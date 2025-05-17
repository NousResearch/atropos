#!/usr/bin/env python3
"""
TextWorldEnv: Trainer environment for Microsoft TextWorld

Wraps the TextWorld game generator and Gym interface to train an LLM
using a best-of-n pattern with function-call style actions. Extends BaseEnv.
"""

import json
import logging
import os
import random
import shutil
import tempfile
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import textworld
import textworld.challenges  # Import the challenges module
import textworld.gym
from textworld import EnvInfos, GameOptions
from pydantic import Field

from atroposlib.envs.base import (
    BaseEnv,
    BaseEnvConfig,
    APIServerConfig,
    ScoredDataGroup,
)
from atroposlib.type_definitions import Message
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer
from atroposlib.utils.tool_call_parser import parse_tool_call

# Import the generation utility
from environments.game_environments.textworld.generation_utils import generate_textworld_game

# Import agent and RM configurations
from environments.agents.atropos_agent import AtroposAgent, AtroposAgentConfig
from environments.agents.atropos_rm import AtroposRM, AtroposRMConfig, RMJudgementLog
from environments.agents.atropos_agent_types import AtroposAgentAction

import asyncio # Add asyncio for gather
import textworld.gym.envs # Import the specific module
from textworld.gym.envs import TextworldGymEnv # Import the class

logger = logging.getLogger(__name__)


# Updated Config Class (as previously defined)
class TextWorldEnvConfig(BaseEnvConfig):
    """
    Configuration for the TextWorld environment trainer.
    """

    env_name: str = "TextWorld"
    max_steps: int = 50
    challenge_name: str = "simple_home"
    challenge_rewards: str = "dense"
    challenge_goal: str = "detailed"
    challenge_test_mode: bool = False
    nb_rooms: int = 5
    nb_objects: int = 10
    quest_min_length: int = 3
    quest_max_length: int = 3
    quest_max_depth: int = 3
    grammar_theme: str = "house"
    grammar_include_adj: bool = True
    game_seed: Optional[int] = None
    max_token_length: int = 16384 # LLM max token length
    max_trajectory_tokens: int = 24576 # Trainer max token length
    agent_config: Optional[AtroposAgentConfig] = None
    rm_config: Optional[AtroposRMConfig] = None
    G_policy_alternatives: int = Field(default=4, description="Number of alternative actions policy agent generates.")
    G_rm_judgements: int = Field(default=1, description="Number of judgements RM makes per policy alternative.")
    rm_reward_discount_factor: float = Field(default=0.99, description="Discount factor for RM Q-value accuracy scoring.")
    debug_mode: bool = False

    # TextWorld specific game generation settings
    game_generation_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "max_score": 10,
            "nb_rooms": 5,
            "nb_objects": 10,
            "quest_length": 3,
            "quest_breadth": 2,
            "include_take_action": True,
            "include_open_action": True,
            "include_drop_action": True,
            "include_go_action": True,
            "include_examine_action": True,
            "include_inventory_action": True,
            "include_look_action": True,
            # Add more as needed based on textworld.GameMaker API
        }
    )
    game_file_path: Optional[str] = None # If using a pre-existing game file

    # Agent and RM configurations
    default_server_config: APIServerConfig = Field(
        default_factory=lambda: APIServerConfig(api_server_type="openai", model_name="gpt-3.5-turbo")
    )
    policy_agent_server_config: Optional[APIServerConfig] = None
    rm_agent_server_config: Optional[APIServerConfig] = None

    atropos_agent_config: AtroposAgentConfig = Field(default_factory=AtroposAgentConfig) # Use default_factory
    atropos_rm_config: AtroposRMConfig = Field(default_factory=AtroposRMConfig) # Use default_factory
    group_size: int = 3 # Number of alternatives
    # Policy search and evaluation parameters
    G_policy_alternatives: int = Field(
        default=3, 
        description="Number of policy alternatives to generate at each step."
    )
    G_rm_judgements: int = Field(
        default=1, 
        description="Number of RM judgements to sample per policy alternative."
    )
    
    rm_reward_discount_factor: float = Field(
        default=0.99, 
        description="Discount factor for future game rewards when training RM."
    )

    class Config:
        extra = 'forbid'
        arbitrary_types_allowed = True

# New EpisodeState for Atropos integration
class TextWorldEpisodeState:
    """
    Stores per-episode state for a TextWorld game when using AtroposAgent and AtroposRM.
    """
    def __init__(self, episode_id: str, game_file: str, textworld_env_instance: TextworldGymEnv, 
                 initial_obs: str, initial_infos: Dict[str, Any], max_steps: int):
        self.episode_id: str = episode_id
        self.game_file: str = game_file
        self.textworld_env: TextworldGymEnv = textworld_env_instance
        self.initial_formatted_obs: str = initial_obs
        self.initial_infos: Dict[str, Any] = initial_infos
        
        self.rm_judgement_history: List[RMJudgementLog] = [] 
        self.policy_step_data: List[ScoredDataGroup] = []
        
        self.cumulative_reward: float = 0.0
        self.max_turns: int = max_steps
        
        self.last_score: float = initial_infos.get("score", 0.0)
        self.moves: int = initial_infos.get("moves", 0)
        self.won: bool = False
        self.lost: bool = False
        self.done: bool = False
        self.last_env_raw_observation: Optional[str] = None
        self.last_env_infos: Optional[Dict[str, Any]] = None


class TextWorldEnv(BaseEnv):
    """
    Trainer environment for TextWorld integrating AtroposAgent and AtroposRM.
    """

    def __init__(
        self,
        config: TextWorldEnvConfig,
        server_configs: List[APIServerConfig], # These are all unique configs needed by env
        slurm: bool = True,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: TextWorldEnvConfig = config 
        self.episodes: Dict[str, TextWorldEpisodeState] = {}
        self._temp_dir = tempfile.mkdtemp(prefix="textworld_env_")
        logger.info(f"TextWorldEnv created temporary directory: {self._temp_dir}")

        # Define the tool for executing TextWorld commands
        self.textworld_tools = [
            {
                "type": "function",
                "function": {
                    "name": "execute_command",
                    "description": "Execute a text command in the adventure game.",
                    "parameters": {
                        "type": "object", # Added for OpenAI compatibility
                        "properties": {
                             "command": { # Changed from command_string to command for consistency with _parse_action
                                "type": "string", 
                                "description": "The full text command to execute (e.g., 'go north', 'take shiny key', 'open wooden door with rusty key')."
                            }
                        },
                        "required": ["command"] # Added for OpenAI compatibility
                    },
                },
            }
        ]
        tools_json = json.dumps(self.textworld_tools, indent=2)

        # Policy agent system prompt
        constructed_system_prompt = (
            "You are a long-thinking AI agent playing a text-based adventure game. Your goal is to follow the objective described "
            "at the start of the game. You interact with the world by providing text commands."
            "\\n\\n"
            "Carefully observe the room descriptions, your inventory, and any feedback from your previous actions. "
            "Think step-by-step about how to achieve the objective."
            "\\n\\n"
            "You MUST first output your thoughts and reasoning process within <think> </think> XML tags. " # Emphasize thinking block
            "After your thoughts, you MUST call the 'execute_command' function to provide your chosen text command. "
            "Do NOT output the command directly as text. Use the tool."
            "\\n\\n"
            f"<tools>\\n{tools_json}\\n</tools>\\n\\n"
            "Your function call should be a JSON object with the function name ('execute_command') and the 'command' argument, "
            "enclosed within <tool_call> </tool_call> tags. Example format:"
            "\\n\\n"
            "<think>\\n"
            "The player is in a dark room. There's a door to the north. The objective is to find the treasure. "
            "I should try opening the door first, or perhaps look for a light source. "
            "Going north seems like a direct approach to explore further. "
            "I will try to go north."
            "\\n</think>\\n"
            "<tool_call>\\n"
            '''{"name": "execute_command", "arguments": {"command": "go north"}}'''
            "\\n</tool_call>\\n\\n" # Corrected example arguments to match tool schema
            "Your response MUST follow this format exactly: <think>...</think> followed by <tool_call>...</tool_call>."
        )
        # Ensure AtroposAgentConfig is instantiated
        agent_cfg = self.config.atropos_agent_config if self.config.atropos_agent_config is not None else AtroposAgentConfig()
        agent_cfg.system_prompt = constructed_system_prompt 
        
        self.agent = AtroposAgent(
            server_client=self.server, 
            tokenizer=self.tokenizer, 
            config=agent_cfg
        )
        # Store the system prompt that will be used for the policy agent message history
        # TODO: Remove this?
        self.policy_agent_system_prompt_content = agent_cfg.system_prompt # Use the updated system_prompt
 
        # Ensure AtroposRMConfig is instantiated
        rm_cfg = self.config.atropos_rm_config if self.config.atropos_rm_config is not None else AtroposRMConfig()

        self.rm = AtroposRM(
            server_client=self.server,
            tokenizer=self.tokenizer,
            config=rm_cfg
        )
 
        if self.config.debug_mode:
            logger.setLevel(logging.DEBUG)

    async def setup(self):
        """Ensure prerequisites are met, e.g., TextWorld is installed and working."""
        # Basic check, can be expanded (e.g., try generating a dummy game)
        try:
            import textworld
            logger.info(f"TextWorld version {textworld.__version__} found.")
        except ImportError:
            logger.error("TextWorld library not found. Please install it to use TextWorldEnv.")
            raise
        # Agent and RM setup (like model loading) is handled by their __init__ methods,
        # which are called in TextWorldEnv.__init__.
        pass

    def _format_observation(self, obs: str, infos: Dict[str, Any]) -> str:
        """Formats the TextWorld observation and additional info for the LLM."""
        # Basic formatting, can be enhanced to include inventory, score, objective, etc.
        # from infos if they are consistently available and useful.
        objective = infos.get("objective", "No objective provided.")
        inventory = infos.get("inventory", "Your inventory is empty.")
        description = infos.get("description", obs) # Use description if available, else raw obs
        feedback = infos.get("feedback", "") # Previous command's feedback

        # Filter out potentially redundant or overly verbose infos if necessary.
        # For now, include key ones.
        formatted_obs = f"Objective: {objective}\n\n"
        formatted_obs += f"Current Location & State:\n{description}\n\n"
        formatted_obs += f"Inventory: {inventory}\n\n"
        if infos.get("last_action"): # Add feedback from last action if it exists
            formatted_obs += f"Feedback from last action ('{infos['last_action']}'):\n{feedback}\n"
        
        # Consider adding score, moves, etc., if strategically important for the agent.
        # current_score = infos.get("score", 0)
        # num_moves = infos.get("moves", 0)
        # formatted_obs += f"Current Score: {current_score}, Moves: {num_moves}\n"
        
        return formatted_obs.strip()

    async def _get_or_create_episode(self, episode_seed: Optional[int] = None) -> Optional[TextWorldEpisodeState]:
        """
        Generates a new TextWorld game, registers it, and initializes an episode state.
        If episode_seed is provided, it's used for game generation.
        The agent's game state is reset here.
        """
        episode_id = f"textworld-episode-{uuid.uuid4().hex}"
        
        current_game_seed = episode_seed if episode_seed is not None else random.randint(0, 0xFFFFFFFF)

        # Create a proper GameOptions object
        options = GameOptions()
        options.seeds = current_game_seed
        options.nb_rooms = self.config.nb_rooms
        options.nb_objects = self.config.nb_objects
        options.chaining.min_length = self.config.quest_min_length
        options.chaining.max_length = self.config.quest_max_length
        options.chaining.max_depth = self.config.quest_max_depth
        options.grammar.theme = self.config.grammar_theme
        options.grammar.include_adj = self.config.grammar_include_adj

        # Create settings dictionary as expected by generate_textworld_game
        challenge_settings = {
            'seed': current_game_seed,
            'rewards': self.config.challenge_rewards,
            'goal': self.config.challenge_goal,
            'test': self.config.challenge_test_mode
        }

        # Call generate_textworld_game with the correct parameters - EXACTLY as in the working code
        try:
            game_file_path, game_object = generate_textworld_game(
                challenge_name=self.config.challenge_name,
                settings=challenge_settings,
                options=options,
                output_folder=self._temp_dir,
                filename_prefix=f"{self.config.challenge_name}_ep{current_game_seed}"
            )
            
            if not game_file_path or not os.path.exists(game_file_path):
                logger.error(f"Failed to generate game file for episode {episode_id} with seed {current_game_seed}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating game for {self.config.challenge_name} challenge: {e}", exc_info=True)
            return None

        requested_infos = EnvInfos(
            description=True, inventory=True, objective=True, score=True, 
            max_score=True, won=True, lost=True, facts=True, 
            last_action=True, feedback=True, moves=True, admissible_commands=True
        )
        
        registered_env_id = textworld.gym.register_game(game_file_path, requested_infos, max_episode_steps=self.config.max_steps, name=episode_id)
        logger.info(f"Registered gym environment: {registered_env_id} (using episode_id: {episode_id})")

        try:
            env = textworld.gym.make(registered_env_id)
            logger.info(f"Gym environment created for {episode_id}.")
            raw_obs, infos = env.reset() # Resets and provides initial state
            formatted_initial_obs = self._format_observation(raw_obs, infos)
            
            # Agent's game state should be reset for a new episode
            self.agent.new_game() # Ensures agent's internal game_log is cleared

            ep_state = TextWorldEpisodeState(
                episode_id=episode_id,
                game_file=game_file_path,
                textworld_env_instance=env,
                initial_obs=formatted_initial_obs, # Pass formatted initial obs
                initial_infos=infos,
                max_steps=self.config.max_steps,
                # policy_system_prompt is handled by agent's config
            )
            self.episodes[episode_id] = ep_state
            # Store initial raw obs and infos for the first step if needed outside _format_observation
            ep_state.last_env_raw_observation = raw_obs 
            ep_state.last_env_infos = infos
            return ep_state
        except Exception as e:
            logger.error(f"Failed to setup gym environment for {game_file_path} (episode {episode_id}): {e}", exc_info=True)
            if os.path.exists(game_file_path):
                try: os.remove(game_file_path)
                except OSError: pass # Ignore if removal fails, temp dir will be cleaned later
            return None

    async def get_next_item(self) -> Optional[Dict[str, Any]]:
        """
        Provides a new, initialized TextWorldEpisodeState for trajectory collection.
        This now directly returns the episode state object. Agent's new_game is called within _get_or_create_episode.
        """
        episode_state = await self._get_or_create_episode(episode_seed=self.config.game_seed)
        if episode_state is None:
            logger.error("Failed to get or create a new TextWorld episode.")
            return None
        
        # self.agent.new_game() is now called in _get_or_create_episode

        return {"episode_state": episode_state, "episode_id": episode_state.episode_id}

    def _parse_action(self, agent_response_text: str) -> Optional[str]:
        """
        Parses the agent's response to extract the TextWorld command string,
        expecting the command to be within a tool call.
        Example tool call: <tool_call>{"name": "execute_command", "arguments": {"command": "go north"}}</tool_call>
        """
        if not agent_response_text:
            logger.warning("[TextWorldEnv._parse_action] Received empty or None agent response text.")
            return None

        # We expect the agent to use a specific tool name for TextWorld commands.
        # This could be made configurable if needed.
        expected_tool_name = "execute_command"
        command_argument_key = "command"

        tool_name, arguments, is_error = parse_tool_call(
            response=agent_response_text,
            # available_tools can be provided if strict validation against a schema is needed,
            # but for now, we primarily check the tool_name and presence of the command argument.
            # Example: available_tools=[{"name": "execute_command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}],
            preferred_tags=["tool_call"] # Standard tag
        )

        if is_error:
            logger.warning(f"[TextWorldEnv._parse_action] Failed to parse tool call from response: '{agent_response_text[:200]}...'")
            return None

        if tool_name != expected_tool_name:
            logger.warning(
                f"[TextWorldEnv._parse_action] Unexpected tool name. Expected '{expected_tool_name}', but got '{tool_name}'. Response: '{agent_response_text[:200]}...'"
            )
            return None

        parsed_command = arguments.get(command_argument_key)

        if not parsed_command or not isinstance(parsed_command, str):
            logger.warning(
                f"[TextWorldEnv._parse_action] Command argument '{command_argument_key}' not found or not a string in tool call arguments. Args: {arguments}. Response: '{agent_response_text[:200]}...'"
            )
            return None
        
        parsed_command = parsed_command.strip()
        if not parsed_command:
            logger.warning(
                 f"[TextWorldEnv._parse_action] Parsed command is an empty string after stripping. Original from tool: '{arguments.get(command_argument_key)}'. Response: '{agent_response_text[:200]}...'"
            )
            return None

        # Optional: Validate against admissible commands if available and strictness is needed, though
        # TextWorld itself will handle invalid commands.
        # infos = ep_state.textworld_env.infos # (Would need ep_state here, or pass infos)
        # if 'admissible_commands' in infos and parsed_command not in infos['admissible_commands']:
        #     logger.warning(f"Parsed command '{parsed_command}' not in admissible commands: {infos['admissible_commands']}")
            # Decide if this is an error or let TextWorld handle it.

        logger.debug(f"[TextWorldEnv._parse_action] Parsed command via tool call: '{parsed_command}'")
        return parsed_command

    async def _next_step(
        self, ep_state: TextWorldEpisodeState, current_turn_num: int
    ) -> Tuple[Optional[ScoredDataGroup], bool]: # Returns (ScoredDataGroup for this turn, is_episode_done)
        """
        Process one step/turn of a TextWorld episode using AtroposAgent and AtroposRM.
        Relies on agent for history management.
        """
        # Log with 1-indexed turn number for human readability
        logger.info(f"[Episode: {ep_state.episode_id} Turn: {current_turn_num + 1}/{ep_state.max_turns}] Starting step.")

        # 1. Determine current observation for the agent
        if current_turn_num == 0:
            current_observation_for_agent = ep_state.initial_formatted_obs
        elif ep_state.last_env_raw_observation is not None and ep_state.last_env_infos is not None:
            current_observation_for_agent = self._format_observation(ep_state.last_env_raw_observation, ep_state.last_env_infos)
        else:
            logger.error(f"[Episode: {ep_state.episode_id} Turn: {current_turn_num + 1}] Missing last observation data. Cannot proceed.")
            return None, True # Critical error
        logger.debug(f"[Episode: {ep_state.episode_id} Turn: {current_turn_num + 1}] Observation for Agent: {current_observation_for_agent[:300]}...")

        # 2. Get Policy Agent Alternatives
        try:
            agent_action_alternatives: List[AtroposAgentAction] = await self.agent.generate_action(
                observation_content=current_observation_for_agent,
                n=self.config.G_policy_alternatives,
            )
        except Exception as e:
            logger.error(f"[Episode: {ep_state.episode_id} Turn: {current_turn_num + 1}] Error during agent.generate_action: {e}", exc_info=True)
            return None, True 

        if not agent_action_alternatives:
            logger.error(f"[Episode: {ep_state.episode_id} Turn: {current_turn_num + 1}] Agent generated no action alternatives. Ending episode.")
            return None, True
        logger.info(f"[Episode: {ep_state.episode_id} Turn: {current_turn_num + 1}] Agent generated {len(agent_action_alternatives)} action alternatives.")

        # The agent has now added a turn to its self.agent.game_log.turn[-1]
        # This turn includes the observation_message and all alternatives, with selected_alternative = None

        # 3. Evaluate Alternatives with RM
        policy_alternatives_for_rm_eval = [] # Stores {parsed_command, raw_agent_response, history_for_rm}
        
        # History up to *before* the current observation that led to these alternatives
        history_of_completed_turns = self.agent._reconstruct_canonical_history() # Ends with last selected assistant action
        
        # The current observation message is stored in the agent's latest (incomplete) turn
        if not self.agent.game_log.turn: # Should not happen if generate_action succeeded
             logger.error(f"[Episode: {ep_state.episode_id} Turn: {current_turn_num + 1}] Agent's game_log is empty after generate_action. Critical error.")
             return None, True
        current_observation_msg_from_agent_log = self.agent.game_log.turn[-1].observation_message


        for alt_action in agent_action_alternatives:
            parsed_cmd = self._parse_action(alt_action.action_text)
            
            # Construct history for RM: completed history + current observation + this specific alternative action
            history_for_rm_alt = list(history_of_completed_turns) 
            history_for_rm_alt.append(current_observation_msg_from_agent_log)
            history_for_rm_alt.append(Message(role="assistant", content=alt_action.action_text))
            
            policy_alternatives_for_rm_eval.append({
                "parsed_command": parsed_cmd,
                "raw_agent_response": alt_action.action_text, # Full <think>...<tool_call>...</tool_call>
                "agent_history_for_rm": history_for_rm_alt 
            })

        alternative_rm_scores: List[float] = []
        all_rm_judgements_this_step: List[RMJudgementLog] = []
        rm_evaluation_tasks = []

        for i, policy_alt_data in enumerate(policy_alternatives_for_rm_eval):
            rm_evaluation_tasks.append(self.rm.generate_g_judgements(
                num_judgements_g=self.config.G_rm_judgements,
                game_history_window=policy_alt_data["agent_history_for_rm"],
                game_seed_for_logging=self.config.game_seed, # Or a per-episode seed
                turn_idx_for_logging=current_turn_num,
                policy_action_candidate_idx_for_logging=i
            ))
        
        if rm_evaluation_tasks:
            try:
                rm_judgement_log_groups = await asyncio.gather(*rm_evaluation_tasks)
            except Exception as e:
                logger.error(f"[Episode: {ep_state.episode_id} Turn: {current_turn_num + 1}] Error during rm.generate_g_judgements: {e}", exc_info=True)
                rm_judgement_log_groups = [[RMJudgementLog(api_error=True, parsed_q_value=0.0)] * self.config.G_rm_judgements] * len(policy_alternatives_for_rm_eval)
        else:
            rm_judgement_log_groups = [] 

        for i, rm_judgements_for_this_alt in enumerate(rm_judgement_log_groups):
            all_rm_judgements_this_step.extend(rm_judgements_for_this_alt)
            valid_q_values = [
                j["parsed_q_value"] for j in rm_judgements_for_this_alt 
                if not j["api_error"] and not j["q_value_parse_error"] and j["parsed_q_value"] is not None
            ]
            if not valid_q_values: # If all RM judgements for an alt failed, or no Q values parsed
                 # Use a default low score (e.g. 0 or a penalty)
                logger.warning(f"[Episode: {ep_state.episode_id} Turn: {current_turn_num + 1} Alt: {i}] No valid Q-values from RM. Assigning 0.0.")
                alternative_rm_scores.append(0.0) 
            else:
                mean_q_value = sum(valid_q_values) / len(valid_q_values)
                alternative_rm_scores.append(mean_q_value)

        ep_state.rm_judgement_history.extend(all_rm_judgements_this_step)

        # 4. Select Best Action
        if not alternative_rm_scores: 
            logger.error(f"[Episode: {ep_state.episode_id} Turn: {current_turn_num + 1}] No RM scores available. Ending episode.")
            return None, True
        logger.debug(f"[Episode: {ep_state.episode_id} Turn: {current_turn_num + 1}] RM scores for alternatives: {alternative_rm_scores}")
            
        best_alternative_idx = alternative_rm_scores.index(max(alternative_rm_scores))
        chosen_policy_alt_data = policy_alternatives_for_rm_eval[best_alternative_idx]
        chosen_action_command = chosen_policy_alt_data["parsed_command"]
        # chosen_agent_raw_response = chosen_policy_alt_data["raw_agent_response"] # raw response of chosen action

        if chosen_action_command is None:
            logger.warning(f"[Episode: {ep_state.episode_id} Turn: {current_turn_num + 1}] Best alternative (idx {best_alternative_idx}) had a None parsed command. Using 'look' as fallback.")
            chosen_action_command = "look"

        logger.info(f"[Episode: {ep_state.episode_id} Turn: {current_turn_num + 1}] Chosen action: '{chosen_action_command}' (from Alt {best_alternative_idx} with RM score {alternative_rm_scores[best_alternative_idx]:.2f})")

        # 5. Record selected action with the Agent
        try:
            self.agent.record_selected_action(selected_action_index=best_alternative_idx)
            logger.debug(f"[Episode: {ep_state.episode_id} Turn: {current_turn_num + 1}] Agent logs updated with selected action index: {best_alternative_idx}")
        except Exception as e:
            logger.error(f"[Episode: {ep_state.episode_id} Turn: {current_turn_num + 1}] Error recording selected action with agent: {e}", exc_info=True)
            return None, True


        # 6. Execute Chosen Action in TextWorld
        try:
            raw_obs_next, score_from_env, done_from_env, infos_next = ep_state.textworld_env.step(chosen_action_command)
        except Exception as e:
            logger.error(f"[Episode: {ep_state.episode_id} Turn: {current_turn_num + 1}] Error stepping TextWorld environment: {e}", exc_info=True)
            ep_state.done = True
            return None, True 

        # 7. Update Episode State
        ep_state.cumulative_reward += score_from_env 
        ep_state.done = done_from_env # Max turns condition handled by the calling loop
        ep_state.last_score = infos_next.get("score", ep_state.last_score)
        ep_state.moves = infos_next.get("moves", ep_state.moves)
        ep_state.won = infos_next.get("won", False)
        ep_state.lost = infos_next.get("lost", False)
        if ep_state.won or ep_state.lost:
            ep_state.done = True
        
        # Store the raw observation and infos for the next step's _format_observation call
        ep_state.last_env_raw_observation = raw_obs_next
        ep_state.last_env_infos = infos_next
        
        # message_history update is now implicitly handled by agent and reconstruction for RM/ScoredDataGroup

        # 8. Prepare ScoredDataGroup for Policy Agent
        sg_tokens: List[List[int]] = []
        sg_masks: List[List[int]] = []
        sg_messages: List[List[Message]] = [] 

        for policy_alt_data in policy_alternatives_for_rm_eval:
            history_to_tokenize = policy_alt_data["agent_history_for_rm"]
            # This history_to_tokenize is [SysPrompt, UserObs1, ..., PrevSelectedAssistAction, CurrentUserObs, CurrentAssistAlternative]
            
            try:
                tokenized_output = tokenize_for_trainer(self.tokenizer, history_to_tokenize)
                sg_tokens.append(tokenized_output["tokens"])
                sg_masks.append(tokenized_output["masks"])
                sg_messages.append(history_to_tokenize)
            except Exception as e:
                logger.error(f"[Episode: {ep_state.episode_id} Turn: {current_turn_num + 1}] Error tokenizing history for ScoredDataGroup: {e}", exc_info=True)
                sg_tokens.append([])
                sg_masks.append([])
                sg_messages.append(history_to_tokenize) 
        
        current_step_scored_data = ScoredDataGroup(
            tokens=sg_tokens,
            masks=sg_masks,
            scores=list(alternative_rm_scores), 
            messages=sg_messages,
            metadata={"turn_number": current_turn_num, "chosen_alternative_index": best_alternative_idx}
        )
        ep_state.policy_step_data.append(current_step_scored_data)
        logger.debug(
            f"[Episode: {ep_state.episode_id} Turn: {current_turn_num + 1}] Constructed ScoredDataGroup with "
            f"{len(sg_tokens)} alternatives. Chosen Idx in metadata: {best_alternative_idx}. "
            f"Scores: {current_step_scored_data.scores}"
        )
        
        return current_step_scored_data, ep_state.done

    async def collect_trajectories(
        self, item: Dict[str, Any]  # Item is Dict[str, Any] from BaseEnv
    ) -> Tuple[List[ScoredDataGroup], List[Dict[str, Any]]]: # Return List[ScoredDataGroup] and List[Item]
        """
        Runs a full TextWorld episode using AtroposAgent and AtroposRM,
        collecting data for each step.
        """
        if not item or "episode_state" not in item:
            logger.error(f"Invalid item received in collect_trajectories. Missing 'episode_state'. Item: {item}")
            return [], []

        ep_state: TextWorldEpisodeState = item["episode_state"]
        if not ep_state:
            logger.error("Episode state is None in collect_trajectories.")
            return [], []

        logger.info(f"Starting trajectory collection for episode: {ep_state.episode_id}, Game file: {ep_state.game_file}")

        all_scored_data_groups_for_episode: List[ScoredDataGroup] = []
        
        try:
            for current_turn_num in range(ep_state.max_turns):
                if ep_state.done:
                    logger.info(f"[Episode: {ep_state.episode_id}] Episode marked done before turn {current_turn_num + 1}. Ending early.")
                    break

                
                scored_data_group_for_turn, episode_is_done_after_step = await self._next_step(
                    ep_state, current_turn_num
                )

                # Need to get the RM scored data groups.

                if scored_data_group_for_turn:
                    all_scored_data_groups_for_episode.append(scored_data_group_for_turn)
                
                if episode_is_done_after_step:
                    logger.info(f"[Episode: {ep_state.episode_id}] Episode finished after turn {current_turn_num + 1}. Won: {ep_state.won}, Lost: {ep_state.lost}, Score: {ep_state.last_score}, Moves: {ep_state.moves}")
                    break
                
                if current_turn_num == ep_state.max_turns - 1 and not ep_state.done:
                    logger.info(f"[Episode: {ep_state.episode_id}] Max turns reached. Won: {ep_state.won}, Lost: {ep_state.lost}, Score: {ep_state.last_score}, Moves: {ep_state.moves}")
                    ep_state.done = True # Ensure it's marked done

        except Exception as e:
            logger.error(f"[Episode: {ep_state.episode_id}] Unexpected error during trajectory collection: {e}", exc_info=True)
            ep_state.done = True # Mark as done to ensure cleanup and proper handling
        finally:
            # Use len(all_scored_data_groups_for_episode) for number of processed turns
            processed_turns_count = len(all_scored_data_groups_for_episode)
            logger.info(f"[Episode: {ep_state.episode_id}] Finalizing episode. Score: {ep_state.last_score}, Won: {ep_state.won}, Lost: {ep_state.lost}, Turns: {processed_turns_count}")
            if ep_state.textworld_env:
                try:
                    ep_state.textworld_env.close()
                    logger.debug(f"[Episode: {ep_state.episode_id}] Closed gym environment.")
                except Exception as e_close:
                    logger.warning(f"[Episode: {ep_state.episode_id}] Error closing gym environment: {e_close}")
            
            if ep_state.game_file and os.path.exists(ep_state.game_file):
                try:
                    os.remove(ep_state.game_file)
                    logger.debug(f"[Episode: {ep_state.episode_id}] Removed game file: {ep_state.game_file}")
                except OSError as e_remove:
                    logger.warning(f"[Episode: {ep_state.episode_id}] Error removing game file {ep_state.game_file}: {e_remove}")

            if ep_state.episode_id in self.episodes:
                del self.episodes[ep_state.episode_id]
                logger.debug(f"[Episode: {ep_state.episode_id}] Removed episode state from active tracking.")

        return all_scored_data_groups_for_episode, [] 

    async def postprocess_histories(
        self,
        trajectories: Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]],
        rollout_info: List[Dict[str, Any]] # Item is Dict[str, Any]
    ) -> Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]:
        """
        Post-processes trajectories, e.g., to calculate final rewards based on episode outcome.
        For TextWorld, this could involve discounting rewards based on game win/loss/score.
        Currently a passthrough.
        """
        if not trajectories:
            return trajectories

        processed_trajectories = trajectories
        if isinstance(trajectories, ScoredDataGroup):
            processed_trajectories = [trajectories]
        
        # Ensure it's a list for consistent logging, though it should be from collect_trajectories
        num_groups_to_log = len(processed_trajectories) if isinstance(processed_trajectories, list) else 1

        logger.debug(f"Postprocessing {num_groups_to_log} ScoredDataGroup(s). (Currently a passthrough)")
        return processed_trajectories

    async def evaluate(self, *args, **kwargs):
        # Implementation pending
        logger.warning("evaluate not yet implemented")
        pass

    async def cleanup(self):
        """Clean up temporary game files and directory."""
        # Clean up registered environments
        # Note: textworld.gym doesn't have a public unregister function easily accessible.
        # We might need to manage env_ids carefully or rely on process exit.
        # For now, focus on deleting the temp dir.
        try:
            if os.path.exists(self._temp_dir):
                shutil.rmtree(self._temp_dir)
                logger.info(f"Cleaned up temporary directory: {self._temp_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up temporary directory {self._temp_dir}: {e}")

        await super().cleanup()

    def __del__(self):
        # Ensure cleanup runs even if explicit cleanup isn't called.
        # Avoid calling async methods like self.cleanup() here.
        # Attempt synchronous removal of the temporary directory.
        try:
            if hasattr(self, '_temp_dir') and os.path.exists(self._temp_dir):
                shutil.rmtree(self._temp_dir)
        except Exception as e:
            # Suppress errors during __del__ as interpreter state is unpredictable.
            # print(f"Error during TextWorldEnv __del__: {e}") # Avoid logger
            pass

    @classmethod
    def config_init(cls) -> Tuple[TextWorldEnvConfig, List[APIServerConfig]]:
        """
        Initializes the environment and server configurations with hardcoded defaults.
        This method provides a standard way to get a runnable configuration without
        needing to parse external files.
        """
        env_config = TextWorldEnvConfig(
            # BaseEnvConfig common parameters (inspired by Blackjack defaults)
            tokenizer_name="NousResearch/Hermes-2-Pro-Llama-3-8B", # Example tokenizer
            group_size=1, # Number of trajectories per run_trajectories_on_item call by BaseEnv
            use_wandb=True,
            rollout_server_url="http://localhost:8000", # Example
            total_steps=1000,
            batch_size=32, # Example
            steps_per_eval=50,
            max_token_length=4096, # Max tokens for a single LLM call (agent/RM)
            wandb_name="textworld_atropos",
            eval_handling=APIServerConfig.EvalHandlingEnum.LIMIT_TRAIN, # Corrected enum access
            eval_limit_ratio=0.1,
            max_steps=50,
            challenge_name="tw-simple",
            game_seed=None, 
            max_trajectory_tokens=32768, # For the combined trajectory data sent to trainer
            agent_config=AtroposAgentConfig(), # Use default agent config
            rm_config=AtroposRMConfig(thinking=True),      # Use default RM config, enable thinking
            G_policy_alternatives=4, 
            G_rm_judgements=1, 
            rm_reward_discount_factor=0.99,
            debug_mode=False
        )
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/Hermes-2-Pro-Llama-3-8B",
                base_url="http://localhost:9004/v1",
                api_key="x",
            )
        ]
        return env_config, server_configs

if __name__ == "__main__":
    TextWorldEnv.cli()
