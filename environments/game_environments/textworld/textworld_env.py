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
                 initial_obs: str, initial_infos: Dict[str, Any], max_steps: int, 
                 policy_system_prompt: str):
        self.episode_id: str = episode_id
        self.game_file: str = game_file
        self.textworld_env: TextworldGymEnv = textworld_env_instance
        self.initial_infos: Dict[str, Any] = initial_infos
        
        # Initialize message_history with system prompt and initial observation
        self.message_history: List[Message] = [
            Message(role="system", content=policy_system_prompt),
            Message(role="user", content=initial_obs)
        ]
        self.rm_judgement_history: List[RMJudgementLog] = [] 
        self.policy_step_data: List[ScoredDataGroup] = []
        
        self.cumulative_reward: float = 0.0
        self.current_turn: int = 0
        self.max_turns: int = max_steps
        
        self.last_score: float = initial_infos.get("score", 0.0)
        self.moves: int = initial_infos.get("moves", 0)
        self.won: bool = False
        self.lost: bool = False
        self.done: bool = False


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
            
            ep_state = TextWorldEpisodeState(
                episode_id=episode_id,
                game_file=game_file_path,
                textworld_env_instance=env,
                initial_obs=formatted_initial_obs,
                initial_infos=infos,
                max_steps=self.config.max_steps,
                policy_system_prompt=self.policy_agent_system_prompt_content
            )
            self.episodes[episode_id] = ep_state
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
        This now directly returns the episode state object.
        """
        # The 'item' from BaseEnv (seed, group_idx) is not directly used here as TextWorld
        # game generation and episode ID are handled internally.
        # We can use a random seed for game generation if self.config.game_seed is None.
        episode_state = await self._get_or_create_episode(episode_seed=self.config.game_seed)
        if episode_state is None:
            logger.error("Failed to get or create a new TextWorld episode.")
            return None # Propagate failure
        
        # The agent's dialogue and memory should be reset at the start of a new episode/trajectory.
        # This is handled by AtroposAgent.start_new_game_dialogue()
        self.agent.start_new_game_dialogue()
        # RM doesn't have explicit per-episode state that needs resetting in the same way currently.

        return {"episode_state": episode_state, "episode_id": episode_state.episode_id} # Wrap in a dict, conform to Item type

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
        """
        logger.info(f"[Episode: {ep_state.episode_id} Turn: {current_turn_num + 1}/{ep_state.max_turns}] Starting step.")

        # 1. Get Policy Agent Alternatives (G_policy_alternatives)
        #    The agent's current message_history is in ep_state.message_history[-1] (user role with current observation)
        #    The agent manages its own memory and full dialogue history internally.
        
        policy_alternatives: List[Dict[str, Any]] = [] # To store {action_text: str, full_history: List[Message], raw_agent_response: str}
        
        # The observation content for the agent is the last user message in its history
        # This was set by _get_or_create_episode or the previous _next_step
        current_observation_content = ep_state.message_history[-1]["content"]
        # The game_history_window for the agent is everything *before* the current observation
        game_history_for_agent = ep_state.message_history[:-1]

        agent_action_tasks = []
        agent_action_tasks.append(self.agent.generate_action(
            game_history_window=game_history_for_agent,
            n=self.config.G_policy_alternatives
        ))
        
        try:
            generated_actions_results = await asyncio.gather(*agent_action_tasks)
            logger.info(f"Generated actions results: {generated_actions_results}")
        except Exception as e:
            logger.error(f"[Episode: {ep_state.episode_id} Turn: {current_turn_num + 1}] Error during agent.generate_action calls: {e}", exc_info=True)
            return None, True # Critical error, end episode

        for raw_agent_response_text, agent_history_after_action in generated_actions_results:
            if raw_agent_response_text is None:
                logger.warning(f"[Episode: {ep_state.episode_id} Turn: {current_turn_num + 1}] Agent returned None action. Skipping alternative.")
                # Potentially add a placeholder or handle this by reducing G for this step
                continue 
            parsed_action_command = self._parse_action(raw_agent_response_text)
            policy_alternatives.append({
                "parsed_command": parsed_action_command, # This is the string like "go north" or None
                "raw_agent_response": raw_agent_response_text, # Full <think>...<tool_call>...</tool_call>
                "agent_history_for_rm": agent_history_after_action # History for RM to evaluate this action
            })

        if not policy_alternatives:
            logger.error(f"[Episode: {ep_state.episode_id} Turn: {current_turn_num + 1}] No valid policy alternatives generated. Ending episode.")
            return None, True

        # 2. Evaluate Alternatives with RM
        alternative_rm_scores: List[float] = []
        all_rm_judgements_this_step: List[RMJudgementLog] = []

        rm_evaluation_tasks = []
        for i, policy_alt in enumerate(policy_alternatives):
            # game_history_window for RM is the agent's history *including* the action it just proposed.
            # This is policy_alt["agent_history_for_rm"]
            if not policy_alt["agent_history_for_rm"]:
                logger.warning(f"[Episode: {ep_state.episode_id} Turn: {current_turn_num + 1} Alt: {i}] Missing agent_history_for_rm. Skipping RM eval for this alt.")
                # We'll need a score for this alternative later, assign a penalty or handle missing.
                # For now, this path means it won't be added to rm_evaluation_tasks.
                continue

            rm_evaluation_tasks.append(self.rm.generate_g_judgements(
                num_judgements_g=self.config.G_rm_judgements,
                game_history_window=policy_alt["agent_history_for_rm"],
                game_seed_for_logging=self.config.game_seed, # Or a per-episode seed if available
                turn_idx_for_logging=current_turn_num,
                policy_action_candidate_idx_for_logging=i
            ))
        
        if rm_evaluation_tasks: # Only gather if there are tasks
            try:
                rm_judgement_log_groups = await asyncio.gather(*rm_evaluation_tasks)
            except Exception as e:
                logger.error(f"[Episode: {ep_state.episode_id} Turn: {current_turn_num + 1}] Error during rm.generate_g_judgements calls: {e}", exc_info=True)
                # This is a critical error for RM evaluation, affects scoring of all alternatives.
                # We might decide to assign a default low score to all, or end the episode.
                # For now, let's assume alternatives without RM scores get a penalty.
                rm_judgement_log_groups = [[RMJudgementLog(api_error=True)] * self.config.G_rm_judgements] * len(policy_alternatives) # Dummy error logs
        else: # No tasks were created (e.g. all policy_alt["agent_history_for_rm"] were None)
            rm_judgement_log_groups = []

        current_alt_idx = 0
        for i, policy_alt in enumerate(policy_alternatives):
            if not policy_alt["agent_history_for_rm"]:
                # This alternative was skipped for RM evaluation
                alternative_rm_scores.append(-float('inf')) # Assign a very low score
                logger.warning(f"[Episode: {ep_state.episode_id} Turn: {current_turn_num+1} Alt: {i}] Alternative had no history for RM, assigned bad score.")
                continue
            
            # Check if we have results for this alternative (in case of errors in gather)
            if current_alt_idx >= len(rm_judgement_log_groups):
                logger.error(f"[Episode: {ep_state.episode_id} Turn: {current_turn_num+1} Alt: {i}] Mismatch between policy_alternatives and rm_judgement_log_groups. Assigning bad score.")
                alternative_rm_scores.append(-float('inf'))
                continue

            rm_judgements_for_this_alt = rm_judgement_log_groups[current_alt_idx]
            current_alt_idx += 1
            all_rm_judgements_this_step.extend(rm_judgements_for_this_alt)
            
            valid_q_values = []
            for rm_log in rm_judgements_for_this_alt:
                if not rm_log["api_error"] and not rm_log["q_value_parse_error"] and rm_log["parsed_q_value"] is not None:
                    valid_q_values.append(rm_log["parsed_q_value"])
                else:
                    # Optionally, add a default or penalty for errored RM judgements to the average
                    valid_q_values.append(0.0) # Defaulting errored/missing Q to 0 for the mean
            
            if valid_q_values:
                mean_q_value = sum(valid_q_values) / len(valid_q_values)
                alternative_rm_scores.append(mean_q_value)
            else:
                # This case should ideally be covered by the default 0.0 above, but as a fallback:
                logger.warning(f"[Episode: {ep_state.episode_id} Turn: {current_turn_num + 1} Alt: {i}] No valid Q-values from RM. Assigning default score.")
                alternative_rm_scores.append(0.0) # Default score if RM completely failed for an alt

        ep_state.rm_judgement_history.extend(all_rm_judgements_this_step)

        # 3. Select Best Action
        if not alternative_rm_scores:
            logger.error(f"[Episode: {ep_state.episode_id} Turn: {current_turn_num + 1}] No RM scores available for any alternative. Ending episode.")
            return None, True
            
        best_alternative_idx = alternative_rm_scores.index(max(alternative_rm_scores))
        chosen_policy_alt = policy_alternatives[best_alternative_idx]
        chosen_action_command = chosen_policy_alt["parsed_command"]
        chosen_agent_raw_response = chosen_policy_alt["raw_agent_response"]

        if chosen_action_command is None:
            logger.warning(f"[Episode: {ep_state.episode_id} Turn: {current_turn_num + 1}] Best alternative (idx {best_alternative_idx}) had a None parsed command. Using 'look' as fallback.")
            chosen_action_command = "look" # Fallback action

        logger.info(f"[Episode: {ep_state.episode_id} Turn: {current_turn_num + 1}] Chosen action: '{chosen_action_command}' (from Alt {best_alternative_idx} with RM score {alternative_rm_scores[best_alternative_idx]:.2f})")

        # 4. Execute Chosen Action in TextWorld
        try:
            obs, score_from_env, done, infos = ep_state.textworld_env.step(chosen_action_command)
        except Exception as e:
            logger.error(f"[Episode: {ep_state.episode_id} Turn: {current_turn_num + 1}] Error stepping TextWorld environment with action '{chosen_action_command}': {e}", exc_info=True)
            ep_state.done = True
            return None, True # Critical error, end episode

        # 5. Update Episode State
        ep_state.cumulative_reward += score_from_env # TextWorld score is often cumulative
        ep_state.current_turn += 1
        ep_state.done = done or ep_state.current_turn >= ep_state.max_turns
        ep_state.last_score = infos.get("score", ep_state.last_score)
        ep_state.moves = infos.get("moves", ep_state.moves)
        ep_state.won = infos.get("won", False)
        ep_state.lost = infos.get("lost", False)
        if ep_state.won or ep_state.lost:
            ep_state.done = True

        # Add chosen agent response and new observation to the main message history for next turn
        ep_state.message_history.append(Message(role="assistant", content=chosen_agent_raw_response)) 
        formatted_new_obs = self._format_observation(obs, infos)
        ep_state.message_history.append(Message(role="user", content=formatted_new_obs))
        
        # Agent memory update is handled internally by AtroposAgent.generate_action
        # based on the chosen_agent_raw_response and its preceding history.
        # However, AtroposAgent.generate_action was called G times. We only want memory for the CHOSEN action.
        # This requires a slight refactor in how memory is committed OR we pass the chosen path back to agent.
        # For now, let's assume AtroposAgent might generate memory for each of G, and we might log it,
        # but its internal FAISS index is what matters for *its next turn*. The current design of 
        # AtroposAgent.generate_action updates its own memory upon generation. This might mean the 
        # agent has memories from unchosen paths. This needs refinement.
        # A potential fix: agent.generate_action returns thought/action but doesn't commit to memory.
        # A new agent method agent.commit_action_and_memory(chosen_history) is called here.
        # For now, this is a known issue with current agent design vs. best-of-N.

        # 6. Prepare ScoredDataGroup for Policy Agent
        #    Tokens and masks need to be generated for each policy alternative's history.
        #    The `agent_history_for_rm` is suitable for this. It ends with the assistant's action.
        sg_tokens: List[List[int]] = []
        sg_masks: List[List[int]] = []
        sg_messages: List[List[Message]] = [] # Full message histories for each policy alt

        for policy_alt in policy_alternatives:
            # policy_alt["agent_history_for_rm"] is [system_prompt, user_obs, assistant_action_N]
            # This is the trajectory to tokenize for the policy trainer for this alternative.
            history_to_tokenize = policy_alt["agent_history_for_rm"]
            if not history_to_tokenize: # Should not happen if policy_alternatives were generated
                sg_tokens.append([])
                sg_masks.append([])
                sg_messages.append([])
                continue
            
            try:
                tokenized_output = tokenize_for_trainer(self.tokenizer, history_to_tokenize)
                sg_tokens.append(tokenized_output["tokens"])
                sg_masks.append(tokenized_output["masks"])
                sg_messages.append(history_to_tokenize)
            except Exception as e:
                logger.error(f"[Episode: {ep_state.episode_id} Turn: {current_turn_num + 1}] Error tokenizing policy alternative: {e}", exc_info=True)
                sg_tokens.append([])
                sg_masks.append([])
                sg_messages.append(history_to_tokenize) # Still save messages if tokenization fails
        
        # Scores are the mean Q-values from RM for now. Will be updated at episode end.
        current_step_scored_data = ScoredDataGroup(
            tokens=sg_tokens,
            masks=sg_masks,
            scores=list(alternative_rm_scores), # Make a copy
            messages=sg_messages,
            # Add metadata that might be useful for end-of-episode reward calculation
            metadata={"turn_number": current_turn_num, "chosen_alternative_index": best_alternative_idx}
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
            logger.info(f"[Episode: {ep_state.episode_id}] Finalizing episode. Score: {ep_state.last_score}, Won: {ep_state.won}, Lost: {ep_state.lost}, Turns: {ep_state.current_turn}")
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
        # Define two server configs: one for agent, one for RM (can be the same model/endpoint)
        # It's assumed server_clients in BaseEnv will be populated based on unique model_names.
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/Hermes-2-Pro-Llama-3-8B", # For Policy Agent
                base_url="http://localhost:9004/v1", # Example endpoint
                api_key="EMPTY",
                # num_requests_for_eval can be set if needed
            ),
            APIServerConfig(
                model_name="gpt4.1-mini", # For Reward Model (example, could be same as agent)
                base_url="http://localhost:9005/v1", # Example different endpoint for RM
                api_key="EMPTY", 
            )
        ]
        return env_config, server_configs

if __name__ == "__main__":
    TextWorldEnv.cli()
