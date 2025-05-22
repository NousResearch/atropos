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
from environments.game_environments.textworld.agents.textworld_memory_manager import TextWorldMemoryManager, MEMORY_SYSTEM_PREREQUISITES_AVAILABLE
from atroposlib.type_definitions import AtroposAgentAction

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

    atropos_agent_config: AtroposAgentConfig = Field(default_factory=AtroposAgentConfig) 
    atropos_rm_config: AtroposRMConfig = Field(default_factory=AtroposRMConfig) 
    # group_size: int = 3 # Number of alternatives -> This is part of BaseEnvConfig now

    # Policy search and evaluation parameters
    # G_policy_alternatives: int = Field( # This is now effectively controlled by atropos_agent_config or direct param to generate_action
    
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
        self.policy_step_data: List[ScoredDataGroup] = [] # Stores ScoredDataGroup for each turn of the canonical path
        
        self.cumulative_reward: float = 0.0 # Overall game score progression
        self.max_turns: int = max_steps
        
        self.last_score: float = initial_infos.get("score", 0.0)
        self.moves: int = initial_infos.get("moves", 0)
        self.won: bool = False
        self.lost: bool = False
        self.done: bool = False
        self.last_env_raw_observation: Optional[str] = None
        self.last_env_infos: Optional[Dict[str, Any]] = None

        # For reward calculation and RM target generation
        self.canonical_rewards: List[float] = [] # Stores immediate rewards from chosen actions
        self.canonical_chosen_alternative_indices: List[int] = [] # Stores index of chosen alternative for each step


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

        # Initialize Memory Manager (shared by Agent)
        self.memory_manager = None
        if self.config.atropos_agent_config.enable_memory and MEMORY_SYSTEM_PREREQUISITES_AVAILABLE:
            logger.info("TextWorldEnv: Initializing TextWorldMemoryManager for AtroposAgent.")
            self.memory_manager = TextWorldMemoryManager(
                embedding_dim_config_val=self.config.atropos_agent_config.embedding_dim,
                player_id_for_logging=f"{self.config.atropos_agent_config.player_id_for_logging}_Memory"
            )
        elif self.config.atropos_agent_config.enable_memory:
            logger.warning("TextWorldEnv: AtroposAgent memory is enabled in config, but prerequisites are not met. Memory will be OFF.")


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
        if self.config.policy_agent_server_config and self.config.policy_agent_server_config.model_name:
            agent_cfg.model_id = self.config.policy_agent_server_config.model_name
        else:
            agent_cfg.model_id = self.config.default_server_config.model_name
        
        self.agent = AtroposAgent(
            server_client=self.server, 
            tokenizer=self.tokenizer, 
            config=agent_cfg,
            memory_manager=self.memory_manager # Pass the memory manager
        )
        # Store the system prompt that will be used for the policy agent message history
        # TODO: Remove this? -> self.agent.system_prompt_content should be the source of truth
        # self.policy_agent_system_prompt_content = agent_cfg.system_prompt # REMOVED - Redundant
 
        # Ensure AtroposRMConfig is instantiated
        rm_cfg = self.config.atropos_rm_config if self.config.atropos_rm_config is not None else AtroposRMConfig()
        # Set RM model_id, defaulting to policy agent's model if not specified
        if self.config.rm_agent_server_config and self.config.rm_agent_server_config.model_name:
            rm_cfg.model_id = self.config.rm_agent_server_config.model_name
        elif agent_cfg.model_id: # Default to policy agent's model_id if RM specific is not set
            rm_cfg.model_id = agent_cfg.model_id
        else: # Fallback to default server config if policy also didn't have one (should not happen with current logic)
            rm_cfg.model_id = self.config.default_server_config.model_name

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
            self.agent.new_game() # Ensures agent's internal game_log and memory manager are cleared

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
        
        # This is the observation_message that was just added to the agent's log by generate_action.
        # This ensures the RM gets the exact same observation the agent based its alternatives on.
        if not self.agent.game_log['turn']: # Should not happen if generate_action succeeded
            logger.error(f"[Episode: {ep_state.episode_id} Turn: {current_turn_num + 1}] Agent's game_log is empty after generate_action. Cannot get observation for RM.")
            # Handle error or return early if critical
            return None, True
        current_observation_msg_from_agent_log = self.agent.game_log['turn'][-1]['observation_message'] # Corrected


        for alt_idx, alt_action in enumerate(agent_action_alternatives):
            parsed_cmd = self._parse_action(alt_action['action_text'])
            
            # Construct history for RM: completed history + current observation + this specific alternative action
            history_for_rm_alt = list(history_of_completed_turns) 
            history_for_rm_alt.append(current_observation_msg_from_agent_log)
            history_for_rm_alt.append(Message(role="assistant", content=alt_action['action_text'], reward=None))
            
            policy_alternatives_for_rm_eval.append({
                "parsed_command": parsed_cmd,
                "raw_agent_response": alt_action['action_text'], # Full <think>...<tool_call>...</tool_call>
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

        # 5. Record selected action with the Agent and allow it to learn/store memory
        try:
            # self.agent.record_selected_action(selected_action_index=best_alternative_idx)
            await self.agent.record_selected_action_and_learn_from_turn(selected_action_index=best_alternative_idx)
            logger.debug(f"[Episode: {ep_state.episode_id} Turn: {current_turn_num + 1}] Agent logs updated and memory processed for selected action index: {best_alternative_idx}")
        except Exception as e:
            logger.error(f"[Episode: {ep_state.episode_id} Turn: {current_turn_num + 1}] Error recording selected action/learning with agent: {e}", exc_info=True)
            return None, True


        # 6. Execute Chosen Action in TextWorld
        try:
            raw_obs_next, immediate_score_from_env, done_from_env, infos_next = ep_state.textworld_env.step(chosen_action_command)
        except Exception as e:
            logger.error(f"[Episode: {ep_state.episode_id} Turn: {current_turn_num + 1}] Error stepping TextWorld environment: {e}", exc_info=True)
            ep_state.done = True # Mark as done
            # Store a placeholder reward if step fails, or decide how to handle
            ep_state.canonical_rewards.append(0.0) # Or a penalty
            ep_state.canonical_chosen_alternative_indices.append(best_alternative_idx)
            return None, True 

        # 7. Update Episode State with outcome of this step
        ep_state.cumulative_reward += immediate_score_from_env # This is the TextWorld score, often sparse
        
        # Store the immediate reward from this step for later GAE/MC calculation.
        # For TextWorld, score_from_env is often the change in total score.
        # If dense rewards are desired, they might need to be engineered (e.g., +0.1 for valid action, -0.1 for invalid).
        # For now, using immediate_score_from_env as the per-step reward.
        ep_state.canonical_rewards.append(immediate_score_from_env)
        ep_state.canonical_chosen_alternative_indices.append(best_alternative_idx)

        ep_state.done = done_from_env 
        ep_state.last_score = infos_next.get("score", ep_state.last_score) # This is cumulative score from TW
        
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
    ) -> Tuple[List[ScoredDataGroup], List[Dict[str, Any]]]: # Return List[ScoredDataGroup] and List[Item for backlog]
        """
        Runs a full TextWorld episode using AtroposAgent and AtroposRM,
        collecting data for each step. Also processes rewards and generates RM training data.
        """
        if not item or "episode_state" not in item:
            logger.error(f"Invalid item received in collect_trajectories. Missing 'episode_state'. Item: {item}")
            return [], []

        ep_state: TextWorldEpisodeState = item["episode_state"]
        if not ep_state: # Should have been caught by the item check, but good practice
            logger.error("Episode state is None in collect_trajectories.")
            return [], []

        logger.info(f"Starting trajectory collection for episode: {ep_state.episode_id}, Game file: {ep_state.game_file}")

        # This list will store ScoredDataGroups for the policy agent for each turn
        policy_sdgs_for_episode: List[ScoredDataGroup] = [] 
        
        try:
            for current_turn_num in range(ep_state.max_turns):
                if ep_state.done:
                    logger.info(f"[Episode: {ep_state.episode_id}] Episode marked done before turn {current_turn_num + 1}. Ending early.")
                    break
                
                scored_data_group_for_turn, episode_is_done_after_step = await self._next_step(
                    ep_state, current_turn_num
                )

                if scored_data_group_for_turn:
                    # Store the SDG from this turn; it will be post-processed later
                    policy_sdgs_for_episode.append(scored_data_group_for_turn) 
                
                if episode_is_done_after_step:
                    logger.info(f"[Episode: {ep_state.episode_id}] Episode finished after turn {current_turn_num + 1}. Won: {ep_state.won}, Lost: {ep_state.lost}, Score: {ep_state.last_score}, Moves: {ep_state.moves}")
                    break
                
                if current_turn_num == ep_state.max_turns - 1 and not ep_state.done:
                    logger.info(f"[Episode: {ep_state.episode_id}] Max turns reached. Won: {ep_state.won}, Lost: {ep_state.lost}, Score: {ep_state.last_score}, Moves: {ep_state.moves}")
                    ep_state.done = True # Ensure it's marked done

        except Exception as e:
            logger.error(f"[Episode: {ep_state.episode_id}] Unexpected error during trajectory collection: {e}", exc_info=True)
            ep_state.done = True 
        finally:
            # --- Post-episode processing ---
            num_canonical_steps = len(ep_state.canonical_rewards)
            
            # 1. Determine final reward for GAE/MC calculation
            final_outcome_reward = 0.0
            if ep_state.won:
                final_outcome_reward = 1.0
            elif ep_state.lost:
                final_outcome_reward = -1.0
            # else, if just max_steps reached without explicit win/loss, reward might be 0 or based on score.
            # For simplicity, using 0 if not win/loss. Could also use normalized ep_state.last_score.
            
            # 2. Calculate true discounted returns for the POLICY agent's chosen actions
            if policy_sdgs_for_episode and num_canonical_steps > 0:
                # Ensure policy_sdgs_for_episode and canonical_rewards align (they should if no errors in _next_step)
                if len(policy_sdgs_for_episode) != num_canonical_steps:
                    logger.error(
                        f"[Episode: {ep_state.episode_id}] Mismatch between policy SDGs ({len(policy_sdgs_for_episode)}) "
                        f"and canonical rewards ({num_canonical_steps}). Reward processing for policy might be incorrect."
                    )
                
                # Calculate discounted returns backwards
                discounted_return = final_outcome_reward 
                for t in range(num_canonical_steps - 1, -1, -1):
                    # The reward r_t is ep_state.canonical_rewards[t]
                    # G_t = r_t + gamma * G_{t+1} (where G_{T} = final_outcome_reward, G_{T+1} for r_T would be 0)
                    # If t is the last step (T-1), then G_t = r_t + gamma * final_outcome_reward (if final_outcome_reward is for state T)
                    # Or, if final_outcome_reward is considered the reward for the last action itself, then it's simpler.
                    
                    # Let's assume canonical_rewards are R_0, R_1, ..., R_{N-1} for N steps.
                    # final_outcome_reward is an additional terminal reward R_N.
                    # G_{N-1} = R_{N-1} + gamma * R_N
                    # G_t = R_t + gamma * G_{t+1}
                    
                    # If current 'discounted_return' is G_{t+1} from previous iteration (or R_N for first iteration)
                    # Current step's immediate reward is ep_state.canonical_rewards[t]
                    current_step_reward = ep_state.canonical_rewards[t]
                    discounted_return = current_step_reward + self.config.rm_reward_discount_factor * discounted_return
                    
                    if t < len(policy_sdgs_for_episode):
                        sdg_t = policy_sdgs_for_episode[t]
                        chosen_idx = ep_state.canonical_chosen_alternative_indices[t]
                        
                        # Update the score of the chosen alternative
                        # Non-chosen alternatives keep their RM-assigned Q-value scores
                        if 0 <= chosen_idx < len(sdg_t["scores"]):
                            # Create a new list for scores to avoid modifying the original if it's shared/logged elsewhere before this
                            new_scores = list(sdg_t["scores"]) 
                            new_scores[chosen_idx] = discounted_return
                            sdg_t["scores"] = new_scores 
                            logger.debug(f"[Episode: {ep_state.episode_id} Turn: {t+1}] Policy chosen action (idx {chosen_idx}) updated MC score to {discounted_return:.4f}")
                        else:
                            logger.warning(f"[Episode: {ep_state.episode_id} Turn: {t+1}] Invalid chosen_idx {chosen_idx} for policy SDG scores.")
                    else:
                        logger.warning(f"[Episode: {ep_state.episode_id}] Index t={t} out of bounds for policy_sdgs_for_episode (len {len(policy_sdgs_for_episode)}).")


            # 3. Generate and send RM training data
            rm_training_data: List[ScoredDataGroup] = []
            if ep_state.rm_judgement_history and num_canonical_steps > 0:
                # Calculate full list of discounted returns for the canonical path once
                canonical_discounted_returns = [0.0] * num_canonical_steps
                current_discounted_return = final_outcome_reward
                for t in range(num_canonical_steps - 1, -1, -1):
                    current_step_reward = ep_state.canonical_rewards[t]
                    current_discounted_return = current_step_reward + self.config.rm_reward_discount_factor * current_discounted_return
                    canonical_discounted_returns[t] = current_discounted_return

                for rm_log_entry in ep_state.rm_judgement_history:
                    turn_idx = rm_log_entry.get("turn_idx_for_logging")
                    policy_alt_idx = rm_log_entry.get("policy_action_candidate_idx_for_logging")

                    # We only train RM on its judgements of the *chosen* policy actions for now
                    if turn_idx is not None and 0 <= turn_idx < num_canonical_steps:
                        chosen_alt_for_this_turn = ep_state.canonical_chosen_alternative_indices[turn_idx]
                        if policy_alt_idx == chosen_alt_for_this_turn:
                            # This RM judgement was for the action that was actually taken
                            true_target_q_value = canonical_discounted_returns[turn_idx]
                            
                            # RM input messages are already List[Dict[str,str]]
                            rm_input_dict_list = rm_log_entry["rm_input_messages"]
                            
                            try:
                                # Tokenize RM's input.
                                # Ensure rm_input_dict_list is correctly formatted if Message type def changes
                                tokenized_rm_input = tokenize_for_trainer(self.tokenizer, rm_input_dict_list)
                                
                                rm_sdg = ScoredDataGroup(
                                    tokens=[tokenized_rm_input["tokens"]],
                                    masks=[tokenized_rm_input["masks"]],
                                    scores=[true_target_q_value], # Target for RM training
                                    messages=[rm_input_dict_list], # Original messages for context/debug
                                    metadata={
                                        "original_rm_q_value": rm_log_entry["parsed_q_value"],
                                        "original_rm_thinking": rm_log_entry["parsed_thinking_block"],
                                        "turn_number": turn_idx,
                                        "policy_action_candidate_idx": policy_alt_idx,
                                        "game_seed": rm_log_entry.get("game_seed_for_logging"),
                                        "episode_id": ep_state.episode_id,
                                        "rm_api_error": rm_log_entry.get("api_error", False),
                                        "rm_q_parse_error": rm_log_entry.get("q_value_parse_error", False),
                                    }
                                )
                                rm_training_data.append(rm_sdg)
                            except Exception as e_tok:
                                logger.error(f"[Episode: {ep_state.episode_id} Turn: {turn_idx+1}] Error tokenizing RM input for training: {e_tok}", exc_info=True)
                        # else: Skip RM judgements for non-chosen policy alternatives for now
                    else:
                        logger.warning(f"[Episode: {ep_state.episode_id}] RM log entry has invalid turn_idx: {turn_idx}")
            
            if rm_training_data:
                logger.info(f"[Episode: {ep_state.episode_id}] Attempting to send {len(rm_training_data)} ScoredDataGroups for RM training.")
                try:
                    # `item` here is the original item passed to collect_trajectories
                    await self.handle_send_to_api(rm_training_data, item=item, do_send_to_api=True)
                except Exception as e_send_rm:
                    logger.error(f"[Episode: {ep_state.episode_id}] Error sending RM training data: {e_send_rm}", exc_info=True)


            # --- Finalize and Cleanup ---
            processed_turns_count = len(policy_sdgs_for_episode) # Number of policy ScoredDataGroups
            logger.info(f"[Episode: {ep_state.episode_id}] Finalizing episode. Score: {ep_state.last_score}, Won: {ep_state.won}, Lost: {ep_state.lost}, Processed Turns for Policy: {processed_turns_count}")
            
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

        # The policy_sdgs_for_episode have been updated in-place with MC returns for chosen actions.
        # BaseEnv will call our overridden postprocess_histories with this data.
        return policy_sdgs_for_episode, [] # Return policy data; backlog is empty for this env structure.

    async def postprocess_histories(
        self,
        trajectories: Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]],
        # rollout_info: List[Dict[str, Any]] # This is not passed by BaseEnv.handle_env
    ) -> Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]:
        """
        Post-processes policy agent trajectories.
        In this TextWorldEnv implementation, the core reward processing (MC returns)
        is done directly at the end of `collect_trajectories` before returning
        the policy ScoredDataGroups. This method can be a passthrough or for
        any final adjustments if needed.
        """
        # The main reward processing for policy agent data (calculating MC discounted returns
        # for chosen actions) is now handled at the end of the `collect_trajectories` method
        # to ensure all episode information (like ep_state.canonical_rewards) is available.
        # This method is called by BaseEnv *after* `collect_trajectories` returns.
        
        if not trajectories:
            logger.debug("postprocess_histories received no trajectories. Passthrough.")
            return trajectories

        # Ensure it's a list for consistent logging/processing, though it should be List[ScoredDataGroup]
        # from collect_trajectories if an episode ran.
        num_groups_to_log = 0
        if isinstance(trajectories, list):
            num_groups_to_log = len([t for t in trajectories if t is not None])
        elif trajectories is not None : # Single ScoredDataGroup
            num_groups_to_log = 1
            
        logger.debug(f"TextWorldEnv.postprocess_histories: Received {num_groups_to_log} ScoredDataGroup(s) for policy. Rewards already processed in collect_trajectories. Passthrough.")
        
        # For now, this is a passthrough as rewards are processed in collect_trajectories.
        return trajectories

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
            
            # atropos_agent_config is now used directly
            atropos_agent_config=AtroposAgentConfig( # Pass any overrides here if needed
                model_id="NousResearch/Hermes-2-Pro-Llama-3-8B" # Ensure model_id matches server
            ), 
            # rm_config is now used directly
            atropos_rm_config=AtroposRMConfig( # Ensure RM also gets a model_id if not covered by default logic
                thinking=True,
                model_id="NousResearch/Hermes-2-Pro-Llama-3-8B" # Default to same model for now
            ),      
            G_policy_alternatives=4, # This should align with how agent.generate_action is called 
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
    # Ensure asyncio event loop is managed correctly if TextWorldEnv.cli() is async
    # For direct script execution, it might be simpler to wrap in an async function and run
    async def main_cli():
        await TextWorldEnv.cli()

    asyncio.run(main_cli()) # Use asyncio.run if TextWorldEnv.cli() is async
