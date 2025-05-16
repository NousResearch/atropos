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

    # Policy search and evaluation parameters
    G_policy_alternatives: int = Field(
        default=3, 
        description="Number of policy alternatives to generate at each step."
    )
    G_rm_judgements: int = Field(
        default=1, 
        description="Number of RM judgements to sample per policy alternative."
    )
    
    # Reward/Value Learning Parameters
    # reward_function_name: str = "textworld_goal_reward" # Example, to be defined
    # reward_model_name: Optional[str] = None # If using a separate reward model
    rm_reward_discount_factor: float = Field(
        default=0.99, 
        description="Discount factor for future game rewards when training RM."
    )

    # tokenizer_name: str = "gpt-4" # From BaseEnvConfig
    # group_size: int = 1 # From BaseEnvConfig (related to batching for data collection)
    # use_wandb: bool = Field(default=False, description="Enable W&B logging for the environment.") # From BaseEnvConfig

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

        # Comment from before regarding agent.start_new_game_dialogue() adding system prompt is noted,
        # but ep_state.message_history is the direct input to agent.generate_action's history_for_llm_call parameter,
        # so it needs to be complete and start with the system prompt here.


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
        super().__init__(config, server_configs, slurm, testing) # BaseEnv populates self.server_clients
        self.config: TextWorldEnvConfig = config 
        self.episodes: Dict[str, TextWorldEpisodeState] = {}
        self._temp_dir = tempfile.mkdtemp(prefix="textworld_env_")
        logger.info(f"TextWorldEnv created temporary directory: {self._temp_dir}")

        # Determine Policy Agent server configuration
        effective_policy_agent_server_config = self.config.policy_agent_server_config or self.config.default_server_config
        if not effective_policy_agent_server_config:
            raise ValueError("No server configuration found for the Policy Agent (checked policy_agent_server_config and default_server_config).")
        
        policy_agent_client = self.server_clients.get(effective_policy_agent_server_config.model_name)
        if not policy_agent_client:
            raise ValueError(
                f"LLM client for Policy Agent model '{effective_policy_agent_server_config.model_name}' not found in server_clients. "
                f"Ensure it was provided in the initial server_configs list to TextWorldEnv."
            )

        # Ensure AtroposAgentConfig is instantiated
        agent_cfg = self.config.atropos_agent_config if self.config.atropos_agent_config is not None else AtroposAgentConfig()

        self.agent = AtroposAgent(
            server_client=policy_agent_client, 
            tokenizer=self.tokenizer, 
            config=agent_cfg
        )
        # Store the system prompt that will be used for the policy agent message history
        self.policy_agent_system_prompt_content = agent_cfg.action_generation_system_prompt
        logger.info(f"TextWorldEnv: Policy Agent initialized with model '{effective_policy_agent_server_config.model_name}'. System Prompt: '{self.policy_agent_system_prompt_content[:100]}...'")


        # Determine RM server configuration
        effective_rm_server_config = self.config.rm_agent_server_config or self.config.default_server_config
        if not effective_rm_server_config:
            raise ValueError("No server configuration found for the RM Agent (checked rm_agent_server_config and default_server_config).")

        rm_agent_client = self.server_clients.get(effective_rm_server_config.model_name)
        if not rm_agent_client:
            logger.warning(
                f"LLM client for RM Agent model '{effective_rm_server_config.model_name}' not found in server_clients. "
                f"Attempting to fall back to Policy Agent's client."
            )
            # Fallback to policy agent's client if RM's specific client is missing
            rm_agent_client = policy_agent_client 
            if rm_agent_client:
                 logger.info(f"RM Agent will use the Policy Agent's LLM client (model: '{effective_policy_agent_server_config.model_name}').")
            else: # Should not happen if policy_agent_client was resolved
                 raise ValueError(f"LLM client for RM Agent model '{effective_rm_server_config.model_name}' not found, and fallback to policy agent client also failed.")
        
        # Ensure AtroposRMConfig is instantiated
        rm_cfg = self.config.atropos_rm_config if self.config.atropos_rm_config is not None else AtroposRMConfig()

        self.rm = AtroposRM(
            server_client=rm_agent_client,
            tokenizer=self.tokenizer,
            config=rm_cfg
        )
        logger.info(f"TextWorldEnv: RM Agent initialized with model '{effective_rm_server_config.model_name}'.")

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

        game_options = {
            "nb_rooms": self.config.nb_rooms,
            "nb_objects": self.config.nb_objects,
            "chaining.min_length": self.config.quest_min_length,
            "chaining.max_length": self.config.quest_max_length,
            "chaining.max_depth": self.config.quest_max_depth,
            "grammar.theme": self.config.grammar_theme,
            "grammar.include_adj": self.config.grammar_include_adj,
            # Add challenge specific options if challenge_name is set
        }
        if self.config.challenge_name:
            challenge = textworld.challenges.CHALLENGES[self.config.challenge_name]
            game_options.update(challenge.options) # Challenge options can override defaults
            # Apply challenge-specific reward/goal settings
            game_options["rewards.dense"] = self.config.challenge_rewards == "dense"
            game_options["rewards.balanced"] = self.config.challenge_rewards == "balanced"
            game_options["rewards.sparse"] = self.config.challenge_rewards == "sparse"
            game_options["goal.detailed"] = self.config.challenge_goal == "detailed"
            game_options["goal.brief"] = self.config.challenge_goal == "brief"
            game_options["goal.none"] = self.config.challenge_goal == "none"
            if self.config.challenge_test_mode:
                game_options["distributions.test"] = True # Use test distribution if specified
        
        # Ensure GameOptions is created before passing to generate_textworld_game
        # The function generate_textworld_game expects a GameOptions object or a dict
        # For clarity, let's create GameOptions explicitly if the utility expects it,
        # otherwise, the dict `game_options` is fine if the utility handles dicts.
        # Assuming generate_textworld_game can handle a dict for options for now.

        game_file_path = generate_textworld_game(
            output_dir=self._temp_dir,
            game_options_dict=game_options,
            game_seed=current_game_seed,
            challenge_name=self.config.challenge_name or "custom" # Provide a default if no challenge
        )

        if not game_file_path:
            logger.error(f"Failed to generate game file for episode {episode_id} with seed {current_game_seed}")
            return None

        requested_infos = EnvInfos(
            description=True, inventory=True, objective=True, score=True, 
            max_score=True, has_won=True, has_lost=True, facts=True, 
            last_action=True, feedback=True, moves=True, inadmissible_commands=True
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

    async def get_next_item(self) -> Optional[TextWorldEpisodeState]: # Changed return type
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

        return episode_state

    def _parse_action(self, agent_response_text: str) -> Optional[str]:
        """
        Parses the agent's response to extract the TextWorld command string.
        Assumes agent is prompted for direct command output (no tool calls for TextWorld actions).
        """
        if not agent_response_text:
            logger.warning("[TextWorldEnv._parse_action] Received empty or None agent response text.")
            return None

        # The agent is prompted to output ONLY the command. Any thinking/XML tags should have been stripped by agent/LLM,
        # or if not, the environment might treat complex outputs as parse failures.
        # For TextWorld, the command is expected to be a simple string.
        # We strip whitespace. Could add more cleaning if LLM sometimes adds quotes or minor fluff.
        
        # Check for common LLM-added prefixes/suffixes if necessary, e.g., if it sometimes says "Command: go north"
        # For now, simple strip assumes clean output as per prompt.
        parsed_command = agent_response_text.strip()

        if not parsed_command:
            logger.warning(
                f"[TextWorldEnv._parse_action] Agent response stripped to empty string. Original: '{agent_response_text[:200]}...'")
            return None
        
        # Optional: Validate against admissible commands if available and strictness is needed, though
        # TextWorld itself will handle invalid commands.
        # infos = ep_state.textworld_env.infos # (Would need ep_state here, or pass infos)
        # if 'admissible_commands' in infos and parsed_command not in infos['admissible_commands']:
        #     logger.warning(f"Parsed command '{parsed_command}' not in admissible commands: {infos['admissible_commands']}")
            # Decide if this is an error or let TextWorld handle it.

        logger.debug(f"[TextWorldEnv._parse_action] Parsed command: '{parsed_command}'")
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
        for i in range(self.config.G_policy_alternatives):
            # Each call to generate_action is independent for now. 
            # Agent uses its internal history, which includes previous chosen actions and memories.
            # For generating diverse alternatives, we might need to explore more advanced sampling techniques
            # or have the agent itself support generating N diverse options from the same core history.
            # Current AtroposAgent.generate_action produces one action.
            agent_action_tasks.append(self.agent.generate_action(
                observation_content=current_observation_content, 
                game_history_window=list(game_history_for_agent) # Pass a copy
            ))
        
        try:
            generated_actions_results = await asyncio.gather(*agent_action_tasks)
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
        ep_state.won = infos.get("has_won", False)
        ep_state.lost = infos.get("has_lost", False)
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
        self, item: Dict[str, Any]
    ) -> Tuple[List[ScoredDataGroup], List[Tuple[int, int]]]:
        """Generate a game using the utility, run episode, collect trajectory."""
        if not all(k in item for k in ["challenge_name", "challenge_settings", "game_options_dict"]):
            logger.error(f"Invalid item received in collect_trajectories. Missing required keys. Item: {item}")
            return [], []

        challenge_name = item["challenge_name"]
        challenge_settings = item["challenge_settings"]
        base_options_dict = item["game_options_dict"]
        seed = challenge_settings.get("seed") # Seed must be in settings

        if seed is None:
             logger.error("Seed missing from challenge_settings in item.")
             return [], []

        # --- 1. Game Generation using Utility --- 
        # Create GameOptions object from the dictionary
        options = GameOptions()
        options.seeds = seed # Ensure seed is set
        # Set base structural parameters on options; grammar details are often handled by challenge itself
        options.nb_rooms = base_options_dict.get("nb_rooms", self.config.nb_rooms)
        options.nb_objects = base_options_dict.get("nb_objects", self.config.nb_objects)
        options.chaining.min_length = base_options_dict.get("quest_min_length", self.config.quest_min_length)
        options.chaining.max_length = base_options_dict.get("quest_max_length", self.config.quest_max_length)
        options.chaining.max_depth = base_options_dict.get("quest_max_depth", self.config.quest_max_depth)
        # Removed grammar settings like theme, include_adj, verb_prob here - handled by challenge/settings dict
        # options.grammar.theme = base_options_dict.get("grammar_theme", self.config.grammar_theme)
        # options.grammar.include_adj = base_options_dict.get("grammar_include_adj", self.config.grammar_include_adj)
        # options.grammar.verb_prob = base_options_dict.get("grammar_verb_prob", self.config.grammar_verb_prob) # <-- REMOVED
        
        # Utility function handles filename and path creation within temp dir
        game_file_path, game_object = generate_textworld_game(
            challenge_name=challenge_name,
            settings=challenge_settings,
            options=options, # Pass the constructed GameOptions
            output_folder=self._temp_dir, # Generate in the env's temp dir
            filename_prefix=f"{challenge_name}_ep{seed}" # More descriptive prefix
        )

        if not game_file_path or not game_object:
            logger.error(f"Failed to generate/compile game for seed {seed} using challenge '{challenge_name}'. See generation_utils logs.")
            # Utility function already logs details
            return [], [] # Return empty trajectory

        # Use the file path returned by the utility
        env_id = f"textworld-env-{uuid.uuid4().hex}"
        # Pass the original dicts for state logging if needed, but main state uses file_path
        ep = TextWorldEpisodeState(env_id, game_file_path, None, "", {}, 0, "") 

        # --- 2. Environment Setup (Remains largely the same) --- 
        try:
            request_infos = EnvInfos()
            request_infos.description = True
            request_infos.inventory = True
            request_infos.objective = True
            request_infos.score = True
            request_infos.feedback = True
            request_infos.won = True
            request_infos.lost = True
            request_infos.max_score = True
            
            registered_env_id = textworld.gym.register_games(
                [game_file_path], # Use the path from the utility
                request_infos=request_infos,
                max_episode_steps=self.config.max_steps,
                name=env_id
            )
            logger.info(f"Registered gym environment: {registered_env_id} (using {env_id})")

            ep.textworld_env = textworld.gym.make(registered_env_id)
            logger.info(f"Gym environment created.")

            obs, infos = ep.textworld_env.reset()
            ep.last_score = infos.get('score', 0)
            infos['_last_score'] = ep.last_score

            ep.message_history = [{"role": "system", "content": self.system_prompt}]
            formatted_initial_obs = self._format_observation(obs, infos)
            ep.message_history.append({"role": "environment", "content": formatted_initial_obs})
            logger.debug(f"Initial observation formatted:\n{formatted_initial_obs}")

        except Exception as e:
            logger.error(f"Failed to setup gym environment for {game_file_path}: {e}", exc_info=True)
            ep.done = True
            if ep.textworld_env:
                try: ep.textworld_env.close()
                except Exception: pass
            # Utility creates the file, so cleanup needs to happen here too on setup failure
            if os.path.exists(game_file_path):
                try: os.remove(game_file_path)
                except OSError: pass
            return [], []
        
        # --- 3. Interaction Loop (Remains the same) ---
        for step_num in range(self.config.max_steps):
            logger.info(f"--- Episode {seed}, Step {step_num+1}/{self.config.max_steps} ---")
            # Build the prompt
            messages = ep.message_history.copy()
            if self.config.thinking_active:
                messages.append({"role": "agent", "content": self.config.thinking_prefill})
            else:
                messages.append({"role": "agent", "content": ""})

            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)

            # Generate candidate completions
            completions = await self.server.completion(
                prompt=prompt,
                n=self.config.group_size,
                max_tokens=self.config.max_token_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )

            # --- Evaluate Alternatives ---
            alt_actions: List[Optional[str]] = [] # Parsed command strings
            alt_scores: List[float] = []
            alt_tokens: List[List[int]] = []
            alt_masks: List[List[int]] = []
            alt_messages: List[List[Message]] = []
            alt_responses: List[str] = []
            alt_format_rewards: List[float] = [0.0] * len(completions.choices)

            # Calculate format/tool scores first
            format_completions_for_scorer = []
            for choice in completions.choices:
                response = choice.text if hasattr(choice, "text") else getattr(choice.message, "content", "")
                full_response = self.config.thinking_prefill + response if self.config.thinking_active else response
                alt_responses.append(full_response)
                format_completions_for_scorer.append([{"role": "assistant", "content": full_response}])

            if self.reward_function:
                try:
                    format_rewards = self.reward_function(format_completions_for_scorer)
                    if format_rewards and len(format_rewards) == len(alt_responses):
                         alt_format_rewards = format_rewards
                    else:
                         logger.warning(f"Reward function returned unexpected number of scores: {len(format_rewards) if format_rewards else 'None'} vs {len(alt_responses)}")
                except Exception as e:
                     logger.error(f"Error applying reward function: {e}", exc_info=True)

            logger.debug(f"Alternative Format/Tool Rewards: {alt_format_rewards}")

            # Process each alternative for parsing and scoring (for selection)
            for i, response in enumerate(alt_responses):
                # Parse action from response
                action_str = self._parse_action(response)
                alt_actions.append(action_str)

                # Calculate score for selection (Format/Tool Score + Parse Penalty)
                current_score = alt_format_rewards[i] # Start with format/tool score
                if action_str is None:
                    current_score += self.config.invalid_action_penalty
                    ep.num_invalid_actions += 1 # Track invalid parses globally for the episode

                alt_scores.append(current_score)

                # Tokenize for training data
                step_msgs: List[Message] = [
                    {"role": m["role"], "content": m["content"]} for m in ep.message_history
                ]
                step_msgs.append({"role": "agent", "content": response})

                try:
                    out = tokenize_for_trainer(self.tokenizer, step_msgs)
                    alt_tokens.append(out["tokens"])
                    alt_masks.append(out["masks"])
                    alt_messages.append(step_msgs)
                except Exception as e:
                     logger.error(f"Error tokenizing alternative {i}: {e}", exc_info=True)
                     # Handle tokenization error - add dummy data? Or discard?
                     # For now, adding placeholders to keep lists aligned, but this might need review
                     alt_tokens.append([])
                     alt_masks.append([])
                     alt_messages.append(step_msgs) # Keep messages for context if needed
                     alt_scores[i] = -float('inf') # Penalize heavily if tokenization fails

            # --- Select Best Action and Step Environment ---
            if not alt_scores: # Should not happen if completions > 0
                 logger.error("No scores generated for alternatives. Aborting step.")
                 break # Or handle differently?

            best_idx = int(max(range(len(alt_scores)), key=lambda i: alt_scores[i]))
            best_action_str = alt_actions[best_idx]
            best_response = alt_responses[best_idx]
            best_format_reward = alt_format_rewards[best_idx]

            # Ensure we have a valid action string to send
            if best_action_str is None:
                logger.warning(f"Best action selected (idx {best_idx}) resulted in None command string. Sending 'wait' instead.")
                best_action_str = "wait"

            logger.info(f"Selected action (idx {best_idx}, score {alt_scores[best_idx]:.3f}): '{best_action_str}'")

            # Step the main environment with the chosen action
            try:
                obs, score, done, infos = ep.textworld_env.step(best_action_str)
                logger.debug(f"Env Step Output: Score={score}, Done={done}, Infos={infos}")
            except Exception as e:
                 logger.error(f"Error stepping environment with action '{best_action_str}': {e}", exc_info=True)
                 # Decide how to proceed: Abort episode? Treat as failed step?
                 ep.done = True # Mark episode as aborted due to env error
                 break

            # --- Update State and History ---
            ep.actions_taken.append(best_action_str)

            # Calculate actual environment reward for this step
            actual_env_reward = score # In TextWorld, score often represents the step reward directly
            # Alternative: Use change in score? Check `infos` for reward signal if `score` isn't it.
            # actual_env_reward = infos.get('score', 0) - ep.last_score
            # Check for win/loss which might provide additional reward signal
            if infos.get('won'): actual_env_reward += 1.0 # Example win bonus
            if infos.get('lost'): actual_env_reward -= 1.0 # Example loss penalty

            ep.step_env_rewards.append(actual_env_reward)
            ep.step_format_rewards.append(best_format_reward)

            # Calculate combined reward for the step taken
            combined_reward = (self.config.environment_reward_weight * actual_env_reward) + \
                              (self.config.format_reward_weight * best_format_reward)
            ep.step_combined_rewards.append(combined_reward)

            # Update episode state based on infos
            ep.last_score = infos.get('score', ep.last_score)
            ep.won = infos.get('won', False)
            ep.lost = infos.get('lost', False)
            infos['_last_score'] = ep.last_score # Pass for formatting

            # Add agent response and env feedback to history
            ep.message_history.append({"role": "agent", "content": best_response})
            formatted_obs = self._format_observation(obs, infos)
            ep.message_history.append({"role": "environment", "content": formatted_obs})

            # Store the scored group for this step
            # The scores here are the selection scores (format/parse)
            ep.trajectory.append(
                ScoredDataGroup(
                    tokens=alt_tokens,
                    masks=alt_masks,
                    scores=alt_scores, # Use the selection scores
                    messages=alt_messages,
                    parsed_action=best_action_str, # Store the chosen action string
                )
            )

            logger.info(f" Step Reward: Env={actual_env_reward:.3f}, Fmt={best_format_reward:.3f}, Comb={combined_reward:.3f}")
            logger.info(f" Episode State: Total Score={ep.last_score}, Won={ep.won}, Lost={ep.lost}")

            if done:
                logger.info(f"Episode finished. Won={ep.won}, Lost={ep.lost}, Score={ep.last_score}")
                break
        # --- End Interaction Loop ---

        # --- 4. Cleanup for Episode (Remains the same) ---
        final_status = "Completed"
        if ep.won: final_status = "Won"
        elif ep.lost: final_status = "Lost"
        elif ep.done: final_status = "Aborted"
        elif step_num == self.config.max_steps - 1: final_status = "Max Steps Reached"

        logger.info(f"Episode {seed} ended. Status: {final_status}, Final Score: {ep.last_score}, Steps: {len(ep.actions_taken)}, Invalid Parses: {ep.num_invalid_actions}")

        if ep.textworld_env:
            try:
                ep.textworld_env.close()
                logger.debug(f"Closed gym environment for {ep.env_id}")
            except Exception as e:
                logger.warning(f"Error closing gym environment {ep.env_id}: {e}")

        # Unregister? Textworld gym doesn't seem to have a clean public unregister.
        # Relying on unique env_ids per run is likely sufficient.

        if os.path.exists(ep.game_file):
            try:
                os.remove(ep.game_file)
                logger.debug(f"Removed temporary game file: {ep.game_file}")
            except OSError as e:
                logger.warning(f"Error removing temporary game file {ep.game_file}: {e}")

        return ep.trajectory, [] # Return trajectory data

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

        super().cleanup()

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
            group_size=cls.DEPRECATED_GROUP_SIZE, # group_size from BaseEnvConfig is now G_policy_alternatives etc.
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
                model_name="gpt-4.1-mini", # For Reward Model (example, could be same as agent)
                base_url="http://localhost:9005/v1", # Example different endpoint for RM
                api_key="EMPTY", 
            )
        ]
        return env_config, server_configs

if __name__ == "__main__":
    TextWorldEnv.cli()
