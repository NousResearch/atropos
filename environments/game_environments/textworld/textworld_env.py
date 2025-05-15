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
from textworld.generator import (  # Import compile_game
    QuestGenerationError,
    compile_game,
)
from textworld.generator.text_grammar import MissingTextGrammar

from helpers.tool_call_parser import parse_tool_call
from trajectoryhandler.envs.base import (
    BaseEnv,
    BaseEnvConfig,
    OpenaiConfig,
    ScoredDataGroup,
)
from trajectoryhandler.envs.reward_fns import registry
from trajectoryhandler.envs.reward_fns.combined_reward import CombinedReward
from trajectoryhandler.type_definitions import Message
from trajectoryhandler.utils.tokenize_for_trainer import tokenize_for_trainer

# Import the generation utility
from environments.game_environments.textworld.generation_utils import generate_textworld_game

logger = logging.getLogger(__name__)


# Placeholder for Config Class
class TextWorldEnvConfig(BaseEnvConfig):
    """
    Configuration for the TextWorld environment trainer.
    """

    env_name: str = "TextWorld"
    temperature: float = 0.7
    top_p: float = 0.9
    max_steps: int = 50  # Max turns per episode

    # Challenge Specification
    challenge_name: str = "tw-simple" # Name of the challenge to generate (e.g., tw-simple)
    challenge_rewards: str = "balanced" # Challenge-specific: dense, balanced, sparse
    challenge_goal: str = "brief" # Challenge-specific: detailed, brief, none
    challenge_test_mode: bool = False # Challenge-specific: Use test distribution

    # Game Generation Parameters (Defaults used if not overridden by challenge)
    nb_rooms: int = 5
    nb_objects: int = 10
    quest_min_length: int = 3 # Corresponds to chaining.min_length
    quest_max_length: int = 3 # Corresponds to chaining.max_length
    quest_max_depth: int = 3 # Corresponds to chaining.max_depth
    grammar_theme: str = "house"
    grammar_include_adj: bool = True
    game_seed: Optional[int] = None # If None, random seed is used per episode

    # Thinking configuration
    thinking_active: bool = True
    thinking_prefill: str = "<think>\n"

    # Reward function configuration
    reward_functions: List[Union[str, Dict[str, Any]]] = ["tool_calling"]
    format_reward_weight: float = 0.3 # Weight for tool calling/format rewards
    environment_reward_weight: float = 0.7 # Weight for game score changes, winning, etc.
    invalid_action_penalty: float = -0.1 # Penalty for invalid/malformed actions


# Placeholder for EpisodeState Class
class EpisodeState:
    """
    Stores per-episode state for a TextWorld game.
    """

    def __init__(self, seed: int, game_options: Dict[str, Any], game_file: str, env_id: str):
        self.seed: int = seed
        self.game_options: Dict[str, Any] = game_options
        self.game_file: str = game_file # Path to the temporary .z8 file
        self.env_id: str = env_id # Unique ID for textworld.gym registration
        self.env: Optional[textworld.gym.Env] = None # Initialized in collect_trajectory
        self.message_history: List[Message] = []
        self.actions_taken: List[str] = [] # Stores the command strings sent to env
        self.step_env_rewards: List[float] = [] # Raw reward from env.step()
        self.step_format_rewards: List[float] = [] # Reward from format/tool scorer
        self.step_combined_rewards: List[float] = [] # Weighted combination
        self.trajectory: List[ScoredDataGroup] = []
        self.last_score: float = 0.0 # Track score changes
        self.won: bool = False
        self.lost: bool = False
        self.aborted: bool = False # If generation or setup failed
        self.num_invalid_actions: int = 0

# Placeholder for TextWorldEnv Class
class TextWorldEnv(BaseEnv):
    """
    Trainer environment for TextWorld using a best-of-n approach with function-call style actions.
    """

    def __init__(
        self,
        config: TextWorldEnvConfig,
        server_configs: List[OpenaiConfig],
        slurm: bool = True,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.episodes: Dict[int, EpisodeState] = {}
        self._temp_dir = tempfile.mkdtemp(prefix="textworld_env_")
        logger.info(f"TextWorldEnv created temporary directory: {self._temp_dir}")

        # Define the single function-calling tool for executing commands
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "execute_command",
                    "description": "Execute a text command in the adventure game.",
                    "parameters": {
                        "command_string": {
                            "type": "string", 
                            "description": "The full text command to execute (e.g., 'go north', 'take shiny key', 'open wooden door with rusty key')."
                        }
                    },
                },
            }
        ]

        # Initialize reward function
        self.reward_function = self._initialize_reward_function()

        tools_json = json.dumps(self.tools, indent=2)
        # Update system prompt for the single tool
        self.system_prompt = (
            "You are an AI agent playing a text-based adventure game. Your goal is to follow the objective described "
            "at the start of the game. You interact with the world by providing text commands."
            "\n\n"
            "Carefully observe the room descriptions, your inventory, and any feedback from your previous actions. "
            "Think step-by-step about how to achieve the objective."
            "\n\n"
            "You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your "
            "chosen text command using the 'execute_command' function call."
            "\n\n"
            f"<tools>\n{tools_json}\n</tools>\n\n"
            "For your function call, return a JSON object with function name ('execute_command') and the command string argument "
            "within <tool_call> </tool_call> tags. Example format:"
            '<tool_call>\n{"arguments": {"command_string": "go north"}, "name": "execute_command"}\n</tool_call>\n\n'
            "Your answer format MUST be:\n"
            "<think>\n"
            "[Your detailed reasoning process about the current situation and your plan.]\n"
            "I will try to [describe action, e.g., go north, take the key].\n"
            "</think>\n\n"
            '<tool_call>\n{"arguments": {"command_string": "your text command here"}, "name": "execute_command"}\n</tool_call>'
        )

    def _initialize_reward_function(self):
        """Initialize the combined reward function for scoring agent responses."""
        if not hasattr(self.config, "reward_functions") or not self.config.reward_functions:
            return None

        reward_configs = []
        for reward_func_config in self.config.reward_functions:
            if isinstance(reward_func_config, str):
                # Handle predefined string names
                if reward_func_config == "tool_calling":
                    tool_calling_config = {
                        "type": "tool_calling",
                        "weight": 1.0, # Weight is applied later during combination
                        "params": {
                            "tools": self.tools,
                            "preferred_tags": ["tool_call"],
                            "check_arguments": True, # Ensure args match schema
                            "allow_extra_params": False,
                        },
                    }
                    reward_configs.append(tool_calling_config)
                elif reward_func_config == "format":
                    # Add format checker if needed, ensures <think> and <tool_call>
                     format_config = {
                         "type": "format",
                         "weight": 1.0,
                         "params": {
                             "preferred_tags": ["think", "tool_call"],
                             "required_tags": ["think", "tool_call"] # Ensure both are present
                         }
                     }
                     reward_configs.append(format_config)
                else:
                    # Assume it's a registered reward function name
                    reward_configs.append({"type": reward_func_config, "weight": 1.0})
            elif isinstance(reward_func_config, dict):
                # Pass through dictionary configuration
                reward_configs.append(reward_func_config)
            else:
                 logger.warning(f"Unsupported reward function configuration type: {type(reward_func_config)}")

        if not reward_configs:
            return None
        elif len(reward_configs) == 1:
            # If only one, weight is applied later, create directly
            config = reward_configs[0]
            config.pop('weight', None) # Remove weight if present, handled later
            return registry.create(config)
        else:
            # Use CombinedReward for multiple functions, weights handled later
            for config in reward_configs:
                 config.pop('weight', None)
            return CombinedReward(rewards=reward_configs, normalization="none")

    async def setup(self):
        """Ensure temporary directory exists."""
        os.makedirs(self._temp_dir, exist_ok=True)
        logger.info("TextWorldEnv setup complete.")

    async def get_next_item(self) -> Dict[str, Any]:
        """Generate game generation settings for a new episode."""
        seed = self.config.game_seed if self.config.game_seed is not None else random.randint(0, 2**31 - 1)
        
        # Prepare the settings dictionary needed by the generation utility
        settings = {
            'seed': seed,
            'rewards': self.config.challenge_rewards,
            'goal': self.config.challenge_goal,
            'test': self.config.challenge_test_mode,
            # Add any other settings required by the specific challenge's make function here
            # based on its signature, potentially drawing from self.config
        }
        
        # Prepare the base GameOptions dictionary (optional, could be created within utility)
        # These might be used by compile_game or specific challenges
        options_dict = {
            "nb_rooms": self.config.nb_rooms,
            "nb_objects": self.config.nb_objects,
            "quest_min_length": self.config.quest_min_length,
            "quest_max_length": self.config.quest_max_length,
            "quest_max_depth": self.config.quest_max_depth,
            "grammar_theme": self.config.grammar_theme,       # Theme might still be useful base option
            "grammar_include_adj": self.config.grammar_include_adj # Include_adj might still be useful
        }
        
        logger.debug(f"Generated item for next episode: challenge='{self.config.challenge_name}', settings={settings}")
        
        # Return structure expected by collect_trajectories
        return {
            "challenge_name": self.config.challenge_name,
            "challenge_settings": settings,
            "game_options_dict": options_dict, # Pass base options separately
        }

    def _format_observation(self, obs: str, infos: Dict[str, Any]) -> str:
        """Format the TextWorld observation and info for the LLM."""
        # Basic formatting, can be enhanced
        formatted = f"Current Location/Situation:\n{obs}\n\n"
        if 'inventory' in infos and infos['inventory']:
            formatted += f"Inventory: {infos['inventory']}\n"
        if 'description' in infos and infos['description'] != obs: # Avoid redundancy if obs is already description
             formatted += f"Description: {infos['description']}\n"
        # Add score, feedback if available and changed
        score = infos.get('score', 0)
        last_score = infos.get('_last_score', 0) # Need to store this in EpisodeState
        if score != last_score:
             formatted += f"Score: {score} (Change: {score - last_score:+})\n"
        if 'feedback' in infos and infos['feedback']:
             formatted += f"Last Action Feedback: {infos['feedback']}\n"

        # Include objective if available (usually only at the start)
        if 'objective' in infos and infos['objective']:
            formatted += f"\nObjective: {infos['objective']}\n"

        # You might want to add admissible commands if needed for prompting/debugging
        # if 'admissible_commands' in infos:
        #     formatted += f"\nPossible Actions: {', '.join(infos['admissible_commands'])}"

        return formatted.strip()

    def _parse_tool_call(self, response: str) -> Tuple[Optional[str], Optional[Dict], bool, Optional[str]]:
        """Parse the tool call from LLM response and extract the command string."""
        tool_name, arguments, is_error = parse_tool_call(
            response, self.tools, ["tool_call"]
        )

        if is_error or tool_name != "execute_command":
            logger.warning(f"Failed to parse tool call or incorrect tool name found (expected 'execute_command'): {response}")
            return None, None, True, tool_name # Action string (None), args, is_error, tool_name

        command_str = arguments.get("command_string")
        
        if not command_str or not isinstance(command_str, str) or not command_str.strip():
             logger.warning(f"Missing or invalid 'command_string' argument in execute_command call: {arguments}")
             return None, arguments, True, tool_name # Invalid command string

        command_str = command_str.strip() # Clean whitespace
        logger.debug(f"Parsed command string: '{command_str}'")
        return command_str, arguments, False, tool_name

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
        ep = EpisodeState(seed, base_options_dict, game_file_path, env_id) 

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

            ep.env = textworld.gym.make(registered_env_id)
            logger.info(f"Gym environment created.")

            obs, infos = ep.env.reset()
            ep.last_score = infos.get('score', 0)
            infos['_last_score'] = ep.last_score

            ep.message_history = [{"role": "system", "content": self.system_prompt}]
            formatted_initial_obs = self._format_observation(obs, infos)
            ep.message_history.append({"role": "environment", "content": formatted_initial_obs})
            logger.debug(f"Initial observation formatted:\n{formatted_initial_obs}")

        except Exception as e:
            logger.error(f"Failed to setup gym environment for {game_file_path}: {e}", exc_info=True)
            ep.aborted = True
            if ep.env:
                try: ep.env.close()
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
                action_str, _, parse_error, _ = self._parse_tool_call(response)
                alt_actions.append(action_str)

                # Calculate score for selection (Format/Tool Score + Parse Penalty)
                current_score = alt_format_rewards[i] # Start with format/tool score
                if parse_error or action_str is None:
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
                obs, score, done, infos = ep.env.step(best_action_str)
                logger.debug(f"Env Step Output: Score={score}, Done={done}, Infos={infos}")
            except Exception as e:
                 logger.error(f"Error stepping environment with action '{best_action_str}': {e}", exc_info=True)
                 # Decide how to proceed: Abort episode? Treat as failed step?
                 ep.aborted = True # Mark episode as aborted due to env error
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
        elif ep.aborted: final_status = "Aborted"
        elif step_num == self.config.max_steps - 1: final_status = "Max Steps Reached"

        logger.info(f"Episode {seed} ended. Status: {final_status}, Final Score: {ep.last_score}, Steps: {len(ep.actions_taken)}, Invalid Parses: {ep.num_invalid_actions}")

        if ep.env:
            try:
                ep.env.close()
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
