#!/usr/bin/env python3
"""
BlackjackEnv: Trainer environment for Gymnasium Blackjack

This wraps Gymnasium's Blackjack-v1 environment to train an LLM via a best-of-n pattern
using function-call style actions. Extends BaseEnv.
"""

import json
import logging
import os
import yaml
from typing import Any, Dict, List, Optional, Tuple, Union
import random
import gymnasium
from tqdm.asyncio import tqdm_asyncio

from atroposlib.utils.tool_call_parser import parse_tool_call
from atroposlib.envs.base import (
    BaseEnv,
    BaseEnvConfig,
    OpenaiConfig,
    ScoredDataGroup,
)
from atroposlib.envs.reward_fns import registry
from atroposlib.envs.reward_fns.combined_reward import CombinedReward
from atroposlib.type_definitions import Message
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

logger = logging.getLogger(__name__)


class BlackjackEnvConfig(BaseEnvConfig):
    """
    Configuration for the Blackjack environment trainer.
    """

    env_name: str = "Blackjack-v1"
    temperature: float = 0.7
    top_p: float = 0.9
    max_turns: Optional[int] = 5
    wandb_name: str = "blackjack"

    # Thinking configuration
    thinking_active: bool = True

    # Evaluation configuration
    eval_episodes: int = 100

    # Reward function configuration
    reward_functions: List[Union[str, Dict[str, Any]]] = []
    format_reward_weight: float = 0.2
    environment_reward_weight: float = 0.8

    # Batch size for this environment
    batch_size: int = 1024

    # Max characters for thinking blocks in history prompts
    max_think_chars_history: int = 3000
    
    # Max tokens for a full trajectory (to prevent exceeding context limits)
    max_trajectory_tokens: int = 24576

    # Debug mode
    debug_mode: bool = False


class BlackjackScoredDataGroup(ScoredDataGroup):
    """
    Represents the scored data for a single step in a Blackjack trajectory, potentially including multiple alternatives.
    """
    seed: int
    tokens: Optional[List[List[int]]] = None
    masks: Optional[List[List[int]]] = None
    scores: Optional[List[float]] = None
    messages: Optional[List[List[Message]]] = None
    parsed_action: Optional[int] = None # Store the chosen action (0=stick, 1=hit, -1=error)


class EpisodeState:
    """
    Stores per-episode state: gym env, history, actions, rewards, trajectory.
    """

    def __init__(self, seed: int, env: gymnasium.Env):
        self.seed: int = seed
        self.env: gymnasium.Env = env
        self.message_history: List[Message] = []
        self.actions: List[int] = []
        self.step_rewards: List[float] = []
        self.trajectory: List[BlackjackScoredDataGroup] = []
        # Add tracking for total score
        self.total_env_reward: float = 0.0
        self.total_format_reward: float = 0.0
        self.total_combined_reward: float = 0.0
        self.num_correct_actions: int = 0
        self.num_total_actions: int = 0


class BlackjackEnv(BaseEnv):
    """
    Trainer environment for Gymnasium Blackjack using a best-of-n approach with function-call style actions.
    """

    def __init__(
        self,
        config: BlackjackEnvConfig,
        server_configs: List[OpenaiConfig],
        slurm: bool = True,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.episodes: Dict[int, EpisodeState] = {}
        self.debug_mode = config.debug_mode # Store debug mode flag
        self.completed_episode_metrics_buffer: List[Dict[str, Any]] = [] # Buffer for step metrics

        # Set logger level based on debug mode
        if self.debug_mode:
            logger.setLevel(logging.DEBUG)
        else:
            # Set default level to WARNING to reduce noise
            if logger.level == logging.NOTSET or logger.level > logging.WARNING:
                 logger.setLevel(logging.WARNING)

        # Define function-calling tool for actions
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "take_action",
                    "description": "Choose to 'hit' or 'stick' in Blackjack.",
                    "parameters": {
                        "action": {"type": "string", "enum": ["hit", "stick"]}
                    },
                },
            }
        ]

        # Initialize reward function
        self.reward_function = self._initialize_reward_function()

        tools_json = json.dumps(self.tools)
        # System prompt instructing the LLM on how to call the action tool
        self.system_prompt = (
            "You are an AI agent playing Blackjack who uses extreme long chains of thought to carefully consider the probabilities and optimal strategy."
            "You need to decide whether to hit or stick based on your current hand and the dealer's showing card.\n\n"
            "You should enclose your thoughts and internal monologue inside <think> </think> tags, and then "
            "provide your decision using the take_action function call. You may use extremely long chains "
            "of thought to carefully consider the probabilities and optimal strategy.\n\n"
            f"<tools>\n{tools_json}\n</tools>\n\n"
            "For your function call, return a JSON object with function name and arguments within <tool_call> </tool_call> "
            "tags with the following schema:\n"
            '<tool_call>\n{"arguments": {"action": "hit"}, "name": "take_action"}}\n</tool_call>\n\n'
            "Your answer format should be:\n"
            "<think>\n"
            "[Your detailed reasoning process about whether to hit or stick]\n"
            "</think>\n\n"
            '<tool_call>\n{"arguments": {"action": "stick"}, "name": "take_action"}}\n</tool_call>\n\n'
            "Remember to carefully consider the probabilities and optimal strategy for Blackjack."
        )

    def _initialize_reward_function(self):
        """Initialize the combined reward function for scoring."""
        if hasattr(self.config, "reward_functions") and self.config.reward_functions:
            # Configure parameters for specific reward functions
            reward_configs = []

            for reward_func in self.config.reward_functions:
                if isinstance(reward_func, str):
                    # String name case - handle known rewards with custom params
                    if reward_func == "format":
                        # Configure format reward with tool_call tags and explicit weight
                        format_config = {
                            "type": "format",
                            "weight": self.config.format_reward_weight,
                            "params": {
                                "preferred_tags": ["think", "tool_call"],
                            },
                        }
                        reward_configs.append(format_config)
                    elif reward_func == "tool_calling":
                        # Configure tool_calling reward with tools and explicit weight
                        tool_calling_config = {
                            "type": "tool_calling",
                            "weight": self.config.format_reward_weight,  # Using format_reward_weight for consistency
                            "params": {
                                "tools": self.tools,
                                "preferred_tags": ["tool_call"],
                                "check_arguments": True,
                            },
                        }
                        reward_configs.append(tool_calling_config)
                    else:
                        # Pass through other reward functions as is
                        reward_configs.append(reward_func)
                else:
                    # Dict case - pass through as is
                    reward_configs.append(reward_func)

            # Create the reward function(s)
            if len(reward_configs) == 1:
                return registry.create(reward_configs[0])
            elif len(reward_configs) > 1:
                return CombinedReward(rewards=reward_configs, normalization="none")

        return None

    def _get_or_create_episode(self, seed: int) -> EpisodeState:
        """Retrieve existing or create a new episode state keyed by seed."""
        if seed not in self.episodes:
            env = gymnasium.make(self.config.env_name)
            obs, _ = env.reset(seed=seed)
            ep = EpisodeState(seed, env)
            # Initialize history
            ep.message_history = [{"role": "system", "content": self.system_prompt}]
            formatted = self._format_observation(obs)
            print("formatted", formatted)
            ep.message_history.append({"role": "environment", "content": formatted})
            self.episodes[seed] = ep
        return self.episodes[seed]

    def _format_observation(self, obs: Tuple[int, int, int]) -> str:
        """Convert Blackjack observation to text for LLM."""
        player_sum, dealer_card, usable_ace = obs
        return (
            f"Your hand sum is {player_sum}. "
            f"Dealer showing: {dealer_card}. "
            f"You have a usable ace: {usable_ace}."
        )

    def _parse_tool_call(self, response: str) -> int:
        """Extract 'hit'/'stick' and map to action 1/0."""
        # Use the tool_call_parser helper
        tool_name, arguments, is_error = parse_tool_call(
            response, self.tools, ["tool_call"]
        )

        # Log the parsing results for debugging
        logger.warning(
            f"Parsed tool call: name={tool_name}, args={arguments}, error={is_error}"
        )

        if is_error:
            logger.warning(f"Failed to parse tool call from response: {response}")
            return -1

        action = arguments.get("action", "").lower()
        if action == "hit":
            return 1
        elif action == "stick":
            return 0
        else:
            logger.warning(f"Invalid action value: {action}")
            return -1

    def _score_response(
        self,
        env_reward: float,
        response_text: str,
        parsed_action: int, # 0=stick, 1=hit, -1=error
        episode_seed: int, # Keep signature consistent, might be useful later
        update_episode_totals: bool = False, # Keep signature consistent
    ) -> float:
        """
        Calculates a combined score for a single agent response based on environment and format rewards.

        Args:
            env_reward: The raw reward obtained from simulating this action in the environment.
            response_text: The full text response generated by the agent (including <think>).
            parsed_action: The action parsed from the response (-1 if parsing failed).
            episode_seed: The seed of the current episode.
            update_episode_totals: Flag (currently unused here) to indicate if episode totals should be updated.

        Returns:
            The combined score.
        """
        format_reward = 0.0
        # Start with the raw environment reward
        current_env_reward = env_reward

        # Penalize for parsing errors
        if parsed_action == -1:
            current_env_reward -= 0.5 # Apply a penalty for invalid action format
            logger.debug(f"[_score_response Seed: {episode_seed}] Penalty applied for invalid action format (-0.5).")

        # Calculate format reward if a reward function is configured
        if self.reward_function:
            # Prepare completions for the reward function
            # Assumes reward function expects a list of lists of messages
            format_completions = [[{"role": "assistant", "content": response_text}]]
            try:
                format_rewards = self.reward_function(format_completions)
                if format_rewards and len(format_rewards) > 0:
                    format_reward = format_rewards[0]
                    logger.debug(f"[_score_response Seed: {episode_seed}] Format reward calculated: {format_reward:.4f}")
            except Exception as e:
                logger.error(f"[_score_response Seed: {episode_seed}] Error calculating format reward: {e}")

        # Combine rewards using configured weights
        env_weight = self.config.environment_reward_weight
        format_weight = self.config.format_reward_weight
        combined_reward = (env_weight * current_env_reward) + (format_weight * format_reward)

        logger.debug(
            f"[_score_response Seed: {episode_seed}] Final Score Calculation: "
            f"Env Reward (raw): {env_reward:.4f}, "
            f"Env Reward (adjusted): {current_env_reward:.4f}, "
            f"Format Reward: {format_reward:.4f}, "
            f"Combined Reward: {combined_reward:.4f}"
        )

        # Note: This function currently doesn't update episode totals itself.
        # That responsibility lies with the caller or the main trajectory loop.

        return combined_reward

    async def _select_best_action(
        self,
        episode: EpisodeState,
        actions: List[int], # Parsed actions (0, 1, or -1)
        responses: List[str] # Full agent responses (<think>...</think><tool_call>...</tool_call>)
    ) -> Tuple[int, List[float]]:
        """
        Simulates and scores multiple candidate actions to select the best one.

        Args:
            episode: The current episode state.
            actions: A list of parsed actions corresponding to the responses.
            responses: A list of full agent responses.

        Returns:
            A tuple containing:
                - The best action selected (0, 1, or -1).
                - A list of combined scores for each action/response.
        """
        if len(actions) != len(responses):
            logger.error(f"[_select_best_action Seed: {episode.seed}] Mismatch between actions ({len(actions)}) and responses ({len(responses)}) count.")
            default_action = next((a for a in actions if a != -1), -1)
            return default_action, [-10.0] * len(actions) # Assign a very low score

        scores = [0.0] * len(actions)
        token_lengths = [0] * len(actions) # For potential tie-breaking later if needed

        try:
            # Simulate each candidate action from the current state
            for idx, (action, response_text) in enumerate(zip(actions, responses)):
                # Replay history *per candidate* for accurate simulation state
                sim_env = gymnasium.make(self.config.env_name)
                sim_obs, sim_info = sim_env.reset(seed=episode.seed)
                valid_sim = True
                for past_action in episode.actions:
                     sim_obs, _, term, trunc, sim_info = sim_env.step(past_action)
                     if term or trunc:
                          logger.warning(f"[_select_best_action Seed: {episode.seed}] Episode terminated during history replay before simulating action {idx}. Assigning low score.")
                          valid_sim = False
                          break
                if not valid_sim:
                      scores[idx] = -10.0 # Penalize if replay shows episode already ended
                      continue

                # Now simulate the actual candidate action
                if action == -1:
                    # Action parsing failed, assign base env_reward of 0 for scoring,
                    # penalty applied within _score_response.
                    env_reward_sim = 0.0
                else:
                    # Perform the step
                    _obs_sim, env_reward_sim, term_sim, trunc_sim, _info_sim = sim_env.step(action)
                    # Use the reward from the simulation step
                    logger.debug(f"[_select_best_action Seed: {episode.seed}] Sim Action {idx} (val:{action}) -> Reward:{env_reward_sim}, Term:{term_sim}")


                # Calculate the combined score using the helper method
                combined_score = self._score_response(
                    env_reward=env_reward_sim,
                    response_text=response_text,
                    parsed_action=action,
                    episode_seed=episode.seed,
                    update_episode_totals=False # Totals updated in main loop
                )
                scores[idx] = combined_score

                # Store token length (might be used by caller)
                token_lengths[idx] = len(self.tokenizer.encode(response_text))

        except Exception as e:
             logger.exception(f"[_select_best_action Seed: {episode.seed}] Error during action simulation/scoring: {e}")
             # Fallback strategy if simulation fails badly
             default_action = next((a for a in actions if a != -1), -1)
             return default_action, [-10.0] * len(actions)


        # Select the best action based on the calculated combined scores
        best_score = float('-inf')
        best_action = -1 # Default to error action if no valid scores
        best_action_idx = -1

        if scores:
            best_score = max(scores)
            potential_best_indices = [i for i, score in enumerate(scores) if score == best_score]

            # Basic tie-breaking: prefer valid actions over invalid ones (-1)
            valid_indices = [i for i in potential_best_indices if actions[i] != -1]
            if valid_indices:
                # If multiple valid actions have the same top score, break ties using token length (shortest first)
                if len(valid_indices) > 1:
                    try:
                        best_action_idx = min(valid_indices, key=lambda i: token_lengths[i])
                        logger.debug(f"[_select_best_action Seed: {episode.seed}] Tie-breaking valid actions based on token length. Chosen index: {best_action_idx}")
                    except IndexError:
                         logger.warning(f"[_select_best_action Seed: {episode.seed}] IndexError during token length tie-breaking. Defaulting to first valid index.")
                         best_action_idx = valid_indices[0] # Fallback
                else:
                    best_action_idx = valid_indices[0]
            elif potential_best_indices:
                 # If all top scores correspond to invalid actions, pick the first one
                 best_action_idx = potential_best_indices[0]
                 logger.debug(f"[_select_best_action Seed: {episode.seed}] All best scores correspond to invalid actions. Choosing first: index {best_action_idx}")
            else:
                 # Should not happen if scores list exists, but handle defensively
                 logger.error(f"[_select_best_action Seed: {episode.seed}] No potential best indices found despite scores existing. Returning default action -1.")
                 best_action_idx = -1 # Ensure index reflects the default action

            if best_action_idx != -1:
                best_action = actions[best_action_idx]
            else:
                 best_action = -1 # Ensure consistency

            logger.info(f"[_select_best_action Seed: {episode.seed}] Selected action: {best_action} (Index: {best_action_idx}, Score: {scores[best_action_idx] if best_action_idx != -1 else 'N/A'}) from scores: {['{:.4f}'.format(s) for s in scores]}")

        else:
            logger.error(f"[_select_best_action Seed: {episode.seed}] No scores calculated. Returning default action -1.")


        return best_action, scores # Return the best action and the list of all scores

    async def collect_trajectory(
        self,
        seed: int,
        interactive: bool = False,
    ) -> List[BlackjackScoredDataGroup]:
        """
        Run a single episode from the given seed, using a best-of-n approach each step.
        Refactored to use _select_best_action.
        Returns a list of BlackjackScoredDataGroup, one per time step.
        """
        ep = self._get_or_create_episode(seed)
        max_turns = self.config.max_turns if self.config.max_turns is not None else 5
        logger.info(f"[Collect Trajectory Seed: {seed}] Starting episode. Max turns: {max_turns}")

        for turn in range(max_turns):
            logger.debug(f"[Collect Trajectory Seed: {seed}] Starting Turn {turn + 1}/{max_turns}")
            # Build the prompt with history
            messages_for_prompt = ep.message_history.copy()

            # Add prefilled thinking starter if enabled
            if self.config.thinking_active:
                messages_for_prompt.append({"role": "agent", "content": "<think>\n"})
            else:
                messages_for_prompt.append({"role": "agent", "content": ""})

            prompt = self.tokenizer.apply_chat_template(messages_for_prompt, tokenize=False)
            logger.debug(f"[Collect Trajectory Seed: {seed} Turn: {turn+1}] Prompting LLM...")

            # Generate group_size candidate completions
            try:
                completions = await self.server.completion(
                    prompt=prompt,
                    n=self.config.group_size,
                    max_tokens=self.config.max_token_length,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                )
            except Exception as api_error:
                 logger.exception(f"[Collect Trajectory Seed: {seed} Turn: {turn+1}] API Error during self.server.completion: {api_error}")
                 # Cannot proceed if API call fails, return trajectory collected so far
                 return self._ensure_trajectory_token_limit(ep.trajectory)

            if not completions or not completions.choices or len(completions.choices) != self.config.group_size:
                 logger.error(f"[Collect Trajectory Seed: {seed} Turn: {turn+1}] API did not return the expected number of choices ({self.config.group_size} vs {len(completions.choices) if completions else 0}). Aborting episode.")
                 return self._ensure_trajectory_token_limit(ep.trajectory) # Return trajectory collected so far


            # Parse actions and collect responses from all completions
            alt_actions: List[int] = []
            alt_responses: List[str] = []
            for choice_idx, choice in enumerate(completions.choices):
                response_text = choice.text if hasattr(choice, "text") else getattr(choice.message, "content", "")
                full_response = ("<think>\n" + response_text) if self.config.thinking_active else response_text
                alt_responses.append(full_response)

                parsed_act = self._parse_tool_call(full_response)
                alt_actions.append(parsed_act)
                logger.debug(f"[Collect Trajectory Seed: {seed} Turn: {turn+1}] Choice {choice_idx}: Parsed Action={parsed_act}, Response Length={len(full_response)}")

            # Select best action using the helper method
            logger.debug(f"[Collect Trajectory Seed: {seed} Turn: {turn+1}] Selecting best action...")
            best_action, scores = await self._select_best_action(ep, alt_actions, alt_responses)


            # Find the index of the chosen best action to get the corresponding response
            best_action_idx = -1
            try:
                # Find the first index corresponding to the best action and its score
                # This handles potential duplicates if tie-breaking wasn't definitive
                best_score_val = max(scores) # The actual best score achieved
                possible_indices = [i for i, (act, score) in enumerate(zip(alt_actions, scores)) if act == best_action and score == best_score_val]
                if possible_indices:
                     best_action_idx = possible_indices[0] # Take the first match
                     logger.info(f"[Collect Trajectory Seed: {seed} Turn: {turn+1}] Best action selected: {best_action} (Index: {best_action_idx}), Score: {scores[best_action_idx]:.4f}")
                else:
                     # Fallback if no exact match (should be rare)
                     logger.warning(f"[Collect Trajectory Seed: {seed} Turn: {turn+1}] Could not find index for best action {best_action} with score {best_score_val}. Trying first occurrence of action.")
                     best_action_idx = alt_actions.index(best_action) # Find first occurrence of the action value
                     logger.info(f"[Collect Trajectory Seed: {seed} Turn: {turn+1}] Fallback - Best action selected: {best_action} (Index: {best_action_idx}), Score: {scores[best_action_idx]:.4f}")

                best_response = alt_responses[best_action_idx]
            except (ValueError, IndexError) as e:
                logger.error(f"[Collect Trajectory Seed: {seed} Turn: {turn+1}] Error finding index for best action {best_action}: {e}. Cannot proceed with episode.")
                # Clean up this partially failed episode before returning
                if seed in self.episodes: # Check before accessing
                    try:
                        self.episodes[seed].env.close()
                    except Exception as close_exc:
                        logger.warning(f"[Collect Trajectory Seed: {seed}] Exception closing env for aborted episode on best_action index error: {close_exc}")
                    del self.episodes[seed]
                return self._ensure_trajectory_token_limit(ep.trajectory) # Abort episode


            # Tokenize all alternatives *before* stepping the main env
            alt_tokens: List[List[int]] = []
            alt_masks: List[List[int]] = []
            alt_messages: List[List[Message]] = []
            tokenization_failed_for_step = False # Flag to track tokenization failure
            for response in alt_responses:
                step_msgs: List[Message] = [
                    {"role": m["role"], "content": m["content"]}
                    for m in ep.message_history
                ]
                step_msgs.append({"role": "agent", "content": response})

                try:
                     out = tokenize_for_trainer(self.tokenizer, step_msgs)
                     alt_tokens.append(out["tokens"])
                     alt_masks.append(out["masks"])
                     alt_messages.append(step_msgs)
                except Exception as tokenization_error:
                      logger.exception(f"[Collect Trajectory Seed: {seed} Turn: {turn+1}] Critical tokenization error for response: {response[:100]}... Error: {tokenization_error}. Aborting episode.")
                      tokenization_failed_for_step = True
                      break # Break from loop over alt_responses

            if tokenization_failed_for_step:
                logger.warning(f"[Collect Trajectory Seed: {seed}] Episode aborted at turn {turn+1} due to tokenization failure.")
                # Clean up this partially failed episode before returning
                if seed in self.episodes: # Check before accessing
                    try:
                        self.episodes[seed].env.close()
                    except Exception as e:
                        logger.warning(f"[Collect Trajectory Seed: {seed}] Exception closing env for aborted episode: {e}")
                    del self.episodes[seed]
                return self._ensure_trajectory_token_limit(ep.trajectory) # Return whatever was collected, could be empty
            
            # The following padding should only happen if tokenization did NOT fail for the step
            # Ensure lists have the expected length
            expected_len = self.config.group_size
            if len(alt_tokens) != expected_len: alt_tokens.extend([[]] * (expected_len - len(alt_tokens)))
            if len(alt_masks) != expected_len: alt_masks.extend([[]] * (expected_len - len(alt_masks)))
            if len(alt_messages) != expected_len: alt_messages.extend([[{"role":"system", "content":"Missing due to prior success but unexpected count"}]] * (expected_len - len(alt_messages)))


            # Step the main environment with the selected best action
            # Handle -1 (invalid format) by defaulting to 'stick'
            env_action = 0 if best_action == -1 else best_action
            if best_action == -1:
                logger.warning(f"[Collect Trajectory Seed: {seed} Turn: {turn+1}] Selected action was invalid format (-1). Stepping env with 'stick' (0).")

            try:
                obs, reward, term, trunc, info = ep.env.step(env_action)
                logger.info(f"[Collect Trajectory Seed: {seed} Turn: {turn+1}] Stepped main env with action {env_action}. Reward: {reward}, Term: {term}, Trunc: {trunc}")
            except Exception as env_step_error:
                logger.exception(f"[Collect Trajectory Seed: {seed} Turn: {turn+1}] Error stepping main environment with action {env_action}: {env_step_error}")
                term = True # Treat env error as terminal
                reward = -1.0 # Penalize heavily
                obs = None


            # Update Episode State (Totals and History)
            ep.actions.append(env_action) # Store the action actually taken in env
            ep.step_rewards.append(reward) # Store raw env reward

            # Re-calculate format reward for the chosen (full) response for accurate tracking
            format_reward_chosen = 0.0
            if self.reward_function:
                 format_completions = [[{"role": "assistant", "content": best_response}]]
                 try:
                     format_rewards = self.reward_function(format_completions)
                     if format_rewards and len(format_rewards) > 0:
                         format_reward_chosen = format_rewards[0]
                 except Exception as e:
                      logger.error(f"[Collect Trajectory Seed: {seed} Turn: {turn+1}] Error re-calculating format reward for chosen action: {e}")

            ep.total_env_reward += reward # Add raw environment reward
            ep.total_format_reward += format_reward_chosen
            combined_reward_step = (self.config.environment_reward_weight * reward) + (self.config.format_reward_weight * format_reward_chosen)
            ep.total_combined_reward += combined_reward_step

            ep.num_total_actions += 1
            if best_action != -1: # Count if the *parsed* action was valid format
                ep.num_correct_actions += 1

            logger.info(
                f"[Collect Trajectory Seed: {seed} Turn: {turn+1}] "
                f"Step Rewards: Env={reward:.2f}, Format={format_reward_chosen:.2f}, Combined={combined_reward_step:.2f}. "
                f"Running Totals: Env={ep.total_env_reward:.2f}, Format={ep.total_format_reward:.2f}, Combined={ep.total_combined_reward:.2f}"
            )

            # Append data for this step to the trajectory
            ep.trajectory.append(
                BlackjackScoredDataGroup(
                    overrides = [],
                    seed=seed, # Add seed here
                    tokens=alt_tokens,
                    masks=alt_masks,
                    scores=scores, # Store scores for all alternatives from _select_best_action
                    messages=alt_messages,
                    parsed_action=best_action, # Store the selected best action (could be -1)
                )
            )

            # Prepare for next turn or break if terminated
            if term or trunc:
                logger.info(f"[Collect Trajectory Seed: {seed}] Episode ended. Term={term}, Trunc={trunc}. Final Reward: {reward}")
                if obs is not None:
                     final_formatted_obs = self._format_observation(obs)
                     # For the final message, no truncation of agent's last thought is needed as it won't form a new prompt
                     ep.message_history.append({"role": "agent", "content": best_response}) # Log full final agent response
                     ep.message_history.append({"role": "environment", "content": f"Final State: {final_formatted_obs} (Reward: {reward})"})
                else:
                     ep.message_history.append({"role": "agent", "content": best_response}) # Log full final agent response
                     ep.message_history.append({"role": "environment", "content": f"Episode terminated with error. (Reward: {reward})"})
                break
            else:
                # Truncate the thinking part of the best response for message history for the *next* turn
                response_for_history = self._truncate_thinking_for_history(best_response, self.config.max_think_chars_history)
                ep.message_history.append({"role": "agent", "content": response_for_history})
                formatted_obs = self._format_observation(obs)
                ep.message_history.append({"role": "environment", "content": formatted_obs})
                logger.debug(f"[Collect Trajectory Seed: {seed} Turn: {turn+1}] New Observation: {formatted_obs}")

        logger.info(f"[Collect Trajectory Seed: {seed}] Finished episode after {len(ep.actions)} steps.")
        logger.info(f"[Collect Trajectory Seed: {seed}] Final Totals: Env Reward={ep.total_env_reward:.2f}, Format Reward={ep.total_format_reward:.2f}, Combined Reward={ep.total_combined_reward:.2f}")
        logger.info(f"[Collect Trajectory Seed: {seed}] Action Accuracy: {ep.num_correct_actions}/{max(1, ep.num_total_actions)} ({ep.num_correct_actions/max(1, ep.num_total_actions):.2%})")

        # --- Collect metrics for this completed episode ---
        final_env_reward_for_outcome = 0
        if ep.step_rewards: # Get the last environment reward
            final_env_reward_for_outcome = ep.step_rewards[-1]
        # Determine game outcome: 1 for win, -1 for loss, 0 for draw
        game_outcome = 0
        if final_env_reward_for_outcome > 0:
            game_outcome = 1
        elif final_env_reward_for_outcome < 0:
            game_outcome = -1
        
        episode_summary_metrics = {
            "seed": seed,
            "total_env_reward": ep.total_env_reward,
            "total_format_reward": ep.total_format_reward,
            "total_combined_reward": ep.total_combined_reward,
            "num_correct_actions": ep.num_correct_actions,
            "num_total_actions": ep.num_total_actions, # This is the number of turns
            "game_outcome": game_outcome, # 1 for win, -1 for loss, 0 for draw
            "num_steps_in_episode": len(ep.actions)
        }
        self.completed_episode_metrics_buffer.append(episode_summary_metrics)
        
        # --- Clean up episode state ---
        # Important for singleton: remove episode from dict to allow fresh starts if seed repeats
        if seed in self.episodes:
            try:
                self.episodes[seed].env.close() # Close the specific gym environment instance
            except Exception as e:
                logger.warning(f"[Collect Trajectory Seed: {seed}] Exception closing env for episode: {e}")
            del self.episodes[seed]
            logger.debug(f"[Collect Trajectory Seed: {seed}] Cleared episode state from self.episodes.")

        return self._ensure_trajectory_token_limit(ep.trajectory)

    async def collect_trajectories(
        self, item: Tuple[int, int]
    ) -> Tuple[List[BlackjackScoredDataGroup], List[Tuple[int, int]]]:
        seed, _ = item
        traj = await self.collect_trajectory(seed)
        # Apply token limit check to trajectory before returning
        if traj:
            traj = self._ensure_trajectory_token_limit(traj)
        return traj, []

    async def score(
        self,
        rollout_group_data: List[BlackjackScoredDataGroup],
    ) -> List[Optional[BlackjackScoredDataGroup]]:
        """
        Applies final scoring adjustments to a completed trajectory, primarily focusing
        on the final outcome (win/loss/draw).

        Args:
            rollout_group_data: The list of ScoredDataGroups representing the trajectory.
            interactive: Flag (currently unused).

        Returns:
            The list of ScoredDataGroups with potentially adjusted scores.
            Returns a list containing None if input is invalid.
        """
        if not rollout_group_data:
            logger.warning("score: Received empty rollout_group_data.")
            return [None] * len(rollout_group_data) # Maintain expected structure if needed downstream

        # --- Determine Final Outcome ---
        # In Blackjack, the outcome is determined by the reward of the final step.
        # We need to find the reward associated with the *best* action chosen in the final step.
        final_step_group = rollout_group_data[-1]
        if not final_step_group or final_step_group["scores"] is None:
            logger.warning("score: Final step group or scores are missing. Cannot determine outcome.")
            # Return original data as we can't adjust scores
            return rollout_group_data

        try:
            best_score_idx = final_step_group["scores"].index(max(final_step_group["scores"]))
            # We need the *original environment reward* associated with this best action.
            # This requires storing the raw env rewards during collection, or re-simulating.
            # For simplicity now, let's assume the combined score reflects the outcome direction.
            # A more robust approach would be to store env_rewards alongside scores.
            # Or, re-simulate the best action of the last step.

            # Let's re-simulate the final chosen action for accuracy:
            seed = final_step_group["seed"]
            best_action_final_step = final_step_group["parsed_action"] # Get the action chosen in the final step

            if best_action_final_step is None or best_action_final_step == -1:
                 logger.warning(f"score [Seed: {seed}]: Invalid best action in final step ({best_action_final_step}). Cannot determine final reward.")
                 final_env_reward = 0 # Default to draw if action invalid
            else:
                # Replay the episode to get the state before the final action
                temp_env = gymnasium.make(self.config.env_name)
                temp_env.reset(seed=seed)
                steps_to_replay = len(rollout_group_data) - 1
                for i in range(steps_to_replay):
                    step_action = rollout_group_data[i]["parsed_action"]
                    if step_action is None or step_action == -1:
                        logger.warning(f"score [Seed: {seed}]: Invalid action encountered during replay at step {i}. Stopping replay.")
                        # If replay fails, we can't accurately get the final reward
                        final_env_reward = 0 # Default to draw
                        break
                    _, _, term, trunc, _ = temp_env.step(step_action)
                    if term or trunc:
                        logger.warning(f"score [Seed: {seed}]: Episode ended prematurely during replay at step {i}. Using reward from that step if possible.")
                        # This case is tricky, the game might have ended before the recorded final step.
                        # We might need a more sophisticated replay logic if this happens often.
                        final_env_reward = 0 # Default for now
                        break
                else: # If loop completed without break
                     # Now perform the final step simulation
                     _, final_env_reward, term, trunc, _ = temp_env.step(best_action_final_step)
                     logger.info(f"score [Seed: {seed}]: Replayed final action {best_action_final_step}, got env_reward: {final_env_reward}")


            # --- Apply Final Bonus/Penalty ---
            # Apply a bonus/penalty based on the definitive win/loss/draw reward
            final_bonus = 0.0
            if final_env_reward > 0: # Win
                final_bonus = 1.0 # Additive bonus for winning
                logger.debug(f"score [Seed: {seed}]: Applying win bonus: +{final_bonus}")
            elif final_env_reward < 0: # Loss
                final_bonus = -1.0 # Additive penalty for losing
                logger.debug(f"score [Seed: {seed}]: Applying loss penalty: {final_bonus}")
            else: # Draw or indeterminate state
                logger.debug(f"score [Seed: {seed}]: No final bonus/penalty (Draw/Indeterminate). Final Env Reward: {final_env_reward}")


            # Adjust score of the best action in the final step
            adjusted_scores = final_step_group["scores"].copy()
            adjusted_scores[best_score_idx] += final_bonus
            final_step_group["scores"] = adjusted_scores # Update the scores in the group


        except (ValueError, IndexError) as e:
            logger.error(f"score [Seed: {seed}]: Error finding best score index or accessing data in final step: {e}. Scores not adjusted.")
            # Return original data if error occurs


        # --- Optional: Tie-breaking based on token length (similar to Hangman) ---
        # Iterate through all steps to apply tie-breaking
        # This might be less critical in Blackjack where turns are few
        for step_group in rollout_group_data:
             if step_group is None or step_group["scores"] is None or step_group["messages"] is None:
                 continue

             scores = step_group["scores"]
             messages = step_group["messages"]
             if len(scores) != len(messages):
                 logger.warning(f"score [Seed: {step_group.get('seed', 'N/A')}]: Mismatch between scores ({len(scores)}) and messages ({len(messages)}) lengths. Skipping tie-breaking for this step.")
                 continue

             # Calculate token lengths (only needed if tie-breaking)
             token_lengths = []
             for msg_list in messages:
                 response_text = msg_list[-1]["content"] if msg_list else ""
                 token_lengths.append(len(self.tokenizer.encode(response_text)))

             # Group indices by score
             score_groups = {}
             for idx, score in enumerate(scores):
                 if score not in score_groups:
                     score_groups[score] = []
                 score_groups[score].append(idx)

             # Apply penalty for ties based on token length
             new_scores = scores.copy()
             for score_val, indices in score_groups.items():
                 if len(indices) > 1:
                     # Sort tied indices by token length (shortest first)
                     try:
                         sorted_indices = sorted(indices, key=lambda i: token_lengths[i])
                         # Penalize longer responses among ties
                         for rank, idx in enumerate(sorted_indices[1:], 1):
                             penalty = 0.0001 * rank # Small penalty
                             new_scores[idx] -= penalty
                             logger.debug(f"score [Seed: {step_group.get('seed', 'N/A')}]: Applied tie-break penalty {-penalty:.5f} to index {idx} (rank {rank}, score {score_val}).")
                     except IndexError:
                          logger.warning(f"score [Seed: {step_group.get('seed', 'N/A')}]: IndexError during tie-breaking. Token lengths ({len(token_lengths)}) might not match indices ({indices}). Skipping tie-breaking for score {score_val}.")


             step_group["scores"] = new_scores # Update scores in the group


        return rollout_group_data

    async def setup(self):
        pass

    async def get_next_item(self) -> Tuple[int, int]:
        import random

        return (random.randint(0, 1000000), 0)

    async def rollout_and_score_eval(self, seed: int) -> Dict[str, Any]:
        """
        Run a single episode for evaluation and return detailed metrics.
        Does not use the best-of-n sampling, but a single completion per step.
        Cleans up the episode state after completion.
        """
        ep = self._get_or_create_episode(seed) # Creates a fresh episode state
        max_turns = self.config.max_turns if self.config.max_turns is not None else 5
        logger.info(f"[Eval Rollout Seed: {seed}] Starting episode. Max turns: {max_turns}")

        # Metrics to collect for this episode
        episode_metrics = {
            "seed": seed,
            "total_env_reward": 0.0,
            "total_format_reward": 0.0,
            "total_combined_reward": 0.0,
            "num_turns": 0,
            "num_correct_actions": 0, # Correctly parsed actions (hit/stick)
            "num_invalid_actions": 0, # Failed to parse action
            "actions_chosen": [], # List of actions (0 for stick, 1 for hit, -1 for error)
            "game_outcome": 0, # -1 for loss, 0 for draw, 1 for win
        }

        for turn in range(max_turns):
            episode_metrics["num_turns"] = turn + 1
            messages_for_prompt = ep.message_history.copy()

            if self.config.thinking_active:
                messages_for_prompt.append({"role": "agent", "content": "<think>\n"})
            else:
                messages_for_prompt.append({"role": "agent", "content": ""})

            prompt = self.tokenizer.apply_chat_template(messages_for_prompt, tokenize=False)

            try:
                completions = await self.server.completion(
                    prompt=prompt,
                    n=1, # Single completion for eval
                    max_tokens=self.config.max_token_length,
                    temperature=self.config.temperature, # Use main config temperature
                    top_p=self.config.top_p,           # Use main config top_p
                    split="eval", # Indicate this is an evaluation call
                )
            except Exception as api_error:
                logger.exception(f"[Eval Rollout Seed: {seed} Turn: {turn+1}] API Error: {api_error}")
                break # End episode on API error

            if not completions or not completions.choices:
                logger.error(f"[Eval Rollout Seed: {seed} Turn: {turn+1}] API did not return any choices. Aborting episode.")
                break

            response_text = completions.choices[0].text if hasattr(completions.choices[0], "text") else getattr(completions.choices[0].message, "content", "")
            full_response = ("<think>\n" + response_text) if self.config.thinking_active else response_text

            parsed_action = self._parse_tool_call(full_response)
            episode_metrics["actions_chosen"].append(parsed_action)

            if parsed_action == -1:
                episode_metrics["num_invalid_actions"] += 1
                env_action = 0 # Default to stick on parse error for env stepping
                logger.warning(f"[Eval Rollout Seed: {seed} Turn: {turn+1}] Invalid action parsed. Defaulting to 'stick'.")
            else:
                episode_metrics["num_correct_actions"] += 1
                env_action = parsed_action

            try:
                obs, reward, term, trunc, info = ep.env.step(env_action)
            except Exception as env_step_error:
                logger.exception(f"[Eval Rollout Seed: {seed} Turn: {turn+1}] Error stepping env: {env_step_error}")
                term = True # Treat as terminal
                reward = -1.0 # Penalize heavily
                obs = None

            # Calculate format reward for this single response
            format_reward_step = 0.0
            if self.reward_function:
                format_completions = [[{"role": "assistant", "content": full_response}]]
                try:
                    format_rewards = self.reward_function(format_completions)
                    if format_rewards and len(format_rewards) > 0:
                        format_reward_step = format_rewards[0]
                except Exception as e:
                    logger.error(f"[Eval Rollout Seed: {seed} Turn: {turn+1}] Error calculating format reward: {e}")

            # Update running totals for the episode
            episode_metrics["total_env_reward"] += reward
            episode_metrics["total_format_reward"] += format_reward_step
            combined_reward_step = (self.config.environment_reward_weight * reward) + \
                                   (self.config.format_reward_weight * format_reward_step)
            episode_metrics["total_combined_reward"] += combined_reward_step

            if term or trunc:
                episode_metrics["game_outcome"] = int(reward) # Store final reward as outcome
                logger.info(f"[Eval Rollout Seed: {seed}] Episode ended. Outcome Reward: {reward}")
                if obs is not None:
                    final_formatted_obs = self._format_observation(obs)
                    ep.message_history.append({"role": "environment", "content": f"Final State: {final_formatted_obs} (Reward: {reward})"})
                else:
                    ep.message_history.append({"role": "environment", "content": f"Episode terminated with error. (Reward: {reward})"})
                break # End of episode
            else:
                ep.message_history.append({"role": "agent", "content": full_response})
                formatted_obs = self._format_observation(obs)
                ep.message_history.append({"role": "environment", "content": formatted_obs})

        logger.info(f"[Eval Rollout Seed: {seed}] Finished episode. Metrics: {episode_metrics}")

        # Clean up episode state for this seed to ensure next eval is fresh
        if seed in self.episodes:
            try:
                self.episodes[seed].env.close()
            except Exception as e:
                logger.warning(f"[Eval Rollout Seed: {seed}] Exception closing env for episode: {e}")
            del self.episodes[seed]

        return episode_metrics

    async def evaluate(self, *args, **kwargs):
        """Run evaluation episodes and aggregate metrics for logging."""
        if not self.config.use_wandb: # Skip if wandb is not enabled
            logger.info("Skipping evaluation as wandb is not enabled.")
            return

        num_eval_episodes = self.config.eval_episodes
        logger.info(f"Starting evaluation for {num_eval_episodes} episodes.")

        eval_tasks = []
        
        for i in range(num_eval_episodes):
            eval_seed = random.randint(1000001, 2000000) # different seed range for eval
            eval_tasks.append(self.rollout_and_score_eval(eval_seed))

        all_episode_metrics = await tqdm_asyncio.gather(*eval_tasks)

        # --- Aggregate Metrics --- 
        if not all_episode_metrics:
            logger.warning("No metrics collected from evaluation episodes.")
            return

        valid_metrics = [m for m in all_episode_metrics if m is not None]
        if not valid_metrics:
            logger.warning("All evaluation episodes resulted in None metrics.")
            return

        num_completed_episodes = len(valid_metrics)

        # Calculate averages
        avg_total_env_reward = sum(m["total_env_reward"] for m in valid_metrics) / num_completed_episodes
        avg_total_format_reward = sum(m["total_format_reward"] for m in valid_metrics) / num_completed_episodes
        avg_total_combined_reward = sum(m["total_combined_reward"] for m in valid_metrics) / num_completed_episodes
        avg_num_turns = sum(m["num_turns"] for m in valid_metrics) / num_completed_episodes

        # Calculate rates
        total_correct_actions = sum(m["num_correct_actions"] for m in valid_metrics)
        total_invalid_actions = sum(m["num_invalid_actions"] for m in valid_metrics)
        total_actions_taken = total_correct_actions + total_invalid_actions
        action_accuracy = total_correct_actions / total_actions_taken if total_actions_taken > 0 else 0
        invalid_action_rate = total_invalid_actions / total_actions_taken if total_actions_taken > 0 else 0

        # Game outcomes
        wins = sum(1 for m in valid_metrics if m["game_outcome"] == 1)
        losses = sum(1 for m in valid_metrics if m["game_outcome"] == -1)
        draws = sum(1 for m in valid_metrics if m["game_outcome"] == 0)

        win_rate = wins / num_completed_episodes if num_completed_episodes > 0 else 0
        loss_rate = losses / num_completed_episodes if num_completed_episodes > 0 else 0
        draw_rate = draws / num_completed_episodes if num_completed_episodes > 0 else 0

        # Action distribution (hit vs stick vs error)
        all_chosen_actions = [action for m in valid_metrics for action in m["actions_chosen"]]
        count_hit = sum(1 for act in all_chosen_actions if act == 1)
        count_stick = sum(1 for act in all_chosen_actions if act == 0)
        count_error_actions = sum(1 for act in all_chosen_actions if act == -1)
        total_parsed_actions_in_eval = len(all_chosen_actions)

        self.eval_metrics = [
            ("eval/avg_total_env_reward", avg_total_env_reward),
            ("eval/avg_total_format_reward", avg_total_format_reward),
            ("eval/avg_total_combined_reward", avg_total_combined_reward),
            ("eval/avg_num_turns", avg_num_turns),
            ("eval/action_accuracy", action_accuracy), # Correctly parsed hit/stick vs total attempts
            ("eval/invalid_action_rate", invalid_action_rate), # Rate of unparsable actions
            ("eval/win_rate", win_rate),
            ("eval/loss_rate", loss_rate),
            ("eval/draw_rate", draw_rate),
            ("eval/num_wins", wins),
            ("eval/num_losses", losses),
            ("eval/num_draws", draws),
            ("eval/num_completed_episodes", num_completed_episodes),
            ("eval/hit_chosen_rate", count_hit / total_parsed_actions_in_eval if total_parsed_actions_in_eval > 0 else 0),
            ("eval/stick_chosen_rate", count_stick / total_parsed_actions_in_eval if total_parsed_actions_in_eval > 0 else 0),
            ("eval/error_action_chosen_rate", count_error_actions / total_parsed_actions_in_eval if total_parsed_actions_in_eval > 0 else 0),
        ]

        logger.info(f"Evaluation completed. Aggregated metrics: {self.eval_metrics}")

    async def wandb_log(self, wandb_metrics: Optional[Dict[str, Any]] = None):
        """
        Log aggregated metrics from completed training episodes and call super().wandb_log.
        """
        if wandb_metrics is None:
            wandb_metrics = {}

        if self.completed_episode_metrics_buffer:
            num_episodes_in_buffer = len(self.completed_episode_metrics_buffer)
            
            # Aggregate metrics from the buffer
            avg_ep_env_reward = sum(m["total_env_reward"] for m in self.completed_episode_metrics_buffer) / num_episodes_in_buffer
            avg_ep_format_reward = sum(m["total_format_reward"] for m in self.completed_episode_metrics_buffer) / num_episodes_in_buffer
            avg_ep_combined_reward = sum(m["total_combined_reward"] for m in self.completed_episode_metrics_buffer) / num_episodes_in_buffer
            
            total_ep_correct_actions = sum(m["num_correct_actions"] for m in self.completed_episode_metrics_buffer)
            total_ep_actions = sum(m["num_total_actions"] for m in self.completed_episode_metrics_buffer)
            avg_ep_action_accuracy = total_ep_correct_actions / total_ep_actions if total_ep_actions > 0 else 0
            
            avg_ep_num_steps = sum(m["num_steps_in_episode"] for m in self.completed_episode_metrics_buffer) / num_episodes_in_buffer

            ep_wins = sum(1 for m in self.completed_episode_metrics_buffer if m["game_outcome"] == 1)
            ep_losses = sum(1 for m in self.completed_episode_metrics_buffer if m["game_outcome"] == -1)
            ep_draws = sum(1 for m in self.completed_episode_metrics_buffer if m["game_outcome"] == 0)

            ep_win_rate = ep_wins / num_episodes_in_buffer if num_episodes_in_buffer > 0 else 0
            ep_loss_rate = ep_losses / num_episodes_in_buffer if num_episodes_in_buffer > 0 else 0
            ep_draw_rate = ep_draws / num_episodes_in_buffer if num_episodes_in_buffer > 0 else 0

            # Add to wandb_metrics dictionary with a specific prefix for training rollouts
            wandb_metrics[f"{self.wandb_prepend or 'blackjack'}_train/avg_episode_env_reward"] = avg_ep_env_reward
            wandb_metrics[f"{self.wandb_prepend or 'blackjack'}_train/avg_episode_format_reward"] = avg_ep_format_reward
            wandb_metrics[f"{self.wandb_prepend or 'blackjack'}_train/avg_episode_combined_reward"] = avg_ep_combined_reward
            wandb_metrics[f"{self.wandb_prepend or 'blackjack'}_train/avg_episode_action_accuracy"] = avg_ep_action_accuracy
            wandb_metrics[f"{self.wandb_prepend or 'blackjack'}_train/avg_episode_num_steps"] = avg_ep_num_steps
            wandb_metrics[f"{self.wandb_prepend or 'blackjack'}_train/episode_win_rate"] = ep_win_rate
            wandb_metrics[f"{self.wandb_prepend or 'blackjack'}_train/episode_loss_rate"] = ep_loss_rate
            wandb_metrics[f"{self.wandb_prepend or 'blackjack'}_train/episode_draw_rate"] = ep_draw_rate
            wandb_metrics[f"{self.wandb_prepend or 'blackjack'}_train/num_episodes_in_log_period"] = num_episodes_in_buffer
            
            logger.info(f"Logging metrics for {num_episodes_in_buffer} completed training episodes.")
            self.completed_episode_metrics_buffer = []
        await super().wandb_log(wandb_metrics)

    @classmethod
    def config_init(
        cls, config_name_or_path: Optional[str] = None
    ) -> Tuple[BlackjackEnvConfig, List[OpenaiConfig]]:
        """Load settings from the local configs directory or an absolute path."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        default_config_filename = "blackjack_default.yaml"

        if config_name_or_path is None:
            cfg_path = os.path.join(current_dir, "configs", default_config_filename)
            logger.info(f"No config specified, using default: {cfg_path}")
        elif os.path.isabs(config_name_or_path):
            cfg_path = config_name_or_path
            logger.info(f"Absolute config path provided: {cfg_path}")
            if not os.path.splitext(cfg_path)[1]: # Check for file extension
                 logger.warning(f"Absolute config path {cfg_path} seems to be missing a file extension.")
        else:
            config_filename = config_name_or_path
            if not config_name_or_path.endswith(".yaml"):
                config_filename += ".yaml"
            cfg_path = os.path.join(current_dir, "configs", config_filename)
            logger.info(f"Relative config name '{config_name_or_path}' provided, resolving to: {cfg_path}")

        logger.debug(f"Final config path to check for existence: {cfg_path}")

        raw_yaml_data = {}
        try:
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    raw_yaml_data = yaml.safe_load(f) or {} # Ensure dict even if file is empty/null
                logger.info(f"Loaded config from {cfg_path}")
            else:
                logger.warning(
                    f"Config file not found at {cfg_path}, using default BlackjackEnvConfig settings and default server config."
                )
                # raw_yaml_data remains empty, leading to defaults.

            # Prepare data for BlackjackEnvConfig
            env_conf_data = raw_yaml_data.copy()
            server_configs_list_from_yaml = env_conf_data.pop("server_configs", []) # Pop for separate processing

            # Apply 'blackjack' section overrides if present
            if "blackjack" in env_conf_data:
                blackjack_overrides = env_conf_data.pop("blackjack")
                if isinstance(blackjack_overrides, dict):
                    env_conf_data.update(blackjack_overrides) # Overrides take precedence
                else:
                    logger.warning(f"'blackjack' section in config YAML is not a dictionary (type: {type(blackjack_overrides)}), ignoring.")
            
            # Pydantic will use BlackjackEnvConfig's defined defaults for any missing keys
            env_conf = BlackjackEnvConfig(**env_conf_data)
            logger.debug(f"Initialized BlackjackEnvConfig: {env_conf}")

            # --- Process server_configs ---
            server_confs = []
            # Check if server_configs_list_from_yaml is not None and is a list,
            # it defaults to [] if key missing, but could be None if YAML has `server_configs: null`
            if isinstance(server_configs_list_from_yaml, list):
                for sc_data in server_configs_list_from_yaml:
                    if not isinstance(sc_data, dict):
                        logger.warning(f"Skipping non-dictionary item in server_configs: {sc_data}")
                        continue
                    
                    current_params = sc_data.copy()

                    # API Key: YAML value (even empty) -> Environment Variable -> "x"
                    resolved_api_key = sc_data.get("api_key")
                    if resolved_api_key is None or resolved_api_key == "":
                        resolved_api_key = os.getenv("OPENAI_API_KEY")
                    if resolved_api_key is None or resolved_api_key == "":
                        resolved_api_key = "x"
                    current_params["api_key"] = resolved_api_key

                    # Apply Blackjack-specific defaults for model and base_url if not in sc_data
                    # These override OpenaiConfig's generic default_factories if needed for Blackjack.
                    if "model_name" not in current_params:
                        current_params["model_name"] = os.getenv("OPENAI_MODEL", "NousResearch/DeepHermes-3-Llama-3-8B-Preview")
                    if "base_url" not in current_params:
                        current_params["base_url"] = os.getenv("OPENAI_API_BASE", "http://localhost:9004/v1")
                    # num_requests_for_eval will use OpenaiConfig's Pydantic default (256) if not in current_params.
                    
                    server_confs.append(OpenaiConfig(**current_params))
            elif "server_configs" not in raw_yaml_data: # server_configs key was completely MISSING
                logger.warning("No 'server_configs' key found in YAML, creating default Blackjack server config.")
                server_confs = [
                    OpenaiConfig(
                        model_name=os.getenv("OPENAI_MODEL", "NousResearch/DeepHermes-3-Llama-3-8B-Preview"),
                        base_url=os.getenv("OPENAI_API_BASE", "http://localhost:9004/v1"),
                        api_key=os.getenv("OPENAI_API_KEY", "x") 
                        # num_requests_for_eval=256 comes from OpenaiConfig Pydantic default
                    )
                ]
            # If server_configs was present but null or not a list, server_confs might be empty or minimally populated.

            return env_conf, server_confs

        except Exception as e:
            cfg_path_for_log = cfg_path if 'cfg_path' in locals() else 'unknown path'
            logger.exception(f"Error loading or parsing config from {cfg_path_for_log}: {e}")
            logger.warning("Falling back to default Blackjack configurations due to error.")
            return BlackjackEnvConfig(), [
                OpenaiConfig(
                    model_name=os.getenv("OPENAI_MODEL", "NousResearch/DeepHermes-3-Llama-3-8B-Preview"),
                    base_url=os.getenv("OPENAI_API_BASE", "http://localhost:9004/v1"),
                    api_key=os.getenv("OPENAI_API_KEY", "x"),
                    num_requests_for_eval=1 # Minimal eval requests on error
                )
            ]

    @classmethod
    def cli(cls):
        super(BlackjackEnv, cls).cli()

    def _truncate_thinking_for_history(self, response_text: str, max_chars_fallback: int) -> str:
        """Helper to truncate the <think> block of a response for message history."""
        try:
            think_start_tag = "<think>"
            think_end_tag = "</think>"
            
            think_start_idx = response_text.find(think_start_tag)
            think_end_idx = response_text.find(think_end_tag)

            if think_start_idx != -1 and think_end_idx != -1 and think_start_idx < think_end_idx:
                # Part 1: Everything up to and including the think_start_tag
                part_before_content = response_text[:think_start_idx + len(think_start_tag)]
                # Part 2: The actual content inside <think>...</think>, stripped of leading/trailing whitespace
                original_think_content = response_text[think_start_idx + len(think_start_tag) : think_end_idx].strip()
                # Part 3: Everything from the think_end_tag to the end of the string
                part_after_content = response_text[think_end_idx:]
                
                truncated_think_content = original_think_content
                is_truncated = False

                if not original_think_content: # Handles empty or whitespace-only think block
                    return response_text # Return original as there's nothing to truncate

                # Try to get the last non-empty paragraph
                paragraphs = [p.strip() for p in original_think_content.split('\n\n') if p.strip()]
                if len(paragraphs) > 0:
                    last_paragraph = paragraphs[-1]
                    # If taking the last paragraph makes it shorter than the original, use it.
                    # This handles cases where the last paragraph is a good summary.
                    if len(last_paragraph) < len(original_think_content):
                        truncated_think_content = last_paragraph
                        is_truncated = True # Considered truncated if we picked a specific, shorter paragraph
                    # If it was a single paragraph, or last paragraph isn't shorter, check against max_chars_fallback
                    elif len(original_think_content) > max_chars_fallback:
                        truncated_think_content = original_think_content[-max_chars_fallback:]
                        is_truncated = True
                elif len(original_think_content) > max_chars_fallback: # No paragraphs found, check length directly
                    truncated_think_content = original_think_content[-max_chars_fallback:]
                    is_truncated = True

                if is_truncated and truncated_think_content: # Prepend "..." if actual truncation happened and content exists
                    # Avoid double "..." if content already started with it (e.g. from previous truncation)
                    if not truncated_think_content.startswith("... "):
                         truncated_think_content = "... " + truncated_think_content.lstrip()
                
                # If truncated_think_content becomes empty or just "...", treat as no meaningful think content left.
                if not truncated_think_content.strip() or truncated_think_content.strip() == "...":
                    final_content_for_block = ""
                else:
                    final_content_for_block = f"\n{truncated_think_content.strip()}\n"
                
                # Use rstrip on part_before_content and lstrip on part_after_content 
                # to allow final_content_for_block to control newlines around the content.
                return f"{part_before_content.rstrip()}{final_content_for_block}{part_after_content.lstrip()}"
            
            return response_text # No valid <think> block found
        except Exception as e:
            logger.error(f"Error in _truncate_thinking_for_history for text '{response_text[:200]}...': {e}", exc_info=True)
            return response_text # Fallback to original on any error

    def _ensure_trajectory_token_limit(self, trajectory: List[BlackjackScoredDataGroup]) -> List[BlackjackScoredDataGroup]:
        """
        Ensure the message histories in a trajectory don't exceed max_trajectory_tokens.
        If they do, trim older messages while maintaining alignment across all alternatives.
        
        Args:
            trajectory: List of BlackjackScoredDataGroup from an episode
            
        Returns:
            The trajectory with potentially trimmed message histories
        """
        if not trajectory:
            return trajectory
        
        # First check if we need trimming at all by examining all alternatives in the last step
        last_step = trajectory[-1]
        if not last_step.get("messages") or not last_step["messages"]:
            return trajectory  # Nothing to trim
            
        # Check the maximum token count across all alternatives in the last step
        max_token_count = 0
        for messages in last_step["messages"]:
            token_count = sum(len(self.tokenizer.encode(msg["content"])) for msg in messages)
            max_token_count = max(max_token_count, token_count)
        
        # If even the largest alternative is under the limit, we're good
        if max_token_count <= self.config.max_trajectory_tokens:
            return trajectory  # No trimming needed
            
        logger.info(f"Message history exceeds token limit (max: {max_token_count} > {self.config.max_trajectory_tokens}). Trimming older messages.")
        
        # We need to trim - process each step in the trajectory
        for step_idx, step in enumerate(trajectory):
            if not step.get("messages") or not step["messages"]:
                continue
                
            # Get message counts to ensure we have the same structure for all alternatives
            alt_count = len(step["messages"])
            if alt_count == 0:
                continue
                
            # Process each alternative's message history
            for alt_idx in range(alt_count):
                messages = step["messages"][alt_idx]
                
                # Continue trimming until we're under the limit
                # We always keep the system message (index 0) and at least the two most recent messages
                while len(messages) > 3:  # At minimum keep system + last two messages
                    # Calculate current token count
                    token_count = sum(len(self.tokenizer.encode(msg["content"])) for msg in messages)
                    
                    if token_count <= self.config.max_trajectory_tokens:
                        break  # We're under the limit
                        
                    # Remove the second message (after system message)
                    # This preserves the system prompt and most recent context
                    try:
                        messages.pop(1)
                    except IndexError:
                        # If we somehow have an unexpected message structure, log and break
                        logger.warning(f"Unexpected message structure while trimming step {step_idx}, alt {alt_idx}. Remaining messages: {len(messages)}")
                        break
                    
                    # Update the messages for this alternative
                    step["messages"][alt_idx] = messages
                    
                    logger.debug(f"Trimmed messages for step {step_idx}, alternative {alt_idx}. New count: {len(messages)}")
        
        # After trimming, we could potentially re-tokenize the message histories to ensure
        # the tokens and masks are aligned with the trimmed messages, but that's expensive.
        # Since we're only concerned with keeping the token count under the limit for the trainer,
        # we'll rely on the trimming logic above.
        
        return trajectory

if __name__ == "__main__":
    # This allows running the environment directly from the command line
    BlackjackEnv.cli()
