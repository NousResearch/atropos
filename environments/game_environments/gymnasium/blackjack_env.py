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

import gymnasium

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

    # Reward function configuration
    reward_functions: List[Union[str, Dict[str, Any]]] = []
    format_reward_weight: float = 0.2
    environment_reward_weight: float = 0.8

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
        logger.debug(
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
                 return ep.trajectory

            if not completions or not completions.choices or len(completions.choices) != self.config.group_size:
                 logger.error(f"[Collect Trajectory Seed: {seed} Turn: {turn+1}] API did not return the expected number of choices ({self.config.group_size} vs {len(completions.choices) if completions else 0}). Aborting episode.")
                 return ep.trajectory # Return trajectory collected so far


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
                return ep.trajectory # Abort episode


            # Tokenize all alternatives *before* stepping the main env
            alt_tokens: List[List[int]] = []
            alt_masks: List[List[int]] = []
            alt_messages: List[List[Message]] = []
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
                      logger.exception(f"[Collect Trajectory Seed: {seed} Turn: {turn+1}] Error tokenizing response: {response[:100]}... Error: {tokenization_error}")
                      alt_tokens.append([])
                      alt_masks.append([])
                      alt_messages.append([{"role":"system", "content":"Tokenization Failed"}])

            # Ensure lists have the expected length even if tokenization failed
            expected_len = self.config.group_size
            if len(alt_tokens) != expected_len: alt_tokens.extend([[]] * (expected_len - len(alt_tokens)))
            if len(alt_masks) != expected_len: alt_masks.extend([[]] * (expected_len - len(alt_masks)))
            if len(alt_messages) != expected_len: alt_messages.extend([[{"role":"system", "content":"Tokenization Failed"}]] * (expected_len - len(alt_messages)))


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

            # Re-calculate format reward for the chosen response for accurate tracking
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
                     ep.message_history.append({"role": "environment", "content": f"Final State: {final_formatted_obs} (Reward: {reward})"})
                else:
                     ep.message_history.append({"role": "environment", "content": f"Episode terminated with error. (Reward: {reward})"})
                break
            else:
                # Add the chosen agent response and the new environment observation to history
                ep.message_history.append({"role": "agent", "content": best_response})
                formatted_obs = self._format_observation(obs)
                ep.message_history.append({"role": "environment", "content": formatted_obs})
                logger.debug(f"[Collect Trajectory Seed: {seed} Turn: {turn+1}] New Observation: {formatted_obs}")

        logger.info(f"[Collect Trajectory Seed: {seed}] Finished episode after {len(ep.actions)} steps.")
        logger.info(f"[Collect Trajectory Seed: {seed}] Final Totals: Env Reward={ep.total_env_reward:.2f}, Format Reward={ep.total_format_reward:.2f}, Combined Reward={ep.total_combined_reward:.2f}")
        logger.info(f"[Collect Trajectory Seed: {seed}] Action Accuracy: {ep.num_correct_actions}/{ep.num_total_actions} ({ep.num_correct_actions/max(1, ep.num_total_actions):.2%})")

        return ep.trajectory

    async def collect_trajectories(
        self, item: Tuple[int, int]
    ) -> Tuple[List[BlackjackScoredDataGroup], List[Tuple[int, int]]]:
        seed, _ = item
        traj = await self.collect_trajectory(seed)
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

    async def evaluate(self, *args, **kwargs):
        pass

    @classmethod
    def config_init(
        cls, config_name_or_path: Optional[str] = None
    ) -> Tuple[BlackjackEnvConfig, List[OpenaiConfig]]:
        """Load settings from the local configs directory or an absolute path."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        default_config_filename = "blackjack_default.yaml"

        if config_name_or_path is None:
            # No input, use default relative to this env
            cfg_path = os.path.join(current_dir, "configs", default_config_filename)
            logger.info(f"No config specified, using default: {cfg_path}")
        elif os.path.isabs(config_name_or_path):
            # Absolute path provided
            cfg_path = config_name_or_path
            logger.info(f"Absolute config path provided: {cfg_path}")
            # Optional: Check if it ends with .yaml, though absolute path should be precise
            if not os.path.splitext(cfg_path)[1]:
                 logger.warning(f"Absolute config path {cfg_path} seems to be missing a file extension.")
        else:
            # Relative name/path provided, assume relative to this env's config dir
            # Ensure it ends with .yaml
            if not config_name_or_path.endswith(".yaml"):
                config_filename = config_name_or_path + ".yaml"
            else:
                config_filename = config_name_or_path
            cfg_path = os.path.join(current_dir, "configs", config_filename)
            logger.info(f"Relative config name '{config_name_or_path}' provided, resolving to: {cfg_path}")

        logger.debug(f"Final config path to check for existence: {cfg_path}")

        raw = {}
        try:
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    raw = yaml.safe_load(f) or {}
                logger.info(f"Loaded config from {cfg_path}")
            else:
                logger.warning(
                    f"Config file not found at {cfg_path}, using default BlackjackEnvConfig settings."
                )
                # raw remains empty, defaults from BlackjackEnvConfig will be used

            # Separate base keys and blackjack-specific keys
            blackjack_specific_raw = raw.pop("blackjack", {}) # Extract and remove blackjack dict
            # Base keys are what remains in raw (excluding server_configs handled separately)
            base_raw = {k: v for k, v in raw.items() if k != "server_configs"}

            # Combine base and specific configs, specific ones overwrite base if names clash
            combined_config_data = base_raw.copy()
            combined_config_data.update(blackjack_specific_raw)

            # Create BlackjackEnvConfig instance using combined data
            # Ensure boolean flags like debug_mode are present if not in combined_data
            if 'debug_mode' not in combined_config_data:
                combined_config_data['debug_mode'] = False # Default if missing

            env_conf = BlackjackEnvConfig(**combined_config_data)
            logger.debug(f"Initialized BlackjackEnvConfig: {env_conf}")

            # Create OpenaiConfig instances from the original raw data
            server_confs = []
            # Use raw.get("server_configs", []) which contains the list of dicts from YAML
            for sc_data in raw.get("server_configs", []):
                # Get values directly from the loaded YAML dict (sc_data)
                # Provide fallbacks if keys are missing in the YAML dict itself
                model_name = sc_data.get("model_name", os.getenv("OPENAI_MODEL", "NousResearch/DeepHermes-3-Llama-3-8B-Preview"))
                base_url = sc_data.get("base_url", os.getenv("OPENAI_API_BASE", "http://localhost:9004/v1"))
                num_requests = sc_data.get("num_requests_for_eval", 256)

                # Special handling for api_key: YAML -> Env Var -> "x"
                api_key = sc_data.get("api_key") # Get from YAML first
                if not api_key: # If missing or empty in YAML
                    api_key = os.getenv("OPENAI_API_KEY") # Try environment variable
                if not api_key: # If still missing or empty
                    api_key = "x" # Default to "x" (for local server assumption)
                    logger.warning("API key not found in config or OPENAI_API_KEY env var. Defaulting to 'x'.")
                else:
                     # Mask the key partially for logging if it's not 'x'
                     masked_key = api_key[:4] + "****" + api_key[-4:] if api_key != "x" and len(api_key) > 8 else api_key
                     logger.debug(f"Using API key: {masked_key}")


                openai_config_args = {
                    "model_name": model_name,
                    "api_key": api_key,
                    "num_requests_for_eval": num_requests,
                    "base_url": base_url,
                }
                logger.warning(f"Creating OpenaiConfig with args: model='{model_name}', base_url='{base_url}', key_present={api_key != 'x'}, requests={num_requests}")
                server_confs.append(OpenaiConfig(**openai_config_args))

            # Provide a default server config ONLY if server_configs was completely missing from YAML
            if "server_configs" not in raw:
                logger.warning("No 'server_configs' section found in YAML, creating default server config.")
                # Default API key logic
                default_api_key = os.getenv("OPENAI_API_KEY")
                if not default_api_key:
                    default_api_key = "x"
                    logger.warning("Defaulting API key to 'x' for default server config.")

                server_confs = [
                    OpenaiConfig(
                        model_name=os.getenv("OPENAI_MODEL", "NousResearch/DeepHermes-3-Llama-3-8B-Preview"),
                        base_url=os.getenv("OPENAI_API_BASE", "http://localhost:9004/v1"),
                        api_key=default_api_key,
                        num_requests_for_eval=256,
                    )
                ]
                logger.warning(f"Created default OpenaiConfig: model='{server_confs[0].model_name}', base_url='{server_confs[0].base_url}', key_present={server_confs[0].api_key != 'x'}")


            return env_conf, server_confs

        except Exception as e:
            logger.exception(f"Error loading config from {cfg_path}: {e}")
            logger.warning("Falling back to default configurations due to error.")
            # Fall back to default configs on error
            return BlackjackEnvConfig(), [
                OpenaiConfig(
                    model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    base_url=os.getenv("OPENAI_API_BASE"),
                    api_key=os.getenv("OPENAI_API_KEY", ""),
                    num_requests_for_eval=1,
                )
            ]

    @classmethod
    def cli(cls):
        super(BlackjackEnv, cls).cli()

if __name__ == "__main__":
    # This allows running the environment directly from the command line
    BlackjackEnv.cli()
