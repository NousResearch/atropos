#!/usr/bin/env python3
"""
BlackjackEnv: Trainer environment for Gymnasium Blackjack

This wraps Gymnasium's Blackjack-v1 environment to train an LLM via a best-of-n pattern
using function-call style actions. Extends BaseEnv.

Sort of inspired by VinePPO, but uses a recursive exact value calculation
instead of Monte Carlo sampling (because the action space is so small and
the environment is deterministic, plus short episode lengths).
"""

import copy
import json
import logging
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import gymnasium
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
    OpenaiConfig,
    ScoredDataGroup,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer_multistep
from atroposlib.utils.tool_call_parser import parse_tool_call

logger = logging.getLogger(__name__)


class BlackjackEnvConfig(BaseEnvConfig):
    env_name: str = "Blackjack-v1"
    temperature: float = 0.7
    top_p: float = 0.9
    max_turns: Optional[int] = 5
    wandb_name: str = "blackjack"
    thinking_active: bool = True
    eval_episodes: int = 100
    max_think_chars_history: int = 3000
    max_trajectory_tokens: int = 24576
    debug_mode: bool = False
    group_size: int = 16
    tiebreak_token_factor: float = 0.01


class BlackjackScoredDataGroup(ScoredDataGroup):
    seed: int
    tokens: Optional[List[List[int]]] = None
    masks: Optional[List[List[int]]] = None
    scores: Optional[List[float]] = None
    messages: Optional[List[List[Dict]]] = None
    parsed_actions: Optional[List[int]] = None


class EpisodeState:
    def __init__(self, seed: int, env: gymnasium.Env):
        self.seed = seed
        self.env = env
        self.message_history: List[Dict] = []
        self.actions: List[int] = []
        self.step_rewards: List[float] = []
        self.total_reward: float = 0.0
        self.num_steps: int = 0
        self.num_correct_actions: int = 0
        self.num_total_actions: int = 0


class BlackjackEnv(BaseEnv):
    def __init__(
        self,
        config: BlackjackEnvConfig,
        server_configs: List[OpenaiConfig],
        slurm: bool = True,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.episodes: Dict[int, EpisodeState] = {}
        self.debug_mode = config.debug_mode
        self.completed_episode_metrics_buffer: List[Dict[str, float]] = []
        if self.debug_mode:
            logger.setLevel(logging.DEBUG)
        else:
            if logger.level == logging.NOTSET or logger.level > logging.WARNING:
                logger.setLevel(logging.WARNING)

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

        tools_json = json.dumps(self.tools)
        self.system_prompt = (
            "You are an AI agent playing Blackjack who uses extreme long chains of thought "
            "to carefully consider the probabilities and optimal strategy. "
            "You need to decide whether to hit or stick based on your current hand and the dealer's showing card.\n\n"
            "You should enclose your thoughts and internal monologue inside <think> </think> tags, and then "
            "provide your decision using the take_action function call. You may use extremely long chains "
            "of thought to carefully consider the probabilities and optimal strategy.\n\n"
            f"<tools>\n{tools_json}\n</tools>\n\n"
            "For your function call, return a JSON object with function name and arguments "
            "within <tool_call> </tool_call> tags with the following schema:\n"
            '<tool_call>\n{"arguments": {"action": "hit"}, "name": "take_action"}\n</tool_call>\n\n'
            "Your full answer format should be:\n"
            "<think>\n[Your detailed reasoning process about whether to hit or stick]\n</think>\n\n"
            '<tool_call>\n{"arguments": {"action": "stick"}, "name": "take_action"}\n</tool_call>\n\n'
            "Remember to carefully consider the probabilities and optimal strategy for Blackjack."
        )

    def _get_or_create_episode(self, seed: int) -> EpisodeState:
        if seed not in self.episodes:
            env = gymnasium.make(self.config.env_name)
            obs, _ = env.reset(seed=seed)
            ep = EpisodeState(seed, env)
            ep.message_history = [{"role": "system", "content": self.system_prompt}]
            ep.message_history.append(
                {"role": "environment", "content": self._format_observation(obs)}
            )
            self.episodes[seed] = ep
        return self.episodes[seed]

    def _format_observation(self, obs: Tuple[int, int, int]) -> str:
        player_sum, dealer_card, usable_ace = obs
        return f"Your hand sum is {player_sum}. Dealer showing: {dealer_card}. You have a usable ace: {usable_ace}."

    def _score_response(
        self,
        env_reward: float,
        response_text: str,
        parsed_action: int,
        episode_seed: int,
    ) -> float:
        """
        Calculates a score for a single agent response based purely on environment reward
        and a penalty for invalid action format.
        """
        current_env_reward = env_reward * 1.0
        # Action is good?
        if parsed_action == -1:
            current_env_reward -= 0.2
        else:
            current_env_reward += 0.2

        # Check the thinking tags exist
        match = re.search(r"<think>(.*?)</think>", response_text)
        if match:
            thinking_content = match.group(1)
            if thinking_content:
                current_env_reward += 0.2
            # Check there's actually valid content (not just whitespace)
            if not thinking_content.strip():
                current_env_reward -= 0.2
        else:
            current_env_reward -= 0.2

        # Calculate the number of tokens in the agent's response
        if response_text:
            num_tokens = len(self.tokenizer.encode(response_text))
        else:
            num_tokens = 0

        # tiebreak & small length penalty
        if self.config.max_token_length > 0:
            token_ratio = min(1.0, num_tokens / self.config.max_token_length)
            tiebreak_bonus = self.config.tiebreak_token_factor * (1.0 - token_ratio)
            current_env_reward += tiebreak_bonus
        return current_env_reward

    def _parse_tool_call(self, response: str) -> int:
        if not response:
            logger.warning(
                "Attempted to parse an empty response string. Returning invalid action (-1)."
            )
            return -1

        parsed_name, parsed_args, is_error = parse_tool_call(
            response, self.tools, ["tool_call"]
        )
        if is_error:
            error_detail = (
                parsed_name
                if isinstance(parsed_name, str) and parsed_name
                else "Parser indicated error, but no specific message was returned in the typical error slot."
            )
            logger.warning(
                f"Failed to parse tool call. Full response: '{response}'. Error detail: {error_detail}"
            )
            return -1

        action = parsed_args.get("action", "").lower()
        if action == "hit":
            return 1
        elif action == "stick":
            return 0
        else:
            logger.warning(
                f"Successfully parsed tool call, but action is invalid. Action: '{action}'. "
                f"Full response: '{response}'. Parsed args: {parsed_args}"
            )
            return -1

    async def _sample_response(self, messages: List[Dict], n: int = 1) -> List[str]:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        try:
            completions = await self.server.completion(
                prompt=prompt,
                n=n,
                max_tokens=self.config.max_token_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
            return [choice.text for choice in completions.choices]
        except Exception as e:
            logger.error(f"API error during completion: {e}")
            return []

    async def _estimate_value(
        self,
        episode_seed_for_sim: int,
        env_actions_to_replay: List[int],
    ) -> float:
        """Calculate exact state value V*(s)

        Args:
            episode_seed_for_sim: The seed of the original episode for deterministic env creation.
            env_actions_to_replay: List of environment actions (0 or 1) taken to reach current state s.
        """
        v_star_cache: Dict[Tuple[int, int, int], float] = {}

        def _get_v_star_recursive(
            obs_tuple: Tuple[int, int, int], current_env: gymnasium.Env
        ) -> float:
            player_sum, dealer_card, usable_ace = obs_tuple

            # Base Case 1: Bust
            if player_sum > 21:
                return -1.0

            # Base Case 2: Check memoization cache
            if obs_tuple in v_star_cache:
                return v_star_cache[obs_tuple]

            env_for_stick = copy.deepcopy(current_env)
            _, reward_stick, _, _, _ = env_for_stick.step(0)
            # stick is terminal, so reward is final outcome
            q_star_stick = reward_stick

            # Q-value for HIT (action 1)
            env_for_hit = copy.deepcopy(current_env)
            obs_hit, reward_hit, term_hit, trunc_hit, _ = env_for_hit.step(1)

            if term_hit or trunc_hit:  # Game ended after hitting
                q_star_hit = reward_hit
            else:  # Game continues, recursively find V* of next state
                # reward_hit is typically 0 if the game didn't end
                q_star_hit = reward_hit + _get_v_star_recursive(obs_hit, env_for_hit)

            v_star = max(q_star_stick, q_star_hit)
            v_star_cache[obs_tuple] = v_star
            return v_star

        sim_env = None
        try:
            sim_env = gymnasium.make(self.config.env_name)
            current_obs, _ = sim_env.reset(seed=episode_seed_for_sim)

            # Replay actions to reach the current state s_t
            is_terminal_after_replay = False
            for action_idx, prev_action in enumerate(env_actions_to_replay):
                current_obs, _, term_replay, trunc_replay, _ = sim_env.step(prev_action)
                if term_replay or trunc_replay:
                    logger.debug(
                        f"[_estimate_value] State became terminal during action replay "
                        f"(action {action_idx+1}/{len(env_actions_to_replay)} of prev_actions). Value is 0."
                    )
                    is_terminal_after_replay = True
                    break

            if is_terminal_after_replay:
                return 0.0
            final_v_star = _get_v_star_recursive(current_obs, sim_env)
            return final_v_star

        except Exception as e:
            logger.error(
                f"[_estimate_value] Error during exact value"
                f" calculation for seed {episode_seed_for_sim}, "
                f"actions {env_actions_to_replay}: {e}",
                exc_info=True,
            )
            return 0.0
        finally:
            if sim_env is not None:
                sim_env.close()

    async def collect_trajectory(self, seed: int) -> List[BlackjackScoredDataGroup]:
        """Collect data for ONE trajectory, evaluating G alternatives per step using MC advantages."""
        G = self.config.group_size
        max_turns = self.config.max_turns or 5

        trajectory_data_for_trainer: List[BlackjackScoredDataGroup] = []
        episode_summary_metrics: Optional[Dict[str, Any]] = None

        logger.info(
            f"[Collect Trajectory Seed: {seed}] Starting trajectory. Group size G={G}."
        )

        try:
            ep = self._get_or_create_episode(seed)
        except Exception as e:
            logger.error(
                f"[Collect Trajectory Seed: {seed}] Failed to create/get episode: {e}",
                exc_info=True,
            )
            return []

        for turn in range(max_turns):
            current_state_messages = ep.message_history.copy()
            logger.debug(
                f"[Collect Trajectory Seed: {seed} Turn: {turn+1}/{max_turns}] "
                f"Current state history length: {len(current_state_messages)}"
            )

            try:
                value_t = await self._estimate_value(
                    episode_seed_for_sim=ep.seed, env_actions_to_replay=ep.actions
                )
                logger.debug(
                    f"[Collect Trajectory Seed: {seed} Turn: {turn+1}] Estimated V(s_t) = {value_t:.4f}"
                )
            except Exception as e_vt:
                logger.error(
                    f"[Collect Trajectory Seed: {seed} Turn: {turn+1}] Error estimating V(s_t): {e_vt}",
                    exc_info=True,
                )
                break

            messages_for_llm = current_state_messages.copy()
            agent_prompt_content = "<think>\n" if self.config.thinking_active else ""
            messages_for_llm.append({"role": "agent", "content": agent_prompt_content})

            try:
                responses = await self._sample_response(messages_for_llm, n=G)
                if len(responses) != G:
                    logger.error(
                        f"[Collect Trajectory Seed: {seed} Turn: {turn+1}] "
                        f"Expected {G} responses, got {len(responses)}. "
                        f"Aborting trajectory."
                    )
                    break
            except Exception as e_sample:
                logger.error(
                    f"[Collect Trajectory Seed: {seed} Turn: {turn+1}] Error sampling responses: {e_sample}",
                    exc_info=True,
                )
                break

            alt_full_responses: List[str] = []
            alt_parsed_actions: List[int] = []
            alt_env_actions: List[int] = []
            alt_raw_rewards: List[float] = []
            alt_combined_rewards: List[float] = []
            alt_next_state_msgs: List[List[Dict]] = []
            alt_is_terminal: List[bool] = []
            alt_tokens: List[List[int]] = []
            alt_masks: List[List[int]] = []
            alt_value_next: List[float] = []
            alt_advantages: List[float] = []

            for i in range(G):
                llm_output_only = responses[i]
                full_agent_response = agent_prompt_content + llm_output_only
                alt_full_responses.append(full_agent_response)

                parsed_action = self._parse_tool_call(full_agent_response)
                alt_parsed_actions.append(parsed_action)

                env_action = parsed_action if parsed_action != -1 else 0
                alt_env_actions.append(env_action)

                sim_env = None
                raw_env_reward_i = 0.0
                term_i, trunc_i = False, False
                next_state_msgs_i = []
                try:
                    sim_env = gymnasium.make(self.config.env_name)
                    _, _ = sim_env.reset(seed=ep.seed)
                    for prev_action in ep.actions:
                        _, _, term_replay, trunc_replay, _ = sim_env.step(prev_action)
                        if term_replay or trunc_replay:
                            logger.error(
                                f"[Collect Trajectory Seed: {seed} Turn: {turn+1} Alt: {i}] "
                                f"Sim env terminated during replay. State mismatch?"
                            )
                            term_i, trunc_i = True, True
                            raw_env_reward_i = 0.0
                            break

                    if not (term_i or trunc_i):
                        sim_obs_next, raw_env_reward_i, term_i, trunc_i, _ = (
                            sim_env.step(env_action)
                        )

                    alt_raw_rewards.append(raw_env_reward_i)
                    alt_is_terminal.append(term_i or trunc_i)

                    combined_reward_i = self._score_response(
                        raw_env_reward_i, full_agent_response, parsed_action, ep.seed
                    )
                    alt_combined_rewards.append(combined_reward_i)

                    current_state_plus_response = current_state_messages + [
                        {"role": "agent", "content": full_agent_response}
                    ]
                    if sim_obs_next is not None:
                        next_state_msgs_i = current_state_plus_response + [
                            {
                                "role": "environment",
                                "content": self._format_observation(sim_obs_next),
                            }
                        ]
                    else:
                        next_state_msgs_i = current_state_plus_response
                    alt_next_state_msgs.append(next_state_msgs_i)

                    tokenized_i = tokenize_for_trainer_multistep(
                        self.tokenizer, next_state_msgs_i
                    )
                    alt_tokens.append(tokenized_i["tokens"])
                    alt_masks.append(tokenized_i["masks"])

                except Exception as e_sim:
                    logger.error(
                        f"[Collect Trajectory Seed: {seed} Turn: {turn+1} Alt: {i}] "
                        f"Error simulating action {env_action}: {e_sim}",
                        exc_info=True,
                    )
                    alt_raw_rewards.append(0.0)
                    alt_combined_rewards.append(-1.0)
                    alt_next_state_msgs.append(
                        current_state_messages
                        + [{"role": "agent", "content": full_agent_response}]
                    )
                    alt_is_terminal.append(True)
                    alt_tokens.append([])
                    alt_masks.append([])
                finally:
                    if sim_env:
                        sim_env.close()

            alt_value_next: List[float] = []
            for i in range(G):
                if not alt_is_terminal[i]:
                    try:
                        actions_to_reach_s_prime = ep.actions + [alt_env_actions[i]]
                        value_next_i = await self._estimate_value(
                            episode_seed_for_sim=ep.seed,
                            env_actions_to_replay=actions_to_reach_s_prime,
                        )
                        alt_value_next.append(value_next_i)
                    except Exception as e_vn:
                        logger.error(
                            f"[Collect Trajectory Seed: {seed} Turn: {turn+1} Alt: {i}] "
                            f"Error estimating V(s'): {e_vn}",
                            exc_info=True,
                        )
                        alt_value_next.append(0.0)
                else:
                    alt_value_next.append(0.0)

            for i in range(G):
                advantage_i = alt_combined_rewards[i] + alt_value_next[i] - value_t
                # If we pass this then instead of raw scores, implicitly, we're
                # doing some credit assignment. Could maybe do bonus on a win too
                # and apply with a discount factor to alts in winning trajectories
                alt_advantages.append(advantage_i)
                logger.debug(
                    f"[Collect Trajectory Seed: {seed} Turn: {turn+1} Alt: {i}] "
                    f"CombinedR={alt_combined_rewards[i]:.2f}, V_t={value_t:.2f}, "
                    f"V_t+1={alt_value_next[i]:.2f} => Advantage={advantage_i:.2f}"
                )

            if (
                len(alt_tokens) != G
                or len(alt_masks) != G
                or len(alt_advantages) != G
                or len(alt_next_state_msgs) != G
                or len(alt_parsed_actions) != G
            ):
                # sanity check
                logger.error(
                    f"[Collect Trajectory Seed: {seed} Turn: {turn+1}] "
                    f"Mismatch in alternative list lengths after processing. "
                    f"Aborting trajectory."
                )
                break

            trajectory_data_for_trainer.append(
                BlackjackScoredDataGroup(
                    seed=ep.seed,
                    tokens=alt_tokens,
                    masks=alt_masks,
                    scores=alt_advantages,
                    messages=alt_next_state_msgs,
                    parsed_actions=alt_parsed_actions,
                )
            )

            # Prepare items for sorting: (-advantage, token_length, original_index)
            sortable_alternatives = []
            for i in range(G):
                # alt_tokens[i] is expected to be a list (possibly empty)
                token_len = len(alt_tokens[i])
                sortable_alternatives.append((-alt_advantages[i], token_len, i))

            sortable_alternatives.sort()

            best_advantage_idx = sortable_alternatives[0][2]

            # Log details of the selected alternative based on the sort
            chosen_advantage_for_log = -sortable_alternatives[0][0]
            chosen_token_length_for_log = sortable_alternatives[0][1]
            logger.debug(
                f"[Collect Trajectory Seed: {seed} Turn: {turn+1}] "
                f"Selected Alt {best_advantage_idx} "
                f"(Adv: {chosen_advantage_for_log:.2f}, "
                f"Tokens: {chosen_token_length_for_log}) "
                f"from {G} alternatives using sort."
            )

            chosen_env_action = alt_env_actions[best_advantage_idx]
            chosen_full_response = alt_full_responses[best_advantage_idx]
            chosen_raw_env_reward = alt_raw_rewards[best_advantage_idx]
            chosen_is_terminal = alt_is_terminal[best_advantage_idx]
            chosen_parsed_action = alt_parsed_actions[best_advantage_idx]

            logger.info(
                f"[Collect Trajectory Seed: {seed} Turn: {turn+1}] Chosen action to step env: "
                f"{chosen_env_action} (from Alt {best_advantage_idx} with "
                f"Adv {chosen_advantage_for_log:.2f})"
            )

            ep.num_total_actions += 1
            if chosen_parsed_action != -1:
                ep.num_correct_actions += 1

            ep.message_history = current_state_messages

            response_for_history = self._truncate_thinking_for_history(
                chosen_full_response, self.config.max_think_chars_history
            )
            ep.message_history.append(
                {"role": "agent", "content": response_for_history}
            )

            try:
                main_obs, main_reward, main_term, main_trunc, _ = ep.env.step(
                    chosen_env_action
                )
                if abs(main_reward - chosen_raw_env_reward) > 1e-6:
                    logger.warning(
                        f"[Collect Trajectory Seed: {seed} Turn: {turn+1}] "
                        f"Mismatch between simulated reward ({chosen_raw_env_reward}) and "
                        f"main env step reward ({main_reward}) for chosen action {chosen_env_action}."
                    )
                if (main_term or main_trunc) != chosen_is_terminal:
                    logger.warning(
                        f"[Collect Trajectory Seed: {seed} Turn: {turn+1}] "
                        f"Mismatch between simulated terminal state ({chosen_is_terminal}) and "
                        f"main env step terminal state ({(main_term or main_trunc)}) "
                        f"for chosen action {chosen_env_action}."
                    )

                term = main_term
                trunc = main_trunc
                obs = main_obs
                ep.actions.append(chosen_env_action)
                ep.step_rewards.append(main_reward)
                ep.num_steps += 1

                if obs:
                    ep.message_history.append(
                        {
                            "role": "environment",
                            "content": self._format_observation(obs),
                        }
                    )
            except Exception as e_main_step:
                logger.error(
                    f"[Collect Trajectory Seed: {seed} Turn: {turn+1}] "
                    f"Error stepping MAIN environment with chosen action {chosen_env_action}: {e_main_step}",
                    exc_info=True,
                )
                term, trunc = True, True

            if term or trunc:
                ep.total_reward = sum(ep.step_rewards)
                logger.info(
                    f"[Collect Trajectory Seed: {seed}] Trajectory ended. "
                    f"Term={term}, Trunc={trunc}. Total raw env reward: {ep.total_reward}"
                )
                break

        final_raw_reward = sum(ep.step_rewards) if ep.step_rewards else 0.0
        logger.info(
            f"[Collect Trajectory Seed: {seed}] Finished collecting trajectory. "
            f"Steps collected: {len(trajectory_data_for_trainer)}, "
            f"Final raw reward: {final_raw_reward:.2f}"
        )

        if ep:
            game_outcome = 0
            if final_raw_reward > 0:
                game_outcome = 1
            elif final_raw_reward < 0:
                game_outcome = -1
            # debugging
            episode_summary_metrics = {
                "seed": ep.seed,
                "total_reward": final_raw_reward,
                "num_steps": ep.num_steps,
                "num_correct_actions": ep.num_correct_actions,
                "num_total_actions": ep.num_total_actions,
                "game_outcome": game_outcome,
            }
            self.completed_episode_metrics_buffer.append(episode_summary_metrics)
            logger.debug(
                f"[Collect Trajectory Seed: {seed}] Added episode summary to buffer: {episode_summary_metrics}"
            )

        if seed in self.episodes:
            try:
                self.episodes[seed].env.close()
            except Exception as e_close:
                logger.warning(
                    f"[Collect Trajectory Seed: {seed}] Exception closing final env: {e_close}"
                )
            del self.episodes[seed]

        return self._ensure_trajectory_token_limit(trajectory_data_for_trainer)

    async def score(
        self, rollout_group_data: List[BlackjackScoredDataGroup]
    ) -> List[Optional[BlackjackScoredDataGroup]]:
        """Pass through rollout data. The 'scores' field in BlackjackScoredDataGroup
        already contains the A*(s,a) advantages from the collection phase.
        """
        logger.info(f"[Score] Processing {len(rollout_group_data)} steps.")
        return rollout_group_data

    async def collect_trajectories(
        self, item: Tuple[int, int]
    ) -> Tuple[List[BlackjackScoredDataGroup], List[Tuple[int, int]]]:
        seed, _ = item
        traj = await self.collect_trajectory(seed)
        if not traj:
            logger.warning(f"[Collect Trajectories] Empty trajectory for seed {seed}.")
        return traj, []

    async def setup(self):
        pass

    async def get_next_item(self) -> Tuple[int, int]:
        return (random.randint(0, 1000000), 0)

    async def rollout_and_score_eval(self, seed: int) -> Dict[str, float]:
        """Run a single episode for evaluation with a single response per step."""
        ep = self._get_or_create_episode(seed)
        max_turns = self.config.max_turns or 5
        metrics = {
            "seed": seed,
            "total_reward": 0.0,
            "num_turns": 0,
            "num_correct_actions": 0,
            "num_invalid_actions": 0,
            "game_outcome": 0,
        }

        for turn in range(max_turns):
            messages = ep.message_history.copy()
            agent_prompt_content = "<think>\n" if self.config.thinking_active else ""
            messages.append({"role": "agent", "content": agent_prompt_content})

            responses = await self._sample_response(messages, n=1)
            if not responses:
                logger.error(
                    f"[Eval Seed: {seed}, Turn: {turn+1}] No response. Aborting."
                )
                break

            llm_output_only = responses[0]
            full_agent_response = agent_prompt_content + llm_output_only

            action = self._parse_tool_call(full_agent_response)
            if action == -1:
                metrics["num_invalid_actions"] += 1
                action = 0
            else:
                metrics["num_correct_actions"] += 1

            try:
                obs, reward, term, trunc, _ = ep.env.step(action)
            except Exception as e:
                logger.error(f"[Eval Seed: {seed}, Turn: {turn+1}] Env error: {e}")
                term = True
                reward = -1.0
                obs = None

            metrics["total_reward"] += reward
            metrics["num_turns"] = turn + 1

            response_for_history = self._truncate_thinking_for_history(
                full_agent_response, self.config.max_think_chars_history
            )

            ep.message_history.append(
                {"role": "agent", "content": response_for_history}
            )

            if obs:
                ep.message_history.append(
                    {"role": "environment", "content": self._format_observation(obs)}
                )

            if term or trunc:
                metrics["game_outcome"] = int(reward)
                logger.info(f"[Eval Seed: {seed}] Episode ended. Outcome: {reward}")
                break

        ep.env.close()
        del self.episodes[seed]
        return metrics

    async def evaluate(self, *args, **kwargs):
        if not self.config.use_wandb:
            logger.info("Skipping evaluation as wandb is not enabled.")
            return
        num_eval_episodes = self.config.eval_episodes
        logger.info(f"Starting evaluation for {num_eval_episodes} episodes.")
        eval_tasks = [
            self.rollout_and_score_eval(random.randint(1000001, 2000000))
            for _ in range(num_eval_episodes)
        ]
        all_metrics = await tqdm_asyncio.gather(*eval_tasks)
        valid_metrics = [m for m in all_metrics if m]
        if not valid_metrics:
            logger.warning("No valid evaluation metrics.")
            return

        num_completed = len(valid_metrics)
        avg_total_reward = sum(m["total_reward"] for m in valid_metrics) / num_completed
        avg_num_turns = sum(m["num_turns"] for m in valid_metrics) / num_completed
        total_correct = sum(m["num_correct_actions"] for m in valid_metrics)
        total_invalid = sum(m["num_invalid_actions"] for m in valid_metrics)
        total_actions = total_correct + total_invalid
        action_accuracy = total_correct / total_actions if total_actions > 0 else 0
        win_rate = (
            sum(1 for m in valid_metrics if m["game_outcome"] == 1) / num_completed
        )
        loss_rate = (
            sum(1 for m in valid_metrics if m["game_outcome"] == -1) / num_completed
        )
        draw_rate = (
            sum(1 for m in valid_metrics if m["game_outcome"] == 0) / num_completed
        )

        self.eval_metrics = [
            ("eval/avg_total_reward", avg_total_reward),
            ("eval/avg_num_turns", avg_num_turns),
            ("eval/action_accuracy", action_accuracy),
            ("eval/win_rate", win_rate),
            ("eval/loss_rate", loss_rate),
            ("eval/draw_rate", draw_rate),
            ("eval/num_completed_episodes", num_completed),
        ]
        logger.info(f"Evaluation completed. Metrics: {self.eval_metrics}")

    async def wandb_log(self, wandb_metrics: Optional[Dict[str, float]] = None):
        if wandb_metrics is None:
            wandb_metrics = {}
        if self.completed_episode_metrics_buffer:
            num_episodes = len(self.completed_episode_metrics_buffer)
            avg_reward = (
                sum(m["total_reward"] for m in self.completed_episode_metrics_buffer)
                / num_episodes
            )
            avg_steps = (
                sum(m["num_steps"] for m in self.completed_episode_metrics_buffer)
                / num_episodes
            )
            win_rate = (
                sum(
                    1
                    for m in self.completed_episode_metrics_buffer
                    if m["game_outcome"] == 1
                )
                / num_episodes
            )
            wandb_metrics[
                f"{self.wandb_prepend or 'blackjack'}_train/avg_episode_reward"
            ] = avg_reward
            wandb_metrics[
                f"{self.wandb_prepend or 'blackjack'}_train/avg_episode_steps"
            ] = avg_steps
            wandb_metrics[
                f"{self.wandb_prepend or 'blackjack'}_train/episode_win_rate"
            ] = win_rate
            wandb_metrics[f"{self.wandb_prepend or 'blackjack'}_train/num_episodes"] = (
                num_episodes
            )
            self.completed_episode_metrics_buffer = []
        await super().wandb_log(wandb_metrics)

    @classmethod
    def config_init(cls) -> Tuple[BlackjackEnvConfig, List[OpenaiConfig]]:
        env_config = BlackjackEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=16,
            use_wandb=True,
            max_num_workers=128,
            rollout_server_url="http://localhost:8000",
            total_steps=2000,
            batch_size=1024,
            steps_per_eval=20,
            max_token_length=1024 * 16,
            inference_weight=1.0,
            wandb_name="fundamental_metric_prediction",
            data_path_to_save_groups=None,
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.1,
            env_name="Blackjack-v1",
            temperature=0.7,
            top_p=0.9,
            max_turns=5,
            thinking_active=True,
            eval_episodes=100,
            max_think_chars_history=3000,
            max_trajectory_tokens=24576,
            debug_mode=False,
            tiebreak_token_factor=0.01,
        )
        server_configs = [
            OpenaiConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9004/v1",
                api_key="x",
                num_requests_for_eval=256,
            )
        ]
        return env_config, server_configs

    def _truncate_thinking_for_history(self, response_text: str, max_chars: int) -> str:
        """Helper to truncate the <think> block of a response for message history."""
        try:
            think_start_tag = "<think>"
            think_end_tag = "</think>"

            think_start_idx = response_text.find(think_start_tag)
            think_end_idx = response_text.find(think_end_tag)

            if (
                think_start_idx != -1
                and think_end_idx != -1
                and think_start_idx < think_end_idx
            ):
                part_before_content = response_text[
                    : think_start_idx + len(think_start_tag)
                ]
                original_think_content = response_text[
                    think_start_idx + len(think_start_tag) : think_end_idx
                ].strip()
                part_after_content = response_text[think_end_idx:]

                truncated_think_content = original_think_content
                is_truncated = False

                if not original_think_content:
                    return response_text

                paragraphs = [
                    p.strip() for p in original_think_content.split("\n\n") if p.strip()
                ]
                if len(paragraphs) > 0:
                    last_paragraph = paragraphs[-1]
                    if len(last_paragraph) < len(original_think_content):
                        truncated_think_content = last_paragraph
                        is_truncated = True
                    elif len(original_think_content) > max_chars:
                        truncated_think_content = original_think_content[-max_chars:]
                        is_truncated = True
                elif len(original_think_content) > max_chars:
                    truncated_think_content = original_think_content[-max_chars:]
                    is_truncated = True

                if is_truncated and truncated_think_content:
                    if not truncated_think_content.startswith("... "):
                        truncated_think_content = (
                            "... " + truncated_think_content.lstrip()
                        )

                if (
                    not truncated_think_content.strip()
                    or truncated_think_content.strip() == "..."
                ):
                    final_content_for_block = ""
                else:
                    final_content_for_block = f"\n{truncated_think_content.strip()}\n"

                return f"{part_before_content.rstrip()}{final_content_for_block}{part_after_content.lstrip()}"

            return response_text
        except Exception as e:
            logger.error(
                f"Error in _truncate_thinking_for_history for text '{response_text[:200]}...': {e}",
                exc_info=True,
            )
            return response_text

    def _ensure_trajectory_token_limit(
        self, trajectory: List[BlackjackScoredDataGroup]
    ) -> List[BlackjackScoredDataGroup]:
        """
        Ensure token sequences in a trajectory don't exceed max_trajectory_tokens.
        Attempts to uniformly truncate older messages (preferably paired turns) from all alternatives within a step.
        The system prompt, last environment observation, and last agent response are preserved as a minimum.
        If a step still exceeds the limit after maximum possible truncation, it is discarded.

        Args:
            trajectory: List of BlackjackScoredDataGroup from an episode

        Returns:
            The trajectory with potentially truncated messages/tokens/masks or filtered steps
        """
        if not trajectory:
            return trajectory

        filtered_trajectory: List[BlackjackScoredDataGroup] = []

        for step_idx, original_step_data in enumerate(trajectory):
            if not (
                original_step_data.get("messages")
                and original_step_data.get("tokens")
                and original_step_data.get("masks")
                and original_step_data.get("seed") is not None
                and original_step_data.get("parsed_actions") is not None
            ):
                logger.warning(
                    f"[_ensure_trajectory_token_limit] Step {step_idx} in MC env "
                    f"is missing critical data. Skipping."
                )
                continue

            max_initial_tokens = 0
            if original_step_data["tokens"]:
                max_initial_tokens = (
                    max(
                        len(alt_tokens)
                        for alt_tokens in original_step_data["tokens"]
                        if isinstance(alt_tokens, list)
                    )
                    if any(
                        isinstance(alt_tokens, list)
                        for alt_tokens in original_step_data["tokens"]
                    )
                    else 0
                )

            if max_initial_tokens <= self.config.max_trajectory_tokens:
                filtered_trajectory.append(original_step_data)
                logger.info(
                    f"[_ensure_trajectory_token_limit] Step {step_idx} compliant in MC env. "
                    f"Max tokens: {max_initial_tokens}"
                )
                continue

            logger.info(
                f"[_ensure_trajectory_token_limit] Step {step_idx} in MC env (max tokens: {max_initial_tokens}) "
                f"exceeds limit ({self.config.max_trajectory_tokens}). Attempting truncation."
            )

            working_messages = [
                msgs_list.copy() for msgs_list in original_step_data["messages"] or []
            ]
            working_tokens = [
                tkns_list.copy() for tkns_list in original_step_data["tokens"] or []
            ]
            working_masks = [
                msks_list.copy() for msks_list in original_step_data["masks"] or []
            ]
            max_current_tokens = max_initial_tokens
            num_alternatives = len(working_messages)

            if num_alternatives == 0:
                logger.warning(
                    f"[_ensure_trajectory_token_limit] Step {step_idx} in MC env has no alternatives"
                    " after copying. Skipping."
                )
                continue

            retokenization_error_this_step = False
            while max_current_tokens > self.config.max_trajectory_tokens:
                target_pop_counts_per_alt = []
                for alt_idx in range(num_alternatives):
                    alt_msg_list = working_messages[alt_idx]
                    num_preserved_at_end = 0
                    if len(alt_msg_list) > 1 and alt_msg_list[-1]["role"] in [
                        "agent",
                        "assistant",
                    ]:
                        num_preserved_at_end = 1
                        if (
                            len(alt_msg_list) > 2
                            and alt_msg_list[-2]["role"] == "environment"
                        ):
                            num_preserved_at_end = 2

                    available_to_pop = len(alt_msg_list) - 1 - num_preserved_at_end

                    if available_to_pop <= 0:
                        target_pop_counts_per_alt.append(0)
                    else:
                        can_pop_pair = (
                            available_to_pop >= 2
                            and len(alt_msg_list) > 2
                            and alt_msg_list[1]["role"] == "environment"
                            and alt_msg_list[2]["role"] in ["agent", "assistant"]
                        )
                        if can_pop_pair:
                            target_pop_counts_per_alt.append(2)
                        else:
                            target_pop_counts_per_alt.append(1)

                positive_pop_counts = [c for c in target_pop_counts_per_alt if c > 0]
                if not positive_pop_counts:
                    break

                min_pop_this_round = min(positive_pop_counts)
                temp_new_alt_tokens = []
                temp_new_alt_masks = []
                max_tokens_after_this_trunc = 0

                for alt_idx in range(num_alternatives):
                    for _ in range(min_pop_this_round):
                        if len(working_messages[alt_idx]) > 1:
                            working_messages[alt_idx].pop(1)
                        else:
                            logger.error(
                                f"[_ensure_trajectory_token_limit] MC env: Critical error during pop for "
                                f"alt {alt_idx}, step {step_idx}. List too short."
                            )
                            retokenization_error_this_step = True
                            break
                    if retokenization_error_this_step:
                        break

                    try:
                        tokenized_alt = tokenize_for_trainer_multistep(
                            self.tokenizer, working_messages[alt_idx]
                        )
                        temp_new_alt_tokens.append(tokenized_alt["tokens"])
                        temp_new_alt_masks.append(tokenized_alt["masks"])
                        max_tokens_after_this_trunc = max(
                            max_tokens_after_this_trunc, len(tokenized_alt["tokens"])
                        )
                    except Exception as e:
                        logger.error(
                            f"[_ensure_trajectory_token_limit] MC env: Error re-tokenizing alt {alt_idx} "
                            f"in step {step_idx} after truncation: {e}"
                        )
                        retokenization_error_this_step = True
                        break

                if retokenization_error_this_step:
                    break

                working_tokens = temp_new_alt_tokens
                working_masks = temp_new_alt_masks
                max_current_tokens = max_tokens_after_this_trunc
                logger.debug(
                    f"[_ensure_trajectory_token_limit] MC env: Step {step_idx}, "
                    f"after uniform pop of {min_pop_this_round}, "
                    f"max tokens: {max_current_tokens}"
                )

            if (
                not retokenization_error_this_step
                and max_current_tokens <= self.config.max_trajectory_tokens
            ):
                updated_step_data: BlackjackScoredDataGroup = {
                    "seed": original_step_data["seed"],
                    "messages": working_messages,
                    "tokens": working_tokens,
                    "masks": working_masks,
                    "scores": original_step_data.get("scores"),
                    "parsed_actions": original_step_data.get("parsed_actions"),
                }
                filtered_trajectory.append(updated_step_data)
                logger.info(
                    f"[_ensure_trajectory_token_limit] MC env: Step {step_idx} successfully processed. "
                    f"Final max tokens: {max_current_tokens}"
                )
            else:
                logger.warning(
                    f"[_ensure_trajectory_token_limit] MC env: Discarding step {step_idx}. "
                    f"Max tokens ({max_current_tokens}) still exceed limit ({self.config.max_trajectory_tokens}) "
                    f"or retokenization error occurred ({retokenization_error_this_step})."
                )

        if len(filtered_trajectory) < len(trajectory):
            logger.warning(
                f"[_ensure_trajectory_token_limit] MC env: Filtered out "
                f"{len(trajectory) - len(filtered_trajectory)} steps "
                f"due to token limit constraints. Original: {len(trajectory)}, Filtered: {len(filtered_trajectory)}"
            )
        return filtered_trajectory

    @classmethod
    def cli(cls):
        super().cli()


if __name__ == "__main__":
    BlackjackEnv.cli()
