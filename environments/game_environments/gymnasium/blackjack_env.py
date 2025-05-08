import gymnasium
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import logging
from tqdm.asyncio import tqdm_asyncio
import json
import os
import yaml
import random
from atroposlib.envs.base import BaseEnv, BaseEnvConfig, OpenaiConfig, ScoredDataGroup
from atroposlib.utils.tool_call_parser import parse_tool_call
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer
from atroposlib.envs.reward_fns import registry
from atroposlib.envs.reward_fns.combined_reward import CombinedReward
from typing import Any

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
    group_size: int = 8  # G for GRPO
    mc_samples: int = 9  # K for MC value estimation
    reward_functions: List[Union[str, Dict[str, Any]]] = []
    format_reward_weight: float = 0.5
    environment_reward_weight: float = 0.5

class BlackjackScoredDataGroup(ScoredDataGroup):
    seed: int # Seed of the trajectory this step belongs to
    # Fields below are lists, where each element corresponds to one of the G alternatives
    tokens: Optional[List[List[int]]] = None # List of token lists for G alternatives
    masks: Optional[List[List[int]]] = None  # List of mask lists for G alternatives
    scores: Optional[List[float]] = None     # List of advantage scores for G alternatives
    messages: Optional[List[List[Dict]]] = None # List of message lists for G alternatives
    parsed_actions: Optional[List[int]] = None # List of parsed actions (0, 1, or -1) for G alternatives
    # Removed singular parsed_action

class EpisodeState:
    def __init__(self, seed: int, env: gymnasium.Env):
        self.seed = seed
        self.env = env
        self.message_history: List[Dict] = []
        self.actions: List[int] = []
        self.step_rewards: List[float] = []
        self.total_reward: float = 0.0
        self.num_steps: int = 0
        self.num_correct_actions: int = 0  # Added for action accuracy tracking
        self.num_total_actions: int = 0    # Added for action accuracy tracking

class BlackjackEnv(BaseEnv):
    def __init__(self, config: BlackjackEnvConfig, server_configs: List[OpenaiConfig], slurm: bool = True, testing: bool = False):
        super().__init__(config, server_configs, slurm, testing)
        self.episodes: Dict[int, EpisodeState] = {}
        self.debug_mode = config.debug_mode
        self.completed_episode_metrics_buffer: List[Dict[str, float]] = []
        if self.debug_mode:
            logger.setLevel(logging.DEBUG)
        else:
            if logger.level == logging.NOTSET or logger.level > logging.WARNING:
                logger.setLevel(logging.WARNING)

        self.tools = [{
            "type": "function",
            "function": {
                "name": "take_action",
                "description": "Choose to 'hit' or 'stick' in Blackjack.",
                "parameters": {"action": {"type": "string", "enum": ["hit", "stick"]}},
            },
        }]

        # Initialize reward function
        self.reward_function = self._initialize_reward_function()

        tools_json = json.dumps(self.tools)
        self.system_prompt = (
            "You are an AI agent playing Blackjack who uses extreme long chains of thought to carefully consider the probabilities and optimal strategy. "
            "You need to decide whether to hit or stick based on your current hand and the dealer's showing card.\n\n"
            "You should enclose your thoughts and internal monologue inside <think> </think> tags, and then "
            "provide your decision using the take_action function call. You may use extremely long chains "
            "of thought to carefully consider the probabilities and optimal strategy.\n\n"
            f"<tools>\n{tools_json}\n</tools>\n\n"
            "For your function call, return a JSON object with function name and arguments within <tool_call> </tool_call> "
            "tags with the following schema:\n"
            '<tool_call>\n{"arguments": {"action": "hit"}, "name": "take_action"}\n</tool_call>\n\n'
            "Your full answer format should be:\n"
            "<think>\n[Your detailed reasoning process about whether to hit or stick]\n</think>\n\n"
            '<tool_call>\n{"arguments": {"action": "stick"}, "name": "take_action"}\n</tool_call>\n\n'
            "Remember to carefully consider the probabilities and optimal strategy for Blackjack."
        )

    def _initialize_reward_function(self):
        """Initialize the combined reward function for scoring."""
        if hasattr(self.config, "reward_functions") and self.config.reward_functions:
            reward_configs = []
            for reward_func_config_item in self.config.reward_functions:
                if isinstance(reward_func_config_item, str):
                    if reward_func_config_item == "format":
                        format_config = {
                            "type": "format",
                            "weight": self.config.format_reward_weight,
                            "params": {
                                "preferred_tags": ["think", "tool_call"],
                            },
                        }
                        reward_configs.append(format_config)
                    elif reward_func_config_item == "tool_calling":
                        tool_calling_config = {
                            "type": "tool_calling",
                            "weight": self.config.format_reward_weight,
                            "params": {
                                "tools": self.tools,
                                "preferred_tags": ["tool_call"],
                                "check_arguments": True,
                            },
                        }
                        reward_configs.append(tool_calling_config)
                    else:
                        logger.warning(f"Unrecognized string reward function type: {reward_func_config_item}. Attempting to use directly.")
                        reward_configs.append(reward_func_config_item) 
                elif isinstance(reward_func_config_item, dict):
                    reward_configs.append(reward_func_config_item)
                else:
                    logger.warning(f"Skipping invalid reward function configuration item: {reward_func_config_item}")
            
            if not reward_configs:
                logger.warning("Reward function configs were processed but resulted in an empty list. No reward function will be active.")
                return None

            if len(reward_configs) == 1:
                try:
                    return registry.create(reward_configs[0])
                except Exception as e:
                    logger.error(f"Failed to create single reward function from config {reward_configs[0]}: {e}")
                    return None
            elif len(reward_configs) > 1:
                try:
                    return CombinedReward(rewards=reward_configs, normalization="none")
                except Exception as e:
                    logger.error(f"Failed to create CombinedReward function from configs: {e}")
                    return None
        logger.info("No reward_functions specified in config or it's empty. No format/tool-call reward function will be active.")
        return None

    def _get_or_create_episode(self, seed: int) -> EpisodeState:
        if seed not in self.episodes:
            env = gymnasium.make(self.config.env_name)
            obs, _ = env.reset(seed=seed)
            ep = EpisodeState(seed, env)
            ep.message_history = [{"role": "system", "content": self.system_prompt}]
            ep.message_history.append({"role": "environment", "content": self._format_observation(obs)})
            self.episodes[seed] = ep
        return self.episodes[seed]

    def _format_observation(self, obs: Tuple[int, int, int]) -> str:
        player_sum, dealer_card, usable_ace = obs
        return f"Your hand sum is {player_sum}. Dealer showing: {dealer_card}. You have a usable ace: {usable_ace}."

    def _score_response(
        self,
        env_reward: float,          # Raw reward from the environment step
        response_text: str,       # Full agent response text (<think> + <tool_call>)
        parsed_action: int,       # Parsed action (0 for stick, 1 for hit, -1 for error)
        episode_seed: int         # Seed of the current episode for logging context
    ) -> float:
        """
        Calculates a combined score for a single agent response based on environment and format rewards.
        """
        format_reward_component = 0.0
        current_env_reward = env_reward # Start with the raw environment reward

        # Apply a penalty if the action parsing failed
        if parsed_action == -1:
            # This penalty is for not adhering to the tool call format that allows parsing.
            # Specific tool argument errors might be handled by the ToolCallingReward function if configured.
            current_env_reward -= 0.5 
            logger.debug(f"[_score_response Seed: {episode_seed}] Penalty applied to env_reward for invalid action format (-0.5). Current env_reward: {current_env_reward}")

        if self.reward_function:
            # The reward_function (e.g., CombinedReward) expects a list of lists of messages.
            # We wrap the single response appropriately.
            # The structure is [[{'role': 'agent', 'content': response_text}]]
            messages_for_format_reward: List[List[Dict[str, str]]] = [[{"role": "agent", "content": response_text}]]
            try:
                # The reward_function should return a list of scores, one for each item in the outer list.
                format_rewards_list = self.reward_function(messages_for_format_reward)
                if format_rewards_list and len(format_rewards_list) > 0:
                    format_reward_component = format_rewards_list[0] # We expect a single score back
                    logger.debug(f"[_score_response Seed: {episode_seed}] Format/ToolCall reward calculated: {format_reward_component:.4f}")
                else:
                    logger.warning(f"[_score_response Seed: {episode_seed}] Format reward function returned empty or invalid result: {format_rewards_list}")
            except Exception as e:
                logger.error(f"[_score_response Seed: {episode_seed}] Error calculating format/tool-call reward: {e}", exc_info=True)
        else:
            logger.debug(f"[_score_response Seed: {episode_seed}] No reward_function active, format_reward_component is 0.")

        # Get weights from config
        env_w = self.config.environment_reward_weight
        fmt_w = self.config.format_reward_weight
        
        # Calculate the final combined score
        combined_score = (env_w * current_env_reward) + (fmt_w * format_reward_component)

        logger.debug(
            f"[_score_response Seed: {episode_seed}] Score Calculation: "
            f"EnvReward(raw): {env_reward:.4f}, EnvReward(adj): {current_env_reward:.4f} (w:{env_w:.2f}), "
            f"FmtReward: {format_reward_component:.4f} (w:{fmt_w:.2f}), "
            f"==> CombinedScore: {combined_score:.4f}"
        )
        return combined_score

    def _parse_tool_call(self, response: str) -> int:
        if not response: # Explicitly check for empty response string
            logger.warning("Attempted to parse an empty response string. Returning invalid action (-1).")
            return -1
        
        parsed_name, parsed_args, is_error = parse_tool_call(response, self.tools, ["tool_call"])
        if is_error:
            error_detail = parsed_name if isinstance(parsed_name, str) and parsed_name else "Parser indicated error, but no specific message was returned in the typical error slot."
            logger.warning(f"Failed to parse tool call. Full response: '{response}'. Error detail: {error_detail}")
            return -1
        
        action = parsed_args.get("action", "").lower()
        if action == "hit":
            return 1
        elif action == "stick":
            return 0
        else:
            logger.warning(f"Successfully parsed tool call, but action is invalid. Action: '{action}'. Full response: '{response}'. Parsed args: {parsed_args}")
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

    async def _estimate_value(self, 
                              episode_seed_for_sim: int, 
                              env_actions_to_replay: List[int], 
                              prompt_messages_for_llm_first_step: List[Dict],
                              K: int) -> float:
        """Estimate state value V(s) using K Monte Carlo rollouts from state s.

        Args:
            episode_seed_for_sim: The seed of the original episode to ensure deterministic env creation.
            env_actions_to_replay: List of environment actions (0 or 1) taken to reach the current state s.
            prompt_messages_for_llm_first_step: Message history up to state s, used to prompt LLM for the first action in simulation.
            K: Number of Monte Carlo samples.
        """
        all_rollout_returns = []
        max_sim_turns = self.config.max_turns or 5 # Use same max_turns as main trajectory for consistency

        for i in range(K):
            sim_env = None # Ensure sim_env is scoped correctly for finally block
            try:
                sim_env = gymnasium.make(self.config.env_name)
                current_sim_obs, _ = sim_env.reset(seed=episode_seed_for_sim)
                
                # Replay history to reach the current state s
                # This brings sim_env to the same state as the main env was in when state s was observed.
                for action_idx, prev_action in enumerate(env_actions_to_replay):
                    current_sim_obs, _, term_replay, trunc_replay, _ = sim_env.step(prev_action)
                    if term_replay or trunc_replay:
                        logger.warning(f"[_estimate_value Sample {i+1}/{K}] Simulation env terminated during action replay (action {action_idx+1}/{len(env_actions_to_replay)} of prev_actions). State s was already terminal. Value is 0.")
                        all_rollout_returns.append(0.0)
                        # Continue to next MC sample if state was already terminal
                        # This means V(s_terminal) = 0, which is correct.
                        # Need to break inner loop and go to next K iteration.
                        break 
                else: # Only execute if the for loop completed without break (i.e., replay didn't terminate)
                    # Now, sim_env is at state s. Start MC rollout from here.
                    rollout_reward_for_this_sample = 0.0
                    # current_mc_messages is the history for the LLM *within this MC rollout*.
                    # It starts with the history that led to state s.
                    current_mc_messages = prompt_messages_for_llm_first_step.copy()
                    term_mc, trunc_mc = False, False

                    for turn_mc in range(max_sim_turns):
                        agent_prompt_content = "<think>\n" if self.config.thinking_active else ""
                        # The messages sent to LLM for this turn of MC rollout
                        messages_for_llm_this_mc_turn = current_mc_messages.copy()
                        messages_for_llm_this_mc_turn.append({"role": "agent", "content": agent_prompt_content})
                        
                        responses = await self._sample_response(messages_for_llm_this_mc_turn, n=1)
                        if not responses:
                            # If API fails during MC rollout, this sample contributes 0 to the value estimate for state s.
                            logger.warning(f"[_estimate_value Sample {i+1}/{K}, Turn {turn_mc+1}] No API response. Ending this MC sample with accumulated reward {rollout_reward_for_this_sample}.")
                            break 
                        
                        llm_output_only = responses[0]
                        full_agent_response = agent_prompt_content + llm_output_only
                        
                        action_mc = self._parse_tool_call(full_agent_response)
                        
                        sim_obs_next, reward_mc_step, term_mc, trunc_mc, _ = sim_env.step(action_mc)
                        rollout_reward_for_this_sample += reward_mc_step # Accumulate reward for this MC rollout
                        
                        # Update message history for the *next* turn of this MC rollout
                        current_mc_messages.append({"role": "agent", "content": full_agent_response})
                        if sim_obs_next is not None:
                            current_mc_messages.append({"role": "environment", "content": self._format_observation(sim_obs_next)})
                        
                        if term_mc or trunc_mc:
                            break # End this MC sample's rollout
                    
                    all_rollout_returns.append(rollout_reward_for_this_sample)

            except Exception as e_mc_sample:
                logger.error(f"[_estimate_value Sample {i+1}/{K}] Unexpected error: {e_mc_sample}", exc_info=True)
                all_rollout_returns.append(0.0) # Assign 0 reward if a sample errors out
            finally:
                if sim_env is not None:
                    sim_env.close()
        
        return np.mean(all_rollout_returns) if all_rollout_returns else 0.0

    async def collect_trajectory(self, seed: int) -> List[BlackjackScoredDataGroup]:
        """Collect data for ONE trajectory, evaluating G alternatives per step using MC advantages."""
        G = self.config.group_size
        K = self.config.mc_samples
        max_turns = self.config.max_turns or 5
        
        # This list will store the BlackjackScoredDataGroup for each step of the single trajectory
        trajectory_data_for_trainer: List[BlackjackScoredDataGroup] = [] 

        logger.info(f"[Collect Trajectory Seed: {seed}] Starting trajectory. Group size G={G}, MC samples K={K}.")

        # --- Initialize the single trajectory episode --- 
        try:
            ep = self._get_or_create_episode(seed) 
        except Exception as e:
            logger.error(f"[Collect Trajectory Seed: {seed}] Failed to create/get episode: {e}", exc_info=True)
            return [] # Cannot proceed

        # --- Main loop for the single trajectory --- 
        for turn in range(max_turns):
            current_state_messages = ep.message_history.copy()
            logger.debug(f"[Collect Trajectory Seed: {seed} Turn: {turn+1}/{max_turns}] Current state history length: {len(current_state_messages)}")
            
            # --- Estimate V(s_t) --- 
            # Actions taken to reach this state are ep.actions
            try:
                value_t = await self._estimate_value(
                    episode_seed_for_sim=ep.seed,
                    env_actions_to_replay=ep.actions, # Actions leading to current state s_t
                    prompt_messages_for_llm_first_step=current_state_messages,
                    K=K
                )
                logger.debug(f"[Collect Trajectory Seed: {seed} Turn: {turn+1}] Estimated V(s_t) = {value_t:.4f}")
            except Exception as e_vt:
                 logger.error(f"[Collect Trajectory Seed: {seed} Turn: {turn+1}] Error estimating V(s_t): {e_vt}", exc_info=True)
                 break # Cannot proceed without V(s_t)

            # --- Sample G alternative responses --- 
            messages_for_llm = current_state_messages.copy()
            agent_prompt_content = "<think>\n" if self.config.thinking_active else ""
            messages_for_llm.append({"role": "agent", "content": agent_prompt_content})
            
            try:
                responses = await self._sample_response(messages_for_llm, n=G)
                if len(responses) != G:
                    logger.error(f"[Collect Trajectory Seed: {seed} Turn: {turn+1}] Expected {G} responses, got {len(responses)}. Aborting trajectory.")
                    break
            except Exception as e_sample:
                 logger.error(f"[Collect Trajectory Seed: {seed} Turn: {turn+1}] Error sampling responses: {e_sample}", exc_info=True)
                 break

            # --- Evaluate each of the G alternatives --- 
            alt_full_responses: List[str] = []
            alt_parsed_actions: List[int] = []
            alt_env_actions: List[int] = [] # Action actually stepped in sim (0 or 1)
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

                # Simulate this alternative action in a temporary environment
                sim_env = None
                raw_env_reward_i = 0.0
                term_i, trunc_i = False, False
                next_state_msgs_i = []
                try:
                    sim_env = gymnasium.make(self.config.env_name)
                    sim_obs, _ = sim_env.reset(seed=ep.seed)
                    # Replay history
                    for prev_action in ep.actions:
                        sim_obs, _, term_replay, trunc_replay, _ = sim_env.step(prev_action)
                        if term_replay or trunc_replay:
                            # This shouldn't happen if V(s_t) was estimated correctly, unless env is stochastic despite seed? Safety check.
                            logger.error(f"[Collect Trajectory Seed: {seed} Turn: {turn+1} Alt: {i}] Sim env terminated during replay. State mismatch?")
                            term_i, trunc_i = True, True # Treat as terminal
                            raw_env_reward_i = 0.0 # No reward if state was already terminal?
                            break 
                   
                    # If replay succeeded, step with the alternative action
                    if not (term_i or trunc_i): 
                        sim_obs_next, raw_env_reward_i, term_i, trunc_i, _ = sim_env.step(env_action)
                   
                    # Store results for this alternative
                    alt_raw_rewards.append(raw_env_reward_i)
                    alt_is_terminal.append(term_i or trunc_i)

                    # Calculate combined reward for this alternative
                    combined_reward_i = self._score_response(raw_env_reward_i, full_agent_response, parsed_action, ep.seed)
                    alt_combined_rewards.append(combined_reward_i)
                   
                    # Construct next state messages for V(s') estimation and tokenization
                    current_state_plus_response = current_state_messages + [{"role": "agent", "content": full_agent_response}]
                    if sim_obs_next is not None:
                        next_state_msgs_i = current_state_plus_response + [{"role": "environment", "content": self._format_observation(sim_obs_next)}]
                    else: # Handle cases where sim_obs_next might be None (e.g., error during step)
                        next_state_msgs_i = current_state_plus_response
                    alt_next_state_msgs.append(next_state_msgs_i)
                   
                    # Tokenize this alternative's full step message history
                    tokenized_i = tokenize_for_trainer(self.tokenizer, next_state_msgs_i)
                    alt_tokens.append(tokenized_i["tokens"])
                    alt_masks.append(tokenized_i["masks"])

                except Exception as e_sim:
                    logger.error(f"[Collect Trajectory Seed: {seed} Turn: {turn+1} Alt: {i}] Error simulating action {env_action}: {e_sim}", exc_info=True)
                    # Handle error by adding placeholder/default values
                    alt_raw_rewards.append(0.0)
                    alt_combined_rewards.append(-1.0) # Penalize simulation error
                    alt_next_state_msgs.append(current_state_messages + [{"role": "agent", "content": full_agent_response}]) # No env obs
                    alt_is_terminal.append(True) # Assume terminal on error
                    alt_tokens.append([])
                    alt_masks.append([])
                finally:
                    if sim_env: sim_env.close()

            # --- Estimate V(s') for all alternatives --- 
            # value_next_list = [0.0] * G # REMOVED - Unused variable
            # alt_value_next will store the estimated V(s') for each alternative
            alt_value_next: List[float] = [] 
            for i in range(G):
                if not alt_is_terminal[i]:
                    try:
                        # Actions to reach s'_i = ep.actions + [env_action_i]
                        actions_to_reach_s_prime = ep.actions + [alt_env_actions[i]]
                        value_next_i = await self._estimate_value(
                            episode_seed_for_sim=ep.seed,
                            env_actions_to_replay=actions_to_reach_s_prime,
                            prompt_messages_for_llm_first_step=alt_next_state_msgs[i],
                            K=K
                        )
                        alt_value_next.append(value_next_i)
                    except Exception as e_vn:
                        logger.error(f"[Collect Trajectory Seed: {seed} Turn: {turn+1} Alt: {i}] Error estimating V(s'): {e_vn}", exc_info=True)
                        alt_value_next.append(0.0) # Default to 0 on error
                else:
                    alt_value_next.append(0.0) # V(terminal) = 0

            # --- Calculate Advantage for all alternatives --- 
            for i in range(G):
                # Advantage = R_combined + gamma * V_raw(s') - V_raw(s) (gamma=1)
                advantage_i = alt_combined_rewards[i] + alt_value_next[i] - value_t
                alt_advantages.append(advantage_i)
                logger.debug(f"[Collect Trajectory Seed: {seed} Turn: {turn+1} Alt: {i}] CombinedR={alt_combined_rewards[i]:.2f}, V_t={value_t:.2f}, V_t+1={alt_value_next[i]:.2f} => Advantage={advantage_i:.2f}")

            # --- Package data for trainer --- 
            # Ensure all lists have G elements, padding if necessary due to errors
            # (Error handling above should add placeholders, but double-check lengths)
            if len(alt_tokens) != G or len(alt_masks) != G or len(alt_advantages) != G or len(alt_next_state_msgs) != G or len(alt_parsed_actions) != G:
                 logger.error(f"[Collect Trajectory Seed: {seed} Turn: {turn+1}] Mismatch in alternative list lengths after processing. Aborting trajectory.")
                 # TODO: Consider more robust padding or error handling
                 break

            trajectory_data_for_trainer.append(BlackjackScoredDataGroup(
                seed=ep.seed,
                tokens=alt_tokens,
                masks=alt_masks,
                scores=alt_advantages, 
                messages=alt_next_state_msgs,
                parsed_actions=alt_parsed_actions
            ))

            # --- Choose action to advance the main trajectory --- 
            best_advantage = -float('inf')
            best_advantage_idx = -1
            valid_indices_for_tiebreak = []

            for i in range(G):
                if alt_advantages[i] > best_advantage:
                    best_advantage = alt_advantages[i]
                    valid_indices_for_tiebreak = [i] # Reset tie-break list
                elif alt_advantages[i] == best_advantage:
                     valid_indices_for_tiebreak.append(i)
            
            if not valid_indices_for_tiebreak: # Should not happen if G >= 1
                logger.error(f"[Collect Trajectory Seed: {seed} Turn: {turn+1}] No best advantage index found. Defaulting to action 0.")
                best_advantage_idx = 0 # Fallback
            elif len(valid_indices_for_tiebreak) == 1:
                best_advantage_idx = valid_indices_for_tiebreak[0]
            else:
                # Tie-breaking: choose the one with the shortest response token length
                try:
                    best_advantage_idx = min(valid_indices_for_tiebreak, key=lambda idx: len(alt_tokens[idx]))
                    logger.debug(f"[Collect Trajectory Seed: {seed} Turn: {turn+1}] Advantage tie break: chose index {best_advantage_idx} based on token length.")
                except IndexError:
                     logger.error(f"[Collect Trajectory Seed: {seed} Turn: {turn+1}] IndexError during tie-breaking. Choosing first tied index {valid_indices_for_tiebreak[0]}.", exc_info=True)
                     best_advantage_idx = valid_indices_for_tiebreak[0]

            # Get the chosen action and response based on best advantage
            chosen_env_action = alt_env_actions[best_advantage_idx]
            chosen_full_response = alt_full_responses[best_advantage_idx]
            chosen_raw_env_reward = alt_raw_rewards[best_advantage_idx]
            chosen_is_terminal = alt_is_terminal[best_advantage_idx]
            
            logger.info(f"[Collect Trajectory Seed: {seed} Turn: {turn+1}] Chosen action to step env: {chosen_env_action} (from Alt {best_advantage_idx} with Adv {alt_advantages[best_advantage_idx]:.2f})")

            # --- Update main episode state --- 
            # Action was already added during simulation loop, need to ensure it matches chosen one?
            # No, ep.actions should store the sequence of *chosen* actions.
            # Let's re-add the chosen action and reward to ep state.
            # The history was advanced with *all* potential responses temporarily during eval loop,
            # we need to reset it and add only the chosen one + actual env observation.
            ep.message_history = current_state_messages # Reset to state before this turn's agent responses
            ep.message_history.append({"role": "agent", "content": chosen_full_response}) # Add chosen response
            
            # Re-step the *main* environment with the chosen action to get definitive obs for history
            # (This seems redundant if the simulation was correct, but safer for state consistency)
            try:
                main_obs, main_reward, main_term, main_trunc, main_info = ep.env.step(chosen_env_action)
                # Sanity check: main_reward should ideally match chosen_raw_env_reward if env is deterministic with seed
                if abs(main_reward - chosen_raw_env_reward) > 1e-6:
                     logger.warning(f"[Collect Trajectory Seed: {seed} Turn: {turn+1}] Mismatch between simulated reward ({chosen_raw_env_reward}) and main env step reward ({main_reward}) for chosen action {chosen_env_action}.")
                if (main_term or main_trunc) != chosen_is_terminal:
                     logger.warning(f"[Collect Trajectory Seed: {seed} Turn: {turn+1}] Mismatch between simulated terminal state ({chosen_is_terminal}) and main env step terminal state ({(main_term or main_trunc)}) for chosen action {chosen_env_action}.")
                
                # Use the results from the main env step
                term = main_term
                trunc = main_trunc
                obs = main_obs
                # Update action/reward lists with the chosen action's outcome
                ep.actions.append(chosen_env_action)
                ep.step_rewards.append(main_reward)
                ep.num_steps += 1 # Increment steps only for the chosen path

                if obs:
                    ep.message_history.append({"role": "environment", "content": self._format_observation(obs)})
            except Exception as e_main_step:
                 logger.error(f"[Collect Trajectory Seed: {seed} Turn: {turn+1}] Error stepping MAIN environment with chosen action {chosen_env_action}: {e_main_step}", exc_info=True)
                 term, trunc = True, True # Abort trajectory if main step fails

            if term or trunc:
                ep.total_reward = sum(ep.step_rewards)
                logger.info(f"[Collect Trajectory Seed: {seed}] Trajectory ended. Term={term}, Trunc={trunc}. Total raw env reward: {ep.total_reward}")
                break # End the turn loop for this trajectory
        
        # --- End of single trajectory collection --- 
        final_raw_reward = sum(ep.step_rewards) if ep.step_rewards else 0.0
        logger.info(f"[Collect Trajectory Seed: {seed}] Finished collecting trajectory. Steps collected: {len(trajectory_data_for_trainer)}, Final raw reward: {final_raw_reward:.2f}")

        # Clean up the main episode environment if it still exists
        if seed in self.episodes:
            try:
                self.episodes[seed].env.close()
            except Exception as e_close:
                logger.warning(f"[Collect Trajectory Seed: {seed}] Exception closing final env: {e_close}")
            del self.episodes[seed]

        return self._ensure_trajectory_token_limit(trajectory_data_for_trainer)

    async def score(self, rollout_group_data: List[BlackjackScoredDataGroup]) -> List[Optional[BlackjackScoredDataGroup]]:
        """Return rollout data with advantages as scores."""
        logger.info(f"[Score] Processing {len(rollout_group_data)} steps.")
        return rollout_group_data

    async def collect_trajectories(self, item: Tuple[int, int]) -> Tuple[List[BlackjackScoredDataGroup], List[Tuple[int, int]]]:
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
                logger.error(f"[Eval Seed: {seed}, Turn: {turn+1}] No response. Aborting.")
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
            
            # Update message history with the full agent response
            ep.message_history.append({"role": "agent", "content": full_agent_response})
            
            if obs:
                ep.message_history.append({"role": "environment", "content": self._format_observation(obs)})
                
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
        eval_tasks = [self.rollout_and_score_eval(random.randint(1000001, 2000000)) for _ in range(num_eval_episodes)]
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
        win_rate = sum(1 for m in valid_metrics if m["game_outcome"] == 1) / num_completed
        loss_rate = sum(1 for m in valid_metrics if m["game_outcome"] == -1) / num_completed
        draw_rate = sum(1 for m in valid_metrics if m["game_outcome"] == 0) / num_completed

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
            avg_reward = sum(m["total_reward"] for m in self.completed_episode_metrics_buffer) / num_episodes
            avg_steps = sum(m["num_steps"] for m in self.completed_episode_metrics_buffer) / num_episodes
            win_rate = sum(1 for m in self.completed_episode_metrics_buffer if m["game_outcome"] == 1) / num_episodes
            wandb_metrics[f"{self.wandb_prepend or 'blackjack'}_train/avg_episode_reward"] = avg_reward
            wandb_metrics[f"{self.wandb_prepend or 'blackjack'}_train/avg_episode_steps"] = avg_steps
            wandb_metrics[f"{self.wandb_prepend or 'blackjack'}_train/episode_win_rate"] = win_rate
            wandb_metrics[f"{self.wandb_prepend or 'blackjack'}_train/num_episodes"] = num_episodes
            self.completed_episode_metrics_buffer = []
        await super().wandb_log(wandb_metrics)

    @classmethod
    def config_init(cls, config_name_or_path: Optional[str] = None) -> Tuple[BlackjackEnvConfig, List[OpenaiConfig]]:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        default_config_filename = "blackjack_default.yaml"
        if config_name_or_path is None:
            cfg_path = os.path.join(current_dir, "configs", default_config_filename)
        elif os.path.isabs(config_name_or_path):
            cfg_path = config_name_or_path
        else:
            config_filename = config_name_or_path + (".yaml" if not config_name_or_path.endswith(".yaml") else "")
            cfg_path = os.path.join(current_dir, "configs", config_filename)

        try:
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    raw_yaml_data = yaml.safe_load(f) or {}
                logger.info(f"Loaded config from {cfg_path}")
            else:
                logger.warning(f"Config not found at {cfg_path}. Using default settings.")
                raw_yaml_data = {}

            env_conf_data = raw_yaml_data.copy()
            server_configs_list = env_conf_data.pop("server_configs", [])
            if "blackjack" in env_conf_data:
                env_conf_data.update(env_conf_data.pop("blackjack"))
            env_conf = BlackjackEnvConfig(**env_conf_data)

            server_confs = []
            for sc_data in server_configs_list:
                if not isinstance(sc_data, dict):
                    continue
                params = sc_data.copy()
                params["api_key"] = params.get("api_key", os.getenv("OPENAI_API_KEY", "x"))
                params["model_name"] = params.get("model_name", os.getenv("OPENAI_MODEL", "NousResearch/DeepHermes-3-Llama-3-8B-Preview"))
                params["base_url"] = params.get("base_url", os.getenv("OPENAI_API_BASE", "http://localhost:9004/v1"))
                server_confs.append(OpenaiConfig(**params))
            if not server_confs:
                server_confs = [OpenaiConfig(
                    model_name=os.getenv("OPENAI_MODEL", "NousResearch/DeepHermes-3-Llama-3-8B-Preview"),
                    base_url=os.getenv("OPENAI_API_BASE", "http://localhost:9004/v1"),
                    api_key=os.getenv("OPENAI_API_KEY", "x")
                )]
            return env_conf, server_confs
        except Exception as e:
            logger.error(f"Error loading config from {cfg_path}: {e}")
            return BlackjackEnvConfig(), [OpenaiConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9004/v1",
                api_key="x"
            )]

    def _truncate_thinking_for_history(self, response_text: str, max_chars: int) -> str:
        try:
            think_start = "<think>"
            think_end = "</think>"
            start_idx = response_text.find(think_start)
            end_idx = response_text.find(think_end)
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                before = response_text[:start_idx + len(think_start)]
                content = response_text[start_idx + len(think_start):end_idx].strip()
                after = response_text[end_idx:]
                if len(content) > max_chars:
                    truncated = "... " + content[-max_chars:].lstrip()
                    return f"{before}\n{truncated}\n{after}"
                return response_text
            return response_text
        except Exception as e:
            logger.error(f"Error truncating thinking: {e}")
            return response_text

    def _ensure_trajectory_token_limit(self, trajectory: List[BlackjackScoredDataGroup]) -> List[BlackjackScoredDataGroup]:
        filtered_trajectory = []
        for step_idx, step_data in enumerate(trajectory):
            if not step_data.get("messages") or not step_data.get("tokens") or not step_data.get("masks"):
                logger.warning(f"[Ensure Token Limit] Step {step_idx} missing data. Skipping.")
                continue
            max_tokens = max(len(t) for t in step_data["tokens"])
            if max_tokens <= self.config.max_trajectory_tokens:
                filtered_trajectory.append(step_data)
                continue
            logger.warning(f"[Ensure Token Limit] Step {step_idx} exceeds token limit ({max_tokens} > {self.config.max_trajectory_tokens}). Discarding.")
        return filtered_trajectory

    @classmethod
    def cli(cls):
        super().cli()

if __name__ == "__main__":
    BlackjackEnv.cli()
