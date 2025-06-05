# Negative reward applied when the first mismatched tool-call causes early termination.
WRONG_CALL_PENALTY = -0.2
# Hard cap on how many new tokens the model may generate in a single turn.
MAX_GEN_PER_TURN = 1024
# Hard cap on how many tool-call turns we will actually roll out
MAX_TOOL_CALL_TURNS = 2
# Whether to validate that all GPT messages have <think> blocks [useful when non-tool call gpt messages are inserted]
VALIDATE_THINK_BLOCKS = True
# Turn-level advantage coefficient (λ in MT-GRPO paper)
# Paper implementation uses 1.0, but we can experiment with different values
TURN_LEVEL_ADVANTAGE_LAMBDA = 0.5  # Configurable: try 0.1, 0.5, 1.0

"""
Multi-Turn Tool-Calling Environment with Turn-Level Advantages
==============================================================

Extends the multi-turn tool-calling environment to implement turn-level credit assignment
following the MT-GRPO approach from "Turn-Level Credit Assignment for Multi-Turn LLM Agents".

Key differences from the base multiturn environment:
    • Computes both turn-level rewards (R_T) and outcome-level rewards (R_O)  
    • Implements MT-GRPO advantage computation:
      - Turn 1: A_T_1 + λ * A_O
      - Turn 2: A_T_2 + λ * A_O  
      - Turn 3: A_O (outcome only)
    • Populates per-token advantages in ScoredDataGroup instead of just scores
    • Enables fine-grained credit assignment across turns

Dataset columns expected
------------------------
* `conversations` – list[dict] with keys `from` and `value`
"""

import json
import random
import re
import asyncio
import ast
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter


import wandb
from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
    Item,
    ScoredDataGroup,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

system_prompt = (
    "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the "
    "problem and deliberate with yourself via systematic reasoning processes to help come to a correct "
    "solution prior to answering. You should enclose your thoughts and internal monologue inside <think> "
    "</think> tags, and then provide your solution or response to the problem."
)


def _validate_reply_and_extract(txt: str):
    """
    Validates that the reply matches the allowed structure:
      - exactly one mandatory <think>…</think> block at the top
      - one or more <tool_call>…</tool_call> blocks
      - nothing else except whitespace/newlines
    Returns list of tool-call JSONs if valid, else None.
    """
    _allowed_re = re.compile(
        r"""^\s*
             <think>[\s\S]*?</think>\s*
             (?:
                 <tool_call>[\s\S]*?</tool_call>\s*
             )+
             \s*$""",
        re.IGNORECASE | re.VERBOSE,
    )
    if not isinstance(txt, str) or not _allowed_re.match(txt):
        return None
    # Extract tool_call JSONs
    matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", txt, re.DOTALL | re.IGNORECASE)
    jsons = []
    for m in matches:
        try:
            jsons.append(json.loads(m))
        except Exception:
            pass
    return jsons


def _json_objects_match(model_json, expected_json):
    """
    True when every key/value in expected_json appears exactly in model_json.
    Nested dicts handled recursively.
    """
    if not isinstance(model_json, dict) or not isinstance(expected_json, dict):
        return False
    for k, v in expected_json.items():
        if k not in model_json:
            return False
        if isinstance(v, dict):
            if not _json_objects_match(model_json[k], v):
                return False
        else:
            if model_json[k] != v:
                return False
    return True


class MultiTurnToolCallingTurnLevelAdvantageEnv(BaseEnv):

    name = "multiturn_tool_use_turnlevel_advantage"

    def __init__(
        self,
        config: BaseEnvConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = True,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        # Load dataset once and cache on this instance
        self.ds = load_dataset("interstellarninja/salesforce_hermes_thinking", split="train")

        self.percent_correct_buffer: List[float] = []
        self.raw_score_buffer: List[float] = []
        self.eval_metrics: List[Tuple[str, float]] = []
        self.rollouts_for_wandb: List = []

        # List of (messages_tuple, expected_calls_by_turn, inter_turns) triples
        self.train_items: List[
            Tuple[Tuple[frozenset, ...], List[List[str]], List[List[Dict[str, str]]]]
        ] = []
        self.test_items: List[
            Tuple[Tuple[frozenset, ...], List[List[str]], List[List[Dict[str, str]]]]
        ] = []
        self.iter = 0

    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
        env_cfg = BaseEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=16,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=2000,
            batch_size=1024,
            steps_per_eval=20,
            max_token_length=1024 * 64,
            inference_weight=1.0,
            wandb_name="multiturn_tool_use_turnlevel_advantage",
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.1,
        )
        server_cfgs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9004/v1",
                api_key="x",
                num_max_requests_at_once=32,
                num_requests_for_eval=256,
            )
        ]
        return env_cfg, server_cfgs

    async def setup(self):
        ds = self.ds.shuffle()

        counts = Counter()
        for row in ds:
            conv = row["conversations"]
            num_turns = 0
            for msg in conv:
                if msg["from"] in ("gpt", "assistant") and re.search(
                    r"<tool_call>", msg["value"], re.IGNORECASE
                ):
                    num_turns += 1
            counts[num_turns] += 1
        print("Tool-call distribution (tool_calls_per_convo → examples):")
        for k in sorted(counts):
            print(f"  {k:2d} → {counts[k]}")

        split = ds.train_test_split(0.02)
        split["train"] = split["train"].shuffle()
        split["test"] = split["test"].shuffle()
        self._prep_items(split["train"], is_train=True)
        self._prep_items(split["test"], is_train=False)

        random.shuffle(self.train_items)
        random.shuffle(self.test_items)

        if not self.train_items:
            raise ValueError("No training items prepared: check dataset formatting.")

    def _prep_items(self, dataset, *, is_train: bool):
        """
        For each conversation, collect all function_calls as a single episode.
        The context is all messages up to (but not including) the first function_call;
        the answer is the list of function_call JSONs (canonical string).
        Each turn can have multiple tool calls.

        We only keep those samples that contain = MAX_TOOL_CALL_TURNS separate messages with <tool_call>.
        """
        target = self.train_items if is_train else self.test_items
        before_len = len(target)

        for row in dataset:
            running_msgs: List[frozenset] = []

            conv = row["conversations"]
            if len(conv) < 3:
                continue
            if conv[0]["from"] != "system" or conv[1]["from"] != "human":
                continue
            
            # Check if conversation has ANY tool calling turns
            has_tool_calls = any(
                msg["from"] in ("gpt", "assistant") and "<tool_call>" in msg["value"].lower()
                for msg in conv
            )
            if not has_tool_calls:
                continue
            
            # Optional: Validate <think> blocks in gpt messages if enabled
            if VALIDATE_THINK_BLOCKS:
                gpt_messages = [msg for msg in conv if msg["from"] in ("gpt", "assistant")]
                if not all("<think>" in msg["value"].lower() for msg in gpt_messages):
                    continue

            if conv and conv[0]["from"] == "system":
                combined_system = system_prompt + "\n\n" + conv[0]["value"]
                running_msgs.append(
                    frozenset({"role": "system", "content": combined_system}.items())
                )
                conv = conv[1:]
            else:
                running_msgs.append(
                    frozenset({"role": "system", "content": system_prompt}.items())
                )

            inter_turns: List[List[Dict[str, str]]] = []
            expected_calls_by_turn: List[List[str]] = []
            buffer: List[Dict[str, str]] = []
            tool_call_turns = 0

            for msg in conv:
                m_from, m_val = msg["from"], msg["value"]

                is_tool_call = (
                    m_from in ("gpt", "assistant")
                    and "<tool_call>" in m_val.lower()
                )
                if is_tool_call:
                    tool_call_turns += 1
                    if expected_calls_by_turn:  # If we have previous turns, save the buffer
                        inter_turns.append(buffer)
                    buffer = []

                    matches = re.findall(
                        r"<tool_call>\s*(.*?)\s*</tool_call>",
                        m_val,
                        re.DOTALL | re.IGNORECASE,
                    )
                    if not matches:
                        continue
                    
                    # Group all tool calls from this message as one turn
                    turn_calls = []
                    for raw in matches:
                        try:
                            obj = json.loads(raw)
                            turn_calls.append(json.dumps(obj, separators=(",", ":")))
                        except Exception:
                            turn_calls.append(raw)
                    expected_calls_by_turn.append(turn_calls)
                    continue

                elif m_from in ("human", "gpt", "assistant"):
                    role = "user" if m_from == "human" else "assistant"
                    if not expected_calls_by_turn:
                        running_msgs.append(
                            frozenset({"role": role, "content": m_val}.items())
                        )
                    else:
                        buffer.append({"role": role, "content": m_val})

                elif m_from == "tool":
                    if not expected_calls_by_turn:
                        running_msgs.append(
                            frozenset({"role": "tool", "content": m_val}.items())
                        )
                    else:
                        buffer.append({"role": "tool", "content": m_val})

            if buffer and expected_calls_by_turn:
                inter_turns.append(buffer)

            while len(inter_turns) < max(0, len(expected_calls_by_turn) - 1):
                inter_turns.append([])

            if tool_call_turns == MAX_TOOL_CALL_TURNS:
                target.append((tuple(running_msgs), expected_calls_by_turn, inter_turns))

        print(f"[prep_items] {'train' if is_train else 'test'}: added {len(target)-before_len} items.")

    def _compute_turn_and_outcome_rewards(self, responses_by_turn: List[str], pred_calls_by_turn: List[List], expected_calls_by_turn: List[List[str]]) -> Tuple[List[float], float]:
        """
        Compute turn-level rewards (R_T) and outcome-level reward (R_O) using our custom approach.
        
        Turn-level rewards: Based on proper <think> blocks + <tool_call> blocks + tool call matches
        Outcome reward: 1.0 if ALL turns complete successfully, 0.0 otherwise
        
        Args:
            responses_by_turn: List of assistant responses for each turn  
            pred_calls_by_turn: List of predicted tool calls for each turn
            expected_calls_by_turn: List of expected tool calls for each turn
            
        Returns:
            Tuple of (turn_rewards, outcome_reward)
        """
        turn_rewards = []
        
        for turn_idx, (response, pred_turn, expected_turn) in enumerate(zip(responses_by_turn, pred_calls_by_turn, expected_calls_by_turn)):
            # Turn-level reward components
            turn_reward = 0.0
            
            # 1. Validate response structure (<think> + <tool_call> blocks)
            validation_result = _validate_reply_and_extract(response)
            has_valid_structure = validation_result is not None
            
            # 2. Check tool call matches
            tool_calls_match = False
            if has_valid_structure and pred_turn and expected_turn:
                # Parse expected calls for this turn
                exp_jsons = []
                for raw in expected_turn:
                    try:
                        exp_jsons.append(json.loads(raw))
                    except json.JSONDecodeError:
                        exp_jsons.append(ast.literal_eval(raw))
                
                # Handle early termination mismatch penalty
                actual_pred_turn = pred_turn
                if pred_turn and pred_turn[-1] == "__MISMATCH__":
                    actual_pred_turn = pred_turn[:-1]
                
                # Check if all tool calls match
                if len(actual_pred_turn) == len(exp_jsons):
                    correct = sum(
                        1 for p, e in zip(actual_pred_turn, exp_jsons) if _json_objects_match(p, e)
                    )
                    tool_calls_match = (correct == len(exp_jsons))
            
            # Compute turn reward
            if has_valid_structure:
                turn_reward += 0.5  # Reward for proper structure
            if tool_calls_match:
                turn_reward += 0.5  # Reward for correct tool calls
            
            # Apply mismatch penalty if needed
            if pred_turn and pred_turn[-1] == "__MISMATCH__":
                turn_reward += WRONG_CALL_PENALTY  # This is negative
            
            turn_rewards.append(turn_reward)
        
        # Outcome reward: 1.0 if all turns complete successfully (no mismatches)
        all_turns_successful = all(
            pred_turn and pred_turn[-1] != "__MISMATCH__" 
            for pred_turn in pred_calls_by_turn
        ) and all(r > 0.5 for r in turn_rewards)  # All turns have at least structure + tool match
        
        outcome_reward = 1.0 if all_turns_successful else 0.0
        
        return turn_rewards, outcome_reward

    def _compute_mt_grpo_advantages(self, turn_rewards_batch: List[List[float]], outcome_rewards_batch: List[float]) -> List[List[float]]:
        """
        Compute MT-GRPO advantages following the paper:
        - Turn 1: A_T_1 + λ * A_O
        - Turn 2: A_T_2 + λ * A_O  
        - Turn 3: A_O (outcome only)
        
        Args:
            turn_rewards_batch: List of turn rewards for each rollout [num_rollouts x num_turns]
            outcome_rewards_batch: List of outcome rewards for each rollout [num_rollouts]
            
        Returns:
            List of advantages for each rollout [num_rollouts x num_turns]
        """
        # Compute standardized turn advantages (A_T)
        turn_advantages_batch = []
        num_turns = len(turn_rewards_batch[0]) if turn_rewards_batch else 0
        
        for turn_idx in range(num_turns):
            turn_rewards_for_this_turn = [rewards[turn_idx] for rewards in turn_rewards_batch]
            mean_turn_reward = np.mean(turn_rewards_for_this_turn)
            std_turn_reward = np.std(turn_rewards_for_this_turn)
            if std_turn_reward == 0:
                std_turn_reward = 1.0  # Avoid division by zero
            
            turn_advantages = [(r - mean_turn_reward) / std_turn_reward for r in turn_rewards_for_this_turn]
            turn_advantages_batch.append(turn_advantages)
        
        # Compute standardized outcome advantages (A_O)
        mean_outcome_reward = np.mean(outcome_rewards_batch)
        std_outcome_reward = np.std(outcome_rewards_batch)
        if std_outcome_reward == 0:
            std_outcome_reward = 1.0  # Avoid division by zero
        
        outcome_advantages = [(r - mean_outcome_reward) / std_outcome_reward for r in outcome_rewards_batch]
        
        # Combine according to MT-GRPO formula
        mt_grpo_advantages = []
        for rollout_idx in range(len(turn_rewards_batch)):
            rollout_advantages = []
            for turn_idx in range(num_turns):
                if turn_idx < num_turns - 1:  # Not the last turn
                    # A_T_i + λ * A_O_i
                    advantage = turn_advantages_batch[turn_idx][rollout_idx] + TURN_LEVEL_ADVANTAGE_LAMBDA * outcome_advantages[rollout_idx]
                else:  # Last turn
                    # A_O_i only
                    advantage = outcome_advantages[rollout_idx]
                rollout_advantages.append(advantage)
            mt_grpo_advantages.append(rollout_advantages)
        
        return mt_grpo_advantages

    def _assign_advantages_to_tokens(self, contexts: List[List[Dict[str, str]]], advantages_by_turn: List[List[float]]) -> List[List[float]]:
        """
        Assign turn-level advantages to tokens in a turn-based manner.
        
        Our approach:
        - Each assistant message = one turn (tool call round)
        - Turn 1, 2, ..., N-1: A_T_i + λ * A_O (turn + outcome advantages)
        - Turn N (final): A_O only (outcome advantage)
        - Assign the turn's advantage to ALL tokens in that assistant message
        
        Args:
            contexts: List of conversation contexts for each rollout
            advantages_by_turn: List of advantages for each turn for each rollout
            
        Returns:
            List of per-token advantages for each rollout
        """
        per_token_advantages = []
        
        for rollout_idx, (context, turn_advantages) in enumerate(zip(contexts, advantages_by_turn)):
            # Tokenize the full conversation to get tokens and masks
            out = tokenize_for_trainer(
                tokenizer=self.tokenizer,
                chat=context,
                include_messages=self.config.include_messages,
            )
            tokens = out["tokens"]
            masks = out["masks"]
            
            # Initialize advantages for all tokens
            token_advantages = [0.0] * len(tokens)
            
            # Find assistant message boundaries and assign turn-specific advantages
            assistant_turn_idx = 0
            current_advantage = turn_advantages[0] if turn_advantages else 0.0
            
            # Track if we're currently in an assistant message
            in_assistant_msg = False
            assistant_start_idx = -1
            
            for i, (token_id, mask) in enumerate(zip(tokens, masks)):
                if mask != -100:  # Non-padding token
                    # Decode a small window to detect message boundaries
                    start_idx = max(0, i - 10)
                    end_idx = min(len(tokens), i + 10)
                    token_window = self.tokenizer.decode(tokens[start_idx:end_idx])
                    
                    # Detect start of new assistant message
                    if "assistant" in token_window.lower() and not in_assistant_msg:
                        in_assistant_msg = True
                        assistant_start_idx = i
                        # Use current turn advantage
                        if assistant_turn_idx < len(turn_advantages):
                            current_advantage = turn_advantages[assistant_turn_idx]
                    
                    # Detect end of assistant message (start of next role)
                    elif ("user" in token_window.lower() or "tool" in token_window.lower()) and in_assistant_msg:
                        # Apply advantage to all tokens in the assistant message we just finished
                        for j in range(assistant_start_idx, i):
                            if masks[j] != -100:
                                token_advantages[j] = current_advantage
                        
                        in_assistant_msg = False
                        assistant_turn_idx += 1
                    
                    # If we're in an assistant message, apply current advantage
                    elif in_assistant_msg:
                        token_advantages[i] = current_advantage
            
            # Handle case where conversation ends with assistant message
            if in_assistant_msg and assistant_start_idx != -1:
                for j in range(assistant_start_idx, len(tokens)):
                    if masks[j] != -100:
                        token_advantages[j] = current_advantage
            
            per_token_advantages.append(token_advantages)
        
        return per_token_advantages

    @staticmethod
    def _score_episode(pred_calls: list, exp_calls: list, lam: float = 0.5) -> float:
        """
        Legacy method for compatibility - now used primarily for evaluation.
        For training, we use turn-level rewards instead.
        """
        exp_jsons: List[dict] = []
        for raw in exp_calls:
            try:
                exp_jsons.append(json.loads(raw))
            except json.JSONDecodeError:
                exp_jsons.append(ast.literal_eval(raw))
        mismatch_penalty = 0.0
        if pred_calls and pred_calls[-1] == "__MISMATCH__":
            pred_calls = pred_calls[:-1]
            mismatch_penalty = WRONG_CALL_PENALTY
        correct = sum(
            1 for p, e in zip(pred_calls, exp_jsons) if _json_objects_match(p, e)
        )
        dense = correct / max(1, len(exp_jsons))
        bonus = 1.0 if correct == len(exp_jsons) else 0.0
        return dense + lam * bonus + mismatch_penalty

    async def rollout_and_score_eval(self, item) -> float:
        messages_tuple, expected_calls_by_turn, inter_turns = item
        base_ctx = [dict(m) for m in messages_tuple]
        ctx = list(base_ctx)
        preds = []
        
        # Iterate through turns instead of individual calls
        for turn_idx, expected_turn_calls in enumerate(expected_calls_by_turn):
            if turn_idx > 0 and turn_idx - 1 < len(inter_turns):
                ctx.extend(inter_turns[turn_idx - 1])
            prompt = self.tokenizer.apply_chat_template(ctx, add_generation_prompt=True, tokenize=False)
            max_toks = max(1, self.config.max_token_length - len(prompt))
            comp = await self.server.completion(
                prompt=prompt, n=1, max_tokens=self.config.max_token_length, temperature=0.0, split="eval"
            )
            reply = comp.choices[0].text
            ctx.append({"role": "assistant", "content": reply})
            tool_jsons = _validate_reply_and_extract(reply)
            if tool_jsons is None:
                break
            preds.extend(tool_jsons)
            # Check if we've processed enough turns
            if turn_idx >= len(expected_calls_by_turn) - 1:
                break
        
        # Flatten expected calls for scoring
        expected_calls_flat = [call for turn_calls in expected_calls_by_turn for call in turn_calls]
        score = self._score_episode(preds, expected_calls_flat)
        return score

    async def evaluate(self, *_, **__):
        subset = self.test_items[: min(128, len(self.test_items))]
        scores = await tqdm_asyncio.gather(*[self.rollout_and_score_eval(it) for it in subset])
        avg_reward = sum(scores) / len(scores)
        pct_exact = sum(1 for s in scores if s >= 1.0) / len(scores)
        self.eval_metrics.append(("eval/avg_reward", avg_reward))
        self.eval_metrics.append(("eval/percent_correct", pct_exact))

    async def get_next_item(self):
        """
        Return the next training item in a strictly sequential (non‐wrapping) order.
        Once we've gone through all items, reshuffle and start over.
        """
        if not self.train_items:
            raise ValueError("train_items is empty – dataset preparation failed.")

        if self.iter >= len(self.train_items):
            random.shuffle(self.train_items)
            self.iter = 0

        itm = self.train_items[self.iter]
        self.iter += 1
        return itm

    async def _build_turn_contexts(self, turn_idx: int, contexts: List[List[Dict[str, str]]], 
                                 inter_turns: List[List[Dict[str, str]]], active: List[bool]) -> Tuple[List[str], List[int]]:
        """Build prompts for the current turn from active rollout contexts."""
        # Add inter-turn context if not the first turn
        if turn_idx > 0 and turn_idx - 1 < len(inter_turns):
            filler = inter_turns[turn_idx - 1]
            for r in range(len(contexts)):
                if active[r]:
                    contexts[r].extend(filler)

        # Build prompts for active rollouts
        prompts, ridx_map = [], []
        for r in range(len(contexts)):
            if not active[r]:
                continue
            ptxt = self.tokenizer.apply_chat_template(
                contexts[r],
                add_generation_prompt=True,
                tokenize=False,
            )
            prompts.append(ptxt)
            ridx_map.append(r)
        
        return prompts, ridx_map

    async def _execute_turn_inference(self, turn_idx: int, prompts: List[str], ridx_map: List[int]) -> List[str]:
        """Execute inference for a turn using optimal batching strategy."""
        if turn_idx == 0:
            # Turn 1: Use n parameter for identical prompts
            return await self._batch_identical_prompts(prompts[0], len(ridx_map), turn_idx)
        else:
            # Later turns: Use parallel requests for heterogeneous prompts
            return await self._batch_heterogeneous_prompts(prompts, turn_idx)

    async def _batch_identical_prompts(self, prompt: str, count: int, turn_idx: int) -> List[str]:
        """Handle identical prompts efficiently using n parameter."""
        print(f"    \033[93m→ TURN {turn_idx+1} prompt full:\033[0m \033[92m{prompt}\033[0m")
        
        resp = await self.server.completion(
            prompt=prompt,
            n=count,
            max_tokens=self.config.max_token_length,
            temperature=0.8,
        )
        choices = [c.text for c in resp.choices]
        
        # Debug: print each rollout
        for i, raw in enumerate(choices):
            print(f"    \033[93m· turn {turn_idx+1} rollout raw [{i}]:\033[0m \033[94m{raw}\033[0m")
            if not raw.strip():
                print(f"      → (empty or error string returned for rollout {i})")
        print("    → All turn 1 rollouts printed; moving on.\n" + "-"*48)
        
        return choices

    async def _batch_heterogeneous_prompts(self, prompts: List[str], turn_idx: int) -> List[str]:
        """Handle heterogeneous prompts using parallel requests."""
        if turn_idx == 1:
            print("=== DEBUG: Now parallelizing Turn 2 prompts ===")
        print(f"    → Parallelizing {len(prompts)} prompts at turn {turn_idx+1}")
        
        # Print each prompt
        for idx_p, p_str in enumerate(prompts):
            print(f"    \033[93m→ TURN-{turn_idx+1} prompt[{idx_p}] full:\033[0m \033[92m{p_str}\033[0m")

        async def _call_single(prompt_str: str) -> str:
            try:
                comp = await self.server.completion(
                    prompt=prompt_str,
                    n=1,
                    max_tokens=self.config.max_token_length,
                    temperature=0.8,
                )
                return comp.choices[0].text
            except Exception as e:
                print(f"    → Turn {turn_idx+1} _call_single exception: {e}")
                return ""

        tasks = [_call_single(p) for p in prompts]
        results = await asyncio.gather(*tasks)

        # Debug: print results for all turns
        choices = []
        for i, rtext in enumerate(results):
            raw = rtext or ""
            print(f"    \033[93m· rollout {i} (Turn {turn_idx+1}) full reply:\033[0m \033[94m{raw}\033[0m\n" + "-"*48)
            if not raw:
                print(f"    → Rollout {i} returned empty or error string")
            choices.append(raw)
        
        return choices

    async def _process_turn_responses(self, turn_idx: int, choices: List[str], ridx_map: List[int],
                                    contexts: List[List[Dict[str, str]]], preds_by_turn: List[List[List]], 
                                    responses_by_turn: List[List[str]], active: List[bool], expected_calls_by_turn: List[List[str]]) -> None:
        """Process and validate responses for a single turn."""
        for txt, r in zip(choices, ridx_map):
            txt = txt or ""
            contexts[r].append({"role": "assistant", "content": txt})
            
            # Store response by turn for reward computation
            while len(responses_by_turn[r]) <= turn_idx:
                responses_by_turn[r].append("")
            responses_by_turn[r][turn_idx] = txt
            
            calls = _validate_reply_and_extract(txt)
            
            # Ensure preds_by_turn has enough turns
            while len(preds_by_turn[r]) <= turn_idx:
                preds_by_turn[r].append([])
            
            if calls is None:
                preds_by_turn[r][turn_idx].append("__MISMATCH__")
                active[r] = False
                continue

            # Get expected calls for this specific turn
            expected_turn_calls = expected_calls_by_turn[turn_idx]
            
            # Check if number of calls matches
            if len(calls) != len(expected_turn_calls):
                preds_by_turn[r][turn_idx].append("__MISMATCH__")
                active[r] = False
                continue
            
            mismatch = False
            for mdl, exp_raw in zip(calls, expected_turn_calls):
                try:
                    exp_obj = json.loads(exp_raw)
                except Exception:
                    exp_obj = ast.literal_eval(exp_raw)
                if not _json_objects_match(mdl, exp_obj):
                    mismatch = True
                    break
                    
            if mismatch:
                preds_by_turn[r][turn_idx].append("__MISMATCH__")
                active[r] = False
            else:
                preds_by_turn[r][turn_idx].extend(calls)

    async def collect_trajectories(
        self,
        item: Tuple[
            Tuple[frozenset, ...],
            List[List[str]],
            List[List[Dict[str, str]]],
        ],
    ) -> Tuple[Optional[ScoredDataGroup], List[Item]]:
        """
        Roll-out multi-turn tool-calling with turn-level advantage computation.
        """
        messages_tuple, expected_calls_by_turn, inter_turns = item
        base_ctx = [dict(m) for m in messages_tuple]

        num_rollouts = self.config.group_size
        contexts: List[List[Dict[str, str]]] = [list(base_ctx) for _ in range(num_rollouts)]
        # Track predictions by turn
        preds_by_turn: List[List[List]] = [[[] for _ in range(MAX_TOOL_CALL_TURNS)] for _ in range(num_rollouts)]
        # Track responses by turn for reward computation
        responses_by_turn: List[List[str]] = [[] for _ in range(num_rollouts)]
        active = [True] * num_rollouts

        max_turns = min(len(expected_calls_by_turn), MAX_TOOL_CALL_TURNS)

        for turn_idx in range(max_turns):
            print(f"[collect_trajectories] Beginning turn {turn_idx+1}/{max_turns} for this group")
            
            # Build contexts and prompts for this turn
            prompts, ridx_map = await self._build_turn_contexts(turn_idx, contexts, inter_turns, active)
            
            if not prompts:
                break

            max_prompt_len = max(len(p) for p in prompts)
            max_gen = min(
                MAX_GEN_PER_TURN,
                max(1, self.config.max_token_length - max_prompt_len),
            )

            # Execute inference for this turn
            choices = await self._execute_turn_inference(turn_idx, prompts, ridx_map)
            
            # Process and validate responses (now stores by turn including responses)
            await self._process_turn_responses(turn_idx, choices, ridx_map, contexts, preds_by_turn, responses_by_turn, active, expected_calls_by_turn)
            
            if not any(active):
                print("    → All roll-outs terminated; stopping further turns.")
                break

            survivors = sum(1 for a in active if a)
            print(f"    → DEBUG: finished turn {turn_idx+1}; {survivors}/{num_rollouts} rollouts still active")

        # Compute turn-level and outcome-level rewards for each rollout using our custom approach
        turn_rewards_batch = []
        outcome_rewards_batch = []
        
        for r in range(num_rollouts):
            turn_rewards, outcome_reward = self._compute_turn_and_outcome_rewards(
                responses_by_turn[r], preds_by_turn[r], expected_calls_by_turn
            )
            turn_rewards_batch.append(turn_rewards)
            outcome_rewards_batch.append(outcome_reward)

        # Compute MT-GRPO advantages
        mt_grpo_advantages = self._compute_mt_grpo_advantages(turn_rewards_batch, outcome_rewards_batch)
        
        # Assign advantages to tokens
        per_token_advantages = self._assign_advantages_to_tokens(contexts, mt_grpo_advantages)

        scored = ScoredDataGroup(tokens=[], masks=[], scores=[], advantages=[])
        
        for r in range(num_rollouts):
            # Use outcome reward as the overall score for compatibility
            score = outcome_rewards_batch[r]
            
            out = tokenize_for_trainer(
                tokenizer=self.tokenizer,
                chat=contexts[r],
                include_messages=self.config.include_messages,
            )
            scored["tokens"].append(out["tokens"])
            scored["masks"].append(out["masks"])
            scored["scores"].append(score if score > 0 else -1.0)
            # Key difference: populate advantages field with per-token advantages
            scored["advantages"].append(per_token_advantages[r])

        # Apply length penalty if needed
        if scored["scores"] and all(s > 0.99 for s in scored["scores"]):
            cutoff = self.config.max_token_length * 0.5
            for i, ln in enumerate([len(t) for t in scored["tokens"]]):
                if ln > cutoff:
                    frac = min((ln - cutoff) / (self.config.max_token_length - cutoff), 1.0)
                    scored["scores"][i] = max(0.0, scored["scores"][i] - frac)

        for s in scored["scores"]:
            self.raw_score_buffer.append(s)
            self.percent_correct_buffer.append(1.0 if s >= 1.0 else 0.0)

        if len(scored["tokens"]) < self.config.group_size or scored["scores"].count(
            scored["scores"][0]
        ) == len(scored["scores"]):
            return None, []

        await self.add_rollouts_for_wandb(scored, item)
        return scored, []

    async def create_rollout_table(self, wdict):
        if self.rollouts_for_wandb:
            table = wandb.Table(columns=["generation", "score", "expected_tool_call"])
            for grp in self.rollouts_for_wandb:
                for g, sc, exp in grp:
                    exp_str = json.dumps(exp, separators=(",", ":"))
                    table.add_data(g, sc, exp_str)
            wdict["train/rollouts"] = table
        self.rollouts_for_wandb = []
        return wdict

    async def add_rollouts_for_wandb(
        self, scored: ScoredDataGroup, item: Item, *, num_keep: int = 1
    ):
        num_keep = min(num_keep, len(scored["tokens"]))
        # Flatten expected calls for wandb logging
        expected_calls_flat = [call for turn_calls in item[1] for call in turn_calls]
        self.rollouts_for_wandb.append(
            [
                (
                    self.tokenizer.decode(scored["tokens"][i]),
                    scored["scores"][i],
                    expected_calls_flat,
                )
                for i in range(num_keep)
            ]
        )
        if len(self.rollouts_for_wandb) > self.config.num_rollouts_to_keep:
            self.rollouts_for_wandb.pop(0)

    async def wandb_log(self, metrics: Optional[Dict] = None):
        metrics = metrics or {}
        metrics = await self.create_rollout_table(metrics)
        if self.raw_score_buffer:
            avg_reward = sum(self.raw_score_buffer) / len(self.raw_score_buffer)
            pct_correct = (
                sum(self.percent_correct_buffer) / len(self.percent_correct_buffer)
            )
            metrics["train/avg_reward"] = avg_reward
            metrics["train/percent_correct"] = pct_correct
            self.raw_score_buffer.clear()
            self.percent_correct_buffer.clear()
        for k, v in self.eval_metrics:
            metrics[k] = v
        await super().wandb_log(metrics)


if __name__ == "__main__":
    MultiTurnToolCallingTurnLevelAdvantageEnv.cli()