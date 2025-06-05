# Negative reward applied when the first mismatched tool-call causes early termination.
WRONG_CALL_PENALTY = -0.2
# Hard cap on how many new tokens the model may generate in a single turn.
MAX_GEN_PER_TURN = 1024
# Hard cap on how many tool-call turns we will actually roll out
MAX_TOOL_CALL_TURNS = 2
# Whether to validate that all GPT messages have <think> blocks [useful when non-tool call gpt messages are inserted]
VALIDATE_THINK_BLOCKS = True

"""
Multi-Turn Tool-Calling Environment
==================================

Extends the single-turn tool-calling environment to conversations that
contain **multiple** function-call / observation pairs.  Each training
item corresponds to *one* episode consisting of all tool-calls in the conversation:

    • We locate every message where `msg["from"] == "gpt" and has <tool_call>`.
    • For each such conversation, we create an item whose **context** is all
      conversation messages upto the next function call turn.
    • Rewards are *episodic*: dense + sparse reward for matching all tool-calls.

Dataset columns expected
------------------------
* `conversations` – list[dict] with keys `from` and `value`
"""

import json
import random
import re
import asyncio
import ast
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


class MultiTurnToolCallingEnv(BaseEnv):

    name = "multiturn_tool_use"

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
            wandb_name="multiturn_tool_use",
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

    @staticmethod
    def _score_episode(pred_calls: list, exp_calls: list, lam: float = 0.5) -> float:
        """
        pred_calls : list of JSON objects (already parsed)
        exp_calls  : list of *canonical* JSON strings from dataset

        Returns dense + sparse reward:
            r = (#correct / N) + lam * 1{all correct} + penalty (if mismatch)
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
                                    contexts: List[List[Dict[str, str]]], preds: List[List], 
                                    active: List[bool], expected_calls_by_turn: List[List[str]]) -> None:
        """Process and validate responses for a single turn."""
        for txt, r in zip(choices, ridx_map):
            txt = txt or ""
            contexts[r].append({"role": "assistant", "content": txt})
            calls = _validate_reply_and_extract(txt)
            
            if calls is None:
                preds[r].append("__MISMATCH__")
                active[r] = False
                continue

            # Get expected calls for this specific turn
            expected_turn_calls = expected_calls_by_turn[turn_idx]
            
            # Check if number of calls matches
            if len(calls) != len(expected_turn_calls):
                preds[r].append("__MISMATCH__")
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
                preds[r].append("__MISMATCH__")
                active[r] = False
            else:
                preds[r].extend(calls)

    async def collect_trajectories(
        self,
        item: Tuple[
            Tuple[frozenset, ...],
            List[List[str]],
            List[List[Dict[str, str]]],
        ],
    ) -> Tuple[Optional[ScoredDataGroup], List[Item]]:
        """
        Roll-out one *tool-call turn* for every rollout in the group.

        ─ Round 0 ────────────────────────────────────────────────────────────
        All roll-outs share an *identical* prompt → send a **single** request
        with `n = group_size`.

        ─ Later rounds ───────────────────────────────────────────────────────
        Prompts are heterogeneous, so we always issue `group_size` independent
        requests in parallel via ``asyncio.gather``.
        """
        messages_tuple, expected_calls_by_turn, inter_turns = item
        base_ctx = [dict(m) for m in messages_tuple]

        num_rollouts = self.config.group_size
        contexts: List[List[Dict[str, str]]] = [list(base_ctx) for _ in range(num_rollouts)]
        preds: List[List] = [[] for _ in range(num_rollouts)]
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
            
            # Process and validate responses
            await self._process_turn_responses(turn_idx, choices, ridx_map, contexts, preds, active, expected_calls_by_turn)
            
            if not any(active):
                print("    → All roll-outs terminated; stopping further turns.")
                break

            survivors = sum(1 for a in active if a)
            print(f"    → DEBUG: finished turn {turn_idx+1}; {survivors}/{num_rollouts} rollouts still active")

        scored = ScoredDataGroup(tokens=[], masks=[], scores=[])
        # Flatten expected calls for scoring (since _score_episode expects flat list)
        expected_calls_flat = [call for turn_calls in expected_calls_by_turn for call in turn_calls]
        for r in range(num_rollouts):
            reward = self._score_episode(preds[r], expected_calls_flat)
            out = tokenize_for_trainer(
                tokenizer=self.tokenizer,
                chat=contexts[r],
                include_messages=self.config.include_messages,
            )
            scored["tokens"].append(out["tokens"])
            scored["masks"].append(out["masks"])
            scored["scores"].append(reward if reward > 0 else -1.0)

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
    MultiTurnToolCallingEnv.cli()