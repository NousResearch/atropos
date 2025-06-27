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

import ast
import asyncio
import json
import random
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

import wandb
from datasets import load_dataset
from pydantic import Field
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

# ------------------------------------------------------------------
# Filter: skip tasks that were already processed in a finished SFT dataset
# ------------------------------------------------------------------
COMPLETED_DATASET_ID = "interstellarninja/toolace_sequential_tool_use_reasoning"
try:
    _done_ds = load_dataset(COMPLETED_DATASET_ID, split="train")
    COMPLETED_TASKS: set[str] = set(_done_ds["task"])
    print(
        f"[filter] Loaded {len(COMPLETED_TASKS):,} completed tasks from {COMPLETED_DATASET_ID}"
    )
except Exception as _exc:
    COMPLETED_TASKS = set()
    print(
        f"[filter] Could not load completed-task dataset: {_exc}. No skipping will occur."
    )

# Easy-to-change constants for experimentation - modify these for quick testing
WRONG_CALL_PENALTY = -0.2
MAX_GEN_PER_TURN = 512
MAX_TOOL_CALL_TURNS_CAP = 3  # Default: keep up to 3 tool-calling turns for multiturn
# Narration turns do NOT count against this cap.
VALIDATE_THINK_BLOCKS = False

GENERATE_ALL_GPT_TURNS = True
# When True, prepend the latest successful assistant tool‑calling example
# (<think>… <tool_call>…) to the system prompt as a live few‑shot.
ADD_DYNAMIC_FEW_SHOT = True

# Supported benchmark categories, aligned with BFCL‑V3
#   "single"     → single‑turn  (one assistant tool‑calling turn)
#   "multistep"  → multi‑step   (≥2 assistant tool‑calling turns, no extra user turns)
#   "multiturn"  → multi‑turn   (≥1 extra user turn after the first tool call)
SCENARIO_CATEGORY = "multistep"  # set to "single" | "multistep" | "multiturn"


class MultiTurnEnvConfig(BaseEnvConfig):
    """Configuration for Multi-Turn Tool Calling Environment."""

    max_tool_call_turns_cap: Optional[int] = Field(
        default=2,
        description="Upper cap on assistant TOOL‑CALLING turns per episode (None → no cap)",
    )
    validate_think_blocks: bool = Field(
        default=True,
        description="Whether to validate that all GPT messages have <think> blocks [useful when non-tool call gpt messages are inserted]",
    )
    generate_all_gpt_turns: bool = Field(
        default=False,
        description=(
            "If True, the environment will emit a GPT turn *after each* <tool_response>. "
            "That reply **must begin** with one <think> … </think> block. "
            "If the dataset expects tool‑calls for that turn, the reply must also contain "
            "the same number of <tool_call> … </tool_call> blocks; otherwise it must contain "
            "no <tool_call> blocks."
        ),
    )
    max_gen_per_turn: int = Field(
        default=1024,
        description="Hard cap on how many new tokens the model may generate in a single turn",
    )
    wrong_call_penalty: float = Field(
        default=-0.2,
        description="Negative reward applied when the first mismatched tool-call causes early termination",
    )
    skip_completed: bool = Field(
        default=True,
        description="Skip any conversation whose first user prompt appears in COMPLETED_TASKS.",
    )
    scenario_category: str = Field(
        default="multiturn",
        description='BFCL‑style scenario type: "single", "multistep", or "multiturn".',
    )
    add_dynamic_few_shot: bool = Field(
        default=True,
        description="Insert most‑recent harvested example into system prompt.",
    )


system_prompt = (
    "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the "
    "problem and deliberate with yourself via systematic reasoning processes to help come to a correct "
    "solution prior to answering. You should enclose your thoughts and internal monologue inside <think> "
    "</think> tags, and then provide your solution or response to the problem."
)

few_shot_example = (
    "Example Reasoning format:\n"
    "<think>\n"
    "Okay, the user asked for a calculation, I need to use python_interpreter tool to compute it.\n"
    "</think>\n"
    "<tool_call>\n"
    '{"name": "python_interpreter", "arguments": {"code": "print(2+2)"}}\n'
    "</tool_call>\n"
)

# Helper line that nudges the model to call tools one‑at‑a‑time.
SEQ_TOOL_HELPER = (
    "You are in sequential tool calling mode. "
    "Call exactly **one** tool, wait for its <tool_response>, "
    "then decide whether to call another. "
    "Never bundle multiple <tool_call> blocks in the same message. "
    "When you get the <tool_response and if you believe the task is complete, "
    "return your reasoning in <think> blocks followed by plain text summary"
)


# ------------------------------------------------------------------
# Helper: normalize assistant tool_call blocks to canonical JSON
# ------------------------------------------------------------------


def _normalize_tool_call_json(txt: str) -> str:
    """
    Normalise assistant replies so that:
      • the original <think> … </think> block is preserved
      • every <tool_call> … </tool_call> block is converted to
        canonical JSON (double‑quoted, valid JSON) even if the
        model used Python literal formatting.

    If normalisation fails we return the original text unchanged.
    """
    # Find the leading <think> … </think>
    m = re.match(r"^\s*(<think>[\s\S]*?</think>)\s*", txt, flags=re.IGNORECASE)
    if not m:
        return txt
    think_part = m.group(1)

    def _convert(match: re.Match) -> str:
        raw = match.group(1).strip()
        # Try literal‑eval first
        try:
            obj = ast.literal_eval(raw)
            return f"<tool_call>{json.dumps(obj, separators=(',', ':'))}</tool_call>"
        except Exception:
            pass
        # Fallback: crude single‑quote → double‑quote substitution
        try:
            json_like = re.sub(r"'([^']*)':", r'"\1":', raw)
            json_like = re.sub(r":\s*'([^']*)'", r':"\1"', json_like)
            json.loads(json_like)  # raises if still invalid
            return f"<tool_call>{json_like}</tool_call>"
        except Exception:
            return match.group(0)  # give up – leave unchanged

    tail = txt[len(m.group(0)) :]
    tail = re.sub(
        r"<tool_call>\s*([\s\S]*?)\s*</tool_call>",
        _convert,
        tail,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # --- beautify: ensure newline separation between blocks ---
    out = think_part + tail
    # ensure each <tool_call> and </tool_call> tag is on its own line
    out = re.sub(r"\s*<tool_call>\s*", "\n<tool_call>\n", out)
    out = re.sub(r"\s*</tool_call>\s*", "\n</tool_call>\n", out)
    return out


# ------------------------------------------------------------------
# Helper: validate a GPT reply that *may* include tool calls
# ------------------------------------------------------------------


def _validate_think_only(txt: str) -> bool:
    """
    A narration / summary turn must:
    • start with exactly one <think> … </think> block
    • contain **no** <tool_call> tags anywhere
    • contain no additional <think> blocks
    Anything after the </think> (user‑visible answer) is allowed.
    """
    txt = _normalize_tool_call_json(txt)
    if not isinstance(txt, str):
        return False

    # Must begin with exactly one think block and no more think blocks after
    think_blocks = re.findall(r"<think>[\s\S]*?</think>", txt, flags=re.IGNORECASE)
    if len(think_blocks) != 1:
        return False
        
    # Must be at the start
    if not re.match(r"^\s*<think>", txt, flags=re.IGNORECASE):
        return False

    # Must not contain any <tool_call>
    if re.search(r"<tool_call\s*>", txt, flags=re.IGNORECASE):
        return False

    return True

def _validate_think_plus_calls(txt: str):
    """
    Validate a GPT reply that should contain exactly one <think> … </think> followed by
    one or more <tool_call> … </tool_call> blocks.
    """
    txt = _normalize_tool_call_json(txt)
    
    # Check for exactly one think block
    think_blocks = re.findall(r"<think>[\s\S]*?</think>", txt, flags=re.IGNORECASE)
    if len(think_blocks) != 1:
        return None
        
    # Must start with the think block
    if not re.match(r"^\s*<think>", txt, flags=re.IGNORECASE):
        return None

    # Must be followed by at least one tool call
    tool_calls = re.findall(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>", txt, flags=re.IGNORECASE)
    if not tool_calls:
        return None

    # Parse tool calls
    tool_jsons = []
    for raw in tool_calls:
        try:
            tool_jsons.append(json.loads(raw))
        except Exception:
            return None
    return tool_jsons


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


# ------------------------------------------------------------------
# Helper: check that tool calls are strictly sequential (no interleaved user or assistant narration)
# ------------------------------------------------------------------
def _check_sequential_tools(conv: List[Dict[str, str]]) -> bool:
    """
    Return True when every assistant tool‑calling turn is followed only by the
    corresponding <tool_response> messages from the system before the next
    assistant <tool_call>. Allow concluding narration after all tool calls are done.
    """
    tool_indices = [
        i
        for i, m in enumerate(conv)
        if m["from"] in ("gpt", "assistant") and "<tool_call>" in m["value"].lower()
    ]

    # No tool calls at all
    if not tool_indices:
        return False

    # Check sequences between tool calls
    for i in range(len(tool_indices) - 1):
        start, end = tool_indices[i], tool_indices[i + 1]
        # Messages strictly between two tool‑calling turns
        in_between = conv[start + 1 : end]
        # Only <tool_response> allowed between tool calls
        if any(m["from"] != "tool" for m in in_between):
            return False

    # For the last tool call, only check up to the next tool response
    # (allow narration after that)
    last_tool_idx = tool_indices[-1]
    next_responses = [
        i
        for i, m in enumerate(conv[last_tool_idx + 1 :], start=last_tool_idx + 1)
        if m["from"] == "tool"
    ]
    if not next_responses:  # No tool response after last tool call
        return False

    # Check sequence after last tool call up to its response
    last_response_idx = next_responses[0]
    in_between = conv[last_tool_idx + 1 : last_response_idx + 1]
    if any(m["from"] != "tool" for m in in_between):
        return False

    return True


class MultiTurnToolCallingEnv(BaseEnv):

    name = "multiturn_tool_use"

    def __init__(
        self,
        config: MultiTurnEnvConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = True,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        # Load dataset once and cache on this instance
        # self.ds = load_dataset("interstellarninja/salesforce_hermes_thinking", split="train")
        self.ds = load_dataset(
            "interstellarninja/toolace_hermes_sequential_tool_use", split="train"
        )

        self.percent_correct_buffer: List[float] = []
        self.raw_score_buffer: List[float] = []
        self.eval_metrics: List[Tuple[str, float]] = []
        self.rollouts_for_wandb: List = []

        # Holds latest good assistant tool‑calling example (<think> + <tool_call>)
        self.dynamic_example: Optional[str] = None

        # List of (messages_tuple, expected_calls_by_turn, inter_turns) triples
        self.train_items: List[
            Tuple[Tuple[frozenset, ...], List[List[str]], List[List[Dict[str, str]]]]
        ] = []
        self.test_items: List[
            Tuple[Tuple[frozenset, ...], List[List[str]], List[List[Dict[str, str]]]]
        ] = []
        self.iter = 0

    @classmethod
    def config_init(cls) -> Tuple[MultiTurnEnvConfig, List[APIServerConfig]]:
        env_cfg = MultiTurnEnvConfig(
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
            # Override config defaults with experimental constants
            wrong_call_penalty=WRONG_CALL_PENALTY,
            max_gen_per_turn=MAX_GEN_PER_TURN,
            max_tool_call_turns_cap=MAX_TOOL_CALL_TURNS_CAP,
            validate_think_blocks=VALIDATE_THINK_BLOCKS,
            generate_all_gpt_turns=GENERATE_ALL_GPT_TURNS,
            skip_completed=True,
            scenario_category=SCENARIO_CATEGORY,
            add_dynamic_few_shot=ADD_DYNAMIC_FEW_SHOT,
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
        sequential_counts = Counter()
        for row in ds:
            conv = row["conversations"]
            num_turns = 0
            for msg in conv:
                if msg["from"] in ("gpt", "assistant") and re.search(
                    r"<tool_call>", msg["value"], re.IGNORECASE
                ):
                    num_turns += 1
            counts[num_turns] += 1
            # Count strictly sequential conversations as well
            if num_turns > 0 and _check_sequential_tools(conv):
                sequential_counts[num_turns] += 1
        print("Tool-call distribution (tool_calls_per_convo → examples):")
        for k in sorted(counts):
            print(f"  {k:2d} → {counts[k]} total, {sequential_counts[k]} sequential")

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
        Process dataset items based on scenario category:
        - "single": exactly one tool-calling turn
        - "multistep": ≥2 sequential tool-calling turns, no human interruption
        - "multiturn": ≥1 tool-calling turn with later human interaction
        """
        target = self.train_items if is_train else self.test_items
        before_len = len(target)

        for row in dataset:
            conv = row["conversations"]
            # Basic validation for all scenarios
            if (
                len(conv) < 3
                or conv[0]["from"] != "system"
                or conv[1]["from"] != "human"
            ):
                continue

            # Skip if task already completed
            if self.config.skip_completed and COMPLETED_TASKS:
                first_user_msg = conv[1]["value"].strip()
                if first_user_msg in COMPLETED_TASKS:
                    continue

            # Find all tool-calling turns
            tool_indices = [
                i
                for i, m in enumerate(conv)
                if m["from"] in ("gpt", "assistant")
                and "<tool_call>" in m["value"].lower()
            ]

            if not tool_indices:  # No tool calls at all
                continue

            # Check for human messages after first tool call
            human_after_first_tool = any(
                i > tool_indices[0] and m["from"] == "human" for i, m in enumerate(conv)
            )

            # ─── SCENARIO-SPECIFIC VALIDATION ───
            valid_scenario = False

            if self.config.scenario_category == "single":
                valid_scenario = len(tool_indices) == 1

            elif self.config.scenario_category == "multistep":
                # Must have ≥2 tool calls, first assistant must be tool call,
                # and must follow sequential pattern
                if len(tool_indices) >= 2:
                    first_assistant_idx = next(
                        (
                            i
                            for i, m in enumerate(conv[2:], start=2)
                            if m["from"] in ("gpt", "assistant")
                        ),
                        None,
                    )
                    if (
                        first_assistant_idx
                        == tool_indices[0]  # First assistant is tool call
                        and not human_after_first_tool  # No human interruption
                        and _check_sequential_tools(conv)
                    ):  # Follows sequential pattern
                        valid_scenario = True

            elif self.config.scenario_category == "multiturn":
                # At least one tool call and a later human turn
                valid_scenario = len(tool_indices) >= 1 and human_after_first_tool

            else:
                raise ValueError(
                    f"Unknown scenario_category={self.config.scenario_category}"
                )

            if not valid_scenario:
                continue

            # ─── PREPARE RUNNING MESSAGES ───
            running_msgs = []
            if self.config.add_dynamic_few_shot and self.dynamic_example:
                example_block = "Example Reasoning format:\n" + self.dynamic_example
            else:
                example_block = few_shot_example

            combined_system = (
                system_prompt + "\n\n" + conv[0]["value"] + "\n\n" + example_block
            )
            running_msgs.append(
                frozenset({"role": "system", "content": combined_system}.items())
            )

            # Add first human message, with helper for multistep
            human_content = conv[1]["value"]
            if self.config.scenario_category == "multistep":
                human_content = f"{human_content}\n\n{SEQ_TOOL_HELPER}"
            running_msgs.append(
                frozenset({"role": "user", "content": human_content}.items())
            )

            # ─── PREPARE EXPECTED CALLS AND INTER-TURNS ───
            expected_calls_by_turn = []
            inter_turns = []

            # Process each tool call turn
            for i, tool_idx in enumerate(tool_indices):
                # Extract tool calls for this turn
                tool_call_msg = conv[tool_idx]["value"]
                matches = re.findall(
                    r"<tool_call>\s*(.*?)\s*</tool_call>",
                    tool_call_msg,
                    re.DOTALL | re.IGNORECASE,
                )
                if not matches:
                    continue

                # Add to expected calls
                turn_calls = []
                for raw in matches:
                    try:
                        obj = json.loads(raw)
                        turn_calls.append(json.dumps(obj, separators=(",", ":")))
                    except Exception:
                        turn_calls.append(raw)
                expected_calls_by_turn.append(turn_calls)

                # Build inter_turns context
                if i < len(tool_indices) - 1:
                    # For multistep: exactly one tool response
                    if self.config.scenario_category == "multistep":
                        tool_response = conv[tool_idx + 1]
                        inter_turn = [
                            {
                                "role": "tool",
                                "content": tool_response["value"],
                            }  # Only include tool response
                        ]
                    # For single/multiturn: collect all messages until next tool call
                    else:
                        next_tool_idx = tool_indices[i + 1]
                        inter_turn = [
                            {
                                "role": m["from"].replace("gpt", "assistant"),
                                "content": m["value"],
                            }
                            for m in conv[tool_idx:next_tool_idx]
                        ]
                    inter_turns.append(inter_turn)

            # Handle final GPT message if it exists and generate_all_gpt_turns is enabled
            if self.config.generate_all_gpt_turns:
                last_tool_response_idx = tool_indices[-1] + 1
                has_final_narration = (
                    last_tool_response_idx + 1 < len(conv)
                    and conv[last_tool_response_idx + 1]["from"] in ("gpt", "assistant")
                    and "<tool_call>" not in conv[last_tool_response_idx + 1]["value"]
                )
                if has_final_narration:
                    expected_calls_by_turn.append(
                        []
                    )  # Empty expected calls for summary
                    final_inter_turn = [
                        {
                            "role": "tool",
                            "content": conv[last_tool_response_idx]["value"],
                        }  # Only include tool response
                    ]
                    inter_turns.append(final_inter_turn)

            # Apply turn cap if configured
            if (
                self.config.max_tool_call_turns_cap is not None
                and len(expected_calls_by_turn) > self.config.max_tool_call_turns_cap
            ):
                cut = self.config.max_tool_call_turns_cap
                expected_calls_by_turn = expected_calls_by_turn[:cut]
                inter_turns = inter_turns[: cut - 1]  # N turns → N-1 gaps

            target.append((tuple(running_msgs), expected_calls_by_turn, inter_turns))

        print(
            f"[prep_items] {'train' if is_train else 'test'}: added {len(target)-before_len} items."
        )

    @staticmethod
    def _score_episode(
        pred_calls: list,
        exp_calls: list,
        lam: float = 0.5,
        wrong_call_penalty: float = -0.2,
    ) -> float:
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
            mismatch_penalty = wrong_call_penalty
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
            prompt = self.tokenizer.apply_chat_template(
                ctx, add_generation_prompt=True, tokenize=False
            )
            max_toks = max(1, self.config.max_token_length - len(prompt))
            max_new = min(self.config.max_gen_per_turn, max_toks)
            comp = await self.server.completion(
                prompt=prompt, n=1, max_tokens=max_new, temperature=0.0, split="eval"
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
        expected_calls_flat = [
            call for turn_calls in expected_calls_by_turn for call in turn_calls
        ]
        score = self._score_episode(
            preds,
            expected_calls_flat,
            wrong_call_penalty=self.config.wrong_call_penalty,
        )
        return score

    async def evaluate(self, *_, **__):
        subset = self.test_items[: min(128, len(self.test_items))]
        scores = await tqdm_asyncio.gather(
            *[self.rollout_and_score_eval(it) for it in subset]
        )
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

    async def _build_turn_contexts(
        self,
        turn_idx: int,
        contexts: List[List[Dict[str, str]]],
        inter_turns: List[List[Dict[str, str]]],
        active: List[bool],
    ) -> Tuple[List[str], List[int]]:
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

    async def _execute_turn_inference(
        self, turn_idx: int, prompts: List[str], ridx_map: List[int]
    ) -> List[str]:
        """Execute inference for a turn using optimal batching strategy."""
        if turn_idx == 0:
            # Turn 1: Use n parameter for identical prompts
            return await self._batch_identical_prompts(
                prompts[0], len(ridx_map), turn_idx
            )
        else:
            # Later turns: Use parallel requests for heterogeneous prompts
            return await self._batch_heterogeneous_prompts(prompts, turn_idx)

    async def _batch_identical_prompts(
        self, prompt: str, count: int, turn_idx: int
    ) -> List[str]:
        """Handle identical prompts efficiently using n parameter."""
        print(
            f"    \033[93m→ TURN {turn_idx+1} prompt full:\033[0m \033[92m{prompt}\033[0m"
        )

        gen_limit = self.config.max_gen_per_turn
        resp = await self.server.completion(
            prompt=prompt,
            n=count,
            max_tokens=gen_limit,
            temperature=0.8,
        )
        choices = [c.text for c in resp.choices]

        # Debug: print each rollout
        for i, raw in enumerate(choices):
            print(
                f"    \033[93m· turn {turn_idx+1} rollout raw [{i}]:\033[0m \033[94m{raw}\033[0m"
            )
            if not raw.strip():
                print(f"      → (empty or error string returned for rollout {i})")
        print("    → All turn 1 rollouts printed; moving on.\n" + "-" * 48)

        return choices

    async def _batch_heterogeneous_prompts(
        self, prompts: List[str], turn_idx: int
    ) -> List[str]:
        """Handle heterogeneous prompts using parallel requests."""
        if turn_idx == 1:
            print("=== DEBUG: Now parallelizing Turn 2 prompts ===")
        print(f"    → Parallelizing {len(prompts)} prompts at turn {turn_idx+1}")

        # Print each prompt
        for idx_p, p_str in enumerate(prompts):
            print(
                f"    \033[93m→ TURN-{turn_idx+1} prompt[{idx_p}] full:\033[0m \033[92m{p_str}\033[0m"
            )

        async def _call_single(prompt_str: str) -> str:
            try:
                gen_limit = self.config.max_gen_per_turn
                comp = await self.server.completion(
                    prompt=prompt_str,
                    n=1,
                    max_tokens=gen_limit,
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
            print(
                f"    \033[93m· rollout {i} (Turn {turn_idx+1}) full reply:\033[0m \033[94m{raw}\033[0m\n"
                + "-" * 48
            )
            if not raw:
                print(f"    → Rollout {i} returned empty or error string")
            choices.append(raw)

        return choices

    async def _process_turn_responses(
        self,
        turn_idx: int,
        choices: List[str],
        ridx_map: List[int],
        contexts: List[List[Dict[str, str]]],
        preds_by_turn: List[List[List]],
        active: List[bool],
        expected_calls_by_turn: List[List[str]],
    ) -> None:
        for txt, r in zip(choices, ridx_map):
            raw_txt = txt or ""
            norm_txt = _normalize_tool_call_json(raw_txt)
            print(f"\n\033[93m=== TURN {turn_idx+1} · ROLLOUT {r} ===\033[0m")
            print(f"\033[95mRaw assistant reply:\033[0m\n\033[94m{raw_txt}\033[0m")

            expected_turn_calls = expected_calls_by_turn[turn_idx]
            is_valid = False

            if expected_turn_calls:  # Turn SHOULD have tool calls
                calls = _validate_think_plus_calls(norm_txt)
                print(f"\033[95mExtracted tool calls:\033[0m {calls}")
                print(f"\033[95mExpected tool calls:\033[0m {expected_turn_calls}")

                if calls is None:
                    preds_by_turn[r][turn_idx].append("__MISMATCH__")
                    active[r] = False
                    continue

                # Check number of calls and content matches
                if len(calls) != len(expected_turn_calls):
                    print(
                        f"\033[91m[DEBUG] Call‑count mismatch — model={len(calls)} vs exp={len(expected_turn_calls)}\033[0m"
                    )
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
                    print("\033[91m[DEBUG] Tool‑call field mismatch detected\033[0m")
                    preds_by_turn[r][turn_idx].append("__MISMATCH__")
                    active[r] = False
                else:
                    is_valid = True
                    preds_by_turn[r][turn_idx].extend(calls)
                    if self.config.add_dynamic_few_shot and calls:
                        self.dynamic_example = norm_txt.strip()

            else:  # Narration / summary turn
                if not _validate_think_only(norm_txt):
                    print(
                        f"[DEBUG] Invalid narration turn for rollout {r}, turn {turn_idx}: missing <think> or contains <tool_call>"
                    )
                    active[r] = False
                else:
                    is_valid = True
                    preds_by_turn[r][turn_idx] = []

            # Only append to context if validation passed
            if is_valid:
                contexts[r].append({"role": "assistant", "content": norm_txt})

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

        # --- Inject extra summary turn if generate_all_gpt_turns is enabled ---
        if self.config.generate_all_gpt_turns:
            expected_calls_by_turn = list(expected_calls_by_turn)
            inter_turns = list(inter_turns)
            expected_calls_by_turn.append(
                []
            )  # Add empty expected call turn for summary
            inter_turns.append([])  # Extend inter_turns to align contexts

        num_rollouts = self.config.group_size
        contexts: List[List[Dict[str, str]]] = [
            list(base_ctx) for _ in range(num_rollouts)
        ]
        preds_by_turn: List[List[List]] = [
            [[] for _ in range(len(expected_calls_by_turn))]
            for _ in range(num_rollouts)
        ]
        active = [True] * num_rollouts

        # --- Compute max_turns to allow summary turn if present ---
        max_turns = min(
            len(expected_calls_by_turn),
            (
                self.config.max_tool_call_turns_cap
                if self.config.max_tool_call_turns_cap is not None
                else len(expected_calls_by_turn)
            ),
        )

        for turn_idx in range(max_turns):
            print(
                f"[collect_trajectories] Beginning turn {turn_idx+1}/{max_turns} for this group"
            )

            # Build contexts and prompts for this turn
            prompts, ridx_map = await self._build_turn_contexts(
                turn_idx, contexts, inter_turns, active
            )

            if not prompts:
                break

            max_prompt_len = max(len(p) for p in prompts)
            max_gen = min(
                self.config.max_gen_per_turn,
                max(1, self.config.max_token_length - max_prompt_len),
            )

            # Execute inference for this turn
            choices = await self._execute_turn_inference(turn_idx, prompts, ridx_map)

            # Process and validate responses
            await self._process_turn_responses(
                turn_idx,
                choices,
                ridx_map,
                contexts,
                preds_by_turn,
                active,
                expected_calls_by_turn,
            )

            if not any(active):
                print("    → All roll-outs terminated; stopping further turns.")
                break

            survivors = sum(1 for a in active if a)
            print(
                f"    → DEBUG: finished turn {turn_idx+1}; {survivors}/{num_rollouts} rollouts still active"
            )

        scored = ScoredDataGroup(tokens=[], masks=[], scores=[])
        # Flatten expected calls for scoring (since _score_episode expects flat list)
        expected_calls_flat = [
            call for turn_calls in expected_calls_by_turn for call in turn_calls
        ]
        for r in range(num_rollouts):
            # flatten per‑turn predictions for scoring
            preds_flat: List = []
            for turn_preds in preds_by_turn[r]:
                preds_flat.extend(turn_preds)
            reward = self._score_episode(
                preds_flat,
                expected_calls_flat,
                wrong_call_penalty=self.config.wrong_call_penalty,
            )
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
                    frac = min(
                        (ln - cutoff) / (self.config.max_token_length - cutoff), 1.0
                    )
                    scored["scores"][i] = max(0.0, scored["scores"][i] - frac)

        for s in scored["scores"]:
            self.raw_score_buffer.append(s)
            self.percent_correct_buffer.append(1.0 if s >= 1.0 else 0.0)

        if len(scored["tokens"]) < self.config.group_size or scored["scores"].count(
            scored["scores"][0]
        ) == len(scored["scores"]):
            return None, []

        # ───────────────────── Final rollout debug ──────────────────────
        print("\n\033[92m=== FINAL ROLLOUT SUMMARY ===\033[0m")
        for r_i, (ctx, score) in enumerate(zip(contexts, scored["scores"])):
            last_assistant = next(
                (m["content"] for m in reversed(ctx) if m["role"] == "assistant"),
                "(no assistant message)",
            )
            print(f"\n\033[96mRollout {r_i} · score={score:.3f}\033[0m")
            print(last_assistant)
            print("-" * 60)
        print("=== END SUMMARY ===\n")
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
            pct_correct = sum(self.percent_correct_buffer) / len(
                self.percent_correct_buffer
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
