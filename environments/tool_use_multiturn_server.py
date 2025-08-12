"""
Multi-Turn Tool-Calling Environment
==================================
 # flake8: noqa: E501

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
import logging
import random
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset
from pydantic import Field
from tqdm.asyncio import tqdm_asyncio

import wandb
from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
    Item,
    ScoredDataGroup,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer


class MultiTurnEnvConfig(BaseEnvConfig):
    """Configuration for Multi-Turn Tool Calling Environment."""

    max_tool_call_turns_cap: Optional[int] = Field(
        default=2,
        description="Upper cap on assistant TOOL‑CALLING turns per episode (None → no cap)",
    )
    validate_think_blocks: bool = Field(
        default=False,
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
    completed_dataset_id: Optional[str] = Field(
        default=None,
        description="Dataset id containing tasks already completed; used to skip duplicates when skip_completed=True. If None, no skipping will occur.",
    )
    scenario_category: str = Field(
        default="multiturn",
        description='BFCL‑style scenario type: "single", "multistep", "multiturn", or "relevance".',
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
)  # noqa: E501

# Helper instructing the model how to handle *relevance* turns
APOLOGY_HELPER = (
    "If none of the available tools can satisfy the user's request, "
    "reply with a <think> block that reasons about the mismatch, then begin your "
    "user‑visible text with an explicit apology such as “I'm sorry” or “Apologies,” "
    "and briefly explain why you cannot use the tools."
)  # noqa: E501

# Helper instructing the model how to request missing info
CLARIFICATION_HELPER = (
    "If the available tools require parameters the user hasn’t provided, "
    "reply with a <think> block that notes the missing details, then begin your "
    "user-visible text with a polite request such as "
    "“Could you please provide the required details?” or "
    "“There’s insufficient information to proceed, may I have the missing data?”"
)  # noqa: E501

# Helper shown right after each <tool_response> when a narration/summary turn is expected
NARRATION_THINK_HELPER = (
    "Reply with a <think> block followed by the user‑visible summary for the tool result above. "
    "Do **not** include any <tool_call> blocks in that tool result summary message. "
    "Do **not** generate <tool_response> blocks yourself since those are provided by the system\n"
    "Example Tool Result Summary:\n"
    "<think>\n"
    "The tool call was successful, the user asked for the weather in SF and the tool returned 70 degrees and sunny.\n"
    "</think>\n"
    "The weather in SF is 70 degrees and sunny."
)  # noqa: E501


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
# Helper: canonicalize tool_call JSON string (for tool_call extraction)
# ------------------------------------------------------------------
def _canonicalise_tool_json(raw: str) -> Optional[str]:
    """
    Try to parse raw as JSON, then as Python literal, and return canonical json.dumps
    with separators=(',', ':'). If neither works, return None.
    """
    try:
        obj = json.loads(raw)
        return json.dumps(obj, separators=(",", ":"))
    except Exception:
        pass
    try:
        obj = ast.literal_eval(raw)
        return json.dumps(obj, separators=(",", ":"))
    except Exception:
        return None


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

    # Must not contain any <tool_response>
    if re.search(r"<tool_response\s*>", txt, flags=re.IGNORECASE):
        return False

    return True


def _validate_think_plus_calls(txt: str) -> Optional[List[dict]]:
    """
    Validate an assistant reply that *must* contain exactly one <think>…</think>
    block followed by one **or more** <tool_call>…</tool_call> blocks.

    Strict rules:

    • The reply must start with the <think> block.
    • Immediately after </think> (and any whitespace / new‑lines) there must be
      a <tool_call>.  No narrative text is allowed before the first tool_call.
    • Only whitespace / new‑lines are allowed between successive <tool_call>
      blocks.
    • After the final </tool_call> only whitespace / new‑lines are allowed.
    • The reply must not contain any <tool_response> blocks.
    • Returns the parsed list of tool‑call JSON objects on success, otherwise
      None.
    """
    txt = _normalize_tool_call_json(txt)

    # Reject if any tool_response slips in
    if re.search(r"<tool_response\s*>", txt, flags=re.IGNORECASE):
        return None

    # Require exactly one leading <think> … </think>
    m = re.match(r"\s*(<think>[\s\S]*?</think>)", txt, flags=re.IGNORECASE)
    if not m:
        return None
    think_block = m.group(1)
    rest = txt[len(think_block) :]

    # Helper regex for a single tool_call including surrounding whitespace
    tc_pattern = r"\s*<tool_call>\s*([\s\S]*?)\s*</tool_call>\s*"

    tool_calls = []
    while True:
        m_tc = re.match(tc_pattern, rest, flags=re.IGNORECASE)
        if not m_tc:
            break
        raw_json = m_tc.group(1)
        canon = _canonicalise_tool_json(raw_json)
        if canon is None:
            return None
        tool_calls.append(json.loads(canon))
        rest = rest[m_tc.end() :]

    # Must have parsed *at least* one tool_call and nothing but whitespace left
    if not tool_calls or rest.strip():
        return None
    return tool_calls


# ------------------------------------------------------------------
# Helper: validate a reply and extract tool calls or narration
# ------------------------------------------------------------------
def _validate_reply_and_extract(txt: str) -> Optional[List[dict]]:
    """
    Unified validator for eval:
    - If the reply contains <tool_call>, validate & return tool calls.
    - If it has no <tool_call>, require a single narration-only <think> block and return [].
    - Return None on validation failure.
    """
    if re.search(r"<tool_call\s*>", txt, flags=re.IGNORECASE):
        return _validate_think_plus_calls(txt)
    # narration-only case
    return [] if _validate_think_only(txt) else None


# ------------------------------------------------------------------
# Helper: robust JSON-like coercion and comparison for tool_call blocks
# ------------------------------------------------------------------
def _coerce_jsonlike(val):
    """
    Best-effort coercion of "JSON-like" values that sometimes arrive
    double-encoded in the source dataset.

    Common pathologies we handle:
        • arguments payload serialized as a *stringified* JSON object.
        • python-primitive booleans represented as strings: "True"/"False".
        • python-literal dict/tuple strings (ast.literal_eval fallback).

    Returns the coerced python object (dict / list / bool / int / str ...).
    If coercion fails, the original value is returned unchanged.
    """
    # Fast path – already non-string
    if not isinstance(val, str):
        return val

    s = val.strip()

    # Bool strings
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    if s.lower() == "null" or s.lower() == "none":
        return None

    # JSON object/array
    if (s.startswith("{") and s.endswith("}")) or (
        s.startswith("[") and s.endswith("]")
    ):
        try:
            return json.loads(s)
        except Exception:
            # fall through to literal_eval
            pass

    # Python literal (single quotes, etc.)
    try:
        return ast.literal_eval(s)
    except Exception:
        return val


def _parse_expected_call(raw_call):
    """
    Parse an expected tool-call that may be *double encoded* (i.e., the
    `arguments` field itself is a JSON-serialized string).  Returns a dict.
    """
    obj = raw_call
    if isinstance(raw_call, str):
        try:
            obj = json.loads(raw_call)
        except Exception:
            obj = _coerce_jsonlike(raw_call)

    # Defensive: if still not a dict, bail out with empty dict so comparison fails cleanly.
    if not isinstance(obj, dict):
        return {}

    # Recursively coerce problematic leaf nodes (esp. obj["arguments"])
    if "arguments" in obj:
        obj["arguments"] = _coerce_jsonlike(obj["arguments"])
    return obj


def _json_objects_match(model_json, expected_json):
    """
    True when every key/value in expected_json appears exactly in model_json.
    Nested dicts handled recursively.

    This variant is *robust* to several dataset irregularities:
      • expected_json["arguments"] provided as a *stringified* JSON object.
      • python-style booleans ("True"/"False") vs JSON booleans (true/false).
      • extraneous ordering / whitespace differences.

    Comparison is asymmetric: model_json must contain >= the expected fields.
    """
    # Coerce string-encoded dicts / bools, etc.
    model_json = _coerce_jsonlike(model_json)
    expected_json = _coerce_jsonlike(expected_json)

    # If expected is dict, model must also be dict
    if isinstance(expected_json, dict):
        if not isinstance(model_json, dict):
            return False
        for k, v in expected_json.items():
            if k not in model_json:
                return False
            if not _json_objects_match(model_json[k], v):
                return False
        return True

    # Non-dict leaves: strict equality after coercion
    return model_json == expected_json


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
            "interstellarninja/toolace_hermes_sequential_tool_use",
            split="train",
            # "interstellarninja/glaive-function-calling-5k", split="train"
            # "interstellarninja/salesforce_hermes_thinking", split="train"
            # "interstellarninja/hermes_salesforce_apigen_tool_use", split="train"
            # "interstellarninja/toolace_hermes_tool_use", split="train"
            # "interstellarninja/nvidia_hermes_when2call", split="train"
            # "NousResearch/salesforce_hermes_tools", split="train"
        )

        self.percent_correct_buffer: List[float] = []
        self.raw_score_buffer: List[float] = []
        self.eval_metrics: List[Tuple[str, float]] = []
        self.rollouts_for_wandb: List = []

        # Holds latest good assistant tool‑calling example (<think> + <tool_call>)
        self.dynamic_example: Optional[str] = None

        # Completed tasks (loaded in setup if enabled)
        self.completed_tasks: set[str] = set()

        # List of (messages_tuple, expected_calls_by_turn, inter_turns) triples
        self.train_items: List[
            Tuple[Tuple[frozenset, ...], List[List[str]], List[List[Dict[str, str]]]]
        ] = []
        self.test_items: List[
            Tuple[Tuple[frozenset, ...], List[List[str]], List[List[Dict[str, str]]]]
        ] = []
        self.iter = 0
        self.max_token_len = 4096

    @classmethod
    def config_init(cls) -> Tuple[MultiTurnEnvConfig, List[APIServerConfig]]:
        env_cfg = MultiTurnEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=8,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=2000,
            batch_size=1024,
            steps_per_eval=10**12,
            max_token_length=4096 * 16,
            inference_weight=1.0,
            wandb_name="multiturn_tool_use",
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.1,
            # Use config defaults; all knobs are overrideable via CLI/YAML
            skip_completed=True,
        )
        # ─── Scenario‑specific tweaks ──────────────────────────────
        # For strict multi‑turn datasets we generate an explicit narration
        # turn after *every* tool response, so we do **not** want the legacy
        # “single extra summary at the very end” mechanic.
        if env_cfg.scenario_category == "multiturn":
            env_cfg.generate_all_gpt_turns = False
        elif env_cfg.scenario_category == "single":
            # For single‑turn episodes we stop after the first assistant
            # tool‑calling turn.  No automatic narration / summary turn.
            env_cfg.generate_all_gpt_turns = False
        elif env_cfg.scenario_category == "relevance":
            # Relevance runs expect a single narration response with no tool calls
            env_cfg.generate_all_gpt_turns = False
            env_cfg.skip_completed = (
                False  # keep all episodes; completed‑task list not relevant
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

        # Populate completed-tasks filter inside setup
        if self.config.skip_completed and self.config.completed_dataset_id:
            try:
                _done_ds = load_dataset(self.config.completed_dataset_id, split="train")
                self.completed_tasks = set(_done_ds["task"])  # type: ignore
                print(
                    f"[filter] Loaded {len(self.completed_tasks):,} completed tasks from {self.config.completed_dataset_id}"
                )
            except Exception as _exc:
                self.completed_tasks = set()
                print(
                    f"[filter] Could not load completed-task dataset: {_exc}. No skipping will occur."
                )

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

        split = ds.train_test_split(0.05)
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
            if self.config.skip_completed and self.completed_tasks:
                first_user_msg = conv[1]["value"].strip()
                if first_user_msg in self.completed_tasks:
                    continue

            # Find all tool-calling turns
            tool_indices = [
                i
                for i, m in enumerate(conv)
                if m["from"] in ("gpt", "assistant")
                and "<tool_call>" in m["value"].lower()
            ]
            # ── SINGLE‑TURN cap (when generate_all_gpt_turns == False) ──
            if (
                self.config.scenario_category == "single"
                and not self.config.generate_all_gpt_turns
                and tool_indices
            ):
                # Keep only the first assistant tool‑calling turn; discard later ones
                tool_indices = tool_indices[:1]
                # ── SINGLE‑TURN truncate conversation after first tool‑call turn ──
                conv = conv[
                    : tool_indices[0] + 1
                ]  # keep up to and including the tool‑call turn

            if self.config.scenario_category == "relevance":
                # ─── RELEVANCE (single narration, no tool calls in prefix) ───
                # Relevance training requires an existing *non‑tool* assistant message
                first_asst_idx = next(
                    (
                        i
                        for i, m in enumerate(conv[2:], start=2)
                        if m["from"] in ("gpt", "assistant")
                    ),
                    None,
                )
                if first_asst_idx is None:
                    continue  # skip conversations lacking the first GPT reply
                asst_msg = conv[first_asst_idx]["value"]
                if "<tool_call" in asst_msg.lower():
                    continue  # first assistant turn itself must not call a tool

                # Truncate the conversation right *before* the first assistant turn
                conv = conv[: first_asst_idx + 1]

                # No tool calls are allowed *in the truncated prefix*
                if any(
                    "<tool_call" in m["value"].lower()
                    for m in conv
                    if m["from"] in ("gpt", "assistant")
                ):
                    continue

                # Build running messages (system + human only)
                running_msgs = []
                # For the relevance scenario we **do not** prepend any tool‑calling example
                combined_system = (
                    system_prompt
                    + "\n\n"
                    + APOLOGY_HELPER
                    + "\n\n"
                    + CLARIFICATION_HELPER
                    + "\n\n"
                    + conv[0]["value"]
                )
                running_msgs.append(
                    frozenset({"role": "system", "content": combined_system}.items())
                )
                running_msgs.append(
                    frozenset({"role": "user", "content": conv[1]["value"]}.items())
                )

                # Expect exactly one assistant turn with NO tool calls
                expected_calls_by_turn = [[]]
                inter_turns = []

                target.append(
                    (tuple(running_msgs), expected_calls_by_turn, inter_turns)
                )
                continue

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
                # ─── STRICT MULTI‑TURN PATTERN ───
                #
                # User
                # Assistant <tool_call>
                # Tool
                # Assistant (summary – *no* <tool_call>)
                # User  …  ↩︎ (before the next tool_call)
                #
                # Every assistant turn must start with <think>; the tool‑call
                # turns are validated via _validate_think_plus_calls(), the
                # narration turns via _validate_think_only().

                # First assistant turn must itself be the first tool‑calling message
                first_asst_idx = next(
                    (
                        i
                        for i, m in enumerate(conv[2:], start=2)
                        if m["from"] in ("gpt", "assistant")
                    ),
                    None,
                )
                if first_asst_idx != tool_indices[0]:
                    continue  # another assistant turn precedes the first tool‑call

                # Must have at least one human after the first tool‑call
                if not human_after_first_tool:
                    continue

                # Must have at least TWO assistant tool‑calling turns overall
                if len(tool_indices) < 2:
                    continue

                expected_calls_by_turn: List[List[str]] = []
                inter_turns: List[List[Dict[str, str]]] = []
                ok = True

                for idx_t, tool_idx in enumerate(tool_indices):
                    # 1. Tool response directly after the tool‑call
                    try:
                        tool_resp_idx = next(
                            j
                            for j in range(tool_idx + 1, len(conv))
                            if conv[j]["from"] == "tool"
                        )
                    except StopIteration:
                        ok = False
                        break

                    # 2. Assistant summary (no <tool_call>) after tool response
                    try:
                        summ_idx = next(
                            j
                            for j in range(tool_resp_idx + 1, len(conv))
                            if conv[j]["from"] in ("gpt", "assistant")
                        )
                    except StopIteration:
                        ok = False
                        break
                    if "<tool_call" in conv[summ_idx]["value"].lower():
                        ok = False
                        break
                    # Narration must start with <think> *only when* validate_think_blocks is True
                    if self.config.validate_think_blocks and not re.match(
                        r"^\s*<think>", conv[summ_idx]["value"], flags=re.IGNORECASE
                    ):
                        ok = False
                        break

                    # 3. If another tool‑call follows, ensure a human turn exists
                    nxt_tool_idx = (
                        tool_indices[idx_t + 1]
                        if idx_t + 1 < len(tool_indices)
                        else None
                    )
                    if nxt_tool_idx is not None:
                        slice_after_summary = conv[summ_idx + 1 : nxt_tool_idx]
                        if not any(m["from"] == "human" for m in slice_after_summary):
                            ok = False
                            break

                    # ─── Build turn A: assistant tool‑call ───
                    tool_call_msg = conv[tool_idx]["value"]
                    # Tool‑calling assistant turn must begin with <think> when flag enabled
                    if self.config.validate_think_blocks and not re.match(
                        r"^\s*<think>", tool_call_msg, flags=re.IGNORECASE
                    ):
                        ok = False
                        break
                    tc_raws = re.findall(
                        r"<tool_call>\s*(.*?)\s*</tool_call>",
                        tool_call_msg,
                        flags=re.DOTALL | re.IGNORECASE,
                    )
                    if not tc_raws:
                        ok = False
                        break
                    turn_calls: List[str] = []
                    for raw in tc_raws:
                        canon = _canonicalise_tool_json(raw)
                        if canon is None:
                            ok = False
                            break
                        turn_calls.append(canon)
                    if not ok:
                        break
                    expected_calls_by_turn.append(turn_calls)

                    # Inter‑turn after tool‑call → tool response AND narration helper
                    inter_turns.append(
                        [
                            {
                                "role": "tool",
                                "content": conv[tool_resp_idx]["value"]
                                + "\n\n"
                                + NARRATION_THINK_HELPER,
                            }
                        ]
                    )

                    # ─── Build turn B: assistant narration (no calls) ───
                    expected_calls_by_turn.append([])

                    # Inter‑turn after narration → up to next tool‑call (user, etc.)
                    slice_end = nxt_tool_idx if nxt_tool_idx is not None else len(conv)
                    next_ctx_slice = [
                        {
                            "role": m["from"].replace("gpt", "assistant"),
                            "content": m["value"],
                        }
                        for m in conv[summ_idx + 1 : slice_end]
                    ]
                    inter_turns.append(next_ctx_slice)

                if not ok:
                    continue  # does not satisfy strict multi‑turn pattern

                # Remove trailing inter‑turn (nothing after final narration)
                if inter_turns:
                    inter_turns.pop()

                # Apply cap *per tool‑calling turn* (ignoring narration turns)
                cap = self.config.max_tool_call_turns_cap
                if cap is not None:
                    keep_turns = 0  # total assistant turns to keep
                    calls_seen = 0  # tool‑calling turns seen so far
                    for idx, calls in enumerate(expected_calls_by_turn):
                        keep_turns += 1
                        if calls:  # non‑empty ⇒ tool‑calling turn
                            calls_seen += 1
                            # Keep the narration turn that immediately follows the last kept call
                            if calls_seen == cap:
                                if idx + 1 < len(expected_calls_by_turn):
                                    keep_turns += 1
                                break
                    expected_calls_by_turn = expected_calls_by_turn[:keep_turns]
                    inter_turns = inter_turns[: keep_turns - 1]

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
                human_content = conv[1]["value"]
                running_msgs.append(
                    frozenset({"role": "user", "content": human_content}.items())
                )

                target.append(
                    (tuple(running_msgs), expected_calls_by_turn, inter_turns)
                )
                continue  # strict multiturn handled

            else:
                raise ValueError(
                    f"Unknown scenario_category={self.config.scenario_category} (expected 'single', 'multistep', 'multiturn', or 'relevance')"
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
                # Enforce <think> only when validate_think_blocks is True
                if self.config.validate_think_blocks and not re.match(
                    r"^\s*<think>", tool_call_msg, flags=re.IGNORECASE
                ):
                    continue
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
                    canon = _canonicalise_tool_json(raw)
                    if canon is None:
                        continue  # skip invalid call; will make mismatch later
                    turn_calls.append(canon)
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
                    # For single: collect all messages until next tool call
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
                            "content": conv[last_tool_response_idx]["value"]
                            + "\n\n"
                            + NARRATION_THINK_HELPER,
                        }
                    ]
                    inter_turns.append(final_inter_turn)

            # Apply turn cap if configured (count only tool-calling turns,
            # and keep narration turn after last kept tool-call if generate_all_gpt_turns is enabled)
            cap = self.config.max_tool_call_turns_cap
            if cap is not None:
                keep_turns = 0  # total assistant turns to keep
                calls_seen = 0  # tool‑calling turns seen so far
                for idx, calls in enumerate(expected_calls_by_turn):
                    keep_turns += 1
                    if calls:  # non‑empty ⇒ tool‑calling turn
                        calls_seen += 1
                        # Keep the narration turn that immediately follows
                        # the *last* retained tool‑call when generate_all_gpt_turns
                        if calls_seen == cap:
                            if self.config.generate_all_gpt_turns and idx + 1 < len(
                                expected_calls_by_turn
                            ):
                                keep_turns += 1
                            break
                expected_calls_by_turn = expected_calls_by_turn[:keep_turns]
                inter_turns = inter_turns[: keep_turns - 1]  # N turns → N‑1 gaps

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
    ) -> Tuple[float, int]:
        """
        Returns (reward, num_correct_calls)

        • dense   = (#correct / N)
        • sparse  = +lam if ALL correct
        • penalty = wrong_call_penalty on first mismatch  (‑0.2 default)

        A "__MISMATCH__" sentinel in pred_calls triggers the penalty.
        """
        # ── Special case: RELEVANCE episodes (no expected tool calls) ──
        if len(exp_calls) == 0:
            # Relevance episode: bonus for explicit apology.
            has_apology = "__APOLOGY__" in pred_calls
            has_info = "__INFO__" in pred_calls
            other_calls = [
                c
                for c in pred_calls
                if c not in ("__APOLOGY__", "__INFO__", "__MISMATCH__")
            ]
            success = ("__MISMATCH__" not in pred_calls) and not other_calls
            if not success:
                return wrong_call_penalty, 0
            base = 1.0
            bonus = 0.1 * int(has_apology) + 0.1 * int(has_info)
            return base + bonus, 0

        # Normalise expected calls (handles double-encoded 'arguments' payloads, etc.)
        exp_jsons: List[dict] = [_parse_expected_call(r) for r in exp_calls]

        mismatch_penalty = 0.0
        if pred_calls and "__MISMATCH__" in pred_calls:
            pred_calls = [c for c in pred_calls if c != "__MISMATCH__"]
            mismatch_penalty = wrong_call_penalty

        # Pad pred list so zip() covers all exp if shorter
        pred_calls += [{}] * (len(exp_jsons) - len(pred_calls))

        correct = sum(
            1 for p, e in zip(pred_calls, exp_jsons) if _json_objects_match(p, e)
        )
        dense = correct / max(1, len(exp_jsons))
        bonus = lam if correct == len(exp_jsons) else 0.0
        return dense + bonus + mismatch_penalty, correct

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
        score, _ = self._score_episode(
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
                    exp_obj = _parse_expected_call(exp_raw)
                    if not _json_objects_match(mdl, exp_obj):
                        mismatch = True
                        break

                if mismatch:
                    print("\033[91m[DEBUG] Tool‑call field mismatch detected\033[0m")
                    preds_by_turn[r][turn_idx].append("__MISMATCH__")
                    active[r] = False
                else:
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
                    # Tag explicit apology wording so the scorer can reward it
                    tags = []
                    if re.search(
                        r"\b(?:sorry|apologies)\b", norm_txt, flags=re.IGNORECASE
                    ):
                        tags.append("__APOLOGY__")
                    if (
                        re.search(
                            r"\binsufficient information\b",
                            norm_txt,
                            flags=re.IGNORECASE,
                        )
                        or re.search(
                            r"\bcould you provide\b", norm_txt, flags=re.IGNORECASE
                        )
                        or re.search(
                            r"\bprovide (?:me )?.*(?:details|information|data)\b",
                            norm_txt,
                            flags=re.IGNORECASE,
                        )
                    ):
                        tags.append("__INFO__")
                    preds_by_turn[r][turn_idx] = tags

            # Always record the assistant's reply so the trajectory is complete,
            # even when the turn is invalid (the rollout will simply receive
            # a negative reward and be terminated early).  This preserves the
            # full dialogue history for RL training.
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

        # --- Compute max_turns ---
        if self.config.scenario_category == "multiturn":
            # expected_calls_by_turn already truncated above, so run the whole sequence
            max_turns = len(expected_calls_by_turn)
        else:
            cap = self.config.max_tool_call_turns_cap
            if cap is None:
                max_turns = len(expected_calls_by_turn)
            else:
                # Count only tool‑calling turns toward the cap while still allowing
                # **one extra narration/summary turn** that immediately follows the
                # last retained tool‑calling turn when `generate_all_gpt_turns` is on.
                tool_turns = 0
                max_turns = 0
                for calls in expected_calls_by_turn:
                    max_turns += 1
                    if calls:  # tool‑calling turn
                        tool_turns += 1
                        if tool_turns >= cap:
                            # If the very next turn is a narration turn (i.e. an empty
                            # expected‑call list) and we are supposed to generate it,
                            # keep **one more** turn so the model can respond.
                            if (
                                self.config.generate_all_gpt_turns
                                and max_turns < len(expected_calls_by_turn)
                                and not expected_calls_by_turn[
                                    max_turns
                                ]  # next turn has no calls
                            ):
                                max_turns += 1
                            break

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
            reward, num_correct = self._score_episode(
                preds_flat,
                expected_calls_flat,
                wrong_call_penalty=self.config.wrong_call_penalty,
            )

            # Dataset‑generation success criterion: need ≥2 validated tool calls
            if self.config.scenario_category == "multiturn" and num_correct < 2:
                reward = -1.0

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

        drop_group = len(scored["tokens"]) < self.config.group_size or scored[
            "scores"
        ].count(scored["scores"][0]) == len(scored["scores"])
        if drop_group and self.config.scenario_category != "relevance":
            return None, []

        # ───────────────────── Final rollout debug ──────────────────────
        print("\n\033[92m=== FINAL ROLLOUT SUMMARY ===\033[0m")  # noqa: E501
        for r_i, (ctx, score) in enumerate(zip(contexts, scored["scores"])):
            last_assistant = next(
                (m["content"] for m in reversed(ctx) if m["role"] == "assistant"),
                "(no assistant message)",
            )
            print(f"\n\033[96mRollout {r_i} · score={score:.3f}\033[0m")
            print(last_assistant)
            print("-" * 60)
        print("=== END SUMMARY ===\n")  # noqa: E501
        await self.add_rollouts_for_wandb(scored, item)
        return scored, []

    async def create_rollout_table(self, wdict):
        """
        Build a WandB table from the buffered rollouts, if any.
        Each entry is (generation_text, score, expected_tool_calls_json_list).
        """
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
