"""
Multi-Turn Tool-Calling Environment with Turn-Level Advantages (MT-GRPO)
=========================================================================

Implements turn-level credit assignment following the MT-GRPO approach from:
"Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Credit Assignment"
(Zeng et al., 2025) - https://arxiv.org/abs/2505.11821

This environment provides a flexible base for multi-turn tool-calling tasks with
fine-grained advantage estimation. Users can customize by overriding:

    • `compute_turn_reward()` - Define turn-level reward signals (R^T_τ)
    • `compute_outcome_reward()` - Define outcome-level reward (R^O)
    • `validate_tool_call_turn()` - Custom tool call validation
    • `validate_summary_turn()` - Custom summary/narration validation
    • `get_next_item()` - Custom data loading

Key MT-GRPO formula (Equation 7 from paper):
    Â_τ = Â^T_τ + λ · Â^O   for all turns τ = 1, 2, ..., T

Where:
    - Â^T_τ = standardized turn-level advantage
    - Â^O = standardized outcome-level advantage
    - λ = turn_level_advantage_lambda (default 1.0)

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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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

# ==============================================================================
# Configuration
# ==============================================================================


class MTGRPOEnvConfig(BaseEnvConfig):
    """
    Configuration for Multi-Turn Tool Calling with Turn-Level Advantages.

    Key MT-GRPO parameters:
        - turn_level_advantage_lambda: λ coefficient for combining turn and outcome advantages
        - wrong_call_penalty: Penalty for incorrect tool calls (negative reward)
        - tool_execution_reward: Reward for valid tool call structure
        - tool_match_reward: Reward for correct tool call content
        - summary_reward: Reward for valid summary/narration turns
    """

    # MT-GRPO specific parameters
    turn_level_advantage_lambda: float = Field(
        default=1.0,
        description="λ coefficient for MT-GRPO advantage: Â_τ = Â^T_τ + λ·Â^O. Paper uses 1.0.",
    )
    wrong_call_penalty: float = Field(
        default=-0.2,
        description="Penalty applied when tool call validation fails (R^T component).",
    )
    tool_execution_reward: float = Field(
        default=0.5,
        description="Reward for valid tool call structure (<think> + <tool_call>).",
    )
    tool_match_reward: float = Field(
        default=0.5,
        description="Reward for tool call content matching expected.",
    )
    summary_reward: float = Field(
        default=1.0,
        description="Reward for valid summary/narration turn.",
    )

    # Environment structure parameters
    max_tool_call_turns_cap: Optional[int] = Field(
        default=2,
        description="Upper cap on tool-calling turns per episode (None → no cap).",
    )
    validate_think_blocks: bool = Field(
        default=True,
        description="Whether to require <think> blocks in assistant messages.",
    )
    generate_all_gpt_turns: bool = Field(
        default=True,
        description="Generate GPT turns after tool responses, including summaries.",
    )
    max_gen_per_turn: int = Field(
        default=2048,
        description="Max tokens the model may generate in a single turn.",
    )

    # Data handling parameters
    skip_completed: bool = Field(
        default=True,
        description="Skip conversations whose first user prompt appears in completed tasks.",
    )
    completed_dataset_id: Optional[str] = Field(
        default=None,
        description="Dataset id for completed tasks (used when skip_completed=True).",
    )
    scenario_category: str = Field(
        default="all",
        description='Scenario type: "single", "multistep", "multiturn", or "all" (accepts everything).',
    )
    min_tool_call_turns: int = Field(
        default=1,
        description="Minimum number of tool-calling turns required (set to 2 for MT-GRPO focus).",
    )
    min_successful_turns: int = Field(
        default=1,
        description="Minimum successful turns to keep a rollout (0 = keep all, even failures).",
    )
    add_dynamic_few_shot: bool = Field(
        default=True,
        description="Insert most-recent successful example into system prompt.",
    )
    use_parallel_requests: bool = Field(
        default=True,
        description="Use parallel requests instead of n parameter for batching.",
    )


# ==============================================================================
# Default Prompts (Override in subclass if needed)
# ==============================================================================

DEFAULT_SYSTEM_PROMPT = (
    "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the "
    "problem and deliberate with yourself via systematic reasoning processes to help come to a correct "
    "solution prior to answering. You should enclose your thoughts and internal monologue inside <think> "
    "</think> tags, and then provide your solution or response to the problem."
)

DEFAULT_FEW_SHOT_EXAMPLE = (
    "Example Reasoning format:\n"
    "<think>\n"
    "Okay, the user asked for a calculation, I need to use python_interpreter tool to compute it.\n"
    "</think>\n"
    "<tool_call>\n"
    '{"name": "python_interpreter", "arguments": {"code": "print(2+2)"}}\n'
    "</tool_call>\n"
)

DEFAULT_SEQ_TOOL_HELPER = (
    "You are in sequential tool calling mode. "
    "Call exactly **one** tool, wait for its <tool_response>, "
    "then decide whether to call another. "
    "Never bundle multiple <tool_call> blocks in the same message. "
    "Do not generate <tool_response> blocks by yourself since system will provide. "
    "When you get the <tool_response> and if you believe the task is complete, "
    "return your reasoning in <think> blocks followed by plain text summary."
)

DEFAULT_NARRATION_HELPER = (
    "Reply with a <think> block followed by the user-visible summary for the tool result above. "
    "Do **not** include any <tool_call> blocks in that tool result summary message. "
    "Do **not** generate <tool_response> blocks yourself since those are provided by the system.\n"
    "Example Tool Result Summary:\n"
    "<think>\n"
    "The tool call was successful, the user asked for the weather in SF and the tool returned 70 degrees and sunny.\n"
    "</think>\n"
    "The weather in SF is 70 degrees and sunny."
)

DEFAULT_CONTINUATION_HELPER = (
    "Continue with your reasoning in <think> blocks. "
    "If you need to call another tool, output <think>...</think> followed by <tool_call>...</tool_call>. "
    "Do **not** generate <tool_response> blocks yourself since those are provided by the system.\n"
    "Example continuation:\n"
    "<think>\n"
    "The first tool returned the data I needed. Now I need to call the next tool to complete the task.\n"
    "</think>\n"
    "<tool_call>\n"
    '{"name": "next_tool", "arguments": {"param": "value"}}\n'
    "</tool_call>"
)


def fix_system_prompt_json_examples(system_prompt: str) -> str:
    """
    Fix system prompts that use Python dict syntax in examples to use proper JSON.

    The dataset often contains examples like:
        <tool_call>
        {'name': <function-name>,'arguments': <args-dict>}
        </tool_call>

    This converts them to proper JSON format.
    """
    # Replace the common Python dict example pattern with proper JSON
    # Pattern 1: {'name': <function-name>,'arguments': <args-dict>}
    system_prompt = re.sub(
        r"<tool_call>\s*\{'name':\s*<function-name>,\s*'arguments':\s*<args-dict>\}\s*</tool_call>",
        '<tool_call>\n{"name": "<function-name>", "arguments": {"<arg1>": "<value1>"}}\n</tool_call>',
        system_prompt,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Pattern 2: Example with single quotes in pydantic schema description
    # Replace single-quoted JSON schema examples
    system_prompt = re.sub(r"'title':\s*'([^']+)'", r'"title": "\1"', system_prompt)
    system_prompt = re.sub(r"'type':\s*'([^']+)'", r'"type": "\1"', system_prompt)
    system_prompt = re.sub(r"'properties':\s*\{", r'"properties": {', system_prompt)
    system_prompt = re.sub(r"'required':\s*\[", r'"required": [', system_prompt)

    # Fix the pydantic model json schema section - replace Python dict with JSON
    # This handles: {'title': 'FunctionCall', 'type': 'object', ...}
    pydantic_pattern = r"\{'title':\s*'FunctionCall'[^}]+\}"
    pydantic_replacement = (
        '{"title": "FunctionCall", "type": "object", "properties": '
        '{"name": {"title": "Name", "type": "string"}, '
        '"arguments": {"title": "Arguments", "type": "object"}}, '
        '"required": ["name", "arguments"]}'
    )
    system_prompt = re.sub(pydantic_pattern, pydantic_replacement, system_prompt)

    return system_prompt


# ==============================================================================
# Utility Functions (Can be used by subclasses)
# ==============================================================================


def strip_tool_response_tags(content: str) -> str:
    """Strip outer <tool_response> tags from content if present."""
    # Match <tool_response>...</tool_response> and extract inner content
    match = re.match(
        r"^\s*<tool_response>\s*([\s\S]*?)\s*</tool_response>\s*$",
        content,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return content


def normalize_tool_call_json(txt: str) -> str:
    """
    Normalize assistant replies to canonical JSON format.
    Preserves <think> block and converts Python-style dicts to JSON.
    """
    m = re.match(r"^\s*(<think>[\s\S]*?</think>)\s*", txt, flags=re.IGNORECASE)
    if not m:
        return txt
    think_part = m.group(1)

    def _convert(match: re.Match) -> str:
        raw = match.group(1).strip()
        try:
            obj = ast.literal_eval(raw)
            return f"<tool_call>{json.dumps(obj, separators=(',', ':'))}</tool_call>"
        except Exception:
            pass
        try:
            json_like = re.sub(r"'([^']*)':", r'"\1":', raw)
            json_like = re.sub(r":\s*'([^']*)'", r':"\1"', json_like)
            json.loads(json_like)
            return f"<tool_call>{json_like}</tool_call>"
        except Exception:
            return match.group(0)

    tail = txt[len(m.group(0)) :]
    tail = re.sub(
        r"<tool_call>\s*([\s\S]*?)\s*</tool_call>",
        _convert,
        tail,
        flags=re.DOTALL | re.IGNORECASE,
    )
    out = think_part + tail
    out = re.sub(r"\s*<tool_call>\s*", "\n<tool_call>\n", out)
    out = re.sub(r"\s*</tool_call>\s*", "\n</tool_call>\n", out)
    return out


def canonicalise_tool_json(raw: str) -> Optional[str]:
    """Parse raw as JSON or Python literal, return canonical JSON string."""
    raw = raw.strip()

    # Reject if it contains XML tags - malformed data
    if re.search(r"</?(?:think|tool_call|tool_response)>", raw, re.IGNORECASE):
        return None

    # First try direct JSON parsing
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "name" in obj:
            return json.dumps(obj, separators=(",", ":"))
    except Exception:
        pass

    # Try Python literal (handles single quotes)
    try:
        obj = ast.literal_eval(raw)
        if isinstance(obj, dict) and "name" in obj:
            return json.dumps(obj, separators=(",", ":"))
    except Exception:
        pass

    return None


def extract_tool_calls(
    txt: str, require_think_block: bool = True
) -> Optional[List[dict]]:
    """
    Extract tool calls from a response.

    Args:
        txt: The response text to parse
        require_think_block: If True, requires <think> block before tool calls.
                            If False, extracts tool calls regardless of think blocks.

    Returns list of parsed tool call dicts, or None if validation fails.
    """
    txt = normalize_tool_call_json(txt)

    if re.search(r"<tool_response\s*>", txt, flags=re.IGNORECASE):
        print("\033[91m[DEBUG extract] Rejected: contains <tool_response>\033[0m")
        return None

    if require_think_block:
        # Strict mode: require <think> block at start
        m = re.match(r"\s*(<think>[\s\S]*?</think>)", txt, flags=re.IGNORECASE)
        if not m:
            print(
                f"\033[91m[DEBUG extract] Rejected: no <think> block "
                f"(require_think={require_think_block})\033[0m"
            )
            print(f"\033[91m[DEBUG extract] Text starts with: '{txt[:150]}...'\033[0m")
            return None
        rest = txt[len(m.group(1)) :]
    else:
        # Flexible mode: skip any leading content before first <tool_call>
        # Find the first <tool_call> tag
        first_tc = re.search(r"<tool_call>", txt, flags=re.IGNORECASE)
        if not first_tc:
            print("\033[91m[DEBUG extract] Rejected: no <tool_call> found\033[0m")
            return None
        rest = txt[first_tc.start() :]

    tc_pattern = r"\s*<tool_call>\s*([\s\S]*?)\s*</tool_call>\s*"

    tool_calls = []
    while True:
        m_tc = re.match(tc_pattern, rest, flags=re.IGNORECASE)
        if not m_tc:
            break
        raw_json = m_tc.group(1)
        print(f"\033[93m[DEBUG extract] Raw tool call content: '{raw_json}'\033[0m")
        canon = canonicalise_tool_json(raw_json)
        if canon is None:
            print("\033[91m[DEBUG extract] Failed to canonicalise tool call\033[0m")
            return None
        print(f"\033[92m[DEBUG extract] Canonicalised: '{canon}'\033[0m")
        tool_calls.append(json.loads(canon))
        rest = rest[m_tc.end() :]

    if not tool_calls:
        print(
            f"\033[91m[DEBUG extract] No tool calls extracted from rest: '{rest[:150]}...'\033[0m"
        )
        return None

    # In strict mode, nothing should remain after tool calls
    # In flexible mode, allow trailing content (like explanations)
    if require_think_block and rest.strip():
        print(
            f"\033[91m[DEBUG extract] Rejected: trailing content after tool calls: '{rest[:100]}'\033[0m"
        )
        return None

    return tool_calls


def validate_think_only(txt: str) -> bool:
    """Validate a narration/summary turn (think block only, no tool calls)."""
    txt = normalize_tool_call_json(txt)
    if not isinstance(txt, str):
        return False

    think_blocks = re.findall(r"<think>[\s\S]*?</think>", txt, flags=re.IGNORECASE)
    if len(think_blocks) != 1:
        return False
    if not re.match(r"^\s*<think>", txt, flags=re.IGNORECASE):
        return False
    if re.search(r"<tool_call\s*>", txt, flags=re.IGNORECASE):
        return False
    if re.search(r"<tool_response\s*>", txt, flags=re.IGNORECASE):
        return False
    return True


def coerce_jsonlike(val):
    """Best-effort coercion of JSON-like values (handles double-encoding)."""
    if not isinstance(val, str):
        return val
    s = val.strip()
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    if s.lower() in ("null", "none"):
        return None
    if (s.startswith("{") and s.endswith("}")) or (
        s.startswith("[") and s.endswith("]")
    ):
        try:
            return json.loads(s)
        except Exception:
            pass
    try:
        return ast.literal_eval(s)
    except Exception:
        return val


def parse_expected_call(raw_call) -> dict:
    """Parse an expected tool call (handles double-encoding)."""
    obj = raw_call
    if isinstance(raw_call, str):
        try:
            obj = json.loads(raw_call)
        except Exception:
            obj = coerce_jsonlike(raw_call)
    if not isinstance(obj, dict):
        return {}
    if "arguments" in obj:
        obj["arguments"] = coerce_jsonlike(obj["arguments"])
    return obj


def json_objects_match(model_json, expected_json) -> bool:
    """Check if model output matches expected (expected is subset of model)."""
    model_json = coerce_jsonlike(model_json)
    expected_json = coerce_jsonlike(expected_json)

    if isinstance(expected_json, dict):
        if not isinstance(model_json, dict):
            return False
        for k, v in expected_json.items():
            if k not in model_json:
                return False
            if not json_objects_match(model_json[k], v):
                return False
        return True
    return model_json == expected_json


# ==============================================================================
# Base MT-GRPO Environment (Abstract methods for customization)
# ==============================================================================


class BaseMTGRPOEnv(BaseEnv):
    """
    Base class for Multi-Turn GRPO environments with turn-level credit assignment.

    Override these methods to customize for your use case:

        # Required overrides:
        - get_next_item(): Return training items
        - setup(): Initialize dataset and items

        # Reward customization (optional):
        - compute_turn_reward(): Custom turn-level reward (R^T_τ)
        - compute_outcome_reward(): Custom outcome reward (R^O)

        # Validation customization (optional):
        - validate_tool_call_turn(): Custom tool call validation
        - validate_summary_turn(): Custom summary validation

        # Prompt customization (optional):
        - get_system_prompt(): Custom system prompt
        - get_few_shot_example(): Custom few-shot example

    The MT-GRPO advantage computation is handled automatically:
        Â_τ = Â^T_τ + λ · Â^O
    """

    name = "base_mtgrpo"

    def __init__(
        self,
        config: MTGRPOEnvConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = True,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)

        self.percent_correct_buffer: List[float] = []
        self.raw_score_buffer: List[float] = []
        self.eval_metrics: List[Tuple[str, float]] = []
        self.rollouts_for_wandb: List = []
        self.dynamic_example: Optional[str] = None
        self.completed_tasks: set[str] = set()
        self.train_items: List = []
        self.test_items: List = []
        self.iter = 0
        # Set max_token_len from config (used by base class for length filtering)
        self.max_token_len = config.max_token_length

    # ==========================================================================
    # CUSTOMIZATION HOOKS - Override these in your subclass
    # ==========================================================================

    def get_system_prompt(self) -> str:
        """
        Return the system prompt for the environment.
        Override to customize the base instructions.
        """
        return DEFAULT_SYSTEM_PROMPT

    def get_few_shot_example(self) -> str:
        """
        Return a few-shot example for the prompt.
        Override to provide domain-specific examples.
        """
        if self.config.add_dynamic_few_shot and self.dynamic_example:
            return "Example Reasoning format:\n" + self.dynamic_example
        return DEFAULT_FEW_SHOT_EXAMPLE

    def get_sequential_helper(self) -> str:
        """Return helper text for sequential tool calling mode."""
        return DEFAULT_SEQ_TOOL_HELPER

    def get_narration_helper(self) -> str:
        """Return helper text for narration/summary turns."""
        return DEFAULT_NARRATION_HELPER

    def get_continuation_helper(self) -> str:
        """Return helper text for continuation after tool response."""
        return DEFAULT_CONTINUATION_HELPER

    def validate_tool_call_turn(
        self,
        response: str,
        expected_calls: List[str],
        pred_calls: List[Any],
    ) -> Tuple[bool, List[Any]]:
        """
        Validate a tool-calling turn and extract predictions.

        Args:
            response: The model's response text
            expected_calls: List of expected tool call JSON strings
            pred_calls: List to append predictions to

        Returns:
            (is_valid, updated_pred_calls)

        Override to implement custom tool call validation logic.
        """
        # Use config to determine if think blocks are required
        require_think = self.config.validate_think_blocks
        calls = extract_tool_calls(response, require_think_block=require_think)

        if calls is None:
            print(
                f"\033[91m[DEBUG] extract_tool_calls returned None (require_think={require_think})\033[0m"
            )
            return False, pred_calls + ["__MISMATCH__"]

        if len(calls) != len(expected_calls):
            print(
                f"\033[91m[DEBUG] Call count mismatch: got {len(calls)}, expected {len(expected_calls)}\033[0m"
            )
            return False, pred_calls + ["__MISMATCH__"]

        for mdl, exp_raw in zip(calls, expected_calls):
            exp_obj = parse_expected_call(exp_raw)
            if not json_objects_match(mdl, exp_obj):
                print(
                    f"\033[91m[DEBUG] JSON mismatch: model={mdl}, expected={exp_obj}\033[0m"
                )
                return False, pred_calls + ["__MISMATCH__"]

        return True, pred_calls + calls

    def validate_summary_turn(self, response: str) -> bool:
        """
        Validate a summary/narration turn (no tool calls expected).

        Args:
            response: The model's response text

        Returns:
            True if valid summary turn

        Override to implement custom summary validation.
        """
        # If think blocks not required, just check there are no tool calls
        if not self.config.validate_think_blocks:
            if re.search(r"<tool_call\s*>", response, flags=re.IGNORECASE):
                return False
            if re.search(r"<tool_response\s*>", response, flags=re.IGNORECASE):
                return False
            return True

        # Strict mode: require think block
        return validate_think_only(response)

    def compute_turn_reward(
        self,
        turn_idx: int,
        response: str,
        expected_calls: List[str],
        pred_calls: List[Any],
        is_valid: bool,
    ) -> float:
        """
        Compute the turn-level reward R^T_τ for a single turn.

        This is called for each turn to compute the intermediate reward signal.
        The paper uses rewards like:
            - Tool Execution Reward (0.5): Valid tool call structure
            - Tool Match Reward (0.5): Correct tool call content

        Args:
            turn_idx: Index of the current turn (0-based)
            response: The model's response text
            expected_calls: List of expected tool call JSON strings (empty for summary turns)
            pred_calls: List of predicted tool calls for this turn
            is_valid: Whether the turn passed validation

        Returns:
            Turn reward value (R^T_τ)

        Override to implement custom turn-level rewards.
        """
        reward = 0.0

        if expected_calls:  # Tool-calling turn
            # Check if structure is valid (has tool_call, optionally with think block)
            require_think = self.config.validate_think_blocks
            has_valid_structure = (
                extract_tool_calls(response, require_think_block=require_think)
                is not None
            )
            if has_valid_structure:
                reward += self.config.tool_execution_reward

            # Check if content matches
            if is_valid:
                reward += self.config.tool_match_reward

        else:  # Summary/narration turn
            if is_valid:
                reward += self.config.summary_reward

        # Apply penalty for mismatches
        if "__MISMATCH__" in pred_calls:
            reward += self.config.wrong_call_penalty

        return reward

    def compute_outcome_reward(
        self,
        turn_rewards: List[float],
        responses: List[str],
        expected_calls_by_turn: List[List[str]],
        completed_turns: int,
    ) -> float:
        """
        Compute the outcome-level reward R^O for the entire trajectory.

        This is the final success/failure signal for the episode.
        The paper uses binary outcome rewards:
            - 1.0 if all turns completed successfully
            - 0.0 otherwise

        Args:
            turn_rewards: List of turn rewards computed so far
            responses: List of model responses for each turn
            expected_calls_by_turn: Expected calls for each turn
            completed_turns: Number of turns that passed validation

        Returns:
            Outcome reward value (R^O)

        Override to implement custom outcome rewards (e.g., answer correctness).
        """
        total_turns = len(expected_calls_by_turn)
        all_successful = completed_turns == total_turns
        return 1.0 if all_successful else 0.0

    # ==========================================================================
    # MT-GRPO CORE LOGIC (Usually don't need to override)
    # ==========================================================================

    def _compute_turn_and_outcome_rewards(
        self,
        responses_by_turn: List[str],
        pred_calls_by_turn: List[List],
        expected_calls_by_turn: List[List[str]],
    ) -> Tuple[List[float], float, int]:
        """
        Compute all turn-level rewards and the outcome reward.

        Returns:
            (turn_rewards, outcome_reward, num_successful_turns)

        Calls the customizable compute_turn_reward() and compute_outcome_reward()
        hooks for each turn.
        """
        turn_rewards = []
        completed_turns = 0
        total_turns = len(expected_calls_by_turn)

        for turn_idx in range(len(responses_by_turn)):
            if turn_idx >= total_turns:
                break

            response = responses_by_turn[turn_idx]
            expected_calls = expected_calls_by_turn[turn_idx]
            pred_calls = (
                pred_calls_by_turn[turn_idx]
                if turn_idx < len(pred_calls_by_turn)
                else []
            )

            # Determine if turn is valid
            if expected_calls:
                is_valid = "__MISMATCH__" not in pred_calls and len(pred_calls) == len(
                    expected_calls
                )
            else:
                is_valid = self.validate_summary_turn(response)

            if is_valid:
                completed_turns += 1

            # Compute turn reward using customizable hook
            turn_reward = self.compute_turn_reward(
                turn_idx, response, expected_calls, pred_calls, is_valid
            )
            turn_rewards.append(turn_reward)

        # Fill missing turns with zero reward
        while len(turn_rewards) < total_turns:
            turn_rewards.append(0.0)

        # Compute outcome reward using customizable hook
        outcome_reward = self.compute_outcome_reward(
            turn_rewards, responses_by_turn, expected_calls_by_turn, completed_turns
        )

        return turn_rewards, outcome_reward, completed_turns

    def _compute_mt_grpo_advantages(
        self,
        turn_rewards_batch: List[List[float]],
        outcome_rewards_batch: List[float],
    ) -> List[List[float]]:
        """
        Compute MT-GRPO advantages following the paper (Equation 7):

            Â_τ = Â^T_τ + λ · Â^O   for ALL turns τ = 1, 2, ..., T

        Where:
            - Â^T_τ = (R^T_τ - mean) / std  (standardized turn advantage)
            - Â^O = (R^O - mean) / std  (standardized outcome advantage)
            - λ = turn_level_advantage_lambda

        This method should NOT be overridden - it implements the core MT-GRPO algorithm.
        """
        if not turn_rewards_batch:
            return []

        num_rollouts = len(turn_rewards_batch)
        max_turns = max(len(rewards) for rewards in turn_rewards_batch)
        lam = self.config.turn_level_advantage_lambda

        # Compute standardized turn advantages Â^T_τ for each turn position
        turn_advantages_by_turn = []
        for turn_idx in range(max_turns):
            turn_rewards = [
                rewards[turn_idx] if turn_idx < len(rewards) else 0.0
                for rewards in turn_rewards_batch
            ]
            mean_turn = np.mean(turn_rewards)
            std_turn = np.std(turn_rewards) or 1.0
            standardized = (np.array(turn_rewards) - mean_turn) / std_turn
            turn_advantages_by_turn.append(standardized)

        # Compute standardized outcome advantage Â^O
        mean_outcome = np.mean(outcome_rewards_batch)
        std_outcome = np.std(outcome_rewards_batch) or 1.0
        outcome_advantages = (
            np.array(outcome_rewards_batch) - mean_outcome
        ) / std_outcome

        # Combine: Â_τ = Â^T_τ + λ · Â^O
        mt_grpo_advantages = []
        for rollout_idx in range(num_rollouts):
            rollout_advantages = []
            actual_num_turns = len(turn_rewards_batch[rollout_idx])

            for turn_idx in range(actual_num_turns):
                A_T = turn_advantages_by_turn[turn_idx][rollout_idx]
                A_O = outcome_advantages[rollout_idx]
                adv = A_T + lam * A_O
                rollout_advantages.append(float(adv))

            mt_grpo_advantages.append(rollout_advantages)

        return mt_grpo_advantages

    def _assign_advantages_to_tokens(
        self,
        tokens: List[int],
        masks: List[int],
        turn_advantages: List[float],
    ) -> List[float]:
        """
        Assign turn-level advantages to tokens based on mask transitions.
        Tokens with mask != -100 are trainable (assistant responses).
        """
        token_advantages = [0.0] * len(tokens)

        if not turn_advantages:
            return token_advantages

        in_trainable_region = False
        current_turn_idx = 0
        current_advantage = turn_advantages[0] if turn_advantages else 0.0

        for i, mask in enumerate(masks):
            if mask != -100:
                if not in_trainable_region:
                    in_trainable_region = True
                    if current_turn_idx < len(turn_advantages):
                        current_advantage = turn_advantages[current_turn_idx]
                token_advantages[i] = current_advantage
            else:
                if in_trainable_region:
                    in_trainable_region = False
                    current_turn_idx += 1

        return token_advantages

    # ==========================================================================
    # TRAJECTORY COLLECTION (Core implementation)
    # ==========================================================================

    async def _build_turn_contexts(
        self,
        turn_idx: int,
        contexts: List[List[Dict[str, str]]],
        inter_turns: List[List[Dict[str, str]]],
        active: List[bool],
    ) -> Tuple[List[str], List[int]]:
        """Build prompts for the current turn from active rollout contexts."""
        if turn_idx > 0 and turn_idx - 1 < len(inter_turns):
            filler = inter_turns[turn_idx - 1]
            print(
                f"    \033[96m[DEBUG _build_turn_contexts] Adding inter_turn {turn_idx-1} to contexts:\033[0m"
            )
            for msg in filler:
                print(
                    f"      role={msg.get('role')}, content_preview={msg.get('content', '')[:200]}..."
                )
            for r in range(len(contexts)):
                if active[r]:
                    contexts[r].extend(filler)

        prompts, ridx_map = [], []
        for r in range(len(contexts)):
            if not active[r]:
                continue
            ptxt = self.tokenizer.apply_chat_template(
                contexts[r], add_generation_prompt=True, tokenize=False
            )
            prompts.append(ptxt)
            ridx_map.append(r)

        return prompts, ridx_map

    async def _execute_turn_inference(
        self,
        turn_idx: int,
        prompts: List[str],
        ridx_map: List[int],
    ) -> List[str]:
        """Execute inference for a turn."""
        if turn_idx == 0 and not self.config.use_parallel_requests:
            return await self._batch_identical_prompts(prompts[0], len(ridx_map))
        else:
            return await self._batch_heterogeneous_prompts(prompts)

    async def _batch_identical_prompts(self, prompt: str, count: int) -> List[str]:
        """Handle identical prompts using n parameter."""
        print(f"    \033[93m→ TURN 1 prompt full:\033[0m \033[92m{prompt}\033[0m")

        resp = await self.server.completion(
            prompt=prompt,
            n=count,
            max_tokens=self.config.max_gen_per_turn,
            temperature=0.8,
        )
        choices = [c.text for c in resp.choices]

        # Debug: print each rollout
        for i, raw in enumerate(choices):
            preview = raw[:1000] + ("..." if len(raw) > 1000 else "")
            print(
                f"    \033[93m· turn 1 rollout raw [{i}] (len={len(raw)}):\033[0m "
                f"\033[94m{preview}\033[0m"
            )
            if not raw.strip():
                print(f"      → (empty or error string returned for rollout {i})")
        print("    → All turn 1 rollouts printed; moving on.\n" + "-" * 48)

        return choices

    async def _batch_heterogeneous_prompts(self, prompts: List[str]) -> List[str]:
        """Handle heterogeneous prompts using parallel requests."""
        print(f"    → Parallelizing {len(prompts)} prompts")

        # Print each prompt - show more for Turn 2+ debugging
        for idx_p, p_str in enumerate(prompts):
            # Check if continuation helper is in the prompt
            has_continuation = "Continue with your reasoning" in p_str
            print(
                f"    \033[93m→ prompt[{idx_p}] (len={len(p_str)}, has_continuation_helper={has_continuation}):\033[0m"
            )
            # Print last 1500 chars to see the tool response and helper
            print(f"    \033[92m...{p_str[-1500:]}\033[0m")

        async def _call_single(prompt_str: str) -> str:
            try:
                comp = await self.server.completion(
                    prompt=prompt_str,
                    n=1,
                    max_tokens=self.config.max_gen_per_turn,
                    temperature=0.8,
                )
                return comp.choices[0].text
            except Exception as e:
                print(f"    → _call_single exception: {e}")
                return ""

        tasks = [_call_single(p) for p in prompts]
        results = await asyncio.gather(*tasks)

        # Debug: print results
        choices = []
        for i, rtext in enumerate(results):
            raw = rtext or ""
            print(
                f"    \033[93m· rollout {i} reply:\033[0m \033[94m{raw[:500]}{'...' if len(raw) > 500 else ''}\033[0m"
            )
            if not raw:
                print(f"    → Rollout {i} returned empty or error string")
            choices.append(raw)
        print("-" * 48)

        return choices

    async def _process_turn_responses(
        self,
        turn_idx: int,
        choices: List[str],
        ridx_map: List[int],
        contexts: List[List[Dict[str, str]]],
        preds_by_turn: List[List[List]],
        responses_by_turn: List[List[str]],
        active: List[bool],
        expected_calls_by_turn: List[List[str]],
    ) -> None:
        """Process and validate responses for a single turn."""
        for txt, r in zip(choices, ridx_map):
            raw_txt = txt or ""
            norm_txt = normalize_tool_call_json(raw_txt)

            print(f"\n\033[93m=== TURN {turn_idx+1} · ROLLOUT {r} ===\033[0m")
            preview = raw_txt[:1500] + ("..." if len(raw_txt) > 1500 else "")
            print(
                f"\033[95mRaw assistant reply (len={len(raw_txt)}):\033[0m\n"
                f"\033[94m{preview}\033[0m"
            )

            responses_by_turn[r].append(norm_txt)
            expected_calls = expected_calls_by_turn[turn_idx]

            if expected_calls:  # Tool-calling turn
                print(f"\033[95mExpected tool calls:\033[0m {expected_calls}")

                is_valid, updated_preds = self.validate_tool_call_turn(
                    norm_txt, expected_calls, preds_by_turn[r][turn_idx]
                )
                preds_by_turn[r][turn_idx] = updated_preds

                print(
                    f"\033[95mExtracted/validated:\033[0m {updated_preds}, valid={is_valid}"
                )

                if not is_valid:
                    print(
                        f"\033[91m[DEBUG] Tool call validation failed for rollout {r}\033[0m"
                    )
                    active[r] = False
                else:
                    # Update dynamic example on success
                    if self.config.add_dynamic_few_shot:
                        self.dynamic_example = norm_txt.strip()

            else:  # Summary turn
                is_valid = self.validate_summary_turn(norm_txt)
                print(f"\033[95mSummary turn validation:\033[0m valid={is_valid}")
                if not is_valid:
                    print(
                        f"\033[91m[DEBUG] Invalid summary turn for rollout {r}\033[0m"
                    )
                    active[r] = False
                preds_by_turn[r][turn_idx] = []

            contexts[r].append({"role": "assistant", "content": norm_txt})

    async def collect_trajectories(
        self,
        item: Tuple[Tuple[frozenset, ...], List[List[str]], List[List[Dict[str, str]]]],
    ) -> Tuple[Optional[ScoredDataGroup], List[Item]]:
        """
        Roll-out multi-turn tool-calling with MT-GRPO turn-level advantage computation.
        """
        messages_tuple, expected_calls_by_turn, inter_turns = item
        base_ctx = [dict(m) for m in messages_tuple]

        num_rollouts = self.config.group_size
        contexts: List[List[Dict[str, str]]] = [
            list(base_ctx) for _ in range(num_rollouts)
        ]
        preds_by_turn: List[List[List]] = [
            [[] for _ in range(len(expected_calls_by_turn))]
            for _ in range(num_rollouts)
        ]
        responses_by_turn: List[List[str]] = [[] for _ in range(num_rollouts)]
        active = [True] * num_rollouts

        max_turns = len(expected_calls_by_turn)
        print(f"[DEBUG] max_turns={max_turns}, len(inter_turns)={len(inter_turns)}")
        print(
            f"[DEBUG] expected_calls_by_turn count per turn: {[len(t) for t in expected_calls_by_turn]}"
        )
        if inter_turns:
            print(
                f"[DEBUG] inter_turns[0] roles: {[m.get('role') for m in inter_turns[0]] if inter_turns else 'empty'}"
            )

        for turn_idx in range(max_turns):
            print(
                f"\n[collect_trajectories] Beginning turn {turn_idx+1}/{max_turns} for this group"
            )

            prompts, ridx_map = await self._build_turn_contexts(
                turn_idx, contexts, inter_turns, active
            )

            if not prompts:
                print("    → No active prompts, stopping.")
                break

            choices = await self._execute_turn_inference(turn_idx, prompts, ridx_map)

            await self._process_turn_responses(
                turn_idx,
                choices,
                ridx_map,
                contexts,
                preds_by_turn,
                responses_by_turn,
                active,
                expected_calls_by_turn,
            )

            survivors = sum(1 for a in active if a)
            print(
                f"    → Finished turn {turn_idx+1}; {survivors}/{num_rollouts} rollouts still active"
            )

            if not any(active):
                print("    → All rollouts terminated; stopping further turns.")
                break

        # Compute rewards and advantages
        turn_rewards_batch = []
        outcome_rewards_batch = []
        successful_turns_batch = []

        for r in range(num_rollouts):
            turn_rewards, outcome_reward, num_successful = (
                self._compute_turn_and_outcome_rewards(
                    responses_by_turn[r], preds_by_turn[r], expected_calls_by_turn
                )
            )
            turn_rewards_batch.append(turn_rewards)
            outcome_rewards_batch.append(outcome_reward)
            successful_turns_batch.append(num_successful)

        # Compute MT-GRPO advantages
        mt_grpo_advantages = self._compute_mt_grpo_advantages(
            turn_rewards_batch, outcome_rewards_batch
        )

        # Build scored data group
        scored = ScoredDataGroup(tokens=[], masks=[], scores=[], advantages=[])

        for r in range(num_rollouts):
            out = tokenize_for_trainer(
                tokenizer=self.tokenizer,
                chat=contexts[r],
                include_messages=self.config.include_messages,
            )

            token_advantages = self._assign_advantages_to_tokens(
                out["tokens"],
                out["masks"],
                mt_grpo_advantages[r] if r < len(mt_grpo_advantages) else [],
            )

            # Compute final score for this rollout
            # Use dense reward (sum of turn rewards / total) + outcome bonus
            # This preserves partial successes for dataset generation
            total_turns = len(expected_calls_by_turn)
            dense_reward = sum(turn_rewards_batch[r]) / max(1, total_turns)
            final_score = dense_reward + outcome_rewards_batch[r]

            # Apply minimum successful turns threshold
            # Rollouts below threshold get negative score (will be filtered out)
            if successful_turns_batch[r] < self.config.min_successful_turns:
                final_score = -1.0

            scored["tokens"].append(out["tokens"])
            scored["masks"].append(out["masks"])
            scored["scores"].append(final_score)
            scored["advantages"].append(token_advantages)

        # Apply length penalty
        if scored["scores"]:
            cutoff = self.config.max_token_length * 0.5
            for i, ln in enumerate([len(t) for t in scored["tokens"]]):
                if ln > cutoff and scored["scores"][i] > 0.99:
                    frac = min(
                        (ln - cutoff) / (self.config.max_token_length - cutoff), 1.0
                    )
                    scored["scores"][i] = max(0.0, scored["scores"][i] - frac)

        # Update metrics
        for s in scored["scores"]:
            self.raw_score_buffer.append(s)
            self.percent_correct_buffer.append(1.0 if s >= 1.0 else 0.0)

        # Validate group
        if len(scored["tokens"]) < self.config.group_size:
            return None, []

        # For dataset generation: keep group if at least one rollout has positive score
        # For RL training: require score diversity (ensure_scores_are_not_same)
        has_positive_score = any(s > 0 for s in scored["scores"])
        if not has_positive_score:
            return None, []  # Drop groups where all rollouts failed

        if self.config.ensure_scores_are_not_same and len(set(scored["scores"])) == 1:
            return None, []

        # Final rollout summary
        print("\n\033[92m=== FINAL ROLLOUT SUMMARY ===\033[0m")
        for r_i, (ctx, score, num_success) in enumerate(
            zip(contexts, scored["scores"], successful_turns_batch)
        ):
            last_assistant = next(
                (m["content"] for m in reversed(ctx) if m["role"] == "assistant"),
                "(no assistant message)",
            )
            print(
                f"\n\033[96mRollout {r_i} · score={score:.3f} · successful_turns={num_success}\033[0m"
            )
            print(f"{last_assistant[:300]}{'...' if len(last_assistant) > 300 else ''}")
            print("-" * 60)
        print("=== END SUMMARY ===\n")

        await self.add_rollouts_for_wandb(scored, item)
        return scored, []

    # ==========================================================================
    # EVALUATION AND LOGGING
    # ==========================================================================

    def _score_episode(
        self,
        pred_calls: list,
        exp_calls: list,
        lam: float = 0.5,
    ) -> Tuple[float, int]:
        """Score an episode for evaluation."""
        exp_jsons = [parse_expected_call(r) for r in exp_calls]

        mismatch_penalty = 0.0
        if pred_calls and "__MISMATCH__" in pred_calls:
            pred_calls = [c for c in pred_calls if c != "__MISMATCH__"]
            mismatch_penalty = self.config.wrong_call_penalty

        pred_calls += [{}] * (len(exp_jsons) - len(pred_calls))

        correct = sum(
            1 for p, e in zip(pred_calls, exp_jsons) if json_objects_match(p, e)
        )
        dense = correct / max(1, len(exp_jsons))
        bonus = lam if correct == len(exp_jsons) else 0.0

        return dense + bonus + mismatch_penalty, correct

    async def rollout_and_score_eval(self, item) -> float:
        """Single evaluation rollout."""
        messages_tuple, expected_calls_by_turn, inter_turns = item
        base_ctx = [dict(m) for m in messages_tuple]
        ctx = list(base_ctx)
        preds = []

        for turn_idx, expected_turn_calls in enumerate(expected_calls_by_turn):
            if turn_idx > 0 and turn_idx - 1 < len(inter_turns):
                ctx.extend(inter_turns[turn_idx - 1])

            prompt = self.tokenizer.apply_chat_template(
                ctx, add_generation_prompt=True, tokenize=False
            )
            max_new = min(
                self.config.max_gen_per_turn,
                max(1, self.config.max_token_length - len(prompt)),
            )

            comp = await self.server.completion(
                prompt=prompt, n=1, max_tokens=max_new, temperature=0.0, split="eval"
            )
            reply = comp.choices[0].text
            ctx.append({"role": "assistant", "content": reply})

            tool_jsons = (
                extract_tool_calls(
                    reply, require_think_block=self.config.validate_think_blocks
                )
                if expected_turn_calls
                else []
            )
            if tool_jsons is None:
                break
            preds.extend(tool_jsons)

            if turn_idx >= len(expected_calls_by_turn) - 1:
                break

        expected_calls_flat = [
            call for turn_calls in expected_calls_by_turn for call in turn_calls
        ]
        score, _ = self._score_episode(preds, expected_calls_flat)
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
        self.eval_metrics.clear()
        await super().wandb_log(metrics)


# ==============================================================================
# Concrete Implementation: Tool-Calling Environment
# ==============================================================================


class MultiTurnToolUseTurnLevelAdvantageEnv(BaseMTGRPOEnv):
    """
    Concrete implementation of MT-GRPO for multi-turn tool-calling tasks.

    This environment:
    - Loads tool-calling conversation datasets
    - Validates tool calls against expected JSON
    - Computes turn-level rewards for tool execution and correctness
    - Uses MT-GRPO advantage estimation for fine-grained credit assignment

    To create your own environment, subclass BaseMTGRPOEnv and override:
    - setup(): Load your dataset
    - get_next_item(): Return training items
    - compute_turn_reward(): Custom turn rewards (optional)
    - compute_outcome_reward(): Custom outcome rewards (optional)
    """

    name = "multiturn_tool_use_turnlevel_advantage"

    def __init__(
        self,
        config: MTGRPOEnvConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = True,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)

        # Try local fixed dataset first, fall back to HuggingFace
        local_path = "data/hermes_reasoning_tool_use_fixed.jsonl"
        import os

        if os.path.exists(local_path):
            print(f"[dataset] Loading from local file: {local_path}")
            self.ds = load_dataset("json", data_files=local_path, split="train")
        else:
            print(
                "[dataset] Loading from HuggingFace: interstellarninja/hermes_reasoning_tool_use"
            )
            self.ds = load_dataset(
                "interstellarninja/hermes_reasoning_tool_use", split="train"
            )

    @classmethod
    def config_init(cls) -> Tuple[MTGRPOEnvConfig, List[APIServerConfig]]:
        env_cfg = MTGRPOEnvConfig(
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
            # MT-GRPO parameters
            turn_level_advantage_lambda=1.0,
            wrong_call_penalty=-0.2,
            tool_execution_reward=0.5,
            tool_match_reward=0.5,
            summary_reward=1.0,
            # Environment parameters
            max_tool_call_turns_cap=3,
            validate_think_blocks=True,  # Dataset has <think> blocks
            generate_all_gpt_turns=True,
            add_dynamic_few_shot=True,
            scenario_category="all",  # Accept all scenario types
            min_tool_call_turns=2,  # Require at least 2 tool-calling turns for MT-GRPO
            min_successful_turns=1,  # Keep rollouts with at least 1 successful turn
            use_parallel_requests=False,
            skip_completed=True,
            completed_dataset_id=None,
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
        """Initialize dataset and prepare training/test items."""
        ds = self.ds.shuffle()

        # Load completed tasks filter
        if self.config.skip_completed and self.config.completed_dataset_id:
            try:
                _done_ds = load_dataset(self.config.completed_dataset_id, split="train")
                self.completed_tasks = set(_done_ds["task"])
                print(f"[filter] Loaded {len(self.completed_tasks):,} completed tasks")
            except Exception as exc:
                self.completed_tasks = set()
                print(f"[filter] Could not load completed-task dataset: {exc}")

        # Statistics
        counts = Counter()
        for row in ds:
            conv = row["conversations"]
            num_turns = sum(
                1
                for msg in conv
                if msg["from"] in ("gpt", "assistant")
                and re.search(r"<tool_call>", msg["value"], re.IGNORECASE)
            )
            counts[num_turns] += 1

        print("Tool-call distribution:")
        for k in sorted(counts):
            print(f"  {k:2d} turns → {counts[k]} examples")

        split = ds.train_test_split(0.02)
        split["train"] = split["train"].shuffle()
        split["test"] = split["test"].shuffle()
        self._prep_items(split["train"], is_train=True)
        self._prep_items(split["test"], is_train=False)

        random.shuffle(self.train_items)
        random.shuffle(self.test_items)

        if not self.train_items:
            raise ValueError("No training items prepared: check dataset formatting.")

    def _check_sequential_tools(self, conv: List[Dict[str, str]]) -> bool:
        """Check if tool calls follow sequential pattern."""
        tool_indices = [
            i
            for i, m in enumerate(conv)
            if m["from"] in ("gpt", "assistant") and "<tool_call>" in m["value"].lower()
        ]
        if not tool_indices:
            return False

        for i in range(len(tool_indices) - 1):
            start, end = tool_indices[i], tool_indices[i + 1]
            in_between = conv[start + 1 : end]
            if any(m["from"] != "tool" for m in in_between):
                return False

        last_tool_idx = tool_indices[-1]
        next_responses = [
            i
            for i, m in enumerate(conv[last_tool_idx + 1 :], start=last_tool_idx + 1)
            if m["from"] == "tool"
        ]
        if not next_responses:
            return False

        return True

    def _prep_items(self, dataset, *, is_train: bool):
        """Process dataset items for training/testing."""
        target = self.train_items if is_train else self.test_items
        before_len = len(target)

        for row in dataset:
            conv = row["conversations"]

            if (
                len(conv) < 3
                or conv[0]["from"] != "system"
                or conv[1]["from"] != "human"
            ):
                continue

            if self.config.skip_completed and self.completed_tasks:
                if conv[1]["value"].strip() in self.completed_tasks:
                    continue

            tool_indices = [
                i
                for i, m in enumerate(conv)
                if m["from"] in ("gpt", "assistant")
                and "<tool_call>" in m["value"].lower()
            ]

            if not tool_indices:
                continue

            # Filter by minimum tool-calling turns
            if len(tool_indices) < self.config.min_tool_call_turns:
                continue

            # Validate scenario
            if self.config.scenario_category == "multistep":
                if len(tool_indices) < 2:
                    continue
                human_after_first_tool = any(
                    i > tool_indices[0] and m["from"] == "human"
                    for i, m in enumerate(conv)
                )
                if human_after_first_tool:
                    continue
                if not self._check_sequential_tools(conv):
                    continue
            elif self.config.scenario_category == "single":
                if len(tool_indices) != 1:
                    continue
            elif self.config.scenario_category == "multiturn":
                # ─── STRICT MULTI-TURN PATTERN ───
                # User → Assistant(tool_call) → Tool → Assistant(summary) → User → ...
                # Each tool call is followed by a summary, then a user turn before next tool call

                human_after_first_tool = any(
                    i > tool_indices[0] and m["from"] == "human"
                    for i, m in enumerate(conv)
                )

                # Must have at least one human after the first tool-call
                if not human_after_first_tool:
                    continue

                # Must have at least TWO tool-calling turns for true multiturn
                if len(tool_indices) < 2:
                    continue

                # First assistant turn must be the first tool-calling message
                first_asst_idx = next(
                    (
                        i
                        for i, m in enumerate(conv[2:], start=2)
                        if m["from"] in ("gpt", "assistant")
                    ),
                    None,
                )
                if first_asst_idx != tool_indices[0]:
                    continue

                # Build multiturn structure
                expected_calls_by_turn = []
                inter_turns = []
                ok = True

                for idx_t, tool_idx in enumerate(tool_indices):
                    # 1. Tool response directly after the tool-call
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

                    # 3. If another tool-call follows, ensure a human turn exists
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

                    # ─── Build turn A: assistant tool-call ───
                    tool_call_msg = conv[tool_idx]["value"]
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
                    turn_calls = []
                    for raw in tc_raws:
                        canon = canonicalise_tool_json(raw)
                        if canon is None:
                            ok = False
                            break
                        turn_calls.append(canon)
                    if not ok:
                        break
                    expected_calls_by_turn.append(turn_calls)

                    # Inter-turn after tool-call → tool response + narration helper
                    tool_content = strip_tool_response_tags(
                        conv[tool_resp_idx]["value"]
                    )
                    inter_turns.append(
                        [
                            {
                                "role": "tool",
                                "content": tool_content
                                + "\n\n"
                                + self.get_narration_helper(),
                            }
                        ]
                    )

                    # ─── Build turn B: assistant narration (no calls) ───
                    expected_calls_by_turn.append([])

                    # Inter-turn after narration → up to next tool-call (user messages, etc.)
                    slice_end = nxt_tool_idx if nxt_tool_idx is not None else len(conv)
                    next_ctx_slice = [
                        {
                            "role": "user" if m["from"] == "human" else "assistant",
                            "content": m["value"],
                        }
                        for m in conv[summ_idx + 1 : slice_end]
                    ]
                    inter_turns.append(next_ctx_slice)

                if not ok:
                    continue

                # Remove trailing inter-turn (nothing after final narration)
                if inter_turns:
                    inter_turns.pop()

                # Apply turn cap
                cap = self.config.max_tool_call_turns_cap
                if cap is not None:
                    keep_turns = 0
                    calls_seen = 0
                    for idx, calls in enumerate(expected_calls_by_turn):
                        keep_turns += 1
                        if calls:
                            calls_seen += 1
                            if calls_seen == cap:
                                if idx + 1 < len(expected_calls_by_turn):
                                    keep_turns += 1
                                break
                    expected_calls_by_turn = expected_calls_by_turn[:keep_turns]
                    inter_turns = inter_turns[: keep_turns - 1]

                # Build system prompt for multiturn
                running_msgs = []
                dataset_system = conv[0]["value"]
                combined_system = dataset_system + "\n\n" + self.get_few_shot_example()
                running_msgs.append(
                    frozenset({"role": "system", "content": combined_system}.items())
                )
                running_msgs.append(
                    frozenset({"role": "user", "content": conv[1]["value"]}.items())
                )

                if len(expected_calls_by_turn) >= 2:
                    target.append(
                        (tuple(running_msgs), expected_calls_by_turn, inter_turns)
                    )
                continue  # multiturn handled

            elif self.config.scenario_category == "all":
                # Accept all scenarios with at least one tool call
                if len(tool_indices) < 1:
                    continue
            else:
                continue

            # Build system prompt
            running_msgs = []
            # Use dataset's system prompt directly (already fixed with proper JSON syntax)
            # Append our few-shot example for reinforcement
            dataset_system = conv[0]["value"]
            combined_system = dataset_system + "\n\n" + self.get_few_shot_example()
            running_msgs.append(
                frozenset({"role": "system", "content": combined_system}.items())
            )

            user_content = conv[1]["value"]
            if self.config.scenario_category == "multistep":
                user_content = f"{user_content}\n\n{self.get_sequential_helper()}"
            running_msgs.append(
                frozenset({"role": "user", "content": user_content}.items())
            )

            # Extract expected calls and inter-turns
            expected_calls_by_turn = []
            inter_turns = []
            skip_conversation = False

            for i, tool_idx in enumerate(tool_indices):
                tool_call_msg = conv[tool_idx]["value"]

                # Validate think blocks if required (skip entire conversation if missing)
                if self.config.validate_think_blocks and not re.match(
                    r"^\s*<think>", tool_call_msg, flags=re.IGNORECASE
                ):
                    skip_conversation = True
                    break

                matches = re.findall(
                    r"<tool_call>\s*(.*?)\s*</tool_call>",
                    tool_call_msg,
                    re.DOTALL | re.IGNORECASE,
                )
                if not matches:
                    skip_conversation = True
                    break

                turn_calls = []
                for raw in matches:
                    canon = canonicalise_tool_json(raw)
                    if canon:
                        turn_calls.append(canon)

                if not turn_calls:
                    skip_conversation = True
                    break

                expected_calls_by_turn.append(turn_calls)

                # Build inter-turn messages (everything between this tool call and the next)
                if i < len(tool_indices) - 1:
                    next_tool_idx = tool_indices[i + 1]
                    inter_msgs = []
                    for msg_idx in range(tool_idx + 1, next_tool_idx):
                        msg = conv[msg_idx]
                        role = (
                            "tool"
                            if msg["from"] == "tool"
                            else ("user" if msg["from"] == "human" else "assistant")
                        )
                        content = msg["value"]
                        # Strip outer <tool_response> tags if present (dataset may have them)
                        if role == "tool":
                            content = strip_tool_response_tags(content)
                        # Add continuation helper to tool responses to remind model about format
                        if role == "tool" and msg_idx == tool_idx + 1:
                            content = content + "\n\n" + self.get_continuation_helper()
                        inter_msgs.append({"role": role, "content": content})
                    if inter_msgs:
                        inter_turns.append(inter_msgs)

            if skip_conversation:
                continue

            # Handle final summary turn
            if self.config.generate_all_gpt_turns and tool_indices:
                last_tool_response_idx = tool_indices[-1] + 1
                has_final_narration = (
                    last_tool_response_idx + 1 < len(conv)
                    and conv[last_tool_response_idx + 1]["from"] in ("gpt", "assistant")
                    and "<tool_call>" not in conv[last_tool_response_idx + 1]["value"]
                )
                if has_final_narration:
                    expected_calls_by_turn.append([])
                    # Strip outer <tool_response> tags if present
                    tool_content = strip_tool_response_tags(
                        conv[last_tool_response_idx]["value"]
                    )
                    inter_turns.append(
                        [
                            {
                                "role": "tool",
                                "content": tool_content
                                + "\n\n"
                                + self.get_narration_helper(),
                            }
                        ]
                    )

            # Apply turn cap
            cap = self.config.max_tool_call_turns_cap
            if cap is not None:
                keep_turns = 0
                calls_seen = 0
                for idx, calls in enumerate(expected_calls_by_turn):
                    keep_turns += 1
                    if calls:
                        calls_seen += 1
                        if calls_seen == cap:
                            if self.config.generate_all_gpt_turns and idx + 1 < len(
                                expected_calls_by_turn
                            ):
                                keep_turns += 1
                            break
                expected_calls_by_turn = expected_calls_by_turn[:keep_turns]
                inter_turns = inter_turns[: keep_turns - 1]

            # Add item - require at least one turn with expected calls
            if len(expected_calls_by_turn) >= 1:
                if (
                    self.config.scenario_category == "multistep"
                    and len(expected_calls_by_turn) >= 2
                ):
                    target.append(
                        (tuple(running_msgs), expected_calls_by_turn, inter_turns)
                    )
                elif self.config.scenario_category in ("single", "multiturn", "all"):
                    target.append(
                        (tuple(running_msgs), expected_calls_by_turn, inter_turns)
                    )

        print(
            f"[prep_items] {'train' if is_train else 'test'}: added {len(target) - before_len} items."
        )

    async def get_next_item(self):
        """Return the next training item."""
        if not self.train_items:
            raise ValueError("train_items is empty – dataset preparation failed.")

        if self.iter >= len(self.train_items):
            random.shuffle(self.train_items)
            self.iter = 0

        itm = self.train_items[self.iter]
        self.iter += 1
        return itm


if __name__ == "__main__":
    MultiTurnToolUseTurnLevelAdvantageEnv.cli()
