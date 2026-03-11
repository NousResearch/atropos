"""
Scoring for T1 tool planning environment.

Parses ground truth Python code with AST to extract tool calls,
then compares against structured tool_calls from the model response.
"""

import ast
import json
import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Tools that T1 considers "real" (not just print or cache ops)
SEARCH_FILTER_TOOLS = {
    "search_hotels",
    "filter_hotels",
    "search_flights",
    "filter_flights",
    "search_restaurants",
    "filter_restaurants",
    "search_attractions",
    "filter_attractions",
    "search_nearest",
    "sort_results",
    "adjust_date",
    "seek_information",
}

CACHE_TOOLS = {"save_to_cache", "get_results_from_cache"}
ALL_TOOLS = SEARCH_FILTER_TOOLS | CACHE_TOOLS


def parse_ground_truth_code(code: str) -> List[Dict[str, Any]]:
    """Parse ground truth Python code into a list of tool calls.

    Args:
        code: Python code string from Filled_Plan column.

    Returns:
        List of {"name": str, "arguments": dict} dicts.
    """
    if not code or not isinstance(code, str):
        return []

    code = code.strip()
    if not code or code == 'print("No planning needed")':
        return []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        logger.warning("Failed to parse ground truth code: %s", code[:100])
        return []

    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Get function name
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            else:
                continue

            if func_name not in ALL_TOOLS:
                continue

            # Extract keyword arguments
            args = {}
            for kw in node.keywords:
                if kw.arg is None:
                    continue
                try:
                    args[kw.arg] = ast.literal_eval(kw.value)
                except (ValueError, TypeError):
                    # For variable references (like prior_results=hotels),
                    # store as string
                    if isinstance(kw.value, ast.Name):
                        args[kw.arg] = kw.value.id
                    else:
                        args[kw.arg] = ast.dump(kw.value)

            calls.append({"name": func_name, "arguments": args})

    return calls


def parse_model_tool_calls(tool_calls: Optional[List[dict]]) -> List[Dict[str, Any]]:
    """Normalize model's structured tool_calls into comparable format.

    Args:
        tool_calls: List of tool call dicts from ChatCompletion response.

    Returns:
        List of {"name": str, "arguments": dict} dicts.
    """
    if not tool_calls:
        return []

    result = []
    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "")
        args_str = func.get("arguments", "{}")
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except (json.JSONDecodeError, TypeError):
            args = {}
        result.append({"name": name, "arguments": args})
    return result


def tool_call_f1(
    ground_truth: List[Dict[str, Any]],
    generated: List[Dict[str, Any]],
) -> Tuple[float, float, float]:
    """Compute precision, recall, F1 on tool names.

    Compares the multiset of tool names called.
    """
    gt_names = Counter(c["name"] for c in ground_truth)
    gen_names = Counter(c["name"] for c in generated)

    # Remove "print" if other tools exist
    if len(gt_names) > 1:
        gt_names.pop("print", None)
    if len(gen_names) > 1:
        gen_names.pop("print", None)

    if not gt_names and not gen_names:
        return 1.0, 1.0, 1.0  # both empty = correct

    tp = sum((gt_names & gen_names).values())
    precision = tp / sum(gen_names.values()) if gen_names else 0.0
    recall = tp / sum(gt_names.values()) if gt_names else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return precision, recall, f1


def tool_param_f1(
    ground_truth: List[Dict[str, Any]],
    generated: List[Dict[str, Any]],
) -> Tuple[float, float, float]:
    """Compute precision, recall, F1 on tool parameters.

    For each matching tool name pair, compares the argument keys and values.
    """
    if not ground_truth and not generated:
        return 1.0, 1.0, 1.0

    # Match tool calls by name (greedy matching)
    gt_remaining = list(ground_truth)
    matched_pairs = []

    for gen_call in generated:
        for i, gt_call in enumerate(gt_remaining):
            if gt_call["name"] == gen_call["name"]:
                matched_pairs.append((gt_call, gen_call))
                gt_remaining.pop(i)
                break

    if not matched_pairs:
        return 0.0, 0.0, 0.0

    total_tp = 0
    total_gt_params = 0
    total_gen_params = 0

    for gt_call, gen_call in matched_pairs:
        gt_args = gt_call.get("arguments", {})
        gen_args = gen_call.get("arguments", {})

        total_gt_params += len(gt_args)
        total_gen_params += len(gen_args)

        for key in gt_args:
            if key in gen_args:
                # Loose comparison — normalize types
                gt_val = gt_args[key]
                gen_val = gen_args[key]
                if _values_match(gt_val, gen_val):
                    total_tp += 1

    precision = total_tp / total_gen_params if total_gen_params > 0 else 0.0
    recall = total_tp / total_gt_params if total_gt_params > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return precision, recall, f1


def _values_match(gt_val: Any, gen_val: Any) -> bool:
    """Loose comparison of argument values."""
    if gt_val == gen_val:
        return True

    # String comparison (case-insensitive)
    if isinstance(gt_val, str) and isinstance(gen_val, str):
        return gt_val.lower().strip() == gen_val.lower().strip()

    # List comparison (order-insensitive for some)
    if isinstance(gt_val, list) and isinstance(gen_val, list):
        if len(gt_val) == len(gen_val):
            return sorted(str(v) for v in gt_val) == sorted(str(v) for v in gen_val)

    # Number comparison
    try:
        if float(gt_val) == float(gen_val):
            return True
    except (ValueError, TypeError):
        pass

    return False


def score_turn(
    ground_truth_code: str,
    model_tool_calls: Optional[List[dict]],
    model_content: Optional[str] = None,
) -> Dict[str, float]:
    """Score a single turn.

    Args:
        ground_truth_code: Python code from Filled_Plan column.
        model_tool_calls: Structured tool_calls from model response.
        model_content: Text content from model response (for seek_information).

    Returns:
        Dict with scores: tool_call_f1, tool_param_f1, correct_no_op, reward
    """
    gt_calls = parse_ground_truth_code(ground_truth_code)
    gen_calls = parse_model_tool_calls(model_tool_calls)

    # Handle "no planning needed" case
    gt_is_noop = len(gt_calls) == 0
    gen_is_noop = len(gen_calls) == 0

    if gt_is_noop and gen_is_noop:
        return {
            "tool_call_f1": 1.0,
            "tool_param_f1": 1.0,
            "correct_no_op": 1.0,
            "reward": 1.0,
        }

    if gt_is_noop and not gen_is_noop:
        return {
            "tool_call_f1": 0.0,
            "tool_param_f1": 0.0,
            "correct_no_op": 0.0,
            "reward": 0.0,
        }

    # Check if this is a seek_information turn
    gt_is_seek = any(c["name"] == "seek_information" for c in gt_calls)

    tc_p, tc_r, tc_f1 = tool_call_f1(gt_calls, gen_calls)
    tp_p, tp_r, tp_f1 = tool_param_f1(gt_calls, gen_calls)

    # Composite reward — graduated so GRPO gets signal even with weak models
    #
    # GT expects tools but model produced none → 0.0 (worst)
    # GT expects tools, model called tools but wrong ones → 0.1 (format credit)
    # GT expects tools, model called some right ones → 0.1 + 0.5*tc_f1 + 0.3*tp_f1
    # Perfect match → 0.1 + 0.5 + 0.3 + 0.1 = 1.0
    if not gt_is_noop and gen_is_noop:
        # GT expects tool calls but model produced none
        reward = 0.0
    else:
        # Model attempted tool calls
        reward = 0.1  # format credit: produced valid tool call structure
        reward += 0.5 * tc_f1
        reward += 0.3 * tp_f1
        # Bonus for getting all tools right
        if tc_f1 == 1.0:
            reward += 0.1

    return {
        "tool_call_precision": tc_p,
        "tool_call_recall": tc_r,
        "tool_call_f1": tc_f1,
        "tool_param_precision": tp_p,
        "tool_param_recall": tp_r,
        "tool_param_f1": tp_f1,
        "correct_no_op": 0.0,
        "is_seek_info": float(gt_is_seek),
        "reward": reward,
    }
