"""
Core T1 tool planning logic — extracted for testability.

These functions do the actual work: generating tool-calling completions
via ManagedServer and scoring them. The env class just orchestrates.

Two modes:
  - Single-turn: generate_tool_completions + score_completions
  - Multi-step: collect_multistep_trajectory walks a full conversation,
    feeding the model's actual responses back at each turn.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from t1_prompts import SYSTEM_PROMPT
from t1_scoring import score_turn
from t1_tools import T1_TOOLS

from atroposlib.envs.base import ScoredDataGroup
from atroposlib.envs.server_handling.managed_server import SequenceNode
from atroposlib.envs.server_handling.server_manager import ServerManager

logger = logging.getLogger(__name__)


async def generate_tool_completions(
    server: ServerManager,
    tokenizer: Any,
    messages: List[Dict[str, str]],
    tools: List[dict] = None,
    n: int = 4,
    max_tokens: int = 512,
    temperature: float = 1.0,
    tool_choice: str = "auto",
    split: str = "train",
    tool_parser: str = "hermes",
) -> Tuple[Any, List[SequenceNode]]:
    """Generate tool-calling completions and return result + tracked nodes.

    Args:
        server: ServerManager with backends configured
        tokenizer: Tokenizer for the model
        messages: Chat messages to complete
        tools: OpenAI function tool definitions (defaults to T1_TOOLS)
        n: Number of completions to generate
        max_tokens: Max tokens per completion
        temperature: Sampling temperature
        tool_choice: "auto", "none", or "required"
        split: "train" or "eval" (for server load balancing)
        tool_parser: vLLM tool parser name

    Returns:
        (ChatCompletion, list of SequenceNodes)
    """
    if tools is None:
        tools = T1_TOOLS

    logger.info(
        f"generate_tool_completions: n={n}, max_tokens={max_tokens}, "
        f"temp={temperature}, tool_choice={tool_choice}, "
        f"num_messages={len(messages)}"
    )

    async with server.managed_server(tokenizer=tokenizer) as managed:
        logger.debug(
            f"  ManagedServer opened (tool_parser={managed._tool_parser_name})"
        )

        result = await managed.chat_completion(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            split=split,
        )

        logger.debug(f"  Got {len(result.choices)} choices")
        for i, c in enumerate(result.choices):
            tc_count = len(c.message.tool_calls) if c.message.tool_calls else 0
            content_preview = (c.message.content or "")[:80]
            logger.debug(
                f"    choice[{i}]: {tc_count} tool_calls, content={content_preview!r}"
            )

        state = managed.get_state()
        nodes = state["nodes"]
        logger.debug(f"  Got {len(nodes)} tracked nodes")

    return result, nodes


def score_completions(
    result: Any,
    nodes: List[SequenceNode],
    gt_code: str,
    min_unmasked_tokens: int = 5,
) -> Tuple[Optional[ScoredDataGroup], List[Dict[str, float]]]:
    """Score completions against ground truth and build a ScoredDataGroup.

    Args:
        result: ChatCompletion from generate_tool_completions
        nodes: SequenceNodes from generate_tool_completions
        gt_code: Ground truth Python code (Filled_Plan)
        min_unmasked_tokens: Skip choices with fewer unmasked tokens

    Returns:
        (ScoredDataGroup or None, list of per-choice score dicts)
    """
    logger.debug(
        f"score_completions: {len(result.choices)} choices, "
        f"{len(nodes)} nodes, gt_code={gt_code[:60]}..."
    )

    all_scores = []
    scores = ScoredDataGroup()
    scores["tokens"] = []
    scores["masks"] = []
    scores["scores"] = []
    scores["inference_logprobs"] = []

    for i, (choice, node) in enumerate(zip(result.choices, nodes)):
        turn_scores = score_turn(
            gt_code, choice.message.tool_calls, choice.message.content
        )
        all_scores.append(turn_scores)
        logger.debug(f"  choice[{i}] scores: {turn_scores}")

        unmasked = len([t for t in node.masked_tokens if t != -100])
        if unmasked < min_unmasked_tokens:
            logger.debug(f"  choice[{i}] skipped: only {unmasked} unmasked tokens")
            continue

        scores["tokens"].append(node.tokens)
        scores["masks"].append(node.masked_tokens)
        scores["inference_logprobs"].append(node.logprobs)
        scores["scores"].append(turn_scores["reward"])

    if not scores["tokens"]:
        logger.debug("  -> None (no valid tokens)")
        return None, all_scores

    if all(s == scores["scores"][0] for s in scores["scores"]):
        logger.debug(f"  -> None (all scores identical: {scores['scores'][0]})")
        return None, all_scores

    logger.debug(f"  -> valid group, scores={scores['scores']}")
    return scores, all_scores


async def collect_multistep_trajectory(
    server: ServerManager,
    tokenizer: Any,
    conversation: List[Dict[str, str]],
    tools: List[dict] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    tool_choice: str = "auto",
    tool_parser: str = "hermes",
) -> Tuple[List[Dict[str, Any]], List[SequenceNode]]:
    """Walk through a full conversation in ONE managed_server session.

    Uses a single ManagedServer context across all turns so sequence tracking
    works properly — each turn extends the previous node, building up a full
    multi-turn trajectory with aligned tokens and logprobs.

    At each user turn:
      1. Add the user message to the running conversation
      2. Generate a model response (n=1) via the SAME managed server
      3. Score against ground truth
      4. Add the model's ACTUAL response (not GT) to conversation history
      5. Continue to next turn regardless of quality

    Nodes are collected ONCE at the end from managed.get_state().

    Args:
        server: ServerManager with backends configured
        tokenizer: Tokenizer for the model
        conversation: List of turn dicts with Role, Filled_Template, Filled_Plan
        tools: Tool definitions (defaults to T1_TOOLS)
        max_tokens: Max tokens per completion
        temperature: Sampling temperature
        tool_choice: "auto", "none", or "required"
        tool_parser: vLLM tool parser name

    Returns:
        (turn_results, nodes) where:
          turn_results: list of per-turn dicts with scores, tool_calls, content
          nodes: list of SequenceNodes from the managed server (one per turn,
                 each extending the previous — full trajectory with tokens/logprobs)
    """
    if tools is None:
        tools = T1_TOOLS

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    turn_results = []

    logger.info(
        f"collect_multistep_trajectory: {len(conversation)} turns, temp={temperature}"
    )

    async with server.managed_server(
        tokenizer=tokenizer, preserve_think_blocks=True
    ) as managed:
        for i, turn in enumerate(conversation):
            role = turn["Role"].strip().lower()

            if role == "assistant":
                if not turn_results:
                    # First assistant turn (greeting) — use GT since model hasn't spoken yet
                    messages.append(
                        {"role": "assistant", "content": turn["Filled_Template"]}
                    )
                    logger.debug(
                        f"  turn[{i}] assistant (GT greeting): {turn['Filled_Template'][:60]}"
                    )
                # Otherwise skip — we already added the model's response after the previous user turn
                continue

            if role != "user":
                continue

            # User turn — add to conversation and generate model response
            messages.append({"role": "user", "content": turn["Filled_Template"]})
            gt_code = turn.get("Filled_Plan", "")

            logger.info(f"  turn[{i}] user: {turn['Filled_Template'][:60]}...")
            logger.debug(f"  turn[{i}] gt_code: {gt_code[:80]}")

            # Generate within the SAME managed server session
            result = await managed.chat_completion(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                n=1,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            choice = result.choices[0]

            # Score this turn
            turn_scores = score_turn(
                gt_code, choice.message.tool_calls, choice.message.content
            )

            tc_count = (
                len(choice.message.tool_calls) if choice.message.tool_calls else 0
            )
            logger.info(
                f"  turn[{i}] result: {tc_count} tool_calls, "
                f"reward={turn_scores['reward']:.2f}, tc_f1={turn_scores['tool_call_f1']:.2f}"
            )

            turn_results.append(
                {
                    "turn_idx": i,
                    "user_message": turn["Filled_Template"],
                    "gt_code": gt_code,
                    "content": choice.message.content,
                    "tool_calls": choice.message.tool_calls,
                    "scores": turn_scores,
                    "messages_so_far": [m.copy() for m in messages],
                }
            )

            # Add model's ACTUAL response to conversation for next turn
            assistant_msg = {"role": "assistant"}
            if choice.message.tool_calls:
                assistant_msg["tool_calls"] = choice.message.tool_calls
                assistant_msg["content"] = choice.message.content or ""
            else:
                assistant_msg["content"] = choice.message.content or ""
            messages.append(assistant_msg)

            logger.debug(
                f"  turn[{i}] added assistant msg to conversation (total: {len(messages)})"
            )

        # Get nodes ONCE at the end — the managed server tracked extending sequences
        state = managed.get_state()
        nodes = state["nodes"]
        logger.info(
            f"  trajectory complete: {len(turn_results)} turns, {len(nodes)} nodes"
        )

    # Summary
    if turn_results:
        avg_reward = sum(r["scores"]["reward"] for r in turn_results) / len(
            turn_results
        )
        avg_tc_f1 = sum(r["scores"]["tool_call_f1"] for r in turn_results) / len(
            turn_results
        )
        logger.info(f"  avg_reward={avg_reward:.3f}, avg_tc_f1={avg_tc_f1:.3f}")

    return turn_results, nodes
