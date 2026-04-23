#!/usr/bin/env python3
"""
Live test for T1 tool planning — runs against an already-running vLLM server.

No pytest fixtures, no subprocess spawning. Just creates a ServerManager
pointed at localhost:9001, calls generate_tool_completions, and prints results.

Usage:
    # With vLLM already running on port 9001:
    python environments/t1_tool_planning/test_t1_live.py

    # Custom port:
    python environments/t1_tool_planning/test_t1_live.py --port 8123

    # Custom model:
    python environments/t1_tool_planning/test_t1_live.py --model Qwen/Qwen3-4B
"""

import argparse
import asyncio
import json
import logging
import os
import sys

# Ensure t1 modules are importable
sys.path.insert(0, os.path.dirname(__file__))

from t1_core import generate_tool_completions, score_completions  # noqa: E402
from t1_prompts import SYSTEM_PROMPT  # noqa: E402
from t1_scoring import score_turn  # noqa: E402
from t1_tools import T1_TOOLS  # noqa: E402

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("test_t1_live")


def make_server_manager(model_name: str, base_url: str):
    """Create a ServerManager pointed at an existing vLLM server."""
    from atroposlib.envs.server_handling.server_baseline import APIServerConfig
    from atroposlib.envs.server_handling.server_manager import ServerManager

    config = APIServerConfig(
        model_name=model_name,
        base_url=base_url,
        api_key="x",
        server_type="vllm",
    )
    server = ServerManager(
        configs=[config],
        slurm=False,
        testing=False,
        tool_parser="hermes",
    )
    return server


def make_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


SAMPLE_CONVERSATIONS = {
    1: [
        {
            "Role": "assistant",
            "Filled_Template": "Hello! I'm your travel assistant. How can I help you today?",
            "Filled_Plan": "",
        },
        {
            "Role": "user",
            "Filled_Template": "I'm looking for hotels in Austin with check-in on May 10, 2025 and check-out on May 15, 2025.",  # noqa: E501
            "Filled_Plan": 'hotels = search_hotels(city="Austin", checkin_date=["May 10, 2025"], checkout_date=["May 15, 2025"])\nsave_to_cache(key="hotels", value=hotels)',  # noqa: E501
        },
    ],
    2: [
        {
            "Role": "assistant",
            "Filled_Template": "Welcome! What can I help you plan?",
            "Filled_Plan": "",
        },
        {
            "Role": "user",
            "Filled_Template": "I need a hotel in New York but I'm not sure about dates yet.",
            "Filled_Plan": 'seek_information("We need to ask for the check-in and check-out dates")',
        },
    ],
    3: [
        {
            "Role": "assistant",
            "Filled_Template": "Hi there! Looking for travel help?",
            "Filled_Plan": "",
        },
        {
            "Role": "user",
            "Filled_Template": "No that's perfect, thanks!",
            "Filled_Plan": 'print("No planning needed")',
        },
    ],
}


def build_messages(conversation: list, turn_index: int) -> list:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for i, turn in enumerate(conversation):
        if i > turn_index:
            break
        role = turn["Role"].strip().lower()
        messages.append({"role": role, "content": turn["Filled_Template"]})
    return messages


async def test_single_completion(server, tokenizer):
    """Test 1: Single completion with tool calling."""
    print("\n" + "=" * 60)
    print("TEST 1: Single tool-calling completion")
    print("=" * 60)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Find me hotels in Austin, checking in May 10 and out May 15, 2025.",
        },
    ]

    result, nodes = await generate_tool_completions(
        server=server,
        tokenizer=tokenizer,
        messages=messages,
        tools=T1_TOOLS,
        n=1,
        max_tokens=500,
        temperature=0.0,
        tool_choice="auto",
    )

    choice = result.choices[0]
    print(f"\nContent: {choice.message.content}")
    print(f"Tool calls: {choice.message.tool_calls}")
    print(f"Finish reason: {choice.finish_reason}")
    print(f"Nodes tracked: {len(nodes)}")

    if nodes:
        node = nodes[0]
        print(f"Token count: {len(node.tokens)}")
        unmasked = len([t for t in node.masked_tokens if t != -100])
        print(f"Unmasked tokens: {unmasked}")
        print(f"Logprobs sample: {node.logprobs[-5:]}")

    # Score against ground truth
    gt_code = 'hotels = search_hotels(city="Austin", checkin_date=["May 10, 2025"], checkout_date=["May 15, 2025"])\nsave_to_cache(key="hotels", value=hotels)'  # noqa: E501
    scores = score_turn(gt_code, choice.message.tool_calls, choice.message.content)
    print(f"\nScores: {json.dumps(scores, indent=2)}")

    return True


async def test_group_completions(server, tokenizer):
    """Test 2: Multiple completions (group_size=4) for GRPO."""
    print("\n" + "=" * 60)
    print("TEST 2: Group completions (n=4) for GRPO")
    print("=" * 60)

    convo = SAMPLE_CONVERSATIONS[1]
    messages = build_messages(convo, turn_index=1)
    gt_code = convo[1]["Filled_Plan"]

    result, nodes = await generate_tool_completions(
        server=server,
        tokenizer=tokenizer,
        messages=messages,
        tools=T1_TOOLS,
        n=4,
        max_tokens=500,
        temperature=1.0,
        tool_choice="auto",
    )

    print(f"\nGot {len(result.choices)} choices, {len(nodes)} nodes")

    for i, choice in enumerate(result.choices):
        tc_count = len(choice.message.tool_calls) if choice.message.tool_calls else 0
        content = (choice.message.content or "")[:60]
        print(f"  choice[{i}]: {tc_count} tool_calls, content={content!r}")

    # Score and build ScoredDataGroup
    scored, all_scores = score_completions(result, nodes, gt_code)

    print("\nPer-choice scores:")
    for i, s in enumerate(all_scores):
        print(
            f"  [{i}] reward={s['reward']:.2f} tc_f1={s['tool_call_f1']:.2f} tp_f1={s['tool_param_f1']:.2f}"
        )

    if scored:
        print(f"\nScoredDataGroup valid: {len(scored['tokens'])} items")
        print(f"  scores: {scored['scores']}")
    else:
        print("\nScoredDataGroup: None (discarded)")

    return True


async def test_noop_turn(server, tokenizer):
    """Test 3: No-op turn (model should NOT call tools)."""
    print("\n" + "=" * 60)
    print("TEST 3: No-op turn")
    print("=" * 60)

    convo = SAMPLE_CONVERSATIONS[3]
    messages = build_messages(convo, turn_index=1)
    gt_code = convo[1]["Filled_Plan"]

    result, nodes = await generate_tool_completions(
        server=server,
        tokenizer=tokenizer,
        messages=messages,
        tools=T1_TOOLS,
        n=1,
        max_tokens=300,
        temperature=0.0,
        tool_choice="auto",
    )

    choice = result.choices[0]
    print(f"\nContent: {(choice.message.content or '')[:100]}")
    print(f"Tool calls: {choice.message.tool_calls}")

    scores = score_turn(gt_code, choice.message.tool_calls, choice.message.content)
    print(f"Scores: {json.dumps(scores, indent=2)}")

    return True


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    args = parser.parse_args()

    base_url = f"http://localhost:{args.port}/v1"
    print(f"Connecting to vLLM at {base_url} (model={args.model})")

    # Check health first
    import requests

    try:
        resp = requests.get(f"http://localhost:{args.port}/health", timeout=5)
        print(f"vLLM health: {resp.status_code}")
        if resp.status_code != 200:
            print("ERROR: vLLM not healthy!")
            return
    except Exception as e:
        print(f"ERROR: Can't reach vLLM: {e}")
        print(
            "Make sure vLLM is running: bash environments/t1_tool_planning/run_vllm.sh"
        )
        return

    server = make_server_manager(args.model, base_url)
    tokenizer = make_tokenizer(args.model)

    print(f"ServerManager created with {len(server.servers)} server(s)")
    print(f"Server type: {type(server.servers[0]).__name__}")
    print(f"Tool parser: {server.tool_parser}")

    passed = 0
    failed = 0

    for test_fn in [test_single_completion, test_group_completions, test_noop_turn]:
        try:
            ok = await test_fn(server, tokenizer)
            if ok:
                passed += 1
                print("\n  ✓ PASSED")
        except Exception as e:
            failed += 1
            print(f"\n  ✗ FAILED: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
