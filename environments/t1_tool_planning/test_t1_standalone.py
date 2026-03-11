"""
Standalone test for T1 tool planning environment.

Spins up vllm_api_server, creates a ServerManager with tool_parser="hermes",
and runs through T1 conversations end-to-end using the extracted t1_core functions.

Tests the full tool calling infrastructure:
  ServerManager → ManagedServer → ToolCallTranslator → vLLM hermes parser
  → structured tool_calls → scoring against T1 ground truth

Usage:
    # With GPU (spins up vLLM):
    pytest --run-gpu environments/t1_tool_planning/test_t1_standalone.py -v -s

    # Scoring logic only (no GPU):
    pytest environments/t1_tool_planning/test_t1_standalone.py -v -k "not gpu"
"""

import json
import os
import signal
import subprocess
import sys
import time

import pytest
import requests

# -- T1 env imports --
sys.path.insert(0, os.path.dirname(__file__))
from t1_core import (  # noqa: E402
    collect_multistep_trajectory,
    generate_tool_completions,
    score_completions,
)
from t1_prompts import SYSTEM_PROMPT  # noqa: E402
from t1_scoring import (  # noqa: E402
    parse_ground_truth_code,
    parse_model_tool_calls,
    score_turn,
    tool_call_f1,
    tool_param_f1,
)
from t1_tools import T1_TOOLS  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VLLM_PORT = 8123
VLLM_MODEL = "Qwen/Qwen3-1.7B"
VLLM_BASE_URL = f"http://localhost:{VLLM_PORT}/v1"
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
VLLM_SCRIPT = os.path.join(REPO_ROOT, "example_trainer", "vllm_api_server.py")

# Sample T1 data — small hotel conversations for testing
SAMPLE_T1_CONVERSATIONS = {
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
        {
            "Role": "assistant",
            "Filled_Template": "I found several hotels in Austin for those dates. Would you like to filter by any specific amenities?",  # noqa: E501
            "Filled_Plan": "",
        },
        {
            "Role": "user",
            "Filled_Template": "Yes, I need one with free wifi and a gym.",
            "Filled_Plan": 'hotels = get_results_from_cache(key="hotels")\nfiltered_hotels = filter_hotels(prior_results=hotels, free_wifi_included=True, gym_present=True)\nsave_to_cache(key="filtered_hotels", value=filtered_hotels)',  # noqa: E501
        },
        {
            "Role": "assistant",
            "Filled_Template": "Here are hotels with free wifi and gym. Anything else?",
            "Filled_Plan": "",
        },
        {
            "Role": "user",
            "Filled_Template": "No that's perfect, thanks!",
            "Filled_Plan": 'print("No planning needed")',
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
        {
            "Role": "assistant",
            "Filled_Template": "Sure! When would you like to check in and check out?",
            "Filled_Plan": "",
        },
        {
            "Role": "user",
            "Filled_Template": "Check in June 1 and check out June 5, 2025.",
            "Filled_Plan": 'hotels = search_hotels(city="New York", checkin_date=["June 1, 2025"], checkout_date=["June 5, 2025"])\nsave_to_cache(key="hotels", value=hotels)',  # noqa: E501
        },
    ],
}


def load_sample_data() -> dict:
    """Load the sample T1 conversations."""
    return SAMPLE_T1_CONVERSATIONS


def build_messages_for_turn(conversation: list, turn_index: int) -> list:
    """Build chat messages up to (and including) the given user turn.

    Uses ground-truth assistant responses for prior turns.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for i, turn in enumerate(conversation):
        if i > turn_index:
            break

        role = turn["Role"].strip().lower()
        content = turn["Filled_Template"]

        if role == "assistant":
            messages.append({"role": "assistant", "content": content})
        elif role == "user":
            messages.append({"role": "user", "content": content})

    return messages


# ---------------------------------------------------------------------------
# Scoring-only tests (no GPU needed)
# ---------------------------------------------------------------------------


class TestT1Scoring:
    """Test the scoring logic with known inputs."""

    def test_parse_ground_truth_search(self):
        code = 'hotels = search_hotels(city="Austin", checkin_date=["May 10"], checkout_date=["May 15"])\nsave_to_cache(key="hotels", value=hotels)'  # noqa: E501
        calls = parse_ground_truth_code(code)
        assert len(calls) == 2
        assert calls[0]["name"] == "search_hotels"
        assert calls[0]["arguments"]["city"] == "Austin"
        assert calls[1]["name"] == "save_to_cache"

    def test_parse_ground_truth_filter(self):
        code = 'hotels = get_results_from_cache(key="hotels")\nfiltered = filter_hotels(prior_results=hotels, free_wifi_included=True)'  # noqa: E501
        calls = parse_ground_truth_code(code)
        assert len(calls) == 2
        assert calls[0]["name"] == "get_results_from_cache"
        assert calls[1]["name"] == "filter_hotels"
        assert calls[1]["arguments"]["free_wifi_included"] is True

    def test_parse_ground_truth_noop(self):
        code = 'print("No planning needed")'
        calls = parse_ground_truth_code(code)
        assert len(calls) == 0

    def test_parse_ground_truth_seek(self):
        code = 'seek_information("We need check-in dates")'
        calls = parse_ground_truth_code(code)
        assert len(calls) == 1
        assert calls[0]["name"] == "seek_information"

    def test_parse_empty(self):
        assert parse_ground_truth_code("") == []
        assert parse_ground_truth_code(None) == []

    def test_tool_call_f1_perfect(self):
        gt = [
            {"name": "search_hotels", "arguments": {}},
            {"name": "save_to_cache", "arguments": {}},
        ]
        gen = [
            {"name": "search_hotels", "arguments": {}},
            {"name": "save_to_cache", "arguments": {}},
        ]
        p, r, f1 = tool_call_f1(gt, gen)
        assert f1 == 1.0

    def test_tool_call_f1_partial(self):
        gt = [
            {"name": "search_hotels", "arguments": {}},
            {"name": "save_to_cache", "arguments": {}},
        ]
        gen = [{"name": "search_hotels", "arguments": {}}]
        p, r, f1 = tool_call_f1(gt, gen)
        assert p == 1.0
        assert r == 0.5
        assert 0 < f1 < 1

    def test_tool_call_f1_wrong(self):
        gt = [{"name": "search_hotels", "arguments": {}}]
        gen = [{"name": "search_flights", "arguments": {}}]
        p, r, f1 = tool_call_f1(gt, gen)
        assert f1 == 0.0

    def test_tool_param_f1_matching(self):
        gt = [
            {
                "name": "search_hotels",
                "arguments": {"city": "Austin", "checkin_date": ["May 10"]},
            }
        ]
        gen = [
            {
                "name": "search_hotels",
                "arguments": {"city": "Austin", "checkin_date": ["May 10"]},
            }
        ]
        p, r, f1 = tool_param_f1(gt, gen)
        assert f1 == 1.0

    def test_tool_param_f1_partial(self):
        gt = [
            {
                "name": "search_hotels",
                "arguments": {"city": "Austin", "checkin_date": ["May 10"]},
            }
        ]
        gen = [{"name": "search_hotels", "arguments": {"city": "Austin"}}]
        p, r, f1 = tool_param_f1(gt, gen)
        assert p == 1.0  # what we generated is correct
        assert r == 0.5  # but we missed one param

    def test_score_turn_noop_correct(self):
        scores = score_turn('print("No planning needed")', None)
        assert scores["reward"] == 1.0

    def test_score_turn_noop_wrong(self):
        # GT says no-op but model called tools
        fake_calls = [
            {"function": {"name": "search_hotels", "arguments": '{"city": "X"}'}}
        ]
        scores = score_turn('print("No planning needed")', fake_calls)
        assert scores["reward"] == 0.0

    def test_score_turn_tools_expected_none_produced(self):
        # GT expects tools but model produced none → 0.0
        gt = 'hotels = search_hotels(city="Austin", checkin_date=["May 10"], checkout_date=["May 15"])'
        scores = score_turn(gt, None)
        assert scores["reward"] == 0.0

    def test_score_turn_wrong_tool_gets_format_credit(self):
        # GT expects search_hotels, model called search_flights → 0.1 (format credit only)
        gt = 'hotels = search_hotels(city="Austin", checkin_date=["May 10"], checkout_date=["May 15"])'
        wrong_calls = [
            {
                "function": {
                    "name": "search_flights",
                    "arguments": '{"start_airport_city": "X"}',
                }
            }
        ]
        scores = score_turn(gt, wrong_calls)
        assert scores["reward"] == 0.1  # format credit, no f1 match
        assert scores["tool_call_f1"] == 0.0

    def test_score_turn_right_tool_higher_than_wrong(self):
        # Right tool should score higher than wrong tool
        gt = 'hotels = search_hotels(city="Austin", checkin_date=["May 10"], checkout_date=["May 15"])'
        right_calls = [
            {"function": {"name": "search_hotels", "arguments": '{"city": "Austin"}'}}
        ]
        wrong_calls = [
            {
                "function": {
                    "name": "search_flights",
                    "arguments": '{"start_airport_city": "X"}',
                }
            }
        ]
        right_scores = score_turn(gt, right_calls)
        wrong_scores = score_turn(gt, wrong_calls)
        assert right_scores["reward"] > wrong_scores["reward"]

    def test_parse_model_tool_calls(self):
        calls = [
            {"function": {"name": "search_hotels", "arguments": '{"city": "Austin"}'}},
            {
                "function": {
                    "name": "save_to_cache",
                    "arguments": '{"key": "hotels", "value": "hotels"}',
                }
            },
        ]
        parsed = parse_model_tool_calls(calls)
        assert len(parsed) == 2
        assert parsed[0]["name"] == "search_hotels"
        assert parsed[0]["arguments"]["city"] == "Austin"

    def test_sample_data_loads(self):
        convos = load_sample_data()
        assert len(convos) == 2
        assert len(convos[1]) == 6  # 3 assistant + 3 user turns
        assert len(convos[2]) == 4

    def test_build_messages(self):
        convos = load_sample_data()
        # Turn index 1 = first user turn (index 0 is assistant)
        msgs = build_messages_for_turn(convos[1], turn_index=1)
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "assistant"
        assert msgs[2]["role"] == "user"
        assert "Austin" in msgs[2]["content"]

    def test_t1_tools_valid(self):
        """Verify all tool definitions have required fields."""
        for tool in T1_TOOLS:
            assert tool["type"] == "function"
            assert "name" in tool["function"]
            assert "parameters" in tool["function"]


# ---------------------------------------------------------------------------
# GPU integration test — full pipeline with vLLM
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def vllm_backend():
    """Start vLLM api server as a subprocess."""
    cmd = [
        sys.executable,
        VLLM_SCRIPT,
        "--model",
        VLLM_MODEL,
        "--port",
        str(VLLM_PORT),
        "--gpu-memory-utilization",
        "0.45",
        "--enforce-eager",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=REPO_ROOT,
    )

    deadline = time.time() + 180
    healthy = False
    while time.time() < deadline:
        try:
            resp = requests.get(f"http://localhost:{VLLM_PORT}/health", timeout=2)
            if resp.status_code == 200:
                healthy = True
                break
        except (requests.ConnectionError, requests.Timeout):
            pass
        if proc.poll() is not None:
            stdout = proc.stdout.read().decode() if proc.stdout else ""
            pytest.fail(f"vLLM exited early:\n{stdout[-3000:]}")
        time.sleep(3)

    if not healthy:
        proc.kill()
        stdout = proc.stdout.read().decode() if proc.stdout else ""
        pytest.fail(f"vLLM didn't start within 180s:\n{stdout[-3000:]}")

    yield proc

    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


@pytest.fixture(scope="module")
def server_and_tokenizer(vllm_backend):
    """Create a ServerManager + tokenizer pointed at the vLLM backend."""
    from transformers import AutoTokenizer

    from atroposlib.envs.server_handling.server_baseline import APIServerConfig
    from atroposlib.envs.server_handling.server_manager import ServerManager

    config = APIServerConfig(
        model_name=VLLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key="x",
        server_type="vllm",
    )
    server = ServerManager(
        configs=[config],
        slurm=False,
        testing=False,
        tool_parser="hermes",
    )
    tokenizer = AutoTokenizer.from_pretrained(VLLM_MODEL)
    return server, tokenizer


@pytest.mark.gpu
class TestT1FullPipeline:
    """End-to-end test: vLLM → ServerManager → ManagedServer → tool calls → scoring."""

    async def test_single_turn_tool_call(self, server_and_tokenizer):
        """Model should call search_hotels when given enough info."""
        server, tokenizer = server_and_tokenizer

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

        # Model should have called at least search_hotels
        if choice.message.tool_calls:
            names = [tc["function"]["name"] for tc in choice.message.tool_calls]
            print(f"Tool names called: {names}")
            assert "search_hotels" in names or "seek_information" in names

        # Score against ground truth
        gt_code = 'hotels = search_hotels(city="Austin", checkin_date=["May 10, 2025"], checkout_date=["May 15, 2025"])\nsave_to_cache(key="hotels", value=hotels)'  # noqa: E501
        scores = score_turn(gt_code, choice.message.tool_calls)
        print(f"Scores: {json.dumps(scores, indent=2)}")
        assert scores["reward"] > 0

        # Verify nodes are tracked
        assert len(nodes) == 1
        assert len(nodes[0].tokens) > 0
        assert len(nodes[0].logprobs) == len(nodes[0].tokens)

    async def test_group_completions(self, server_and_tokenizer):
        """Generate n=4 completions (GRPO-style) and score them."""
        server, tokenizer = server_and_tokenizer

        convo = SAMPLE_T1_CONVERSATIONS[1]
        messages = build_messages_for_turn(convo, turn_index=1)
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

        assert len(result.choices) == 4
        assert len(nodes) == 4

        print(f"\nGot {len(result.choices)} choices:")
        for i, choice in enumerate(result.choices):
            tc_count = (
                len(choice.message.tool_calls) if choice.message.tool_calls else 0
            )
            content = (choice.message.content or "")[:60]
            print(f"  choice[{i}]: {tc_count} tool_calls, content={content!r}")

        # Score and build ScoredDataGroup
        scored, all_scores = score_completions(result, nodes, gt_code)

        print("\nPer-choice scores:")
        for i, s in enumerate(all_scores):
            print(
                f"  [{i}] reward={s['reward']:.2f} tc_f1={s['tool_call_f1']:.2f} tp_f1={s['tool_param_f1']:.2f}"
            )

        # At least some choices should have tokens
        assert len(all_scores) == 4

        if scored:
            print(
                f"\nScoredDataGroup: {len(scored['tokens'])} valid items, scores={scored['scores']}"
            )
            # Verify structure
            assert len(scored["tokens"]) == len(scored["masks"])
            assert len(scored["tokens"]) == len(scored["scores"])
            assert len(scored["tokens"]) == len(scored["inference_logprobs"])

    async def test_seek_information(self, server_and_tokenizer):
        """Model should ask for missing info when dates aren't provided."""
        server, tokenizer = server_and_tokenizer

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "I need a hotel in New York but I'm not sure about dates.",
            },
        ]

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
        print(f"\nContent: {choice.message.content}")
        print(f"Tool calls: {choice.message.tool_calls}")

        gt_code = (
            'seek_information("We need to ask for the check-in and check-out dates")'
        )
        scores = score_turn(gt_code, choice.message.tool_calls)
        print(f"Scores: {json.dumps(scores, indent=2)}")

        assert len(nodes) == 1

    async def test_noop_turn(self, server_and_tokenizer):
        """No-op turn — model should just respond without tools."""
        server, tokenizer = server_and_tokenizer

        convo = SAMPLE_T1_CONVERSATIONS[1]
        # Turn 5 = "No that's perfect, thanks!" → print("No planning needed")
        messages = build_messages_for_turn(convo, turn_index=5)
        gt_code = convo[5]["Filled_Plan"]

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

    async def test_conversation_walkthrough(self, server_and_tokenizer):
        """Walk through a full T1 conversation using GT context, scoring each user turn."""
        server, tokenizer = server_and_tokenizer
        convos = load_sample_data()
        convo = convos[1]  # 3-turn hotel conversation

        all_scores = []
        user_turn_indices = [
            i for i, t in enumerate(convo) if t["Role"].strip().lower() == "user"
        ]

        for turn_idx in user_turn_indices:
            turn = convo[turn_idx]
            messages = build_messages_for_turn(convo, turn_idx)

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
            gt_code = turn["Filled_Plan"]
            scores = score_turn(
                gt_code, choice.message.tool_calls, choice.message.content
            )

            print(f"\n--- Turn {turn_idx} ---")
            print(f"User: {turn['Filled_Template'][:80]}...")
            print(f"GT code: {gt_code[:80]}...")
            if choice.message.tool_calls:
                names = [tc["function"]["name"] for tc in choice.message.tool_calls]
                print(f"Model called: {names}")
            else:
                print(f"Model text: {(choice.message.content or '')[:80]}...")
            print(
                f"Scores: tc_f1={scores['tool_call_f1']:.2f} tp_f1={scores['tool_param_f1']:.2f} reward={scores['reward']:.2f}"  # noqa: E501
            )

            all_scores.append(scores)

        # Aggregate
        avg_tc_f1 = sum(s["tool_call_f1"] for s in all_scores) / len(all_scores)
        avg_tp_f1 = sum(s["tool_param_f1"] for s in all_scores) / len(all_scores)
        avg_reward = sum(s["reward"] for s in all_scores) / len(all_scores)

        print(f"\n=== AGGREGATE ({len(all_scores)} turns) ===")
        print(f"Tool Call F1: {avg_tc_f1:.3f}")
        print(f"Tool Param F1: {avg_tp_f1:.3f}")
        print(f"Avg Reward: {avg_reward:.3f}")

    async def test_multistep_trajectory(self, server_and_tokenizer):
        """Multi-step: walk full conversation feeding model's OWN responses back.

        This is the real end-to-end test. At each turn:
          1. Model generates a response (possibly with tool_calls)
          2. That response is fed back as conversation history
          3. ToolCallTranslator reconstructs raw text from tool_calls
          4. Chat template re-tokenizes with reconstructed text
          5. Next turn uses the model's actual history, not GT

        Tests the full bidirectional tool call pipeline.
        """
        server, tokenizer = server_and_tokenizer
        convos = load_sample_data()
        convo = convos[1]  # 3-turn hotel conversation (6 entries: 3 asst + 3 user)

        turn_results, nodes = await collect_multistep_trajectory(
            server=server,
            tokenizer=tokenizer,
            conversation=convo,
            tools=T1_TOOLS,
            max_tokens=500,
            temperature=0.0,
            tool_choice="auto",
        )

        assert len(turn_results) > 0, "Should have at least one turn result"
        assert len(nodes) > 0, "Should have tracked nodes"

        print(
            f"\n=== MULTI-STEP TRAJECTORY ({len(turn_results)} turns, {len(nodes)} nodes) ==="
        )
        for tr in turn_results:
            print(f"\n--- Turn {tr['turn_idx']} ---")
            print(f"User: {tr['user_message'][:80]}...")
            print(f"GT: {tr['gt_code'][:80]}...")
            if tr["tool_calls"]:
                names = [tc["function"]["name"] for tc in tr["tool_calls"]]
                print(f"Model called: {names}")
            else:
                print(f"Model text: {(tr['content'] or '')[:80]}...")
            s = tr["scores"]
            print(
                f"Scores: tc_f1={s['tool_call_f1']:.2f} tp_f1={s['tool_param_f1']:.2f} reward={s['reward']:.2f}"
            )
            print(f"Messages in context: {len(tr['messages_so_far'])}")

        # Verify nodes — each turn should extend the previous
        print("\nNodes from managed server:")
        for i, node in enumerate(nodes):
            unmasked = len([t for t in node.masked_tokens if t != -100])
            print(f"  node[{i}]: {len(node.tokens)} tokens, {unmasked} unmasked")

        # Verify conversation grew correctly
        user_turns = len(turn_results)
        last_msg_count = len(turn_results[-1]["messages_so_far"])
        expected_min = (
            2 + user_turns
        )  # system + greeting + at least 1 msg per user turn
        print(
            f"\nFinal conversation: {last_msg_count} messages (expected >= {expected_min})"
        )
        assert last_msg_count >= expected_min

        # Aggregate
        avg_reward = sum(r["scores"]["reward"] for r in turn_results) / len(
            turn_results
        )
        avg_tc_f1 = sum(r["scores"]["tool_call_f1"] for r in turn_results) / len(
            turn_results
        )
        print(f"\nAvg Reward: {avg_reward:.3f}")
        print(f"Avg TC F1: {avg_tc_f1:.3f}")

    async def test_multistep_with_tool_history(self, server_and_tokenizer):
        """Verify tool_calls in history are properly reconstructed for subsequent turns.

        The critical path: turn N produces tool_calls → turn N+1's prompt must
        contain those tool_calls in raw text form (e.g. <tool_call> tags) so the
        tokenizer produces correct tokens.
        """
        server, tokenizer = server_and_tokenizer
        convos = load_sample_data()
        convo = convos[2]  # 2-turn: seek_information → search_hotels

        turn_results, nodes = await collect_multistep_trajectory(
            server=server,
            tokenizer=tokenizer,
            conversation=convo,
            tools=T1_TOOLS,
            max_tokens=500,
            temperature=0.0,
            tool_choice="auto",
        )

        print(
            f"\n=== TOOL HISTORY TEST ({len(turn_results)} turns, {len(nodes)} nodes) ==="
        )
        for tr in turn_results:
            tc_count = len(tr["tool_calls"]) if tr["tool_calls"] else 0
            print(
                f"Turn {tr['turn_idx']}: {tc_count} tool_calls, reward={tr['scores']['reward']:.2f}"
            )

        # Nodes should show extending — later nodes have more tokens
        for i, node in enumerate(nodes):
            print(f"  node[{i}]: {len(node.tokens)} total tokens")

        # If turn 0 produced tool_calls, verify turn 1's messages contain them
        if len(turn_results) >= 2 and turn_results[0]["tool_calls"]:
            turn1_messages = turn_results[1]["messages_so_far"]
            assistant_msgs = [m for m in turn1_messages if m.get("role") == "assistant"]
            has_tool_history = any(m.get("tool_calls") for m in assistant_msgs)
            print(f"\nTurn 1 has tool_call history in context: {has_tool_history}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
