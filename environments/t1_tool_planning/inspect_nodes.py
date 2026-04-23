#!/usr/bin/env python3
"""Inspect multi-step node output to verify extending works correctly."""

import asyncio
import logging
import os
import signal
import subprocess
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(__file__))
logging.basicConfig(level=logging.WARNING)

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
MODEL = "Qwen/Qwen3-1.7B"
PORT = 8123


def start_vllm():
    cmd = [
        sys.executable,
        "-m",
        "example_trainer.vllm_api_server",
        "--model",
        MODEL,
        "--port",
        str(PORT),
        "--gpu-memory-utilization",
        "0.45",
        "--enforce-eager",
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=REPO_ROOT
    )
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            r = requests.get(f"http://localhost:{PORT}/health", timeout=2)
            if r.status_code == 200:
                print("vLLM ready")
                return proc
        except Exception:
            pass
        if proc.poll() is not None:
            out = proc.stdout.read().decode()[-2000:]
            print(f"vLLM died:\n{out}")
            sys.exit(1)
        time.sleep(3)
    proc.kill()
    print("vLLM timeout")
    sys.exit(1)


async def main():
    from t1_core import collect_multistep_trajectory
    from t1_tools import T1_TOOLS
    from transformers import AutoTokenizer

    from atroposlib.envs.server_handling.server_baseline import APIServerConfig
    from atroposlib.envs.server_handling.server_manager import ServerManager

    config = APIServerConfig(
        model_name=MODEL,
        base_url=f"http://localhost:{PORT}/v1",
        api_key="x",
        server_type="vllm",
    )
    server = ServerManager(
        configs=[config], slurm=False, testing=False, tool_parser="hermes"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    convo = [
        {
            "Role": "assistant",
            "Filled_Template": "Hello! How can I help?",
            "Filled_Plan": "",
        },
        {
            "Role": "user",
            "Filled_Template": "Find hotels in Austin, check-in May 10, check-out May 15, 2025.",
            "Filled_Plan": 'search_hotels(city="Austin", checkin_date=["May 10, 2025"], checkout_date=["May 15, 2025"])',  # noqa: E501
        },
        {
            "Role": "assistant",
            "Filled_Template": "Found some. Filter?",
            "Filled_Plan": "",
        },
        {
            "Role": "user",
            "Filled_Template": "Yes, free wifi.",
            "Filled_Plan": "filter_hotels(prior_results=hotels, free_wifi_included=True)",
        },
    ]

    turn_results, nodes = await collect_multistep_trajectory(
        server=server,
        tokenizer=tokenizer,
        conversation=convo,
        tools=T1_TOOLS,
        max_tokens=500,
        temperature=0.0,
        tool_choice="auto",
    )

    print(f"\nNodes: {len(nodes)}")
    node = nodes[0]

    unmasked_idx = [i for i, t in enumerate(node.masked_tokens) if t != -100]
    masked_idx = [i for i, t in enumerate(node.masked_tokens) if t == -100]
    first_u = unmasked_idx[0] if unmasked_idx else 0

    print(
        f"Total: {len(node.tokens)} | Masked: {len(masked_idx)} | Unmasked: {len(unmasked_idx)}"
    )

    # Check contiguity
    gaps = []
    for j in range(1, len(unmasked_idx)):
        if unmasked_idx[j] != unmasked_idx[j - 1] + 1:
            gaps.append((unmasked_idx[j - 1], unmasked_idx[j]))
    print(f"Unmasked contiguous: {not gaps}  Gaps: {gaps}")

    # Decode
    prompt_text = tokenizer.decode(node.tokens[:first_u], skip_special_tokens=False)
    comp_tokens = [node.tokens[i] for i in unmasked_idx]
    comp_text = tokenizer.decode(comp_tokens, skip_special_tokens=False)

    print("\n--- PROMPT TAIL (last 150 chars) ---")
    print(prompt_text[-150:])
    print("\n--- COMPLETION (unmasked, first 400 chars) ---")
    print(comp_text[:400])
    print("\n--- COMPLETION (unmasked, last 200 chars) ---")
    print(comp_text[-200:])

    print(f"\nPrompt logprobs sample (should be 1.0): {node.logprobs[:3]}")
    print(f"Completion logprobs sample: {[node.logprobs[i] for i in unmasked_idx[:5]]}")

    for tr in turn_results:
        tc = len(tr["tool_calls"]) if tr["tool_calls"] else 0
        print(
            f"\nTurn {tr['turn_idx']}: {tc} tool_calls, reward={tr['scores']['reward']:.2f}"
        )
        if tr["tool_calls"]:
            for t in tr["tool_calls"]:
                print(f"  {t['function']['name']}({t['function']['arguments'][:80]})")
        else:
            print(f"  text: {(tr['content'] or '')[:80]}")


if __name__ == "__main__":
    proc = start_vllm()
    try:
        asyncio.run(main())
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
