import argparse
import asyncio
import hashlib
import json
import os
import statistics
import sys
import time
from typing import Dict, List

from transformers import AutoTokenizer

# Add atropos to path
sys.path.append(os.getcwd())

from atroposlib.envs.server_handling.server_manager import (
    APIServerConfig,
    ServerManager,
)
from atroposlib.envs.server_handling.sglang_stateful_server import StatefulSGLangServer


# ---------------------------------------------------------------------------
# HARDWARE BENCHMARK SUITE
# ---------------------------------------------------------------------------
async def run_benchmark(
    worker_urls: List[str], num_conversations: int = 5, turns_per_conv: int = 4
):
    print(f"Benchmarking Stateless vs Stateful SGLang ({len(worker_urls)} workers)")

    configs = [
        APIServerConfig(
            base_url=url,
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            server_type="sglang",
            health_check=True,
        )
        for url in worker_urls
    ]

    manager = ServerManager(configs=configs)
    await asyncio.sleep(5)

    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    results = {
        "stateless": {"ttfts": [], "total_times": []},
        "stateful": {"ttfts": [], "total_times": []},
    }

    async def benchmark_mode(mode_name: str, use_stateful: bool):
        print(f"Running {mode_name}...")

        for i in range(num_conversations):
            session_id = f"bench-{mode_name}-{i}"
            messages = []

            for t in range(turns_per_conv):
                messages.append(
                    {"role": "user", "content": f"Explain topic {t} in one sentence."}
                )
                start_time = time.time()
                s_id = session_id if use_stateful else None

                async with manager.managed_server(
                    session_id=s_id, tokenizer=tokenizer
                ) as managed:
                    res = await managed.chat_completion(
                        messages=messages, max_tokens=10
                    )
                    ttft = time.time() - start_time

                    if t > 0:
                        results[mode_name]["ttfts"].append(ttft)

                    messages.append(
                        {"role": "assistant", "content": res.choices[0].message.content}
                    )

    await benchmark_mode("stateless", use_stateful=False)
    await benchmark_mode("stateful", use_stateful=True)

    def get_stats(mode):
        ttfts = results[mode]["ttfts"]
        if not ttfts:
            return 0.0, 0.0
        return statistics.mean(ttfts), statistics.stdev(ttfts)

    mean_sl, std_sl = get_stats("stateless")
    mean_sf, std_sf = get_stats("stateful")

    print(f"\nResults (Latency T2-T{turns_per_conv}):")
    print(f"Stateless: {mean_sl:.4f}s (std={std_sl:.4f})")
    print(f"Stateful:  {mean_sf:.4f}s (std={std_sf:.4f})")

    if mean_sl > 0:
        improvement = (mean_sl - mean_sf) / mean_sl * 100
        print(f"Latency Reduction: {improvement:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers",
        nargs="+",
        default=["http://localhost:30001", "http://localhost:30002"],
    )
    parser.add_argument("--convs", type=int, default=3)
    args = parser.parse_args()

    asyncio.run(run_benchmark(args.workers, num_conversations=args.convs))
