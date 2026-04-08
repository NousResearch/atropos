import asyncio
import json
import os
import sys
import hashlib
import argparse
import time
import statistics
from typing import List, Dict
from transformers import AutoTokenizer

# Add atropos to path
sys.path.append(os.getcwd())

from atroposlib.envs.server_handling.server_manager import ServerManager, APIServerConfig
from atroposlib.envs.server_handling.sglang_stateful_server import StatefulSGLangServer

# ---------------------------------------------------------------------------
# HARDWARE BENCHMARK SUITE
# ---------------------------------------------------------------------------
async def run_benchmark(worker_urls: List[str], num_conversations: int = 5, turns_per_conv: int = 4):
    print(f"\n{'='*60}")
    print(f"BENCHMARKING: STATELESS vs STATEFUL SGLANG")
    print(f"HARDWARE: {len(worker_urls)}x GPU Workers")
    print(f"{'='*60}\n")

    configs = [
        APIServerConfig(
            base_url=url,
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            server_type="sglang",
            health_check=True
        ) for url in worker_urls
    ]
    
    manager = ServerManager(configs=configs)
    
    # Wait for health stabilization
    print("Stabilizing workers...")
    await asyncio.sleep(8)
    
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    results = {
        "stateless": {"ttfts": [], "total_times": []},
        "stateful": {"ttfts": [], "total_times": []}
    }

    async def benchmark_mode(mode_name: str, use_stateful: bool):
        print(f"\n--- Running {mode_name.upper()} Mode ---")
        
        # We'll use a dummy flag in ServerManager to bypass stateful if needed
        # Or just toggle the server_type to 'openai' for stateless simulation 
        # but better to use the same class and just not pass session_ids.
        
        for i in range(num_conversations):
            session_id = f"bench-{mode_name}-{i}"
            messages = []
            
            for t in range(turns_per_conv):
                # Simple prompt that grows slightly
                messages.append({"role": "user", "content": f"Explain topic {t} in one sentence."})
                
                start_time = time.time()
                
                # Pass session_id only in stateful mode
                s_id = session_id if use_stateful else None
                
                async with manager.managed_server(session_id=s_id, tokenizer=tokenizer) as managed:
                    res = await managed.chat_completion(messages=messages, max_tokens=10)
                    ttft = time.time() - start_time # Approximation for non-streaming
                    
                    # Store TTFT for Turn 2+ (where cache hit matters)
                    if t > 0:
                        results[mode_name]["ttfts"].append(ttft)
                    
                    # Update messages with assistant response
                    messages.append({"role": "assistant", "content": res.choices[0].message.content})
            
            print(f"  Conversation {i+1}/{num_conversations} complete.")

    # 1. Run Stateless
    await benchmark_mode("stateless", use_stateful=False)
    
    # 2. Run Stateful
    await benchmark_mode("stateful", use_stateful=True)

    # -----------------------------------------------------------------------
    # RESULTS ANALYSIS
    # -----------------------------------------------------------------------
    print(f"\n\n{'='*60}")
    print(f"FINAL PERFORMANCE NUMBERS (T2-T{turns_per_conv} Latency)")
    print(f"{'='*60}")
    
    def get_stats(mode):
        ttfts = results[mode]["ttfts"]
        if not ttfts: return "N/A", "N/A"
        return statistics.mean(ttfts), statistics.stdev(ttfts)

    mean_sl, std_sl = get_stats("stateless")
    mean_sf, std_sf = get_stats("stateful")

    print(f"{'Mode':<15} | {'Avg TTFT (s)':<15} | {'Stdev':<10}")
    print(f"{'-'*45}")
    print(f"{'Stateless':<15} | {mean_sl:<15.4f} | {std_sl:<10.4f}")
    print(f"{'Stateful':<15} | {mean_sf:<15.4f} | {std_sf:<10.4f}")
    
    if mean_sl != "N/A" and mean_sf != "N/A":
        improvement = (mean_sl - mean_sf) / mean_sl * 100
        print(f"\nLATENCY REDUCTION: {improvement:.2f}%")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", nargs="+", default=["http://localhost:30001", "http://localhost:30002"])
    parser.add_argument("--convs", type=int, default=3)
    args = parser.parse_args()
    
    asyncio.run(run_benchmark(args.workers, num_conversations=args.convs))
