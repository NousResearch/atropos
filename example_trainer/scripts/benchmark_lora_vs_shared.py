#!/usr/bin/env python3
"""
Benchmark LoRA vs Shared vLLM inference performance.

This script:
1. Starts two vLLM instances (one with LoRA, one without)
2. Optionally loads a LoRA adapter
3. Sends identical prompts to both
4. Measures and compares TPS (tokens per second)

Usage:
    python benchmark_lora_vs_shared.py --model Qwen/Qwen3-4B-Instruct-2507
    python benchmark_lora_vs_shared.py --model Qwen/Qwen3-4B-Instruct-2507 --lora-path ./checkpoints/final_adapter
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from typing import Optional
from pathlib import Path

import requests

# Force unbuffered output for log files
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None

def log(msg: str):
    """Print with immediate flush."""
    print(msg, flush=True)

# Complex math prompt that requires extended reasoning
BENCHMARK_PROMPT = """You are a mathematics expert. Solve this problem step by step, showing all your work:

A rectangular garden has a perimeter of 56 meters. The length is 4 meters more than twice the width. 

1) Set up the equations
2) Solve for width and length
3) Calculate the area
4) If we want to put a circular fountain in the center with radius equal to 1/4 of the width, what area remains for planting?
5) Express the planting area as a percentage of the total garden area

Show all calculations clearly and verify your answer."""

# Longer prompt for extended generation
LONG_PROMPT = """Write a detailed technical explanation of how transformer neural networks work, covering:

1. The attention mechanism - explain self-attention, multi-head attention, and how queries, keys, and values work
2. The encoder-decoder architecture vs decoder-only models
3. Positional encoding - why it's needed and different approaches
4. Layer normalization and residual connections
5. The feed-forward network component
6. How training works with cross-entropy loss and backpropagation through attention

Include mathematical formulas where appropriate and explain the intuition behind each component. This should be comprehensive enough for someone with basic ML knowledge to understand transformers deeply."""


def wait_for_server(port: int, timeout: int = 300) -> bool:
    """Wait for vLLM server to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"http://localhost:{port}/health", timeout=5)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def start_vllm_server(
    model: str,
    port: int,
    gpu_id: int,
    enable_lora: bool = False,
    max_lora_rank: int = 32,
    log_file: str = "vllm.log",
) -> subprocess.Popen:
    """Start a vLLM server."""
    # Find the vllm_api_server.py script relative to this script
    script_dir = Path(__file__).parent.parent  # example_trainer/
    vllm_server_path = script_dir / "vllm_api_server.py"
    
    if not vllm_server_path.exists():
        log(f"ERROR: vllm_api_server.py not found at {vllm_server_path}")
        raise FileNotFoundError(f"vllm_api_server.py not found at {vllm_server_path}")
    
    cmd = [
        sys.executable, str(vllm_server_path),
        "--model", model,
        "--port", str(port),
        "--gpu-memory-utilization", "0.45",
        "--max-model-len", "8192",
        "--dtype", "bfloat16",
    ]
    
    if enable_lora:
        cmd.extend([
            "--enable-lora",
            "--max-lora-rank", str(max_lora_rank),
            "--enforce-eager",  # Required for LoRA
        ])
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    log(f"Starting vLLM: CUDA_VISIBLE_DEVICES={gpu_id}")
    log(f"Command: {' '.join(cmd)}")
    
    log_f = open(log_file, "w")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
    )
    log(f"Started vLLM process PID={proc.pid}, logging to {log_file}")
    return proc


def load_lora_adapter(port: int, adapter_path: str) -> bool:
    """Load a LoRA adapter into vLLM."""
    try:
        resp = requests.post(
            f"http://localhost:{port}/lora/load",
            json={"adapter_path": adapter_path, "adapter_name": "benchmark_adapter"},
            timeout=30,
        )
        return resp.status_code == 200
    except Exception as e:
        log(f"Failed to load LoRA adapter: {e}")
        return False


def benchmark_inference(
    port: int,
    prompt: str,
    max_tokens: int = 2048,
    num_runs: int = 3,
) -> dict:
    """Benchmark inference on a vLLM server."""
    results = {
        "times": [],
        "tokens": [],
        "tps": [],
    }
    
    for i in range(num_runs):
        start = time.time()
        try:
            resp = requests.post(
                f"http://localhost:{port}/generate",
                json={
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                },
                timeout=300,
            )
            elapsed = time.time() - start
            
            if resp.status_code == 200:
                data = resp.json()
                output_text = data.get("text", [""])[0]
                # Rough token count (words * 1.3)
                output_tokens = len(output_text.split()) * 1.3
                
                results["times"].append(elapsed)
                results["tokens"].append(output_tokens)
                results["tps"].append(output_tokens / elapsed if elapsed > 0 else 0)
                
                log(f"  Run {i+1}: {elapsed:.2f}s, ~{output_tokens:.0f} tokens, {output_tokens/elapsed:.1f} TPS")
            else:
                log(f"  Run {i+1}: FAILED ({resp.status_code})")
        except Exception as e:
            log(f"  Run {i+1}: ERROR - {e}")
    
    if results["times"]:
        results["avg_time"] = sum(results["times"]) / len(results["times"])
        results["avg_tokens"] = sum(results["tokens"]) / len(results["tokens"])
        results["avg_tps"] = sum(results["tps"]) / len(results["tps"])
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark LoRA vs Shared vLLM inference")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507",
                        help="Model to benchmark")
    parser.add_argument("--lora-path", type=str, default=None,
                        help="Path to LoRA adapter (optional)")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Max tokens to generate")
    parser.add_argument("--num-runs", type=int, default=3,
                        help="Number of benchmark runs per server")
    parser.add_argument("--lora-gpu", type=int, default=0,
                        help="GPU for LoRA server")
    parser.add_argument("--shared-gpu", type=int, default=1,
                        help="GPU for shared/base server")
    parser.add_argument("--lora-port", type=int, default=9001,
                        help="Port for LoRA server")
    parser.add_argument("--shared-port", type=int, default=9002,
                        help="Port for shared/base server")
    parser.add_argument("--prompt", type=str, choices=["math", "long"], default="long",
                        help="Which prompt to use")
    parser.add_argument("--skip-lora", action="store_true",
                        help="Skip LoRA server (test base only)")
    parser.add_argument("--skip-shared", action="store_true",
                        help="Skip shared/base server (test LoRA only)")
    args = parser.parse_args()
    
    prompt = LONG_PROMPT if args.prompt == "long" else BENCHMARK_PROMPT
    
    procs = []
    
    def cleanup():
        log("\nCleaning up...")
        for p in procs:
            try:
                p.terminate()
                p.wait(timeout=5)
            except Exception:
                p.kill()
    
    signal.signal(signal.SIGINT, lambda s, f: (cleanup(), sys.exit(0)))
    signal.signal(signal.SIGTERM, lambda s, f: (cleanup(), sys.exit(0)))
    
    try:
        log("=" * 70)
        log("vLLM Inference Benchmark: LoRA vs Base Model")
        log("=" * 70)
        log(f"Model: {args.model}")
        log(f"LoRA adapter: {args.lora_path or 'None (base model only)'}")
        log(f"Max tokens: {args.max_tokens}")
        log(f"Num runs: {args.num_runs}")
        log(f"Prompt type: {args.prompt}")
        log("=" * 70)
        
        # Start LoRA server
        if not args.skip_lora:
            log(f"\n[1/4] Starting LoRA-enabled vLLM on GPU {args.lora_gpu}, port {args.lora_port}...")
            log("      Flags: --enable-lora --enforce-eager (no CUDA graphs)")
            lora_proc = start_vllm_server(
                args.model, args.lora_port, args.lora_gpu,
                enable_lora=True, log_file="benchmark_lora.log"
            )
            procs.append(lora_proc)
        
        # Start base/shared server
        if not args.skip_shared:
            log(f"\n[2/4] Starting base vLLM on GPU {args.shared_gpu}, port {args.shared_port}...")
            log("      Flags: (none) - uses CUDA graphs for faster inference")
            shared_proc = start_vllm_server(
                args.model, args.shared_port, args.shared_gpu,
                enable_lora=False, log_file="benchmark_shared.log"
            )
            procs.append(shared_proc)
        
        # Wait for servers
        log("\n[3/4] Waiting for servers to be ready...")
        
        lora_ready = False
        shared_ready = False
        
        if not args.skip_lora:
            log(f"  Waiting for LoRA server (port {args.lora_port})...")
            lora_ready = wait_for_server(args.lora_port, timeout=300)
            if lora_ready:
                log(f"  ✓ LoRA server ready")
                
                # Load LoRA adapter if provided
                if args.lora_path:
                    log(f"  Loading LoRA adapter from {args.lora_path}...")
                    if load_lora_adapter(args.lora_port, args.lora_path):
                        log(f"  ✓ LoRA adapter loaded")
                    else:
                        log(f"  ✗ Failed to load LoRA adapter")
            else:
                log(f"  ✗ LoRA server failed to start")
        
        if not args.skip_shared:
            log(f"  Waiting for base server (port {args.shared_port})...")
            shared_ready = wait_for_server(args.shared_port, timeout=300)
            if shared_ready:
                log(f"  ✓ Base server ready")
            else:
                log(f"  ✗ Base server failed to start")
        
        # Run benchmarks
        log("\n[4/4] Running benchmarks...")
        log("-" * 70)
        
        lora_results = None
        shared_results = None
        
        if lora_ready and not args.skip_lora:
            log(f"\nLoRA Server (--enable-lora --enforce-eager):")
            lora_results = benchmark_inference(
                args.lora_port, prompt, args.max_tokens, args.num_runs
            )
        
        if shared_ready and not args.skip_shared:
            log(f"\nBase Server (CUDA graphs enabled):")
            shared_results = benchmark_inference(
                args.shared_port, prompt, args.max_tokens, args.num_runs
            )
        
        # Print comparison
        log("\n" + "=" * 70)
        log("RESULTS SUMMARY")
        log("=" * 70)
        
        if lora_results and "avg_tps" in lora_results:
            log(f"\nLoRA Mode (--enable-lora --enforce-eager):")
            log(f"  Avg time:   {lora_results['avg_time']:.2f}s")
            log(f"  Avg tokens: {lora_results['avg_tokens']:.0f}")
            log(f"  Avg TPS:    {lora_results['avg_tps']:.1f}")
        
        if shared_results and "avg_tps" in shared_results:
            log(f"\nBase Mode (CUDA graphs):")
            log(f"  Avg time:   {shared_results['avg_time']:.2f}s")
            log(f"  Avg tokens: {shared_results['avg_tokens']:.0f}")
            log(f"  Avg TPS:    {shared_results['avg_tps']:.1f}")
        
        if lora_results and shared_results and "avg_tps" in lora_results and "avg_tps" in shared_results:
            speedup = shared_results["avg_tps"] / lora_results["avg_tps"] if lora_results["avg_tps"] > 0 else 0
            time_diff = lora_results["avg_time"] - shared_results["avg_time"]
            log(f"\nComparison:")
            log(f"  Base is {speedup:.2f}x faster in TPS")
            log(f"  Base saves {time_diff:.2f}s per request")
            log(f"  --enforce-eager overhead: ~{(1 - 1/speedup) * 100:.1f}%")
        
        log("\n" + "=" * 70)
        log("Note: The main difference is --enforce-eager which disables CUDA graphs.")
        log("This is REQUIRED for LoRA hot-swapping but costs ~10-30% performance.")
        log("=" * 70)
        
    finally:
        cleanup()


if __name__ == "__main__":
    main()
