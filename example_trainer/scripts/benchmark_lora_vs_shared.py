#!/usr/bin/env python3
"""
Benchmark LoRA inference modes to find the fastest approach.

This script tests multiple vLLM configurations to determine:
1. Does --enable-lora force eager mode even without --enforce-eager?
2. What's the actual TPS difference between configurations?
3. Is there ANY way to get fast LoRA inference?

Configurations tested:
- BASE: No LoRA flags (CUDA graphs enabled) - baseline
- LORA_EAGER: --enable-lora --enforce-eager (required for hot-swap)
- LORA_NO_EAGER: --enable-lora only (does vLLM force eager anyway?)

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
    mode: str = "base",  # "base", "lora_eager", "lora_no_eager"
    max_lora_rank: int = 32,
    log_file: str = "vllm.log",
) -> subprocess.Popen:
    """
    Start a vLLM server with different configurations.
    
    Modes:
    - base: No LoRA, CUDA graphs enabled (fastest)
    - lora_eager: --enable-lora --enforce-eager (slow, but supports hot-swap)
    - lora_no_eager: --enable-lora only (test if vLLM forces eager anyway)
    """
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
    
    if mode == "lora_eager":
        cmd.extend([
            "--enable-lora",
            "--max-lora-rank", str(max_lora_rank),
            "--enforce-eager",
        ])
        log(f"Mode: LORA_EAGER (--enable-lora --enforce-eager)")
    elif mode == "lora_no_eager":
        cmd.extend([
            "--enable-lora",
            "--max-lora-rank", str(max_lora_rank),
            # NOTE: NOT adding --enforce-eager - testing if vLLM forces it anyway
        ])
        log(f"Mode: LORA_NO_EAGER (--enable-lora only, NO --enforce-eager)")
    else:
        log(f"Mode: BASE (no LoRA flags, CUDA graphs enabled)")
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    log(f"GPU: {gpu_id}")
    log(f"Command: {' '.join(cmd)}")
    
    log_f = open(log_file, "w")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
    )
    log(f"Started vLLM PID={proc.pid}, log: {log_file}")
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
    parser = argparse.ArgumentParser(description="Benchmark LoRA inference configurations")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507",
                        help="Model to benchmark")
    parser.add_argument("--lora-path", type=str, default=None,
                        help="Path to LoRA adapter (optional)")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Max tokens to generate")
    parser.add_argument("--num-runs", type=int, default=3,
                        help="Number of benchmark runs per server")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU to use (tests run sequentially)")
    parser.add_argument("--port", type=int, default=9001,
                        help="Port for vLLM server")
    parser.add_argument("--prompt", type=str, choices=["math", "long"], default="long",
                        help="Which prompt to use")
    parser.add_argument("--modes", type=str, default="all",
                        help="Comma-separated modes to test: base,lora_eager,lora_no_eager or 'all'")
    args = parser.parse_args()
    
    prompt = LONG_PROMPT if args.prompt == "long" else BENCHMARK_PROMPT
    
    # Parse modes to test
    if args.modes == "all":
        modes_to_test = ["base", "lora_no_eager", "lora_eager"]
    else:
        modes_to_test = [m.strip() for m in args.modes.split(",")]
    
    results = {}
    current_proc = None
    
    def cleanup():
        log("\nCleaning up...")
        if current_proc:
            try:
                current_proc.terminate()
                current_proc.wait(timeout=5)
            except Exception:
                try:
                    current_proc.kill()
                except Exception:
                    pass
    
    signal.signal(signal.SIGINT, lambda s, f: (cleanup(), sys.exit(0)))
    signal.signal(signal.SIGTERM, lambda s, f: (cleanup(), sys.exit(0)))
    
    try:
        log("=" * 70)
        log("vLLM LoRA Inference Configuration Benchmark")
        log("=" * 70)
        log(f"Model: {args.model}")
        log(f"LoRA adapter: {args.lora_path or 'None'}")
        log(f"Max tokens: {args.max_tokens}")
        log(f"Num runs: {args.num_runs}")
        log(f"Modes to test: {modes_to_test}")
        log("=" * 70)
        log("")
        log("QUESTION: Does --enable-lora force eager mode even without --enforce-eager?")
        log("=" * 70)
        
        # Test each mode sequentially (same GPU, restart between tests)
        for i, mode in enumerate(modes_to_test):
            log(f"\n[{i+1}/{len(modes_to_test)}] Testing mode: {mode.upper()}")
            log("-" * 70)
            
            # Start server
            current_proc = start_vllm_server(
                args.model, args.port, args.gpu,
                mode=mode, log_file=f"benchmark_{mode}.log"
            )
            
            # Wait for ready
            log(f"  Waiting for server (port {args.port})...")
            if not wait_for_server(args.port, timeout=300):
                log(f"  ✗ Server failed to start! Check benchmark_{mode}.log")
                results[mode] = {"error": "Server failed to start"}
                current_proc.terminate()
                current_proc = None
                continue
            
            log(f"  ✓ Server ready")
            
            # Load LoRA adapter if provided and mode supports it
            if args.lora_path and mode in ["lora_eager", "lora_no_eager"]:
                log(f"  Loading LoRA adapter...")
                if load_lora_adapter(args.port, args.lora_path):
                    log(f"  ✓ Adapter loaded")
                else:
                    log(f"  ⚠ Failed to load adapter (continuing anyway)")
            
            # Check the log file for CUDA graph status
            log(f"  Checking CUDA graph status in log...")
            try:
                with open(f"benchmark_{mode}.log", "r") as f:
                    log_content = f.read()
                    if "Cudagraph is disabled" in log_content:
                        log(f"  ⚠ CUDA GRAPHS DISABLED (eager mode)")
                    elif "cudagraph" in log_content.lower():
                        # Look for other cudagraph messages
                        for line in log_content.split("\n"):
                            if "cudagraph" in line.lower():
                                log(f"  Log: {line.strip()[:80]}")
                    else:
                        log(f"  (No cudagraph message found in log)")
            except Exception as e:
                log(f"  (Could not read log: {e})")
            
            # Run benchmark
            log(f"\n  Running {args.num_runs} inference requests...")
            mode_results = benchmark_inference(
                args.port, prompt, args.max_tokens, args.num_runs
            )
            results[mode] = mode_results
            
            # Terminate server
            log(f"  Stopping server...")
            current_proc.terminate()
            try:
                current_proc.wait(timeout=10)
            except Exception:
                current_proc.kill()
            current_proc = None
            
            # Wait for port to be free
            time.sleep(3)
        
        # Print comparison
        log("\n" + "=" * 70)
        log("RESULTS SUMMARY")
        log("=" * 70)
        
        valid_results = {k: v for k, v in results.items() if "avg_tps" in v}
        
        for mode, res in valid_results.items():
            log(f"\n{mode.upper()}:")
            log(f"  Avg time:   {res['avg_time']:.2f}s")
            log(f"  Avg tokens: {res['avg_tokens']:.0f}")
            log(f"  Avg TPS:    {res['avg_tps']:.1f}")
        
        # Compare
        if "base" in valid_results:
            base_tps = valid_results["base"]["avg_tps"]
            log(f"\n" + "-" * 70)
            log("COMPARISON TO BASE (CUDA graphs enabled):")
            for mode, res in valid_results.items():
                if mode != "base":
                    ratio = res["avg_tps"] / base_tps if base_tps > 0 else 0
                    slowdown = (1 - ratio) * 100
                    log(f"  {mode}: {res['avg_tps']:.1f} TPS ({ratio:.2f}x base, {slowdown:.1f}% slower)")
        
        # Key finding
        log("\n" + "=" * 70)
        log("KEY FINDING:")
        if "lora_no_eager" in valid_results and "lora_eager" in valid_results:
            no_eager_tps = valid_results["lora_no_eager"]["avg_tps"]
            eager_tps = valid_results["lora_eager"]["avg_tps"]
            if abs(no_eager_tps - eager_tps) < eager_tps * 0.1:  # Within 10%
                log("  ⚠ --enable-lora FORCES eager mode regardless of --enforce-eager flag!")
                log("  ⚠ There is NO WAY to get CUDA graphs with LoRA enabled in vLLM.")
            else:
                log("  ✓ --enable-lora without --enforce-eager is FASTER!")
                log(f"  ✓ lora_no_eager: {no_eager_tps:.1f} TPS vs lora_eager: {eager_tps:.1f} TPS")
        
        if "base" in valid_results and "lora_eager" in valid_results:
            base_tps = valid_results["base"]["avg_tps"]
            lora_tps = valid_results["lora_eager"]["avg_tps"]
            log(f"\n  Base model (no LoRA): {base_tps:.1f} TPS")
            log(f"  LoRA enabled:         {lora_tps:.1f} TPS")
            log(f"  Slowdown factor:      {base_tps/lora_tps:.1f}x")
        
        log("=" * 70)
        
    finally:
        cleanup()


if __name__ == "__main__":
    main()
