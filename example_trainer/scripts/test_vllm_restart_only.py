#!/usr/bin/env python3
"""
Minimal test for vLLM restart cycle - no training, just launch/terminate/relaunch.
Tests whether GPU memory is properly released between restarts.
"""
import os
import sys
import time
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--port", type=int, default=9099)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--memory-util", type=float, default=0.3)
    parser.add_argument("--restarts", type=int, default=3, help="Number of restart cycles to test")
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    import torch
    from trainers import _launch_vllm_with_lora, _terminate_vllm
    from config import TrainingConfig
    
    print("=" * 60)
    print("vLLM RESTART CYCLE TEST")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Port: {args.port}")
    print(f"GPU: {args.gpu}")
    print(f"Memory utilization: {args.memory_util}")
    print(f"Restart cycles: {args.restarts}")
    print("=" * 60)
    
    # Check initial GPU memory
    if torch.cuda.is_available():
        free_mem = torch.cuda.mem_get_info()[0] / 1e9
        total_mem = torch.cuda.mem_get_info()[1] / 1e9
        print(f"\nInitial GPU memory: {free_mem:.1f}/{total_mem:.1f} GB free")
    
    # Create a minimal config
    config = TrainingConfig(
        model_name=args.model,
        vllm_port=args.port,
        vllm_gpu_memory_utilization=args.memory_util,
        max_model_len=4096,  # Small for quick test
        lora_r=16,
        lora_alpha=32,
        weight_bridge_mode="lora_restart",
        save_path="/tmp/vllm_restart_test",
    )
    
    # Create dummy adapter directory
    os.makedirs(config.save_path, exist_ok=True)
    adapter_path = os.path.join(config.save_path, "dummy_adapter")
    
    # We need to create a real adapter for vLLM to load
    # Let's skip the adapter for this test and just test launch/terminate
    print("\n" + "=" * 60)
    print("Testing vLLM launch/terminate cycle (no adapter)")
    print("=" * 60)
    
    from vllm_manager import kill_process_on_port, wait_for_vllm_ready
    import subprocess
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    server_script = os.path.join(os.path.dirname(script_dir), "vllm_api_server.py")
    
    for cycle in range(args.restarts):
        print(f"\n{'='*60}")
        print(f"CYCLE {cycle + 1}/{args.restarts}")
        print(f"{'='*60}")
        
        # Check memory before launch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free_mem = torch.cuda.mem_get_info()[0] / 1e9
            total_mem = torch.cuda.mem_get_info()[1] / 1e9
            print(f"[Before launch] GPU memory: {free_mem:.1f}/{total_mem:.1f} GB free ({100*free_mem/total_mem:.0f}%)")
        
        # Launch vLLM (without LoRA for simplicity)
        print(f"\n[{cycle+1}] Launching vLLM...")
        cmd = [
            "python", server_script,
            "--model", args.model,
            "--port", str(args.port),
            "--gpu-memory-utilization", str(args.memory_util),
            "--max-model-len", "4096",
        ]
        print(f"  Command: {' '.join(cmd)}")
        
        log_file = f"/tmp/vllm_restart_test/vllm_cycle_{cycle}.log"
        with open(log_file, "w") as f:
            proc = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )
        print(f"  PID: {proc.pid}")
        print(f"  Log: {log_file}")
        
        # Wait for vLLM to be ready
        print(f"  Waiting for vLLM to be ready...")
        start_time = time.time()
        if wait_for_vllm_ready(args.port, timeout=300):
            elapsed = time.time() - start_time
            print(f"  ✓ vLLM ready in {elapsed:.1f}s")
        else:
            print(f"  ✗ vLLM failed to start!")
            print(f"  Check log: {log_file}")
            with open(log_file, "r") as f:
                print(f"  Last 20 lines:\n{''.join(f.readlines()[-20:])}")
            proc.kill()
            return 1
        
        # Check memory after launch
        if torch.cuda.is_available():
            free_mem = torch.cuda.mem_get_info()[0] / 1e9
            total_mem = torch.cuda.mem_get_info()[1] / 1e9
            print(f"[After launch] GPU memory: {free_mem:.1f}/{total_mem:.1f} GB free ({100*free_mem/total_mem:.0f}%)")
        
        # Keep vLLM running for a bit
        print(f"\n  Letting vLLM run for 5s...")
        time.sleep(5)
        
        # Terminate vLLM
        print(f"\n[{cycle+1}] Terminating vLLM...")
        _terminate_vllm(proc, args.port)
        
        # Check memory after terminate
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free_mem = torch.cuda.mem_get_info()[0] / 1e9
            total_mem = torch.cuda.mem_get_info()[1] / 1e9
            print(f"[After terminate] GPU memory: {free_mem:.1f}/{total_mem:.1f} GB free ({100*free_mem/total_mem:.0f}%)")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE!")
    print("=" * 60)
    
    if torch.cuda.is_available():
        free_mem = torch.cuda.mem_get_info()[0] / 1e9
        total_mem = torch.cuda.mem_get_info()[1] / 1e9
        print(f"Final GPU memory: {free_mem:.1f}/{total_mem:.1f} GB free ({100*free_mem/total_mem:.0f}%)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
