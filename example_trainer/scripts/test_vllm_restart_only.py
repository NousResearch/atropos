#!/usr/bin/env python3
"""
Minimal test for vLLM restart cycle - no training, just launch/terminate/relaunch.
Tests whether GPU memory is properly released between restarts.

Run from atropos directory:
    python example_trainer/scripts/test_vllm_restart_only.py --restarts 3 --gpu 0
"""
import os
import sys
import time
import argparse
import subprocess
import signal


def kill_process_on_port(port: int) -> None:
    """Kill any process using the specified port."""
    try:
        subprocess.run(f"fuser -k {port}/tcp", shell=True, capture_output=True, timeout=10)
    except Exception:
        pass


def wait_for_vllm_ready(port: int, timeout: int = 300) -> bool:
    """Wait for vLLM to be ready on the specified port."""
    import urllib.request
    import urllib.error
    
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5)
            if req.status == 200:
                return True
        except (urllib.error.URLError, Exception):
            pass
        time.sleep(5)
        elapsed = int(time.time() - start)
        print(f"    Waiting... ({elapsed}s / {timeout}s)")
    return False


def terminate_vllm(proc, port: int) -> None:
    """Terminate vLLM process and release GPU memory."""
    print(f"  Terminating vLLM on port {port}...")
    
    # Get current GPU device
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    
    # Phase 1: Kill the process group (kills all children too)
    if proc is not None:
        print(f"  Killing process group (PID: {proc.pid})...")
        try:
            # Kill entire process group - this gets all child processes
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception as e:
            print(f"  Warning: {e}")
    
    # Phase 2: Kill by port (catches anything still running)
    kill_process_on_port(port)
    time.sleep(2)
    
    # Phase 3: Kill ALL vLLM-related processes
    print("  Killing all vLLM-related processes...")
    kill_commands = [
        f"fuser -k {port}/tcp",
        "pkill -9 -f 'vllm.*EngineCore'",
        "pkill -9 -f 'vllm_api_server'",
        "pkill -9 -f 'from vllm'",
        "pkill -9 -f 'multiprocessing.spawn'",
    ]
    for cmd in kill_commands:
        try:
            subprocess.run(cmd, shell=True, capture_output=True, timeout=5)
        except Exception:
            pass
    
    # Phase 4: Check for zombie GPU processes
    print(f"  Checking for zombie GPU processes on GPU {gpu_id}...")
    try:
        result = subprocess.run(
            f"nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits -i {gpu_id}",
            shell=True, capture_output=True, text=True, timeout=10
        )
        if result.stdout.strip():
            print(f"  Found GPU processes:\n{result.stdout}")
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 1:
                        pid = parts[0].strip()
                        if pid and pid != str(os.getpid()):
                            print(f"    Killing zombie GPU process: {pid}")
                            try:
                                subprocess.run(f"kill -9 {pid}", shell=True, timeout=5)
                            except Exception:
                                pass
    except Exception as e:
        print(f"  Warning: nvidia-smi check failed: {e}")
    
    # Phase 5: Wait for GPU memory release
    print("  Waiting for GPU memory release...")
    import torch
    for i in range(12):  # 60 seconds total
        time.sleep(5)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free_mem = torch.cuda.mem_get_info()[0] / 1e9
            total_mem = torch.cuda.mem_get_info()[1] / 1e9
            print(f"    [{(i+1)*5}s] GPU memory: {free_mem:.1f}/{total_mem:.1f} GB free ({100*free_mem/total_mem:.0f}%)")
            if free_mem > total_mem * 0.5:
                print(f"  ✓ Sufficient memory available ({free_mem:.1f} GB)")
                break
    
    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free_mem = torch.cuda.mem_get_info()[0] / 1e9
        total_mem = torch.cuda.mem_get_info()[1] / 1e9
        print(f"  ✓ Final GPU memory: {free_mem:.1f}/{total_mem:.1f} GB free ({100*free_mem/total_mem:.0f}%)")
    
    print("  ✓ vLLM terminated")


def main():
    parser = argparse.ArgumentParser(description="Test vLLM restart cycle")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--port", type=int, default=9099)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--memory-util", type=float, default=0.3)
    parser.add_argument("--restarts", type=int, default=3, help="Number of restart cycles")
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    import torch
    
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
    
    # Find server script (relative to this script's location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    server_script = os.path.join(os.path.dirname(script_dir), "vllm_api_server.py")
    
    if not os.path.exists(server_script):
        print(f"ERROR: Cannot find vllm_api_server.py at {server_script}")
        return 1
    
    log_dir = "/tmp/vllm_restart_test"
    os.makedirs(log_dir, exist_ok=True)
    
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
        
        # Launch vLLM
        print(f"\n[{cycle+1}] Launching vLLM...")
        cmd = [
            "python", server_script,
            "--model", args.model,
            "--port", str(args.port),
            "--gpu-memory-utilization", str(args.memory_util),
            "--max-model-len", "4096",
        ]
        print(f"  Command: {' '.join(cmd)}")
        
        log_file = f"{log_dir}/vllm_cycle_{cycle}.log"
        with open(log_file, "w") as f:
            proc = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
                start_new_session=True,  # Creates new process group for easy cleanup
            )
        print(f"  PID: {proc.pid} (process group: {os.getpgid(proc.pid)})")
        print(f"  Log: {log_file}")
        
        # Wait for vLLM to be ready
        print(f"  Waiting for vLLM to be ready...")
        start_time = time.time()
        if wait_for_vllm_ready(args.port, timeout=300):
            elapsed = time.time() - start_time
            print(f"  ✓ vLLM ready in {elapsed:.1f}s")
        else:
            print(f"  ✗ vLLM failed to start!")
            print(f"  Check log: tail -50 {log_file}")
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
        terminate_vllm(proc, args.port)
    
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
