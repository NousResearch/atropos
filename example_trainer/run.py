#!/usr/bin/env python3
"""
Unified GRPO trainer with integrated vLLM server (shared_vllm mode).

Combines vLLM server startup and trainer into a single command:
    python example_trainer/run.py --model Qwen/Qwen3-4B-Instruct --training-steps 20

This script:
1. Starts vLLM server with shared weights enabled
2. Waits for vLLM to be ready and bridge config to be created
3. Starts the GRPO trainer in shared_vllm mode
4. Handles cleanup on exit

For other modes (legacy, LoRA), use grpo.py instead.
"""

import atexit
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

from .cli import create_unified_parser, config_from_args
from .config import TrainingConfig
from .trainers import train_shared_vllm


def wait_for_vllm(port: int, timeout: int = 300) -> bool:
    """Wait for vLLM server to be ready."""
    print(f"[Run] Waiting for vLLM server on port {port}...")
    start = time.time()
    
    while time.time() - start < timeout:
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            if response.status_code == 200:
                print(f"[Run] ✓ vLLM server is ready (took {time.time() - start:.1f}s)")
                return True
        except requests.exceptions.ConnectionError:
            pass
        except Exception as e:
            print(f"[Run] Health check error: {e}")
        
        time.sleep(2)
    
    print(f"[Run] ✗ vLLM server failed to start within {timeout}s")
    return False


def wait_for_bridge_config(config_path: str, timeout: int = 60) -> bool:
    """Wait for vLLM bridge config to be created."""
    print(f"[Run] Waiting for bridge config at {config_path}...")
    start = time.time()
    
    while time.time() - start < timeout:
        if os.path.exists(config_path):
            try:
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                if config.get('ipc_handles') and len(config['ipc_handles']) > 0:
                    print(f"[Run] ✓ Bridge config ready with {len(config['ipc_handles'])} IPC handles")
                    return True
            except Exception:
                pass
        time.sleep(1)
    
    print(f"[Run] ✗ Bridge config not created within {timeout}s")
    return False


def main():
    # Parse args using shared CLI module
    parser = create_unified_parser()
    args = parser.parse_args()
    
    # Create log directory
    log_dir = getattr(args, 'log_dir', './logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Bridge config path
    bridge_config_path = "./vllm_bridge_config.json"
    
    # Clean up old bridge config
    if os.path.exists(bridge_config_path):
        os.remove(bridge_config_path)
        print(f"[Run] Removed old bridge config")
    
    # === Print Configuration ===
    print(f"\n{'='*60}")
    print("STARTING UNIFIED GRPO TRAINER (shared_vllm mode)")
    print(f"{'='*60}")
    print(f"Model: {args.model_name}")
    print(f"vLLM port: {args.vllm_port}")
    print(f"GPU memory utilization: {args.gpu_memory_utilization}")
    print(f"Training steps: {args.training_steps}")
    print(f"Optimizer: {args.optimizer}")
    print(f"GRPO: kl_coef={args.kl_coef}, clip_eps={args.clip_eps}")
    print(f"{'='*60}\n")
    
    # Get the path to vllm_api_server.py
    script_dir = Path(__file__).parent
    vllm_server_script = script_dir / "vllm_api_server.py"
    
    if not vllm_server_script.exists():
        print(f"[Run] ✗ vLLM server script not found at {vllm_server_script}")
        sys.exit(1)
    
    # Extract device index from args.device
    device_index = "0"
    if ":" in args.device:
        device_index = args.device.split(":")[1]
    
    # Build vLLM environment
    vllm_env = os.environ.copy()
    vllm_env["VLLM_ENABLE_SHARED_WEIGHTS"] = "1"
    vllm_env["VLLM_BRIDGE_CONFIG_PATH"] = bridge_config_path
    vllm_env["CUDA_VISIBLE_DEVICES"] = device_index
    vllm_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Build vLLM command
    vllm_cmd = [
        sys.executable, "-u", str(vllm_server_script),
        "--model", args.model_name,
        "--port", str(args.vllm_port),
        "--dtype", args.dtype,
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-model-len", str(args.max_model_len),
        "--enforce-eager",  # Required for shared weights
    ]
    
    vllm_log_path = os.path.join(log_dir, "vllm.log")
    print(f"[Run] Starting vLLM server (log: {vllm_log_path})...")
    
    vllm_log = open(vllm_log_path, "w")
    vllm_process = subprocess.Popen(
        vllm_cmd,
        env=vllm_env,
        stdout=vllm_log,
        stderr=subprocess.STDOUT,
    )
    
    # Register cleanup
    def cleanup():
        print("\n[Run] Cleaning up...")
        if vllm_process.poll() is None:
            print("[Run] Terminating vLLM server...")
            vllm_process.terminate()
            try:
                vllm_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                vllm_process.kill()
        vllm_log.close()
        print("[Run] Cleanup complete.")
    
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
    
    # Wait for vLLM to be ready
    if not wait_for_vllm(args.vllm_port, timeout=500):
        print("[Run] ✗ vLLM server failed to start. Check logs at:", vllm_log_path)
        sys.exit(1)
    
    # Wait for bridge config
    if not wait_for_bridge_config(bridge_config_path, timeout=60):
        print("[Run] ✗ Bridge config not created. Check vLLM logs.")
        sys.exit(1)
    
    # === Start Trainer ===
    print(f"\n[Run] Starting GRPO trainer...")
    
    # Build config - override some fields for shared_vllm mode
    config = TrainingConfig(
        model_name=args.model_name,
        lr=args.lr,
        training_steps=args.training_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optimizer=args.optimizer,
        device="cuda:0",  # Always 0 since we set CUDA_VISIBLE_DEVICES
        save_path=args.save_path,
        checkpoint_interval=args.checkpoint_interval,
        # GRPO hyperparameters
        kl_coef=args.kl_coef,
        clip_eps=args.clip_eps,
        use_reference_logprobs=not getattr(args, "no_reference_logprobs", False),
        # vLLM settings
        vllm_port=args.vllm_port,
        vllm_gpu_memory_utilization=args.gpu_memory_utilization,
        vllm_config_path=bridge_config_path,
        # Mode settings
        weight_bridge_mode="shared_vllm",  # Always shared_vllm for run.py
        atropos_url=args.atropos_url,
        # Logging
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        benchmark=True,  # Always show timing info for run.py
        debug_loading=getattr(args, "debug_loading", False),
    )
    
    try:
        train_shared_vllm(config)
        print("\n[Run] ✓ Training completed successfully!")
    except Exception as e:
        print(f"\n[Run] ✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
