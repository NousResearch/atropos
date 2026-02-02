#!/usr/bin/env python3
"""
Unified GRPO trainer with integrated vLLM server.

Combines vLLM server startup and trainer into a single command:
    python -m example_trainer.run --model Qwen/Qwen3-4B-Instruct-2507 --training-steps 20

This script:
1. Starts vLLM server with shared weights enabled
2. Waits for vLLM to be ready and bridge config to be created
3. Starts the GRPO trainer
4. Handles cleanup on exit
"""

import argparse
import atexit
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests
import torch


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
            # Check if it has IPC handles
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
    parser = argparse.ArgumentParser(
        description="Unified GRPO trainer with integrated vLLM server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # === Model ===
    parser.add_argument(
        "--model", "--model-name",
        type=str,
        required=True,
        dest="model_name",
        help="Model to train (e.g., Qwen/Qwen3-4B-Instruct-2507)",
    )
    
    # === Training ===
    parser.add_argument("--training-steps", type=int, default=20, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--seq-len", type=int, default=2048, help="Max sequence length for training")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument(
        "--optimizer",
        choices=["adamw", "adamw_8bit", "adamw_cpu", "adafactor"],
        default="adamw_8bit",
        help="Optimizer to use",
    )
    parser.add_argument("--save-path", type=str, default="trained_model_checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=3)
    
    # === GRPO/PPO Hyperparameters ===
    parser.add_argument(
        "--kl-coef", type=float, default=0.1,
        help="KL divergence penalty coefficient. Higher = more conservative updates (default: 0.1)",
    )
    parser.add_argument(
        "--clip-eps", type=float, default=0.2,
        help="PPO clipping epsilon. Clips ratio to [1-eps, 1+eps] (default: 0.2)",
    )
    parser.add_argument(
        "--no-reference-logprobs", action="store_true",
        help="Disable use of inference logprobs as reference policy (not recommended)",
    )
    
    # === vLLM Server ===
    parser.add_argument("--vllm-port", type=int, default=9001, help="Port for vLLM server")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5, help="vLLM GPU memory fraction")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Max context length for vLLM")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "auto"])
    
    # === Atropos API ===
    parser.add_argument("--atropos-url", type=str, default="http://localhost:8002", help="Atropos API URL")
    
    # === Logging ===
    parser.add_argument("--wandb-project", type=str, default=None, help="W&B project name")
    parser.add_argument("--use-wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--log-dir", type=str, default="./logs", help="Directory for log files")
    
    # === Device ===
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device for training")
    
    args = parser.parse_args()
    
    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Bridge config path
    bridge_config_path = "./vllm_bridge_config.json"
    
    # Clean up old bridge config
    if os.path.exists(bridge_config_path):
        os.remove(bridge_config_path)
        print(f"[Run] Removed old bridge config")
    
    # === Start vLLM Server ===
    print(f"\n{'='*60}")
    print("STARTING UNIFIED GRPO TRAINER")
    print(f"{'='*60}")
    print(f"Model: {args.model_name}")
    print(f"vLLM port: {args.vllm_port}")
    print(f"Training steps: {args.training_steps}")
    print(f"Optimizer: {args.optimizer}")
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
    
    # Build vLLM command
    vllm_env = os.environ.copy()
    vllm_env["VLLM_ENABLE_SHARED_WEIGHTS"] = "1"
    vllm_env["VLLM_BRIDGE_CONFIG_PATH"] = bridge_config_path
    vllm_env["CUDA_VISIBLE_DEVICES"] = device_index
    vllm_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    vllm_cmd = [
        sys.executable, "-u", str(vllm_server_script),
        "--model", args.model_name,
        "--port", str(args.vllm_port),
        "--dtype", args.dtype,
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-model-len", str(args.max_model_len),
        "--enforce-eager",  # Required for shared weights
    ]
    
    vllm_log_path = os.path.join(args.log_dir, "vllm.log")
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
    if not wait_for_vllm(args.vllm_port, timeout=300):
        print("[Run] ✗ vLLM server failed to start. Check logs at:", vllm_log_path)
        sys.exit(1)
    
    # Wait for bridge config
    if not wait_for_bridge_config(bridge_config_path, timeout=60):
        print("[Run] ✗ Bridge config not created. Check vLLM logs.")
        sys.exit(1)
    
    # === Start Trainer ===
    print(f"\n[Run] Starting GRPO trainer...")
    
    # Import and run trainer directly (same process)
    # Use absolute imports since this script may be run directly
    from example_trainer.trainers import train_shared_vllm
    from example_trainer.config import TrainingConfig
    
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
        vllm_port=args.vllm_port,
        vllm_config_path=bridge_config_path,
        atropos_url=args.atropos_url,
        weight_bridge_mode="shared_vllm",
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        checkpoint_interval=args.checkpoint_interval,
        # GRPO hyperparameters
        kl_coef=args.kl_coef,
        clip_eps=args.clip_eps,
        use_reference_logprobs=not args.no_reference_logprobs,
        benchmark=True,  # Always show timing info
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
