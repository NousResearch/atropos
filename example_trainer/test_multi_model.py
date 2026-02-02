#!/usr/bin/env python3
"""
Multi-model test suite for shared_vllm trainer.

Tests the trainer against diverse models to verify robustness.
Supports both parallel (different GPUs) and sequential execution.

With --auto-env, each model gets its own isolated stack:
    - run-api (port 8002 + offset)
    - gsm8k environment (with model-specific tokenizer)
    - vLLM server (port 9001 + offset)
    - trainer

Usage:
    # RECOMMENDED: Fully automated parallel test (each model gets isolated stack)
    python -m example_trainer.test_multi_model \
        --models qwen3-4b hermes-8b nemotron-14b devstral-24b \
        --parallel \
        --gpus 0 1 2 3 \
        --auto-env
    
    # Sequential test on one GPU
    python -m example_trainer.test_multi_model \
        --models qwen3-4b hermes-8b \
        --sequential \
        --gpu 0 \
        --auto-env
    
    # Manual mode (you must start run-api and gsm8k_server yourself)
    # First start: run-api --port 8002 &
    # Then start gsm8k for your model
    python -m example_trainer.test_multi_model \
        --models qwen3-4b \
        --sequential \
        --gpu 0 \
        --atropos-url http://localhost:8002

Port allocation with --auto-env:
    Model 0: run-api:8002, vLLM:9001
    Model 1: run-api:8003, vLLM:9002
    Model 2: run-api:8004, vLLM:9003
    ...
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import threading


@dataclass
class ModelConfig:
    """Configuration for a test model."""
    name: str
    model_id: str
    gpu_memory_utilization: float = 0.5
    max_model_len: int = 4096
    dtype: str = "bfloat16"
    training_steps: int = 10
    notes: str = ""


# Define test models
# Memory estimates for B200 (183GB):
#   - Model weights (bf16): 2 bytes/param
#   - Gradients: ~same as weights
#   - 8-bit optimizer: ~1 byte/param
#   - KV cache: depends on max_model_len
TEST_MODELS: Dict[str, ModelConfig] = {
    "qwen3-4b": ModelConfig(
        name="qwen3-4b",
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        gpu_memory_utilization=0.4,  # ~73GB for vLLM
        max_model_len=8192,          # Plenty of room on B200
        notes="Small 4B model, good baseline test (~8GB weights)",
    ),
    "hermes-8b": ModelConfig(
        name="hermes-8b",
        model_id="NousResearch/Hermes-3-Llama-3.1-8B",
        gpu_memory_utilization=0.45,  # ~82GB for vLLM
        max_model_len=8192,           # 8K context fits well
        notes="Llama 8B architecture (~16GB weights)",
    ),
    "nemotron-14b": ModelConfig(
        name="nemotron-14b",
        model_id="nvidia/Nemotron-Cascade-14B-Thinking",
        gpu_memory_utilization=0.5,   # ~91GB for vLLM
        max_model_len=32768,          # 32K context for thinking
        notes="14B thinking model (~28GB weights), needs room for long CoT",
    ),
    "devstral-24b": ModelConfig(
        name="devstral-24b",
        model_id="mistralai/Devstral-Small-2-24B-Instruct-2512",
        gpu_memory_utilization=0.55,  # ~100GB for vLLM
        max_model_len=16384,          # 16K context (conservative for 24B)
        notes="Large 24B Mistral (~48GB weights), largest model",
    ),
}


def get_test_dir(base_dir: str, model_name: str, timestamp: str) -> Path:
    """Get unique test directory for a model run."""
    return Path(base_dir) / f"{model_name}_{timestamp}"


def start_run_api(
    port: int,
    log_path: Path,
) -> subprocess.Popen:
    """Start a run-api instance on a specific port."""
    cmd = [sys.executable, "-m", "atroposlib.cli.run_api", "--port", str(port)]
    
    log_file = open(log_path, "w")
    process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        # Don't buffer output
        bufsize=1,
    )
    return process


def wait_for_run_api(port: int, timeout: int = 60) -> bool:
    """Wait for run-api to be ready."""
    import requests
    start = time.time()
    while time.time() - start < timeout:
        try:
            # run-api uses /status or / endpoint, not /health
            resp = requests.get(f"http://localhost:{port}/status", timeout=5)
            if resp.status_code == 200:
                return True
        except:
            pass
        try:
            # Fallback to root endpoint
            resp = requests.get(f"http://localhost:{port}/", timeout=5)
            if resp.status_code == 200:
                return True
        except:
            pass
        time.sleep(2)
    return False


def start_gsm8k_env(
    model_id: str,
    vllm_port: int,
    run_api_port: int,
    log_path: Path,
    atropos_root: Path,
) -> subprocess.Popen:
    """Start a gsm8k environment process for a specific model."""
    gsm8k_script = atropos_root / "environments" / "gsm8k_server.py"
    cmd = [
        sys.executable, "-u", str(gsm8k_script), "serve",
        "--env.rollout_server_url", f"http://localhost:{run_api_port}",
        "--env.tokenizer_name", model_id,
        "--env.use_wandb", "false",
        "--env.total_steps", "10000",
        "--env.batch_size", "64",
        "--env.group_size", "8",
        "--openai.model_name", model_id,
        "--openai.base_url", f"http://localhost:{vllm_port}/v1",
        "--openai.api_key", "x",
        "--openai.server_type", "openai",
    ]
    
    log_file = open(log_path, "w")
    process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=str(atropos_root),  # Run from atropos root
    )
    return process


def run_model_test(
    model_config: ModelConfig,
    gpu_id: int,
    atropos_url: str,
    atropos_port: int,
    base_dir: str,
    timestamp: str,
    training_steps: int,
    vllm_port_offset: int = 0,
    auto_env: bool = False,
) -> Dict:
    """
    Run a complete training test for a single model.
    
    Returns dict with test results.
    """
    model_name = model_config.name
    test_dir = get_test_dir(base_dir, model_name, timestamp).resolve()  # Make absolute
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Unique paths for this model (all absolute)
    vllm_port = 9001 + vllm_port_offset
    bridge_config_path = test_dir / "vllm_bridge_config.json"
    checkpoint_dir = test_dir / "checkpoints"
    log_dir = test_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    vllm_log = log_dir / "vllm.log"
    trainer_log = log_dir / "trainer.log"
    
    # Each model gets unique ports
    run_api_port = 8002 + vllm_port_offset
    
    result = {
        "model": model_config.model_id,
        "model_name": model_name,
        "gpu": gpu_id,
        "vllm_port": vllm_port,
        "run_api_port": run_api_port,
        "test_dir": str(test_dir),
        "status": "pending",
        "error": None,
        "start_time": None,
        "end_time": None,
        "duration_seconds": None,
        "real_time_alignment": None,
        "final_gpu_memory": None,
    }
    
    print(f"\n{'='*60}")
    print(f"[{model_name}] Starting test on GPU {gpu_id}")
    print(f"[{model_name}] Model: {model_config.model_id}")
    print(f"[{model_name}] vLLM port: {vllm_port}")
    print(f"[{model_name}] Test dir: {test_dir}")
    print(f"{'='*60}\n")
    
    result["start_time"] = datetime.now().isoformat()
    start_time = time.time()
    
    env_process = None
    run_api_process = None
    trainer_process = None
    
    # Get atropos root directory (used for vLLM and gsm8k scripts)
    script_dir = Path(__file__).parent
    atropos_root = script_dir.parent.resolve()
    
    try:
        # === Start run-api (if auto_env) ===
        if auto_env:
            run_api_log = log_dir / "run_api.log"
            print(f"[{model_name}] Starting run-api on port {run_api_port}...")
            run_api_process = start_run_api(run_api_port, run_api_log)
            
            if not wait_for_run_api(run_api_port, timeout=60):
                # Check if process died
                if run_api_process.poll() is not None:
                    print(f"[{model_name}] run-api process exited with code {run_api_process.returncode}")
                # Print log contents for debugging
                if run_api_log.exists():
                    print(f"[{model_name}] run-api log contents:")
                    print(run_api_log.read_text()[-2000:])  # Last 2000 chars
                raise RuntimeError(f"run-api failed to start on port {run_api_port}")
            print(f"[{model_name}] ✓ run-api ready on port {run_api_port}")
            
            # Update atropos_url to use this model's run-api
            atropos_url = f"http://localhost:{run_api_port}"
        
        # === Start gsm8k Environment (if auto_env) ===
        if auto_env:
            env_log = log_dir / "env.log"
            print(f"[{model_name}] Starting gsm8k environment (tokenizer: {model_config.model_id})...")
            env_process = start_gsm8k_env(
                model_config.model_id, vllm_port, run_api_port, env_log, atropos_root
            )
            time.sleep(10)  # Give it time to initialize and connect
            print(f"[{model_name}] ✓ gsm8k environment started")
        
        # === Start Unified vLLM + Trainer (run.py) ===
        # Using run.py ensures vLLM is a CHILD of the trainer process,
        # which is required for CUDA IPC with ptrace_scope=1
        run_script = script_dir / "run.py"
        
        run_env = os.environ.copy()
        run_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        run_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        run_cmd = [
            sys.executable, "-u", str(run_script),
            "--model", model_config.model_id,
            "--vllm-port", str(vllm_port),
            "--gpu-memory-utilization", str(model_config.gpu_memory_utilization),
            "--max-model-len", str(model_config.max_model_len),
            "--dtype", model_config.dtype,
            "--atropos-url", atropos_url,
            "--training-steps", str(training_steps),
            "--optimizer", "adamw_8bit",
            "--save-path", str(checkpoint_dir),
            "--checkpoint-interval", "5",
            "--log-dir", str(log_dir),
        ]
        
        print(f"[{model_name}] Starting unified trainer (vLLM + GRPO) for {training_steps} steps...")
        with open(trainer_log, "w") as tlog:
            trainer_process = subprocess.Popen(
                run_cmd,
                env=run_env,
                stdout=tlog,
                stderr=subprocess.STDOUT,
                cwd=str(atropos_root),  # Run from atropos root
            )
            trainer_process.wait()
        
        if trainer_process.returncode != 0:
            raise RuntimeError(f"Unified trainer exited with code {trainer_process.returncode}")
        
        result["status"] = "success"
        print(f"[{model_name}] ✓ Training completed successfully!")
        
        # Parse trainer log for metrics
        try:
            with open(trainer_log, "r") as f:
                log_content = f.read()
            
            # Extract real-time alignment
            if "Mean diff:" in log_content:
                import re
                match = re.search(r"Mean diff: ([\d.]+)", log_content)
                if match:
                    result["real_time_alignment"] = float(match.group(1))
            
            # Extract final GPU memory
            if "GPU mem:" in log_content:
                matches = re.findall(r"GPU mem: ([\d.]+)GB", log_content)
                if matches:
                    result["final_gpu_memory"] = float(matches[-1])
        except Exception as e:
            print(f"[{model_name}] Warning: Could not parse log: {e}")
        
    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        print(f"[{model_name}] ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Note: vLLM is managed by run.py and cleaned up automatically
        
        # Cleanup gsm8k environment
        if env_process and env_process.poll() is None:
            print(f"[{model_name}] Terminating gsm8k environment...")
            env_process.terminate()
            try:
                env_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                env_process.kill()
        
        # Cleanup run-api
        if run_api_process and run_api_process.poll() is None:
            print(f"[{model_name}] Terminating run-api...")
            run_api_process.terminate()
            try:
                run_api_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                run_api_process.kill()
    
    result["end_time"] = datetime.now().isoformat()
    result["duration_seconds"] = time.time() - start_time
    
    return result


def run_parallel_tests(
    models: List[ModelConfig],
    gpu_ids: List[int],
    atropos_url: str,
    atropos_port: int,
    base_dir: str,
    training_steps: int,
    auto_env: bool = False,
) -> List[Dict]:
    """Run tests for multiple models in parallel."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []
    threads = []
    result_lock = threading.Lock()
    
    def run_and_store(model, gpu, port_offset):
        result = run_model_test(
            model, gpu, atropos_url, atropos_port, base_dir, timestamp,
            training_steps, port_offset, auto_env
        )
        with result_lock:
            results.append(result)
    
    # Start threads
    for i, (model, gpu) in enumerate(zip(models, gpu_ids)):
        t = threading.Thread(target=run_and_store, args=(model, gpu, i))
        t.start()
        threads.append(t)
        time.sleep(5)  # Stagger starts slightly
    
    # Wait for all to complete
    for t in threads:
        t.join()
    
    return results


def run_sequential_tests(
    models: List[ModelConfig],
    gpu_id: int,
    atropos_url: str,
    atropos_port: int,
    base_dir: str,
    training_steps: int,
    auto_env: bool = False,
) -> List[Dict]:
    """Run tests for multiple models sequentially on one GPU."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []
    
    for i, model in enumerate(models):
        result = run_model_test(
            model, gpu_id, atropos_url, atropos_port, base_dir, timestamp,
            training_steps, port_offset=0, auto_env=auto_env
        )
        results.append(result)
        
        # Give GPU time to fully release memory
        time.sleep(10)
    
    return results


def print_summary(results: List[Dict]):
    """Print summary of test results."""
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for r in results:
        status_icon = "✓" if r["status"] == "success" else "✗"
        duration = f"{r['duration_seconds']:.1f}s" if r['duration_seconds'] else "N/A"
        alignment = f"{r['real_time_alignment']:.4f}" if r['real_time_alignment'] else "N/A"
        memory = f"{r['final_gpu_memory']:.1f}GB" if r['final_gpu_memory'] else "N/A"
        
        print(f"\n{status_icon} {r['model_name']}")
        print(f"    Model: {r['model']}")
        print(f"    GPU: {r['gpu']}, vLLM port: {r['vllm_port']}, run-api port: {r.get('run_api_port', 'N/A')}")
        print(f"    Status: {r['status']}")
        print(f"    Duration: {duration}")
        print(f"    Real-time alignment: {alignment}")
        print(f"    GPU memory: {memory}")
        if r["error"]:
            print(f"    Error: {r['error']}")
        print(f"    Logs: {r['test_dir']}/logs/")
    
    # Summary stats
    successes = sum(1 for r in results if r["status"] == "success")
    failures = len(results) - successes
    
    print(f"\n{'='*80}")
    print(f"TOTAL: {successes} passed, {failures} failed")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-model test suite for shared_vllm trainer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all models in parallel (one per GPU)
    python -m example_trainer.test_multi_model --parallel
    
    # Run specific models
    python -m example_trainer.test_multi_model --models hermes-8b qwen3-4b --parallel
    
    # Run sequentially on GPU 0
    python -m example_trainer.test_multi_model --sequential --gpu 0
    
Available models: """ + ", ".join(TEST_MODELS.keys())
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(TEST_MODELS.keys()),
        default=["qwen3-4b", "hermes-8b"],
        help="Models to test",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run models in parallel on different GPUs",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run models sequentially on one GPU",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=None,
        help="GPU IDs to use (for parallel mode)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID (for sequential mode)",
    )
    parser.add_argument(
        "--atropos-url",
        type=str,
        default="http://localhost:8002",
        help="Atropos API URL",
    )
    parser.add_argument(
        "--atropos-port",
        type=int,
        default=8002,
        help="Atropos API port (for spawning multiple if needed)",
    )
    parser.add_argument(
        "--training-steps",
        type=int,
        default=10,
        help="Number of training steps per model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./multi_model_tests",
        help="Base directory for test outputs",
    )
    parser.add_argument(
        "--auto-env",
        action="store_true",
        help="Automatically start gsm8k environment for each model (requires run-api to be running)",
    )
    
    args = parser.parse_args()
    
    if not args.parallel and not args.sequential:
        args.sequential = True  # Default to sequential
    
    # Get model configs
    models = [TEST_MODELS[name] for name in args.models]
    
    print(f"\n{'#'*60}")
    print("# MULTI-MODEL SHARED_VLLM TRAINER TEST SUITE")
    print(f"{'#'*60}")
    print(f"\nModels to test: {[m.name for m in models]}")
    print(f"Mode: {'Parallel' if args.parallel else 'Sequential'}")
    print(f"Training steps per model: {args.training_steps}")
    print(f"Output directory: {args.output_dir}")
    print(f"Atropos URL: {args.atropos_url}")
    
    # Run tests
    if args.auto_env:
        print(f"Auto-env: Will start gsm8k environment per model")
    
    if args.parallel:
        gpus = args.gpus or list(range(len(models)))
        if len(gpus) < len(models):
            print(f"\nWarning: Not enough GPUs ({len(gpus)}) for models ({len(models)})")
            print("Some models will share GPUs")
            gpus = gpus * (len(models) // len(gpus) + 1)
        
        print(f"Using GPUs: {gpus[:len(models)]}")
        results = run_parallel_tests(
            models, gpus[:len(models)],
            args.atropos_url, args.atropos_port,
            args.output_dir, args.training_steps,
            auto_env=args.auto_env
        )
    else:
        print(f"Using GPU: {args.gpu}")
        results = run_sequential_tests(
            models, args.gpu,
            args.atropos_url, args.atropos_port,
            args.output_dir, args.training_steps,
            auto_env=args.auto_env
        )
    
    # Print summary
    print_summary(results)
    
    # Save results to JSON
    results_file = Path(args.output_dir) / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Exit with error code if any failed
    if any(r["status"] != "success" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
