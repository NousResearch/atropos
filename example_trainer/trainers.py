"""
Training mode implementations for GRPO trainer.

Contains the three main training modes:
- train_legacy: Checkpoint-based training with vLLM restarts
- train_shared_vllm: Single-copy mode with CUDA IPC
- train_lora: LoRA adapter training with hot-swap
"""

import os
import time
from typing import Optional

import requests
import torch
from torch.optim import AdamW

from .api import check_atropos_api, register_trainer
from .checkpointing import save_checkpoint, save_lora_checkpoint
from .config import TrainingConfig
from .data import get_data
from .model import load_model_and_tokenizer, PEFT_AVAILABLE
from .training import (
    finalize_training,
    log_metrics,
    run_training_step,
    setup_wandb,
)
from .vllm_manager import (
    check_vllm_health,
    check_vllm_process_health,
    launch_vllm_server,
    terminate_vllm_process,
    set_vllm_process,
)


def train_legacy(config: TrainingConfig):
    """
    Legacy GRPO training with periodic vLLM restarts.

    This mode:
    1. Trains model on trainer GPU
    2. Saves checkpoints periodically
    3. Restarts vLLM to load new weights

    Use for:
    - Simple setup
    - When trainer and vLLM on different GPUs
    """
    training_start_time = time.time()

    # === Setup ===
    use_wandb = setup_wandb(config)
    model, tokenizer = load_model_and_tokenizer(config)
    optimizer = AdamW(model.parameters(), lr=config.lr)

    print(f"\n{'='*60}")
    print("LEGACY MODE (checkpoint + vLLM restart)")
    print(f"{'='*60}")
    print(f"Training for {config.training_steps} steps on {config.device}")
    print(f"vLLM restart interval: every {config.vllm_restart_interval} steps")
    print(f"Save path: {config.save_path}")
    print(f"{'='*60}\n")

    os.makedirs(config.save_path, exist_ok=True)

    # Check Atropos API
    if not check_atropos_api(url=config.atropos_url, timeout=30):
        raise RuntimeError(f"Atropos API not reachable at {config.atropos_url}")
    register_trainer(config)

    # Launch initial vLLM server
    vllm_proc = launch_vllm_server(config, config.model_name)
    set_vllm_process(vllm_proc)

    # === Benchmark tracking ===
    benchmark_stats = {
        "step_times": [],
        "sync_times": [],
        "data_fetch_times": [],
        "gpu_memories": [],
    }

    # === Training Loop ===
    batches = []
    for step in range(config.training_steps):
        print(f"\nStep {step+1}/{config.training_steps}")

        # Fetch data
        data_fetch_start = time.time()
        if len(batches) == 0:
            batches, _ = get_data(config.batch_size, config.seq_len, config.atropos_url,
                                  extract_inference_logprobs=False)
        token_batches, label_batches, advantage_batches, temperature_batches = batches.pop(0)
        data_fetch_time = time.time() - data_fetch_start
        benchmark_stats["data_fetch_times"].append(data_fetch_time)

        # Check if we should sync (save checkpoint + restart vLLM)
        should_sync = (step + 1) % config.vllm_restart_interval == 0 or step == config.training_steps - 1
        if should_sync:
            terminate_vllm_process()

        # Training step
        step_start = time.time()
        metrics = run_training_step(
            model, optimizer,
            token_batches, label_batches, advantage_batches, temperature_batches,
            config,
        )
        step_time = time.time() - step_start
        benchmark_stats["step_times"].append(step_time)

        # GPU memory tracking
        gpu_mem_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        gpu_mem_reserved_gb = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        benchmark_stats["gpu_memories"].append(gpu_mem_gb)

        # Sync (checkpoint + restart)
        sync_time = 0
        if should_sync:
            sync_start = time.time()
            checkpoint_path = save_checkpoint(model, tokenizer, config.save_path, step + 1)
            torch.cuda.empty_cache()
            vllm_proc = launch_vllm_server(config, checkpoint_path)
            set_vllm_process(vllm_proc)
            sync_time = time.time() - sync_start
            benchmark_stats["sync_times"].append(sync_time)

        # Update metrics
        metrics.update({
            "step_time": step_time,
            "sync_time": sync_time,
            "data_fetch_time": data_fetch_time,
            "gpu_memory_gb": gpu_mem_gb,
            "gpu_memory_reserved_gb": gpu_mem_reserved_gb,
        })

        log_metrics(metrics, step + 1, use_wandb, benchmark=config.benchmark)
        check_vllm_process_health()

    # === Cleanup ===
    save_checkpoint(model, tokenizer, config.save_path, config.training_steps, is_final=True)
    finalize_training(use_wandb, training_start_time, "legacy", config.training_steps, benchmark_stats, config.benchmark)


def train_shared_vllm(config: TrainingConfig):
    """
    GRPO training with shared vLLM weights (single-copy mode).

    This mode:
    1. Attaches to vLLM's weight tensors via CUDA IPC
    2. optimizer.step() modifies vLLM's weights in-place
    3. vLLM immediately uses updated weights (no restart!)

    Requirements:
    - vLLM running with VLLM_ENABLE_SHARED_WEIGHTS=1
    - Trainer on same GPU(s) as vLLM
    """
    training_start_time = time.time()

    # === Setup ===
    use_wandb = setup_wandb(config)

    print(f"\n{'='*60}")
    print("SINGLE-COPY MODE (CUDA IPC)")
    print(">>> TRUE shared memory - only ONE model copy!")
    print(">>> Trainer uses vLLM's tensors directly!")
    print(f"{'='*60}")
    print(f"Model: {config.model_name}")
    print(f"Save path: {config.save_path}")
    print(f"{'='*60}\n")

    # Attach to vLLM's shared tensors
    print("[1/2] Attaching to vLLM's shared tensors...")
    model, tokenizer = load_model_and_tokenizer(config, single_copy=True)

    if model is None:
        raise RuntimeError(
            "Single-copy mode failed. Make sure:\n"
            "1. vLLM is running with VLLM_ENABLE_SHARED_WEIGHTS=1\n"
            "2. Trainer is on the SAME GPUs as vLLM\n"
            "3. vllm_bridge_config.json exists with IPC handles"
        )

    optimizer = AdamW(model.parameters(), lr=config.lr)

    # === Real-time weight sharing verification ===
    # This bypasses Atropos and directly tests if vLLM sees the same weights
    print("\n[Weight Sharing Verification - Real-time Test]")
    
    try:
        import requests
        
        # Test prompt
        test_prompt = "The capital of France is"
        test_tokens = tokenizer.encode(test_prompt, return_tensors="pt").to(model.device)
        
        # 1. Get logprobs from vLLM directly
        vllm_url = f"http://localhost:{config.vllm_port}"
        response = requests.post(
            f"{vllm_url}/generate",
            json={
                "prompt": test_prompt,
                "max_tokens": 5,
                "temperature": 1.0,
                "logprobs": 1,  # Return top-1 logprob
            },
            timeout=30,
        )
        
        if response.status_code == 200:
            result = response.json()
            # Extract logprobs for generated tokens
            vllm_logprobs = []
            vllm_tokens = []
            for item in result.get("logprobs", [[]])[0]:
                if item:
                    token_id = int(list(item.keys())[0])
                    logprob = float(list(item.values())[0])
                    vllm_logprobs.append(logprob)
                    vllm_tokens.append(token_id)
            
            if vllm_tokens:
                # 2. Compute same logprobs with trainer's model
                with torch.no_grad():
                    # Prepare input: prompt + generated tokens (except last)
                    full_input = torch.cat([
                        test_tokens,
                        torch.tensor([vllm_tokens[:-1]], device=model.device)
                    ], dim=1) if len(vllm_tokens) > 1 else test_tokens
                    
                    outputs = model(full_input)
                    logits = outputs.logits
                    
                    # Get logprobs for the positions where vLLM generated tokens
                    trainer_logprobs = []
                    for i, token_id in enumerate(vllm_tokens):
                        pos = test_tokens.shape[1] - 1 + i  # Position in sequence
                        if pos < logits.shape[1]:
                            log_probs = torch.log_softmax(logits[0, pos], dim=-1)
                            trainer_logprobs.append(log_probs[token_id].item())
                
                if trainer_logprobs and vllm_logprobs:
                    # Compare
                    vllm_mean = sum(vllm_logprobs[:len(trainer_logprobs)]) / len(trainer_logprobs)
                    trainer_mean = sum(trainer_logprobs) / len(trainer_logprobs)
                    diff = abs(vllm_mean - trainer_mean)
                    
                    print(f"  Test prompt: '{test_prompt}'")
                    print(f"  Generated tokens: {vllm_tokens}")
                    print(f"  vLLM logprobs:    {[f'{lp:.4f}' for lp in vllm_logprobs[:len(trainer_logprobs)]]}")
                    print(f"  Trainer logprobs: {[f'{lp:.4f}' for lp in trainer_logprobs]}")
                    print(f"  Mean diff: {diff:.4f}")
                    
                    if diff < 0.05:
                        print(f"  ✓ WEIGHTS ARE SHARED! Diff < 0.05")
                    elif diff < 0.2:
                        print(f"  ⚠ Small diff ({diff:.4f}) - likely numerical precision differences")
                    else:
                        print(f"  ✗ LARGE DIFF ({diff:.4f}) - weights may NOT be shared!")
                        print(f"    Check: Is vLLM running with --enforce-eager?")
                else:
                    print(f"  Could not compute comparison (no overlapping tokens)")
            else:
                print(f"  Could not extract vLLM logprobs from response")
        else:
            print(f"  Could not reach vLLM at {vllm_url} (status {response.status_code})")
            
    except Exception as e:
        print(f"  Real-time verification failed: {e}")
        print(f"  (This is optional - training will continue)")
    
    print(f"\n[2/2] Starting training for {config.training_steps} steps")
    print("NOTE: vLLM sees weight updates immediately after each step!")
    print("-" * 60)

    os.makedirs(config.save_path, exist_ok=True)

    # Check Atropos API
    print(f"\n[Setup] Connecting to Atropos API at {config.atropos_url}...")
    if not check_atropos_api(url=config.atropos_url, timeout=30):
        raise RuntimeError(f"Atropos API not reachable at {config.atropos_url}")
    register_trainer(config)

    # === Benchmark tracking ===
    benchmark_stats = {
        "step_times": [],
        "sync_times": [],
        "data_fetch_times": [],
        "gpu_memories": [],
    }

    # === Training Loop ===
    batches = []
    inference_logprobs = None
    for step in range(config.training_steps):
        print(f"\nStep {step+1}/{config.training_steps}")

        # Fetch data (with inference logprobs for alignment check)
        data_fetch_start = time.time()
        if len(batches) == 0:
            batches, inference_logprobs = get_data(
                config.batch_size, config.seq_len, config.atropos_url,
                extract_inference_logprobs=True,  # Enable logprob alignment check
            )
        token_batches, label_batches, advantage_batches, temperature_batches = batches.pop(0)
        data_fetch_time = time.time() - data_fetch_start
        benchmark_stats["data_fetch_times"].append(data_fetch_time)

        # Training step (with logprob alignment check)
        step_start = time.time()
        metrics = run_training_step(
            model, optimizer,
            token_batches, label_batches, advantage_batches, temperature_batches,
            config,
            inference_logprobs=inference_logprobs,  # Pass for alignment validation
        )
        step_time = time.time() - step_start
        benchmark_stats["step_times"].append(step_time)
        
        # Clear inference logprobs after use (will be refreshed with new data)
        inference_logprobs = None

        # GPU memory tracking
        gpu_mem_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        gpu_mem_reserved_gb = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        benchmark_stats["gpu_memories"].append(gpu_mem_gb)

        # In single-copy mode, weights are updated in-place (no sync needed!)
        sync_time = 0.0
        print(f"  [SINGLE-COPY] Weights updated in-place - step {step+1}")
        benchmark_stats["sync_times"].append(sync_time)

        # Update metrics
        metrics.update({
            "step_time": step_time,
            "sync_time": sync_time,
            "data_fetch_time": data_fetch_time,
            "gpu_memory_gb": gpu_mem_gb,
            "gpu_memory_reserved_gb": gpu_mem_reserved_gb,
        })

        log_metrics(metrics, step + 1, use_wandb, benchmark=config.benchmark)

        # Periodic checkpoint (for recovery, not for vLLM sync)
        if (step + 1) % config.vllm_restart_interval == 0:
            save_checkpoint(model, tokenizer, config.save_path, step + 1)

    # === Cleanup ===
    save_checkpoint(model, tokenizer, config.save_path, config.training_steps, is_final=True)
    finalize_training(use_wandb, training_start_time, "shared_vllm", config.training_steps, benchmark_stats, config.benchmark)


def train_lora(config: TrainingConfig):
    """
    GRPO training with LoRA adapters.

    This mode:
    1. Freezes base model, trains only LoRA adapter weights
    2. Saves lightweight adapter checkpoints
    3. Hot-swaps adapters in vLLM via API

    Benefits:
    - Much faster training (fewer parameters)
    - Smaller checkpoints
    - Adapters can be hot-swapped without restart

    Requirements:
    - External vLLM server running with --enable-lora
    """
    if not PEFT_AVAILABLE:
        raise RuntimeError("PEFT library required for LoRA mode. Install with: pip install peft")

    training_start_time = time.time()

    # === Setup ===
    use_wandb = setup_wandb(config)

    print(f"\n{'='*60}")
    print("LORA MODE (adapter-only training)")
    print(f"{'='*60}")
    print(f"Base model: {config.model_name}")
    print(f"LoRA config: r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"Save path: {config.save_path}")
    print(f"vLLM port: {config.vllm_port}")
    print(f"{'='*60}\n")

    # Check external vLLM server
    print("[1/3] Checking external vLLM server...")
    if not check_vllm_health(config.vllm_port):
        print(f"\nERROR: vLLM server not running on port {config.vllm_port}")
        print("\nLoRA mode requires an external vLLM server. Start it first:")
        print(f"  python example_trainer/vllm_api_server.py --model {config.model_name} "
              f"--port {config.vllm_port} --enable-lora --enforce-eager")
        raise RuntimeError(f"External vLLM server required on port {config.vllm_port}")
    print(f"vLLM server healthy on port {config.vllm_port}")

    # Load model with LoRA adapters
    print("[2/3] Loading model with LoRA adapters...")
    model, tokenizer = load_model_and_tokenizer(config)

    # Only optimize LoRA parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=config.lr)

    print(f"[3/3] Starting training for {config.training_steps} steps")
    print("-" * 60)

    os.makedirs(config.save_path, exist_ok=True)

    # Check Atropos API
    if not check_atropos_api(url=config.atropos_url, timeout=30):
        raise RuntimeError(f"Atropos API not reachable at {config.atropos_url}")
    register_trainer(config)

    # === Benchmark tracking ===
    benchmark_stats = {
        "step_times": [],
        "sync_times": [],
        "data_fetch_times": [],
        "gpu_memories": [],
    }

    # === Training Loop ===
    batches = []
    for step in range(config.training_steps):
        print(f"\nStep {step+1}/{config.training_steps}")

        # Fetch data
        data_fetch_start = time.time()
        if len(batches) == 0:
            batches, _ = get_data(config.batch_size, config.seq_len, config.atropos_url,
                                  extract_inference_logprobs=False)
        token_batches, label_batches, advantage_batches, temperature_batches = batches.pop(0)
        data_fetch_time = time.time() - data_fetch_start
        benchmark_stats["data_fetch_times"].append(data_fetch_time)

        # Training step
        step_start = time.time()
        metrics = run_training_step(
            model, optimizer,
            token_batches, label_batches, advantage_batches, temperature_batches,
            config,
        )
        step_time = time.time() - step_start
        benchmark_stats["step_times"].append(step_time)

        # GPU memory tracking
        gpu_mem_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        gpu_mem_reserved_gb = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        benchmark_stats["gpu_memories"].append(gpu_mem_gb)

        # Periodic adapter save + hot-swap
        sync_time = 0
        should_sync = (step + 1) % config.vllm_restart_interval == 0
        if should_sync:
            sync_start = time.time()
            adapter_path = save_lora_checkpoint(model, config.save_path, step + 1)
            _hotswap_lora_adapter(config.vllm_port, adapter_path, f"step_{step + 1}")
            sync_time = time.time() - sync_start
            benchmark_stats["sync_times"].append(sync_time)

        # Update metrics
        metrics.update({
            "step_time": step_time,
            "sync_time": sync_time,
            "data_fetch_time": data_fetch_time,
            "gpu_memory_gb": gpu_mem_gb,
            "gpu_memory_reserved_gb": gpu_mem_reserved_gb,
        })

        log_metrics(metrics, step + 1, use_wandb, benchmark=config.benchmark)

    # === Cleanup ===
    final_sync_start = time.time()
    final_adapter_path = save_lora_checkpoint(model, config.save_path, config.training_steps, is_final=True)
    _hotswap_lora_adapter(config.vllm_port, final_adapter_path, "final")
    final_sync_time = time.time() - final_sync_start
    benchmark_stats["sync_times"].append(final_sync_time)

    finalize_training(use_wandb, training_start_time, "lora_only", config.training_steps, benchmark_stats, config.benchmark)

    # Save tokenizer
    tokenizer_path = os.path.join(config.save_path, "tokenizer")
    tokenizer.save_pretrained(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")


def _hotswap_lora_adapter(
    port: int,
    adapter_path: str,
    adapter_name: Optional[str] = None,
) -> bool:
    """
    Request vLLM to hot-swap to a new LoRA adapter.

    Tries:
    1. Native vLLM endpoint: /v1/load_lora_adapter
    2. Custom endpoint: /lora/load
    """
    base_url = f"http://localhost:{port}"
    name = adapter_name or os.path.basename(adapter_path)

    # Try native vLLM endpoint first
    try:
        response = requests.post(
            f"{base_url}/v1/load_lora_adapter",
            json={"lora_name": name, "lora_path": adapter_path},
            timeout=30,
        )
        if response.status_code == 200:
            print(f"  [LORA] ✓ Hot-swapped adapter: {name}")
            return True
    except Exception:
        pass

    # Try custom endpoint
    try:
        response = requests.post(
            f"{base_url}/lora/load",
            json={"adapter_path": adapter_path, "adapter_name": name},
            timeout=30,
        )
        if response.status_code == 200:
            print(f"  [LORA] ✓ Hot-swapped adapter via custom API: {name}")
            return True
        else:
            print(f"  [LORA] ✗ Hot-swap failed: {response.text}")
            return False
    except Exception as e:
        print(f"  [LORA] ✗ Hot-swap request failed: {e}")
        return False

