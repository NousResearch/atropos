"""
Training mode implementations for GRPO trainer.

Contains the four main training modes:
- train_legacy: Checkpoint-based training with vLLM restarts
- train_shared_vllm: Single-copy mode with CUDA IPC
- train_lora: LoRA adapter training with HTTP hot-swap
- train_lora_restart: LoRA training with vLLM restarts (FAST mode)
"""

import os
import subprocess
import sys
import time
from typing import Iterable, Optional

import requests
import torch
from torch.optim import AdamW

from .api import check_atropos_api, register_trainer


def create_optimizer(model: torch.nn.Module, config) -> torch.optim.Optimizer:
    """
    Create optimizer based on config.optimizer setting.

    Options:
    - 'adamw': Standard AdamW (full precision, ~32GB GPU for 8B model)
    - 'adamw_8bit': 8-bit AdamW from bitsandbytes (~8GB GPU, requires bitsandbytes)
    - 'adafactor': Adafactor without momentum (~8GB GPU, no extra dependencies)
    """
    return create_optimizer_for_params(model.parameters(), config)


def create_optimizer_for_params(
    params: Iterable[torch.nn.Parameter], config
) -> torch.optim.Optimizer:
    """Create optimizer for a specific parameter iterable."""
    params = list(params)

    if config.optimizer == "adamw_8bit":
        try:
            import bitsandbytes as bnb

            optimizer = bnb.optim.AdamW8bit(params, lr=config.lr)
            print("[Setup] Using 8-bit AdamW (saves ~24GB optimizer memory)")
            return optimizer
        except ImportError:
            print("[Setup] WARNING: bitsandbytes not installed, falling back to AdamW")
            print("[Setup] Install with: pip install bitsandbytes")

    if config.optimizer == "adafactor":
        try:
            from transformers.optimization import Adafactor

            optimizer = Adafactor(
                params,
                lr=config.lr,
                scale_parameter=False,
                relative_step=False,
            )
            print("[Setup] Using Adafactor (no momentum, saves ~24GB)")
            return optimizer
        except ImportError:
            print("[Setup] WARNING: transformers Adafactor not available, using AdamW")

    # Default: standard AdamW
    optimizer = AdamW(params, lr=config.lr)
    print("[Setup] Using standard AdamW (requires ~32GB for optimizer states)")
    return optimizer


from .checkpointing import save_checkpoint, save_lora_checkpoint  # noqa: E402
from .config import TrainingConfig  # noqa: E402
from .data import get_data  # noqa: E402
from .model import PEFT_AVAILABLE, load_model_and_tokenizer  # noqa: E402
from .training import (  # noqa: E402
    finalize_training,
    log_metrics,
    run_training_step,
    setup_wandb,
)
from .vllm_manager import (  # noqa: E402
    check_vllm_health,
    check_vllm_process_health,
    launch_vllm_server,
    set_vllm_process,
    terminate_vllm_process,
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
    optimizer = create_optimizer(model, config)

    print("\n" + "=" * 60)
    print("LEGACY MODE (checkpoint + vLLM restart)")
    print("=" * 60)
    print(f"Training for {config.training_steps} steps on {config.device}")
    print(f"vLLM restart interval: every {config.vllm_restart_interval} steps")
    print(f"Save path: {config.save_path}")
    print("=" * 60 + "\n")

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

        # Fetch data (with inference logprobs for proper GRPO)
        data_fetch_start = time.time()
        if len(batches) == 0:
            batches, _ = get_data(
                config.batch_size,
                config.seq_len,
                config.atropos_url,
                extract_inference_logprobs=True,
            )
        batch_data = batches.pop(0)
        token_batches, label_batches, advantage_batches, temperature_batches = (
            batch_data[:4]
        )
        inference_logprob_batches = batch_data[4] if len(batch_data) > 4 else None
        distill_token_id_batches = batch_data[5] if len(batch_data) > 5 else None
        distill_logprob_batches = batch_data[6] if len(batch_data) > 6 else None
        data_fetch_time = time.time() - data_fetch_start
        benchmark_stats["data_fetch_times"].append(data_fetch_time)

        # Check if we should sync (save checkpoint + restart vLLM)
        should_sync = (
            step + 1
        ) % config.vllm_restart_interval == 0 or step == config.training_steps - 1
        if should_sync:
            terminate_vllm_process()

        # Training step (with proper GRPO using inference logprobs)
        step_start = time.time()
        metrics = run_training_step(
            model,
            optimizer,
            token_batches,
            label_batches,
            advantage_batches,
            temperature_batches,
            config,
            inference_logprob_batches=inference_logprob_batches,
            distill_token_id_batches=distill_token_id_batches,
            distill_logprob_batches=distill_logprob_batches,
        )
        step_time = time.time() - step_start
        benchmark_stats["step_times"].append(step_time)

        # GPU memory tracking
        gpu_mem_gb = (
            torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        )
        gpu_mem_reserved_gb = (
            torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        )
        benchmark_stats["gpu_memories"].append(gpu_mem_gb)

        # Sync (checkpoint + restart)
        sync_time = 0
        if should_sync:
            sync_start = time.time()
            checkpoint_path = save_checkpoint(
                model, tokenizer, config.save_path, step + 1
            )
            torch.cuda.empty_cache()
            vllm_proc = launch_vllm_server(config, checkpoint_path)
            set_vllm_process(vllm_proc)
            sync_time = time.time() - sync_start
            benchmark_stats["sync_times"].append(sync_time)

        # Update metrics
        metrics.update(
            {
                "step_time": step_time,
                "sync_time": sync_time,
                "data_fetch_time": data_fetch_time,
                "gpu_memory_gb": gpu_mem_gb,
                "gpu_memory_reserved_gb": gpu_mem_reserved_gb,
            }
        )

        log_metrics(metrics, step + 1, use_wandb, benchmark=config.benchmark)
        check_vllm_process_health()

    # === Cleanup ===
    save_checkpoint(
        model, tokenizer, config.save_path, config.training_steps, is_final=True
    )
    finalize_training(
        use_wandb,
        training_start_time,
        "legacy",
        config.training_steps,
        benchmark_stats,
        config.benchmark,
    )


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

    print("\n" + "=" * 60)
    print("SINGLE-COPY MODE (CUDA IPC)")
    print(">>> Trainer uses vLLM's tensors directly!")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Save path: {config.save_path}")
    print("=" * 60 + "\n")

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

    optimizer = create_optimizer(model, config)

    # === Real-time weight sharing verification ===
    print("\n[Weight Sharing Verification]")

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
    for step in range(config.training_steps):
        print(f"\nStep {step+1}/{config.training_steps}")

        # Fetch data (with inference logprobs for proper GRPO loss)
        data_fetch_start = time.time()
        if len(batches) == 0:
            batches, _ = get_data(
                config.batch_size,
                config.seq_len,
                config.atropos_url,
                extract_inference_logprobs=True,  # Enable proper GRPO with reference logprobs
            )
        batch_data = batches.pop(0)
        token_batches, label_batches, advantage_batches, temperature_batches = (
            batch_data[:4]
        )
        inference_logprob_batches = batch_data[4] if len(batch_data) > 4 else None
        distill_token_id_batches = batch_data[5] if len(batch_data) > 5 else None
        distill_logprob_batches = batch_data[6] if len(batch_data) > 6 else None
        data_fetch_time = time.time() - data_fetch_start
        benchmark_stats["data_fetch_times"].append(data_fetch_time)

        # Training step with proper GRPO (importance sampling + KL penalty)
        step_start = time.time()
        metrics = run_training_step(
            model,
            optimizer,
            token_batches,
            label_batches,
            advantage_batches,
            temperature_batches,
            config,
            inference_logprob_batches=inference_logprob_batches,  # Pass for GRPO ratio computation
            distill_token_id_batches=distill_token_id_batches,
            distill_logprob_batches=distill_logprob_batches,
        )
        step_time = time.time() - step_start
        benchmark_stats["step_times"].append(step_time)

        # GPU memory tracking
        gpu_mem_gb = (
            torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        )
        gpu_mem_reserved_gb = (
            torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        )
        benchmark_stats["gpu_memories"].append(gpu_mem_gb)

        # In single-copy mode, weights are updated in-place (no sync needed!)
        sync_time = 0.0
        print(f"  [SINGLE-COPY] Weights updated in-place - step {step+1}")
        benchmark_stats["sync_times"].append(sync_time)

        # Update metrics
        metrics.update(
            {
                "step_time": step_time,
                "sync_time": sync_time,
                "data_fetch_time": data_fetch_time,
                "gpu_memory_gb": gpu_mem_gb,
                "gpu_memory_reserved_gb": gpu_mem_reserved_gb,
            }
        )

        log_metrics(metrics, step + 1, use_wandb, benchmark=config.benchmark)

        # Periodic checkpoint (for recovery, not for vLLM sync)
        if (
            config.checkpoint_interval > 0
            and (step + 1) % config.checkpoint_interval == 0
        ):
            save_checkpoint(model, tokenizer, config.save_path, step + 1)

    # === Cleanup ===
    save_checkpoint(
        model, tokenizer, config.save_path, config.training_steps, is_final=True
    )
    finalize_training(
        use_wandb,
        training_start_time,
        "shared_vllm",
        config.training_steps,
        benchmark_stats,
        config.benchmark,
    )


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
        raise RuntimeError(
            "PEFT library required for LoRA mode. Install with: pip install peft"
        )

    training_start_time = time.time()

    # === Setup ===
    use_wandb = setup_wandb(config)

    print("\n" + "=" * 60)
    print("LORA MODE (adapter-only training)")
    print("=" * 60)
    print(f"Base model: {config.model_name}")
    print(f"LoRA config: r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"Save path: {config.save_path}")
    print(f"vLLM port: {config.vllm_port}")
    print("=" * 60 + "\n")

    # Check external vLLM server
    print("[1/3] Checking external vLLM server...")
    if not check_vllm_health(config.vllm_port):
        print(f"\nERROR: vLLM server not running on port {config.vllm_port}")
        print("\nLoRA mode requires an external vLLM server. Start it first:")
        print(
            f"  python example_trainer/vllm_api_server.py --model {config.model_name} "
            f"--port {config.vllm_port} --enable-lora --enforce-eager"
        )
        raise RuntimeError(f"External vLLM server required on port {config.vllm_port}")
    print(f"vLLM server healthy on port {config.vllm_port}")

    # Load model with LoRA adapters
    print("[2/3] Loading model with LoRA adapters...")
    model, tokenizer = load_model_and_tokenizer(config)

    # Only optimize LoRA parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = create_optimizer_for_params(trainable_params, config)

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

        # Fetch data (with inference logprobs for proper GRPO)
        data_fetch_start = time.time()
        if len(batches) == 0:
            batches, _ = get_data(
                config.batch_size,
                config.seq_len,
                config.atropos_url,
                extract_inference_logprobs=True,
            )
        batch_data = batches.pop(0)
        token_batches, label_batches, advantage_batches, temperature_batches = (
            batch_data[:4]
        )
        inference_logprob_batches = batch_data[4] if len(batch_data) > 4 else None
        distill_token_id_batches = batch_data[5] if len(batch_data) > 5 else None
        distill_logprob_batches = batch_data[6] if len(batch_data) > 6 else None
        data_fetch_time = time.time() - data_fetch_start
        benchmark_stats["data_fetch_times"].append(data_fetch_time)

        # Training step with proper GRPO
        step_start = time.time()
        metrics = run_training_step(
            model,
            optimizer,
            token_batches,
            label_batches,
            advantage_batches,
            temperature_batches,
            config,
            inference_logprob_batches=inference_logprob_batches,
            distill_token_id_batches=distill_token_id_batches,
            distill_logprob_batches=distill_logprob_batches,
        )
        step_time = time.time() - step_start
        benchmark_stats["step_times"].append(step_time)

        # GPU memory tracking
        gpu_mem_gb = (
            torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        )
        gpu_mem_reserved_gb = (
            torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        )
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
        metrics.update(
            {
                "step_time": step_time,
                "sync_time": sync_time,
                "data_fetch_time": data_fetch_time,
                "gpu_memory_gb": gpu_mem_gb,
                "gpu_memory_reserved_gb": gpu_mem_reserved_gb,
            }
        )

        log_metrics(metrics, step + 1, use_wandb, benchmark=config.benchmark)

    # === Cleanup ===
    final_sync_start = time.time()
    final_adapter_path = save_lora_checkpoint(
        model, config.save_path, config.training_steps, is_final=True
    )
    _hotswap_lora_adapter(config.vllm_port, final_adapter_path, "final")
    final_sync_time = time.time() - final_sync_start
    benchmark_stats["sync_times"].append(final_sync_time)

    finalize_training(
        use_wandb,
        training_start_time,
        "lora_only",
        config.training_steps,
        benchmark_stats,
        config.benchmark,
    )

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


def train_lora_restart(config: TrainingConfig):
    """
    GRPO training with LoRA adapters using vLLM restarts (FAST mode).

    This mode:
    1. Freezes base model, trains only LoRA adapter weights
    2. Runs vLLM WITHOUT --enforce-eager (keeps some CUDA optimizations)
    3. Restarts vLLM every N steps with the new adapter pre-loaded

    Performance comparison (Qwen3-4B @ 8k context):
    - lora_only (--enforce-eager): ~13 TPS (SLOW - CUDA graphs disabled)
    - lora_restart (no --enforce-eager): ~108 TPS (8x FASTER)
    - base model (no LoRA): ~172 TPS (baseline)

    The restart overhead (~45s) is much less than the 8x inference slowdown.

    Requirements:
    - No external vLLM needed - this mode manages vLLM internally
    - Requires PEFT library for LoRA
    """
    if not PEFT_AVAILABLE:
        raise RuntimeError(
            "PEFT library required for LoRA mode. Install with: pip install peft"
        )

    training_start_time = time.time()

    # === Setup ===
    use_wandb = setup_wandb(config)

    print("\n" + "=" * 60)
    print("LORA RESTART MODE (fast inference with CUDA graphs)")
    print("=" * 60)
    print(f"Base model: {config.model_name}")
    print(f"LoRA config: r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"Save path: {config.save_path}")
    print(f"vLLM port: {config.vllm_port}")
    print(f"Restart interval: every {config.vllm_restart_interval} steps")
    print("=" * 60)
    print("NOTE: This mode restarts vLLM without --enforce-eager for faster inference.")
    print("      Expected: ~108 TPS (vs ~13 TPS with --enforce-eager = 8x speedup)")
    print("=" * 60 + "\n")

    # Load model with LoRA adapters for training
    print("[1/4] Loading model with LoRA adapters...")
    model, tokenizer = load_model_and_tokenizer(config)

    # Only optimize LoRA parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = create_optimizer_for_params(trainable_params, config)

    os.makedirs(config.save_path, exist_ok=True)

    # Save initial adapter
    print("[2/4] Saving initial LoRA adapter...")
    initial_adapter_path = save_lora_checkpoint(model, config.save_path, 0)
    current_adapter_path = initial_adapter_path

    # Launch vLLM with the initial adapter
    print("[3/4] Launching vLLM with CUDA graphs (no --enforce-eager)...")
    vllm_proc = _launch_vllm_with_lora(config, current_adapter_path)
    if vllm_proc is None:
        raise RuntimeError("Failed to launch vLLM")

    print(f"[4/4] Starting training for {config.training_steps} steps")
    print("-" * 60)

    # Check Atropos API
    if not check_atropos_api(url=config.atropos_url, timeout=30):
        _terminate_vllm(vllm_proc, config.vllm_port)
        raise RuntimeError(f"Atropos API not reachable at {config.atropos_url}")
    register_trainer(config)

    # === Benchmark tracking ===
    benchmark_stats = {
        "step_times": [],
        "sync_times": [],
        "data_fetch_times": [],
        "gpu_memories": [],
        "restart_times": [],
    }

    # === Training Loop ===
    batches = []
    for step in range(config.training_steps):
        print(f"\nStep {step+1}/{config.training_steps}")

        # Fetch data (with inference logprobs for proper GRPO)
        data_fetch_start = time.time()
        if len(batches) == 0:
            batches, _ = get_data(
                config.batch_size,
                config.seq_len,
                config.atropos_url,
                extract_inference_logprobs=True,
            )
        batch_data = batches.pop(0)
        token_batches, label_batches, advantage_batches, temperature_batches = (
            batch_data[:4]
        )
        inference_logprob_batches = batch_data[4] if len(batch_data) > 4 else None
        distill_token_id_batches = batch_data[5] if len(batch_data) > 5 else None
        distill_logprob_batches = batch_data[6] if len(batch_data) > 6 else None
        data_fetch_time = time.time() - data_fetch_start
        benchmark_stats["data_fetch_times"].append(data_fetch_time)

        # Training step with proper GRPO
        step_start = time.time()
        metrics = run_training_step(
            model,
            optimizer,
            token_batches,
            label_batches,
            advantage_batches,
            temperature_batches,
            config,
            inference_logprob_batches=inference_logprob_batches,
            distill_token_id_batches=distill_token_id_batches,
            distill_logprob_batches=distill_logprob_batches,
        )
        step_time = time.time() - step_start
        benchmark_stats["step_times"].append(step_time)

        # GPU memory tracking
        gpu_mem_gb = (
            torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        )
        gpu_mem_reserved_gb = (
            torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        )
        benchmark_stats["gpu_memories"].append(gpu_mem_gb)

        # Periodic adapter save + vLLM restart
        sync_time = 0
        should_sync = (step + 1) % config.vllm_restart_interval == 0
        if should_sync and (step + 1) < config.training_steps:  # Don't restart on last step
            sync_start = time.time()
            
            # Save new adapter
            current_adapter_path = save_lora_checkpoint(model, config.save_path, step + 1)
            
            # Restart vLLM with new adapter
            print("  [RESTART] Restarting vLLM with new adapter...")
            _terminate_vllm(vllm_proc, config.vllm_port)
            vllm_proc = _launch_vllm_with_lora(config, current_adapter_path)
            if vllm_proc is None:
                raise RuntimeError("Failed to restart vLLM")
            
            sync_time = time.time() - sync_start
            benchmark_stats["sync_times"].append(sync_time)
            benchmark_stats["restart_times"].append(sync_time)
            print(f"  [RESTART] vLLM restarted in {sync_time:.1f}s")

        # Update metrics
        metrics.update(
            {
                "step_time": step_time,
                "sync_time": sync_time,
                "data_fetch_time": data_fetch_time,
                "gpu_memory_gb": gpu_mem_gb,
                "gpu_memory_reserved_gb": gpu_mem_reserved_gb,
            }
        )

        log_metrics(metrics, step + 1, use_wandb, benchmark=config.benchmark)

    # === Cleanup ===
    print("\nSaving final adapter...")
    final_sync_start = time.time()
    final_adapter_path = save_lora_checkpoint(
        model, config.save_path, config.training_steps, is_final=True
    )
    final_sync_time = time.time() - final_sync_start
    benchmark_stats["sync_times"].append(final_sync_time)

    # Terminate vLLM
    _terminate_vllm(vllm_proc, config.vllm_port)

    finalize_training(
        use_wandb,
        training_start_time,
        "lora_restart",
        config.training_steps,
        benchmark_stats,
        config.benchmark,
    )

    # Save tokenizer
    tokenizer_path = os.path.join(config.save_path, "tokenizer")
    tokenizer.save_pretrained(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")
    print(f"Final adapter saved to {final_adapter_path}")


# Global counter for vLLM restarts (for unique log files)
_vllm_restart_counter = 0


def _launch_vllm_with_lora(config: TrainingConfig, adapter_path: str) -> Optional[subprocess.Popen]:
    """
    Launch vLLM with a LoRA adapter (no --enforce-eager for faster inference).
    
    Unlike lora_only mode, this does NOT use --enforce-eager, so we get
    ~108 TPS instead of ~13 TPS (8x faster).
    """
    global _vllm_restart_counter
    from .vllm_manager import kill_process_on_port, wait_for_vllm_ready
    
    # Kill any existing process on the port
    print(f"  Cleaning up port {config.vllm_port}...")
    kill_process_on_port(config.vllm_port)
    
    # Clear CUDA cache before starting new vLLM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Wait for port and GPU memory to be fully released
    time.sleep(5)
    
    # Find the vllm_api_server.py script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    server_script = os.path.join(script_dir, "vllm_api_server.py")
    
    # Build command - NO --enforce-eager for faster inference (~108 TPS vs ~13 TPS)
    cmd = [
        sys.executable, server_script,
        "--model", config.model_name,
        "--port", str(config.vllm_port),
        "--gpu-memory-utilization", str(config.vllm_gpu_memory_utilization),
        "--max-model-len", str(config.max_model_len),
        "--enable-lora",
        "--max-lora-rank", str(max(config.lora_r * 2, 32)),
        # Note: NOT adding --enforce-eager - this gives us ~8x faster inference!
        # Without --enforce-eager, vLLM can use more optimizations.
    ]
    
    # Set environment for GPU selection
    env = os.environ.copy()
    if config.vllm_gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(config.vllm_gpu)
        print(f"  GPU: {config.vllm_gpu} (via CUDA_VISIBLE_DEVICES)")
    else:
        print("  GPU: Same as trainer (inherited CUDA_VISIBLE_DEVICES)")
    
    print(f"  Launching: {' '.join(cmd)}")
    print(f"  Adapter: {adapter_path}")
    
    # Log vLLM output to file for debugging (unique file per restart)
    vllm_log_path = os.path.join(config.save_path, f"vllm_restart_{_vllm_restart_counter}.log")
    _vllm_restart_counter += 1
    print(f"  vLLM log: {vllm_log_path}")
    
    try:
        vllm_log_file = open(vllm_log_path, "w")
        # Start in new session so we can kill entire process group later
        proc = subprocess.Popen(
            cmd, env=env, stdout=vllm_log_file, stderr=subprocess.STDOUT,
            start_new_session=True  # Creates new process group for easy cleanup
        )
        print(f"  vLLM PID: {proc.pid} (process group: {os.getpgid(proc.pid)})")
        print("  NOTE: vLLM without --enforce-eager compiles CUDA graphs on startup (takes 1-3 min)...")
        
        # Wait for server to be ready (longer timeout for CUDA graph compilation)
        if not wait_for_vllm_ready(config.vllm_port, timeout=300):
            print("  ERROR: vLLM failed to start after 300s")
            print(f"  Check log: {vllm_log_path}")
            # Print last 30 lines of the log
            try:
                with open(vllm_log_path, 'r') as f:
                    lines = f.readlines()
                    print("  Last 30 lines of vLLM log:")
                    for line in lines[-30:]:
                        print(f"    {line.rstrip()}")
            except Exception as e:
                print(f"  Could not read log: {e}")
            proc.terminate()
            return None
        
        # Load the LoRA adapter
        print("  Loading LoRA adapter...")
        try:
            resp = requests.post(
                f"http://localhost:{config.vllm_port}/lora/load",
                json={"adapter_path": adapter_path, "adapter_name": "training_adapter"},
                timeout=60,
            )
            if resp.status_code == 200:
                print("  ✓ Adapter loaded successfully")
            else:
                print(f"  WARNING: Adapter load returned {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"  WARNING: Could not load adapter: {e}")
            # Continue anyway - base model inference still works
        
        return proc
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def _terminate_vllm(proc: Optional[subprocess.Popen], port: int = 9001) -> None:
    """Terminate a vLLM process and release GPU resources."""
    import signal
    import subprocess as sp
    
    print(f"  Terminating vLLM on port {port}...")
    
    # Get current GPU device
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    
    # Phase 1: Kill the process group if we have a handle (kills all children too)
    main_pid = None
    if proc is not None:
        main_pid = proc.pid
        print(f"  Killing process group (PID: {main_pid})...")
        try:
            # Kill entire process group - this gets all child processes
            os.killpg(os.getpgid(main_pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception as e:
            print(f"  Warning: {e}")
    
    # Phase 2: Kill by port (catches anything still running)
    from .vllm_manager import kill_process_on_port
    kill_process_on_port(port)
    time.sleep(2)
    
    # Phase 3: Aggressively kill ALL vLLM-related processes
    print("  Killing all vLLM-related processes...")
    kill_commands = [
        f"fuser -k {port}/tcp",
        "pkill -9 -f 'vllm.*EngineCore'",
        "pkill -9 -f 'vllm_api_server'",
        "pkill -9 -f 'from vllm'",
        "pkill -9 -f 'multiprocessing.spawn'",
        "pkill -9 -f 'ray::IDLE'",  # Ray workers if any
    ]
    for cmd in kill_commands:
        try:
            sp.run(cmd, shell=True, capture_output=True, timeout=5)
        except Exception:
            pass
    
    # Phase 4: Use nvidia-smi to find and kill GPU processes (nuclear option)
    print(f"  Checking for zombie GPU processes on GPU {gpu_id}...")
    try:
        result = sp.run(
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
                        # Don't kill the current Python process (trainer)
                        if pid and pid != str(os.getpid()) and pid != str(main_pid):
                            print(f"    Killing zombie GPU process: {pid}")
                            try:
                                sp.run(f"kill -9 {pid}", shell=True, timeout=5)
                            except Exception:
                                pass
    except Exception as e:
        print(f"  Warning: nvidia-smi check failed: {e}")
    
    # Phase 5: Wait for GPU memory release - CRITICAL
    # The CUDA driver needs time to actually free memory after process death
    print("  Waiting for GPU memory release...")
    for i in range(12):  # 60 seconds total (longer wait)
        time.sleep(5)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free_mem = torch.cuda.mem_get_info()[0] / 1e9
            total_mem = torch.cuda.mem_get_info()[1] / 1e9
            print(f"    [{(i+1)*5}s] GPU memory: {free_mem:.1f}/{total_mem:.1f} GB free ({100*free_mem/total_mem:.0f}%)")
            # If we have enough memory (>50% free), break early
            if free_mem > total_mem * 0.5:
                print(f"  ✓ Sufficient memory available ({free_mem:.1f} GB)")
                break
    
    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free_mem = torch.cuda.mem_get_info()[0] / 1e9
        total_mem = torch.cuda.mem_get_info()[1] / 1e9
        print(f" Final GPU memory: {free_mem:.1f}/{total_mem:.1f} GB free ({100*free_mem/total_mem:.0f}%)")
        
        if free_mem < total_mem * 0.3:
            print("  WARNING: Low GPU memory! May fail to restart vLLM.")
            print("  Consider reducing --vllm-gpu-memory-utilization")
    
    print("  vLLM terminated")


