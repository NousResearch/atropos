"""
Training mode implementations for GRPO trainer.

Contains the four main training modes:
- train_legacy: Checkpoint-based training with vLLM restarts
- train_shared_vllm: Single-copy mode with CUDA IPC
- train_lora: LoRA adapter training with HTTP hot-swap
- train_lora_nccl: LoRA adapter training with NCCL direct transfer (torchtitan-style)
"""

import os
import time
from typing import Optional

import requests
import torch
from torch.optim import AdamW

from .api import check_atropos_api, register_trainer


class CPUOffloadAdamW(torch.optim.Optimizer):
    """
    AdamW with optimizer states offloaded to CPU.

    Full precision (no quantization), but states stay on CPU RAM instead of GPU.
    Trade-off: Slower (~2x) but uses ~0GB GPU memory for optimizer states.
    """

    def __init__(
        self, params, lr=1e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def _init_state(self, p):
        """Lazily initialize state on CPU."""
        state = self.state[p]
        if len(state) == 0:
            state["step"] = 0
            # Store on CPU in FP32
            state["exp_avg"] = torch.zeros_like(p, device="cpu", dtype=torch.float32)
            state["exp_avg_sq"] = torch.zeros_like(p, device="cpu", dtype=torch.float32)
        return state

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self._init_state(p)

                state["step"] += 1

                # Move states to GPU for computation
                exp_avg = state["exp_avg"].to(p.device)
                exp_avg_sq = state["exp_avg_sq"].to(p.device)

                # AdamW update
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] / bias_correction1

                # Update weights
                denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(group["eps"])
                p.addcdiv_(exp_avg, denom, value=-step_size)

                # Weight decay
                if group["weight_decay"] != 0:
                    p.add_(p, alpha=-group["lr"] * group["weight_decay"])

                # Move states back to CPU (non-blocking for better perf)
                state["exp_avg"].copy_(exp_avg.cpu())
                state["exp_avg_sq"].copy_(exp_avg_sq.cpu())

        return loss


def create_optimizer(model: torch.nn.Module, config) -> torch.optim.Optimizer:
    """
    Create optimizer based on config.optimizer setting.

    Options:
    - 'adamw': Standard AdamW (full precision, ~32GB GPU for 4B model)
    - 'adamw_8bit': 8-bit AdamW from bitsandbytes (~8GB GPU, requires bitsandbytes)
    - 'adamw_cpu': AdamW with CPU offload (~0GB GPU, slower but full precision)
    - 'adafactor': Adafactor without momentum (~8GB GPU, no extra dependencies)
    """
    if config.optimizer == "adamw_8bit":
        try:
            import bitsandbytes as bnb

            optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=config.lr)
            print("[Setup] Using 8-bit AdamW (saves ~24GB optimizer memory)")
            return optimizer
        except ImportError:
            print("[Setup] WARNING: bitsandbytes not installed, falling back to AdamW")
            print("[Setup] Install with: pip install bitsandbytes")

    if config.optimizer == "adamw_cpu":
        optimizer = CPUOffloadAdamW(model.parameters(), lr=config.lr)
        print(
            "[Setup] Using AdamW with CPU offload (full precision, ~0GB GPU for states)"
        )
        print(
            "[Setup] NOTE: ~2x slower due to CPU<->GPU transfers, but no quantization"
        )
        return optimizer

    if config.optimizer == "adafactor":
        try:
            from transformers.optimization import Adafactor

            optimizer = Adafactor(
                model.parameters(),
                lr=config.lr,
                scale_parameter=False,
                relative_step=False,
            )
            print("[Setup] Using Adafactor (no momentum, saves ~24GB)")
            return optimizer
        except ImportError:
            print("[Setup] WARNING: transformers Adafactor not available, using AdamW")

    # Default: standard AdamW
    optimizer = AdamW(model.parameters(), lr=config.lr)
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


def train_lora_nccl(config: TrainingConfig):
    """
    GRPO training with LoRA adapters using NCCL direct weight transfer.
    
    This mode (inspired by torchtitan):
    1. Freezes base model, trains only LoRA adapter weights
    2. Uses NCCL to broadcast weights directly to vLLM (zero disk I/O)
    3. Weight updates are immediate - no HTTP API calls
    
    Benefits over train_lora():
    - Much faster weight sync (NCCL vs HTTP+disk)
    - Lower latency for on-policy training
    - No checkpoint files during training
    
    Requirements:
    - External vLLM server running with NCCL receiver enabled
    - Trainer and vLLM must be in the same NCCL process group
    """
    if not PEFT_AVAILABLE:
        raise RuntimeError(
            "PEFT library required for LoRA mode. Install with: pip install peft"
        )
    
    training_start_time = time.time()
    
    # === Setup ===
    use_wandb = setup_wandb(config)
    
    print("\n" + "=" * 60)
    print("LORA NCCL MODE (torchtitan-style direct weight transfer)")
    print("=" * 60)
    print(f"Base model: {config.model_name}")
    print(f"LoRA config: r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"Save path: {config.save_path}")
    print(f"vLLM port: {config.vllm_port}")
    print(f"NCCL init: {config.nccl_init_method}")
    print("=" * 60 + "\n")
    
    # Check external vLLM server
    print("[1/5] Checking external vLLM server...")
    if not check_vllm_health(config.vllm_port):
        print(f"\nERROR: vLLM server not running on port {config.vllm_port}")
        print("\nLoRA NCCL mode requires an external vLLM server. Start it first:")
        print(
            f"  python example_trainer/vllm_api_server.py "
            f"--model {config.model_name} --port {config.vllm_port} --enable-lora --enforce-eager"
        )
        raise RuntimeError(f"External vLLM server required on port {config.vllm_port}")
    print(f"vLLM server healthy on port {config.vllm_port}")
    
    # Load model with LoRA adapters
    print("[2/5] Loading model with LoRA adapters...")
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Only optimize LoRA parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=config.lr)
    
    # Import NCCL bridge components
    from .nccl_weight_bridge import (
        NCCLBridgeConfig,
        NCCLWeightBridge,
        create_trainer_param_to_vllm_mapping,
        export_bridge_config,
        get_lora_params,
    )
    
    # Pre-register params to get metadata for vLLM
    lora_params = get_lora_params(model)
    param_names = sorted(lora_params.keys())
    param_shapes = {name: list(p.shape) for name, p in lora_params.items()}
    param_dtypes = {name: str(p.dtype) for name, p in lora_params.items()}
    
    param_metadata = {
        "param_names": param_names,
        "param_shapes": param_shapes,
        "param_dtypes": param_dtypes,
        "num_params": len(param_names),
    }
    
    param_mappings = create_trainer_param_to_vllm_mapping(
        param_names,
        model_name=config.model_name
    )
    
    # Tell vLLM to start its NCCL receiver FIRST (it will join as rank 1)
    print("[3/5] Starting NCCL receiver on vLLM server...")
    vllm_base_url = f"http://localhost:{config.vllm_port}"
    try:
        response = requests.post(
            f"{vllm_base_url}/nccl/start_receiver",
            json={
                "init_method": config.nccl_init_method,
                "world_size": config.nccl_world_size,
                "param_metadata": param_metadata,
                "param_mappings": param_mappings,
            },
            timeout=30,
        )
        resp_data = response.json()
        if response.status_code != 200 or resp_data.get("status") == "error":
            raise RuntimeError(f"Failed to start NCCL receiver on vLLM: {resp_data}")
        print(f"  vLLM NCCL receiver started: {resp_data}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to contact vLLM server: {e}")
    
    # Wait for vLLM to be in "connecting" state
    import time as time_module
    print("  Waiting for vLLM NCCL receiver to initialize...")
    for i in range(10):
        time_module.sleep(1)
        try:
            status_resp = requests.get(f"{vllm_base_url}/nccl/status", timeout=5)
            status = status_resp.json()
            print(f"    vLLM NCCL status: {status.get('status', 'unknown')}")
            if status.get("status") == "error":
                raise RuntimeError(f"vLLM NCCL setup failed: {status.get('error')}")
            if status.get("status") in ["connecting", "connected"]:
                break
        except Exception as e:
            print(f"    Status check error: {e}")
    
    # Now setup trainer's NCCL bridge (joins as rank 0)
    print("[4/5] Setting up trainer NCCL weight bridge...")
    nccl_config = NCCLBridgeConfig(
        rank=0,  # Trainer is always rank 0
        world_size=config.nccl_world_size,
        init_method=config.nccl_init_method,
    )
    
    bridge = NCCLWeightBridge(nccl_config)
    if not bridge.setup():
        # Try to stop vLLM receiver on failure
        try:
            requests.post(f"{vllm_base_url}/nccl/stop_receiver", timeout=5)
        except Exception:
            pass
        raise RuntimeError("Failed to setup NCCL bridge")
    
    # Register parameters with the bridge (we already have the metadata)
    bridge.param_names = param_names
    bridge.param_shapes = {name: tuple(shape) for name, shape in param_shapes.items()}
    bridge.param_dtypes = param_dtypes
    
    # Export config for debugging/recovery
    bridge_config_path = os.path.join(config.save_path, "nccl_bridge_config.json")
    os.makedirs(config.save_path, exist_ok=True)
    export_bridge_config(
        bridge_config_path,
        param_metadata,
        param_mappings,
        config.nccl_init_method,
        config.nccl_world_size,
    )
    
    print(f"[5/5] Starting training for {config.training_steps} steps")
    print("-" * 60)
    
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
    
    # Send initial weights to vLLM
    print("Sending initial LoRA weights to vLLM...")
    initial_sync_time = bridge.send_lora_weights(model, step=0)
    print(f"  Initial sync completed in {initial_sync_time:.3f}s")
    
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
        
        # NCCL weight sync (every step for on-policy, or periodic)
        sync_time = 0
        should_sync = (
            config.nccl_sync_every_step or 
            (step + 1) % config.vllm_restart_interval == 0
        )
        if should_sync:
            sync_start = time.time()
            bridge.send_lora_weights(model, step=step + 1)
            sync_time = time.time() - sync_start
            benchmark_stats["sync_times"].append(sync_time)
            print(f"  [NCCL] Weights synced in {sync_time:.3f}s")
        
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
        
        # Periodic checkpoint (for recovery only, not for vLLM sync)
        if (
            config.checkpoint_interval > 0
            and (step + 1) % config.checkpoint_interval == 0
        ):
            save_lora_checkpoint(model, config.save_path, step + 1)
    
    # === Cleanup ===
    # Final sync
    print("\nSending final weights...")
    final_sync_time = bridge.send_lora_weights(model, step=config.training_steps)
    benchmark_stats["sync_times"].append(final_sync_time)
    
    # Save final checkpoint
    final_adapter_path = save_lora_checkpoint(
        model, config.save_path, config.training_steps, is_final=True
    )
    
    # Cleanup bridge
    bridge.cleanup()
    
    finalize_training(
        use_wandb,
        training_start_time,
        "lora_nccl",
        config.training_steps,
        benchmark_stats,
        config.benchmark,
    )
    
    # Save tokenizer
    tokenizer_path = os.path.join(config.save_path, "tokenizer")
    tokenizer.save_pretrained(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")
    print(f"Final adapter saved to {final_adapter_path}")
