import argparse
import atexit
import json
import math
import os
import random
import shutil
import string
import subprocess
import time
from typing import List, Literal, Optional, Tuple

import numpy as np
import requests
import torch
import torch.nn.functional as F
import wandb  # Added for logging
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import weight bridge for shared vLLM mode
try:
    from example_trainer.vllm_weight_bridge import (
        BridgeConfig,
        VLLMWeightBridge,
        create_bridge_from_training_config,
    )
    BRIDGE_AVAILABLE = True
except ImportError:
    BRIDGE_AVAILABLE = False

# Import PEFT for LoRA training
try:
    from peft import LoraConfig, TaskType, get_peft_model, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# Global variable to keep track of the vLLM process
vllm_process = None


def cleanup_vllm():
    global vllm_process
    if vllm_process:
        print("\nTerminating vLLM process...")
        vllm_process.terminate()
        try:
            vllm_process.wait(timeout=5)  # Wait a bit for graceful shutdown
            print("vLLM process terminated.")
        except subprocess.TimeoutExpired:
            print("vLLM process did not terminate gracefully, killing.")
            vllm_process.kill()
            vllm_process.wait()
            print("vLLM process killed.")
        vllm_process = None


# Register the cleanup function to be called on script exit
atexit.register(cleanup_vllm)


class TrainingConfig(BaseModel):
    """
    Training details, model, etc
    """

    model_name: str = Field(..., description="Name of the base model to train")
    lr: float = Field(1e-5, description="Learning rate for the optimizer")
    training_steps: int = Field(
        10, description="Number of training steps"
    )  # Renamed from epochs
    batch_size: int = Field(
        2, description="Batch size for training (will be handled by get_data)"
    )
    seq_len: int = Field(2048, description="Sequence length for training")
    gradient_accumulation_steps: int = Field(
        32, description="Number of gradient accumulation steps"
    )
    device: str = Field(
        "cuda" if torch.cuda.is_available() else "cpu", description="Device to train on"
    )
    save_path: str = Field(
        "trained_model_checkpoints", description="Base path to save model checkpoints"
    )
    vllm_restart_interval: int = Field(
        3, description="Restart vLLM every N training steps"
    )
    vllm_port: int = Field(9001, description="Port for the vLLM server")

    # Wandb configuration
    use_wandb: bool = Field(
        False, description="Whether to use Weights & Biases for logging"
    )
    wandb_project: Optional[str] = Field(None, description="Wandb project name")
    wandb_group: Optional[str] = Field(None, description="Wandb group name")

    # Pipeline / weight bridge configuration
    weight_bridge_mode: Literal["shared_vllm", "lora_only", "none"] = Field(
        "none",
        description=(
            "How to synchronize weights with inference server. "
            "'shared_vllm': attach to vLLM's shared memory tensors and update in-place. "
            "'lora_only': keep base model frozen, train/swap LoRA adapters. "
            "'none': legacy mode, restart vLLM with new checkpoint files."
        ),
    )
    trainer_rank: int = Field(
        0,
        description="Rank of this trainer in the distributed group (for shared_vllm mode)",
    )
    world_size: int = Field(
        1,
        description="Total processes in the distributed group (for shared_vllm mode)",
    )
    init_method: str = Field(
        "env://",
        description=(
            "PyTorch distributed init method URL. "
            "Use 'env://' to read MASTER_ADDR/MASTER_PORT from environment, "
            "or 'tcp://host:port' for explicit rendezvous."
        ),
    )
    num_inference_nodes: int = Field(
        0,
        description=(
            "Number of inference nodes (vLLM servers) to coordinate with. "
            "0 means single-node local mode."
        ),
    )

    # LoRA configuration (for lora_only mode)
    lora_r: int = Field(16, description="LoRA rank (dimension of low-rank matrices)")
    lora_alpha: int = Field(32, description="LoRA alpha (scaling factor)")
    lora_dropout: float = Field(0.05, description="Dropout probability for LoRA layers")
    lora_target_modules: Optional[List[str]] = Field(
        None,
        description=(
            "List of module names to apply LoRA to. "
            "If None, defaults to ['q_proj', 'v_proj'] for most models."
        ),
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
def register_trainer(config: TrainingConfig):
    """
    Register the trainer with the Atropos API
    """
    requests.post(
        "http://localhost:8000/register",
        json={
            "wandb_group": config.wandb_group,
            "wandb_project": config.wandb_project,
            "batch_size": config.batch_size * config.gradient_accumulation_steps,
            "max_token_len": config.seq_len,
            "starting_step": 0,
            "checkpoint_dir": config.save_path,
            "save_checkpoint_interval": config.training_steps,
            "num_steps": config.training_steps,
        },
        timeout=10,
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
def get_batch():
    data = requests.get("http://localhost:8000/batch", timeout=10).json()
    return data


def pad_data_to_good_offset(data, batch_size: int):
    max_token_len = max(
        [max([len(x) for x in item["tokens"]]) for item in data["batch"]]
    )
    # usually 64 is a good choice to ensure nonweird scaling behavior on GPUS
    # so we pad to the nearest multiple of 64
    good_multiple = 64
    if (max_token_len - 1) % (good_multiple) != 0:
        max_token_len = math.ceil((max_token_len - 1) / (good_multiple)) * good_multiple
        token_setup_len = (
            max_token_len + 1
        )  # add 1 so we can make it causal at the proper length
    else:
        token_setup_len = max_token_len
        max_token_len = (
            max_token_len - 1
        )  # since it's causal we need to remove the last bit...
    # pad all tokens to max_token_len and add to lists
    input_ids = list()
    labels = list()
    advantages = list()
    lengths = list()
    temperatures = list()
    for item in data["batch"]:
        scores = item["scores"]
        scores = np.array(scores)
        # check if we have more than 1 score...
        if len(scores) > 1:
            scores = scores - scores.mean()
            scores = scores / max(scores.std(), 1e-8)
        item["scores"] = scores
        if item["overrides"] is not None:
            for i in range(len(item["overrides"])):
                if item["overrides"][i].get("set_advantage_to_zero", False):
                    item["scores"][i] = 0
        for i in range(len(item["tokens"])):
            lengths.append(
                math.ceil((len(item["tokens"][i]) - 1) / (good_multiple))
                * good_multiple
            )
            label_item = np.concatenate(
                [
                    np.array(item["masks"][i]),
                    np.full(
                        max(0, token_setup_len - len(item["tokens"][i])),
                        -100,
                        dtype=np.int32,
                    ),
                ]
            )
            item["tokens"][i] = np.concatenate(
                [
                    np.array(item["tokens"][i]),
                    np.zeros(
                        max(0, token_setup_len - len(item["tokens"][i])), dtype=np.int32
                    ),
                ]
            )
            input_ids.append(item["tokens"][i][:-1])
            labels.append(label_item[1:])
            advantages.append(item["scores"][i])
            # per-sample override -> group generation_params -> group_overrides - > 1.0
            # need to update docs since this lets you set the temperature for each sample from the override
            t = 1.0
            if (
                item.get("overrides")
                and i < len(item["overrides"])
                and isinstance(item["overrides"][i], dict)
                and ("temperature" in item["overrides"][i])
            ):
                t = float(item["overrides"][i]["temperature"])
            elif item.get("generation_params") and (
                "temperature" in item["generation_params"]
            ):
                t = float(item["generation_params"]["temperature"])
            elif item.get("group_overrides") and (
                "temperature" in item["group_overrides"]
            ):
                t = float(item["group_overrides"]["temperature"])
            temperatures.append(t)
    # combine all lists into tensors
    token_batches = []
    label_batches = []
    advantage_batches = []
    temperature_batches = []
    for i in range(len(input_ids) // batch_size):
        token_batches.append(
            torch.tensor(
                np.stack(input_ids[i * batch_size : (i + 1) * batch_size], axis=0)
            )
        )
        label_batches.append(
            torch.tensor(
                np.stack(labels[i * batch_size : (i + 1) * batch_size], axis=0)
            )
        )
        advantage_batches.append(
            torch.tensor(
                np.stack(advantages[i * batch_size : (i + 1) * batch_size], axis=0)
            ).view(-1, 1)
        )
        # Temperatures: one per sample, shaped for broadcasting to [B, 1, 1]
        temperature_batches.append(
            torch.tensor(
                np.array(
                    temperatures[i * batch_size : (i + 1) * batch_size],
                    dtype=np.float32,
                )
            ).view(-1, 1, 1)
        )

    return token_batches, label_batches, advantage_batches, temperature_batches


def get_data(
    batch_size: int, seq_len: int
) -> List[
    Tuple[
        List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]
    ]
]:
    """
    getting data from the api
    """
    batches = []
    while True:
        data = get_batch()
        if data["batch"] is not None:
            # Save the batch
            with open("temp.json", "w", encoding="utf-8") as f:
                json.dump(data, f)
            # In case the inference runs ahead of the training, we loop until we don't have any more data
            batches.append(pad_data_to_good_offset(data, batch_size))
        elif len(batches) > 0:
            # Return the batches
            return batches
        else:
            time.sleep(1)


# =============================================================================
# Common Training Helpers (shared across all modes)
# =============================================================================


def setup_wandb(config: TrainingConfig) -> bool:
    """
    Initialize Weights & Biases logging if enabled.

    Args:
        config: Training configuration

    Returns:
        True if wandb is active, False otherwise
    """
    if not config.use_wandb:
        return False

    if not config.wandb_project:
        print("Warning: wandb_project not set, disabling wandb.")
        return False

    # Generate random group name if not provided
    if not config.wandb_group:
        config.wandb_group = "".join(
            random.choices(string.ascii_letters + string.digits, k=8)
        )

    try:
        wandb.init(
            project=config.wandb_project,
            group=config.wandb_group,
            config=config.dict(),
        )
        print(
            f"Wandb logging enabled. Run: {wandb.run.name} "
            f"(Project: {config.wandb_project})"
        )
        return True
    except Exception as e:
        print(f"Error initializing wandb: {e}. Disabling wandb.")
        return False


def load_model_and_tokenizer(
    config: TrainingConfig,
    bridge: Optional["VLLMWeightBridge"] = None,
) -> Tuple[torch.nn.Module, "AutoTokenizer"]:
    """
    Load or attach to model based on weight_bridge_mode.

    Args:
        config: Training configuration
        bridge: Optional weight bridge for shared_vllm mode

    Returns:
        Tuple of (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    if config.weight_bridge_mode == "shared_vllm" and bridge is not None:
        print("[Setup] Loading model for shared vLLM mode...")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name, torch_dtype=torch.bfloat16
        )
        model.to(config.device)
        bridge.attach_to_vllm_weights(dict(model.named_parameters()))

    elif config.weight_bridge_mode == "lora_only":
        model = _load_model_with_lora(config)

    else:
        print("[Setup] Loading model for legacy mode...")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name, torch_dtype=torch.bfloat16
        )
        model.to(config.device)

    # Enable gradient checkpointing (saves memory)
    # For LoRA, use PEFT's method; for others, use standard method
    if config.weight_bridge_mode == "lora_only":
        # PEFT models need gradient_checkpointing enabled on base model
        # and require use_reentrant=False for proper gradient flow
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_enable()
    
    model.train()

    return model, tokenizer


def _load_model_with_lora(config: TrainingConfig) -> torch.nn.Module:
    """
    Load base model and wrap with LoRA adapters.

    Args:
        config: Training configuration with LoRA settings

    Returns:
        PEFT model with LoRA adapters applied
    """
    if not PEFT_AVAILABLE:
        raise RuntimeError(
            "PEFT library not available. Install with: pip install peft"
    )

    print("[Setup] Loading base model for LoRA mode...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name, torch_dtype=torch.bfloat16
    )
    base_model.to(config.device)

    # Determine target modules
    target_modules = config.lora_target_modules
    if target_modules is None:
        # Default modules for most transformer models
        target_modules = ["q_proj", "v_proj"]

    print(f"[Setup] Applying LoRA: r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"[Setup] Target modules: {target_modules}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    return model


def save_lora_checkpoint(
    model: torch.nn.Module,
    save_path: str,
    step: int,
    is_final: bool = False,
) -> str:
    """
    Save LoRA adapter checkpoint.

    Args:
        model: PEFT model with LoRA adapters
        save_path: Base directory for checkpoints
        step: Current training step
        is_final: Whether this is the final checkpoint

    Returns:
        Path where adapter was saved
    """
    if is_final:
        adapter_path = os.path.join(save_path, "final_adapter")
    else:
        adapter_path = os.path.join(save_path, f"adapter_step_{step}")

    print(f"  Saving LoRA adapter to {adapter_path}...")

    if os.path.exists(adapter_path):
        shutil.rmtree(adapter_path)
    os.makedirs(adapter_path, exist_ok=True)

    # Save only the adapter weights (much smaller than full model)
    model.save_pretrained(adapter_path)

    print("  Adapter saved.")
    return adapter_path


def compute_grpo_loss(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    advantages: torch.Tensor,
    temperatures: torch.Tensor,
    gradient_accumulation_steps: int,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute GRPO loss for a single micro-batch.

    Args:
        model: The model to compute loss for
        tokens: Input token IDs [batch, seq_len]
        labels: Target labels [batch, seq_len]
        advantages: Advantage values [batch, 1]
        temperatures: Temperature values [batch, 1, 1]
        gradient_accumulation_steps: Number of accumulation steps

    Returns:
        Tuple of (loss tensor, metrics dict)
    """
    # Forward pass
    outputs = model(tokens)
    logits = outputs.logits

    # Temperature scaling
    t = temperatures.to(logits.device, logits.dtype)
    t = torch.where(t <= 0, torch.ones_like(t), t)
    logits = logits / t

    # Log probabilities per token
    logp_per_token = -F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(labels.shape)

    # Masking based on labels != -100
    mask = (labels != -100).float()

    # Compute metrics (no grad needed)
    with torch.no_grad():
        pos = (advantages > 0).float()
        neg = (advantages <= 0).float()
        mask_float = mask.to(logp_per_token.dtype)
        mask_sum = mask_float.sum(dim=-1).clamp_min(1e-8)
        avg_logp = (logp_per_token * mask_float).sum(dim=-1) / mask_sum
        pos_logp = (logp_per_token * pos).mean().item()
        neg_logp = (logp_per_token * neg).mean().item()

    # GRPO loss
    grpo_loss_term = torch.exp(logp_per_token - logp_per_token.detach())
    grpo_loss = (
        ((-grpo_loss_term * mask).sum(-1) / mask.sum(-1))
        * advantages.to(logp_per_token.device)
    ).mean() / gradient_accumulation_steps

    metrics = {
        "pos_logp": pos_logp,
        "neg_logp": neg_logp,
        "avg_logp": avg_logp,
        "pos_count": pos.sum().item(),
        "neg_count": neg.sum().item(),
    }

    return grpo_loss, metrics


def run_training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    token_batches: List[torch.Tensor],
    label_batches: List[torch.Tensor],
    advantage_batches: List[torch.Tensor],
    temperature_batches: List[torch.Tensor],
    config: TrainingConfig,
) -> dict:
    """
    Run a single training step (forward, backward, optimizer step).

    Args:
        model: The model to train
        optimizer: The optimizer
        token_batches: List of token tensors
        label_batches: List of label tensors
        advantage_batches: List of advantage tensors
        temperature_batches: List of temperature tensors
        config: Training configuration

    Returns:
        Dict of training metrics for this step
    """
    total_loss = 0.0
    total_pos_logp = 0.0
    total_neg_logp = 0.0
    total_pos = 0.0
    total_neg = 0.0

    # Accumulate gradients over micro-batches
    for tokens, labels, advantages, temperatures in zip(
        token_batches, label_batches, advantage_batches, temperature_batches
    ):
        tokens = tokens.to(config.device)
        labels = labels.to(config.device)
        advantages = advantages.to(config.device)

        loss, metrics = compute_grpo_loss(
            model, tokens, labels, advantages, temperatures,
            config.gradient_accumulation_steps
        )

        loss.backward()
        total_loss += loss.item()
        total_pos_logp += metrics["pos_logp"]
        total_neg_logp += metrics["neg_logp"]
        total_pos += metrics["pos_count"]
        total_neg += metrics["neg_count"]

    # Gradient clipping and optimizer step
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    # Normalize metrics
        if total_pos > 0:
            total_pos_logp /= total_pos
        if total_neg > 0:
            total_neg_logp /= total_neg

    return {
        "loss": total_loss,
        "grad_norm": grad_norm.item(),
        "pos_logp": total_pos_logp,
        "neg_logp": total_neg_logp,
    }


def save_checkpoint(
    model: torch.nn.Module,
    tokenizer: "AutoTokenizer",
    save_path: str,
    step: int,
    is_final: bool = False,
) -> str:
    """
    Save model checkpoint.

    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        save_path: Base directory for checkpoints
        step: Current training step
        is_final: Whether this is the final checkpoint

    Returns:
        Path where checkpoint was saved
    """
    if is_final:
        checkpoint_path = os.path.join(save_path, "final_model")
    else:
        checkpoint_path = os.path.join(save_path, f"step_{step}")

    print(f"  Saving checkpoint to {checkpoint_path}...")

    if os.path.exists(checkpoint_path):
        shutil.rmtree(checkpoint_path)
    os.makedirs(checkpoint_path, exist_ok=True)

    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)

    print("  Checkpoint saved.")
    return checkpoint_path


def log_metrics(
    metrics: dict,
    step: int,
    use_wandb: bool,
    extra_metrics: Optional[dict] = None,
) -> None:
    """
    Log training metrics to console and optionally wandb.

    Args:
        metrics: Dict of metrics from training step
        step: Current step number
        use_wandb: Whether to log to wandb
        extra_metrics: Optional additional metrics to log
    """
    print(f"  Loss: {metrics['loss']:.4f}, Grad norm: {metrics['grad_norm']:.4f}")

    if use_wandb:
        log_dict = {
            "train/loss": metrics["loss"],
            "train/grad_norm": metrics["grad_norm"],
            "train/pos_logp": metrics["pos_logp"],
            "train/neg_logp": metrics["neg_logp"],
        }
        if extra_metrics:
            log_dict.update(extra_metrics)
        wandb.log(log_dict, step=step)


def finalize_training(
    use_wandb: bool,
    training_start_time: Optional[float] = None,
    mode: str = "unknown",
    total_steps: int = 0,
) -> None:
    """Clean up after training and log benchmark summary."""
    print("\nTraining finished.")
    
    # Log benchmark summary
    if training_start_time is not None:
        total_time = time.time() - training_start_time
        gpu_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        
        print(f"\n{'='*60}")
        print(f"BENCHMARK SUMMARY ({mode})")
        print(f"{'='*60}")
        print(f"  Total training time: {total_time:.2f}s ({total_time/60:.2f} min)")
        print(f"  Total steps: {total_steps}")
        print(f"  Avg time per step: {total_time/max(total_steps,1):.2f}s")
        print(f"  Peak GPU memory: {gpu_mem_gb:.2f} GB")
        print(f"{'='*60}\n")
        
        if use_wandb:
            wandb.summary["benchmark/total_time_seconds"] = total_time
            wandb.summary["benchmark/total_time_minutes"] = total_time / 60
            wandb.summary["benchmark/avg_step_time_seconds"] = total_time / max(total_steps, 1)
            wandb.summary["benchmark/peak_gpu_memory_gb"] = gpu_mem_gb
            wandb.summary["benchmark/mode"] = mode
            wandb.summary["benchmark/total_steps"] = total_steps
    
    if use_wandb:
        wandb.finish()


def train(config: TrainingConfig):
    """
    Legacy GRPO training with periodic vLLM restarts.

    This mode saves checkpoints to disk and restarts vLLM to pick up new weights.
    Use weight_bridge_mode='shared_vllm' for in-place weight updates without restarts.
    """
    global vllm_process
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
    register_trainer(config)

    # Launch initial vLLM server
    vllm_process = _launch_vllm_server(config, config.model_name)

    # === Training Loop ===
    batches = []
    for step in range(config.training_steps):
        print(f"\nStep {step+1}/{config.training_steps}")

        # Get training data
        if len(batches) == 0:
            batches = get_data(config.batch_size, config.seq_len)
        token_batches, label_batches, advantage_batches, temperature_batches = batches.pop(0)

        # Terminate vLLM before training step (to free GPU memory)
        should_sync = (step + 1) % config.vllm_restart_interval == 0 or step == config.training_steps - 1
        if should_sync:
            _terminate_vllm_process()

        # Run training step using common helper
        metrics = run_training_step(
            model, optimizer,
            token_batches, label_batches, advantage_batches, temperature_batches,
            config
        )

        # Log metrics
        log_metrics(metrics, step + 1, use_wandb, {
            "train/learning_rate": optimizer.param_groups[0]["lr"],
        })

        # Save checkpoint and restart vLLM
        if should_sync:
            checkpoint_path = save_checkpoint(model, tokenizer, config.save_path, step + 1)
            torch.cuda.empty_cache()
            vllm_process = _launch_vllm_server(config, checkpoint_path)

        # Check for unexpected vLLM termination
        _check_vllm_health()

    # === Cleanup ===
    save_checkpoint(model, tokenizer, config.save_path, config.training_steps, is_final=True)
    finalize_training(use_wandb, training_start_time, "legacy", config.training_steps)


# =============================================================================
# vLLM Process Management (Legacy Mode Only)
# =============================================================================


def _launch_vllm_server(config: TrainingConfig, model_path: str) -> Optional[subprocess.Popen]:
    """Launch a vLLM server process."""
    global vllm_process

    vllm_command = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(config.vllm_port),
        "--dtype", "auto",
        "--gpu-memory-utilization", "0.45",
        "--disable-log-requests",
    ]
    # Add served-model-name if using checkpoint path
    if model_path != config.model_name:
        vllm_command.extend(["--served-model-name", config.model_name])

    print(f"  Launching vLLM: {' '.join(vllm_command)}")

    try:
        proc = subprocess.Popen(vllm_command)
        print(f"  vLLM launched with PID: {proc.pid}")

        # Check for immediate startup errors
        try:
            proc.communicate(timeout=2)
            if proc.returncode is not None and proc.returncode != 0:
                print("  WARNING: vLLM failed to start")
                return None
        except subprocess.TimeoutExpired:
            print("  vLLM process started (check logs for details)")

        return proc

    except FileNotFoundError:
        print("  ERROR: vLLM not found. Is it installed?")
        return None
    except Exception as e:
        print(f"  ERROR launching vLLM: {e}")
        return None


def _terminate_vllm_process() -> None:
    """Terminate the running vLLM process if any."""
    global vllm_process

    if vllm_process is None:
        return

    print("  Terminating vLLM process...")
    vllm_process.terminate()
    try:
        vllm_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        print("  vLLM did not terminate gracefully, killing...")
        vllm_process.kill()
        vllm_process.wait()
    vllm_process = None


def _check_vllm_health() -> None:
    """Check if vLLM process terminated unexpectedly."""
    global vllm_process

    if vllm_process is not None and vllm_process.poll() is not None:
        print(f"  WARNING: vLLM terminated unexpectedly (code: {vllm_process.returncode})")
        vllm_process = None


def train_shared_vllm(config: TrainingConfig):
    """
    GRPO training with shared vLLM weights.

    Instead of saving checkpoints and restarting vLLM, this mode:
    1. Joins the same distributed group as vLLM
    2. Attaches to vLLM's weight tensors directly
    3. optimizer.step() modifies vLLM's weights in-place
    4. vLLM immediately uses updated weights (no restart!)
    """
    if not BRIDGE_AVAILABLE:
        raise RuntimeError(
            "vLLM weight bridge not available. "
            "Ensure vllm_weight_bridge.py is in the same directory."
        )

    training_start_time = time.time()

    # === Setup ===
    use_wandb = setup_wandb(config)

    print(f"\n{'='*60}")
    print("SHARED VLLM MODE (in-place weight updates)")
    print(f"{'='*60}")
    print(f"Model: {config.model_name}")
    print(f"Distributed: rank={config.trainer_rank}/{config.world_size}")
    print(f"Init method: {config.init_method}")
    print(f"Inference nodes: {config.num_inference_nodes}")
    print(f"Save path: {config.save_path}")
    print(f"{'='*60}\n")

    # Initialize weight bridge
    print("[1/3] Initializing weight bridge...")
    bridge = create_bridge_from_training_config(config)

    # Load model with bridge attachment
    print("[2/3] Loading model with shared weights...")
    model, tokenizer = load_model_and_tokenizer(config, bridge=bridge)
    optimizer = AdamW(model.parameters(), lr=config.lr)

    print(f"[3/3] Starting training for {config.training_steps} steps")
    print("NOTE: vLLM sees weight updates immediately after each step!")
    print("-" * 60)

    os.makedirs(config.save_path, exist_ok=True)
    register_trainer(config)

    # === Training Loop ===
    batches = []
    for step in range(config.training_steps):
        print(f"\nStep {step+1}/{config.training_steps}")

        # Get training data
        if len(batches) == 0:
            batches = get_data(config.batch_size, config.seq_len)
        token_batches, label_batches, advantage_batches, temperature_batches = batches.pop(0)

        # Run training step using common helper
        metrics = run_training_step(
            model, optimizer,
            token_batches, label_batches, advantage_batches, temperature_batches,
            config
        )

        # Notify vLLM that weights have been updated
        bridge.notify_update()
        print(f"  [SHARED] Weights updated in-place - vLLM now using step {step+1} weights")

        # Log metrics
        log_metrics(metrics, step + 1, use_wandb, {
            "train/learning_rate": optimizer.param_groups[0]["lr"],
            "bridge/update_count": step + 1,
        })

        # Periodic checkpoint save (for recovery, not for vLLM sync)
        if (step + 1) % config.vllm_restart_interval == 0:
            save_checkpoint(model, tokenizer, config.save_path, step + 1)

    # === Cleanup ===
    bridge.cleanup()
    save_checkpoint(model, tokenizer, config.save_path, config.training_steps, is_final=True)
    finalize_training(use_wandb, training_start_time, "shared_vllm", config.training_steps)


def train_lora(config: TrainingConfig):
    """
    GRPO training with LoRA adapters.

    This mode keeps the base model frozen and only trains LoRA adapter weights.
    Benefits:
    - Much faster training (fewer parameters)
    - Smaller checkpoint sizes (adapter only, not full model)
    - Adapters can be hot-swapped in vLLM without full restart
    """
    if not PEFT_AVAILABLE:
        raise RuntimeError(
            "PEFT library required for LoRA mode. Install with: pip install peft"
        )

    training_start_time = time.time()

    # === Setup ===
    use_wandb = setup_wandb(config)

    print(f"\n{'='*60}")
    print("LORA MODE (adapter-only training)")
    print(f"{'='*60}")
    print(f"Base model: {config.model_name}")
    print(f"LoRA config: r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"Save path: {config.save_path}")
    print(f"{'='*60}\n")

    # Load model with LoRA adapters
    print("[1/2] Loading model with LoRA adapters...")
    model, tokenizer = load_model_and_tokenizer(config)

    # Only optimize LoRA parameters (base model is frozen)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=config.lr)

    print(f"[2/2] Starting training for {config.training_steps} steps")
    print("-" * 60)

    os.makedirs(config.save_path, exist_ok=True)
    register_trainer(config)

    # Launch vLLM with base model (adapters loaded separately)
    _launch_vllm_server(config, config.model_name)

    # === Training Loop ===
    batches = []
    for step in range(config.training_steps):
        print(f"\nStep {step+1}/{config.training_steps}")

        # Get training data
        if len(batches) == 0:
            batches = get_data(config.batch_size, config.seq_len)
        token_batches, label_batches, advantage_batches, temperature_batches = batches.pop(0)

        # Run training step
        metrics = run_training_step(
            model, optimizer,
            token_batches, label_batches, advantage_batches, temperature_batches,
            config
        )

        # Log metrics
        log_metrics(metrics, step + 1, use_wandb, {
            "train/learning_rate": optimizer.param_groups[0]["lr"],
            "lora/trainable_params": sum(p.numel() for p in trainable_params),
        })

        # Periodic adapter save
        if (step + 1) % config.vllm_restart_interval == 0:
            adapter_path = save_lora_checkpoint(model, config.save_path, step + 1)
            print(f"  [LORA] Adapter ready for hot-swap at: {adapter_path}")
            # Note: vLLM adapter hot-swap would be triggered here via API call

    # === Cleanup ===
    _terminate_vllm_process()

    # Save final adapter
    save_lora_checkpoint(model, config.save_path, config.training_steps, is_final=True)
    finalize_training(use_wandb, training_start_time, "lora_only", config.training_steps)

    # Also save tokenizer for convenience
    tokenizer_path = os.path.join(config.save_path, "tokenizer")
    tokenizer.save_pretrained(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the GRPO trainer."""
    parser = argparse.ArgumentParser(
        description="GRPO Trainer with optional shared-weight vLLM integration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Core training arguments ---
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="HuggingFace model identifier (e.g., 'Qwen/Qwen2.5-1.5B-Instruct')",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--training-steps",
        type=int,
        default=10,
        help="Number of training steps to run",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for training",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=32,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (cuda/cpu)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="trained_model_checkpoints",
        help="Directory to save model checkpoints",
    )

    # --- vLLM arguments ---
    parser.add_argument(
        "--vllm-restart-interval",
        type=int,
        default=3,
        help="Restart vLLM every N training steps (legacy mode only)",
    )
    parser.add_argument(
        "--vllm-port",
        type=int,
        default=9001,
        help="Port for the vLLM server",
    )

    # --- Wandb arguments ---
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help="Wandb group name",
    )

    # --- Pipeline / weight bridge arguments ---
    parser.add_argument(
        "--weight-bridge-mode",
        type=str,
        choices=["shared_vllm", "lora_only", "none"],
        default="none",
        help=(
            "Weight sync mode: "
            "'shared_vllm' = attach to vLLM shared memory, "
            "'lora_only' = train LoRA adapters only, "
            "'none' = legacy restart-based sync"
        ),
    )
    parser.add_argument(
        "--trainer-rank",
        type=int,
        default=0,
        help="Rank of this trainer in the distributed group",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Total processes in the distributed group",
    )
    parser.add_argument(
        "--init-method",
        type=str,
        default="env://",
        help="PyTorch distributed init method (e.g., 'env://', 'tcp://host:port')",
    )
    parser.add_argument(
        "--num-inference-nodes",
        type=int,
        default=0,
        help="Number of inference nodes to coordinate with (0 = single-node local)",
    )

    # --- LoRA arguments ---
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (dimension of low-rank matrices)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha (scaling factor, typically 2x rank)",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="Dropout probability for LoRA layers",
    )
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        nargs="+",
        default=None,
        help="Module names to apply LoRA to (default: q_proj v_proj)",
    )

    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> TrainingConfig:
    """Build a TrainingConfig from parsed CLI arguments."""
    return TrainingConfig(
        model_name=args.model_name,
        lr=args.lr,
        training_steps=args.training_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        device=args.device,
        save_path=args.save_path,
        vllm_restart_interval=args.vllm_restart_interval,
        vllm_port=args.vllm_port,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
        weight_bridge_mode=args.weight_bridge_mode,
        trainer_rank=args.trainer_rank,
        world_size=args.world_size,
        init_method=args.init_method,
        num_inference_nodes=args.num_inference_nodes,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
    )


# Example usage (optional, can be run from another script)
if __name__ == "__main__":
    args = parse_args()
    training_config = config_from_args(args)

    print(f"Weight bridge mode: {training_config.weight_bridge_mode}")

    if training_config.weight_bridge_mode == "shared_vllm":
        # Shared vLLM mode: attach to vLLM's weights, update in-place
        train_shared_vllm(training_config)

    elif training_config.weight_bridge_mode == "lora_only":
        # LoRA mode: freeze base model, train adapters only
        train_lora(training_config)

    else:
        # Legacy mode: periodic checkpoint saves + vLLM restarts
        train(training_config)
