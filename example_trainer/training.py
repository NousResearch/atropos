"""
Training utilities for GRPO trainer.

Contains loss computation, training step logic, and metric logging.

Includes logprob alignment tracking to verify that training logprobs match
inference logprobs at initialization (validates shared_vllm mode is working).
"""

import random
import string
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import wandb

from .config import TrainingConfig


# Global storage for logprob alignment stats
_logprob_alignment_stats: Dict[str, float] = {}


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


def compute_grpo_loss(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    advantages: torch.Tensor,
    temperatures: torch.Tensor,
    gradient_accumulation_steps: int,
    inference_logprobs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute GRPO (Group Relative Policy Optimization) loss for a single micro-batch.

    The GRPO loss encourages the model to:
    - Increase probability for tokens with positive advantages
    - Decrease probability for tokens with negative advantages
    
    Args:
        model: The model to compute loss for
        tokens: Input token IDs [batch, seq_len]
        labels: Target labels [batch, seq_len], -100 for masked positions
        advantages: Advantage values [batch, 1]
        temperatures: Temperature values [batch, 1, 1]
        gradient_accumulation_steps: Number of accumulation steps (for scaling)
        inference_logprobs: Optional logprobs from inference for alignment check

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
        
        # Collect training logprobs for masked positions (generated tokens only)
        # Keep as PyTorch tensor (supports bfloat16 natively)
        training_logprobs_flat = logp_per_token[mask.bool()].detach()

    # GRPO loss: weighted log probabilities by advantages
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
        "training_logprobs": training_logprobs_flat,  # For alignment check
    }

    return grpo_loss, metrics


def compute_logprob_alignment(
    inference_logprobs: List[np.ndarray],
    training_logprobs: List[torch.Tensor],
) -> Dict[str, float]:
    """
    Compute alignment stats between inference and training logprobs.
    
    At initialization (step 0), these should match closely if the model
    weights are correctly shared between training and inference.
    
    Args:
        inference_logprobs: Logprobs from vLLM inference (numpy arrays)
        training_logprobs: Logprobs computed during training forward pass (PyTorch tensors, bfloat16 supported)
        
    Returns:
        Dict of alignment statistics
    """
    if not inference_logprobs or not training_logprobs:
        return {}
    
    # Process inference logprobs (numpy)
    inf_flat = np.concatenate(inference_logprobs)
    # Filter out placeholder values (1.0 or 0.0 used for prompt tokens)
    inf_mask = (inf_flat != 1.0) & (inf_flat != 0.0)
    inf_filtered = inf_flat[inf_mask]
    
    # Process training logprobs (PyTorch - supports bfloat16 natively)
    train_flat = torch.cat(training_logprobs)
    
    # Compute stats using PyTorch for training (keeps bfloat16 precision)
    stats = {}
    
    if len(inf_filtered) > 0:
        stats["logprobs/inference_mean"] = float(np.mean(inf_filtered))
        stats["logprobs/inference_std"] = float(np.std(inf_filtered))
    
    if train_flat.numel() > 0:
        # PyTorch operations - fully support bfloat16
        stats["logprobs/training_mean"] = train_flat.mean().item()
        stats["logprobs/training_std"] = train_flat.std().item()
    
    # Compute diff (key metric for alignment validation)
    if "logprobs/inference_mean" in stats and "logprobs/training_mean" in stats:
        stats["logprobs/diff"] = stats["logprobs/inference_mean"] - stats["logprobs/training_mean"]
        
        # At step 0, this diff should be very close to 0 if weights are shared correctly
        # A large diff indicates the training model is using different weights than vLLM
    
    return stats


def run_training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    token_batches: List[torch.Tensor],
    label_batches: List[torch.Tensor],
    advantage_batches: List[torch.Tensor],
    temperature_batches: List[torch.Tensor],
    config: TrainingConfig,
    inference_logprobs: Optional[List[np.ndarray]] = None,
) -> dict:
    """
    Run a single training step with gradient accumulation.

    Performs:
    1. Forward pass through all micro-batches
    2. Backward pass with gradient accumulation
    3. Gradient clipping
    4. Optimizer step
    5. (Optional) Logprob alignment check
    
    Args:
        model: The model to train
        optimizer: The optimizer
        token_batches: List of token tensors (micro-batches)
        label_batches: List of label tensors
        advantage_batches: List of advantage tensors
        temperature_batches: List of temperature tensors
        config: Training configuration
        inference_logprobs: Optional logprobs from inference for alignment check

    Returns:
        Dict of training metrics for this step
    """
    global _logprob_alignment_stats
    
    total_loss = 0.0
    total_pos_logp = 0.0
    total_neg_logp = 0.0
    total_pos = 0.0
    total_neg = 0.0
    grad_norm = 0.0
    all_training_logprobs: List[torch.Tensor] = []

    # Accumulate gradients over micro-batches
    for tokens, labels, advantages, temperatures in zip(
        token_batches, label_batches, advantage_batches, temperature_batches
    ):
        tokens = tokens.to(config.device)
        labels = labels.to(config.device)
        advantages = advantages.to(config.device)

        loss, metrics = compute_grpo_loss(
            model,
            tokens,
            labels,
            advantages,
            temperatures,
            config.gradient_accumulation_steps,
        )

        loss.backward()
        total_loss += loss.item()
        total_pos_logp += metrics["pos_logp"]
        total_neg_logp += metrics["neg_logp"]
        total_pos += metrics["pos_count"]
        total_neg += metrics["neg_count"]
        
        # Collect training logprobs for alignment check
        if "training_logprobs" in metrics:
            all_training_logprobs.append(metrics["training_logprobs"])

    # Gradient clipping and optimizer step
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()

    # Normalize metrics by count
    num_batches = len(token_batches) if token_batches else 1
    if total_pos > 0:
        total_pos_logp /= num_batches
    if total_neg > 0:
        total_neg_logp /= num_batches

    result = {
        "loss": total_loss,
        "grad_norm": grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm,
        "pos_logp": total_pos_logp,
        "neg_logp": total_neg_logp,
        "pos_count": total_pos,
        "neg_count": total_neg,
    }
    
    # Compute logprob alignment stats
    if inference_logprobs is not None and all_training_logprobs:
        alignment_stats = compute_logprob_alignment(inference_logprobs, all_training_logprobs)
        _logprob_alignment_stats.update(alignment_stats)
        result["logprob_alignment"] = alignment_stats
    
    return result


def log_metrics(
    metrics: dict,
    step: int,
    use_wandb: bool,
    extra_metrics: Optional[dict] = None,
    benchmark: bool = False,
) -> None:
    """
    Log training metrics to console and optionally wandb.

    Args:
        metrics: Dict of metrics from training step
        step: Current step number
        use_wandb: Whether to log to wandb
        extra_metrics: Optional additional metrics to log
        benchmark: Whether to show timing/benchmark info
    """
    global _logprob_alignment_stats
    
    # Build timing string (only if benchmark enabled)
    timing_str = ""
    if benchmark:
        if "step_time" in metrics:
            timing_str += f", Step time: {metrics['step_time']:.2f}s"
        if "sync_time" in metrics and metrics["sync_time"] > 0:
            timing_str += f", Sync time: {metrics['sync_time']:.2f}s"
        if "data_fetch_time" in metrics:
            timing_str += f", Data fetch: {metrics['data_fetch_time']:.2f}s"
        if "gpu_memory_gb" in metrics:
            timing_str += f", GPU mem: {metrics['gpu_memory_gb']:.2f}GB"

    # Show loss with more precision since GRPO loss is often very small
    loss_str = (
        f"{metrics['loss']:.6f}"
        if abs(metrics["loss"]) < 0.01
        else f"{metrics['loss']:.4f}"
    )
    print(f"  Loss: {loss_str}, Grad norm: {metrics['grad_norm']:.4f}{timing_str}")

    # Show GRPO-specific metrics if available
    if "pos_count" in metrics or "neg_count" in metrics:
        pos_count = metrics.get("pos_count", 0)
        neg_count = metrics.get("neg_count", 0)
        pos_logp = metrics.get("pos_logp", 0)
        neg_logp = metrics.get("neg_logp", 0)
        print(
            f"    Advantages: +{int(pos_count)} / -{int(neg_count)}, "
            f"LogP: pos={pos_logp:.3f}, neg={neg_logp:.3f}"
        )
    
    # Show logprob alignment stats (important for shared_vllm validation!)
    if "logprob_alignment" in metrics:
        alignment = metrics["logprob_alignment"]
        if "logprobs/diff" in alignment:
            diff = alignment["logprobs/diff"]
            inf_mean = alignment.get("logprobs/inference_mean", 0)
            train_mean = alignment.get("logprobs/training_mean", 0)
            
            # At step 0, diff should be ~0 if weights are shared correctly
            status = "OK" if abs(diff) < 0.1 else "MISMATCH!"
            print(f"    LogProb Alignment: inf={inf_mean:.4f}, train={train_mean:.4f}, "
                  f"diff={diff:.4f} [{status}]")

    if use_wandb:
        log_dict = {
            "train/loss": metrics["loss"],
            "train/grad_norm": metrics["grad_norm"],
            "train/pos_logp": metrics.get("pos_logp", 0),
            "train/neg_logp": metrics.get("neg_logp", 0),
        }
        # Add timing metrics if present
        for key in ["step_time", "sync_time", "data_fetch_time", 
                    "gpu_memory_gb", "gpu_memory_reserved_gb"]:
            if key in metrics:
                log_dict[f"train/{key}"] = metrics[key]
        
        # Add logprob alignment stats (key for shared_vllm validation!)
        if _logprob_alignment_stats:
            log_dict.update(_logprob_alignment_stats)
        
        if extra_metrics:
            log_dict.update(extra_metrics)
        wandb.log(log_dict, step=step)


def finalize_training(
    use_wandb: bool,
    training_start_time: Optional[float] = None,
    mode: str = "unknown",
    total_steps: int = 0,
    benchmark_stats: Optional[dict] = None,
    benchmark: bool = False,
) -> None:
    """
    Clean up after training and log benchmark summary.

    Args:
        use_wandb: Whether wandb is enabled
        training_start_time: Start time of training
        mode: Training mode name
        total_steps: Total steps completed
        benchmark_stats: Dict with lists of per-step metrics
        benchmark: Whether to print benchmark summary to console
    """
    print("\nTraining finished.")

    if benchmark_stats is None:
        benchmark_stats = {}

    if training_start_time is not None:
        total_time = time.time() - training_start_time
        peak_gpu_mem_gb = (
            torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        )

        # Calculate averages from collected stats
        step_times = benchmark_stats.get("step_times", [])
        sync_times = benchmark_stats.get("sync_times", [])
        data_fetch_times = benchmark_stats.get("data_fetch_times", [])
        gpu_memories = benchmark_stats.get("gpu_memories", [])

        avg_step_time = sum(step_times) / len(step_times) if step_times else 0
        total_step_time = sum(step_times)
        avg_sync_time = sum(sync_times) / len(sync_times) if sync_times else 0
        total_sync_time = sum(sync_times)
        avg_data_fetch = sum(data_fetch_times) / len(data_fetch_times) if data_fetch_times else 0
        total_data_fetch = sum(data_fetch_times)
        avg_gpu_mem = sum(gpu_memories) / len(gpu_memories) if gpu_memories else 0

        if benchmark:
            print(f"\n{'='*70}")
            print(f"BENCHMARK SUMMARY ({mode})")
            print(f"{'='*70}")
            print(f"  Total training time:     {total_time:.2f}s ({total_time/60:.2f} min)")
            print(f"  Total steps:             {total_steps}")
            print("  ")
            print("  TIMING BREAKDOWN:")
            print(f"    Avg step time:         {avg_step_time:.2f}s")
            print(f"    Total step time:       {total_step_time:.2f}s")
            print(f"    Avg sync time:         {avg_sync_time:.2f}s (x{len(sync_times)} syncs)")
            print(f"    Total sync time:       {total_sync_time:.2f}s")
            print(f"    Avg data fetch time:   {avg_data_fetch:.2f}s")
            print(f"    Total data fetch time: {total_data_fetch:.2f}s")
            print("  ")
            print("  MEMORY:")
            print(f"    Peak GPU memory:       {peak_gpu_mem_gb:.2f} GB")
            print(f"    Avg GPU memory:        {avg_gpu_mem:.2f} GB")
            print(f"{'='*70}\n")

        if use_wandb:
            wandb.summary["benchmark/total_time_seconds"] = total_time
            wandb.summary["benchmark/total_time_minutes"] = total_time / 60
            wandb.summary["benchmark/mode"] = mode
            wandb.summary["benchmark/total_steps"] = total_steps
            wandb.summary["benchmark/avg_step_time_seconds"] = avg_step_time
            wandb.summary["benchmark/peak_gpu_memory_gb"] = peak_gpu_mem_gb
            wandb.summary["benchmark/avg_gpu_memory_gb"] = avg_gpu_mem
            wandb.finish()
    elif use_wandb:
        wandb.finish()

