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
    clip_eps: float = 0.2,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute GRPO (Group Relative Policy Optimization) loss for a single micro-batch.

    This implements proper GRPO/PPO with:
    - Importance sampling ratio: policy(a|s) / policy_old(a|s)
    - PPO-style clipping to prevent large updates

    The loss encourages the model to:
    - Increase probability for tokens with positive advantages
    - Decrease probability for tokens with negative advantages

    Args:
        model: The model to compute loss for
        tokens: Input token IDs [batch, seq_len]
        labels: Target labels [batch, seq_len], -100 for masked positions
        advantages: Advantage values [batch, 1]
        temperatures: Temperature values [batch, 1, 1]
        gradient_accumulation_steps: Number of accumulation steps (for scaling)
        inference_logprobs: Logprobs from inference (π_old), aligned with labels [batch, seq_len]
        clip_eps: PPO clipping epsilon. Clips ratio to [1-eps, 1+eps]

    Returns:
        Tuple of (loss tensor, metrics dict)
    """
    # Forward pass
    outputs = model(tokens)
    logits = outputs.logits

    # Temperature scaling for training otherwise likely ratio is off
    t = temperatures.to(logits.device, logits.dtype)
    t = torch.where(t <= 0, torch.ones_like(t), t)
    scaled_logits = logits / t

    # Log probabilities per token (current policy π)
    logp_per_token = -F.cross_entropy(
        scaled_logits.view(-1, scaled_logits.size(-1)),
        labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(labels.shape)

    # Masking based on labels != -100
    mask = (labels != -100).float()
    mask_sum = mask.sum(dim=-1).clamp_min(1e-8)

    # Expand advantages to match token shape [batch, 1] -> [batch, seq_len]
    adv_expanded = advantages.expand_as(logp_per_token).to(logp_per_token.device)

    # Track logprobs for alignment verification
    inference_logprobs_flat = None
    logprob_diff_mean = 0.0
    logprob_diff_abs_mean = 0.0
    logprob_diff_max = 0.0

    # === GRPO/PPO Loss Computation ===
    if inference_logprobs is not None:
        # Move inference logprobs to correct device/dtype
        ref_logprobs = inference_logprobs.to(
            logp_per_token.device, logp_per_token.dtype
        )

        # NOTE: inference_logprobs uses 1.0 for masked (prompt) positions, actual negative values for generated
        with torch.no_grad():
            # Only look at generated positions (where mask == 1)
            ref_at_generated = (ref_logprobs * mask).sum() / mask.sum()
            train_at_generated = (logp_per_token * mask).sum() / mask.sum()

            # Extract logprobs at generated positions for alignment tracking
            inference_logprobs_flat = ref_logprobs[mask.bool()].detach()
            training_at_mask = logp_per_token[mask.bool()].detach()

            # Token-level difference: THE key metric for alignment verification
            # If weights are truly shared, this should be ~0 at step start
            token_diff = training_at_mask - inference_logprobs_flat
            logprob_diff_mean = token_diff.mean().item()
            logprob_diff_abs_mean = token_diff.abs().mean().item()
            logprob_diff_max = token_diff.abs().max().item()

            # Check if ref logprobs are negative (as they should be for generated tokens)
            # If ref_at_generated is close to 1.0, that means the 1.0 placeholder is being used
            if ref_at_generated > 0.5:
                print(
                    f"    [WARNING] ref_logprobs avg {ref_at_generated:.3f} (should be negative!)"
                )
                print(
                    "    [WARNING] This suggests inference_logprobs alignment is wrong"
                )
            elif abs(ref_at_generated - train_at_generated) > 2.0:
                print(
                    f"    [DEBUG] Logprob gap: ref={ref_at_generated:.3f}, train={train_at_generated:.3f}"
                )

        # Compute importance sampling ratio: policy(a|s) / policy_old(a|s) = exp(log policy - log policy_old)
        log_ratio = logp_per_token - ref_logprobs
        ratio = torch.exp(log_ratio)

        # PPO-style clipping
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

        # Surrogate objectives
        surr1 = ratio * adv_expanded
        surr2 = clipped_ratio * adv_expanded

        # Pessimistic bound: min for positive advantages, max for negative
        # This is equivalent to: -min(ratio * A, clipped_ratio * A) when A > 0
        # -max(ratio * A, clipped_ratio * A) when A < 0
        policy_loss_per_token = -torch.where(
            adv_expanded >= 0,
            torch.min(surr1, surr2),
            torch.max(surr1, surr2),
        )

        # Average over tokens, then over batch
        policy_loss = ((policy_loss_per_token * mask).sum(dim=-1) / mask_sum).mean()

        total_loss = policy_loss / gradient_accumulation_steps

        # Compute metrics for logging
        with torch.no_grad():
            # Fraction of tokens where ratio was clipped
            clipped_fraction = (
                (ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)
            ).float()
            clipped_fraction = (clipped_fraction * mask).sum() / mask.sum()

            # Mean ratio for monitoring
            mean_ratio = (ratio * mask).sum() / mask.sum()

            # For backward compatibility: collect training logprobs
            raw_logp_per_token = -F.cross_entropy(
                outputs.logits.view(-1, outputs.logits.size(-1)),
                labels.view(-1),
                reduction="none",
                ignore_index=-100,
            ).view(labels.shape)
            training_logprobs_flat = raw_logp_per_token[mask.bool()].detach()
    else:
        # Fail loudly
        raise ValueError(
            "GRPO requires inference_logprobs for importance sampling!\n"
            "\n"
            "This error means the environment isn't providing logprobs. To fix:\n"
            "  1. Use --openai.server_type vllm (not 'openai')\n"
            "  2. Ensure vLLM is returning logprobs in /generate response\n"
            "  3. Check that gsm8k_server is configured correctly\n"
            "\n"
            "Without inference logprobs, training will cause reward hacking."
        )

    # === Compute Additional Metrics ===
    with torch.no_grad():
        pos = (advantages > 0).float()
        neg = (advantages <= 0).float()
        mask_float = mask.to(logp_per_token.dtype)
        avg_logp = (logp_per_token * mask_float).sum(dim=-1) / mask_sum
        pos_logp = (logp_per_token * pos).mean().item()
        neg_logp = (logp_per_token * neg).mean().item()

        # Interpretable metric: advantage-weighted average logprob
        interpretable_loss = (avg_logp * advantages.squeeze()).mean().item()

    metrics = {
        "pos_logp": pos_logp,
        "neg_logp": neg_logp,
        "avg_logp": avg_logp,
        "pos_count": pos.sum().item(),
        "neg_count": neg.sum().item(),
        "training_logprobs": training_logprobs_flat,
        "inference_logprobs": inference_logprobs_flat,
        "interpretable_loss": interpretable_loss,
        # GRPO-specific metrics
        "mean_ratio": mean_ratio.item() if torch.is_tensor(mean_ratio) else mean_ratio,
        "clipped_fraction": (
            clipped_fraction.item()
            if torch.is_tensor(clipped_fraction)
            else clipped_fraction
        ),
        # Token-level alignment metrics (key for verifying weight sharing)
        "logprob_diff_mean": logprob_diff_mean,
        "logprob_diff_abs_mean": logprob_diff_abs_mean,
        "logprob_diff_max": logprob_diff_max,
    }

    return total_loss, metrics


def run_training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    token_batches: List[torch.Tensor],
    label_batches: List[torch.Tensor],
    advantage_batches: List[torch.Tensor],
    temperature_batches: List[torch.Tensor],
    config: TrainingConfig,
    step_idx: int,
    inference_logprob_batches: Optional[List[torch.Tensor]] = None,
) -> dict:
    """
    Run a single training step with gradient accumulation.

    Performs:
    1. Forward pass through all micro-batches with proper GRPO loss
    2. Backward pass with gradient accumulation
    3. Gradient clipping
    4. Optimizer step

    Args:
        model: The model to train
        optimizer: The optimizer
        token_batches: List of token tensors (micro-batches)
        label_batches: List of label tensors
        advantage_batches: List of advantage tensors
        temperature_batches: List of temperature tensors
        config: Training configuration (includes clip_eps, warmup_steps)
        step_idx: Current global training step (0-based)
        inference_logprob_batches: Batched logprobs from inference (π_old), aligned with labels

    Returns:
        Dict of training metrics for this step
    """
    total_loss = 0.0
    total_pos_logp = 0.0
    total_neg_logp = 0.0
    total_pos = 0.0
    total_neg = 0.0
    total_mean_ratio = 0.0
    total_clipped_fraction = 0.0
    total_logprob_diff_mean = 0.0
    total_logprob_diff_abs_mean = 0.0
    total_logprob_diff_max = 0.0
    grad_norm = 0.0
    all_training_logprobs: List[torch.Tensor] = []
    all_inference_logprobs: List[torch.Tensor] = []

    # Get GRPO hyperparameters from config
    clip_eps = getattr(config, "clip_eps", 0.2)

    # Apply linear warmup to optimizer LR for early-step stability.
    warmup_steps = max(0, int(getattr(config, "warmup_steps", 0)))
    if warmup_steps > 0 and step_idx < warmup_steps:
        warmup_scale = float(step_idx + 1) / float(max(1, warmup_steps))
        current_lr = float(config.lr) * warmup_scale
    else:
        current_lr = float(config.lr)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    # Accumulate gradients over micro-batches
    num_batches = len(token_batches) if token_batches else 1

    for batch_idx, (tokens, labels, advantages, temperatures) in enumerate(
        zip(token_batches, label_batches, advantage_batches, temperature_batches)
    ):
        tokens = tokens.to(config.device)
        labels = labels.to(config.device)
        advantages = advantages.to(config.device)

        # Get corresponding inference logprobs batch if available
        inf_logprobs = None
        if inference_logprob_batches is not None and batch_idx < len(
            inference_logprob_batches
        ):
            inf_logprobs = inference_logprob_batches[batch_idx]

        loss, metrics = compute_grpo_loss(
            model,
            tokens,
            labels,
            advantages,
            temperatures,
            config.gradient_accumulation_steps,
            inference_logprobs=inf_logprobs,
            clip_eps=clip_eps,
        )

        loss.backward()
        total_loss += loss.item()
        total_pos_logp += metrics["pos_logp"]
        total_neg_logp += metrics["neg_logp"]
        total_pos += metrics["pos_count"]
        total_neg += metrics["neg_count"]

        # Accumulate GRPO-specific metrics
        total_mean_ratio += metrics.get("mean_ratio", 1.0)
        total_clipped_fraction += metrics.get("clipped_fraction", 0.0)

        # Accumulate token-level alignment metrics
        total_logprob_diff_mean += metrics.get("logprob_diff_mean", 0.0)
        total_logprob_diff_abs_mean += metrics.get("logprob_diff_abs_mean", 0.0)
        total_logprob_diff_max = max(
            total_logprob_diff_max, metrics.get("logprob_diff_max", 0.0)
        )

        # Collect logprobs for alignment monitoring
        if "training_logprobs" in metrics and metrics["training_logprobs"] is not None:
            all_training_logprobs.append(metrics["training_logprobs"])
        if (
            "inference_logprobs" in metrics
            and metrics["inference_logprobs"] is not None
        ):
            all_inference_logprobs.append(metrics["inference_logprobs"])

    # Gradient clipping and optimizer step
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()

    # Help prevent memory fragmentation
    torch.cuda.empty_cache()

    # Normalize metrics by batch count
    if total_pos > 0:
        total_pos_logp /= num_batches
    if total_neg > 0:
        total_neg_logp /= num_batches

    result = {
        "loss": total_loss,
        "lr": current_lr,
        "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else grad_norm,
        "pos_logp": total_pos_logp,
        "neg_logp": total_neg_logp,
        "pos_count": total_pos,
        "neg_count": total_neg,
        # GRPO-specific metrics (averaged over batches)
        "mean_ratio": total_mean_ratio / num_batches,
        "clipped_fraction": total_clipped_fraction / num_batches,
    }

    # Compute logprob alignment stats for monitoring
    # This proves weight sharing is working: inference & training logprobs should converge
    if all_training_logprobs:
        train_flat = torch.cat(all_training_logprobs)
        if train_flat.numel() > 0:
            _logprob_alignment_stats["logprobs/training_mean"] = (
                train_flat.mean().item()
            )
            _logprob_alignment_stats["logprobs/training_std"] = train_flat.std().item()

    if all_inference_logprobs:
        inf_flat = torch.cat(all_inference_logprobs)
        if inf_flat.numel() > 0:
            _logprob_alignment_stats["logprobs/inference_mean"] = inf_flat.mean().item()
            _logprob_alignment_stats["logprobs/inference_std"] = inf_flat.std().item()

    # Token-level alignment metrics - THE key metric for verifying weight sharing
    # diff_abs_mean close to 0 = weights are truly shared
    _logprob_alignment_stats["alignment/diff_mean"] = (
        total_logprob_diff_mean / num_batches
    )
    _logprob_alignment_stats["alignment/diff_abs_mean"] = (
        total_logprob_diff_abs_mean / num_batches
    )
    _logprob_alignment_stats["alignment/diff_max"] = total_logprob_diff_max

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

    # Primary metrics line: Loss and grad norm
    loss_str = (
        f"{metrics['loss']:.6f}"
        if abs(metrics["loss"]) < 0.01
        else f"{metrics['loss']:.4f}"
    )
    print(f"  Loss: {loss_str}, Grad norm: {metrics['grad_norm']:.4f}{timing_str}")

    # GRPO metrics line: ratio and clipping
    mean_ratio = metrics.get("mean_ratio", 1.0)
    clipped_frac = metrics.get("clipped_fraction", 0)

    print(f"    GRPO: ratio={mean_ratio:.3f}, clipped={clipped_frac*100:.1f}%")

    # Advantage distribution
    if "pos_count" in metrics or "neg_count" in metrics:
        pos_count = metrics.get("pos_count", 0)
        neg_count = metrics.get("neg_count", 0)
        pos_logp = metrics.get("pos_logp", 0)
        neg_logp = metrics.get("neg_logp", 0)
        print(
            f"    Advantages: +{int(pos_count)} / -{int(neg_count)}, "
            f"LogP: pos={pos_logp:.3f}, neg={neg_logp:.3f}"
        )

    if use_wandb:
        log_dict = {
            "train/loss": metrics["loss"],
            "train/grad_norm": metrics["grad_norm"],
            "train/lr": metrics.get("lr", 0.0),
            "train/pos_logp": metrics.get("pos_logp", 0),
            "train/neg_logp": metrics.get("neg_logp", 0),
            # GRPO-specific metrics
            "grpo/mean_ratio": mean_ratio,
            "grpo/clipped_fraction": clipped_frac,
        }
        # Add timing metrics if present
        for key in [
            "step_time",
            "sync_time",
            "data_fetch_time",
            "gpu_memory_gb",
            "gpu_memory_reserved_gb",
        ]:
            if key in metrics:
                log_dict[f"train/{key}"] = metrics[key]

        # Add logprob alignment stats
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
        avg_data_fetch = (
            sum(data_fetch_times) / len(data_fetch_times) if data_fetch_times else 0
        )
        total_data_fetch = sum(data_fetch_times)
        avg_gpu_mem = sum(gpu_memories) / len(gpu_memories) if gpu_memories else 0

        if benchmark:
            print(f"\n{'='*70}")
            print(f"BENCHMARK SUMMARY ({mode})")
            print(f"{'='*70}")
            print(
                f"  Total training time:     {total_time:.2f}s ({total_time/60:.2f} min)"
            )
            print(f"  Total steps:             {total_steps}")
            print("  ")
            print("  TIMING BREAKDOWN:")
            print(f"    Avg step time:         {avg_step_time:.2f}s")
            print(f"    Total step time:       {total_step_time:.2f}s")
            print(
                f"    Avg sync time:         {avg_sync_time:.2f}s (x{len(sync_times)} syncs)"
            )
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
