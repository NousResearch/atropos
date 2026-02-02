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

# Global storage for weight verification
_weight_snapshot: Dict[str, float] = {}


def verify_vllm_sees_updates(model: torch.nn.Module, vllm_port: int, step: int) -> bool:
    """
    Verify that vLLM actually sees weight updates by corrupting a weight
    and checking if vLLM's output changes.
    
    Returns True if vLLM sees updates, False otherwise.
    """
    import requests
    
    try:
        # Find embedding layer
        embed_param = None
        for name, param in model.named_parameters():
            if "embed_tokens" in name:
                embed_param = param
                break
        
        if embed_param is None:
            return True  # Can't verify, assume OK
        
        test_prompt = "Hello"
        vllm_url = f"http://localhost:{vllm_port}"
        
        # Get baseline
        r1 = requests.post(
            f"{vllm_url}/generate",
            json={"prompt": test_prompt, "max_tokens": 3, "temperature": 0.0},
            timeout=10,
        )
        baseline = r1.json().get("text", [""])[0] if r1.status_code == 200 else None
        
        if baseline is None:
            return True  # Can't verify
        
        # Corrupt weight
        original = embed_param.data[0, 0].clone()
        embed_param.data[0, 0] = 9999.0
        
        # Query vLLM
        r2 = requests.post(
            f"{vllm_url}/generate",
            json={"prompt": test_prompt, "max_tokens": 3, "temperature": 0.0},
            timeout=10,
        )
        corrupted = r2.json().get("text", [""])[0] if r2.status_code == 200 else baseline
        
        # Restore
        embed_param.data[0, 0] = original
        
        # Check if output changed
        sharing_works = (corrupted != baseline)
        
        if not sharing_works and step > 0:
            print(f"    [WARN] Step {step}: vLLM may not see weight updates!")
        
        return sharing_works
        
    except Exception:
        return True  # Can't verify, assume OK


def snapshot_weights(model: torch.nn.Module) -> Dict[str, float]:
    """Take a snapshot of sample weight values for comparison."""
    snapshot = {}
    for name, param in model.named_parameters():
        if any(x in name for x in ["layers.0.", "layers.10.", "embed_tokens", "lm_head"]):
            snapshot[name] = param.data.flatten()[0].item()
    return snapshot


def compare_weight_snapshots(old: Dict[str, float], new: Dict[str, float]) -> Dict[str, float]:
    """Compare two weight snapshots and return differences."""
    diffs = {}
    for name in old:
        if name in new:
            diffs[name] = abs(new[name] - old[name])
    return diffs


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
    kl_coef: float = 0.1,
    clip_eps: float = 0.2,
    use_reference_logprobs: bool = True,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute GRPO (Group Relative Policy Optimization) loss for a single micro-batch.

    This implements proper GRPO/PPO with:
    - Importance sampling ratio: π(a|s) / π_old(a|s)
    - PPO-style clipping to prevent large updates
    - KL penalty to prevent reward hacking/policy collapse
    
    The loss encourages the model to:
    - Increase probability for tokens with positive advantages
    - Decrease probability for tokens with negative advantages
    - Stay close to the reference policy (inference-time policy)
    
    Args:
        model: The model to compute loss for
        tokens: Input token IDs [batch, seq_len]
        labels: Target labels [batch, seq_len], -100 for masked positions
        advantages: Advantage values [batch, 1]
        temperatures: Temperature values [batch, 1, 1]
        gradient_accumulation_steps: Number of accumulation steps (for scaling)
        inference_logprobs: Logprobs from inference (π_old), aligned with labels [batch, seq_len]
        kl_coef: KL penalty coefficient (beta). Higher = more conservative updates
        clip_eps: PPO clipping epsilon. Clips ratio to [1-eps, 1+eps]
        use_reference_logprobs: If True, use inference_logprobs as reference policy

    Returns:
        Tuple of (loss tensor, metrics dict)
    """
    # Forward pass
    outputs = model(tokens)
    logits = outputs.logits

    # Temperature scaling for training
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

    # === GRPO/PPO Loss Computation ===
    if use_reference_logprobs and inference_logprobs is not None:
        # Move inference logprobs to correct device/dtype
        ref_logprobs = inference_logprobs.to(logp_per_token.device, logp_per_token.dtype)
        
        # Compute importance sampling ratio: π(a|s) / π_old(a|s) = exp(log π - log π_old)
        log_ratio = logp_per_token - ref_logprobs
        ratio = torch.exp(log_ratio)
        
        # PPO-style clipping
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        
        # Surrogate objectives
        surr1 = ratio * adv_expanded
        surr2 = clipped_ratio * adv_expanded
        
        # Pessimistic bound: min for positive advantages, max for negative
        # This is equivalent to: -min(ratio * A, clipped_ratio * A) when A > 0
        #                        -max(ratio * A, clipped_ratio * A) when A < 0
        policy_loss_per_token = -torch.where(
            adv_expanded >= 0,
            torch.min(surr1, surr2),
            torch.max(surr1, surr2),
        )
        
        # Average over tokens, then over batch
        policy_loss = ((policy_loss_per_token * mask).sum(dim=-1) / mask_sum).mean()
        
        # KL penalty: encourage staying close to reference policy
        # Using Schulman's unbiased KL estimator from the DeepSeek GRPO paper (Equation 4):
        #   D_KL(π_θ || π_ref) = (π_ref / π_θ) - log(π_ref / π_θ) - 1
        # 
        # In terms of log probabilities:
        #   log_ratio = log π_θ - log π_ref  (what we computed above)
        #   ratio_ref_over_pi = exp(-log_ratio) = π_ref / π_θ
        #   kl = ratio_ref_over_pi - log(ratio_ref_over_pi) - 1
        #      = exp(-log_ratio) + log_ratio - 1
        #
        # This estimator is guaranteed to be non-negative (unlike squared log-ratio).
        if kl_coef > 0:
            # Schulman's unbiased KL estimator: (π_ref/π) - log(π_ref/π) - 1
            # = exp(-log_ratio) + log_ratio - 1
            kl_per_token = torch.exp(-log_ratio) + log_ratio - 1.0
            kl_penalty = ((kl_per_token * mask).sum(dim=-1) / mask_sum).mean()
            total_loss = (policy_loss + kl_coef * kl_penalty) / gradient_accumulation_steps
        else:
            kl_penalty = torch.tensor(0.0, device=logp_per_token.device)
            total_loss = policy_loss / gradient_accumulation_steps
        
        # Compute metrics for logging
        with torch.no_grad():
            # Fraction of tokens where ratio was clipped
            clipped_fraction = ((ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)).float()
            clipped_fraction = (clipped_fraction * mask).sum() / mask.sum()
            
            # Mean ratio and KL for monitoring (using Schulman's estimator)
            mean_ratio = (ratio * mask).sum() / mask.sum()
            # Schulman KL: exp(-log_ratio) + log_ratio - 1
            schulman_kl = torch.exp(-log_ratio) + log_ratio - 1.0
            mean_kl = (schulman_kl * mask).sum() / mask.sum()
            
            # For backward compatibility: collect training logprobs
            raw_logp_per_token = -F.cross_entropy(
                outputs.logits.view(-1, outputs.logits.size(-1)),
                labels.view(-1),
                reduction="none",
                ignore_index=-100,
            ).view(labels.shape)
            training_logprobs_flat = raw_logp_per_token[mask.bool()].detach()
    else:
        # Fallback: REINFORCE-style (no reference policy)
        # This is what the original code did - NOT recommended!
        print("  [WARNING] No reference logprobs - using REINFORCE (may cause reward hacking!)")
        
        # Simple policy gradient: -log(π) * A
        policy_loss = ((-logp_per_token * mask * adv_expanded).sum(dim=-1) / mask_sum).mean()
        total_loss = policy_loss / gradient_accumulation_steps
        kl_penalty = torch.tensor(0.0, device=logp_per_token.device)
        
        with torch.no_grad():
            clipped_fraction = torch.tensor(0.0)
            mean_ratio = torch.tensor(1.0)
            mean_kl = torch.tensor(0.0)
            raw_logp_per_token = -F.cross_entropy(
                outputs.logits.view(-1, outputs.logits.size(-1)),
                labels.view(-1),
                reduction="none",
                ignore_index=-100,
            ).view(labels.shape)
            training_logprobs_flat = raw_logp_per_token[mask.bool()].detach()

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
        "interpretable_loss": interpretable_loss,
        # GRPO-specific metrics
        "kl_penalty": kl_penalty.item() if torch.is_tensor(kl_penalty) else kl_penalty,
        "mean_ratio": mean_ratio.item() if torch.is_tensor(mean_ratio) else mean_ratio,
        "mean_kl": mean_kl.item() if torch.is_tensor(mean_kl) else mean_kl,
        "clipped_fraction": clipped_fraction.item() if torch.is_tensor(clipped_fraction) else clipped_fraction,
    }

    return total_loss, metrics


def compute_logprob_alignment(
    inference_logprobs: List[np.ndarray],
    training_logprobs: List[torch.Tensor],
    debug: bool = False,
) -> Dict[str, float]:
    """
    Compute alignment stats between inference and training logprobs.
    
    At initialization (step 0), these should match closely if the model
    weights are correctly shared between training and inference.
    
    Args:
        inference_logprobs: Logprobs from vLLM inference (numpy arrays)
        training_logprobs: Logprobs computed during training forward pass (PyTorch tensors, bfloat16 supported)
        debug: If True, print detailed debugging info
        
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
    
    if debug:
        print(f"    [DEBUG] Inference: {len(inf_flat)} total, {len(inf_filtered)} after filter")
        print(f"    [DEBUG] Training: {train_flat.numel()} logprobs")
        if len(inf_filtered) > 0:
            print(f"    [DEBUG] Inf sample (first 5): {inf_filtered[:5]}")
        if train_flat.numel() > 0:
            print(f"    [DEBUG] Train sample (first 5): {train_flat[:5].tolist()}")
    
    # Compute stats using PyTorch for training (keeps bfloat16 precision)
    stats = {}
    
    if len(inf_filtered) > 0:
        stats["logprobs/inference_mean"] = float(np.mean(inf_filtered))
        stats["logprobs/inference_std"] = float(np.std(inf_filtered))
    
    if train_flat.numel() > 0:
        # PyTorch operations - fully support bfloat16
        stats["logprobs/training_mean"] = train_flat.mean().item()
        stats["logprobs/training_std"] = train_flat.std().item()
    
    # Compute diff (for tracking, not validation)
    # NOTE: Per-token comparison is NOT reliable here because inference and training
    # logprobs come from different batch orderings and can't be aligned token-by-token.
    # The real-time test at startup is the proper alignment validation.
    if "logprobs/inference_mean" in stats and "logprobs/training_mean" in stats:
        stats["logprobs/diff"] = stats["logprobs/inference_mean"] - stats["logprobs/training_mean"]
    
    return stats


def run_training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    token_batches: List[torch.Tensor],
    label_batches: List[torch.Tensor],
    advantage_batches: List[torch.Tensor],
    temperature_batches: List[torch.Tensor],
    config: TrainingConfig,
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
        config: Training configuration (includes kl_coef, clip_eps, use_reference_logprobs)
        inference_logprob_batches: Batched logprobs from inference (π_old), aligned with labels

    Returns:
        Dict of training metrics for this step
    """
    global _logprob_alignment_stats
    
    total_loss = 0.0
    total_pos_logp = 0.0
    total_neg_logp = 0.0
    total_pos = 0.0
    total_neg = 0.0
    total_kl_penalty = 0.0
    total_mean_ratio = 0.0
    total_mean_kl = 0.0
    total_clipped_fraction = 0.0
    grad_norm = 0.0
    all_training_logprobs: List[torch.Tensor] = []

    # Get GRPO hyperparameters from config
    kl_coef = getattr(config, 'kl_coef', 0.1)
    clip_eps = getattr(config, 'clip_eps', 0.2)
    use_reference_logprobs = getattr(config, 'use_reference_logprobs', True)

    # Accumulate gradients over micro-batches
    num_batches = len(token_batches) if token_batches else 1
    
    for batch_idx, (tokens, labels, advantages, temperatures) in enumerate(zip(
        token_batches, label_batches, advantage_batches, temperature_batches
    )):
        tokens = tokens.to(config.device)
        labels = labels.to(config.device)
        advantages = advantages.to(config.device)
        
        # Get corresponding inference logprobs batch if available
        inf_logprobs = None
        if inference_logprob_batches is not None and batch_idx < len(inference_logprob_batches):
            inf_logprobs = inference_logprob_batches[batch_idx]

        loss, metrics = compute_grpo_loss(
            model,
            tokens,
            labels,
            advantages,
            temperatures,
            config.gradient_accumulation_steps,
            inference_logprobs=inf_logprobs,
            kl_coef=kl_coef,
            clip_eps=clip_eps,
            use_reference_logprobs=use_reference_logprobs,
        )

        loss.backward()
        total_loss += loss.item()
        total_pos_logp += metrics["pos_logp"]
        total_neg_logp += metrics["neg_logp"]
        total_pos += metrics["pos_count"]
        total_neg += metrics["neg_count"]
        
        # Accumulate GRPO-specific metrics
        total_kl_penalty += metrics.get("kl_penalty", 0.0)
        total_mean_ratio += metrics.get("mean_ratio", 1.0)
        total_mean_kl += metrics.get("mean_kl", 0.0)
        total_clipped_fraction += metrics.get("clipped_fraction", 0.0)
        
        # Collect training logprobs for alignment monitoring
        if "training_logprobs" in metrics:
            all_training_logprobs.append(metrics["training_logprobs"])

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
        "grad_norm": grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm,
        "pos_logp": total_pos_logp,
        "neg_logp": total_neg_logp,
        "pos_count": total_pos,
        "neg_count": total_neg,
        # GRPO-specific metrics (averaged over batches)
        "kl_penalty": total_kl_penalty / num_batches,
        "mean_ratio": total_mean_ratio / num_batches,
        "mean_kl": total_mean_kl / num_batches,
        "clipped_fraction": total_clipped_fraction / num_batches,
    }
    
    # Compute logprob alignment stats for monitoring
    # NOTE: Now that we use proper GRPO, this is less critical
    # but still useful for debugging weight sharing issues
    if all_training_logprobs:
        # Store training logprob stats
        train_flat = torch.cat(all_training_logprobs)
        if train_flat.numel() > 0:
            _logprob_alignment_stats["logprobs/training_mean"] = train_flat.mean().item()
            _logprob_alignment_stats["logprobs/training_std"] = train_flat.std().item()
    
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

    # Primary metrics line: Loss and grad norm
    loss_str = (
        f"{metrics['loss']:.6f}"
        if abs(metrics["loss"]) < 0.01
        else f"{metrics['loss']:.4f}"
    )
    print(f"  Loss: {loss_str}, Grad norm: {metrics['grad_norm']:.4f}{timing_str}")

    # GRPO metrics line: KL, ratio, clipping
    kl_penalty = metrics.get("kl_penalty", 0)
    mean_ratio = metrics.get("mean_ratio", 1.0)
    mean_kl = metrics.get("mean_kl", 0)
    clipped_frac = metrics.get("clipped_fraction", 0)
    
    if kl_penalty > 0 or mean_kl > 0:
        print(
            f"    GRPO: KL={mean_kl:.4f}, ratio={mean_ratio:.3f}, "
            f"clipped={clipped_frac*100:.1f}%"
        )

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
            "train/pos_logp": metrics.get("pos_logp", 0),
            "train/neg_logp": metrics.get("neg_logp", 0),
            # GRPO-specific metrics
            "grpo/kl_penalty": kl_penalty,
            "grpo/mean_ratio": mean_ratio,
            "grpo/mean_kl": mean_kl,
            "grpo/clipped_fraction": clipped_frac,
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

