"""
Checkpoint saving utilities for GRPO trainer.

Handles saving model checkpoints for different training modes:
- Full model checkpoints (legacy and shared_vllm modes)
- LoRA adapter checkpoints

IMPORTANT: For shared_vllm mode, the model parameters are VIEWS into vLLM's
fused tensors (qkv_proj, gate_up_proj). This module handles unfusing them
back to HuggingFace format for safe checkpoint saving.
"""

import os
import shutil
from typing import Dict

import torch


def _ensure_contiguous_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """
    Create a state dict with contiguous tensors for safe saving.
    
    This is critical for shared_vllm mode where parameters are views into
    vLLM's fused tensors. Views may share storage and not be contiguous,
    which can cause issues when saving.
    
    Returns:
        State dict with all tensors made contiguous (copied if necessary)
    """
    state_dict = {}
    for name, param in model.named_parameters():
        # Check if tensor is a view (non-contiguous or shares storage)
        if not param.is_contiguous() or param.storage_offset() != 0:
            # Make a contiguous copy - this "unfuses" the view
            state_dict[name] = param.detach().clone().contiguous()
        else:
            state_dict[name] = param.detach()
    
    # Also include buffers
    for name, buffer in model.named_buffers():
        if not buffer.is_contiguous() or buffer.storage_offset() != 0:
            state_dict[name] = buffer.detach().clone().contiguous()
        else:
            state_dict[name] = buffer.detach()
    
    return state_dict


def save_checkpoint(
    model: torch.nn.Module,
    tokenizer,
    save_path: str,
    step: int,
    is_final: bool = False,
    safe_mode: bool = True,
) -> str:
    """
    Save full model checkpoint.

    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        save_path: Base directory for checkpoints
        step: Current training step
        is_final: Whether this is the final checkpoint
        safe_mode: If True, ensure all tensors are contiguous before saving.
                   This is important for shared_vllm mode where params are
                   views into fused vLLM tensors.

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

    if safe_mode:
        # For shared_vllm mode: ensure views are properly unfused
        print("  [Checkpoint] Using safe mode - ensuring contiguous tensors...")
        state_dict = _ensure_contiguous_state_dict(model)
        
        # Count how many were non-contiguous (views into fused tensors)
        view_count = sum(
            1 for name, param in model.named_parameters()
            if not param.is_contiguous() or param.storage_offset() != 0
        )
        if view_count > 0:
            print(f"  [Checkpoint] Unfused {view_count} view tensors (qkv/gate_up fusions)")
        
        # Save state dict manually, then save config separately
        torch.save(state_dict, os.path.join(checkpoint_path, "pytorch_model.bin"))
        model.config.save_pretrained(checkpoint_path)
    else:
        # Standard save (may have issues with view tensors)
        model.save_pretrained(checkpoint_path)
    
    tokenizer.save_pretrained(checkpoint_path)

    print("  Checkpoint saved.")
    return checkpoint_path


def save_lora_checkpoint(
    model: torch.nn.Module,
    save_path: str,
    step: int,
    is_final: bool = False,
) -> str:
    """
    Save LoRA adapter checkpoint.

    Only saves the LoRA adapter weights, not the full model.
    This results in much smaller checkpoint files.

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

