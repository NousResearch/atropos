"""
Checkpoint saving utilities for GRPO trainer.

Handles saving model checkpoints for different training modes:
- Full model checkpoints (legacy and shared_vllm modes)
- LoRA adapter checkpoints
"""

import os
import shutil

import torch


def save_checkpoint(
    model: torch.nn.Module,
    tokenizer,
    save_path: str,
    step: int,
    is_final: bool = False,
) -> str:
    """
    Save full model checkpoint.

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

