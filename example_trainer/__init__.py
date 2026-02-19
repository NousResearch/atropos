"""
GRPO (Group Relative Policy Optimization) Trainer

A training framework for fine-tuning language models with reinforcement learning,
designed to work with the Atropos environment system.

Supports three training modes:
- Legacy: Checkpoint-based training with vLLM restarts
- Shared vLLM: Single-copy mode with CUDA IPC (no model duplication!)
- LoRA: Adapter-only training with hot-swap capability
- LoRA restart: Adapter training with periodic fast vLLM restarts

Usage:
    # As CLI
    python -m example_trainer.grpo --model-name Qwen/Qwen2.5-3B-Instruct --training-steps 100

    # As library
    from example_trainer import (
        TrainingConfig,
        train_legacy,
        train_shared_vllm,
        train_lora,
        train_lora_restart,
    )

    config = TrainingConfig(model_name="Qwen/Qwen2.5-3B-Instruct", training_steps=100)
    train_legacy(config)
"""

from .cli import config_from_args, parse_args
from .config import TrainingConfig
from .trainers import train_legacy, train_lora, train_lora_restart, train_shared_vllm

__all__ = [
    "TrainingConfig",
    "train_legacy",
    "train_shared_vllm",
    "train_lora",
    "train_lora_restart",
    "parse_args",
    "config_from_args",
]
