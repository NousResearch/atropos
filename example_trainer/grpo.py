#!/usr/bin/env python3
"""
GRPO Trainer - Main Entry Point

This is the command-line entry point for the GRPO trainer.
For the actual implementation, see the modular files:

- config.py      - TrainingConfig class
- api.py         - Atropos API communication
- data.py        - Data processing and batching
- model.py       - Model loading and shared memory
- training.py    - Loss computation and training step
- checkpointing.py - Checkpoint saving
- vllm_manager.py  - vLLM process management
- trainers.py    - Training mode implementations
- cli.py         - CLI argument parsing

Usage:
    # Legacy mode (checkpoint + restart)
    python grpo.py --model-name Qwen/Qwen2.5-3B-Instruct --training-steps 100

    # Single-copy mode (shared memory)
    python grpo.py --model-name Qwen/Qwen2.5-3B-Instruct --weight-bridge-mode shared_vllm

    # LoRA mode (adapter training)
    python grpo.py --model-name Qwen/Qwen2.5-3B-Instruct --weight-bridge-mode lora_only
"""

from .cli import parse_args, config_from_args
from .trainers import train_legacy, train_shared_vllm, train_lora


def main():
    """Main entry point for GRPO trainer."""
    args = parse_args()
    config = config_from_args(args)

    print(f"Weight bridge mode: {config.weight_bridge_mode}")

    if config.weight_bridge_mode == "shared_vllm":
        # Single-copy mode: attach to vLLM's weights, update in-place
        train_shared_vllm(config)

    elif config.weight_bridge_mode == "lora_only":
        # LoRA mode: freeze base model, train adapters only
        train_lora(config)

    else:
        # Legacy mode: periodic checkpoint saves + vLLM restarts
        train_legacy(config)


if __name__ == "__main__":
    main()

