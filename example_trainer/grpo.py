#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) Trainer.

Supports four training modes:
- none (legacy): Periodic checkpoint saves + vLLM restarts
- shared_vllm: Single-copy mode with CUDA IPC weight sharing
- lora_only: LoRA adapter training with HTTP hot-swap (SLOW - needs --enforce-eager)
- lora_restart: LoRA training with vLLM restarts (FAST - CUDA graphs enabled)

Usage:
    # Legacy mode (manages vLLM internally)
    python -m example_trainer.grpo --model-name Qwen/Qwen2.5-3B-Instruct

    # Shared vLLM mode (requires external vLLM with VLLM_ENABLE_SHARED_WEIGHTS=1)
    python -m example_trainer.grpo --model-name Qwen/Qwen2.5-3B-Instruct \\
        --weight-bridge-mode shared_vllm

    # LoRA mode with HTTP hot-swap (SLOW - 13 TPS due to --enforce-eager)
    python -m example_trainer.grpo --model-name Qwen/Qwen2.5-3B-Instruct \\
        --weight-bridge-mode lora_only --lora-r 16 --lora-alpha 32

    # LoRA mode with vLLM restarts (FAST - 170 TPS with CUDA graphs)
    python -m example_trainer.grpo --model-name Qwen/Qwen2.5-3B-Instruct \\
        --weight-bridge-mode lora_restart --lora-r 16 --lora-alpha 32 \\
        --vllm-restart-interval 3
"""

from .cli import config_from_args, parse_args
from .trainers import train_legacy, train_lora, train_lora_restart, train_shared_vllm


def main():
    """Main entry point for GRPO trainer."""
    args = parse_args()
    config = config_from_args(args)

    print("\n" + "=" * 60)
    print("GRPO TRAINER")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Mode: {config.weight_bridge_mode}")
    print(f"Training steps: {config.training_steps}")
    print(f"GRPO: clip_eps={config.clip_eps}")
    print(f"{'='*60}\n")

    if config.weight_bridge_mode == "shared_vllm":
        # Single-copy mode: attach to vLLM's weights, update in-place
        train_shared_vllm(config)

    elif config.weight_bridge_mode == "lora_only":
        # LoRA mode: freeze base model, train adapters only (HTTP hot-swap)
        # WARNING: This is SLOW (~13 TPS) because it requires --enforce-eager
        train_lora(config)

    elif config.weight_bridge_mode == "lora_restart":
        # LoRA mode with vLLM restarts (FAST - uses CUDA graphs)
        # Restarts vLLM every vllm_restart_interval steps with new adapter
        train_lora_restart(config)

    else:
        # Legacy mode: periodic checkpoint saves + vLLM restarts
        train_legacy(config)


if __name__ == "__main__":
    main()
