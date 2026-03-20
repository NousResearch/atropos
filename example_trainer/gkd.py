#!/usr/bin/env python3
"""
Dedicated GKD trainer entry point.

Unlike `example_trainer.grpo`, this path is explicitly distillation-first:
- on-policy student rollouts come from the Atropos environment
- teacher supervision is consumed from distill_token_ids/distill_logprobs
- no GRPO ratio objective is used

This keeps teacher distillation conceptually separate from reward-based GRPO.
"""

from .cli import config_from_gkd_args, parse_gkd_args
from .trainers import train_legacy, train_lora, train_lora_restart, train_shared_vllm


def main():
    """Main entry point for the dedicated GKD trainer."""
    args = parse_gkd_args()
    config = config_from_gkd_args(args)

    print("\n" + "=" * 60)
    print("GKD TRAINER")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Mode: {config.weight_bridge_mode}")
    print(f"Training steps: {config.training_steps}")
    print(
        "Divergence: "
        f"{config.distill_loss_type}"
        + (
            f" (beta={config.distill_jsd_beta})"
            if config.distill_loss_type == "jsd"
            else ""
        )
    )
    print(
        f"Distill coef: {config.distill_coef}, temperature: {config.distill_temperature}"
    )
    print(f"{'='*60}\n")

    if config.weight_bridge_mode == "shared_vllm":
        train_shared_vllm(config)
    elif config.weight_bridge_mode == "lora_only":
        train_lora(config)
    elif config.weight_bridge_mode == "lora_restart":
        train_lora_restart(config)
    else:
        train_legacy(config)


if __name__ == "__main__":
    main()
