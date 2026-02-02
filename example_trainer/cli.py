"""
Command-line interface for GRPO trainer.

Provides argument parsing and configuration building.
"""

import argparse

import torch

from .config import TrainingConfig


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the GRPO trainer.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="GRPO Trainer with optional shared-weight vLLM integration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # === Core Training Arguments ===
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
        "--optimizer",
        type=str,
        choices=["adamw", "adamw_8bit", "adamw_cpu", "adafactor"],
        default="adamw_8bit",
        help="Optimizer: 'adamw' (full precision, ~32GB GPU), "
             "'adamw_8bit' (8-bit states, ~8GB GPU), "
             "'adamw_cpu' (CPU offload, ~0GB GPU, slower), "
             "'adafactor' (no momentum, ~8GB GPU)",
    )
    
    # === GRPO/PPO Hyperparameters ===
    parser.add_argument(
        "--kl-coef",
        type=float,
        default=0.1,
        help=(
            "KL divergence penalty coefficient (beta). "
            "Controls policy deviation from reference. "
            "Higher = more conservative, prevents reward hacking. "
            "0 = disabled (not recommended)."
        ),
    )
    parser.add_argument(
        "--clip-eps",
        type=float,
        default=0.2,
        help=(
            "PPO-style clipping epsilon. "
            "Clips importance ratio to [1-eps, 1+eps]. "
            "Prevents destabilizing large policy updates."
        ),
    )
    parser.add_argument(
        "--no-reference-logprobs",
        action="store_true",
        help=(
            "Disable use of inference logprobs as reference policy. "
            "Falls back to REINFORCE-style updates (not recommended)."
        ),
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
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=3,
        help="Save checkpoint every N training steps (0 = only save final)",
    )

    # === vLLM Arguments ===
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
    parser.add_argument(
        "--atropos-url",
        type=str,
        default="http://localhost:8000",
        help="URL of the Atropos API/environment server (e.g., gsm8k_server)",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.45,
        help="GPU memory utilization for vLLM server (0.0-1.0)",
    )

    # === Wandb Arguments ===
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

    # === Training Mode Arguments ===
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

    # === LoRA Arguments ===
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

    # === Single-Copy Mode Arguments ===
    parser.add_argument(
        "--single-copy",
        action="store_true",
        help=(
            "Enable TRUE single-copy mode (shared_vllm mode only). "
            "Trainer attaches to vLLM's model tensors via CUDA IPC. "
            "Only ONE copy of the model exists in GPU memory! "
            "Requires trainer and vLLM to be on the SAME GPU(s). "
            "vLLM must be started with VLLM_ENABLE_SHARED_WEIGHTS=1."
        ),
    )
    parser.add_argument(
        "--vllm-config-path",
        type=str,
        default=None,
        help=(
            "Explicit path to vllm_bridge_config.json. "
            "If not provided, auto-detects from LOGDIR, current directory, "
            "or /tmp/atropos_bridge. "
            "This file contains CUDA IPC handles created by vLLM."
        ),
    )

    # === Debug Flags ===
    parser.add_argument(
        "--debug-loading",
        action="store_true",
        help=(
            "Enable verbose debug output during model loading and IPC attachment. "
            "Useful for diagnosing single-copy mode issues."
        ),
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help=(
            "Enable benchmark timing output showing step time, sync time, "
            "data fetch time, and GPU memory usage per step."
        ),
    )

    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> TrainingConfig:
    """
    Build a TrainingConfig from parsed CLI arguments.
    
    Args:
        args: Parsed argparse namespace
        
    Returns:
        TrainingConfig instance
    """
    return TrainingConfig(
        model_name=args.model_name,
        lr=args.lr,
        training_steps=args.training_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optimizer=args.optimizer,
        device=args.device,
        save_path=args.save_path,
        checkpoint_interval=getattr(args, "checkpoint_interval", 3),
        # GRPO/PPO hyperparameters
        kl_coef=getattr(args, "kl_coef", 0.1),
        clip_eps=getattr(args, "clip_eps", 0.2),
        use_reference_logprobs=not getattr(args, "no_reference_logprobs", False),
        # vLLM settings
        vllm_restart_interval=args.vllm_restart_interval,
        vllm_port=args.vllm_port,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
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
        single_copy=getattr(args, "single_copy", False),
        vllm_config_path=getattr(args, "vllm_config_path", None),
        debug_loading=getattr(args, "debug_loading", False),
        benchmark=getattr(args, "benchmark", False),
        atropos_url=getattr(args, "atropos_url", "http://localhost:8000"),
    )

