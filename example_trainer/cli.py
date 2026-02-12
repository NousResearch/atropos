"""
Command-line interface for GRPO trainer.

Provides modular argument group builders and configuration building.
This is the SINGLE SOURCE OF TRUTH for all CLI arguments.
"""

import argparse

import torch

from .config import TrainingConfig

# =============================================================================
# Argument Group Builders (modular, reusable)
# =============================================================================


def add_model_args(parser: argparse.ArgumentParser) -> None:
    """Add model-related arguments."""
    group = parser.add_argument_group("Model")
    group.add_argument(
        "--model",
        "--model-name",
        type=str,
        required=True,
        dest="model_name",
        help="HuggingFace model identifier (e.g., 'Qwen/Qwen2.5-1.5B-Instruct')",
    )


def add_training_args(parser: argparse.ArgumentParser) -> None:
    """Add core training arguments."""
    group = parser.add_argument_group("Training")
    group.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate for the optimizer",
    )
    group.add_argument(
        "--training-steps",
        type=int,
        default=10,
        help="Number of training steps to run",
    )
    group.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for training",
    )
    group.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    group.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=32,
        help="Number of gradient accumulation steps",
    )
    group.add_argument(
        "--optimizer",
        type=str,
        choices=["adamw", "adamw_8bit", "adamw_cpu", "adafactor"],
        default="adamw_8bit",
        help="Optimizer: 'adamw' (full precision), 'adamw_8bit' (8-bit states), "
        "'adamw_cpu' (CPU offload), 'adafactor' (no momentum)",
    )
    group.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (cuda/cpu)",
    )
    group.add_argument(
        "--save-path",
        type=str,
        default="trained_model_checkpoints",
        help="Directory to save model checkpoints",
    )
    group.add_argument(
        "--checkpoint-interval",
        type=int,
        default=3,
        help="Save checkpoint every N training steps (0 = only save final)",
    )


def add_grpo_args(parser: argparse.ArgumentParser) -> None:
    """Add GRPO/PPO hyperparameter arguments."""
    group = parser.add_argument_group("GRPO/PPO Hyperparameters")
    group.add_argument(
        "--kl-coef",
        type=float,
        default=0.1,
        help="KL divergence penalty coefficient (beta). Higher = more conservative.",
    )
    group.add_argument(
        "--clip-eps",
        type=float,
        default=0.2,
        help="PPO-style clipping epsilon. Clips ratio to [1-eps, 1+eps].",
    )
    group.add_argument(
        "--no-reference-logprobs",
        action="store_true",
        help="Disable use of inference logprobs as reference policy (not recommended).",
    )


def add_vllm_args(parser: argparse.ArgumentParser) -> None:
    """Add vLLM server arguments."""
    group = parser.add_argument_group("vLLM Server")
    group.add_argument(
        "--vllm-port",
        type=int,
        default=9001,
        help="Port for the vLLM server",
    )
    group.add_argument(
        "--gpu-memory-utilization",
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.45,
        dest="gpu_memory_utilization",
        help="GPU memory utilization for vLLM server (0.0-1.0)",
    )
    group.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum context length for vLLM",
    )
    group.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "auto"],
        help="Data type for model weights",
    )
    group.add_argument(
        "--vllm-restart-interval",
        type=int,
        default=3,
        help="Restart vLLM every N training steps (legacy mode only)",
    )


def add_atropos_args(parser: argparse.ArgumentParser) -> None:
    """Add Atropos API arguments."""
    group = parser.add_argument_group("Atropos API")
    group.add_argument(
        "--atropos-url",
        type=str,
        default="http://localhost:8000",
        help="URL of the Atropos API/environment server",
    )


def add_wandb_args(parser: argparse.ArgumentParser) -> None:
    """Add Weights & Biases arguments."""
    group = parser.add_argument_group("Weights & Biases")
    group.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    group.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Wandb project name",
    )
    group.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help="Wandb group name",
    )


def add_mode_args(parser: argparse.ArgumentParser) -> None:
    """Add training mode arguments."""
    group = parser.add_argument_group("Training Mode")
    group.add_argument(
        "--weight-bridge-mode",
        type=str,
        choices=["shared_vllm", "lora_only", "lora_nccl", "none"],
        default="none",
        help="Weight sync mode: 'shared_vllm', 'lora_only', 'lora_nccl', or 'none' (legacy)",
    )
    group.add_argument(
        "--vllm-config-path",
        type=str,
        default=None,
        help="Explicit path to vllm_bridge_config.json (auto-detected if not provided)",
    )


def add_lora_args(parser: argparse.ArgumentParser) -> None:
    """Add LoRA-specific arguments."""
    group = parser.add_argument_group("LoRA Configuration")
    group.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    group.add_argument(
        "--lora-alpha", type=int, default=32, help="LoRA alpha (scaling factor)"
    )
    group.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    group.add_argument(
        "--lora-target-modules",
        type=str,
        nargs="+",
        default=None,
        help="Module names to apply LoRA to (default: q_proj v_proj)",
    )


def add_nccl_args(parser: argparse.ArgumentParser) -> None:
    """Add NCCL weight bridge arguments (for lora_nccl mode)."""
    group = parser.add_argument_group("NCCL Weight Bridge (lora_nccl mode)")
    group.add_argument(
        "--nccl-init-method",
        type=str,
        default="tcp://localhost:29500",
        help="NCCL process group init method (tcp://host:port)",
    )
    group.add_argument(
        "--nccl-world-size",
        type=int,
        default=2,
        help="Total processes in NCCL group (trainer + vLLM instances)",
    )
    group.add_argument(
        "--nccl-sync-every-step",
        action="store_true",
        default=True,
        help="Sync weights after every step (true on-policy)",
    )
    group.add_argument(
        "--no-nccl-sync-every-step",
        action="store_false",
        dest="nccl_sync_every_step",
        help="Sync weights only at vllm_restart_interval",
    )


def add_distributed_args(parser: argparse.ArgumentParser) -> None:
    """Add distributed training arguments."""
    group = parser.add_argument_group("Distributed Training")
    group.add_argument("--trainer-rank", type=int, default=0, help="Trainer rank")
    group.add_argument("--world-size", type=int, default=1, help="World size")
    group.add_argument(
        "--init-method", type=str, default="env://", help="Distributed init method"
    )
    group.add_argument(
        "--num-inference-nodes", type=int, default=0, help="Number of inference nodes"
    )


def add_debug_args(parser: argparse.ArgumentParser) -> None:
    """Add debug/benchmark arguments."""
    group = parser.add_argument_group("Debug & Benchmarking")
    group.add_argument(
        "--debug-loading",
        action="store_true",
        help="Enable verbose debug output during model loading",
    )
    group.add_argument(
        "--benchmark",
        action="store_true",
        help="Enable benchmark timing output",
    )
    group.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Directory for log files",
    )


# =============================================================================
# Parser Builders
# =============================================================================


def create_base_parser(description: str) -> argparse.ArgumentParser:
    """Create a base parser with common formatting."""
    return argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )


def create_full_parser() -> argparse.ArgumentParser:
    """
    Create a parser with ALL arguments (for grpo.py multi-mode entry point).
    """
    parser = create_base_parser("GRPO Trainer - Multi-mode training")

    add_model_args(parser)
    add_training_args(parser)
    add_grpo_args(parser)
    add_vllm_args(parser)
    add_atropos_args(parser)
    add_wandb_args(parser)
    add_mode_args(parser)
    add_lora_args(parser)
    add_nccl_args(parser)
    add_distributed_args(parser)
    add_debug_args(parser)

    return parser


def create_unified_parser() -> argparse.ArgumentParser:
    """
    Create a parser for run.py (unified shared_vllm mode with integrated vLLM).
    """
    parser = create_base_parser(
        "Unified GRPO Trainer - Starts vLLM server and trainer in one command"
    )

    add_model_args(parser)
    add_training_args(parser)
    add_grpo_args(parser)
    add_vllm_args(parser)
    add_atropos_args(parser)
    add_wandb_args(parser)
    add_debug_args(parser)

    return parser


# =============================================================================
# Legacy API (backwards compatibility)
# =============================================================================


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the GRPO trainer (grpo.py).

    Returns:
        Parsed arguments namespace
    """
    parser = create_full_parser()
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
        vllm_restart_interval=getattr(args, "vllm_restart_interval", 3),
        vllm_port=args.vllm_port,
        vllm_gpu_memory_utilization=getattr(args, "gpu_memory_utilization", 0.45),
        max_model_len=getattr(args, "max_model_len", 4096),
        dtype=getattr(args, "dtype", "bfloat16"),
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_group=getattr(args, "wandb_group", None),
        weight_bridge_mode=getattr(args, "weight_bridge_mode", "none"),
        trainer_rank=getattr(args, "trainer_rank", 0),
        world_size=getattr(args, "world_size", 1),
        init_method=getattr(args, "init_method", "env://"),
        num_inference_nodes=getattr(args, "num_inference_nodes", 0),
        lora_r=getattr(args, "lora_r", 16),
        lora_alpha=getattr(args, "lora_alpha", 32),
        lora_dropout=getattr(args, "lora_dropout", 0.05),
        lora_target_modules=getattr(args, "lora_target_modules", None),
        vllm_config_path=getattr(args, "vllm_config_path", None),
        debug_loading=getattr(args, "debug_loading", False),
        benchmark=getattr(args, "benchmark", False),
        atropos_url=getattr(args, "atropos_url", "http://localhost:8000"),
        # NCCL settings (for lora_nccl mode)
        nccl_init_method=getattr(args, "nccl_init_method", "tcp://localhost:29500"),
        nccl_world_size=getattr(args, "nccl_world_size", 2),
        nccl_sync_every_step=getattr(args, "nccl_sync_every_step", True),
    )
