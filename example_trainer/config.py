"""
Training configuration for GRPO trainer.

This module contains the TrainingConfig class which defines all training
parameters, model settings, and operational modes.
"""

from typing import List, Literal, Optional

import torch
from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    """
    Training configuration for GRPO trainer.
    
    Supports three training modes:
    - 'none' (legacy): Periodic checkpoint saves + vLLM restarts
    - 'shared_vllm': Attach to vLLM's shared memory tensors, update in-place
    - 'lora_only': Freeze base model, train LoRA adapters only
    """

    # === Model Configuration ===
    model_name: str = Field(..., description="Name of the base model to train")
    
    # === Training Hyperparameters ===
    lr: float = Field(1e-5, description="Learning rate for the optimizer")
    training_steps: int = Field(10, description="Number of training steps")
    batch_size: int = Field(2, description="Batch size for training")
    seq_len: int = Field(2048, description="Sequence length for training")
    gradient_accumulation_steps: int = Field(
        32, description="Number of gradient accumulation steps"
    )
    optimizer: Literal["adamw", "adamw_8bit", "adamw_cpu", "adafactor"] = Field(
        "adamw_8bit",
        description="Optimizer to use: 'adamw' (full precision, ~32GB GPU), "
                    "'adamw_8bit' (8-bit states, ~8GB GPU, requires bitsandbytes), "
                    "'adamw_cpu' (CPU offload, ~0GB GPU, slower), "
                    "'adafactor' (no momentum, ~8GB GPU)"
    )
    
    # === Device & Storage ===
    device: str = Field(
        "cuda" if torch.cuda.is_available() else "cpu", 
        description="Device to train on"
    )
    save_path: str = Field(
        "trained_model_checkpoints", 
        description="Base path to save model checkpoints"
    )
    
    # === vLLM Server Configuration ===
    vllm_restart_interval: int = Field(
        3, description="Restart vLLM every N training steps (legacy mode)"
    )
    vllm_port: int = Field(9001, description="Port for the vLLM server")
    vllm_gpu_memory_utilization: float = Field(
        0.45, description="GPU memory utilization for vLLM server (0.0-1.0)"
    )

    # === Weights & Biases Configuration ===
    use_wandb: bool = Field(
        False, description="Whether to use Weights & Biases for logging"
    )
    wandb_project: Optional[str] = Field(None, description="Wandb project name")
    wandb_group: Optional[str] = Field(None, description="Wandb group name")

    # === Training Mode Configuration ===
    weight_bridge_mode: Literal["shared_vllm", "lora_only", "none"] = Field(
        "none",
        description=(
            "How to synchronize weights with inference server. "
            "'shared_vllm': attach to vLLM's shared memory tensors and update in-place. "
            "'lora_only': keep base model frozen, train/swap LoRA adapters. "
            "'none': legacy mode, restart vLLM with new checkpoint files."
        ),
    )
    
    # === Distributed Training Configuration ===
    trainer_rank: int = Field(
        0, description="Rank of this trainer in the distributed group"
    )
    world_size: int = Field(
        1, description="Total processes in the distributed group"
    )
    init_method: str = Field(
        "env://",
        description=(
            "PyTorch distributed init method URL. "
            "Use 'env://' to read MASTER_ADDR/MASTER_PORT from environment, "
            "or 'tcp://host:port' for explicit rendezvous."
        ),
    )
    num_inference_nodes: int = Field(
        0,
        description=(
            "Number of inference nodes (vLLM servers) to coordinate with. "
            "0 means single-node local mode."
        ),
    )

    # === LoRA Configuration ===
    lora_r: int = Field(16, description="LoRA rank (dimension of low-rank matrices)")
    lora_alpha: int = Field(32, description="LoRA alpha (scaling factor)")
    lora_dropout: float = Field(0.05, description="Dropout probability for LoRA layers")
    lora_target_modules: Optional[List[str]] = Field(
        None,
        description=(
            "List of module names to apply LoRA to. "
            "If None, defaults to ['q_proj', 'v_proj'] for most models."
        ),
    )

    # === Single-Copy Mode Configuration ===
    single_copy: bool = Field(
        False,
        description=(
            "Enable TRUE single-copy mode via CUDA IPC. "
            "The trainer attaches to vLLM's model tensors directly, "
            "meaning only ONE copy of the model exists in GPU memory. "
            "Requires trainer and vLLM to be on the SAME GPU(s). "
            "vLLM must be started with VLLM_ENABLE_SHARED_WEIGHTS=1."
        ),
    )
    vllm_config_path: Optional[str] = Field(
        None,
        description=(
            "Explicit path to vllm_bridge_config.json. "
            "If not provided, auto-detects from LOGDIR environment variable, "
            "current directory, or /tmp/atropos_bridge. "
            "This file is created by vLLM when VLLM_ENABLE_SHARED_WEIGHTS=1 "
            "and contains CUDA IPC handles for single-copy mode."
        ),
    )

    # === Debug & Benchmark Flags ===
    debug_loading: bool = Field(
        False,
        description=(
            "Enable verbose debug output during model loading and IPC attachment. "
            "Useful for diagnosing single-copy mode issues."
        ),
    )
    benchmark: bool = Field(
        False,
        description=(
            "Enable benchmark timing output showing step time, sync time, "
            "data fetch time, and GPU memory usage per step."
        ),
    )
    
    # === Atropos API Configuration ===
    atropos_url: str = Field(
        "http://localhost:8000",
        description=(
            "URL of the Atropos API server (environment server). "
            "Default is http://localhost:8000. Change for concurrent tests."
        ),
    )

