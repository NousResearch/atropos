"""
vLLM Patching Module - Enables shared memory weight updates.

This module patches vLLM's GPUModelRunner to:
1. Call share_memory_() on model weights after loading
2. Spawn a daemon process that receives NCCL weight updates from trainers
3. Enable real-time weight synchronization without restarting vLLM

Usage:
    # Import this BEFORE importing vllm
    from example_trainer.vllm_patching import apply_patches
    apply_patches()
    
    # Then import vllm normally
    from vllm import AsyncLLM
"""

from .patched_gpu_runner import PatchedGPUModelRunner, apply_patches
from .weight_updater import weight_updater_process
from .distributed_utils import (
    init_process_group,
    broadcast_object_list,
    get_inference_urls,
    get_json_data,
)

__all__ = [
    "PatchedGPUModelRunner",
    "apply_patches",
    "weight_updater_process",
    "init_process_group",
    "broadcast_object_list",
    "get_inference_urls",
    "get_json_data",
]


