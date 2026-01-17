"""
vLLM Patching Module - Enables CUDA IPC shared memory for single-copy training.

This module patches vLLM's GPUModelRunner to:
1. Call share_memory_() on model weights after loading
2. Export CUDA IPC handles to vllm_bridge_config.json
3. Enable the trainer to attach to vLLM's tensors directly

The result: ONE copy of model weights in GPU memory, shared between
vLLM (inference) and the trainer (gradient updates).

Usage:
    # Set environment BEFORE importing
    import os
    os.environ["VLLM_ENABLE_SHARED_WEIGHTS"] = "1"
    
    # Import and apply patches BEFORE importing vllm
    from example_trainer.vllm_patching import apply_patches
    apply_patches()
    
    # Then import vllm normally
    from vllm import AsyncLLM
"""

from .patched_gpu_runner import (
    PatchedGPUModelRunner,
    apply_patches,
    get_patched_runner,
    is_patched,
)

__all__ = [
    "PatchedGPUModelRunner",
    "apply_patches",
    "get_patched_runner",
    "is_patched",
]
