"""
Patched GPU Model Runner - Enables shared memory weight updates.

This patches vLLM's GPUModelRunner to:
1. Call share_memory_() on model weights after loading
2. Spawn a daemon process that receives NCCL weight updates from trainers

The key insight is that share_memory_() makes tensors accessible from 
multiple processes. The daemon receives updates via NCCL and copies them
directly into the shared tensors, which vLLM reads for inference.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
import torch.multiprocessing as mp

# Lazy imports to avoid circular dependencies
if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner


# Flag to track if patches have been applied
_PATCHES_APPLIED = False


def apply_patches() -> None:
    """
    Apply patches to vLLM's GPUModelRunner.
    
    This must be called BEFORE importing vLLM's engine classes.
    Safe to call multiple times (idempotent).
    
    Usage:
        from example_trainer.vllm_patching import apply_patches
        apply_patches()
        
        from vllm import AsyncLLM  # Now uses patched runner
    """
    global _PATCHES_APPLIED
    
    if _PATCHES_APPLIED:
        return
    
    try:
        import vllm.v1.worker.gpu_worker
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner as OriginalRunner
        
        # Create patched class
        PatchedRunner = _create_patched_runner(OriginalRunner)
        
        # Replace in vllm module
        vllm.v1.worker.gpu_worker.GPUModelRunner = PatchedRunner
        
        _PATCHES_APPLIED = True
        print("[vLLM Patch] ✓ GPUModelRunner patched for shared memory updates")
        
    except ImportError as e:
        print(f"[vLLM Patch] Warning: Could not apply patches: {e}")
        print("[vLLM Patch] Shared memory updates will not be available")


def _create_patched_runner(BaseRunner: type) -> type:
    """
    Create a patched GPUModelRunner class.
    
    Returns a new class that inherits from the original and adds
    shared memory + daemon functionality.
    """
    from .weight_updater import weight_updater_process
    
    class PatchedGPUModelRunner(BaseRunner):
        """
        Patched GPUModelRunner that enables shared memory weight updates.
        
        After loading the model, this:
        1. Calls share_memory_() on all parameters to make them accessible
           from other processes
        2. Spawns a daemon process that joins NCCL groups with the trainer
           and receives weight updates
           
        The daemon copies updates directly into the shared tensors, so
        vLLM immediately sees the new weights for inference.
        """
        
        def load_model(self, *args, **kwargs) -> None:
            """Load model and set up shared memory + update daemon."""
            # Call original load_model
            super().load_model(*args, **kwargs)
            
            # Check if shared memory updates are enabled
            enable_shared = os.environ.get("VLLM_ENABLE_SHARED_WEIGHTS", "0") == "1"
            num_inference_nodes = int(os.environ.get("NUM_INFERENCE_NODES", -1))
            
            if not enable_shared and num_inference_nodes < 0:
                print("[vLLM Patch] Shared weights disabled (set VLLM_ENABLE_SHARED_WEIGHTS=1 to enable)")
                return
            
            print("[vLLM Patch] Setting up shared memory weight updates...")
            
            try:
                self._setup_shared_memory()
                self._spawn_weight_updater()
                print("[vLLM Patch] ✓ Shared memory updates enabled")
            except Exception as e:
                print(f"[vLLM Patch] Warning: Failed to set up shared memory: {e}")
                import traceback
                traceback.print_exc()
        
        def _setup_shared_memory(self) -> None:
            """Move model tensors to shared memory."""
            # Make entire model shareable
            self.model.share_memory()
            
            # Also share_memory_() on each parameter individually
            # (some implementations may need this)
            state_dict = self.model.state_dict()
            for key, val in state_dict.items():
                if val.is_cuda or val.device.type == 'cuda':
                    # For CUDA tensors, we need to ensure they're in shared memory
                    val.share_memory_()
            
            print(f"[vLLM Patch] Shared {len(state_dict)} tensors in model")
        
        def _spawn_weight_updater(self) -> None:
            """Spawn the daemon process for receiving weight updates."""
            try:
                from vllm.distributed import get_tensor_model_parallel_rank
            except ImportError:
                # Fallback for older vLLM versions
                get_tensor_model_parallel_rank = lambda: 0
            
            # Get model configuration
            state_dict = self.model.state_dict()
            
            # Get attention head counts
            hf_config = self.model_config.hf_text_config
            num_heads = getattr(hf_config, "num_attention_heads", 0)
            num_kv_heads = self.model_config.get_total_num_kv_heads()
            
            # Get parallel configuration
            tp_rank = get_tensor_model_parallel_rank()
            gpu_id = torch.cuda.device(self.device).idx if hasattr(self.device, 'idx') else 0
            
            print(f"[vLLM Patch] Spawning updater: tp_rank={tp_rank}, gpu={gpu_id}")
            
            # Spawn daemon process
            ctx = mp.get_context("spawn")
            self.weight_updater_process = ctx.Process(
                target=weight_updater_process,
                args=(
                    state_dict,
                    num_heads,
                    num_kv_heads,
                    tp_rank,
                    self.parallel_config.tensor_parallel_size,
                    gpu_id,
                ),
                daemon=True,
            )
            self.weight_updater_process.start()
            
            print(f"[vLLM Patch] Weight updater daemon started (PID: {self.weight_updater_process.pid})")
    
    return PatchedGPUModelRunner


class PatchedGPUModelRunner:
    """
    Placeholder class for type checking.
    
    The actual patched class is created dynamically by _create_patched_runner()
    to properly inherit from vLLM's GPUModelRunner.
    """
    pass


