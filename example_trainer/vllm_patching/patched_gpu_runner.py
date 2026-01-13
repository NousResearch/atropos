"""
Patched GPU Model Runner - Enables shared memory weight updates.

This patches vLLM's GPUModelRunner to:
1. Call share_memory_() on model weights after loading
2. Spawn a daemon process that receives NCCL weight updates from trainers

The key insight is that share_memory_() makes tensors accessible from 
multiple processes. The daemon receives updates via NCCL and copies them
directly into the shared tensors, which vLLM reads for inference.

CRITICAL: This module must be imported and apply_patches() called BEFORE
any vLLM imports. The patches MUST happen before vLLM caches module references.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

# Lazy imports to avoid circular dependencies
if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner


# Flag to track if patches have been applied
_PATCHES_APPLIED = False
_PATCHED_RUNNER_CLASS = None


def apply_patches() -> bool:
    """
    Apply patches to vLLM's GPUModelRunner in ALL locations.
    
    This must be called BEFORE importing vLLM's engine classes.
    Safe to call multiple times (idempotent).
    
    Returns True if patches were applied successfully.
    
    Usage:
        # CRITICAL: Import and call BEFORE any vLLM imports!
        import os
        os.environ["VLLM_ENABLE_SHARED_WEIGHTS"] = "1"
        
        from example_trainer.vllm_patching import apply_patches
        apply_patches()
        
        # Now import vLLM
        from vllm import AsyncLLM  # Uses patched runner
    """
    global _PATCHES_APPLIED, _PATCHED_RUNNER_CLASS
    
    if _PATCHES_APPLIED:
        return True
    
    try:
        # Import the source module and get original class
        import vllm.v1.worker.gpu_model_runner as gpu_model_runner_module
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner as OriginalRunner
        
        # Create the patched class
        PatchedRunner = _create_patched_runner(OriginalRunner)
        _PATCHED_RUNNER_CLASS = PatchedRunner
        
        # =================================================================
        # PATCH 1: Replace in source module
        # =================================================================
        gpu_model_runner_module.GPUModelRunner = PatchedRunner
        print("[vLLM Patch] ✓ Patched vllm.v1.worker.gpu_model_runner.GPUModelRunner")
        
        # =================================================================
        # PATCH 2: Replace in gpu_worker module (main usage location)
        # =================================================================
        try:
            import vllm.v1.worker.gpu_worker as gpu_worker_module
            gpu_worker_module.GPUModelRunner = PatchedRunner
            print("[vLLM Patch] ✓ Patched vllm.v1.worker.gpu_worker.GPUModelRunner")
        except ImportError:
            pass
        
        # =================================================================
        # PATCH 3: Update sys.modules entry for source module
        # =================================================================
        # This ensures new imports get the patched version
        if 'vllm.v1.worker.gpu_model_runner' in sys.modules:
            sys.modules['vllm.v1.worker.gpu_model_runner'].GPUModelRunner = PatchedRunner
        
        # =================================================================
        # PATCH 4: Patch GPUWorker if already imported
        # =================================================================
        try:
            if 'vllm.v1.worker.gpu_worker' in sys.modules:
                worker_module = sys.modules['vllm.v1.worker.gpu_worker']
                if hasattr(worker_module, 'GPUWorker'):
                    # Update any class-level references
                    worker_module.GPUModelRunner = PatchedRunner
        except Exception:
            pass
        
        _PATCHES_APPLIED = True
        print("[vLLM Patch] ✓ GPUModelRunner patched for shared memory updates")
        return True
        
    except ImportError as e:
        print(f"[vLLM Patch] Warning: Could not apply patches: {e}")
        print("[vLLM Patch] This may be due to vLLM version incompatibility")
        print("[vLLM Patch] Shared memory updates will not be available")
        return False
    except Exception as e:
        print(f"[vLLM Patch] Error applying patches: {e}")
        import traceback
        traceback.print_exc()
        return False


def _create_patched_runner(BaseRunner: type) -> type:
    """
    Create a patched GPUModelRunner class.
    
    Returns a new class that inherits from the original and adds
    shared memory + daemon functionality.
    """
    import torch
    import torch.multiprocessing as mp
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
        
        _shared_memory_setup_done = False
        weight_updater_process = None
        
        def load_model(self, *args, **kwargs) -> None:
            """Load model and set up shared memory + update daemon."""
            print(f"[vLLM Patch] PatchedGPUModelRunner.load_model() called!")
            
            # Call original load_model
            super().load_model(*args, **kwargs)
            
            print(f"[vLLM Patch] Model loaded, checking shared weights setup...")
            
            # Check if shared memory updates are enabled
            enable_shared = os.environ.get("VLLM_ENABLE_SHARED_WEIGHTS", "0") == "1"
            num_inference_nodes = int(os.environ.get("NUM_INFERENCE_NODES", "-1"))
            
            print(f"[vLLM Patch] VLLM_ENABLE_SHARED_WEIGHTS={enable_shared}, NUM_INFERENCE_NODES={num_inference_nodes}")
            
            if not enable_shared and num_inference_nodes < 0:
                print("[vLLM Patch] Shared weights disabled (set VLLM_ENABLE_SHARED_WEIGHTS=1 to enable)")
                return
            
            if self._shared_memory_setup_done:
                print("[vLLM Patch] Shared memory already set up, skipping")
                return
            
            print("[vLLM Patch] Setting up shared memory weight updates...", flush=True)
            
            try:
                self._setup_shared_memory()
                PatchedGPUModelRunner._shared_memory_setup_done = True
                print("[vLLM Patch] ✓ Shared memory setup complete!", flush=True)
            except Exception as e:
                print(f"[vLLM Patch] ERROR in _setup_shared_memory: {e}", flush=True)
                import traceback
                traceback.print_exc()
                return
            
            # Spawn weight updater daemon (optional - can be skipped for HTTP-only mode)
            skip_daemon = os.environ.get("VLLM_SKIP_WEIGHT_DAEMON", "0") == "1"
            if skip_daemon:
                print("[vLLM Patch] Skipping weight updater daemon (VLLM_SKIP_WEIGHT_DAEMON=1)", flush=True)
                return
            
            try:
                print("[vLLM Patch] Spawning weight updater daemon...", flush=True)
                self._spawn_weight_updater()
                print("[vLLM Patch] ✓ Weight updater daemon spawned!", flush=True)
            except Exception as e:
                print(f"[vLLM Patch] ERROR spawning weight updater: {e}", flush=True)
                import traceback
                traceback.print_exc()
                print("[vLLM Patch] Continuing without daemon (HTTP-only mode)", flush=True)
        
        def _setup_shared_memory(self) -> None:
            """Move model tensors to shared memory and export param info."""
            import json
            from pathlib import Path
            
            print("[vLLM Patch] _setup_shared_memory() starting...")
            
            # Get state dict
            state_dict = self.model.state_dict()
            print(f"[vLLM Patch] Model has {len(state_dict)} parameters")
            
            # Make entire model shareable via share_memory_() on each tensor
            shared_count = 0
            for key, val in state_dict.items():
                try:
                    if val.is_cuda:
                        val.share_memory_()
                        shared_count += 1
                except Exception as e:
                    print(f"[vLLM Patch] Warning: Could not share {key}: {e}")
            
            print(f"[vLLM Patch] Called share_memory_() on {shared_count} CUDA tensors")
            
            # Also try calling share_memory() on the model itself
            try:
                self.model.share_memory()
                print("[vLLM Patch] Called model.share_memory()")
            except Exception as e:
                print(f"[vLLM Patch] Note: model.share_memory() not available: {e}")
            
            # Export parameter info to JSON for trainer
            log_dir = os.environ.get("LOGDIR", ".")
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            json_path = Path(log_dir) / "vllm_bridge_config.json"
            
            param_mappings = {}
            param_names = []
            for name, tensor in state_dict.items():
                param_mappings[name] = {
                    "vllm_name": name,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                }
                param_names.append(name)
            
            # Get model info
            model_name = "unknown"
            tp_degree = 1
            try:
                model_name = str(self.model_config.model)
                tp_degree = self.parallel_config.tensor_parallel_size
            except Exception as e:
                print(f"[vLLM Patch] Warning: Could not get model config: {e}")
            
            info = {
                "model": model_name,
                "tp_degree": tp_degree,
                "dp_shard_degree": 1,
                "param_mappings": param_mappings,
                "param_names": sorted(param_names),
                "shared_weights_enabled": True,
                "num_params": len(param_names),
            }
            
            try:
                with open(json_path, "w") as f:
                    json.dump(info, f, indent=2)
                print(f"[vLLM Patch] ✓ Exported {len(param_mappings)} params to {json_path}")
            except Exception as e:
                print(f"[vLLM Patch] ERROR: Failed to export params: {e}")
                import traceback
                traceback.print_exc()
        
        def _spawn_weight_updater(self) -> None:
            """Spawn the daemon process for receiving weight updates."""
            print("[vLLM Patch] _spawn_weight_updater() called", flush=True)
            
            try:
                from vllm.distributed import get_tensor_model_parallel_rank
                print("[vLLM Patch] Imported get_tensor_model_parallel_rank", flush=True)
            except ImportError as e:
                print(f"[vLLM Patch] Could not import get_tensor_model_parallel_rank: {e}", flush=True)
                get_tensor_model_parallel_rank = lambda: 0
            
            # Get model configuration
            state_dict = self.model.state_dict()
            print(f"[vLLM Patch] Got state_dict with {len(state_dict)} params", flush=True)
            
            # Get attention head counts
            hf_config = self.model_config.hf_text_config
            num_heads = getattr(hf_config, "num_attention_heads", 0)
            num_kv_heads = self.model_config.get_total_num_kv_heads()
            print(f"[vLLM Patch] num_heads={num_heads}, num_kv_heads={num_kv_heads}", flush=True)
            
            # Get parallel configuration
            tp_rank = get_tensor_model_parallel_rank()
            print(f"[vLLM Patch] tp_rank={tp_rank}", flush=True)
            
            # Get GPU ID
            gpu_id = 0
            try:
                if hasattr(self, 'device'):
                    if hasattr(self.device, 'index'):
                        gpu_id = self.device.index or 0
                    elif isinstance(self.device, int):
                        gpu_id = self.device
            except Exception:
                gpu_id = tp_rank
            
            print(f"[vLLM Patch] Spawning weight updater: tp_rank={tp_rank}, gpu={gpu_id}", flush=True)
            
            # Spawn daemon process
            print("[vLLM Patch] Creating spawn context...", flush=True)
            ctx = mp.get_context("spawn")
            
            print("[vLLM Patch] Creating Process...", flush=True)
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
            
            print("[vLLM Patch] Starting daemon process...", flush=True)
            self.weight_updater_process.start()
            
            print(f"[vLLM Patch] ✓ Weight updater daemon started (PID: {self.weight_updater_process.pid})", flush=True)
    
    # Set proper class name
    PatchedGPUModelRunner.__name__ = "PatchedGPUModelRunner"
    PatchedGPUModelRunner.__qualname__ = "PatchedGPUModelRunner"
    
    return PatchedGPUModelRunner


def get_patched_runner() -> type | None:
    """Get the patched runner class if patches have been applied."""
    return _PATCHED_RUNNER_CLASS


def is_patched() -> bool:
    """Check if patches have been applied."""
    return _PATCHES_APPLIED


# Placeholder class for type checking
class PatchedGPUModelRunner:
    """
    Placeholder class for type checking.
    
    The actual patched class is created dynamically by _create_patched_runner()
    to properly inherit from vLLM's GPUModelRunner.
    """
    pass
