"""
Patched GPU Model Runner - Enables CUDA IPC for single-copy training.

This patches vLLM's GPUModelRunner to:
1. Call share_memory_() on model weights after loading
2. Export CUDA IPC handles to vllm_bridge_config.json

The key insight is that CUDA IPC handles allow the trainer process to
attach to the EXACT SAME GPU memory that vLLM uses. This means:
- ONE copy of model weights in GPU memory
- Trainer's optimizer.step() updates vLLM's weights directly
- No synchronization needed - vLLM immediately sees new weights

CRITICAL: This module must be imported and apply_patches() called BEFORE
any vLLM imports. The patches MUST happen before vLLM caches module references.
"""

from __future__ import annotations

import os
import shutil
import sys

# Flag to track if patches have been applied
_PATCHES_APPLIED = False
_PATCHED_RUNNER_CLASS = None


def _patch_lora_triton_for_blackwell() -> bool:
    """
    Patch vLLM's LoRA Triton kernels to disable GDC (Grid Dependency Control).

    GDC is a Blackwell-specific feature that causes Triton compilation to fail
    on B200 GPUs. This patches the kernel_utils.py to disable GDC.

    Returns True if patch was applied successfully.
    """
    try:
        import vllm

        vllm_path = vllm.__path__[0]
        kernel_utils_path = f"{vllm_path}/lora/ops/triton_ops/kernel_utils.py"

        # Check if file exists
        if not os.path.exists(kernel_utils_path):
            print("[vLLM Patch] LoRA kernel_utils.py not found, skipping GDC patch")
            return False

        with open(kernel_utils_path, "r") as f:
            content = f.read()

        # Check if already patched
        if "PATCHED FOR B200" in content:
            print("[vLLM Patch] LoRA GDC already patched for B200")
            return True

        modified = False

        # Patch USE_GDC = True -> False
        if "USE_GDC = True" in content:
            content = content.replace(
                "USE_GDC = True",
                "USE_GDC = False  # PATCHED FOR B200 - GDC causes Triton compilation failure",
            )
            modified = True

        # Patch USE_GDC: tl.constexpr = True -> False
        if "USE_GDC: tl.constexpr = True" in content:
            content = content.replace(
                "USE_GDC: tl.constexpr = True",
                "USE_GDC: tl.constexpr = False  # PATCHED FOR B200",
            )
            modified = True

        # Patch the gdc_wait call itself
        if "tl.extra.cuda.gdc_wait()" in content:
            content = content.replace(
                "tl.extra.cuda.gdc_wait()",
                "pass  # tl.extra.cuda.gdc_wait() PATCHED FOR B200 - disabled",
            )
            modified = True

        if modified:
            with open(kernel_utils_path, "w") as f:
                f.write(content)
            print(f"[vLLM Patch] ✓ Patched LoRA Triton GDC in {kernel_utils_path}")

            # Clear Triton cache to force recompilation
            triton_cache = os.path.expanduser("~/.triton/cache")
            if os.path.exists(triton_cache):
                try:
                    shutil.rmtree(triton_cache)
                    print("[vLLM Patch] ✓ Cleared Triton cache")
                except Exception as e:
                    print(f"[vLLM Patch] Warning: Could not clear Triton cache: {e}")

            return True
        else:
            print("[vLLM Patch] No GDC patterns found to patch")
            return False

    except Exception as e:
        print(f"[vLLM Patch] Warning: Could not patch LoRA GDC: {e}")
        return False


def apply_patches() -> bool:
    """
    Apply patches to vLLM's GPUModelRunner in ALL locations.

    This must be called BEFORE importing vLLM's engine classes.
    Safe to call multiple times (idempotent).

    Also patches LoRA Triton kernels to disable GDC for B200 compatibility.

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

    # First, patch LoRA Triton for B200 compatibility
    _patch_lora_triton_for_blackwell()

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
        if "vllm.v1.worker.gpu_model_runner" in sys.modules:
            sys.modules["vllm.v1.worker.gpu_model_runner"].GPUModelRunner = (
                PatchedRunner
            )

        # =================================================================
        # PATCH 4: Patch GPUWorker if already imported
        # =================================================================
        try:
            if "vllm.v1.worker.gpu_worker" in sys.modules:
                worker_module = sys.modules["vllm.v1.worker.gpu_worker"]
                if hasattr(worker_module, "GPUWorker"):
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
    CUDA IPC export functionality for single-copy training.
    """

    class PatchedGPUModelRunner(BaseRunner):
        """
        Patched GPUModelRunner that enables CUDA IPC for single-copy training.

        After loading the model, this:
        1. Calls share_memory_() on all parameters
        2. Exports CUDA IPC handles to vllm_bridge_config.json

        The trainer reads these IPC handles and attaches to the SAME
        GPU memory, so optimizer.step() updates weights that vLLM
        immediately sees for inference.
        """

        _shared_memory_setup_done = False

        def load_model(self, *args, **kwargs) -> None:
            """Load model and set up shared memory + update daemon."""
            print("[vLLM Patch] PatchedGPUModelRunner.load_model() called!")

            # Call original load_model
            super().load_model(*args, **kwargs)

            print("[vLLM Patch] Model loaded, checking shared weights setup...")

            # Check if shared memory updates are enabled
            enable_shared = os.environ.get("VLLM_ENABLE_SHARED_WEIGHTS", "0") == "1"
            num_inference_nodes = int(os.environ.get("NUM_INFERENCE_NODES", "-1"))

            print(
                f"[vLLM Patch] VLLM_ENABLE_SHARED_WEIGHTS={enable_shared}, NUM_INFERENCE_NODES={num_inference_nodes}"
            )

            if not enable_shared and num_inference_nodes < 0:
                print(
                    "[vLLM Patch] Shared weights disabled (set VLLM_ENABLE_SHARED_WEIGHTS=1 to enable)"
                )
                return

            if self._shared_memory_setup_done:
                print("[vLLM Patch] Shared memory already set up, skipping")
                return

            print("[vLLM Patch] Setting up shared memory weight updates...", flush=True)

            try:
                self._setup_shared_memory()
                PatchedGPUModelRunner._shared_memory_setup_done = True
                print("[vLLM Patch] ✓ Shared memory setup complete!", flush=True)
                print(
                    "[vLLM Patch] ✓ IPC handles exported - trainer can now attach!",
                    flush=True,
                )
            except Exception as e:
                print(f"[vLLM Patch] ERROR in _setup_shared_memory: {e}", flush=True)
                import traceback

                traceback.print_exc()
                return

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
            # Allow explicit config path via env var, otherwise use LOGDIR
            config_path = os.environ.get("VLLM_BRIDGE_CONFIG_PATH")
            if config_path:
                json_path = Path(config_path)
                json_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                log_dir = os.environ.get("LOGDIR", ".")
                Path(log_dir).mkdir(parents=True, exist_ok=True)
                json_path = Path(log_dir) / "vllm_bridge_config.json"

            param_mappings = {}
            param_names = []
            ipc_handles = {}

            for name, tensor in state_dict.items():
                param_mappings[name] = {
                    "vllm_name": name,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "device": str(tensor.device),
                }
                param_names.append(name)

                # Export CUDA IPC handles for true single-copy mode
                if tensor.is_cuda:
                    try:
                        import base64

                        storage = tensor.untyped_storage()
                        share_data = storage._share_cuda_()

                        # share_data is a tuple of 8 items - we need ALL of them:
                        # [0] = device index (int)
                        # [1] = cudaIpcMemHandle_t (bytes)
                        # [2] = storage size (int)
                        # [3] = storage offset in original (int)
                        # [4] = ref counter handle (bytes - filename)
                        # [5] = ref counter offset (int)
                        # [6] = event handle (bytes)
                        # [7] = event sync required (bool)

                        ipc_handles[name] = {
                            "device_index": share_data[0],
                            "ipc_handle_b64": base64.b64encode(share_data[1]).decode(
                                "ascii"
                            ),
                            "storage_size": share_data[2],
                            "storage_offset_orig": share_data[3],
                            "ref_counter_handle_b64": base64.b64encode(
                                share_data[4]
                            ).decode("ascii"),
                            "ref_counter_offset": share_data[5],
                            "event_handle_b64": base64.b64encode(share_data[6]).decode(
                                "ascii"
                            ),
                            "event_sync_required": share_data[7],
                            # Tensor metadata for reconstruction
                            "tensor_storage_offset": tensor.storage_offset(),
                            "shape": list(tensor.shape),
                            "stride": list(tensor.stride()),
                            "dtype": str(tensor.dtype),
                        }
                    except Exception as e:
                        print(
                            f"[vLLM Patch] Could not get IPC handle for {name}: {e}",
                            flush=True,
                        )
                        import traceback

                        traceback.print_exc()

            print(
                f"[vLLM Patch] Exported {len(ipc_handles)} IPC handles for single-copy mode",
                flush=True,
            )

            # Get model info
            model_name = "unknown"
            tp_degree = 1
            try:
                model_name = str(self.model_config.model)
                tp_degree = self.parallel_config.tensor_parallel_size
            except Exception as e:
                print(f"[vLLM Patch] Warning: Could not get model config: {e}")

            import base64

            # Convert bytes to base64 for JSON serialization
            def serialize_ipc_handles(handles):
                result = {}
                for k, v in handles.items():
                    if isinstance(v, bytes):
                        result[k] = {"_bytes_b64_": base64.b64encode(v).decode("ascii")}
                    elif isinstance(v, dict):
                        result[k] = serialize_ipc_handles(v)
                    else:
                        result[k] = v
                return result

            serialized_ipc_handles = (
                serialize_ipc_handles(ipc_handles) if ipc_handles else {}
            )

            info = {
                "model": model_name,
                "tp_degree": tp_degree,
                "dp_shard_degree": 1,
                "param_mappings": param_mappings,
                "param_names": sorted(param_names),
                "ipc_handles": serialized_ipc_handles,
                "shared_weights_enabled": True,
                "num_params": len(param_names),
                "single_copy_enabled": len(ipc_handles) > 0,
            }

            try:
                with open(json_path, "w") as f:
                    json.dump(info, f, indent=2)
                print(
                    f"[vLLM Patch] ✓ Exported {len(param_mappings)} params to {json_path}"
                )
            except Exception as e:
                print(f"[vLLM Patch] ERROR: Failed to export params: {e}")
                import traceback

                traceback.print_exc()

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
