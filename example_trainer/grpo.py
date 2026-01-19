import argparse
import atexit
import json
import math
import os
import random
import shutil
import string
import subprocess
import time
from typing import List, Literal, Optional, Tuple

import numpy as np
import requests
import torch
import torch.nn.functional as F
import wandb  # Added for logging
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

# Weight bridge removed - single-copy mode uses direct CUDA IPC instead

# Import PEFT for LoRA training
try:
    from peft import LoraConfig, TaskType, get_peft_model

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# Global variable to keep track of the vLLM process
vllm_process = None


def cleanup_vllm():
    global vllm_process
    if vllm_process:
        print("\nTerminating vLLM process...")
        vllm_process.terminate()
        try:
            vllm_process.wait(timeout=5)  # Wait a bit for graceful shutdown
            print("vLLM process terminated.")
        except subprocess.TimeoutExpired:
            print("vLLM process did not terminate gracefully, killing.")
            vllm_process.kill()
            vllm_process.wait()
            print("vLLM process killed.")
        vllm_process = None


# Register the cleanup function to be called on script exit
atexit.register(cleanup_vllm)


class TrainingConfig(BaseModel):
    """
    Training details, model, etc
    """

    model_name: str = Field(..., description="Name of the base model to train")
    lr: float = Field(1e-5, description="Learning rate for the optimizer")
    training_steps: int = Field(
        10, description="Number of training steps"
    )  # Renamed from epochs
    batch_size: int = Field(
        2, description="Batch size for training (will be handled by get_data)"
    )
    seq_len: int = Field(2048, description="Sequence length for training")
    gradient_accumulation_steps: int = Field(
        32, description="Number of gradient accumulation steps"
    )
    device: str = Field(
        "cuda" if torch.cuda.is_available() else "cpu", description="Device to train on"
    )
    save_path: str = Field(
        "trained_model_checkpoints", description="Base path to save model checkpoints"
    )
    vllm_restart_interval: int = Field(
        3, description="Restart vLLM every N training steps"
    )
    vllm_port: int = Field(9001, description="Port for the vLLM server")
    vllm_gpu_memory_utilization: float = Field(
        0.45, description="GPU memory utilization for vLLM server (0.0-1.0)"
    )

    # Wandb configuration
    use_wandb: bool = Field(
        False, description="Whether to use Weights & Biases for logging"
    )
    wandb_project: Optional[str] = Field(None, description="Wandb project name")
    wandb_group: Optional[str] = Field(None, description="Wandb group name")

    # Pipeline / weight bridge configuration
    weight_bridge_mode: Literal["shared_vllm", "lora_only", "none"] = Field(
        "none",
        description=(
            "How to synchronize weights with inference server. "
            "'shared_vllm': attach to vLLM's shared memory tensors and update in-place. "
            "'lora_only': keep base model frozen, train/swap LoRA adapters. "
            "'none': legacy mode, restart vLLM with new checkpoint files."
        ),
    )
    trainer_rank: int = Field(
        0,
        description="Rank of this trainer in the distributed group (for shared_vllm mode)",
    )
    world_size: int = Field(
        1,
        description="Total processes in the distributed group (for shared_vllm mode)",
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

    # LoRA configuration (for lora_only mode)
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

    # Single-copy mode (TRUE shared memory - no extra model copy)
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

    # Debug flags
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


def check_atropos_api(timeout: float = 30.0) -> bool:
    """
    Check if the Atropos API server is reachable.

    Args:
        timeout: Maximum time to wait for the server

    Returns:
        True if server is reachable
    """
    import time as _time

    start = _time.time()
    while _time.time() - start < timeout:
        try:
            response = requests.get("http://localhost:8000/info", timeout=2)
            if response.status_code == 200:
                print("[Trainer] ✓ Atropos API server is reachable")
                return True
        except requests.exceptions.ConnectionError:
            pass
        except Exception as e:
            print(f"[Trainer] Waiting for Atropos API... ({e})")
        _time.sleep(1)

    print("[Trainer] ⚠ Warning: Atropos API server not reachable")
    return False


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
def register_trainer(config: TrainingConfig):
    """
    Register the trainer with the Atropos API.

    Verifies registration succeeded before returning.
    """
    response = requests.post(
        "http://localhost:8000/register",
        json={
            # wandb fields are required strings - use empty string if None
            "wandb_group": config.wandb_group or "",
            "wandb_project": config.wandb_project or "",
            "batch_size": config.batch_size * config.gradient_accumulation_steps,
            "max_token_len": config.seq_len,
            "starting_step": 0,
            "checkpoint_dir": config.save_path,
            "save_checkpoint_interval": config.training_steps,
            "num_steps": config.training_steps,
        },
        timeout=10,
    )

    # Check for HTTP errors
    response.raise_for_status()

    # Verify we got a valid response with UUID
    data = response.json()
    if "uuid" not in data:
        raise RuntimeError(f"Registration failed: {data}")

    print(f"[Trainer] ✓ Registered with Atropos API (uuid: {data['uuid']})")


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
def get_batch():
    data = requests.get("http://localhost:8000/batch", timeout=10).json()

    # Check if there was an error (trainer not registered)
    if data.get("status") == "error":
        raise RuntimeError(f"Atropos API error: {data.get('message', 'Unknown error')}")

    return data


def pad_data_to_good_offset(data, batch_size: int):
    max_token_len = max(
        [max([len(x) for x in item["tokens"]]) for item in data["batch"]]
    )
    # usually 64 is a good choice to ensure nonweird scaling behavior on GPUS
    # so we pad to the nearest multiple of 64
    good_multiple = 64
    if (max_token_len - 1) % (good_multiple) != 0:
        max_token_len = math.ceil((max_token_len - 1) / (good_multiple)) * good_multiple
        token_setup_len = (
            max_token_len + 1
        )  # add 1 so we can make it causal at the proper length
    else:
        token_setup_len = max_token_len
        max_token_len = (
            max_token_len - 1
        )  # since it's causal we need to remove the last bit...
    # pad all tokens to max_token_len and add to lists
    input_ids = list()
    labels = list()
    advantages = list()
    lengths = list()
    temperatures = list()
    for item in data["batch"]:
        scores = item["scores"]
        scores = np.array(scores)
        # check if we have more than 1 score...
        if len(scores) > 1:
            scores = scores - scores.mean()
            scores = scores / max(scores.std(), 1e-8)
        item["scores"] = scores
        if item["overrides"] is not None:
            for i in range(len(item["overrides"])):
                if item["overrides"][i].get("set_advantage_to_zero", False):
                    item["scores"][i] = 0
        for i in range(len(item["tokens"])):
            lengths.append(
                math.ceil((len(item["tokens"][i]) - 1) / (good_multiple))
                * good_multiple
            )
            label_item = np.concatenate(
                [
                    np.array(item["masks"][i]),
                    np.full(
                        max(0, token_setup_len - len(item["tokens"][i])),
                        -100,
                        dtype=np.int32,
                    ),
                ]
            )
            item["tokens"][i] = np.concatenate(
                [
                    np.array(item["tokens"][i]),
                    np.zeros(
                        max(0, token_setup_len - len(item["tokens"][i])), dtype=np.int32
                    ),
                ]
            )
            input_ids.append(item["tokens"][i][:-1])
            labels.append(label_item[1:])
            advantages.append(item["scores"][i])
            # per-sample override -> group generation_params -> group_overrides - > 1.0
            # need to update docs since this lets you set the temperature for each sample from the override
            t = 1.0
            if (
                item.get("overrides")
                and i < len(item["overrides"])
                and isinstance(item["overrides"][i], dict)
                and ("temperature" in item["overrides"][i])
            ):
                t = float(item["overrides"][i]["temperature"])
            elif item.get("generation_params") and (
                "temperature" in item["generation_params"]
            ):
                t = float(item["generation_params"]["temperature"])
            elif item.get("group_overrides") and (
                "temperature" in item["group_overrides"]
            ):
                t = float(item["group_overrides"]["temperature"])
            temperatures.append(t)
    # combine all lists into tensors
    token_batches = []
    label_batches = []
    advantage_batches = []
    temperature_batches = []
    for i in range(len(input_ids) // batch_size):
        token_batches.append(
            torch.tensor(
                np.stack(input_ids[i * batch_size : (i + 1) * batch_size], axis=0)
            )
        )
        label_batches.append(
            torch.tensor(
                np.stack(labels[i * batch_size : (i + 1) * batch_size], axis=0)
            )
        )
        advantage_batches.append(
            torch.tensor(
                np.stack(advantages[i * batch_size : (i + 1) * batch_size], axis=0)
            ).view(-1, 1)
        )
        # Temperatures: one per sample, shaped for broadcasting to [B, 1, 1]
        temperature_batches.append(
            torch.tensor(
                np.array(
                    temperatures[i * batch_size : (i + 1) * batch_size],
                    dtype=np.float32,
                )
            ).view(-1, 1, 1)
        )

    return token_batches, label_batches, advantage_batches, temperature_batches


def get_data(
    batch_size: int, seq_len: int
) -> List[
    Tuple[
        List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]
    ]
]:
    """
    getting data from the api
    """
    batches = []
    while True:
        data = get_batch()
        if data["batch"] is not None:
            # Save the batch
            with open("temp.json", "w", encoding="utf-8") as f:
                json.dump(data, f)
            # In case the inference runs ahead of the training, we loop until we don't have any more data
            batches.append(pad_data_to_good_offset(data, batch_size))
        elif len(batches) > 0:
            # Return the batches
            return batches
        else:
            time.sleep(1)


# =============================================================================
# Common Training Helpers (shared across all modes)
# =============================================================================


def setup_wandb(config: TrainingConfig) -> bool:
    """
    Initialize Weights & Biases logging if enabled.

    Args:
        config: Training configuration

    Returns:
        True if wandb is active, False otherwise
    """
    if not config.use_wandb:
        return False

    if not config.wandb_project:
        print("Warning: wandb_project not set, disabling wandb.")
        return False

    # Generate random group name if not provided
    if not config.wandb_group:
        config.wandb_group = "".join(
            random.choices(string.ascii_letters + string.digits, k=8)
        )

    try:
        wandb.init(
            project=config.wandb_project,
            group=config.wandb_group,
            config=config.dict(),
        )
        print(
            f"Wandb logging enabled. Run: {wandb.run.name} "
            f"(Project: {config.wandb_project})"
        )
        return True
    except Exception as e:
        print(f"Error initializing wandb: {e}. Disabling wandb.")
        return False


def _attach_to_vllm_shared_tensors(
    config: TrainingConfig,
    bridge_config_path: str,
) -> Optional[torch.nn.Module]:
    """
    Attach to vLLM's shared tensors via CUDA IPC (true single-copy mode).

    This creates a model whose parameters point to the SAME GPU memory as vLLM,
    meaning only ONE copy of the model exists in GPU memory.

    Args:
        config: Training configuration
        bridge_config_path: Path to vllm_bridge_config.json

    Returns:
        Model with parameters pointing to vLLM's tensors, or None if not possible
    """
    print(f"[Setup] Reading bridge config from: {bridge_config_path}")
    try:
        with open(bridge_config_path, "r") as f:
            bridge_config = json.load(f)
        print(f"[Setup] Bridge config keys: {list(bridge_config.keys())}")
    except Exception as e:
        print(f"[Setup] Could not read bridge config: {e}")
        return None

    single_copy_enabled = bridge_config.get("single_copy_enabled", False)
    print(f"[Setup] single_copy_enabled in config: {single_copy_enabled}")

    if not single_copy_enabled:
        print("[Setup] Single-copy mode not available (single_copy_enabled=False)")
        print("[Setup] Make sure vLLM was started with VLLM_ENABLE_SHARED_WEIGHTS=1")
        return None

    ipc_handles_raw = bridge_config.get("ipc_handles", {})
    print(f"[Setup] IPC handles count: {len(ipc_handles_raw)}")
    if not ipc_handles_raw:
        print("[Setup] No IPC handles found in bridge config")
        return None

    # Deserialize base64-encoded bytes back to bytes
    import base64

    def deserialize_ipc_handles(handles):
        result = {}
        for k, v in handles.items():
            if isinstance(v, dict):
                if "_bytes_b64_" in v:
                    result[k] = base64.b64decode(v["_bytes_b64_"])
                else:
                    result[k] = deserialize_ipc_handles(v)
            else:
                result[k] = v
        return result

    ipc_handles = deserialize_ipc_handles(ipc_handles_raw)

    print(f"[Setup] Attaching to vLLM's shared tensors ({len(ipc_handles)} tensors)...")
    print("[Setup] TRUE SINGLE-COPY MODE - No additional model memory!")

    # Load model config (not weights) to get architecture
    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(config.model_name)

    # Create empty model on meta device (no memory allocation)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            model_config,
            torch_dtype=torch.bfloat16,
        )

    # Get parameter names from the empty model
    param_names = list(model.state_dict().keys())
    print(f"[Setup] Model architecture has {len(param_names)} parameters", flush=True)

    # Initialize CUDA before IPC operations
    # Get the device indices we'll be using
    device_indices = set()
    for name, info in ipc_handles.items():
        if "device_index" in info:
            device_indices.add(info["device_index"])

    print(f"[Setup] IPC handles span devices: {sorted(device_indices)}", flush=True)

    # Initialize CUDA context on each device
    for dev_idx in sorted(device_indices):
        print(f"[Setup] Initializing CUDA on device {dev_idx}...", flush=True)
        torch.cuda.set_device(dev_idx)
        torch.cuda.synchronize(dev_idx)
        print(f"[Setup] ✓ Device {dev_idx} ready", flush=True)

    # Map vLLM tensor names to HuggingFace model parameter names
    hf_state_dict = {}
    vllm_to_hf_mapping = _create_vllm_to_hf_mapping(model, ipc_handles)

    attached_count = 0
    for hf_name, vllm_name in vllm_to_hf_mapping.items():
        if vllm_name not in ipc_handles:
            continue

        ipc_info = ipc_handles[vllm_name]

        try:
            # Reconstruct tensor from IPC handle
            # We need all 8 items from the original _share_cuda_() call
            if "ipc_handle_b64" not in ipc_info:
                print(f"[Setup] Missing ipc_handle_b64 for {hf_name}")
                continue

            # DEBUG: Only try first tensor to see if IPC works at all
            if attached_count == 0 and config.debug_loading:
                print(f"[Setup DEBUG] Attempting first tensor: {hf_name}", flush=True)
                print(
                    f"[Setup DEBUG] device_index: {ipc_info['device_index']}",
                    flush=True,
                )
                print(
                    f"[Setup DEBUG] storage_size: {ipc_info['storage_size']}",
                    flush=True,
                )
                print(f"[Setup DEBUG] shape: {ipc_info['shape']}", flush=True)

            # Decode all the bytes fields from base64
            device_index = ipc_info["device_index"]
            ipc_handle = base64.b64decode(ipc_info["ipc_handle_b64"])
            storage_size = ipc_info["storage_size"]
            storage_offset_orig = ipc_info["storage_offset_orig"]
            ref_counter_handle = base64.b64decode(ipc_info["ref_counter_handle_b64"])
            ref_counter_offset = ipc_info["ref_counter_offset"]
            event_handle = base64.b64decode(ipc_info["event_handle_b64"])
            event_sync_required = ipc_info["event_sync_required"]

            if attached_count == 0 and config.debug_loading:
                print(
                    f"[Setup DEBUG] Decoded IPC handle, len={len(ipc_handle)}",
                    flush=True,
                )
                print("[Setup DEBUG] About to call _new_shared_cuda...", flush=True)

            # Reconstruct the 8-tuple that _new_shared_cuda expects
            share_tuple = (
                device_index,
                ipc_handle,
                storage_size,
                storage_offset_orig,
                ref_counter_handle,
                ref_counter_offset,
                event_handle,
                event_sync_required,
            )

            # Create storage from IPC handle (needs all 8 items)
            storage = torch.UntypedStorage._new_shared_cuda(*share_tuple)

            if attached_count == 0 and config.debug_loading:
                print(
                    f"[Setup DEBUG] Storage created! size={storage.size()}", flush=True
                )

            # Reconstruct tensor
            dtype = getattr(torch, ipc_info["dtype"].replace("torch.", ""))
            tensor = torch.tensor([], dtype=dtype, device=f"cuda:{device_index}")
            tensor.set_(
                storage,
                storage_offset=ipc_info["tensor_storage_offset"],
                size=ipc_info["shape"],
                stride=ipc_info["stride"],
            )

            if attached_count == 0 and config.debug_loading:
                print(f"[Setup DEBUG] Tensor set! shape={tensor.shape}", flush=True)

            # Make tensor require gradients for training
            tensor.requires_grad_(True)

            hf_state_dict[hf_name] = tensor
            attached_count += 1

            if attached_count == 1 and config.debug_loading:
                print(
                    "[Setup DEBUG] ✓ First tensor attached successfully!", flush=True
                )

        except Exception as e:
            print(f"[Setup] Failed to attach {hf_name}: {e}", flush=True)
            import traceback

            traceback.print_exc()
            continue

    if attached_count == 0:
        print("[Setup] Could not attach any tensors, falling back to regular loading")
        return None

    print(f"[Setup] ✓ Attached {attached_count} tensors to vLLM's shared memory")

    # Load state dict into model
    model.load_state_dict(hf_state_dict, strict=False, assign=True)

    # Initialize any remaining meta tensors (buffers like rotary embeddings)
    # These are not in vLLM's state_dict but need to be initialized
    device = f"cuda:{list(device_indices)[0]}" if device_indices else "cuda:0"

    # =========================================================================
    # DIAGNOSTIC: Count what's on meta vs cuda after load_state_dict
    # =========================================================================
    meta_params = []
    cuda_params = []
    for name, param in model.named_parameters():
        if param.device.type == "meta":
            meta_params.append(name)
        elif param.device.type == "cuda":
            cuda_params.append(name)

    meta_buffers = []
    cuda_buffers = []
    for name, buffer in model.named_buffers():
        if buffer.device.type == "meta":
            meta_buffers.append(name)
        elif buffer.device.type == "cuda":
            cuda_buffers.append(name)

    if config.debug_loading:
        print("\n[DIAGNOSTIC] After load_state_dict:")
        print(f"  - Parameters on CUDA: {len(cuda_params)}")
        print(f"  - Parameters on META: {len(meta_params)}")
        print(f"  - Buffers on CUDA: {len(cuda_buffers)}")
        print(f"  - Buffers on META: {len(meta_buffers)}")

        if meta_params:
            print("\n[DIAGNOSTIC] First 10 META parameters:")
            for name in meta_params[:10]:
                param = dict(model.named_parameters())[name]
                print(
                    f"    {name}: shape={param.shape}, dtype={param.dtype}, device={param.device}"
                )

        if meta_buffers:
            print("\n[DIAGNOSTIC] META buffers:")
            for name in meta_buffers[:10]:
                buffer = dict(model.named_buffers())[name]
                print(
                    f"    {name}: shape={buffer.shape}, dtype={buffer.dtype}, device={buffer.device}"
                )

    # =========================================================================
    # Helper function to navigate module hierarchy
    # =========================================================================
    def get_parent_and_name(model, full_name):
        """Get parent module and attribute name from full parameter name."""
        parts = full_name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent, parts[-1]

    # =========================================================================
    # Initialize remaining meta parameters
    # NOTE: Can't use param.data = ... on meta tensors!
    # Must use setattr() to replace the entire parameter in the parent module
    # =========================================================================
    meta_count = 0

    for name in meta_params:
        param = dict(model.named_parameters()).get(name)
        if param is None:
            continue

        try:
            if config.debug_loading:
                print(f"[DIAGNOSTIC] Initializing meta param: {name}")
                print(
                    f"  - Old: device={param.device}, dtype={param.dtype}, shape={param.shape}"
                )

            # Create new parameter with actual data on CUDA
            new_data = torch.zeros(param.shape, dtype=param.dtype, device=device)
            new_param = torch.nn.Parameter(new_data, requires_grad=param.requires_grad)

            if config.debug_loading:
                print(
                    f"  - New: device={new_param.device}, dtype={new_param.dtype}, shape={new_param.shape}"
                )

            # Replace in parent module using setattr (NOT param.data = ...)
            parent, attr_name = get_parent_and_name(model, name)
            if config.debug_loading:
                print(f"  - Parent module: {type(parent).__name__}, attr: {attr_name}")

            setattr(parent, attr_name, new_param)
            meta_count += 1
            if config.debug_loading:
                print("  - ✓ Replaced successfully!")

        except Exception as e:
            if config.debug_loading:
                print(f"[DIAGNOSTIC] FAILED to initialize {name}: {e}")
            import traceback

            traceback.print_exc()

    # =========================================================================
    # Initialize remaining meta buffers
    # =========================================================================
    for name in meta_buffers:
        buffer = dict(model.named_buffers()).get(name)
        if buffer is None:
            continue

        try:
            if config.debug_loading:
                print(f"[DIAGNOSTIC] Initializing meta buffer: {name}")
                print(
                    f"  - Old: device={buffer.device}, dtype={buffer.dtype}, shape={buffer.shape}"
                )

            # For buffers like inv_freq, we need proper initialization
            if "inv_freq" in name:
                # Rotary embedding inverse frequencies
                dim = buffer.shape[0] * 2  # inv_freq has shape [dim/2]
                base = 10000.0  # Default RoPE base
                inv_freq = 1.0 / (
                    base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
                )
                new_buffer = inv_freq.to(dtype=buffer.dtype, device=device)
                if config.debug_loading:
                    print(f"  - Computed inv_freq with dim={dim}, base={base}")
            else:
                # Other buffers - initialize with zeros
                new_buffer = torch.zeros(
                    buffer.shape, dtype=buffer.dtype, device=device
                )

            if config.debug_loading:
                print(
                    f"  - New: device={new_buffer.device}, dtype={new_buffer.dtype}, shape={new_buffer.shape}"
                )

            # Replace in parent module
            parent, attr_name = get_parent_and_name(model, name)
            if config.debug_loading:
                print(f"  - Parent module: {type(parent).__name__}, attr: {attr_name}")

            parent.register_buffer(attr_name, new_buffer)
            meta_count += 1
            if config.debug_loading:
                print("  - ✓ Replaced successfully!")

        except Exception as e:
            if config.debug_loading:
                print(f"[DIAGNOSTIC] FAILED to initialize buffer {name}: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n[Setup] Initialized {meta_count} remaining meta tensors")

    return model


def _create_vllm_to_hf_mapping(model: torch.nn.Module, ipc_handles: dict) -> dict:
    """
    Create mapping from HuggingFace parameter names to vLLM tensor names.

    vLLM uses slightly different naming conventions than HuggingFace.
    This function creates the bidirectional mapping.
    """
    hf_params = set(model.state_dict().keys())
    vllm_params = set(ipc_handles.keys())

    mapping = {}

    for hf_name in hf_params:
        # Try direct match first
        if hf_name in vllm_params:
            mapping[hf_name] = hf_name
            continue

        # Try common transformations
        # vLLM often uses 'model.' prefix
        vllm_name = f"model.{hf_name}" if not hf_name.startswith("model.") else hf_name
        if vllm_name in vllm_params:
            mapping[hf_name] = vllm_name
            continue

        # Remove 'model.' prefix if present
        if hf_name.startswith("model."):
            vllm_name = hf_name[6:]
            if vllm_name in vllm_params:
                mapping[hf_name] = vllm_name

    return mapping


def load_model_and_tokenizer(
    config: TrainingConfig,
    single_copy: bool = False,
) -> Tuple[torch.nn.Module, "AutoTokenizer"]:
    """
    Load or attach to model based on weight_bridge_mode.

    Args:
        config: Training configuration
        single_copy: If True, attach to vLLM's shared tensors via CUDA IPC

    Returns:
        Tuple of (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Single-copy mode: attach to vLLM's shared tensors via CUDA IPC
    if single_copy or config.weight_bridge_mode == "shared_vllm":
        # Check for explicit path first
        if config.vllm_config_path and os.path.exists(config.vllm_config_path):
            config_path = config.vllm_config_path
            print(f"[Setup] Using explicit vLLM config path: {config_path}")
        else:
            # Auto-detect from common locations
            possible_paths = [
                os.environ.get("LOGDIR", "."),
                ".",
                "/tmp/atropos_bridge",
                os.path.dirname(os.path.abspath(__file__)),
            ]

            config_path = None
            for log_dir in possible_paths:
                candidate = os.path.join(log_dir, "vllm_bridge_config.json")
                if os.path.exists(candidate):
                    config_path = candidate
                    print(f"[Setup] Found vLLM config at: {candidate}")
                    break

            if config_path is None:
                checked = [
                    os.path.join(p, "vllm_bridge_config.json") for p in possible_paths
                ]
                raise RuntimeError(
                    f"[Setup] Could not find vllm_bridge_config.json\n"
                    f"Checked: {checked}\n"
                    f"Tip: Use --vllm-config-path to specify the path explicitly\n"
                    f"Make sure vLLM is running with VLLM_ENABLE_SHARED_WEIGHTS=1 and LOGDIR set"
                )

        model = _attach_to_vllm_shared_tensors(config, config_path)
        if model is not None:
            print("[Setup] ✓ Single-copy mode active - using vLLM's tensors directly!")
            model.train()
            return model, tokenizer
        else:
            raise RuntimeError(
                "[Setup] Single-copy mode FAILED to attach to vLLM's tensors.\n"
                "Check:\n"
                "  1. vLLM running with VLLM_ENABLE_SHARED_WEIGHTS=1\n"
                "  2. vllm_bridge_config.json exists with ipc_handles\n"
                "  3. Trainer is on SAME GPUs as vLLM"
            )

    elif config.weight_bridge_mode == "lora_only":
        model = _load_model_with_lora(config)

    else:
        print("[Setup] Loading model for legacy mode...")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name, torch_dtype=torch.bfloat16
        )
        model.to(config.device)

    # Enable gradient checkpointing (saves memory)
    # For LoRA, use PEFT's method; for others, use standard method
    if config.weight_bridge_mode == "lora_only":
        # PEFT models need gradient_checkpointing enabled on base model
        # and require use_reentrant=False for proper gradient flow
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    else:
        # Standard gradient checkpointing
        model.gradient_checkpointing_enable()

    model.train()

    return model, tokenizer


def _load_model_with_lora(config: TrainingConfig) -> torch.nn.Module:
    """
    Load base model and wrap with LoRA adapters.

    Args:
        config: Training configuration with LoRA settings

    Returns:
        PEFT model with LoRA adapters applied
    """
    if not PEFT_AVAILABLE:
        raise RuntimeError("PEFT library not available. Install with: pip install peft")

    print("[Setup] Loading base model for LoRA mode...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name, torch_dtype=torch.bfloat16
    )
    base_model.to(config.device)

    # Determine target modules
    target_modules = config.lora_target_modules
    if target_modules is None:
        # Default modules for most transformer models
        target_modules = ["q_proj", "v_proj"]

    print(f"Applying LoRA: r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"Target modules: {target_modules}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    return model


def save_lora_checkpoint(
    model: torch.nn.Module,
    save_path: str,
    step: int,
    is_final: bool = False,
) -> str:
    """
    Save LoRA adapter checkpoint.

    Args:
        model: PEFT model with LoRA adapters
        save_path: Base directory for checkpoints
        step: Current training step
        is_final: Whether this is the final checkpoint

    Returns:
        Path where adapter was saved
    """
    if is_final:
        adapter_path = os.path.join(save_path, "final_adapter")
    else:
        adapter_path = os.path.join(save_path, f"adapter_step_{step}")

    print(f"  Saving LoRA adapter to {adapter_path}...")

    if os.path.exists(adapter_path):
        shutil.rmtree(adapter_path)
    os.makedirs(adapter_path, exist_ok=True)

    # Save only the adapter weights (much smaller than full model)
    model.save_pretrained(adapter_path)

    print("  Adapter saved.")
    return adapter_path


def compute_grpo_loss(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    advantages: torch.Tensor,
    temperatures: torch.Tensor,
    gradient_accumulation_steps: int,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute GRPO loss for a single micro-batch.

    Args:
        model: The model to compute loss for
        tokens: Input token IDs [batch, seq_len]
        labels: Target labels [batch, seq_len]
        advantages: Advantage values [batch, 1]
        temperatures: Temperature values [batch, 1, 1]
        gradient_accumulation_steps: Number of accumulation steps

    Returns:
        Tuple of (loss tensor, metrics dict)
    """
    # Forward pass
    outputs = model(tokens)
    logits = outputs.logits

    # Temperature scaling
    t = temperatures.to(logits.device, logits.dtype)
    t = torch.where(t <= 0, torch.ones_like(t), t)
    logits = logits / t

    # Log probabilities per token
    logp_per_token = -F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(labels.shape)

    # Masking based on labels != -100
    mask = (labels != -100).float()

    # Compute metrics (no grad needed)
    with torch.no_grad():
        pos = (advantages > 0).float()
        neg = (advantages <= 0).float()
        mask_float = mask.to(logp_per_token.dtype)
        mask_sum = mask_float.sum(dim=-1).clamp_min(1e-8)
        avg_logp = (logp_per_token * mask_float).sum(dim=-1) / mask_sum
        pos_logp = (logp_per_token * pos).mean().item()
        neg_logp = (logp_per_token * neg).mean().item()

    # GRPO loss
    grpo_loss_term = torch.exp(logp_per_token - logp_per_token.detach())
    grpo_loss = (
        ((-grpo_loss_term * mask).sum(-1) / mask.sum(-1))
        * advantages.to(logp_per_token.device)
    ).mean() / gradient_accumulation_steps

    metrics = {
        "pos_logp": pos_logp,
        "neg_logp": neg_logp,
        "avg_logp": avg_logp,
        "pos_count": pos.sum().item(),
        "neg_count": neg.sum().item(),
    }

    return grpo_loss, metrics


def run_training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    token_batches: List[torch.Tensor],
    label_batches: List[torch.Tensor],
    advantage_batches: List[torch.Tensor],
    temperature_batches: List[torch.Tensor],
    config: TrainingConfig,
) -> dict:
    """
    Run a single training step (forward, backward, optimizer step).

    Args:
        model: The model to train
        optimizer: The optimizer
        token_batches: List of token tensors
        label_batches: List of label tensors
        advantage_batches: List of advantage tensors
        temperature_batches: List of temperature tensors
        config: Training configuration

    Returns:
        Dict of training metrics for this step
    """
    total_loss = 0.0
    total_pos_logp = 0.0
    total_neg_logp = 0.0
    total_pos = 0.0
    total_neg = 0.0

    # Accumulate gradients over micro-batches
    for tokens, labels, advantages, temperatures in zip(
        token_batches, label_batches, advantage_batches, temperature_batches
    ):
        tokens = tokens.to(config.device)
        labels = labels.to(config.device)
        advantages = advantages.to(config.device)

        loss, metrics = compute_grpo_loss(
            model,
            tokens,
            labels,
            advantages,
            temperatures,
            config.gradient_accumulation_steps,
        )

        loss.backward()
        total_loss += loss.item()
        total_pos_logp += metrics["pos_logp"]
        total_neg_logp += metrics["neg_logp"]
        total_pos += metrics["pos_count"]
        total_neg += metrics["neg_count"]

        # Gradient clipping and optimizer step
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        # Normalize metrics
        if total_pos > 0:
            total_pos_logp /= total_pos
        if total_neg > 0:
            total_neg_logp /= total_neg

    return {
        "loss": total_loss,
        "grad_norm": grad_norm.item(),
        "pos_logp": total_pos_logp,
        "neg_logp": total_neg_logp,
    }


def save_checkpoint(
    model: torch.nn.Module,
    tokenizer: "AutoTokenizer",
    save_path: str,
    step: int,
    is_final: bool = False,
) -> str:
    """
    Save model checkpoint.

    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        save_path: Base directory for checkpoints
        step: Current training step
        is_final: Whether this is the final checkpoint

    Returns:
        Path where checkpoint was saved
    """
    if is_final:
        checkpoint_path = os.path.join(save_path, "final_model")
    else:
        checkpoint_path = os.path.join(save_path, f"step_{step}")

    print(f"  Saving checkpoint to {checkpoint_path}...")

    if os.path.exists(checkpoint_path):
        shutil.rmtree(checkpoint_path)
    os.makedirs(checkpoint_path, exist_ok=True)

    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)

    print("  Checkpoint saved.")
    return checkpoint_path


def log_metrics(
    metrics: dict,
    step: int,
    use_wandb: bool,
    extra_metrics: Optional[dict] = None,
    benchmark: bool = False,
) -> None:
    """
    Log training metrics to console and optionally wandb.

    Args:
        metrics: Dict of metrics from training step
        step: Current step number
        use_wandb: Whether to log to wandb
        extra_metrics: Optional additional metrics to log
        benchmark: Whether to show timing/benchmark info
    """
    # Console output with timing info (only if benchmark enabled)
    timing_str = ""
    if benchmark:
        if "step_time" in metrics:
            timing_str += f", Step time: {metrics['step_time']:.2f}s"
        if "sync_time" in metrics and metrics["sync_time"] > 0:
            timing_str += f", Sync time: {metrics['sync_time']:.2f}s"
        if "data_fetch_time" in metrics:
            timing_str += f", Data fetch: {metrics['data_fetch_time']:.2f}s"
        if "gpu_memory_gb" in metrics:
            timing_str += f", GPU mem: {metrics['gpu_memory_gb']:.2f}GB"

    # Show loss with more precision since GRPO loss is often very small
    loss_str = (
        f"{metrics['loss']:.6f}"
        if abs(metrics["loss"]) < 0.01
        else f"{metrics['loss']:.4f}"
    )
    print(f"  Loss: {loss_str}, Grad norm: {metrics['grad_norm']:.4f}{timing_str}")

    # Show GRPO-specific metrics if available
    if "pos_count" in metrics or "neg_count" in metrics:
        pos_count = metrics.get("pos_count", 0)
        neg_count = metrics.get("neg_count", 0)
        pos_logp = metrics.get("pos_logp", 0)
        neg_logp = metrics.get("neg_logp", 0)
        print(
            f"    Advantages: +{int(pos_count)} / -{int(neg_count)}, LogP: pos={pos_logp:.3f}, neg={neg_logp:.3f}"
        )

    if use_wandb:
        log_dict = {
            "train/loss": metrics["loss"],
            "train/grad_norm": metrics["grad_norm"],
            "train/pos_logp": metrics["pos_logp"],
            "train/neg_logp": metrics["neg_logp"],
        }
        # Add timing metrics if present
        if "step_time" in metrics:
            log_dict["train/step_time"] = metrics["step_time"]
        if "sync_time" in metrics:
            log_dict["train/sync_time"] = metrics["sync_time"]
        if "data_fetch_time" in metrics:
            log_dict["train/data_fetch_time"] = metrics["data_fetch_time"]
        if "gpu_memory_gb" in metrics:
            log_dict["train/gpu_memory_gb"] = metrics["gpu_memory_gb"]
        if "gpu_memory_reserved_gb" in metrics:
            log_dict["train/gpu_memory_reserved_gb"] = metrics["gpu_memory_reserved_gb"]
        if extra_metrics:
            log_dict.update(extra_metrics)
        wandb.log(log_dict, step=step)


def finalize_training(
    use_wandb: bool,
    training_start_time: Optional[float] = None,
    mode: str = "unknown",
    total_steps: int = 0,
    benchmark_stats: Optional[dict] = None,
    benchmark: bool = False,
) -> None:
    """Clean up after training and log benchmark summary.

    Args:
        use_wandb: Whether wandb is enabled
        training_start_time: Start time of training
        mode: Training mode name
        total_steps: Total steps completed
        benchmark_stats: Dict with lists of per-step metrics:
            - step_times: List of step durations
            - sync_times: List of sync durations
            - data_fetch_times: List of data fetch durations
            - gpu_memories: List of GPU memory readings (GB)
        benchmark: Whether to print benchmark summary to console
    """
    print("\nTraining finished.")

    # Default empty stats
    if benchmark_stats is None:
        benchmark_stats = {}

    # Log benchmark summary
    if training_start_time is not None:
        total_time = time.time() - training_start_time
        peak_gpu_mem_gb = (
            torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        )

        # Calculate averages from collected stats
        step_times = benchmark_stats.get("step_times", [])
        sync_times = benchmark_stats.get("sync_times", [])
        data_fetch_times = benchmark_stats.get("data_fetch_times", [])
        gpu_memories = benchmark_stats.get("gpu_memories", [])

        avg_step_time = sum(step_times) / len(step_times) if step_times else 0
        total_step_time = sum(step_times)
        avg_sync_time = sum(sync_times) / len(sync_times) if sync_times else 0
        total_sync_time = sum(sync_times)
        avg_data_fetch = (
            sum(data_fetch_times) / len(data_fetch_times) if data_fetch_times else 0
        )
        total_data_fetch = sum(data_fetch_times)
        avg_gpu_mem = sum(gpu_memories) / len(gpu_memories) if gpu_memories else 0

        # Print benchmark summary only if benchmark flag is enabled
        if benchmark:
            print(f"\n{'='*70}")
            print(f"BENCHMARK SUMMARY ({mode})")
            print(f"{'='*70}")
            print(
                f"  Total training time:     {total_time:.2f}s ({total_time/60:.2f} min)"
            )
            print(f"  Total steps:             {total_steps}")
            print("  ")
            print("  TIMING BREAKDOWN:")
            print(f"    Avg step time:         {avg_step_time:.2f}s")
            print(f"    Total step time:       {total_step_time:.2f}s")
            print(
                f"    Avg sync time:         {avg_sync_time:.2f}s (x{len(sync_times)} syncs)"
            )
            print(f"    Total sync time:       {total_sync_time:.2f}s")
            print(f"    Avg data fetch time:   {avg_data_fetch:.2f}s")
            print(f"    Total data fetch time: {total_data_fetch:.2f}s")
            print("  ")
            print("  MEMORY:")
            print(f"    Peak GPU memory:       {peak_gpu_mem_gb:.2f} GB")
            print(f"    Avg GPU memory:        {avg_gpu_mem:.2f} GB")
            print(f"{'='*70}\n")

        if use_wandb:
            # Total time metrics
            wandb.summary["benchmark/total_time_seconds"] = total_time
            wandb.summary["benchmark/total_time_minutes"] = total_time / 60
            wandb.summary["benchmark/mode"] = mode
            wandb.summary["benchmark/total_steps"] = total_steps

            # Step timing metrics
            wandb.summary["benchmark/avg_step_time_seconds"] = avg_step_time
            wandb.summary["benchmark/total_step_time_seconds"] = total_step_time

            # Sync timing metrics
            wandb.summary["benchmark/avg_sync_time_seconds"] = avg_sync_time
            wandb.summary["benchmark/total_sync_time_seconds"] = total_sync_time
            wandb.summary["benchmark/num_syncs"] = len(sync_times)

            # Data fetch timing metrics
            wandb.summary["benchmark/avg_data_fetch_time_seconds"] = avg_data_fetch
            wandb.summary["benchmark/total_data_fetch_time_seconds"] = total_data_fetch

            # Memory metrics
            wandb.summary["benchmark/peak_gpu_memory_gb"] = peak_gpu_mem_gb
            wandb.summary["benchmark/avg_gpu_memory_gb"] = avg_gpu_mem

    if use_wandb:
        wandb.finish()


def train(config: TrainingConfig):
    """
    Legacy GRPO training with periodic vLLM restarts.

    This mode saves checkpoints to disk and restarts vLLM to pick up new weights.
    Use weight_bridge_mode='shared_vllm' for in-place weight updates without restarts.
    """
    global vllm_process
    training_start_time = time.time()

    # === Setup ===
    use_wandb = setup_wandb(config)
    model, tokenizer = load_model_and_tokenizer(config)
    optimizer = AdamW(model.parameters(), lr=config.lr)

    print(f"\n{'='*60}")
    print("LEGACY MODE (checkpoint + vLLM restart)")
    print(f"{'='*60}")
    print(f"Training for {config.training_steps} steps on {config.device}")
    print(f"vLLM restart interval: every {config.vllm_restart_interval} steps")
    print(f"Save path: {config.save_path}")
    print(f"{'='*60}\n")

    os.makedirs(config.save_path, exist_ok=True)
    register_trainer(config)

    # Launch initial vLLM server
    vllm_process = _launch_vllm_server(config, config.model_name)

    # === Benchmark tracking ===
    benchmark_stats = {
        "step_times": [],
        "sync_times": [],
        "data_fetch_times": [],
        "gpu_memories": [],
    }

    # === Training Loop ===
    batches = []
    for step in range(config.training_steps):
        print(f"\nStep {step+1}/{config.training_steps}")

        # Track data fetch time
        data_fetch_start = time.time()
        if len(batches) == 0:
            batches = get_data(config.batch_size, config.seq_len)
        token_batches, label_batches, advantage_batches, temperature_batches = (
            batches.pop(0)
        )
        data_fetch_time = time.time() - data_fetch_start
        benchmark_stats["data_fetch_times"].append(data_fetch_time)

        # Terminate vLLM before training step (to free GPU memory)
        should_sync = (
            step + 1
        ) % config.vllm_restart_interval == 0 or step == config.training_steps - 1
        if should_sync:
            _terminate_vllm_process()

        # Track step time
        step_start = time.time()

        # Run training step using common helper
        metrics = run_training_step(
            model,
            optimizer,
            token_batches,
            label_batches,
            advantage_batches,
            temperature_batches,
            config,
        )

        step_time = time.time() - step_start
        benchmark_stats["step_times"].append(step_time)

        # Track GPU memory
        if torch.cuda.is_available():
            gpu_mem_gb = torch.cuda.memory_allocated() / 1e9
            gpu_mem_reserved_gb = torch.cuda.memory_reserved() / 1e9
            benchmark_stats["gpu_memories"].append(gpu_mem_gb)
        else:
            gpu_mem_gb = 0
            gpu_mem_reserved_gb = 0

        # Track sync time
        sync_time = 0
        if should_sync:
            sync_start = time.time()
            checkpoint_path = save_checkpoint(
                model, tokenizer, config.save_path, step + 1
            )
            torch.cuda.empty_cache()
            vllm_process = _launch_vllm_server(config, checkpoint_path)
            sync_time = time.time() - sync_start
            benchmark_stats["sync_times"].append(sync_time)

        # Add timing metrics
        metrics["step_time"] = step_time
        metrics["sync_time"] = sync_time
        metrics["data_fetch_time"] = data_fetch_time
        metrics["gpu_memory_gb"] = gpu_mem_gb
        metrics["gpu_memory_reserved_gb"] = gpu_mem_reserved_gb

        # Log metrics
        log_metrics(
            metrics,
            step + 1,
            use_wandb,
            {
                "train/learning_rate": optimizer.param_groups[0]["lr"],
            },
            benchmark=config.benchmark,
        )

        # Check for unexpected vLLM termination
        _check_vllm_process_health()

    # === Cleanup ===
    save_checkpoint(
        model, tokenizer, config.save_path, config.training_steps, is_final=True
    )
    finalize_training(
        use_wandb,
        training_start_time,
        "legacy",
        config.training_steps,
        benchmark_stats,
        benchmark=config.benchmark,
    )


# =============================================================================
# vLLM Process Management (Legacy Mode Only)
# =============================================================================


def _launch_vllm_server(
    config: TrainingConfig, model_path: str
) -> Optional[subprocess.Popen]:
    """Launch a vLLM server process using our custom vllm_api_server.py.

    Uses the custom server instead of standard vLLM because:
    - Standard vLLM only has /v1/completions (OpenAI-compatible)
    - Our custom server has /generate endpoint needed by VLLMServer class
    - This allows proper tokens_and_logprobs_completion support
    """
    vllm_process

    # Use our custom vllm_api_server.py instead of standard vLLM
    # This provides the /generate endpoint that VLLMServer needs
    script_dir = os.path.dirname(os.path.abspath(__file__))
    custom_server_path = os.path.join(script_dir, "vllm_api_server.py")

    vllm_command = [
        "python",
        custom_server_path,
        "--model",
        model_path,
        "--port",
        str(config.vllm_port),
        "--gpu-memory-utilization",
        str(config.vllm_gpu_memory_utilization),
    ]
    # Add served-model-name if using checkpoint path
    if model_path != config.model_name:
        vllm_command.extend(["--served-model-name", config.model_name])

    print(f"  Launching vLLM: {' '.join(vllm_command)}")

    try:
        proc = subprocess.Popen(vllm_command)
        print(f"  vLLM launched with PID: {proc.pid}")

        # Check for immediate startup errors
        try:
            proc.communicate(timeout=2)
            if proc.returncode is not None and proc.returncode != 0:
                print("  WARNING: vLLM failed to start")
                return None
        except subprocess.TimeoutExpired:
            print("  vLLM process started (check logs for details)")

        return proc

    except FileNotFoundError:
        print("  ERROR: vLLM not found. Is it installed?")
        return None
    except Exception as e:
        print(f"  ERROR launching vLLM: {e}")
        return None


def _terminate_vllm_process() -> None:
    """Terminate the running vLLM process if any."""
    global vllm_process

    if vllm_process is None:
        return

    print("  Terminating vLLM process...")
    vllm_process.terminate()
    try:
        vllm_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        print("  vLLM did not terminate gracefully, killing...")
        vllm_process.kill()
        vllm_process.wait()
    vllm_process = None


def _check_vllm_process_health() -> None:
    """Check if vLLM process terminated unexpectedly (legacy mode)."""
    global vllm_process

    if vllm_process is not None and vllm_process.poll() is not None:
        print(
            f"  WARNING: vLLM terminated unexpectedly (code: {vllm_process.returncode})"
        )
        vllm_process = None


def train_shared_vllm(config: TrainingConfig):
    """
    GRPO training with shared vLLM weights.

    Instead of saving checkpoints and restarting vLLM, this mode:
    1. Joins the same distributed group as vLLM
    2. Attaches to vLLM's weight tensors directly via CUDA IPC
    3. optimizer.step() modifies vLLM's weights in-place
    4. vLLM immediately uses updated weights (no restart!)

    Requirements:
    - vLLM running with VLLM_ENABLE_SHARED_WEIGHTS=1
    - Trainer on same GPU(s) as vLLM (for IPC to work)
    """
    training_start_time = time.time()

    # === Setup ===
    use_wandb = setup_wandb(config)

    print(f"\n{'='*60}")
    print("SINGLE-COPY MODE (CUDA IPC)")
    print(">>> TRUE shared memory - only ONE model copy!")
    print(">>> Trainer uses vLLM's tensors directly!")
    print(f"{'='*60}")
    print(f"Model: {config.model_name}")
    print(f"Distributed: rank={config.trainer_rank}/{config.world_size}")
    print(f"Init method: {config.init_method}")
    print(f"Save path: {config.save_path}")
    print(f"{'='*60}\n")

    # Single-copy mode: attach directly to vLLM's tensors via CUDA IPC
    print("[1/2] Attaching to vLLM's shared tensors...")
    model, tokenizer = load_model_and_tokenizer(config, single_copy=True)

    if model is None:
        raise RuntimeError(
            "Single-copy mode failed. Make sure:\n"
            "1. vLLM is running with VLLM_ENABLE_SHARED_WEIGHTS=1\n"
            "2. Trainer is on the SAME GPUs as vLLM\n"
            "3. vllm_bridge_config.json exists with IPC handles"
        )

    optimizer = AdamW(
        model.parameters(), lr=config.lr
    )  # maybe we need to make this configurable in the future

    print(f"[2/2] Starting training for {config.training_steps} steps")
    print("NOTE: vLLM sees weight updates immediately after each step!")
    print("-" * 60)

    os.makedirs(config.save_path, exist_ok=True)

    # Check Atropos API and register BEFORE training loop
    print("\n[Setup] Connecting to Atropos API...")
    if not check_atropos_api(timeout=30):
        raise RuntimeError(
            "Atropos API server not reachable. " "Please start it with: run-api"
        )
    register_trainer(config)

    # === Benchmark tracking ===
    benchmark_stats = {
        "step_times": [],
        "sync_times": [],  # For shared mode, this is the notify_update time
        "data_fetch_times": [],
        "gpu_memories": [],
    }

    # === Training Loop ===
    batches = []
    for step in range(config.training_steps):
        print(f"\nStep {step+1}/{config.training_steps}")

        # Track data fetch time
        data_fetch_start = time.time()
        if len(batches) == 0:
            batches = get_data(config.batch_size, config.seq_len)
        token_batches, label_batches, advantage_batches, temperature_batches = (
            batches.pop(0)
        )
        data_fetch_time = time.time() - data_fetch_start
        benchmark_stats["data_fetch_times"].append(data_fetch_time)

        # Track step time
        step_start = time.time()

        # Run training step using common helper
        metrics = run_training_step(
            model,
            optimizer,
            token_batches,
            label_batches,
            advantage_batches,
            temperature_batches,
            config,
        )

        step_time = time.time() - step_start
        benchmark_stats["step_times"].append(step_time)

        # Track GPU memory
        if torch.cuda.is_available():
            gpu_mem_gb = torch.cuda.memory_allocated() / 1e9
            gpu_mem_reserved_gb = torch.cuda.memory_reserved() / 1e9
            benchmark_stats["gpu_memories"].append(gpu_mem_gb)
        else:
            gpu_mem_gb = 0
            gpu_mem_reserved_gb = 0

        # In single-copy mode, weights are already updated in-place (same GPU memory!)
        # No synchronization needed - vLLM sees changes immediately
        sync_time = 0.0
        print(f"  [SINGLE-COPY] Weights updated in-place - step {step+1}")
        benchmark_stats["sync_times"].append(sync_time)

        # Add timing metrics
        metrics["step_time"] = step_time
        metrics["sync_time"] = sync_time
        metrics["data_fetch_time"] = data_fetch_time
        metrics["gpu_memory_gb"] = gpu_mem_gb
        metrics["gpu_memory_reserved_gb"] = gpu_mem_reserved_gb

        # Log metrics
        log_metrics(
            metrics,
            step + 1,
            use_wandb,
            {
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "train/update_count": step + 1,
            },
            benchmark=config.benchmark,
        )

        # Periodic checkpoint save (for recovery, not for vLLM sync)
        if (step + 1) % config.vllm_restart_interval == 0:
            save_checkpoint(model, tokenizer, config.save_path, step + 1)

    # === Cleanup ===
    save_checkpoint(
        model, tokenizer, config.save_path, config.training_steps, is_final=True
    )
    finalize_training(
        use_wandb,
        training_start_time,
        "shared_vllm",
        config.training_steps,
        benchmark_stats,
        benchmark=config.benchmark,
    )


def _check_vllm_health(port: int) -> bool:
    """Check if external vLLM server is running and healthy."""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def _hotswap_lora_adapter(port: int, adapter_path: str) -> bool:
    """
    Request vLLM to hot-swap to a new LoRA adapter.

    Args:
        port: vLLM server port
        adapter_path: Path to the saved adapter directory

    Returns:
        True if successful, False otherwise
    """
    try:
        response = requests.post(
            f"http://localhost:{port}/lora/load",
            json={"adapter_path": adapter_path},
            timeout=30,
        )
        if response.status_code == 200:
            print(f"  [LORA] Hot-swapped adapter: {adapter_path}")
            return True
        else:
            print(f"  [LORA] Hot-swap failed: {response.text}")
            return False
    except Exception as e:
        print(f"  [LORA] Hot-swap request failed: {e}")
        return False


def train_lora(config: TrainingConfig):
    """
    GRPO training with LoRA adapters.

    This mode keeps the base model frozen and only trains LoRA adapter weights.

    REQUIRES: External vLLM server running via vllm_api_server.py

    Benefits:
    - Much faster training (fewer parameters)
    - Smaller checkpoint sizes (adapter only, not full model)
    - Adapters can be hot-swapped in vLLM via /lora/load endpoint
    """
    if not PEFT_AVAILABLE:
        raise RuntimeError(
            "PEFT library required for LoRA mode. Install with: pip install peft"
        )

    training_start_time = time.time()

    # === Setup ===
    use_wandb = setup_wandb(config)

    print(f"\n{'='*60}")
    print("LORA MODE (adapter-only training)")
    print(f"{'='*60}")
    print(f"Base model: {config.model_name}")
    print(f"LoRA config: r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"Save path: {config.save_path}")
    print(f"vLLM port: {config.vllm_port}")
    print(f"{'='*60}\n")

    # Check that external vLLM is running
    print("[1/3] Checking external vLLM server...")
    if not _check_vllm_health(config.vllm_port):
        print(f"\nERROR: vLLM server not running on port {config.vllm_port}")
        print("\nLoRA mode requires an external vLLM server. Start it first:")
        print("  python example_trainer/vllm_api_server.py \\")
        print(f"    --model {config.model_name} \\")
        print(f"    --port {config.vllm_port} \\")
        print("    --gpu-memory-utilization 0.45")
        raise RuntimeError(f"External vLLM server required on port {config.vllm_port}")
    print(f"vLLM server healthy on port {config.vllm_port}")

    # Load model with LoRA adapters
    print("[2/3] Loading model with LoRA adapters...")
    model, tokenizer = load_model_and_tokenizer(config)

    # Only optimize LoRA parameters (base model is frozen)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=config.lr)

    print(f"[3/3] Starting training for {config.training_steps} steps")
    print("-" * 60)

    os.makedirs(config.save_path, exist_ok=True)
    register_trainer(config)

    # NOTE: No vLLM launch here - using external vLLM server

    # === Benchmark tracking ===
    benchmark_stats = {
        "step_times": [],
        "sync_times": [],  # For LoRA mode, this is adapter save + hot-swap time
        "data_fetch_times": [],
        "gpu_memories": [],
    }

    # === Training Loop ===
    batches = []
    for step in range(config.training_steps):
        print(f"\nStep {step+1}/{config.training_steps}")

        # Track data fetch time
        data_fetch_start = time.time()
        if len(batches) == 0:
            batches = get_data(config.batch_size, config.seq_len)
        token_batches, label_batches, advantage_batches, temperature_batches = (
            batches.pop(0)
        )
        data_fetch_time = time.time() - data_fetch_start
        benchmark_stats["data_fetch_times"].append(data_fetch_time)

        # Track step time
        step_start = time.time()

        # Run training step
        metrics = run_training_step(
            model,
            optimizer,
            token_batches,
            label_batches,
            advantage_batches,
            temperature_batches,
            config,
        )

        step_time = time.time() - step_start
        benchmark_stats["step_times"].append(step_time)

        # Track GPU memory
        if torch.cuda.is_available():
            gpu_mem_gb = torch.cuda.memory_allocated() / 1e9
            gpu_mem_reserved_gb = torch.cuda.memory_reserved() / 1e9
            benchmark_stats["gpu_memories"].append(gpu_mem_gb)
        else:
            gpu_mem_gb = 0
            gpu_mem_reserved_gb = 0

        # Track sync time (adapter save + hot-swap)
        sync_time = 0
        should_sync = (step + 1) % config.vllm_restart_interval == 0
        if should_sync:
            sync_start = time.time()
            adapter_path = save_lora_checkpoint(model, config.save_path, step + 1)
            # Try to hot-swap the adapter in vLLM (non-blocking, best effort)
            _hotswap_lora_adapter(config.vllm_port, adapter_path)
            sync_time = time.time() - sync_start
            benchmark_stats["sync_times"].append(sync_time)

        # Add timing metrics
        metrics["step_time"] = step_time
        metrics["sync_time"] = sync_time
        metrics["data_fetch_time"] = data_fetch_time
        metrics["gpu_memory_gb"] = gpu_mem_gb
        metrics["gpu_memory_reserved_gb"] = gpu_mem_reserved_gb

        # Log metrics
        log_metrics(
            metrics,
            step + 1,
            use_wandb,
            {
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "lora/trainable_params": sum(p.numel() for p in trainable_params),
            },
            benchmark=config.benchmark,
        )

    # === Cleanup ===
    # NOTE: No vLLM termination - external server keeps running

    # Save final adapter (track this sync time too)
    final_sync_start = time.time()
    final_adapter_path = save_lora_checkpoint(
        model, config.save_path, config.training_steps, is_final=True
    )

    # Hot-swap to final adapter
    _hotswap_lora_adapter(config.vllm_port, final_adapter_path)
    final_sync_time = time.time() - final_sync_start
    benchmark_stats["sync_times"].append(final_sync_time)

    finalize_training(
        use_wandb,
        training_start_time,
        "lora_only",
        config.training_steps,
        benchmark_stats,
        benchmark=config.benchmark,
    )

    # Also save tokenizer for convenience
    tokenizer_path = os.path.join(config.save_path, "tokenizer")
    tokenizer.save_pretrained(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the GRPO trainer."""
    parser = argparse.ArgumentParser(
        description="GRPO Trainer with optional shared-weight vLLM integration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Core training arguments ---
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

    # --- vLLM arguments ---
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
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.45,
        help="GPU memory utilization for vLLM server (0.0-1.0)",
    )

    # --- Wandb arguments ---
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

    # --- Pipeline / weight bridge arguments ---
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

    # --- LoRA arguments ---
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

    # --- Debug flags ---
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
    """Build a TrainingConfig from parsed CLI arguments."""
    return TrainingConfig(
        model_name=args.model_name,
        lr=args.lr,
        training_steps=args.training_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        device=args.device,
        save_path=args.save_path,
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
    )


# Example usage (optional, can be run from another script)
if __name__ == "__main__":
    args = parse_args()
    training_config = config_from_args(args)

    print(f"Weight bridge mode: {training_config.weight_bridge_mode}")

    if training_config.weight_bridge_mode == "shared_vllm":
        # Shared vLLM mode: attach to vLLM's weights, update in-place
        train_shared_vllm(training_config)

    elif training_config.weight_bridge_mode == "lora_only":
        # LoRA mode: freeze base model, train adapters only
        train_lora(training_config)

    else:
        # Legacy mode: periodic checkpoint saves + vLLM restarts
        train(training_config)
