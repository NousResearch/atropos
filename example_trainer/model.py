"""
Model loading and shared memory utilities for GRPO trainer.

Handles:
- Standard model loading (legacy mode)
- LoRA model loading and wrapping
- Single-copy mode: Attaching to vLLM's shared tensors via CUDA IPC
"""

import base64
import json
import os
import re
from typing import Dict, Optional, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .config import TrainingConfig

# Import PEFT for LoRA training
try:
    from peft import LoraConfig, TaskType, get_peft_model

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


def _get_attention_implementation() -> str:
    """
    Determine the best attention implementation to use.
    Priority:
    1. Flash Attention 2 (if flash_attn library is available and works)
    2. SDPA (PyTorch's scaled dot-product attention)

    Returns:
        Tuple of (attn_implementation string, human-readable name)
    """
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def _load_model_with_attention(
    model_name_or_config,
    torch_dtype=torch.bfloat16,
    from_config: bool = False,
) -> torch.nn.Module:
    """
    Load a model with the best available attention implementation.

    Args:
        model_name_or_config: Either a model name (str) or a model config object
        torch_dtype: Data type for model weights
        from_config: If True, use from_config (for meta device loading - no weights)
                    If False, use from_pretrained (downloads and loads weights)

    Returns:
        Loaded model with appropriate attention implementation
    """
    # Select the loader function based on mode
    # from_config: creates empty shell (meta device), from_pretrained: loads weights
    loader = (
        AutoModelForCausalLM.from_config
        if from_config
        else AutoModelForCausalLM.from_pretrained
    )

    # Try attention implementations in order of preference
    for attn_impl in ["flash_attention_2", "sdpa"]:
        # Skip flash_attention_2 if not available
        if (
            attn_impl == "flash_attention_2"
            and _get_attention_implementation() != "flash_attention_2"
        ):
            continue

        try:
            model = loader(
                model_name_or_config,
                torch_dtype=torch_dtype,
                attn_implementation=attn_impl,
            )
            print(f"[Setup] Using {attn_impl.replace('_', ' ').title()}")
            return model
        except Exception as e:
            if attn_impl == "flash_attention_2":
                print(f"[Setup] Flash Attention 2 failed ({e}), trying SDPA...")
                continue
            raise

    # Should never reach here, but just in case
    raise RuntimeError("Failed to load model with any attention implementation")


def load_model_and_tokenizer(
    config: TrainingConfig,
    single_copy: bool = False,
) -> Tuple[torch.nn.Module, AutoTokenizer]:
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
        config_path = _find_vllm_config(config)
        model = _attach_to_vllm_shared_tensors(config, config_path)

        if model is not None:
            print("[Setup] ✓ Single-copy mode active - using vLLM's tensors directly!")
            _apply_train_layer_filter(model, config.train_layer_indices)
            # Enable gradient checkpointing to save memory (was missing before!)
            _setup_gradient_checkpointing(model, config)
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

    elif config.weight_bridge_mode in ("lora_only", "lora_restart"):
        # Both lora_only and lora_restart use PEFT LoRA adapters
        model = _load_model_with_lora(config)

    else:
        # Legacy mode: load full model
        print("[Setup] Loading model for legacy mode...")
        model = _load_model_with_attention(config.model_name)
        model.to(config.device)
        _apply_train_layer_filter(model, config.train_layer_indices)

    # Enable gradient checkpointing
    _setup_gradient_checkpointing(model, config)
    model.train()

    return model, tokenizer


def _find_vllm_config(config: TrainingConfig) -> str:
    """Find the vllm_bridge_config.json file."""
    # Check explicit path first
    if config.vllm_config_path and os.path.exists(config.vllm_config_path):
        print(f"[Setup] Using explicit vLLM config path: {config.vllm_config_path}")
        return config.vllm_config_path

    # Auto-detect from common locations
    possible_paths = [
        os.environ.get("LOGDIR", "."),
        ".",
        "/tmp/atropos_bridge",
        os.path.dirname(os.path.abspath(__file__)),
    ]
    # Look through possible
    for log_dir in possible_paths:
        candidate = os.path.join(log_dir, "vllm_bridge_config.json")
        if os.path.exists(candidate):
            print(f"[Setup] Found vLLM config at: {candidate}")
            return candidate

    checked = [os.path.join(p, "vllm_bridge_config.json") for p in possible_paths]
    raise RuntimeError(
        f"[Setup] Could not find vllm_bridge_config.json\n"
        f"Checked: {checked}\n"
        f"Tip: Use --vllm-config-path to specify the path explicitly\n"
        f"Make sure vLLM is running with VLLM_ENABLE_SHARED_WEIGHTS=1 and LOGDIR set"
    )


def _load_model_with_lora(config: TrainingConfig) -> torch.nn.Module:
    """
    Load base model and wrap with LoRA adapters.

    Args:
        config: Training configuration with LoRA settings

    Returns:
        PEFT model with LoRA adapters applied
    """
    if not PEFT_AVAILABLE:  # Yeah no PEFT is needed no matter what bless huggingface
        raise RuntimeError("PEFT library not available. Install with: pip install peft")

    print("[Setup] Loading base model for LoRA mode...")
    base_model = _load_model_with_attention(config.model_name)
    base_model.to(config.device)

    # Determine target modules
    target_modules = config.lora_target_modules
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]
    layer_indices = config.lora_layer_indices

    if layer_indices is not None:
        num_hidden_layers = getattr(base_model.config, "num_hidden_layers", None)
        if num_hidden_layers is None:
            raise RuntimeError(
                "Model config does not expose num_hidden_layers; cannot validate "
                "--lora-layer-indices for this architecture."
            )
        invalid = [idx for idx in layer_indices if idx >= num_hidden_layers]
        if invalid:
            raise ValueError(
                f"Invalid --lora-layer-indices {invalid} for model with "
                f"{num_hidden_layers} layers (valid range: 0-{num_hidden_layers - 1})"
            )

    print(f"Applying LoRA: r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"Target modules: {target_modules}")
    if layer_indices is not None:
        print(
            f"Applying LoRA only to layers: {layer_indices} "
            f"(total {len(layer_indices)})"
        )

    lora_kwargs = dict(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    if layer_indices is not None:
        lora_kwargs["layers_to_transform"] = layer_indices
    lora_config = LoraConfig(**lora_kwargs)

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    return model


def _apply_train_layer_filter(
    model: torch.nn.Module, layer_indices: Optional[list[int]]
) -> None:
    """
    Freeze all parameters except selected transformer block indices.

    Applies to full-model modes (shared_vllm / legacy), not LoRA.
    """
    if layer_indices is None:
        return

    num_hidden_layers = getattr(model.config, "num_hidden_layers", None)
    if num_hidden_layers is None:
        num_hidden_layers = getattr(model.config, "n_layer", None)
    if num_hidden_layers is None:
        raise RuntimeError(
            "Model config does not expose num_hidden_layers or n_layer; "
            "cannot validate --train-layer-indices for this architecture."
        )

    invalid = [idx for idx in layer_indices if idx >= num_hidden_layers]
    if invalid:
        raise ValueError(
            f"Invalid --train-layer-indices {invalid} for model with "
            f"{num_hidden_layers} layers (valid range: 0-{num_hidden_layers - 1})"
        )

    allowed = set(layer_indices)
    layer_pattern = re.compile(r"\.layers\.(\d+)\.")
    trainable_params = 0
    total_params = 0

    for name, param in model.named_parameters():
        match = layer_pattern.search(name)
        should_train = bool(match and int(match.group(1)) in allowed)
        param.requires_grad_(should_train)
        total_params += param.numel()
        if should_train:
            trainable_params += param.numel()

    if trainable_params == 0:
        raise RuntimeError(
            "--train-layer-indices did not match any trainable parameters. "
            "Check architecture naming and selected indices."
        )

    pct = 100.0 * trainable_params / max(total_params, 1)
    print(
        f"[Setup] Training only transformer layers {sorted(allowed)} "
        f"({trainable_params}/{total_params} params, {pct:.2f}%)"
    )


def _setup_gradient_checkpointing(
    model: torch.nn.Module, config: TrainingConfig
) -> None:
    """Configure gradient checkpointing for the model."""
    # Disable KV cache - incompatible with gradient checkpointing
    model.config.use_cache = False

    if config.weight_bridge_mode in ("lora_only", "lora_restart"):
        # PEFT models need special handling - enable_input_require_grads is CRITICAL
        # Without this, the LoRA parameters won't receive gradients!
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    else:
        model.gradient_checkpointing_enable()


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
    # Load the bridge that we just searched for
    try:
        with open(bridge_config_path, "r") as f:
            bridge_config = json.load(f)
        print(f"[Setup] Bridge config keys: {list(bridge_config.keys())}")
    except Exception as e:
        print(f"[Setup] Could not read bridge config: {e}")
        return None

    single_copy_enabled = bridge_config.get("single_copy_enabled", False)
    print(f"[Setup] single_copy_enabled in config: {single_copy_enabled}")
    # If single copy is not enable here then we exist because VLLM is likely botched
    if not single_copy_enabled:
        print("[Setup] Single-copy mode not available (single_copy_enabled=False)")
        print("[Setup] Make sure vLLM was started with VLLM_ENABLE_SHARED_WEIGHTS=1")
        return None
    # Get IPC handles from bridge config - memory pointers to shared weight tensors
    ipc_handles_raw = bridge_config.get("ipc_handles", {})
    print(f"[Setup] IPC handles count: {len(ipc_handles_raw)}")
    if not ipc_handles_raw:
        print("[Setup] No IPC handles found in bridge config")
        return None

    # Deserialize base64-encoded bytes
    ipc_handles = _deserialize_ipc_handles(ipc_handles_raw)

    print(f"[Setup] Attaching to vLLM's shared tensors ({len(ipc_handles)} tensors)...")
    print("[Setup] TRUE SINGLE-COPY MODE - No additional model memory!")

    # Load model config (not weights) to get architecture
    # doesn't store the buffers just basically the schematics. This is the
    # the blueprint for the house not the actual house
    model_config = AutoConfig.from_pretrained(config.model_name)

    # Create empty model on meta device (no memory allocation)
    # Try Flash Attention 2 first (matches vLLM better), fall back to SDPA
    with torch.device("meta"):
        model = _load_model_with_attention(model_config, from_config=True)

    param_names = list(model.state_dict().keys())
    print(f"[Setup] Model architecture has {len(param_names)} parameters", flush=True)

    # Initialize CUDA on devices used by vLLM
    device_indices = _initialize_cuda_devices(ipc_handles)

    # Create mapping from HF names to vLLM tensors
    vllm_to_hf_mapping = _create_vllm_to_hf_mapping(
        model, ipc_handles, debug=config.debug_loading
    )

    # Reconstruct tensors and build state dict
    hf_state_dict, attached_count, fused_count = _reconstruct_shared_tensors(
        ipc_handles, vllm_to_hf_mapping, config
    )

    print(
        f"[Setup] Attached {attached_count} tensors ({fused_count} from fused layers)"
    )

    if attached_count == 0:
        print("[Setup] Could not attach any tensors, falling back to regular loading")
        return None

    # Validate mapping coverage
    _validate_mapping_coverage(model, hf_state_dict, attached_count)

    # Load state dict into model
    model.load_state_dict(hf_state_dict, strict=False, assign=True)

    # Initialize remaining meta tensors
    device = f"cuda:{list(device_indices)[0]}" if device_indices else "cuda:0"
    _initialize_meta_tensors(model, device, config)

    # Final validation - ensure nothing is left on meta device
    _validate_no_meta_tensors(model)

    print("[Setup] ✓ All tensors successfully initialized on CUDA")
    return model


def _deserialize_ipc_handles(handles_raw: dict) -> dict:
    """Deserialize base64-encoded bytes in IPC handles."""

    def deserialize(handles):
        result = {}
        for k, v in handles.items():
            if isinstance(v, dict):
                if "_bytes_b64_" in v:
                    result[k] = base64.b64decode(v["_bytes_b64_"])
                else:
                    result[k] = deserialize(v)
            else:
                result[k] = v
        return result

    return deserialize(handles_raw)


def _initialize_cuda_devices(ipc_handles: dict) -> set:
    """Initialize CUDA context on devices used by IPC handles."""
    device_indices = set()
    for name, info in ipc_handles.items():
        if "device_index" in info:
            device_indices.add(info["device_index"])

    print(f"[Setup] IPC handles span devices: {sorted(device_indices)}", flush=True)

    for dev_idx in sorted(device_indices):
        print(f"[Setup] Initializing CUDA on device {dev_idx}...", flush=True)
        torch.cuda.set_device(dev_idx)
        torch.cuda.synchronize(dev_idx)
        print(f"[Setup] ✓ Device {dev_idx} ready", flush=True)

    return device_indices


def _reconstruct_shared_tensors(
    ipc_handles: dict,
    vllm_to_hf_mapping: dict,
    config: TrainingConfig,
) -> Tuple[dict, int, int]:
    """Reconstruct tensors from IPC handles and build state dict."""
    hf_state_dict = {}
    vllm_tensor_cache: Dict[str, torch.Tensor] = {}
    attached_count = 0
    fused_count = 0

    def reconstruct_vllm_tensor(vllm_name: str) -> Optional[torch.Tensor]:
        if vllm_name in vllm_tensor_cache:
            return vllm_tensor_cache[vllm_name]

        if vllm_name not in ipc_handles:
            return None

        ipc_info = ipc_handles[vllm_name]
        if "ipc_handle_b64" not in ipc_info:
            return None

        try:
            device_index = ipc_info["device_index"]
            ipc_handle = base64.b64decode(ipc_info["ipc_handle_b64"])
            storage_size = ipc_info["storage_size"]
            storage_offset_orig = ipc_info["storage_offset_orig"]
            ref_counter_handle = base64.b64decode(ipc_info["ref_counter_handle_b64"])
            ref_counter_offset = ipc_info["ref_counter_offset"]
            event_handle = base64.b64decode(ipc_info["event_handle_b64"])
            event_sync_required = ipc_info["event_sync_required"]

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

            storage = torch.UntypedStorage._new_shared_cuda(*share_tuple)
            dtype = getattr(torch, ipc_info["dtype"].replace("torch.", ""))
            tensor = torch.tensor([], dtype=dtype, device=f"cuda:{device_index}")
            tensor.set_(
                storage,
                storage_offset=ipc_info["tensor_storage_offset"],
                size=ipc_info["shape"],
                stride=ipc_info["stride"],
            )

            vllm_tensor_cache[vllm_name] = tensor
            return tensor

        except Exception as e:
            print(f"[Setup] Failed to reconstruct {vllm_name}: {e}", flush=True)
            return None

    for hf_name, mapping_info in vllm_to_hf_mapping.items():
        try:
            if isinstance(mapping_info, dict):
                # Fused mapping - slice the source tensor
                vllm_name = mapping_info["source"]
                slice_start, slice_end = mapping_info["slice"]
                slice_dim = mapping_info["dim"]

                full_tensor = reconstruct_vllm_tensor(vllm_name)
                if full_tensor is None:
                    continue

                # Create VIEW (not copy) into the fused tensor
                if slice_dim == 0:
                    tensor = full_tensor[slice_start:slice_end]
                else:
                    tensor = full_tensor.narrow(
                        slice_dim, slice_start, slice_end - slice_start
                    )

                tensor.requires_grad_(True)
                hf_state_dict[hf_name] = tensor
                fused_count += 1
                attached_count += 1

            else:
                # Direct mapping
                vllm_name = mapping_info
                tensor = reconstruct_vllm_tensor(vllm_name)
                if tensor is None:
                    continue

                tensor.requires_grad_(True)
                hf_state_dict[hf_name] = tensor
                attached_count += 1

        except Exception as e:
            print(f"[Setup] Failed to attach {hf_name}: {e}", flush=True)

    return hf_state_dict, attached_count, fused_count


def _validate_mapping_coverage(
    model: torch.nn.Module,
    hf_state_dict: dict,
    attached_count: int,
) -> None:
    """Validate that enough parameters were mapped."""
    hf_param_count = len(list(model.named_parameters()))
    mapping_coverage = attached_count / hf_param_count if hf_param_count > 0 else 0

    # Note: attached_count may be > param_count because state_dict includes buffers
    # while named_parameters only counts trainable params
    print(
        f"[Setup] Mapping coverage: {attached_count} tensors for {hf_param_count} parameters "
        f"(>100% is OK - includes buffers)"
    )

    if mapping_coverage < 0.90:
        unmapped_params = set(model.state_dict().keys()) - set(hf_state_dict.keys())
        warning_msg = (
            f"[Setup] WARNING: Low mapping coverage ({mapping_coverage:.1%})\n"
        )
        warning_msg += f"Unmapped parameters ({len(unmapped_params)}):\n"
        for name in list(unmapped_params)[:20]:
            warning_msg += f"  - {name}\n"
        print(warning_msg)

        if mapping_coverage < 0.50:
            raise RuntimeError(
                f"[Setup] CRITICAL: Only {mapping_coverage:.1%} of parameters mapped!"
            )
    else:
        print(f"[Setup] ✓ Good mapping coverage ({mapping_coverage:.1%})")


def _initialize_meta_tensors(
    model: torch.nn.Module,
    device: str,
    config: TrainingConfig,
) -> None:
    """Initialize any remaining meta tensors after loading."""
    meta_params = [
        name for name, p in model.named_parameters() if p.device.type == "meta"
    ]
    meta_buffers = [
        name for name, b in model.named_buffers() if b.device.type == "meta"
    ]

    if config.debug_loading:
        print(
            f"\n[DIAGNOSTIC] Meta params: {len(meta_params)}, Meta buffers: {len(meta_buffers)}"
        )

    def get_parent_and_name(model, full_name):
        parts = full_name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent, parts[-1]

    meta_count = 0

    # Initialize meta parameters
    for name in meta_params:
        param = dict(model.named_parameters()).get(name)
        if param is None:
            continue

        try:
            new_data = torch.zeros(param.shape, dtype=param.dtype, device=device)
            new_param = torch.nn.Parameter(new_data, requires_grad=param.requires_grad)
            parent, attr_name = get_parent_and_name(model, name)
            setattr(parent, attr_name, new_param)
            meta_count += 1
        except Exception as e:
            if config.debug_loading:
                print(f"[DIAGNOSTIC] FAILED to initialize {name}: {e}")

    # Initialize meta buffers
    for name in meta_buffers:
        buffer = dict(model.named_buffers()).get(name)
        if buffer is None:
            continue

        try:
            if "inv_freq" in name:
                dim = buffer.shape[0] * 2
                # Get rope_theta from model config (default 10000.0 for LLaMA, but Qwen3 uses 5000000!)
                rope_theta = getattr(model.config, "rope_theta", 10000.0)
                inv_freq = 1.0 / (
                    rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
                )
                new_buffer = inv_freq.to(dtype=buffer.dtype, device=device)
                print(f"[Setup] Initialized {name} with rope_theta={rope_theta}")
            else:
                new_buffer = torch.zeros(
                    buffer.shape, dtype=buffer.dtype, device=device
                )

            parent, attr_name = get_parent_and_name(model, name)
            parent.register_buffer(attr_name, new_buffer)
            meta_count += 1
        except Exception as e:
            if config.debug_loading:
                print(f"[DIAGNOSTIC] FAILED to initialize buffer {name}: {e}")

    print(f"\n[Setup] Initialized {meta_count} remaining meta tensors")


def _validate_no_meta_tensors(model: torch.nn.Module) -> None:
    """Ensure no parameters or buffers are still on meta device."""
    final_meta_params = [
        name for name, p in model.named_parameters() if p.device.type == "meta"
    ]
    final_meta_buffers = [
        name for name, b in model.named_buffers() if b.device.type == "meta"
    ]

    if final_meta_params or final_meta_buffers:
        error_msg = "[Setup] CRITICAL ERROR: Some tensors are still on meta device!\n"
        error_msg += "The model would produce GARBAGE output.\n\n"

        if final_meta_params:
            error_msg += f"Meta parameters ({len(final_meta_params)}):\n"
            for name in final_meta_params[:20]:
                error_msg += f"  - {name}\n"

        if final_meta_buffers:
            error_msg += f"\nMeta buffers ({len(final_meta_buffers)}):\n"
            for name in final_meta_buffers[:20]:
                error_msg += f"  - {name}\n"

        raise RuntimeError(error_msg)


def _create_vllm_to_hf_mapping(
    model: torch.nn.Module,
    ipc_handles: dict,
    debug: bool = False,
) -> dict:
    """
    Create mapping from HuggingFace parameter names to vLLM tensor names.

    Handles fused layers:
    - qkv_proj (vLLM) = q_proj + k_proj + v_proj (HF)
    - gate_up_proj (vLLM) = gate_proj + up_proj (HF)

    Uses actual tensor shapes from HF model to determine slice sizes,
    rather than calculating from config (which can be wrong for some models).
    """
    hf_state_dict = model.state_dict()
    print("Here is the HF state dict so that we can get a better view ")
    hf_params = set(hf_state_dict.keys())
    vllm_params = set(ipc_handles.keys())

    # Get model config for fallback dimension calculations
    model_config = model.config
    hidden_size = getattr(model_config, "hidden_size", 4096)
    num_attention_heads = getattr(model_config, "num_attention_heads", 32)
    num_key_value_heads = getattr(
        model_config, "num_key_value_heads", num_attention_heads
    )
    intermediate_size = getattr(model_config, "intermediate_size", hidden_size * 4)

    # Try to get head_dim from config (some models like Qwen3 have this)
    head_dim = getattr(model_config, "head_dim", None)
    if head_dim is None:
        head_dim = hidden_size // num_attention_heads

    # Determine QKV sizes from ACTUAL HF model tensor shapes
    # Look for a q_proj weight in the model to get the actual size
    q_size = None
    k_size = None
    v_size = None

    for name, param in hf_state_dict.items():
        if "q_proj.weight" in name and q_size is None:
            q_size = param.shape[0]  # Output dimension
        elif "k_proj.weight" in name and k_size is None:
            k_size = param.shape[0]
        elif "v_proj.weight" in name and v_size is None:
            v_size = param.shape[0]
        if q_size and k_size and v_size:
            break

    # Fallback to calculated values if not found
    if q_size is None:
        q_size = num_attention_heads * head_dim
    if k_size is None:
        k_size = num_key_value_heads * head_dim
    if v_size is None:
        v_size = num_key_value_heads * head_dim

    # Also get gate/up sizes from actual HF model
    gate_size = None
    up_size = None

    for name, param in hf_state_dict.items():
        if "gate_proj.weight" in name and gate_size is None:
            gate_size = param.shape[0]
        elif "up_proj.weight" in name and up_size is None:
            up_size = param.shape[0]
        if gate_size and up_size:
            break

    # Fallback
    if gate_size is None:
        gate_size = intermediate_size
    if up_size is None:
        up_size = intermediate_size

    # Always print sizes for debugging weight sharing issues
    print(
        f"[Mapping] Model config: hidden={hidden_size}, heads={num_attention_heads}, "
        f"kv_heads={num_key_value_heads}, head_dim={head_dim}"
    )
    print(f"[Mapping] QKV sizes from HF model: q={q_size}, k={k_size}, v={v_size}")
    print(f"[Mapping] Gate/Up sizes from HF model: gate={gate_size}, up={up_size}")

    mapping = {}

    def find_vllm_name(hf_name: str) -> Optional[str]:
        if hf_name in vllm_params:
            return hf_name
        if not hf_name.startswith("model."):
            candidate = f"model.{hf_name}"
            if candidate in vllm_params:
                return candidate
        if hf_name.startswith("model."):
            candidate = hf_name[6:]
            if candidate in vllm_params:
                return candidate
        return None

    def find_fused_source(hf_name: str, fused_suffix: str) -> Optional[str]:
        for unfused in ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]:
            if unfused in hf_name:
                fused_name = hf_name.replace(unfused, fused_suffix)
                found = find_vllm_name(fused_name)
                if found:
                    return found
        return None

    for hf_name in hf_params:
        # Try direct match first
        vllm_name = find_vllm_name(hf_name)
        if vllm_name:
            mapping[hf_name] = vllm_name
            continue

        # Check for QKV fusion
        if any(x in hf_name for x in ["q_proj", "k_proj", "v_proj"]):
            fused_name = find_fused_source(hf_name, "qkv_proj")
            if fused_name:
                if "q_proj" in hf_name:
                    start, end = 0, q_size
                elif "k_proj" in hf_name:
                    start, end = q_size, q_size + k_size
                else:
                    start, end = q_size + k_size, q_size + k_size + v_size

                mapping[hf_name] = {
                    "source": fused_name,
                    "slice": (start, end),
                    "dim": 0,
                    "type": "qkv_fusion",
                }
                continue

        # Check for Gate/Up fusion
        if any(x in hf_name for x in ["gate_proj", "up_proj"]):
            fused_name = find_fused_source(hf_name, "gate_up_proj")
            if fused_name:
                if "gate_proj" in hf_name:
                    start, end = 0, gate_size
                else:
                    start, end = gate_size, gate_size + up_size

                mapping[hf_name] = {
                    "source": fused_name,
                    "slice": (start, end),
                    "dim": 0,
                    "type": "gate_up_fusion",
                }
                continue

    if debug:
        direct = sum(1 for v in mapping.values() if isinstance(v, str))
        fused = sum(1 for v in mapping.values() if isinstance(v, dict))
        print(
            f"[Mapping] Total: {len(mapping)} mapped ({direct} direct, {fused} fused)"
        )

    return mapping
