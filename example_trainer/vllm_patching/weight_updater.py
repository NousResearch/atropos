"""
Weight Updater Process - Daemon that receives NCCL weight updates.

This process runs as a daemon spawned by the patched vLLM GPUModelRunner.
It joins NCCL process groups with the trainer and receives weight updates,
copying them directly into vLLM's shared memory tensors.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist

from .distributed_utils import (
    init_process_group,
    get_inference_urls,
    get_hostnames,
    get_json_data,
    get_name_conversions,
    permute,
    permute_1d,
)


def weight_updater_process(
    state_dict: Dict[str, torch.Tensor],
    num_q_heads: int,
    num_kv_heads: int,
    tp_rank: int,
    tp_size: int,
    gpu_id: int,
) -> None:
    """
    Daemon process that receives weight updates from trainers via NCCL.
    
    This runs inside a subprocess spawned by PatchedGPUModelRunner. It:
    1. Joins NCCL/Gloo process groups with the trainer
    2. Receives weight update broadcasts from rank 0 (trainer)
    3. Copies updated weights directly into the shared state_dict
    
    Since state_dict tensors have share_memory_() called on them, the main
    vLLM process immediately sees the updates for inference.
    
    Args:
        state_dict: Model state dict with shared memory tensors
        num_q_heads: Number of query attention heads (for permutation)
        num_kv_heads: Number of key/value attention heads
        tp_rank: Tensor parallel rank of this worker
        tp_size: Total tensor parallel size
        gpu_id: GPU device ID for this worker
    """
    # Configuration from environment
    num_inference_nodes = int(os.environ.get("NUM_INFERENCE_NODES", 0))
    cuda_devices = str(os.environ.get("CUDA_VISIBLE_DEVICES", "0")).split(",")
    debug = int(os.environ.get("WEIGHT_UPDATER_DEBUG", 0))
    
    # Determine world size based on setup
    if num_inference_nodes > 0:
        # Multi-node: 8 GPUs per node
        world_size = num_inference_nodes * 8
        ranks_per_node = 8
    else:
        # Single node: typically 4 inference GPUs
        world_size = 4
        ranks_per_node = 4
    
    # Get network info
    hostnames = get_hostnames()
    master_addr, master_gloo_addr, master_inference_addr, urls = get_inference_urls(
        num_inference_nodes
    )
    
    if master_addr is None:
        print(f"[Updater] Master address not found, exiting", flush=True)
        return
    
    # Set CUDA device
    torch.cuda.set_device(tp_rank)
    
    print(
        f"[Updater] Starting on TP rank {tp_rank}/{tp_size}, "
        f"q_heads={num_q_heads}, kv_heads={num_kv_heads}, gpu_id={gpu_id}",
        flush=True,
    )
    print(f"[Updater] Master: {master_addr}, world_size={world_size}", flush=True)
    
    # Determine this worker's rank within the inference group
    rank = -1
    if num_inference_nodes == 0:
        # Single node: skip first N GPUs (used by trainer)
        rank = int(cuda_devices[gpu_id]) - (8 - ranks_per_node)
    else:
        # Multi-node: find which inference node we're on
        for i, url in enumerate(urls):
            if hostnames and url in hostnames:
                rank = ranks_per_node * i + int(cuda_devices[gpu_id])
                break
    
    if rank < 0:
        print(f"[Updater] Could not determine rank, exiting", flush=True)
        return
    
    # Load config from vLLM
    print("[Updater] Loading bridge config...", flush=True)
    try:
        json_data = get_json_data()
    except Exception as e:
        print(f"[Updater] Failed to load config: {e}", flush=True)
        return
    
    param_name_list = sorted(json_data.get("param_mappings", {}).keys())
    num_training_gpus = json_data.get("dp_shard_degree", 1) * json_data.get("tp_degree", 1)
    total_group_size = num_training_gpus + world_size
    
    # Offset rank by training GPUs
    rank = rank + num_training_gpus
    
    print(f"[Updater] Total group size: {total_group_size}", flush=True)
    print(f"[Updater] Training GPUs: {num_training_gpus}", flush=True)
    print(f"[Updater] My rank: {rank}", flush=True)
    
    # Initialize process groups
    print("[Updater] Creating process groups...", flush=True)
    
    try:
        # Gloo group for coordination
        gloo_group = init_process_group(
            backend="gloo",
            init_method=f"tcp://{master_addr}",
            world_size=total_group_size,
            rank=rank,
            group_name="gloo_group",
        )
        print("[Updater] ✓ Gloo group created", flush=True)
        
        # NCCL group for tensor transfers
        nccl_group = init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_addr}",
            world_size=total_group_size,
            rank=rank,
            group_name="weight_update_group",
        )
        print("[Updater] ✓ NCCL group created", flush=True)
        
    except Exception as e:
        print(f"[Updater] Failed to create process groups: {e}", flush=True)
        return
    
    # Get device for tensors
    my_device = next(iter(state_dict.values())).device
    
    # Write dtype mapping if rank 0
    if rank == num_training_gpus:  # First inference rank
        _write_dtype_mapping(state_dict, json_data)
    
    print("[Updater] Entering update loop...", flush=True)
    
    # Buffers for merged QKV and gate_up projections
    qkv_buffer = {}
    gate_up_buffer = {}
    qkv_bias_buffer = {}
    w1w3_buffer = {}
    
    with torch.no_grad():
        while True:
            try:
                # Receive parameter index from trainer (rank 0)
                obj_indx = torch.zeros(1, dtype=torch.long, device=my_device)
                dist.broadcast(obj_indx, src=0, group=nccl_group)
                
                tt_indx = obj_indx.item()
                
                # -1 signals no update this round (heartbeat)
                if tt_indx == -1:
                    continue
                
                # Get parameter info
                if tt_indx >= len(param_name_list):
                    print(f"[Updater] Invalid index {tt_indx}, skipping", flush=True)
                    continue
                    
                tt_name = param_name_list[tt_indx]
                param_info = json_data["param_mappings"].get(tt_name, {})
                vllm_name = param_info.get("vllm_name", tt_name)
                local_shape = param_info.get("local_shape", [])
                
                if vllm_name not in state_dict:
                    if debug:
                        print(f"[Updater] {vllm_name} not in state_dict, skipping", flush=True)
                    continue
                
                target_dtype = state_dict[vllm_name].dtype
                
                if debug:
                    print(
                        f"[Updater] Receiving {tt_name} -> {vllm_name}, "
                        f"shape={local_shape}, dtype={target_dtype}",
                        flush=True,
                    )
                
                # Gather tensors from all training ranks
                tensor_list = [
                    torch.zeros(
                        local_shape if idx < num_training_gpus else [1],
                        dtype=target_dtype,
                        device=my_device,
                    )
                    for idx in range(total_group_size)
                ]
                
                dist.all_gather(
                    tensor_list,
                    torch.zeros(1, dtype=target_dtype, device=my_device),
                    group=nccl_group,
                )
                
                # Only keep training tensors
                tensor_list = tensor_list[:num_training_gpus]
                
                # Merge tensors from different parallel configurations
                tensor = _merge_tensors(
                    tensor_list,
                    json_data,
                    param_info,
                    state_dict[vllm_name],
                )
                
                # Apply updates (handling merged QKV, gate_up, etc.)
                _apply_weight_update(
                    state_dict,
                    vllm_name,
                    tt_name,
                    tensor,
                    param_info,
                    num_q_heads,
                    num_kv_heads,
                    qkv_buffer,
                    gate_up_buffer,
                    qkv_bias_buffer,
                    w1w3_buffer,
                    debug,
                )
                
            except Exception as e:
                print(f"[Updater] Error in update loop: {e}", flush=True)
                import traceback
                traceback.print_exc()
                time.sleep(1)


def _write_dtype_mapping(
    state_dict: Dict[str, torch.Tensor],
    json_data: Dict[str, Any],
) -> None:
    """Write dtype mapping file for trainer reference."""
    try:
        log_dir = os.environ.get("LOGDIR", ".")
        name_conversions = get_name_conversions(json_data.get("param_mappings", {}))
        
        weight_dtypes = {}
        for name in state_dict.keys():
            tt_names = name_conversions.get(name, [name])
            for tt_name in tt_names:
                weight_dtypes[tt_name] = str(state_dict[name].dtype).split(".")[-1]
        
        with open(f"{log_dir}/vllm_dtypes.json", "w") as f:
            json.dump(weight_dtypes, f, indent=2)
            
        print("[Updater] Wrote dtype mapping", flush=True)
    except Exception as e:
        print(f"[Updater] Failed to write dtype mapping: {e}", flush=True)


def _merge_tensors(
    tensor_list: List[torch.Tensor],
    json_data: Dict[str, Any],
    param_info: Dict[str, Any],
    target_tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Merge tensors from distributed training into single tensor.
    
    Handles FSDP (data parallel) and TP (tensor parallel) sharding.
    """
    dp_shard_degree = json_data.get("dp_shard_degree", 1)
    tp_degree = json_data.get("tp_degree", 1)
    tp_shard_dim = param_info.get("tp_shard_dim", 0)
    
    if dp_shard_degree > 1:
        # First merge across data parallel dimension
        tp_tensors = []
        for i in range(tp_degree):
            dp_tensors = tensor_list[i::tp_degree]
            tp_tensors.append(torch.cat(dp_tensors, dim=0))
        
        # Then merge across tensor parallel dimension if needed
        if tp_degree > 1:
            if tp_tensors[0].shape == target_tensor.shape:
                tensor = tp_tensors[0].contiguous()
            else:
                tensor = torch.cat(tp_tensors, dim=tp_shard_dim).contiguous()
        else:
            tensor = tp_tensors[0].contiguous()
    else:
        # No FSDP, just merge TP shards
        tensor = torch.cat(tensor_list, dim=tp_shard_dim).contiguous()
    
    # Cast to target dtype if needed
    if tensor.dtype != target_tensor.dtype:
        tensor = tensor.to(target_tensor.dtype)
    
    return tensor


def _apply_weight_update(
    state_dict: Dict[str, torch.Tensor],
    vllm_name: str,
    tt_name: str,
    tensor: torch.Tensor,
    param_info: Dict[str, Any],
    num_q_heads: int,
    num_kv_heads: int,
    qkv_buffer: Dict[str, torch.Tensor],
    gate_up_buffer: Dict[str, torch.Tensor],
    qkv_bias_buffer: Dict[str, torch.Tensor],
    w1w3_buffer: Dict[str, torch.Tensor],
    debug: bool,
) -> None:
    """
    Apply weight update to state_dict, handling merged projections.
    
    vLLM often merges QKV projections and gate/up projections into single
    tensors for efficiency. This handles unpacking and merging correctly.
    """
    needs_permute = param_info.get("needs_permute", False)
    shape = param_info.get("shape", list(tensor.shape))
    
    def _debug_diff(name: str, old: torch.Tensor, new: torch.Tensor) -> None:
        if debug:
            diff = (new.float() - old.float()).abs()
            print(
                f"[WEIGHT DIFF] {name}: mean={diff.mean().item():.6e}, "
                f"std={diff.std().item():.6e}",
                flush=True,
            )
    
    # Handle merged QKV projection weights
    if "qkv_proj.weight" in vllm_name:
        key_val = "q" if ".wq." in tt_name or "q_proj" in tt_name else \
                  "v" if ".wv." in tt_name or "v_proj" in tt_name else "k"
        
        if key_val == "q" and needs_permute:
            tensor = permute(tensor, num_q_heads)
        elif key_val == "k" and needs_permute:
            tensor = permute(tensor, num_kv_heads)
        
        qkv_buffer[key_val] = tensor
        
        if len(qkv_buffer) == 3:
            merged = torch.cat([qkv_buffer["q"], qkv_buffer["k"], qkv_buffer["v"]], dim=0)
            _debug_diff(vllm_name, state_dict[vllm_name].data, merged)
            state_dict[vllm_name].data.copy_(merged.contiguous())
            qkv_buffer.clear()
    
    # Handle merged gate/up projection weights
    elif "gate_up_proj.weight" in vllm_name:
        key_val = "w1" if ".w1." in tt_name or "gate_proj" in tt_name else "w3"
        gate_up_buffer[key_val] = tensor
        
        if len(gate_up_buffer) == 2:
            merged = torch.cat([gate_up_buffer["w1"], gate_up_buffer["w3"]], dim=0)
            _debug_diff(vllm_name, state_dict[vllm_name].data, merged)
            state_dict[vllm_name].data.copy_(merged.contiguous())
            gate_up_buffer.clear()
    
    # Handle merged w1/w3 weights (alternative naming)
    elif "w13_weight" in vllm_name:
        key_val = "w1" if ".w1" in tt_name else "w3"
        w1w3_buffer[key_val] = tensor
        
        if len(w1w3_buffer) == 2:
            merged = torch.cat([w1w3_buffer["w1"], w1w3_buffer["w3"]], dim=1)
            _debug_diff(vllm_name, state_dict[vllm_name].data, merged)
            state_dict[vllm_name].data.copy_(merged.contiguous())
            w1w3_buffer.clear()
    
    # Handle merged QKV bias
    elif "qkv_proj.bias" in vllm_name:
        key_val = "q" if ".wq." in tt_name else "v" if ".wv." in tt_name else "k"
        
        if key_val == "q" and needs_permute:
            tensor = permute_1d(tensor, num_q_heads)
        elif key_val == "k" and needs_permute:
            tensor = permute_1d(tensor, num_kv_heads)
        
        qkv_bias_buffer[key_val] = tensor
        
        if len(qkv_bias_buffer) == 3:
            merged = torch.cat([qkv_bias_buffer["q"], qkv_bias_buffer["k"], qkv_bias_buffer["v"]], dim=0)
            _debug_diff(vllm_name, state_dict[vllm_name].data, merged)
            state_dict[vllm_name].data.copy_(merged.contiguous())
            qkv_bias_buffer.clear()
    
    # Handle regular weights (possibly needing permutation)
    elif needs_permute:
        if len(shape) == 2:
            tensor = permute(tensor, shape[0]).contiguous()
        elif len(shape) == 1:
            tensor = permute_1d(tensor, shape[0]).contiguous()
        
        _debug_diff(vllm_name, state_dict[vllm_name].data, tensor)
        state_dict[vllm_name].data.copy_(tensor)
    
    # Simple weight copy
    else:
        _debug_diff(vllm_name, state_dict[vllm_name].data, tensor)
        state_dict[vllm_name].data.copy_(tensor)


