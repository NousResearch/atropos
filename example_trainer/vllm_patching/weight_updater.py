"""
Weight Updater Process - Daemon that receives NCCL weight updates.

This process runs as a daemon spawned by the patched vLLM GPUModelRunner.
It joins NCCL process groups with the trainer and receives weight updates,
copying them directly into vLLM's shared memory tensors.
"""

from __future__ import annotations

import os
import time
from typing import Dict

import torch
import torch.distributed as dist

from .distributed_utils import (
    init_process_group,
    get_inference_urls,
    get_hostnames,
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
    debug = int(os.environ.get("WEIGHT_UPDATER_DEBUG", 0))
    
    # Get network info
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
    
    # For single-node mode (num_inference_nodes=0):
    # - Trainer is rank 0
    # - Inference daemon is rank 1 (or tp_rank + 1 for multi-GPU)
    # Total world size = 1 trainer + 1 inference = 2
    #
    # For multi-node mode:
    # - More complex, based on SLURM node allocation
    
    if num_inference_nodes == 0:
        # Single node: simple setup
        # World = [trainer (rank 0), inference daemon (rank 1)]
        num_training_ranks = 1
        num_inference_ranks = 1
        world_size = num_training_ranks + num_inference_ranks
        rank = num_training_ranks + tp_rank  # Daemon is rank 1
    else:
        # Multi-node: 8 GPUs per node
        hostnames = get_hostnames()
        cuda_devices = str(os.environ.get("CUDA_VISIBLE_DEVICES", "0")).split(",")
        ranks_per_node = 8
        world_size = num_inference_nodes * ranks_per_node
        
        rank = -1
        for i, url in enumerate(urls or []):
            if hostnames and url in hostnames:
                rank = ranks_per_node * i + int(cuda_devices[gpu_id])
                break
        
        if rank < 0:
            print(f"[Updater] Could not determine rank for multi-node, exiting", flush=True)
            return
    
    print(f"[Updater] Master: {master_addr}, world_size={world_size}, my_rank={rank}", flush=True)
    
    # Use state_dict keys as parameter list (we already have the model!)
    param_name_list = sorted(state_dict.keys())
    print(f"[Updater] Model has {len(param_name_list)} parameters", flush=True)
    
    # Use the world_size and rank we already calculated
    total_group_size = world_size
    
    # For single-node mode, trainer is rank 0
    num_training_ranks = 1 if num_inference_nodes == 0 else 1
    
    print(f"[Updater] Total group size: {total_group_size}", flush=True)
    print(f"[Updater] Training ranks: {num_training_ranks}", flush=True)
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
        import traceback
        traceback.print_exc()
        return
    
    # Get device for tensors
    my_device = next(iter(state_dict.values())).device
    
    # Build param info dict from state_dict
    param_info_dict = {}
    for name, tensor in state_dict.items():
        param_info_dict[name] = {
            "shape": list(tensor.shape),
            "dtype": tensor.dtype,
        }
    
    print("[Updater] Entering update loop...", flush=True)
    print(f"[Updater] Waiting for weight updates from trainer (rank 0)...", flush=True)
    
    update_count = 0
    
    with torch.no_grad():
        while True:
            try:
                # Receive parameter index from trainer (rank 0)
                obj_indx = torch.zeros(1, dtype=torch.long, device=my_device)
                dist.broadcast(obj_indx, src=0, group=nccl_group)
                
                tt_indx = obj_indx.item()
                
                # -1 signals heartbeat (no update)
                if tt_indx == -1:
                    continue
                
                # -2 signals shutdown
                if tt_indx == -2:
                    print("[Updater] Received shutdown signal", flush=True)
                    break
                
                # Get parameter info
                if tt_indx < 0 or tt_indx >= len(param_name_list):
                    if debug:
                        print(f"[Updater] Invalid index {tt_indx}, skipping", flush=True)
                    continue
                
                param_name = param_name_list[tt_indx]
                
                if param_name not in state_dict:
                    if debug:
                        print(f"[Updater] {param_name} not in state_dict, skipping", flush=True)
                    continue
                
                target_tensor = state_dict[param_name]
                target_shape = list(target_tensor.shape)
                target_dtype = target_tensor.dtype
                
                # Receive the tensor from trainer
                # Trainer sends via broadcast, we receive
                received_tensor = torch.zeros(target_shape, dtype=target_dtype, device=my_device)
                dist.broadcast(received_tensor, src=0, group=nccl_group)
                
                # Copy to shared memory
                state_dict[param_name].data.copy_(received_tensor)
                
                update_count += 1
                if debug or (update_count % 50 == 0):
                    print(f"[Updater] Updated {param_name} (#{update_count})", flush=True)
                
            except Exception as e:
                print(f"[Updater] Error in update loop: {e}", flush=True)
                import traceback
                traceback.print_exc()
                time.sleep(1)


# Note: Advanced multi-GPU tensor parallelism support removed for simplicity.
# For single-node mode, we use direct tensor broadcast which is sufficient.
