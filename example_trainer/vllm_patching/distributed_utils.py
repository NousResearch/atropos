"""
Distributed utilities for vLLM weight synchronization.

Provides process group initialization and communication helpers
for coordinating weight updates between trainer and vLLM.
"""

from __future__ import annotations

import json
import os
import socket
import time
from collections import defaultdict
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist


def init_process_group(
    backend: Optional[str] = None,
    init_method: Optional[str] = None,
    timeout: Optional[timedelta] = None,
    world_size: int = -1,
    rank: int = -1,
    store: Optional[Any] = None,
    group_name: str = "",
    pg_options: Optional[Any] = None,
) -> dist.ProcessGroup:
    """
    Initialize a custom process group for weight synchronization.
    
    This creates a named process group that coexists with vLLM's internal
    process groups, enabling direct tensor communication between trainer
    and inference processes.
    
    Args:
        backend: "nccl" for GPU, "gloo" for CPU
        init_method: Rendezvous URL (e.g., "tcp://host:port")
        timeout: How long to wait for other ranks
        world_size: Total number of processes
        rank: This process's rank
        store: Optional torch.distributed Store
        group_name: Name for this process group (must match across ranks)
        pg_options: Backend-specific options
        
    Returns:
        ProcessGroup for collective operations
    """
    from torch.distributed.distributed_c10d import (
        _new_process_group_helper,
        _world,
        Backend,
        default_pg_timeout,
        PrefixStore,
        rendezvous,
    )

    assert (store is None) or (init_method is None), \
        "Cannot specify both init_method and store."

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("undefined")

    if timeout is None:
        timeout = default_pg_timeout

    # Create store via rendezvous if not provided
    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)
        store = PrefixStore(group_name, store)

    # Handle PyTorch version differences for pg_options parameter
    pg_options_param_name = (
        "backend_options" if str(torch.__version__) >= "2.6" else "pg_options"
    )

    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        **{pg_options_param_name: pg_options},
        timeout=timeout,
    )

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}

    return pg


def broadcast_object_list(
    object_list: List[Any],
    src: Optional[int] = None,
    group: Optional[dist.ProcessGroup] = None,
    device: Optional[torch.device] = None,
    group_src: Optional[int] = None,
) -> None:
    """
    Broadcast a list of objects from source rank to all other ranks.
    
    Modified from torch.distributed.broadcast_object_list to work correctly
    with custom process groups where rank 0 may not be the default group's rank 0.
    
    Args:
        object_list: List of objects to broadcast (modified in-place on receivers)
        src: Global source rank (deprecated, use group_src)
        group: Process group to use
        device: Device for temporary tensors
        group_src: Source rank within the group
    """
    global_src = group_src if group_src is not None else src
    current_device = device

    # Broadcast object sizes first
    object_sizes_tensor = torch.empty(
        len(object_list), dtype=torch.long, device=current_device
    )
    dist.broadcast(object_sizes_tensor, src=global_src, group=group)

    # Broadcast serialized objects
    object_tensor = torch.empty(
        torch.sum(object_sizes_tensor).item(),
        dtype=torch.uint8,
        device=current_device,
    )
    dist.broadcast(object_tensor, src=global_src, group=group)

    # Deserialize objects
    offset = 0
    for i, obj_size in enumerate(object_sizes_tensor):
        obj_view = object_tensor[offset : offset + obj_size]
        obj_view = obj_view.type(torch.uint8)
        offset += obj_size
        object_list[i] = dist.distributed_c10d._tensor_to_object(
            obj_view, obj_size, group
        )


def get_inference_urls(num_inference_nodes: int = 0) -> Tuple[Optional[str], ...]:
    """
    Get URLs for inference server communication.
    
    Parses SLURM environment or uses localhost for single-machine setup.
    
    Args:
        num_inference_nodes: Number of dedicated inference nodes.
            0 = single machine, trainer and vLLM share the node
            >0 = multi-node, last N nodes are for inference
            
    Returns:
        Tuple of (master_addr, master_gloo_addr, master_inference_addr, nodelist)
        Returns (None, None, None, None) if not in a valid setup.
    """
    if num_inference_nodes > 0:
        # Multi-node SLURM setup
        slurm_nodelist = os.environ.get("SLURM_JOB_NODELIST")
        if not slurm_nodelist:
            return None, None, None, None
            
        # Parse SLURM node list
        nodelist = (
            os.popen(f'scontrol show hostnames {slurm_nodelist}')
            .read()
            .strip()
            .split("\n")
        )
        nodelist = [node for node in nodelist if node]
        
        # First node is master for process groups
        master_server = f"{nodelist[0]}:26756"
        master_gloo_server = f"{nodelist[0]}:26757"
        
        # Last N nodes are inference nodes
        inference_nodes = nodelist[-num_inference_nodes:]
        master_inference_server = f"{inference_nodes[0]}:26758"
        
        return master_server, master_gloo_server, master_inference_server, inference_nodes
        
    elif num_inference_nodes == 0:
        # Single machine setup
        master_server = "localhost:26756"
        master_gloo_server = "localhost:26757"
        master_inference_server = "localhost:26758"
        nodelist = ["localhost"]
        
        return master_server, master_gloo_server, master_inference_server, nodelist
        
    else:
        return None, None, None, None


def get_hostnames() -> Optional[List[str]]:
    """
    Get the hostnames for this machine.
    
    Parses /etc/hosts to find all hostnames associated with this machine's IP.
    
    Returns:
        List of [ip, hostname1, hostname2, ...] or None if not found.
    """
    my_ip = socket.gethostbyname(socket.gethostname())
    my_hostname = socket.gethostname()
    
    try:
        with open("/etc/hosts", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split()
                    if len(parts) >= 2 and ((parts[0] == my_ip) or (my_hostname in parts)):
                        ip = parts[0]
                        if ip.startswith("127."):
                            continue
                        return parts
    except Exception:
        pass
        
    return None


def get_json_data(log_dir: Optional[str] = None, timeout: int = 300) -> Dict[str, Any]:
    """
    Load the bridge configuration JSON from vLLM.
    
    Waits for the file to be created by vLLM's weight bridge setup.
    
    Args:
        log_dir: Directory containing the JSON file (defaults to LOGDIR env var)
        timeout: Maximum seconds to wait for file
        
    Returns:
        Parsed JSON data with parameter mappings and configuration.
        
    Raises:
        ValueError: If LOGDIR not set and log_dir not provided
        FileNotFoundError: If file not found after timeout
    """
    if log_dir is None:
        log_dir = os.environ.get("LOGDIR")
    if log_dir is None:
        raise ValueError("LOGDIR environment variable not set and log_dir not provided")
    
    json_path = os.path.join(log_dir, "vllm_bridge_config.json")
    
    wait_time = 0
    while not os.path.exists(json_path):
        if wait_time >= timeout:
            raise FileNotFoundError(f"Config file not found after {timeout}s: {json_path}")
        if wait_time % 10 == 0:
            print(f"[Updater] Waiting for {json_path}...", flush=True)
        time.sleep(1)
        wait_time += 1
    
    # Wait a moment for file to finish writing
    time.sleep(0.5)
    
    with open(json_path, "r") as f:
        return json.load(f)


def get_name_conversions(param_mappings: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Build reverse mapping from vLLM names to trainer names.
    
    Args:
        param_mappings: Dict mapping trainer param names to vLLM info
        
    Returns:
        Dict mapping vLLM names to list of trainer names
    """
    name_conversions = defaultdict(list)
    for name, info in param_mappings.items():
        vllm_name = info.get("vllm_name", name)
        name_conversions[vllm_name].append(name)
    return name_conversions


# Permutation functions for rotary embeddings
def permute(w: torch.Tensor, n_heads: int) -> torch.Tensor:
    """
    Permute weight tensor for sliced rotary embeddings.
    
    Args:
        w: Weight tensor of shape [dim1, dim2]
        n_heads: Number of attention heads
        
    Returns:
        Permuted tensor for rotary embedding compatibility
    """
    dim1 = w.shape[0]
    dim2 = w.shape[1]
    return (
        w.view(n_heads, dim1 // n_heads // 2, 2, dim2)
        .transpose(1, 2)
        .reshape(dim1, dim2)
    )


def permute_1d(w: torch.Tensor, n_heads: int) -> torch.Tensor:
    """
    Permute 1D weight tensor (bias) for sliced rotary embeddings.
    
    Args:
        w: Weight tensor of shape [dim1]
        n_heads: Number of attention heads
        
    Returns:
        Permuted tensor
    """
    dim1 = w.shape[0]
    return w.view(n_heads, dim1 // n_heads // 2, 2).transpose(1, 2).reshape(dim1)


