"""
vLLM Weight Bridge - Trainer-side integration for shared memory weight updates.

This module coordinates weight updates between the trainer and vLLM inference.

ARCHITECTURE:
    The patched vLLM server (using vllm_patching/) runs a daemon process that:
    1. Joins NCCL process groups with the trainer
    2. Receives weight updates via all_gather
    3. Copies updates into vLLM's shared memory tensors
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                       SHARED MEMORY (via share_memory_())               │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │                         Model Weights                            │   │
    │  │              (accessible from MULTIPLE processes)                │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    │           ▲                                          ▲                 │
    │           │ Reads                                    │ Writes          │
    │  ┌────────┴────────┐                     ┌───────────┴───────────┐    │
    │  │  vLLM Worker    │                     │  weight_updater       │    │
    │  │  (inference)    │                     │  daemon process       │    │
    │  └─────────────────┘                     └───────────┬───────────┘    │
    │                                                      │ NCCL            │
    │                                                      ▼                 │
    │                                          ┌─────────────────────┐      │
    │                                          │  Trainer Process    │      │
    │                                          │  (this bridge)      │      │
    │                                          └─────────────────────┘      │
    └─────────────────────────────────────────────────────────────────────────┘

MODES:
    LOCAL MODE (num_inference_nodes=0):
        - Single machine setup
        - Trainer and vLLM share the same node
        - NCCL for weight broadcast to vLLM's daemon
        
    DISTRIBUTED MODE (num_inference_nodes>0):
        - Multi-node setup with dedicated inference nodes
        - Last N nodes run vLLM inference
        - NCCL spans across nodes for weight updates
"""

from __future__ import annotations

import json
import os
import socket
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn


# =============================================================================
# Process Group Initialization
# =============================================================================


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
    
    Creates a named group that coexists with vLLM's internal process groups.
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

    backend = Backend(backend) if backend else Backend("undefined")
    timeout = timeout or default_pg_timeout

    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)
        store = PrefixStore(group_name, store)

    # Handle PyTorch version differences
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


def get_inference_urls(num_inference_nodes: int = 0) -> Tuple[Optional[str], ...]:
    """
    Get URLs for inference server communication.
    
    Returns:
        Tuple of (master_addr, master_gloo_addr, master_inference_addr, nodelist)
    """
    if num_inference_nodes > 0:
        slurm_nodelist = os.environ.get("SLURM_JOB_NODELIST")
        if not slurm_nodelist:
            return None, None, None, None
            
        nodelist = (
            os.popen(f'scontrol show hostnames {slurm_nodelist}')
            .read().strip().split("\n")
        )
        nodelist = [n for n in nodelist if n]
        
        master_server = f"{nodelist[0]}:26756"
        master_gloo_server = f"{nodelist[0]}:26757"
        inference_nodes = nodelist[-num_inference_nodes:]
        master_inference_server = f"{inference_nodes[0]}:26758"
        
        return master_server, master_gloo_server, master_inference_server, inference_nodes
        
    elif num_inference_nodes == 0:
        return "localhost:26756", "localhost:26757", "localhost:26758", ["localhost"]
    else:
        return None, None, None, None


# =============================================================================
# Bridge Configuration
# =============================================================================


@dataclass
class BridgeConfig:
    """Configuration for the vLLM weight bridge."""
    
    # Process group settings
    trainer_rank: int = 0
    world_size: int = 1
    init_method: str = "env://"
    num_inference_nodes: int = 0
    
    # Model settings
    model_name: str = ""
    device: str = "cuda"
    
    # Synchronization settings
    timeout_seconds: float = 300.0
    log_dir: Optional[str] = None
    
    # vLLM server URL for HTTP-based sync (fallback)
    vllm_api_url: str = "http://localhost:9001"
    
    # Derived from environment
    num_gpus_per_node: int = field(default_factory=lambda: torch.cuda.device_count())
    
    @property
    def is_local_mode(self) -> bool:
        """Local mode: single machine, uses NCCL to daemon on same node."""
        return self.num_inference_nodes == 0
    
    @property
    def uses_nccl(self) -> bool:
        """Whether NCCL is used for weight synchronization."""
        return self.num_inference_nodes >= 0
    
    @classmethod
    def from_training_config(cls, config: Any) -> "BridgeConfig":
        """Create BridgeConfig from a TrainingConfig object."""
        return cls(
            trainer_rank=getattr(config, 'trainer_rank', 0),
            world_size=getattr(config, 'world_size', 1),
            init_method=getattr(config, 'init_method', 'env://'),
            num_inference_nodes=getattr(config, 'num_inference_nodes', 0),
            model_name=config.model_name,
            device=config.device,
            log_dir=os.environ.get("LOGDIR"),
            vllm_api_url=f"http://localhost:{getattr(config, 'vllm_port', 9001)}",
        )


# =============================================================================
# Weight Bridge Class
# =============================================================================


class VLLMWeightBridge:
    """
    Bridge for synchronizing model weights between trainer and vLLM.
    
    This class:
    1. Initializes NCCL process groups with vLLM's weight updater daemon
    2. Broadcasts weight updates after each optimizer.step()
    3. Ensures vLLM immediately uses updated weights for inference
    
    Usage:
        bridge = VLLMWeightBridge(config)
        bridge.initialize()
        
        for batch in data:
            loss = compute_loss(model, batch)
            loss.backward()
            optimizer.step()
            bridge.broadcast_weights(model)  # vLLM now uses new weights
    """
    
    def __init__(self, config: BridgeConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Process groups
        self.nccl_group: Optional[dist.ProcessGroup] = None
        self.gloo_group: Optional[dist.ProcessGroup] = None
        
        # Parameter mappings (loaded from vLLM's JSON)
        self.param_mappings: Dict[str, Any] = {}
        self.param_name_list: List[str] = []
        
        # State
        self._initialized: bool = False
        self._update_count: int = 0
        
        # Derived config
        self._num_training_gpus: int = 0
        self._total_group_size: int = 0
    
    def initialize(self) -> None:
        """
        Initialize the bridge: create process groups and load mappings.
        
        Must be called before any weight synchronization.
        """
        if self._initialized:
            return
        
        print(f"[Bridge] Initializing weight bridge (rank {self.config.trainer_rank})")
        
        if self.config.uses_nccl:
            self._initialize_nccl_mode()
        else:
            self._initialize_http_mode()
        
        self._initialized = True
    
    def _initialize_nccl_mode(self) -> None:
        """Initialize NCCL-based weight synchronization."""
        print("[Bridge] Using NCCL mode for weight synchronization")
        
        # Get rendezvous URLs
        master_addr, master_gloo_addr, _, nodelist = get_inference_urls(
            self.config.num_inference_nodes
        )
        
        if master_addr is None:
            raise RuntimeError(
                "Could not determine inference URLs. "
                "Set NUM_INFERENCE_NODES environment variable."
            )
        
        print(f"[Bridge] Master address: {master_addr}")
        print(f"[Bridge] Inference nodes: {nodelist}")
        
        # Load parameter mappings from vLLM
        self._load_param_mappings()
        
        # Calculate group sizes
        # For single-node mode (num_inference_nodes=0):
        # - Simple setup: 1 trainer + 1 inference daemon = 2 ranks
        # For multi-node mode:
        # - More complex based on SLURM allocation
        
        if self.config.num_inference_nodes == 0:
            # Single node: simple 2-rank setup
            self._num_training_gpus = 1
            num_inference_gpus = 1
        else:
            # Multi-node: 8 GPUs per node
            self._num_training_gpus = self.config.world_size * 8
            num_inference_gpus = self.config.num_inference_nodes * 8
        
        self._total_group_size = self._num_training_gpus + num_inference_gpus
        
        print(f"[Bridge] Training ranks: {self._num_training_gpus}")
        print(f"[Bridge] Inference ranks: {num_inference_gpus}")
        print(f"[Bridge] Total group size: {self._total_group_size}")
        
        # Create Gloo group (for coordination)
        print("[Bridge] Creating Gloo process group...")
        self.gloo_group = init_process_group(
            backend="gloo",
            init_method=f"tcp://{master_addr}",
            world_size=self._total_group_size,
            rank=self.config.trainer_rank,
            group_name="gloo_group",
        )
        print("[Bridge] ✓ Gloo group created")
        
        # Create NCCL group (for tensor transfers)
        print("[Bridge] Creating NCCL process group...")
        self.nccl_group = init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_addr}",
            world_size=self._total_group_size,
            rank=self.config.trainer_rank,
            group_name="weight_update_group",
        )
        print("[Bridge] ✓ NCCL group created")
    
    def _initialize_http_mode(self) -> None:
        """Initialize HTTP-based weight synchronization (fallback)."""
        print("[Bridge] Using HTTP mode for weight synchronization")
        print(f"[Bridge] vLLM API URL: {self.config.vllm_api_url}")
        
        # Verify vLLM server is reachable
        try:
            import requests
            response = requests.get(f"{self.config.vllm_api_url}/health", timeout=5)
            if response.status_code == 200:
                print("[Bridge] ✓ vLLM server is reachable")
            else:
                print(f"[Bridge] Warning: vLLM health check returned {response.status_code}")
        except Exception as e:
            print(f"[Bridge] Warning: Could not reach vLLM: {e}")
    
    def _load_param_mappings(self) -> None:
        """Load parameter name mappings from vLLM's config file."""
        log_dir = self.config.log_dir or os.environ.get("LOGDIR", ".")
        json_path = Path(log_dir) / "vllm_bridge_config.json"
        
        # Wait for file (vLLM needs time to load model and export params)
        wait_time = 0
        max_wait = min(self.config.timeout_seconds, 120)  # Max 2 minutes
        while not json_path.exists() and wait_time < max_wait:
            if wait_time % 10 == 0:
                print(f"[Bridge] Waiting for {json_path}... ({wait_time}s)")
            time.sleep(1)
            wait_time += 1
        
        if not json_path.exists():
            print(f"[Bridge] Warning: Config file not found after {wait_time}s")
            print("[Bridge] Will use trainer's model params directly")
            self.param_mappings = {}
            self.param_name_list = []
            return
        
        time.sleep(1.0)  # Wait for file to finish writing
        
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            
            self.param_mappings = data.get("param_mappings", {})
            self.param_name_list = data.get("param_names", sorted(self.param_mappings.keys()))
            
            print(f"[Bridge] Loaded {len(self.param_name_list)} vLLM parameter names")
        except Exception as e:
            print(f"[Bridge] Warning: Failed to load config: {e}")
            self.param_mappings = {}
            self.param_name_list = []
    
    def set_param_list_from_model(self, model: nn.Module) -> None:
        """
        Set param list from the trainer's model.
        
        Call this if vLLM's param names don't match the trainer's.
        """
        self.param_name_list = sorted(name for name, _ in model.named_parameters())
        print(f"[Bridge] Using trainer's {len(self.param_name_list)} parameter names")
    
    def broadcast_weights(self, model: nn.Module) -> None:
        """
        Broadcast all model weights to vLLM inference workers.
        
        Call this after optimizer.step() to push updated weights.
        
        Args:
            model: The model whose weights to broadcast
        """
        if not self._initialized:
            raise RuntimeError("Bridge not initialized. Call initialize() first.")
        
        if self.nccl_group is None:
            # HTTP mode - just notify
            self._notify_update_http()
            return
        
        self._update_count += 1
        start_time = time.time()
        
        state_dict = dict(model.named_parameters())
        num_params = 0
        
        with torch.no_grad():
            for idx, param_name in enumerate(self.param_name_list):
                # Get tensor for this parameter
                if param_name not in state_dict:
                    continue
                
                tensor = state_dict[param_name].data
                
                # Step 1: Broadcast parameter index
                idx_tensor = torch.tensor([idx], dtype=torch.long, device=self.device)
                dist.broadcast(idx_tensor, src=0, group=self.nccl_group)
                
                # Step 2: Broadcast the actual tensor
                dist.broadcast(tensor.contiguous(), src=0, group=self.nccl_group)
                
                num_params += 1
        
        elapsed = time.time() - start_time
        print(f"[Bridge] Broadcast {num_params} params, update #{self._update_count} ({elapsed:.2f}s)")
    
    def broadcast_single_param(
        self, 
        model: nn.Module, 
        param_name: str
    ) -> None:
        """
        Broadcast a single parameter to vLLM.
        
        Useful for incremental updates or debugging.
        """
        if self.nccl_group is None:
            return
        
        if param_name not in self.param_name_list:
            print(f"[Bridge] Warning: {param_name} not in param list")
            return
        
        idx = self.param_name_list.index(param_name)
        state_dict = dict(model.named_parameters())
        
        if param_name not in state_dict:
            return
        
        with torch.no_grad():
            idx_tensor = torch.tensor([idx], dtype=torch.long, device=self.device)
            dist.broadcast(idx_tensor, src=0, group=self.nccl_group)
            
            tensor = state_dict[param_name].data
            local_shape = self.param_mappings[param_name].get(
                "local_shape", list(tensor.shape)
            )
            
            tensor_list = [
                torch.zeros(local_shape, dtype=tensor.dtype, device=self.device)
                for _ in range(self._total_group_size)
            ]
            dist.all_gather(tensor_list, tensor, group=self.nccl_group)
    
    def notify_update(self) -> None:
        """
        Notify vLLM that weights have been updated.
        
        In NCCL mode, this is a no-op (updates are immediate).
        In HTTP mode, sends a notification to vLLM.
        """
        self._update_count += 1
        
        if self.nccl_group is None:
            self._notify_update_http()
    
    def _notify_update_http(self) -> None:
        """Notify vLLM via HTTP (fallback mode)."""
        try:
            import requests
            response = requests.post(
                f"{self.config.vllm_api_url}/bridge/notify_update",
                json={
                    "update_count": self._update_count,
                    "trainer_rank": self.config.trainer_rank,
                    "timestamp": time.time(),
                },
                timeout=5,
            )
            if response.status_code != 200:
                print(f"[Bridge] Warning: notify_update returned {response.status_code}")
        except Exception as e:
            print(f"[Bridge] Warning: Could not notify vLLM: {e}")
    
    def send_heartbeat(self) -> None:
        """
        Send heartbeat signal to keep inference workers alive.
        
        In NCCL mode, sends -1 as the parameter index to signal
        "no update this round".
        """
        if self.nccl_group is None:
            return
        
        with torch.no_grad():
            idx_tensor = torch.tensor([-1], dtype=torch.long, device=self.device)
            dist.broadcast(idx_tensor, src=0, group=self.nccl_group)
    
    def cleanup(self) -> None:
        """Clean up resources."""
        print("[Bridge] Cleaning up...")
        
        # Send shutdown signal (optional)
        if self.nccl_group is not None:
            try:
                # Send -2 to signal shutdown (if implemented in updater)
                with torch.no_grad():
                    idx_tensor = torch.tensor([-2], dtype=torch.long, device=self.device)
                    dist.broadcast(idx_tensor, src=0, group=self.nccl_group)
            except Exception:
                pass
        
        self._initialized = False
        print("[Bridge] Cleanup complete")


# =============================================================================
# Factory Function
# =============================================================================


def create_bridge_from_training_config(config: Any) -> VLLMWeightBridge:
    """
    Create a VLLMWeightBridge from a TrainingConfig object.
    
    Args:
        config: TrainingConfig with model and distributed settings
        
    Returns:
        Initialized VLLMWeightBridge ready for use
    """
    bridge_config = BridgeConfig.from_training_config(config)
    bridge = VLLMWeightBridge(bridge_config)
    bridge.initialize()
    return bridge


def export_param_mappings(
    model: nn.Module,
    model_name: str,
    tp_degree: int = 1,
    dp_shard_degree: int = 1,
    log_dir: Optional[str] = None,
) -> None:
    """
    Export parameter mappings to JSON for vLLM to read.
    
    Call this from the trainer BEFORE starting vLLM.
    
    Args:
        model: The model being trained
        model_name: HuggingFace model name
        tp_degree: Tensor parallel degree
        dp_shard_degree: Data parallel shard degree (FSDP)
        log_dir: Directory to write config file
    """
    log_dir = log_dir or os.environ.get("LOGDIR", ".")
    json_path = Path(log_dir) / "vllm_bridge_config.json"
    
    param_mappings = {}
    
    for name, param in model.named_parameters():
        param_mappings[name] = {
            "vllm_name": name,  # May need transformation for some models
            "shape": list(param.shape),
            "local_shape": list(param.shape),  # For FSDP, this would be shard shape
            "dtype": str(param.dtype),
            "tp_shard_dim": 0,
            "needs_permute": False,  # Set True for rotary embedding weights
        }
    
    config = {
        "model": model_name,
        "tp_degree": tp_degree,
        "dp_shard_degree": dp_shard_degree,
        "param_mappings": param_mappings,
    }
    
    with open(json_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"[Bridge] Exported param mappings to {json_path}")
