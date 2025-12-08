"""
vLLM Weight Bridge - Integration between trainer and vLLM inference.

This module provides two modes for coordinating weight updates:

LOCAL MODE (num_inference_nodes=0):
    - Trainer and vLLM run as separate processes on the same machine
    - Communication via HTTP to vLLM's /bridge/* endpoints
    - No NCCL process groups needed
    - Simpler setup, suitable for single-machine training

DISTRIBUTED MODE (num_inference_nodes>0):
    - Trainer and vLLM join the same NCCL process group
    - Direct tensor sharing via shared GPU memory
    - Lower latency, but requires coordinated setup

Architecture (Local Mode):
    ┌─────────────────┐         ┌─────────────────┐
    │ Trainer Process │  HTTP   │  vLLM Process   │
    │ (training)      │────────▶│  (inference)    │
    └─────────────────┘         └─────────────────┘

Architecture (Distributed Mode):
    ┌─────────────────────────────────────────┐
    │        Shared GPU Memory (NCCL)         │
    │   Model weights owned by vLLM process   │
    └─────────────────────────────────────────┘
            ▲                       ▲
            │ forward pass          │ optimizer.step()
    ┌───────┴───────┐       ┌───────┴───────┐
    │ vLLM Process  │       │Trainer Process│
    └───────────────┘       └───────────────┘
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
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM


# =============================================================================
# Process Group Initialization Helpers
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

    This is based on torch.distributed internals but allows creating a named
    group that coexists with the default process group (used by vLLM internally).

    Args:
        backend: "nccl" for GPU, "gloo" for CPU
        init_method: Rendezvous URL (e.g., "tcp://host:port" or "env://")
        timeout: How long to wait for other ranks
        world_size: Total number of processes in the group
        rank: This process's rank in the group
        store: Optional torch.distributed Store object
        group_name: Name for this process group (must match across all ranks)
        pg_options: Backend-specific options

    Returns:
        A ProcessGroup object for collective operations
    """
    from torch.distributed.distributed_c10d import (
        _new_process_group_helper,
        _world,
        Backend,
        default_pg_timeout,
        PrefixStore,
        rendezvous,
    )

    assert (store is None) or (
        init_method is None
    ), "Cannot specify both init_method and store."

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

    # Rendezvous with other processes
    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)
        # Use a PrefixStore to avoid key collisions with other groups
        store = PrefixStore(group_name, store)

    # PyTorch 2.6+ renamed pg_options to backend_options
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
    src: int,
    group: dist.ProcessGroup,
    device: Optional[torch.device] = None,
) -> None:
    """
    Broadcast a list of picklable objects from src rank to all other ranks.

    This is a simplified version of torch.distributed.broadcast_object_list
    that works correctly with custom process groups.

    Args:
        object_list: List of objects to broadcast (modified in-place on receivers)
        src: Source rank that has the data
        group: Process group to use
        device: Device for intermediate tensors
    """
    current_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Broadcast object sizes first
    object_sizes_tensor = torch.empty(
        len(object_list), dtype=torch.long, device=current_device
    )
    dist.broadcast(object_sizes_tensor, src=src, group=group)

    # Broadcast serialized objects
    object_tensor = torch.empty(
        torch.sum(object_sizes_tensor).item(),
        dtype=torch.uint8,
        device=current_device,
    )
    dist.broadcast(object_tensor, src=src, group=group)

    # Deserialize on receiving ranks
    offset = 0
    for i, obj_size in enumerate(object_sizes_tensor):
        obj_view = object_tensor[offset : offset + obj_size]
        obj_view = obj_view.type(torch.uint8)
        offset += obj_size
        object_list[i] = dist._tensor_to_object(obj_view, obj_size, group)


# =============================================================================
# Environment and URL Helpers
# =============================================================================


def get_inference_urls(num_inference_nodes: int) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[List[str]]]:
    """
    Get rendezvous URLs for connecting to inference nodes.

    In SLURM environments, parses SLURM_JOB_NODELIST to find inference servers.
    For local testing, returns localhost URLs.

    Args:
        num_inference_nodes: Number of inference nodes (from config)

    Returns:
        Tuple of (master_server, master_gloo_server, master_inference_server, nodelist)
        All None if inference nodes not configured.
    """
    if num_inference_nodes > 0:
        # Multi-node SLURM environment
        nodelist_raw = os.popen(
            f'scontrol show hostnames {os.environ.get("SLURM_JOB_NODELIST", "")}'
        ).read()
        nodelist = [n for n in nodelist_raw.split("\n") if n]

        if not nodelist:
            return None, None, None, None

        master_server = f"{nodelist[0]}:26756"
        master_gloo_server = f"{nodelist[0]}:26757"
        # Inference nodes are the last N nodes
        inference_nodes = nodelist[-num_inference_nodes:]
        master_inference_server = f"{inference_nodes[0]}:26758"

        return master_server, master_gloo_server, master_inference_server, inference_nodes

    elif num_inference_nodes == 0:
        # Single-node local mode
        return "localhost:26756", "localhost:26757", "localhost:26758", ["localhost"]

    else:
        return None, None, None, None


def get_local_hostname() -> Optional[List[str]]:
    """Get the local hostname(s) from /etc/hosts for rank determination."""
    my_ip = socket.gethostbyname(socket.gethostname())
    my_hostname = socket.gethostname()

    try:
        with open("/etc/hosts", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split()
                    if len(parts) >= 2 and (parts[0] == my_ip or my_hostname in parts):
                        ip = parts[0]
                        if ip.startswith("127."):
                            continue
                        return parts
    except FileNotFoundError:
        pass

    return [my_ip, my_hostname]


# =============================================================================
# Tensor Mapping and Permutation Helpers
# =============================================================================


def permute_for_rotary(w: torch.Tensor, n_heads: int) -> torch.Tensor:
    """
    Permute weight tensor for sliced rotary embeddings.

    vLLM and some model implementations use different layouts for Q/K projections.
    This converts between them.
    """
    dim1, dim2 = w.shape[0], w.shape[1]
    return (
        w.view(n_heads, dim1 // n_heads // 2, 2, dim2)
        .transpose(1, 2)
        .reshape(dim1, dim2)
    )


def permute_for_rotary_1d(w: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Permute 1D tensor (bias) for sliced rotary embeddings."""
    dim1 = w.shape[0]
    return w.view(n_heads, dim1 // n_heads // 2, 2).transpose(1, 2).reshape(dim1)


def get_name_conversions(param_mappings: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Build a mapping from vLLM parameter names to trainer parameter names.

    vLLM may split or combine parameters differently than HuggingFace models.
    This helps translate between naming conventions.
    """
    name_conversions = defaultdict(list)
    for name, info in param_mappings.items():
        vllm_name = info.get("vllm_name", name)
        name_conversions[vllm_name].append(name)
    return dict(name_conversions)


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

    # vLLM server URL for HTTP-based sync (local mode)
    vllm_api_url: str = "http://localhost:9001"

    # Derived from environment
    num_gpus_per_node: int = field(default_factory=lambda: torch.cuda.device_count())

    @property
    def is_local_mode(self) -> bool:
        """
        Local mode: single machine, no NCCL process groups needed.
        Communication happens via HTTP to vLLM server.
        """
        return self.num_inference_nodes == 0

    @classmethod
    def from_training_config(cls, config: Any) -> "BridgeConfig":
        """Create BridgeConfig from a TrainingConfig object."""
        return cls(
            trainer_rank=config.trainer_rank,
            world_size=config.world_size,
            init_method=config.init_method,
            num_inference_nodes=config.num_inference_nodes,
            model_name=config.model_name,
            device=config.device,
            log_dir=os.environ.get("LOGDIR"),
            vllm_api_url=f"http://localhost:{getattr(config, 'vllm_port', 9001)}",
        )


# =============================================================================
# Main Bridge Class
# =============================================================================


class VLLMWeightBridge:
    """
    Bridge for sharing model weights between trainer and vLLM inference server.

    This class handles:
    1. Joining the distributed process group with vLLM workers
    2. Attaching to vLLM's model weight tensors
    3. Providing a model interface for the trainer to optimize
    4. Synchronizing updates so vLLM sees changes immediately

    Usage:
        bridge = VLLMWeightBridge(config)
        bridge.initialize()
        model = bridge.get_trainable_model()
        optimizer = AdamW(model.parameters(), lr=1e-5)

        for batch in data:
            loss = compute_loss(model, batch)
            loss.backward()
            optimizer.step()
            bridge.notify_update()  # vLLM now uses new weights
    """

    def __init__(self, config: BridgeConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Process groups (initialized in initialize())
        self.nccl_group: Optional[dist.ProcessGroup] = None
        self.gloo_group: Optional[dist.ProcessGroup] = None

        # Parameter mappings (loaded from vLLM's JSON)
        self.param_mappings: Dict[str, Any] = {}
        self.name_conversions: Dict[str, List[str]] = {}

        # Shared tensors (attached in attach_to_vllm_weights())
        self.shared_state_dict: Dict[str, torch.Tensor] = {}

        # Model for training (created in get_trainable_model())
        self._model: Optional[nn.Module] = None

        # Synchronization state
        self._update_count: int = 0
        self._initialized: bool = False

    def initialize(self) -> None:
        """
        Initialize the bridge: join process groups and load parameter mappings.

        In local mode (num_inference_nodes=0), skips NCCL setup and uses HTTP.
        In distributed mode, creates NCCL/Gloo process groups.

        This must be called before any other methods.
        """
        if self._initialized:
            return

        print(f"[Bridge] Initializing weight bridge for rank {self.config.trainer_rank}")

        if self.config.is_local_mode:
            self._initialize_local_mode()
        else:
            self._initialize_distributed_mode()

        self._initialized = True

    def _initialize_local_mode(self) -> None:
        """
        Initialize for local single-machine mode.

        In local mode:
        - No NCCL process groups (trainer and vLLM are separate processes)
        - Communication via HTTP to vLLM's bridge endpoints
        - Trainer loads its own model copy, updates are synced via checkpoints
        """
        print("[Bridge] Using LOCAL MODE (HTTP-based sync, no NCCL)")
        print(f"[Bridge] vLLM API URL: {self.config.vllm_api_url}")

        # Verify vLLM server is reachable
        try:
            import requests
            response = requests.get(f"{self.config.vllm_api_url}/health", timeout=5)
            if response.status_code == 200:
                print("[Bridge] vLLM server is reachable")
            else:
                print(f"[Bridge] Warning: vLLM health check returned {response.status_code}")
        except Exception as e:
            print(f"[Bridge] Warning: Could not reach vLLM server: {e}")
            print("[Bridge] Training will continue, but vLLM sync may not work")

        # Load parameter mappings if available (optional in local mode)
        try:
            self._load_param_mappings()
        except RuntimeError:
            print("[Bridge] Parameter mapping file not found (optional in local mode)")
            self.param_mappings = {}

    def _initialize_distributed_mode(self) -> None:
        """
        Initialize for distributed multi-node mode.

        Creates NCCL and Gloo process groups for direct tensor sharing.
        """
        print("[Bridge] Using DISTRIBUTED MODE (NCCL tensor sharing)")

        # Get rendezvous URLs
        master_addr, master_gloo_addr, master_inference_addr, nodelist = get_inference_urls(
            self.config.num_inference_nodes
        )

        if master_addr is None:
            raise RuntimeError(
                "Could not determine inference server URLs. "
                "Set NUM_INFERENCE_NODES environment variable or check SLURM_JOB_NODELIST."
            )

        print(f"[Bridge] Master address: {master_addr}")
        print(f"[Bridge] Inference nodes: {nodelist}")

        # Load parameter mappings from vLLM
        self._load_param_mappings()

        # Calculate total group size (trainers + inference workers)
        num_training_gpus = self._get_num_training_gpus()
        # In distributed mode, each inference node contributes num_gpus_per_node workers
        num_inference_gpus = self.config.num_inference_nodes * self.config.num_gpus_per_node

        total_group_size = num_training_gpus + num_inference_gpus
        trainer_rank_in_group = self.config.trainer_rank

        print(f"[Bridge] Training GPUs: {num_training_gpus}, Inference GPUs: {num_inference_gpus}")
        print(f"[Bridge] Total group size: {total_group_size}, Trainer rank: {trainer_rank_in_group}")

        # Initialize NCCL group for tensor transfers
        self.nccl_group = init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_addr}",
            world_size=total_group_size,
            rank=trainer_rank_in_group,
            group_name="weight_update_group",
            timeout=timedelta(seconds=self.config.timeout_seconds),
        )
        print("[Bridge] NCCL process group initialized")

        # Initialize Gloo group for metadata/coordination
        self.gloo_group = init_process_group(
            backend="gloo",
            init_method=f"tcp://{master_gloo_addr}",
            world_size=total_group_size,
            rank=trainer_rank_in_group,
            group_name="gloo_group",
            timeout=timedelta(seconds=self.config.timeout_seconds),
        )
        print("[Bridge] Gloo process group initialized")

    def _load_param_mappings(self) -> None:
        """Load parameter name mappings from vLLM's exported JSON."""
        log_dir = self.config.log_dir or os.environ.get("LOGDIR", ".")
        json_path = Path(log_dir) / "vllm_bridge_config.json"

        # Wait for vLLM to write the mapping file
        wait_time = 0
        while not json_path.exists() and wait_time < self.config.timeout_seconds:
            print(f"[Bridge] Waiting for {json_path} to be created...")
            time.sleep(1)
            wait_time += 1

        if not json_path.exists():
            raise RuntimeError(
                f"Parameter mapping file not found at {json_path}. "
                "Make sure vLLM is running and has exported its parameter mappings."
            )

        # Small delay to ensure file is fully written
        time.sleep(1)

        with open(json_path, "r") as f:
            data = json.load(f)

        self.param_mappings = data.get("param_mappings", {})
        self.name_conversions = get_name_conversions(self.param_mappings)

        print(f"[Bridge] Loaded mappings for {len(self.param_mappings)} parameters")

    def _get_num_training_gpus(self) -> int:
        """Get number of training GPUs from param mappings or config."""
        if self.param_mappings:
            # Try to get from vLLM's exported info
            return self.param_mappings.get("dp_shard_degree", 1) * self.param_mappings.get("tp_degree", 1)
        return self.config.world_size

    def attach_to_vllm_weights(self, vllm_state_dict: Dict[str, torch.Tensor]) -> None:
        """
        Attach to vLLM's weight tensors.

        After this call, self.shared_state_dict contains references to the
        actual tensors that vLLM uses for inference. Modifying these tensors
        will immediately affect vLLM's outputs.

        Args:
            vllm_state_dict: vLLM's model state_dict (actual tensors, not copies)
        """
        self.shared_state_dict = vllm_state_dict
        print(f"[Bridge] Attached to {len(vllm_state_dict)} vLLM weight tensors")

        # Log tensor info for debugging
        for name, tensor in list(vllm_state_dict.items())[:5]:
            print(f"[Bridge]   {name}: {tensor.shape}, {tensor.dtype}, {tensor.device}")
        if len(vllm_state_dict) > 5:
            print(f"[Bridge]   ... and {len(vllm_state_dict) - 5} more")

    def get_trainable_model(self) -> nn.Module:
        """
        Get a model whose parameters point to vLLM's shared tensors.

        This creates a HuggingFace model structure but replaces all parameters
        with references to the shared tensors. When the optimizer updates these
        parameters, it modifies vLLM's weights directly.

        Returns:
            An nn.Module ready for training with shared weights
        """
        if self._model is not None:
            return self._model

        if not self.shared_state_dict:
            raise RuntimeError(
                "Must call attach_to_vllm_weights() before get_trainable_model()"
            )

        print(f"[Bridge] Creating trainable model for {self.config.model_name}")

        # Load model config (not weights)
        model_config = AutoConfig.from_pretrained(self.config.model_name)

        # Create model with empty weights
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(model_config)

        # Replace each parameter with the shared tensor
        self._replace_parameters_with_shared(model)

        model.to(self.device)
        self._model = model

        print(f"[Bridge] Trainable model ready with {sum(p.numel() for p in model.parameters())} parameters")
        return model

    def _replace_parameters_with_shared(self, model: nn.Module) -> None:
        """
        Replace model parameters with references to shared vLLM tensors.

        This is the key operation that makes weight sharing work. After this,
        model.parameters() returns tensors that ARE vLLM's weights.
        """
        replaced_count = 0
        missing_params = []

        for name, param in model.named_parameters():
            # Convert HuggingFace param name to vLLM param name
            vllm_name = self._hf_to_vllm_name(name)

            if vllm_name in self.shared_state_dict:
                shared_tensor = self.shared_state_dict[vllm_name]

                # Create a new Parameter that wraps the shared tensor
                # The key is that we're not copying - we're referencing the same storage
                new_param = nn.Parameter(shared_tensor, requires_grad=True)

                # Replace the parameter in the model
                self._set_parameter(model, name, new_param)
                replaced_count += 1
            else:
                missing_params.append(name)

        print(f"[Bridge] Replaced {replaced_count} parameters with shared tensors")
        if missing_params:
            print(f"[Bridge] Warning: {len(missing_params)} parameters not found in shared state:")
            for p in missing_params[:5]:
                print(f"[Bridge]   {p}")

    def _hf_to_vllm_name(self, hf_name: str) -> str:
        """
        Convert a HuggingFace parameter name to vLLM's naming convention.

        vLLM may merge QKV projections, use different layer naming, etc.
        This handles the translation.
        """
        # Check if we have an explicit mapping
        for vllm_name, hf_names in self.name_conversions.items():
            if hf_name in hf_names:
                return vllm_name

        # Common transformations
        # vLLM often uses: model.layers.N.self_attn.qkv_proj
        # HF uses: model.layers.N.self_attn.q_proj, k_proj, v_proj

        # For now, try the name as-is
        return hf_name

    def _set_parameter(self, model: nn.Module, name: str, new_param: nn.Parameter) -> None:
        """Set a parameter by dotted name path."""
        parts = name.split(".")
        module = model
        for part in parts[:-1]:
            module = getattr(module, part)
        setattr(module, parts[-1], new_param)

    def broadcast_weights_to_inference(self) -> None:
        """
        Broadcast updated weights from trainer to inference workers.

        Call this after optimizer.step() to push the new weights to all
        vLLM inference processes. They will use the updated weights for
        subsequent requests.
        """
        if not self._initialized:
            raise RuntimeError("Bridge not initialized. Call initialize() first.")

        param_names = sorted(self.param_mappings.keys())

        with torch.no_grad():
            for idx, param_name in enumerate(param_names):
                # Signal which parameter we're broadcasting
                idx_tensor = torch.tensor([idx], dtype=torch.long, device=self.device)
                dist.broadcast(idx_tensor, src=0, group=self.nccl_group)

                # Get the tensor for this parameter
                vllm_name = self.param_mappings[param_name].get("vllm_name", param_name)
                if vllm_name not in self.shared_state_dict:
                    continue

                tensor = self.shared_state_dict[vllm_name]
                local_shape = self.param_mappings[param_name].get("local_shape", list(tensor.shape))

                # Gather from all training ranks, then broadcast to inference
                # (This handles FSDP/TP sharding if present)
                dist.all_gather(
                    [torch.zeros(local_shape, dtype=tensor.dtype, device=self.device)
                     for _ in range(dist.get_world_size(self.nccl_group))],
                    tensor,
                    group=self.nccl_group,
                )

        self._update_count += 1
        print(f"[Bridge] Broadcast update #{self._update_count} complete")

    def notify_update(self) -> None:
        """
        Notify inference workers that weights have been updated.

        In local mode: sends HTTP request to vLLM's /bridge/notify_update endpoint
        In distributed mode: broadcasts update counter via Gloo
        """
        self._update_count += 1

        if self.config.is_local_mode:
            self._notify_update_http()
        elif self.gloo_group is not None:
            self._notify_update_distributed()

    def _notify_update_http(self) -> None:
        """Notify vLLM via HTTP (local mode)."""
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
            # Don't fail training if vLLM notification fails
            print(f"[Bridge] Warning: Could not notify vLLM: {e}")

    def _notify_update_distributed(self) -> None:
        """Notify via Gloo broadcast (distributed mode)."""
        update_tensor = torch.tensor([self._update_count], dtype=torch.long)
        dist.broadcast(update_tensor, src=0, group=self.gloo_group)

    def barrier(self) -> None:
        """Wait for all processes in the group to reach this point."""
        if self.nccl_group is not None:
            dist.barrier(group=self.nccl_group)

    def cleanup(self) -> None:
        """Clean up process groups and resources."""
        if self.nccl_group is not None:
            dist.destroy_process_group(self.nccl_group)
            self.nccl_group = None

        if self.gloo_group is not None:
            dist.destroy_process_group(self.gloo_group)
            self.gloo_group = None

        self._initialized = False
        print("[Bridge] Cleaned up")


# =============================================================================
# Convenience Functions
# =============================================================================


def create_bridge_from_training_config(config: Any) -> VLLMWeightBridge:
    """
    Create and initialize a VLLMWeightBridge from a TrainingConfig.

    Args:
        config: TrainingConfig object with bridge settings

    Returns:
        Initialized VLLMWeightBridge ready for use
    """
    bridge_config = BridgeConfig.from_training_config(config)
    bridge = VLLMWeightBridge(bridge_config)
    bridge.initialize()
    return bridge

