"""
NCCL Weight Bridge for LoRA Training.

Implements torchtitan-style direct NCCL weight transfer between trainer and vLLM.
This eliminates disk I/O for weight synchronization.

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    NCCL Process Group                                │
    │  ┌─────────────────────┐                ┌─────────────────────────┐ │
    │  │   Trainer (rank 0)  │ ──NCCL send──> │   vLLM (rank 1+)        │ │
    │  │   - LoRA weights    │                │   - Receives weights    │ │
    │  │   - broadcast()     │                │   - Updates state_dict  │ │
    │  └─────────────────────┘                └─────────────────────────┘ │
    └─────────────────────────────────────────────────────────────────────┘

Usage (Trainer side):
    bridge = NCCLWeightBridge(
        rank=0,
        world_size=2,
        init_method="tcp://localhost:29500"
    )
    bridge.setup()
    
    # After training step
    bridge.send_lora_weights(model)

Usage (vLLM side):
    bridge = NCCLWeightBridge(
        rank=1,
        world_size=2,
        init_method="tcp://localhost:29500"
    )
    bridge.setup()
    
    # In background thread
    bridge.receive_and_update_weights(vllm_state_dict, param_mappings)
"""

import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist


@dataclass
class NCCLBridgeConfig:
    """Configuration for NCCL weight bridge."""
    
    # Process group settings
    rank: int = 0
    world_size: int = 2
    init_method: str = "tcp://localhost:29500"
    timeout_seconds: int = 120  # Reduced from 300 for faster failure
    
    # LoRA settings
    lora_param_patterns: List[str] = field(default_factory=lambda: [
        "lora_A", "lora_B", "lora_a", "lora_b"
    ])


def is_lora_param(name: str, patterns: Optional[List[str]] = None) -> bool:
    """Check if a parameter name corresponds to a LoRA weight."""
    if patterns is None:
        patterns = ["lora_A", "lora_B", "lora_a", "lora_b", 
                   "lora_a_stacked", "lora_b_stacked"]
    return any(p in name for p in patterns)


def get_lora_params(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Extract LoRA parameters from a model."""
    lora_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad and is_lora_param(name):
            lora_params[name] = param
    return lora_params


class NCCLWeightBridge:
    """
    NCCL-based weight bridge for synchronizing LoRA weights between trainer and vLLM.
    
    Inspired by torchtitan's sglang_handling.py approach.
    """
    
    def __init__(self, config: NCCLBridgeConfig):
        self.config = config
        self.rank = config.rank
        self.world_size = config.world_size
        
        self.nccl_group: Optional[dist.ProcessGroup] = None
        self.gloo_group: Optional[dist.ProcessGroup] = None
        
        self.is_initialized = False
        self.update_count = 0
        self.last_update_time = 0.0
        
        # Parameter registry (filled during first sync)
        self.param_names: List[str] = []
        self.param_shapes: Dict[str, Tuple[int, ...]] = {}
        self.param_dtypes: Dict[str, torch.dtype] = {}
        
        # Receiver state (vLLM side)
        self._receiver_thread: Optional[threading.Thread] = None
        self._stop_receiver = threading.Event()
        self._state_dict_ref: Optional[Dict[str, torch.Tensor]] = None
        self._param_mappings: Dict[str, str] = {}
        
    def setup(self) -> bool:
        """
        Initialize NCCL process group.
        
        Returns:
            True if setup successful, False otherwise.
        """
        if self.is_initialized:
            return True
            
        try:
            # Clean up any existing distributed state
            self._cleanup_env_for_new_group()
            
            timeout = timedelta(seconds=self.config.timeout_seconds)
            
            # Initialize NCCL group for tensor transfers
            print(f"[NCCLBridge] Initializing NCCL group (rank={self.rank}, world={self.world_size})")
            print(f"[NCCLBridge] Init method: {self.config.init_method}")
            self.nccl_group = self._init_process_group(
                backend="nccl",
                init_method=self.config.init_method,
                world_size=self.world_size,
                rank=self.rank,
                group_name="lora_weight_nccl",
                timeout=timeout,
            )
            
            # Note: We skip Gloo group - metadata is passed via HTTP before NCCL setup
            self.gloo_group = None
            
            self.is_initialized = True
            print(f"[NCCLBridge] ✓ Initialized successfully (rank {self.rank})")
            return True
            
        except Exception as e:
            print(f"[NCCLBridge] ✗ Failed to initialize: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _cleanup_env_for_new_group(self):
        """Remove environment variables that interfere with new process groups."""
        # Save and remove torch distributed env vars
        env_vars_to_clear = [
            "LOCAL_RANK", "RANK", "WORLD_SIZE", "GROUP_RANK",
            "GROUP_WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR",
            "MASTER_PORT"
        ]
        self._saved_env = {}
        for var in env_vars_to_clear:
            if var in os.environ:
                self._saved_env[var] = os.environ.pop(var)
    
    def _restore_env(self):
        """Restore saved environment variables."""
        for var, value in getattr(self, '_saved_env', {}).items():
            os.environ[var] = value
    
    def _init_process_group(
        self,
        backend: str,
        init_method: str,
        world_size: int,
        rank: int,
        group_name: str,
        timeout: timedelta,
    ) -> dist.ProcessGroup:
        """
        Initialize a new process group without affecting the global state.
        
        Based on torchtitan's init_process_group implementation.
        """
        from torch.distributed.distributed_c10d import (
            _new_process_group_helper,
            _world,
            Backend,
            default_pg_timeout,
            PrefixStore,
            rendezvous,
        )
        
        backend_obj = Backend(backend)
        
        if timeout is None:
            timeout = default_pg_timeout
            
        # Create rendezvous store
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)
        
        # Use a PrefixStore to avoid key collisions
        store = PrefixStore(group_name, store)
        
        # Handle PyTorch version differences
        pg_options_param_name = (
            "backend_options" if str(torch.__version__) >= "2.6" else "pg_options"
        )
        
        pg, _ = _new_process_group_helper(
            world_size,
            rank,
            [],
            backend_obj,
            store,
            group_name=group_name,
            **{pg_options_param_name: None},
            timeout=timeout,
        )
        
        _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}
        
        return pg
    
    def register_params(self, model: torch.nn.Module) -> Dict[str, Any]:
        """
        Register LoRA parameters from the model.
        
        Returns:
            Dictionary with parameter metadata for vLLM side.
        """
        lora_params = get_lora_params(model)
        
        self.param_names = sorted(lora_params.keys())
        self.param_shapes = {name: tuple(p.shape) for name, p in lora_params.items()}
        self.param_dtypes = {name: p.dtype for name, p in lora_params.items()}
        
        metadata = {
            "param_names": self.param_names,
            "param_shapes": {k: list(v) for k, v in self.param_shapes.items()},
            "param_dtypes": {k: str(v) for k, v in self.param_dtypes.items()},
            "num_params": len(self.param_names),
        }
        
        print(f"[NCCLBridge] Registered {len(self.param_names)} LoRA parameters")
        return metadata
    
    def send_lora_weights(
        self,
        model: torch.nn.Module,
        step: Optional[int] = None,
    ) -> float:
        """
        Send LoRA weights to vLLM via NCCL broadcast.
        
        Args:
            model: Model with LoRA adapters
            step: Optional training step for logging
            
        Returns:
            Time taken for the transfer in seconds.
        """
        if not self.is_initialized:
            raise RuntimeError("NCCLBridge not initialized. Call setup() first.")
        
        if self.rank != 0:
            raise RuntimeError("send_lora_weights() should only be called from rank 0 (trainer)")
        
        start_time = time.time()
        
        # Get LoRA parameters
        lora_params = get_lora_params(model)
        
        if not self.param_names:
            self.register_params(model)
        
        # Send step index first (so receivers know an update is coming)
        step_tensor = torch.tensor([step if step is not None else self.update_count], 
                                   dtype=torch.long, device="cuda")
        dist.broadcast(step_tensor, src=0, group=self.nccl_group)
        
        # Send each parameter
        for name in self.param_names:
            param = lora_params[name]
            # Ensure contiguous for efficient transfer
            param_data = param.detach().contiguous()
            dist.broadcast(param_data, src=0, group=self.nccl_group)
        
        # Send completion signal
        done_tensor = torch.tensor([1], dtype=torch.long, device="cuda")
        dist.broadcast(done_tensor, src=0, group=self.nccl_group)
        
        elapsed = time.time() - start_time
        self.update_count += 1
        self.last_update_time = time.time()
        
        if step is not None:
            print(f"[NCCLBridge] Sent LoRA weights (step {step}) in {elapsed:.3f}s")
        
        return elapsed
    
    def receive_lora_weights(
        self,
        on_receive: Optional[Callable[[int, Dict[str, torch.Tensor]], None]] = None,
    ) -> Tuple[int, Dict[str, torch.Tensor]]:
        """
        Receive LoRA weights from trainer via NCCL broadcast.
        
        This is a BLOCKING call that waits for the trainer to send weights.
        For non-blocking continuous receive, use start_receiver().
        
        Args:
            on_receive: Optional callback with (step, weights_dict)
            
        Returns:
            Tuple of (step_number, dict of param_name -> tensor)
        """
        if not self.is_initialized:
            raise RuntimeError("NCCLBridge not initialized. Call setup() first.")
        
        if self.rank == 0:
            raise RuntimeError("receive_lora_weights() should only be called from rank > 0 (vLLM)")
        
        device = "cuda"
        
        # Receive step index
        step_tensor = torch.zeros(1, dtype=torch.long, device=device)
        dist.broadcast(step_tensor, src=0, group=self.nccl_group)
        step = step_tensor.item()
        
        if step < 0:
            # Negative step means shutdown signal
            return step, {}
        
        # Receive each parameter
        received_weights = {}
        for name in self.param_names:
            shape = self.param_shapes[name]
            dtype_str = self.param_dtypes[name]
            # Handle dtype string conversion
            if isinstance(dtype_str, str):
                dtype = getattr(torch, dtype_str.replace("torch.", ""))
            else:
                dtype = dtype_str
            
            # Create buffer and receive
            buffer = torch.zeros(shape, dtype=dtype, device=device)
            dist.broadcast(buffer, src=0, group=self.nccl_group)
            received_weights[name] = buffer
        
        # Receive completion signal
        done_tensor = torch.zeros(1, dtype=torch.long, device=device)
        dist.broadcast(done_tensor, src=0, group=self.nccl_group)
        
        self.update_count += 1
        self.last_update_time = time.time()
        
        print(f"[NCCLBridge] Received LoRA weights (step {step})")
        
        if on_receive:
            on_receive(step, received_weights)
        
        return step, received_weights
    
    def start_receiver(
        self,
        state_dict: Dict[str, torch.Tensor],
        param_mappings: Dict[str, str],
        on_update: Optional[Callable[[int], None]] = None,
    ):
        """
        Start background thread to receive weight updates (vLLM side).
        
        Args:
            state_dict: vLLM's model state dict (will be updated in-place)
            param_mappings: Mapping from trainer param names to vLLM param names
            on_update: Optional callback called after each update with step number
        """
        if self.rank == 0:
            raise RuntimeError("start_receiver() should not be called from rank 0 (trainer)")
        
        self._state_dict_ref = state_dict
        self._param_mappings = param_mappings
        self._stop_receiver.clear()
        
        def receiver_loop():
            print(f"[NCCLBridge] Receiver thread started (rank {self.rank})")
            device = "cuda"
            
            while not self._stop_receiver.is_set():
                try:
                    # Wait for step index
                    step_tensor = torch.zeros(1, dtype=torch.long, device=device)
                    dist.broadcast(step_tensor, src=0, group=self.nccl_group)
                    step = step_tensor.item()
                    
                    if step < 0:
                        # Negative step means shutdown signal
                        print("[NCCLBridge] Received shutdown signal")
                        break
                    
                    # Receive each parameter
                    for name in self.param_names:
                        shape = self.param_shapes[name]
                        dtype_str = self.param_dtypes[name]
                        dtype = getattr(torch, str(dtype_str).replace("torch.", ""))
                        
                        # Create buffer and receive
                        buffer = torch.zeros(shape, dtype=dtype, device=device)
                        dist.broadcast(buffer, src=0, group=self.nccl_group)
                        
                        # Map to vLLM param name and update
                        vllm_name = self._param_mappings.get(name, name)
                        if vllm_name in self._state_dict_ref:
                            # Reshape if needed for vLLM stacked format
                            target = self._state_dict_ref[vllm_name]
                            if buffer.shape != target.shape:
                                buffer = self._reshape_for_vllm(buffer, vllm_name, target.shape)
                            
                            target.data.copy_(buffer)
                    
                    # Wait for completion signal
                    done_tensor = torch.zeros(1, dtype=torch.long, device=device)
                    dist.broadcast(done_tensor, src=0, group=self.nccl_group)
                    
                    self.update_count += 1
                    self.last_update_time = time.time()
                    
                    print(f"[NCCLBridge] Received weight update (step {step})")
                    
                    if on_update:
                        on_update(step)
                        
                except Exception as e:
                    if not self._stop_receiver.is_set():
                        print(f"[NCCLBridge] Receiver error: {e}")
                        import traceback
                        traceback.print_exc()
                    break
            
            print("[NCCLBridge] Receiver thread exiting")
        
        self._receiver_thread = threading.Thread(target=receiver_loop, daemon=True)
        self._receiver_thread.start()
    
    def stop_receiver(self):
        """Stop the receiver thread."""
        self._stop_receiver.set()
        
        # Send shutdown signal if we're the trainer
        if self.rank == 0 and self.is_initialized:
            try:
                shutdown_tensor = torch.tensor([-1], dtype=torch.long, device="cuda")
                dist.broadcast(shutdown_tensor, src=0, group=self.nccl_group)
            except Exception:
                pass
        
        if self._receiver_thread and self._receiver_thread.is_alive():
            self._receiver_thread.join(timeout=5.0)
    
    def _reshape_for_vllm(
        self,
        tensor: torch.Tensor,
        vllm_name: str,
        target_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Reshape LoRA tensor from PEFT format to vLLM stacked format.
        
        vLLM expects:
        - Attention LoRA: [1, 1, rank, dim] or [1, 1, dim, rank]
        - MoE LoRA: [num_experts, rank, dim]
        """
        # Check if this is an attention LoRA (needs [1, 1, ...] prefix)
        is_attention_lora = any(
            proj in vllm_name
            for proj in ["qkv_proj", "o_proj", "q_proj", "k_proj", "v_proj"]
        )
        
        if is_attention_lora and len(tensor.shape) == 2 and len(target_shape) == 4:
            # [rank, dim] -> [1, 1, rank, dim]
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        
        if tensor.shape != target_shape:
            raise ValueError(
                f"Shape mismatch for {vllm_name}: got {tensor.shape}, expected {target_shape}"
            )
        
        return tensor
    
    def cleanup(self):
        """Clean up process groups and threads."""
        self.stop_receiver()
        self._restore_env()
        
        # Note: We don't destroy the process groups as they may still be in use
        # by other parts of the system. They will be cleaned up on process exit.
        
        self.is_initialized = False
        print("[NCCLBridge] Cleaned up")


def create_trainer_param_to_vllm_mapping(
    trainer_param_names: List[str],
    model_name: str = "llama",
) -> Dict[str, str]:
    """
    Create mapping from PEFT trainer parameter names to vLLM parameter names.
    
    PEFT names: base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
    vLLM names: model.layers.0.self_attn.qkv_proj.q_proj.lora_a_stacked
    
    Args:
        trainer_param_names: List of parameter names from the trainer model
        model_name: Model architecture name for architecture-specific mappings
        
    Returns:
        Dictionary mapping trainer names to vLLM names
    """
    mapping = {}
    
    for name in trainer_param_names:
        if not is_lora_param(name):
            continue
            
        # Remove PEFT prefixes
        vllm_name = name
        for prefix in ["base_model.model.", "base_model."]:
            if vllm_name.startswith(prefix):
                vllm_name = vllm_name[len(prefix):]
        
        # Handle attention projections (qkv fusion)
        for proj in ["q_proj", "k_proj", "v_proj"]:
            if f".{proj}.lora_" in vllm_name:
                # Map to vLLM's fused qkv_proj format
                if ".lora_A" in vllm_name or ".lora_a" in vllm_name:
                    suffix = "lora_a_stacked"
                else:
                    suffix = "lora_b_stacked"
                    
                # self_attn.q_proj.lora_A.weight -> self_attn.qkv_proj.q_proj.lora_a_stacked
                vllm_name = vllm_name.replace(f".{proj}.lora_A.weight", f".qkv_proj.{proj}.{suffix}")
                vllm_name = vllm_name.replace(f".{proj}.lora_B.weight", f".qkv_proj.{proj}.{suffix}")
                vllm_name = vllm_name.replace(f".{proj}.lora_a.weight", f".qkv_proj.{proj}.{suffix}")
                vllm_name = vllm_name.replace(f".{proj}.lora_b.weight", f".qkv_proj.{proj}.{suffix}")
                break
        
        # Handle o_proj
        if ".o_proj.lora_" in vllm_name:
            suffix = "lora_a_stacked" if ("lora_A" in vllm_name or "lora_a" in vllm_name) else "lora_b_stacked"
            vllm_name = vllm_name.replace(".o_proj.lora_A.weight", f".o_proj.o_proj.{suffix}")
            vllm_name = vllm_name.replace(".o_proj.lora_B.weight", f".o_proj.o_proj.{suffix}")
            vllm_name = vllm_name.replace(".o_proj.lora_a.weight", f".o_proj.o_proj.{suffix}")
            vllm_name = vllm_name.replace(".o_proj.lora_b.weight", f".o_proj.o_proj.{suffix}")
        
        # Handle MLP projections
        for mlp_proj in ["gate_proj", "up_proj", "down_proj"]:
            if f".{mlp_proj}.lora_" in vllm_name:
                suffix = "lora_a_stacked" if ("lora_A" in vllm_name or "lora_a" in vllm_name) else "lora_b_stacked"
                vllm_name = vllm_name.replace(f".{mlp_proj}.lora_A.weight", f".{mlp_proj}.{suffix}")
                vllm_name = vllm_name.replace(f".{mlp_proj}.lora_B.weight", f".{mlp_proj}.{suffix}")
                vllm_name = vllm_name.replace(f".{mlp_proj}.lora_a.weight", f".{mlp_proj}.{suffix}")
                vllm_name = vllm_name.replace(f".{mlp_proj}.lora_b.weight", f".{mlp_proj}.{suffix}")
                break
        
        mapping[name] = vllm_name
    
    return mapping


def export_bridge_config(
    config_path: str,
    param_metadata: Dict[str, Any],
    param_mappings: Dict[str, str],
    nccl_init_method: str,
    world_size: int = 2,
):
    """
    Export bridge configuration to JSON for vLLM to read.
    
    Args:
        config_path: Path to write the config
        param_metadata: Parameter metadata from register_params()
        param_mappings: Trainer to vLLM parameter name mappings
        nccl_init_method: NCCL init method (e.g., "tcp://localhost:29500")
        world_size: Total number of processes in the group
    """
    config = {
        "nccl_enabled": True,
        "nccl_init_method": nccl_init_method,
        "world_size": world_size,
        "param_metadata": param_metadata,
        "param_mappings": param_mappings,
    }
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"[NCCLBridge] Exported config to {config_path}")
