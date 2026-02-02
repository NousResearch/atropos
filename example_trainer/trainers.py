"""
Training mode implementations for GRPO trainer.

Contains the three main training modes:
- train_legacy: Checkpoint-based training with vLLM restarts
- train_shared_vllm: Single-copy mode with CUDA IPC
- train_lora: LoRA adapter training with hot-swap
"""

import os
import time
from typing import Optional

import requests
import torch
from torch.optim import AdamW

from .api import check_atropos_api, register_trainer


class CPUOffloadAdamW(torch.optim.Optimizer):
    """
    AdamW with optimizer states offloaded to CPU.
    
    Full precision (no quantization), but states stay on CPU RAM instead of GPU.
    Trade-off: Slower (~2x) but uses ~0GB GPU memory for optimizer states.
    """
    def __init__(self, params, lr=1e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def _init_state(self, p):
        """Lazily initialize state on CPU."""
        state = self.state[p]
        if len(state) == 0:
            state['step'] = 0
            # Store on CPU in FP32
            state['exp_avg'] = torch.zeros_like(p, device='cpu', dtype=torch.float32)
            state['exp_avg_sq'] = torch.zeros_like(p, device='cpu', dtype=torch.float32)
        return state
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self._init_state(p)
                
                state['step'] += 1
                
                # Move states to GPU for computation
                exp_avg = state['exp_avg'].to(p.device)
                exp_avg_sq = state['exp_avg_sq'].to(p.device)
                
                # AdamW update
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] / bias_correction1
                
                # Update weights
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                p.addcdiv_(exp_avg, denom, value=-step_size)
                
                # Weight decay
                if group['weight_decay'] != 0:
                    p.add_(p, alpha=-group['lr'] * group['weight_decay'])
                
                # Move states back to CPU (non-blocking for better perf)
                state['exp_avg'].copy_(exp_avg.cpu())
                state['exp_avg_sq'].copy_(exp_avg_sq.cpu())
        
        return loss


def create_optimizer(model: torch.nn.Module, config) -> torch.optim.Optimizer:
    """
    Create optimizer based on config.optimizer setting.
    
    Options:
    - 'adamw': Standard AdamW (full precision, ~32GB GPU for 4B model)
    - 'adamw_8bit': 8-bit AdamW from bitsandbytes (~8GB GPU, requires bitsandbytes)
    - 'adamw_cpu': AdamW with CPU offload (~0GB GPU, slower but full precision)
    - 'adafactor': Adafactor without momentum (~8GB GPU, no extra dependencies)
    """
    if config.optimizer == "adamw_8bit":
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=config.lr)
            print(f"[Setup] Using 8-bit AdamW (saves ~24GB optimizer memory)")
            return optimizer
        except ImportError:
            print("[Setup] WARNING: bitsandbytes not installed, falling back to AdamW")
            print("[Setup] Install with: pip install bitsandbytes")
    
    if config.optimizer == "adamw_cpu":
        optimizer = CPUOffloadAdamW(model.parameters(), lr=config.lr)
        print(f"[Setup] Using AdamW with CPU offload (full precision, ~0GB GPU for states)")
        print(f"[Setup] NOTE: ~2x slower due to CPU<->GPU transfers, but no quantization")
        return optimizer
    
    if config.optimizer == "adafactor":
        try:
            from transformers.optimization import Adafactor
            optimizer = Adafactor(
                model.parameters(),
                lr=config.lr,
                scale_parameter=False,
                relative_step=False,
            )
            print(f"[Setup] Using Adafactor (no momentum, saves ~24GB)")
            return optimizer
        except ImportError:
            print("[Setup] WARNING: transformers Adafactor not available, using AdamW")
    
    # Default: standard AdamW
    optimizer = AdamW(model.parameters(), lr=config.lr)
    print(f"[Setup] Using standard AdamW (requires ~32GB for optimizer states)")
    return optimizer


from .checkpointing import save_checkpoint, save_lora_checkpoint
from .config import TrainingConfig
from .data import get_data
from .model import load_model_and_tokenizer, PEFT_AVAILABLE
from .training import (
    finalize_training,
    log_metrics,
    run_training_step,
    setup_wandb,
)
from .vllm_manager import (
    check_vllm_health,
    check_vllm_process_health,
    launch_vllm_server,
    terminate_vllm_process,
    set_vllm_process,
)


def train_legacy(config: TrainingConfig):
    """
    Legacy GRPO training with periodic vLLM restarts.

    This mode:
    1. Trains model on trainer GPU
    2. Saves checkpoints periodically
    3. Restarts vLLM to load new weights

    Use for:
    - Simple setup
    - When trainer and vLLM on different GPUs
    """
    training_start_time = time.time()

    # === Setup ===
    use_wandb = setup_wandb(config)
    model, tokenizer = load_model_and_tokenizer(config)
    optimizer = create_optimizer(model, config)

    print(f"\n{'='*60}")
    print("LEGACY MODE (checkpoint + vLLM restart)")
    print(f"{'='*60}")
    print(f"Training for {config.training_steps} steps on {config.device}")
    print(f"vLLM restart interval: every {config.vllm_restart_interval} steps")
    print(f"Save path: {config.save_path}")
    print(f"{'='*60}\n")

    os.makedirs(config.save_path, exist_ok=True)

    # Check Atropos API
    if not check_atropos_api(url=config.atropos_url, timeout=30):
        raise RuntimeError(f"Atropos API not reachable at {config.atropos_url}")
    register_trainer(config)

    # Launch initial vLLM server
    vllm_proc = launch_vllm_server(config, config.model_name)
    set_vllm_process(vllm_proc)

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

        # Fetch data
        data_fetch_start = time.time()
        if len(batches) == 0:
            batches, _ = get_data(config.batch_size, config.seq_len, config.atropos_url,
                                  extract_inference_logprobs=False)
        token_batches, label_batches, advantage_batches, temperature_batches = batches.pop(0)
        data_fetch_time = time.time() - data_fetch_start
        benchmark_stats["data_fetch_times"].append(data_fetch_time)

        # Check if we should sync (save checkpoint + restart vLLM)
        should_sync = (step + 1) % config.vllm_restart_interval == 0 or step == config.training_steps - 1
        if should_sync:
            terminate_vllm_process()

        # Training step
        step_start = time.time()
        metrics = run_training_step(
            model, optimizer,
            token_batches, label_batches, advantage_batches, temperature_batches,
            config,
        )
        step_time = time.time() - step_start
        benchmark_stats["step_times"].append(step_time)

        # GPU memory tracking
        gpu_mem_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        gpu_mem_reserved_gb = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        benchmark_stats["gpu_memories"].append(gpu_mem_gb)

        # Sync (checkpoint + restart)
        sync_time = 0
        if should_sync:
            sync_start = time.time()
            checkpoint_path = save_checkpoint(model, tokenizer, config.save_path, step + 1)
            torch.cuda.empty_cache()
            vllm_proc = launch_vllm_server(config, checkpoint_path)
            set_vllm_process(vllm_proc)
            sync_time = time.time() - sync_start
            benchmark_stats["sync_times"].append(sync_time)

        # Update metrics
        metrics.update({
            "step_time": step_time,
            "sync_time": sync_time,
            "data_fetch_time": data_fetch_time,
            "gpu_memory_gb": gpu_mem_gb,
            "gpu_memory_reserved_gb": gpu_mem_reserved_gb,
        })

        log_metrics(metrics, step + 1, use_wandb, benchmark=config.benchmark)
        check_vllm_process_health()

    # === Cleanup ===
    save_checkpoint(model, tokenizer, config.save_path, config.training_steps, is_final=True)
    finalize_training(use_wandb, training_start_time, "legacy", config.training_steps, benchmark_stats, config.benchmark)


def train_shared_vllm(config: TrainingConfig):
    """
    GRPO training with shared vLLM weights (single-copy mode).

    This mode:
    1. Attaches to vLLM's weight tensors via CUDA IPC
    2. optimizer.step() modifies vLLM's weights in-place
    3. vLLM immediately uses updated weights (no restart!)

    Requirements:
    - vLLM running with VLLM_ENABLE_SHARED_WEIGHTS=1
    - Trainer on same GPU(s) as vLLM
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
    print(f"Save path: {config.save_path}")
    print(f"{'='*60}\n")

    # Attach to vLLM's shared tensors
    print("[1/2] Attaching to vLLM's shared tensors...")
    model, tokenizer = load_model_and_tokenizer(config, single_copy=True)

    if model is None:
        raise RuntimeError(
            "Single-copy mode failed. Make sure:\n"
            "1. vLLM is running with VLLM_ENABLE_SHARED_WEIGHTS=1\n"
            "2. Trainer is on the SAME GPUs as vLLM\n"
            "3. vllm_bridge_config.json exists with IPC handles"
        )

    optimizer = create_optimizer(model, config)

    # === Real-time weight sharing verification ===
    print("\n[Weight Sharing Verification]")
    
    # First, check if we can modify a weight and see the change
    probe_param = None
    for name, param in model.named_parameters():
        if "layers.0.self_attn.q_proj.weight" in name:
            probe_param = param
            probe_name = name
            break
    
    if probe_param is not None:
        original_val = probe_param.data[0, 0].clone()
        print(f"  Testing tensor: {probe_name}")
        print(f"  Original value [0,0]: {original_val.item():.6f}")
        print(f"  Data pointer: {probe_param.data.data_ptr()}")
        
        # Modify the weight
        probe_param.data[0, 0] = original_val + 0.001
        new_val = probe_param.data[0, 0].item()
        print(f"  After +0.001:  [0,0]: {new_val:.6f}")
        
        # Restore
        probe_param.data[0, 0] = original_val
        restored_val = probe_param.data[0, 0].item()
        print(f"  Restored:      [0,0]: {restored_val:.6f}")
        
        if abs(new_val - original_val.item() - 0.001) < 0.0001:
            print(f"  ✓ Trainer CAN modify the tensor")
        else:
            print(f"  ✗ Modification didn't stick - tensor may be a copy!")
    
    # === CRITICAL TEST: Does vLLM SEE weight modifications? ===
    print(f"\n  [CRITICAL] Testing if vLLM sees weight modifications...")
    try:
        import requests
        
        test_prompt = "2+2="
        vllm_url = f"http://localhost:{config.vllm_port}"
        
        # Get baseline output from vLLM
        response1 = requests.post(
            f"{vllm_url}/generate",
            json={"prompt": test_prompt, "max_tokens": 3, "temperature": 0.0},
            timeout=30,
        )
        baseline_output = response1.json().get("text", [""])[0] if response1.status_code == 200 else "ERROR"
        
        # CORRUPT a weight dramatically (this should break the model)
        embed_param = None
        for name, param in model.named_parameters():
            if "embed_tokens" in name:
                embed_param = param
                break
        
        if embed_param is not None:
            original_embed = embed_param.data[0, :10].clone()
            
            # Corrupt the embedding with extreme values
            embed_param.data[0, :10] = 1000.0
            
            # Query vLLM again - if sharing works, output should be GARBAGE
            response2 = requests.post(
                f"{vllm_url}/generate",
                json={"prompt": test_prompt, "max_tokens": 3, "temperature": 0.0},
                timeout=30,
            )
            corrupted_output = response2.json().get("text", [""])[0] if response2.status_code == 200 else "ERROR"
            
            # Restore the embedding
            embed_param.data[0, :10] = original_embed
            
            # Query vLLM again - should be back to normal
            response3 = requests.post(
                f"{vllm_url}/generate",
                json={"prompt": test_prompt, "max_tokens": 3, "temperature": 0.0},
                timeout=30,
            )
            restored_output = response3.json().get("text", [""])[0] if response3.status_code == 200 else "ERROR"
            
            print(f"    Baseline vLLM output:  '{baseline_output}'")
            print(f"    Corrupted vLLM output: '{corrupted_output}'")
            print(f"    Restored vLLM output:  '{restored_output}'")
            
            # Check if vLLM saw the corruption
            if corrupted_output != baseline_output:
                print(f"  ✓✓✓ vLLM SEES WEIGHT UPDATES! Output changed when weights corrupted.")
                if restored_output == baseline_output:
                    print(f"  ✓✓✓ Output restored after weight restoration. SHARING IS WORKING!")
                else:
                    print(f"  ⚠ Output didn't fully restore - may need vLLM cache clear")
            else:
                print(f"  ✗✗✗ vLLM DID NOT SEE CORRUPTION - SHARING IS BROKEN!")
                print(f"      vLLM may have internal weight copies/cache.")
                print(f"      The IPC attachment gives write access but vLLM doesn't read from it.")
    except Exception as e:
        import traceback
        print(f"  Critical test failed: {e}")
        traceback.print_exc()
    
    # Now test vLLM logprobs vs trainer logprobs
    print(f"\n  Testing logprob alignment with vLLM...")
    try:
        import requests
        
        test_prompt = "The capital of France is"
        test_tokens = tokenizer.encode(test_prompt, return_tensors="pt").to(model.device)
        
        # Get completion from vLLM
        vllm_url = f"http://localhost:{config.vllm_port}"
        response = requests.post(
            f"{vllm_url}/generate",
            json={
                "prompt": test_prompt,
                "max_tokens": 3,
                "temperature": 0.0,  # Greedy for determinism
                "logprobs": 1,
            },
            timeout=30,
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Parse vLLM response - format is [[{token_id: logprob}, ...], ...]
            vllm_logprobs = []
            vllm_tokens = []
            
            logprobs_data = result.get("logprobs", [])
            if logprobs_data and len(logprobs_data) > 0:
                for token_logprob_list in logprobs_data[0]:  # First completion
                    if isinstance(token_logprob_list, dict):
                        # Format: {token_id: logprob}
                        for tid, lp in token_logprob_list.items():
                            vllm_tokens.append(int(tid))
                            vllm_logprobs.append(float(lp))
                            break  # Only first (top) logprob
                    elif isinstance(token_logprob_list, list) and len(token_logprob_list) > 0:
                        # Format: [{token_id: logprob}]
                        item = token_logprob_list[0]
                        if isinstance(item, dict):
                            for tid, lp in item.items():
                                vllm_tokens.append(int(tid))
                                vllm_logprobs.append(float(lp))
                                break
            
            print(f"  vLLM generated: {tokenizer.decode(vllm_tokens) if vllm_tokens else 'N/A'}")
            print(f"  vLLM tokens: {vllm_tokens}")
            print(f"  vLLM logprobs: {vllm_logprobs}")
            
            if vllm_tokens:
                # Compute trainer logprobs for the same sequence
                with torch.no_grad():
                    # Build full sequence: prompt + generated tokens
                    full_seq = list(test_tokens[0].cpu().numpy()) + vllm_tokens
                    full_input = torch.tensor([full_seq[:-1]], device=model.device)  # Input is all but last
                    
                    outputs = model(full_input)
                    logits = outputs.logits[0]  # [seq_len, vocab]
                    
                    # Get logprobs at positions corresponding to generated tokens
                    trainer_logprobs = []
                    prompt_len = test_tokens.shape[1]
                    for i, token_id in enumerate(vllm_tokens):
                        pos = prompt_len - 1 + i  # Position to predict this token
                        if pos < logits.shape[0]:
                            log_probs = torch.log_softmax(logits[pos].float(), dim=-1)
                            trainer_logprobs.append(log_probs[token_id].item())
                    
                    print(f"  Trainer logprobs: {[f'{lp:.4f}' for lp in trainer_logprobs]}")
                    
                    if trainer_logprobs and vllm_logprobs:
                        for i, (vlp, tlp) in enumerate(zip(vllm_logprobs, trainer_logprobs)):
                            diff = abs(vlp - tlp)
                            status = "✓" if diff < 0.25 else "⚠"  # 0.25 threshold accounts for impl differences
                            print(f"    Token {i}: vLLM={vlp:.4f}, Trainer={tlp:.4f}, diff={diff:.4f} {status}")
                        
                        mean_diff = sum(abs(v-t) for v,t in zip(vllm_logprobs, trainer_logprobs)) / len(trainer_logprobs)
                        print(f"  Mean diff: {mean_diff:.4f}")
                        
                        if mean_diff < 0.05:
                            print(f"  ✓ PERFECT ALIGNMENT - weights shared and same compute path")
                        elif mean_diff < 0.25:
                            print(f"  ✓ WEIGHTS ARE SHARED (diff {mean_diff:.2f} is due to different forward pass implementations)")
                            print(f"    vLLM uses Flash Attention, trainer uses HuggingFace - small diff is expected!")
                        else:
                            print(f"  ⚠ Large diff ({mean_diff:.2f}) - may indicate issue with weight sharing")
        else:
            print(f"  vLLM request failed: {response.status_code}")
            
    except Exception as e:
        import traceback
        print(f"  Verification error: {e}")
        traceback.print_exc()
    
    print(f"\n[2/2] Starting training for {config.training_steps} steps")
    print("NOTE: vLLM sees weight updates immediately after each step!")
    print("-" * 60)

    os.makedirs(config.save_path, exist_ok=True)

    # Check Atropos API
    print(f"\n[Setup] Connecting to Atropos API at {config.atropos_url}...")
    if not check_atropos_api(url=config.atropos_url, timeout=30):
        raise RuntimeError(f"Atropos API not reachable at {config.atropos_url}")
    register_trainer(config)

    # === Benchmark tracking ===
    benchmark_stats = {
        "step_times": [],
        "sync_times": [],
        "data_fetch_times": [],
        "gpu_memories": [],
    }

    # === Training Loop ===
    batches = []
    inference_logprobs = None
    for step in range(config.training_steps):
        print(f"\nStep {step+1}/{config.training_steps}")

        # Fetch data (with inference logprobs for alignment check)
        data_fetch_start = time.time()
        if len(batches) == 0:
            batches, inference_logprobs = get_data(
                config.batch_size, config.seq_len, config.atropos_url,
                extract_inference_logprobs=True,  # Enable logprob alignment check
            )
        token_batches, label_batches, advantage_batches, temperature_batches = batches.pop(0)
        data_fetch_time = time.time() - data_fetch_start
        benchmark_stats["data_fetch_times"].append(data_fetch_time)

        # Training step (with logprob alignment check)
        step_start = time.time()
        metrics = run_training_step(
            model, optimizer,
            token_batches, label_batches, advantage_batches, temperature_batches,
            config,
            inference_logprobs=inference_logprobs,  # Pass for alignment validation
        )
        step_time = time.time() - step_start
        benchmark_stats["step_times"].append(step_time)
        
        # Clear inference logprobs after use (will be refreshed with new data)
        inference_logprobs = None

        # GPU memory tracking
        gpu_mem_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        gpu_mem_reserved_gb = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        benchmark_stats["gpu_memories"].append(gpu_mem_gb)

        # In single-copy mode, weights are updated in-place (no sync needed!)
        sync_time = 0.0
        print(f"  [SINGLE-COPY] Weights updated in-place - step {step+1}")
        benchmark_stats["sync_times"].append(sync_time)

        # Update metrics
        metrics.update({
            "step_time": step_time,
            "sync_time": sync_time,
            "data_fetch_time": data_fetch_time,
            "gpu_memory_gb": gpu_mem_gb,
            "gpu_memory_reserved_gb": gpu_mem_reserved_gb,
        })

        log_metrics(metrics, step + 1, use_wandb, benchmark=config.benchmark)

        # Periodic checkpoint (for recovery, not for vLLM sync)
        if config.checkpoint_interval > 0 and (step + 1) % config.checkpoint_interval == 0:
            save_checkpoint(model, tokenizer, config.save_path, step + 1)

    # === Cleanup ===
    save_checkpoint(model, tokenizer, config.save_path, config.training_steps, is_final=True)
    finalize_training(use_wandb, training_start_time, "shared_vllm", config.training_steps, benchmark_stats, config.benchmark)


def train_lora(config: TrainingConfig):
    """
    GRPO training with LoRA adapters.

    This mode:
    1. Freezes base model, trains only LoRA adapter weights
    2. Saves lightweight adapter checkpoints
    3. Hot-swaps adapters in vLLM via API

    Benefits:
    - Much faster training (fewer parameters)
    - Smaller checkpoints
    - Adapters can be hot-swapped without restart

    Requirements:
    - External vLLM server running with --enable-lora
    """
    if not PEFT_AVAILABLE:
        raise RuntimeError("PEFT library required for LoRA mode. Install with: pip install peft")

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

    # Check external vLLM server
    print("[1/3] Checking external vLLM server...")
    if not check_vllm_health(config.vllm_port):
        print(f"\nERROR: vLLM server not running on port {config.vllm_port}")
        print("\nLoRA mode requires an external vLLM server. Start it first:")
        print(f"  python example_trainer/vllm_api_server.py --model {config.model_name} "
              f"--port {config.vllm_port} --enable-lora --enforce-eager")
        raise RuntimeError(f"External vLLM server required on port {config.vllm_port}")
    print(f"vLLM server healthy on port {config.vllm_port}")

    # Load model with LoRA adapters
    print("[2/3] Loading model with LoRA adapters...")
    model, tokenizer = load_model_and_tokenizer(config)

    # Only optimize LoRA parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=config.lr)

    print(f"[3/3] Starting training for {config.training_steps} steps")
    print("-" * 60)

    os.makedirs(config.save_path, exist_ok=True)

    # Check Atropos API
    if not check_atropos_api(url=config.atropos_url, timeout=30):
        raise RuntimeError(f"Atropos API not reachable at {config.atropos_url}")
    register_trainer(config)

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

        # Fetch data
        data_fetch_start = time.time()
        if len(batches) == 0:
            batches, _ = get_data(config.batch_size, config.seq_len, config.atropos_url,
                                  extract_inference_logprobs=False)
        token_batches, label_batches, advantage_batches, temperature_batches = batches.pop(0)
        data_fetch_time = time.time() - data_fetch_start
        benchmark_stats["data_fetch_times"].append(data_fetch_time)

        # Training step
        step_start = time.time()
        metrics = run_training_step(
            model, optimizer,
            token_batches, label_batches, advantage_batches, temperature_batches,
            config,
        )
        step_time = time.time() - step_start
        benchmark_stats["step_times"].append(step_time)

        # GPU memory tracking
        gpu_mem_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        gpu_mem_reserved_gb = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        benchmark_stats["gpu_memories"].append(gpu_mem_gb)

        # Periodic adapter save + hot-swap
        sync_time = 0
        should_sync = (step + 1) % config.vllm_restart_interval == 0
        if should_sync:
            sync_start = time.time()
            adapter_path = save_lora_checkpoint(model, config.save_path, step + 1)
            _hotswap_lora_adapter(config.vllm_port, adapter_path, f"step_{step + 1}")
            sync_time = time.time() - sync_start
            benchmark_stats["sync_times"].append(sync_time)

        # Update metrics
        metrics.update({
            "step_time": step_time,
            "sync_time": sync_time,
            "data_fetch_time": data_fetch_time,
            "gpu_memory_gb": gpu_mem_gb,
            "gpu_memory_reserved_gb": gpu_mem_reserved_gb,
        })

        log_metrics(metrics, step + 1, use_wandb, benchmark=config.benchmark)

    # === Cleanup ===
    final_sync_start = time.time()
    final_adapter_path = save_lora_checkpoint(model, config.save_path, config.training_steps, is_final=True)
    _hotswap_lora_adapter(config.vllm_port, final_adapter_path, "final")
    final_sync_time = time.time() - final_sync_start
    benchmark_stats["sync_times"].append(final_sync_time)

    finalize_training(use_wandb, training_start_time, "lora_only", config.training_steps, benchmark_stats, config.benchmark)

    # Save tokenizer
    tokenizer_path = os.path.join(config.save_path, "tokenizer")
    tokenizer.save_pretrained(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")


def _hotswap_lora_adapter(
    port: int,
    adapter_path: str,
    adapter_name: Optional[str] = None,
) -> bool:
    """
    Request vLLM to hot-swap to a new LoRA adapter.

    Tries:
    1. Native vLLM endpoint: /v1/load_lora_adapter
    2. Custom endpoint: /lora/load
    """
    base_url = f"http://localhost:{port}"
    name = adapter_name or os.path.basename(adapter_path)

    # Try native vLLM endpoint first
    try:
        response = requests.post(
            f"{base_url}/v1/load_lora_adapter",
            json={"lora_name": name, "lora_path": adapter_path},
            timeout=30,
        )
        if response.status_code == 200:
            print(f"  [LORA] ✓ Hot-swapped adapter: {name}")
            return True
    except Exception:
        pass

    # Try custom endpoint
    try:
        response = requests.post(
            f"{base_url}/lora/load",
            json={"adapter_path": adapter_path, "adapter_name": name},
            timeout=30,
        )
        if response.status_code == 200:
            print(f"  [LORA] ✓ Hot-swapped adapter via custom API: {name}")
            return True
        else:
            print(f"  [LORA] ✗ Hot-swap failed: {response.text}")
            return False
    except Exception as e:
        print(f"  [LORA] ✗ Hot-swap request failed: {e}")
        return False

