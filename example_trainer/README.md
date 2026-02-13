# GRPO Trainer

A modular training framework for fine-tuning language models with **Group Relative Policy Optimization (GRPO)**, designed to work with the Atropos environment system.

## Module Structure

```
example_trainer/
├── grpo.py              # CLI entry point (dispatches to 4 training modes)
├── run.py               # Unified launcher for shared_vllm mode (starts vLLM+trainer)
├── config.py            # TrainingConfig Pydantic model (all hyperparameters)
├── cli.py               # CLI argument parsing (modular, single source of truth)
├── api.py               # Atropos API communication (registration, batch fetching)
├── data.py              # Data fetching, preprocessing, logprob alignment
├── model.py             # Model loading, CUDA IPC, tensor mapping (QKV/Gate fusion)
├── training.py          # GRPO loss (importance sampling, KL penalty, clipping)
├── checkpointing.py     # Save models & LoRA adapters (handles fused tensor unfusing)
├── vllm_manager.py      # vLLM process lifecycle (launch, health, termination)
├── trainers.py          # 4 training mode implementations + optimizer selection
├── vllm_api_server.py   # Custom vLLM server with /generate endpoint + LoRA
├── vllm_patching/       # CUDA IPC patches for weight sharing 
│   └── patched_gpu_runner.py
└── scripts/             # Helper scripts and benchmarks
    ├── test_lora_mode.sh
    ├── test_single_copy_mode.sh
    └── compare_all_modes_math_zero.sh
```


## GRPO Training Loop

```
1. Generate multiple responses to the same prompt
2. Score each response (reward)
3. Compute ADVANTAGE = reward - mean(rewards)
4. Train: increase probability of above-average responses
    decrease probability of below-average responses
```

### Key Concepts

| Concept | What It Means |
|---------|---------------|
| **Advantage** | How much better/worse than average a response was |
| **Importance Sampling** | Corrects for policy drift during training |
| **KL Penalty** | Prevents the model from changing too drastically from base |
| **Clipping** | Limits update magnitude for stability |


## System Architecture

```
Data Flow:
1. Environment generates prompts → calls vLLM → scores responses
2. Environment sends trajectories to run-api
3. Trainer fetches batches from run-api
4. Trainer updates model weights
5. Weight synchronization:
   - shared_vllm: vLLM sees updates immediately via CUDA IPC (zero-copy)
   - lora_only: Trainer pushes adapter to vLLM via HTTP (slow)
   - lora_restart: Trainer restarts vLLM with new adapter (fast)
   - none (legacy): Trainer saves checkpoint and restarts vLLM
```

---

## Four Training Modes

| Mode | Description | Memory | Inference Speed | Best For |
|------|-------------|--------|-----------------|----------|
| **shared_vllm** | Single-copy via CUDA IPC | 1x model | ~172 TPS | Same GPU, maximum efficiency |
| **lora_restart** | LoRA + vLLM restarts | 1x + adapter | ~108 TPS | LoRA training with speed |
| **lora_only** | LoRA + HTTP hot-swap | 1x + adapter | ~13 TPS ⚠️ | Debugging only |
| **none** (legacy) | Full model, restart vLLM | 2x model | ~172 TPS | simple setup |

### ⚠️ IMPORTANT: `lora_only` Performance Warning

The `lora_only` mode requires `--enforce-eager` which **disables CUDA graphs**, resulting in:
- **8x slower inference** (~13 TPS vs ~108 TPS)
- Training that takes **4x longer** (401 min vs 132 min for 120 steps)

**Use `lora_restart` instead** - it runs vLLM without `--enforce-eager` for 8x faster inference.

### Recommendation

**Use `shared_vllm`** for production training when:
- You have enough GPU memory for the full model
- You want fastest training (no overhead)
- Trainer and vLLM are on the same GPU(s)

**Use `lora_restart`** when:
- You want LoRA's memory efficiency
- You can tolerate ~45s restart overhead every N steps

**Avoid `lora_only`** unless you're debugging - the 8x inference penalty is severe.

**Use `none` (legacy)** mode when:
- You want the simplest setup without CUDA IPC or LoRA

---

## Quick Start: LoRA Training (Recommended)

### Step 1: Install Dependencies
- They are listed in the requirements.txt file that you can see

### Step 2: Start All Components

**Terminal 1: API Server**
```bash
run-api --port 8002
```

**Terminal 2: vLLM Server**
```bash
python -m example_trainer.vllm_api_server \
    --model NousResearch/Hermes-3-Llama-3.1-8B \
    --port 9001 \
    --gpu-memory-utilization 0.5 \
    --max-model-len 4096 \
    --dtype bfloat16 \
    --enable-lora \
    --enforce-eager
```

**Terminal 3: Environment**
```bash
# Important: Use server_type=vllm to get logprobs (required for GRPO)
python environments/gsm8k_server.py serve \
    --env.group_size 4 \
    --env.max_num 200 \
    --slurm.num_requests_per_time_interval 16 \
    --slurm.time_interval 10 \
    --openai.api_key "dummy" \
    --openai.base_url "http://localhost:9001/v1" \
    --openai.model_name "NousResearch/Hermes-3-Llama-3.1-8B" \
    --openai.server_type vllm
```

**Terminal 4: Trainer**
```bash
python -m example_trainer.grpo \
    --model-name NousResearch/Hermes-3-Llama-3.1-8B \
    --weight-bridge-mode lora_only \
    --vllm-port 9001 \
    --atropos-url "http://localhost:8002" \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --lr 1e-5 \
    --training-steps 30 \
    --kl-coef 0.1 \
    --clip-eps 0.2 \
    --vllm-restart-interval 5 \
    --save-path ./lora_checkpoints \
    --wandb-project "grpo-training"
```

### Startup Order

```bash
# CRITICAL: Follow this exact order!
# 1. Start API first
run-api --port 8002

# 2. Wait 5s, then start vLLM
# Check health: curl http://localhost:9001/health
python -m example_trainer.vllm_api_server --model ... --enable-lora --enforce-eager

# 3. Wait for vLLM health endpoint to return 200
while ! curl -s http://localhost:9001/health > /dev/null; do sleep 1; done

# 4. Start environment (MUST use --openai.server_type vllm for logprobs)
python environments/gsm8k_server.py serve ...

# 5. Start trainer (will register with API and begin training)
python -m example_trainer.grpo --weight-bridge-mode lora_only ...
```

---

##  Shared vLLM Mode 

Single-copy mode shares GPU memory between vLLM and the trainer - zero model duplication!

### How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SINGLE GPU (CUDA IPC)                            │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Model Weights (ONE copy in GPU memory)          │   │
│  │               (accessible via CUDA IPC handles)              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│           ▲                                          ▲              │
│           │ Reads (inference)                        │ Writes       │
│  ┌────────┴────────┐                     ┌───────────┴───────────┐ │
│  │  vLLM Worker    │                     │  Trainer Process      │ │
│  │                 │                     │  (attached via IPC)   │ │
│  └─────────────────┘                     └───────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### Running Shared vLLM Mode

**Terminal 1: API**
```bash
run-api --port 8002
```

**Terminal 2: vLLM with Shared Weights**
```bash
VLLM_ENABLE_SHARED_WEIGHTS=1 LOGDIR=/tmp/grpo_training \
python -m example_trainer.vllm_api_server \
    --model NousResearch/Hermes-3-Llama-3.1-8B \
    --port 9001 \
    --gpu-memory-utilization 0.45 \
    --enforce-eager
```

**Terminal 3: Environment**
```bash
# Important: Use server_type=vllm to get logprobs (required for GRPO)
python environments/gsm8k_server.py serve \
    --openai.base_url "http://localhost:9001/v1" \
    --openai.model_name "NousResearch/Hermes-3-Llama-3.1-8B" \
    --openai.server_type vllm \
    --env.group_size 4 \
    --slurm.num_requests_per_time_interval 16 \
    --slurm.time_interval 10
```

**Terminal 4: Trainer**
```bash
python -m example_trainer.grpo \
    --model-name NousResearch/Hermes-3-Llama-3.1-8B \
    --weight-bridge-mode shared_vllm \
    --vllm-port 9001 \
    --vllm-config-path /tmp/grpo_training/vllm_bridge_config.json \
    --atropos-url "http://localhost:8002" \
    --kl-coef 0.1 \
    --clip-eps 0.2
```

### Or Use the Unified Launcher

```bash
# Single command starts both vLLM and trainer
VLLM_ENABLE_SHARED_WEIGHTS=1 python -m example_trainer.run \
    --model-name NousResearch/Hermes-3-Llama-3.1-8B \
    --atropos-url "http://localhost:8002" \
    --training-steps 30
```

---

## Best Practices & Lessons Learned


### 1. Use `--openai.server_type vllm` for Training

**CRITICAL:** The atropos environment MUST use `server_type=vllm` to get logprobs for proper GRPO training.

Only `server_type=vllm` calls the `/generate` endpoint which returns token-level logprobs. These logprobs serve as the reference policy (π_old) for importance sampling in GRPO.

```bash
# CORRECT - gets logprobs for training (REQUIRED!)
--openai.server_type vllm

# WRONG for training - no logprobs, training will FAIL
--openai.server_type openai
```

**What happens without logprobs:**
- The trainer will raise an error: "GRPO requires inference_logprobs for importance sampling!"
- Without the reference policy, GRPO degenerates to vanilla REINFORCE (leads to reward hacking)

**How logprobs flow through the system:**
1. Environment calls vLLM `/generate` with `logprobs=true`
2. vLLM returns token-level logprobs for each generated token
3. Environment embeds these in trajectory data sent to API
4. Trainer extracts and aligns logprobs with training labels
5. GRPO loss uses logprobs as π_old for importance sampling ratio

### 2. KL Coefficient and Clipping Are Essential

**CRITICAL:** Without these hyperparameters, training WILL collapse (reward hacking):

```bash
--kl-coef 0.1      # Prevents policy from drifting too far from reference
--clip-eps 0.2     # Limits importance sampling ratio to [0.8, 1.2]
```

**Why these matter:**
- **KL Penalty** (β): Penalizes the policy for deviating from the reference policy (inference-time policy)
  - Uses Schulman's unbiased estimator: `exp(-log_ratio) + log_ratio - 1`
  - Higher β = more conservative updates
  - Set to 0 to disable (NOT recommended - leads to instability)

- **PPO Clipping** (ε): Clips the importance sampling ratio to `[1-ε, 1+ε]`
  - Prevents catastrophically large policy updates
  - Takes pessimistic bound (conservative update)

**Symptoms of missing/misconfigured KL/clipping:**
- Accuracy drops dramatically (e.g., 59% → 7%)
- Loss goes to very negative values (< -10)
- Model outputs become repetitive/degenerate
- `mean_ratio` diverges far from 1.0
- `mean_kl` explodes (> 1.0)

**Healthy training metrics:**
- `mean_ratio`: 0.8 - 1.2 (close to 1.0)
- `mean_kl`: 0.01 - 0.1
- `clipped_fraction`: < 0.3 (< 30% of tokens clipped)

### 3. Memory Budgeting for Large Models

| Model Size | GPU Memory | Recommended Settings |
|------------|------------|----------------------|
| 8B | 80GB | `--gpu-memory-utilization 0.5` |
| 14B | 80GB | `--gpu-memory-utilization 0.45`, `--batch-size 2` |
| 24B | 192GB (B200) | `--gpu-memory-utilization 0.30`, `--optimizer adafactor` |


### 4. Optimizer Selection

The trainer supports multiple optimizer options to trade off between speed, memory, and precision:

| Optimizer | GPU Memory for States | Speed | Precision | Dependencies |
|-----------|----------------------|-------|-----------|--------------|
| `adamw` | ~32GB (for 8B model) | Fastest | Full FP32 | None |
| `adamw_8bit` (default) | ~8GB | Fast | 8-bit quantized | `bitsandbytes` |
| `adafactor` | ~8GB | Fast | Full (no momentum) | `transformers` |
| `adamw_cpu` | ~0GB (on CPU) | ~2x slower | Full FP32 | None |

**Usage:**
```bash
# 8-bit AdamW (default) - recommended for memory-constrained setups
--optimizer adamw_8bit

# Standard AdamW - full precision
--optimizer adamw

# Adafactor - no momentum states, good for large models
--optimizer adafactor

# CPU offload - experimental, use when nothing else fits
--optimizer adamw_cpu
```

**Recommendations:**
- **8B models on 80GB:** Use `adamw` (fastest)
- **14B+ models on 80GB:** Use `adamw_8bit` or `adafactor`
- **24B models:** Use `adafactor` with reduced batch size
- **adamw_cpu:** Experimental - not well tested, ~2x slower due to CPU↔GPU transfers

**Potential Risks:**
- `adamw_8bit`: Quantization may slightly affect convergence in edge cases; generally safe
- `adafactor`: No momentum can make training slightly less stable; use with larger batch sizes
- `adamw_cpu`: Significantly slower; only use when you have no other option

---

## Tensor Mapping (vLLM ↔ HuggingFace)

### The Problem

vLLM fuses certain layers for efficiency, but HuggingFace keeps them separate:

```
HuggingFace Model:              vLLM Model:
├── q_proj [4096, 4096]         ├── qkv_proj [12288, 4096]  ← FUSED!
├── k_proj [1024, 4096]         │   (contains q, k, v concatenated)
├── v_proj [1024, 4096]         │
│                               │
├── gate_proj [14336, 4096]     ├── gate_up_proj [28672, 4096]  ← FUSED!
├── up_proj [14336, 4096]       │   (contains gate and up concatenated)
```

### How We Solve It

The trainer creates **views** into vLLM's fused tensors:

```python
# vLLM has: qkv_proj.weight [12288, 4096]
# We need:  q_proj [4096], k_proj [1024], v_proj [1024]

# Get sizes from model config
q_size = num_heads * head_dim           # e.g., 4096
k_size = num_kv_heads * head_dim        # e.g., 1024
v_size = num_kv_heads * head_dim        # e.g., 1024

# Create views (no copy!)
hf_model.q_proj.weight = vllm_qkv[0:4096, :]      # First chunk
hf_model.k_proj.weight = vllm_qkv[4096:5120, :]   # Second chunk
hf_model.v_proj.weight = vllm_qkv[5120:6144, :]   # Third chunk
```

### Key Insight: Views Share Memory

```python
# These point to the SAME GPU memory:
trainer_q_proj.data_ptr() == vllm_qkv_proj.data_ptr()  # True!

# So when optimizer updates trainer weights:
optimizer.step()  # Updates trainer_q_proj

# vLLM sees the change immediately (same memory)!
```

### The Config File

vLLM exports tensor mappings to `vllm_bridge_config.json`:

```json
{
  "model": "NousResearch/Hermes-3-Llama-3.1-8B",
  "param_mappings": {
    "model.layers.0.self_attn.qkv_proj.weight": {
      "ipc_handle": "base64_encoded_cuda_ipc_handle",
      "shape": [6144, 4096],
      "dtype": "bfloat16"
    }
  }
}
```

---

## ❓ FAQ

### Q: How do I debug logprob alignment issues?

**A:** Look for these log messages during training:
```
[WARNING] ref_logprobs at generated positions avg 0.85 (should be negative!)
[WARNING] This suggests inference_logprobs alignment is wrong
```

This means inference logprobs aren't being passed correctly. Debug steps:

1. **Check environment server type:**
   ```bash
   # Must be 'vllm', NOT 'openai'
   --openai.server_type vllm
   ```

2. **Verify vLLM returns logprobs:**
   ```bash
   curl -X POST http://localhost:9001/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Hello", "max_tokens": 5}'
   # Response should include "logprobs": [...]
   ```

3. **Check data.py logs:**
   ```
   [Data] ✓ inference_logprobs found in batch (sample len: 128)
   ```

4. **Monitor alignment metrics in training logs:**
   - `alignment/diff_mean` should be close to 0 at step start
   - `alignment/diff_abs_mean` < 0.1 = good alignment
   - Large values = weights not properly shared or logprobs misaligned


##  Troubleshooting

### "Atropos API not reachable"

```bash
# Start the API server first
run-api --port 8002
```

### "404 Not Found" on /generate

You're using a vLLM server that doesn't expose `/generate`. Use our custom server:

```bash
python -m example_trainer.vllm_api_server ...  # Has /generate
# NOT: python -m vllm.entrypoints.openai.api_server  # Only has /v1/*
```

### "Cannot re-initialize CUDA in forked subprocess"

vLLM v1 engine issue. We disable it by default, but if you see this:

```bash
VLLM_USE_V1=0 python -m example_trainer.vllm_api_server ...
```

### "LogProb Alignment: MISMATCH!"

Weight updates aren't visible to inference. Fix:

```bash
# Add --enforce-eager to vLLM
python vllm_api_server.py --model $MODEL --enforce-eager
```

### OOM (Out of Memory)

Reduce memory usage:

```bash
--gpu-memory-utilization 0.4   # Less vLLM memory
--batch-size 2                  # Smaller batches
--gradient-accumulation-steps 8 # Compensate with accumulation
--seq-len 1024                  # Shorter sequences
--optimizer adafactor           # Uses less memory than AdamW
```

### "FlexibleArgumentParser" import error

vLLM version incompatibility. Our server handles this automatically, but make sure you're using:

```bash
python -m example_trainer.vllm_api_server  # NOT direct vllm commands
```


## 📊 Monitoring Training

### WandB Logging

```bash
--use-wandb \
--wandb-project "my-grpo-training" \
--wandb-run-name "hermes-8b-gsm8k"
```

---

##  CLI Reference

### Essential Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-name` or `--model` | (required) | HuggingFace model ID |
| `--weight-bridge-mode` | `none` | `shared_vllm`, `lora_only`, `lora_restart`, or `none` |
| `--training-steps` | 10 | Number of training steps |
| `--batch-size` | 2 | Micro-batch size |
| `--gradient-accumulation-steps` | 32 | Effective batch = batch × accum |
| `--seq-len` | 2048 | Maximum sequence length |

### GRPO Hyperparameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--kl-coef` | 0.1 | KL penalty strength (higher = more conservative) |
| `--clip-eps` | 0.2 | PPO clipping range [1-ε, 1+ε] |
| `--lr` | 1e-5 | Learning rate (NOT --learning-rate) |

### LoRA Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--lora-r` | 16 | LoRA rank (dimension of low-rank matrices) |
| `--lora-alpha` | 32 | LoRA alpha scaling factor |
| `--lora-dropout` | 0.05 | LoRA dropout probability |
| `--lora-target-modules` | None | Module names to apply LoRA (default: `q_proj v_proj`) |

### vLLM Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--vllm-port` | 9001 | vLLM server port |
| `--vllm-config-path` | auto | Path to bridge config (shared mode) |
| `--gpu-memory-utilization` | 0.45 | vLLM GPU memory fraction |
| `--vllm-gpu` | None | GPU ID for vLLM (None = same as trainer) |
| `--max-model-len` | 4096 | Maximum context length |
| `--dtype` | `bfloat16` | Model dtype: `bfloat16`, `float16`, or `auto` |
| `--vllm-restart-interval` | 3 | Restart vLLM every N steps (legacy/lora_restart) |

---

## Module Documentation

| Module | Purpose |
|--------|---------|
| `grpo.py` | CLI entry point, dispatches to training modes (4 modes) |
| `run.py` | Unified launcher for shared_vllm mode (starts vLLM + trainer) |
| `cli.py` | Single source of truth for all CLI arguments (modular builders) |
| `config.py` | `TrainingConfig` Pydantic model with all hyperparameters |
| `api.py` | Communication with Atropos API (registration, batch fetching) |
| `data.py` | Batch preprocessing, padding, logprob extraction and alignment |
| `model.py` | Model loading, CUDA IPC attachment, tensor mapping (QKV/Gate fusion) |
| `training.py` | GRPO loss computation (importance sampling, KL penalty, clipping) |
| `trainers.py` | Mode-specific training loops (4 implementations + optimizer selection) |
| `vllm_api_server.py` | Custom vLLM server with `/generate` endpoint and LoRA support |
| `vllm_manager.py` | vLLM process lifecycle management (launch, health checks, termination) |
| `checkpointing.py` | Save/load checkpoints and adapters (handles fused tensor unfusing) |

---

## Code Execution Flow

### High-Level Flow (All Modes)

```
1. CLI Parsing (cli.py)
   ↓
2. Config Creation (config.py)
   ↓
3. Mode Dispatcher (grpo.py or run.py)
   ↓
4. Trainer Function (trainers.py)
   ├─ Setup Phase
   │  ├─ Initialize W&B (training.py)
   │  ├─ Load Model (model.py)
   │  ├─ Create Optimizer (trainers.py)
   │  ├─ Check Atropos API (api.py)
   │  ├─ Register Trainer (api.py)
   │  └─ Launch/Connect vLLM (vllm_manager.py or external)
   │
   └─ Training Loop
      ├─ Fetch Batch (api.py → data.py)
      │  ├─ Poll /batch endpoint
      │  ├─ Pad sequences (data.py)
      │  ├─ Extract inference logprobs (data.py)
      │  └─ Normalize advantages (data.py)
      │
      ├─ Training Step (training.py)
      │  ├─ For each micro-batch:
      │  │  ├─ Forward pass (model)
      │  │  ├─ Compute GRPO loss (training.py)
      │  │  │  ├─ Temperature scaling
      │  │  │  ├─ Compute log probabilities
      │  │  │  ├─ Importance sampling ratio (using inference logprobs)
      │  │  │  ├─ PPO clipping
      │  │  │  ├─ Schulman KL penalty
      │  │  │  └─ Return loss + metrics
      │  │  └─ Backward pass (accumulate gradients)
      │  ├─ Clip gradients (norm=1.0)
      │  ├─ Optimizer step
      │  └─ Zero gradients
      │
      ├─ Weight Sync (mode-dependent)
      │  ├─ shared_vllm: No sync needed (weights shared via CUDA IPC)
      │  ├─ lora_only: HTTP POST to /lora/load
      │  ├─ lora_restart: Save adapter + terminate + relaunch vLLM
      │  └─ none: Save checkpoint + terminate + relaunch vLLM
      │
      ├─ Log Metrics (training.py)
      │  ├─ Console output
      │  └─ W&B logging (if enabled)
      │
      └─ Periodic Checkpoint (checkpointing.py)
         ├─ Ensure tensors are contiguous (unfuse views)
         ├─ Save state dict
         └─ Free GPU memory
```

### Mode-Specific Details

#### shared_vllm Mode

```python
# Entry: grpo.py → trainers.train_shared_vllm()

1. Model Loading (model.py):
   - Find vllm_bridge_config.json
   - Load IPC handles (CUDA memory pointers)
   - Create empty model on meta device
   - Reconstruct tensors from IPC handles
   - Map vLLM fused tensors → HF unfused parameters
     * qkv_proj → q_proj, k_proj, v_proj (views)
     * gate_up_proj → gate_proj, up_proj (views)
   - Initialize remaining meta tensors (buffers, etc.)

2. Training Loop:
   - optimizer.step() directly modifies vLLM's tensors
   - No weight synchronization needed!
   - Checkpoints: Unfuse views before saving (checkpointing.py)

3. Tensor Mapping (model.py:_create_vllm_to_hf_mapping):
   - Reads actual HF tensor shapes from model.state_dict()
   - Creates slice mappings for fused layers
   - Example: q_proj = qkv_proj[0:4096, :]
```

#### lora_restart Mode

```python
# Entry: grpo.py → trainers.train_lora_restart()

1. Model Loading (model.py):
   - Load base model with PEFT
   - Apply LoRA config to target modules
   - Freeze base weights, only LoRA trainable

2. vLLM Management:
   - Launch: _launch_vllm_with_lora()
     * NO --enforce-eager flag (CUDA graphs enabled)
     * Pre-load initial adapter
   - Periodic Restart:
     * Save new adapter (checkpointing.py)
     * Terminate vLLM aggressively (_terminate_vllm)
       - Kill process group
       - Kill by port (fuser)
       - Kill by process name patterns
       - Wait for GPU memory release (critical!)
     * Relaunch with new adapter

3. Performance:
   - ~108 TPS (CUDA graphs enabled)
   - ~45s restart overhead
   - Much faster than lora_only (~8x speedup)
```

#### lora_only Mode

```python
# Entry: grpo.py → trainers.train_lora()

1. Model Loading: Same as lora_restart

2. vLLM: External server (must be pre-started)
   - MUST use --enforce-eager (disables CUDA graphs)
   - MUST use --enable-lora

3. Weight Sync: _hotswap_lora_adapter()
   - Tries /v1/load_lora_adapter (native vLLM)
   - Falls back to /lora/load (custom endpoint)

4. Performance:
   - ~13 TPS (CUDA graphs disabled)
   - No restart overhead
   - 8x slower than lora_restart!
```

#### none (legacy) Mode

```python
# Entry: grpo.py → trainers.train_legacy()

1. Model Loading: Full model (model.py)

2. vLLM Management:
   - Launch: vllm_manager.launch_vllm_server()
   - Periodic Restart:
     * Save full checkpoint (checkpointing.py)
     * Terminate vLLM (vllm_manager.terminate_vllm_process)
     * Relaunch with new checkpoint

3. Use Case:
   - Different GPUs for trainer and vLLM
   - Simple setup without CUDA IPC or LoRA
```

### Data Flow Detail (data.py)

```python
# api.get_batch() → data.get_data() → data.pad_data_to_good_offset()

1. Batch Structure from API:
   {
     "batch": [
       {
         "tokens": [[tok1, tok2, ...], ...],  # group_size sequences
         "masks": [[mask1, mask2, ...], ...],  # -100 for prompt, token_id for generated
         "scores": [score1, score2, ...],      # rewards
         "inference_logprobs": [[lp1, lp2, ...], ...],  # CRITICAL for GRPO!
         "generation_params": {"temperature": 1.0},
         ...
       }
     ]
   }

2. Preprocessing (pad_data_to_good_offset):
   - Normalize advantages (mean=0, std=1 per group)
   - Pad sequences to multiple of 64
   - Align inference_logprobs with labels:
     * 1.0 for prompt tokens (masked)
     * Actual negative logprobs for generated tokens
     * Shift by 1 for causal alignment
   - Extract temperatures (priority: override > generation_params > 1.0)
   - Batch into micro-batches

3. Output:
   - token_batches: [B, seq_len]
   - label_batches: [B, seq_len]  # -100 for masked
   - advantage_batches: [B, 1]
   - temperature_batches: [B, 1, 1]
   - inference_logprob_batches: [B, seq_len]  # aligned with labels!
```

### GRPO Loss Computation (training.py)

```python
# training.compute_grpo_loss()

1. Forward Pass:
   - Get logits from model
   - Apply temperature scaling (from data)
   - Compute log probabilities per token

2. Reference Policy (π_old):
   - Extract from inference_logprobs (from vLLM at generation time)
   - Already aligned with labels by data.py

3. Importance Sampling:
   - log_ratio = log π_new(a|s) - log π_old(a|s)
   - ratio = exp(log_ratio)
   - Clipped ratio = clip(ratio, 1-ε, 1+ε)

4. Policy Loss:
   - surr1 = ratio * advantage
   - surr2 = clipped_ratio * advantage
   - policy_loss = -min(surr1, surr2)  # pessimistic bound

5. KL Penalty (Schulman's estimator):
   - kl = exp(-log_ratio) + log_ratio - 1
   - Guaranteed non-negative, unbiased

6. Total Loss:
   - loss = policy_loss + β * kl_penalty
   - Scaled by 1/gradient_accumulation_steps

7. Metrics:
   - mean_ratio: Average importance sampling ratio
   - mean_kl: Average KL divergence
   - clipped_fraction: % of tokens clipped
   - alignment/* : Token-level logprob alignment (verifies weight sharing)
```
