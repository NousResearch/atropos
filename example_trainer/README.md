# GRPO Trainer

A modular training framework for fine-tuning language models with **Group Relative Policy Optimization (GRPO)**, designed to work with the Atropos environment system.

## Module Structure

```
example_trainer/
├── grpo.py              # CLI entry point (dispatches to trainers)
├── run.py               # Unified launcher for shared_vllm mode
├── config.py            # TrainingConfig dataclass
├── cli.py               # CLI argument parsing (single source of truth)
├── api.py               # Atropos API communication
├── data.py              # Data fetching & preprocessing
├── model.py             # Model loading & CUDA IPC shared memory
├── training.py          # GRPO loss computation & training step
├── checkpointing.py     # Save models & LoRA adapters
├── vllm_manager.py      # vLLM process management
├── trainers.py          # Training mode implementations
├── vllm_api_server.py   # Custom vLLM server (streamlined for training)
├── vllm_patching/       # CUDA IPC patches for weight sharing
│   └── patched_gpu_runner.py
└── scripts/             # Helper scripts
    ├── test_lora_mode.sh
    └── test_single_copy_mode.sh
```


GRPO Training Loop

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

Data Flow:
1. Environment generates prompts → calls vLLM → scores responses
2. Environment sends trajectories to run-api
3. Trainer fetches batches from run-api
4. Trainer updates model weights
5. (shared_vllm) vLLM sees updates immediately via CUDA IPC
   (lora_only) Trainer pushes adapter to vLLM periodically
```

---

## Three Training Modes

| Mode | Description | Memory | Best For |
|------|-------------|--------|----------|
| **shared_vllm** | Single-copy via CUDA IPC | 1x model | Same GPU, maximum efficiency |
| **lora_only** | Train adapters, HTTP hot-swap | 1x + small adapter | Simple setup, debugging |
| **legacy** | Full model, restart vLLM | 2x model | Different GPUs, simple setup |

### Recommendation

**Start with `lora_only`** - it's the easiest to set up and debug.

**Use `shared_vllm`** for production training when you need:
- Fastest weight synchronization (CUDA IPC, zero-copy updates)
- True on-policy training (vLLM sees updates immediately)

**Use `shared_vllm`** for single-GPU training when you need maximum efficiency.

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
python environments/gsm8k_server.py serve \
    --env.group_size 4 \
    --env.max_num 200 \
    --slurm.num_requests_per_time_interval 16 \
    --slurm.time_interval 10 \
    --openai.api_key "dummy" \
    --openai.base_url "http://localhost:9001" \
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
    --learning-rate 1e-5 \
    --training-steps 30 \
    --kl-coef 0.1 \
    --clip-eps 0.2 \
    --vllm-restart-interval 5 \
    --save-path ./lora_checkpoints \
    --wandb-project "grpo-training"
```

### Startup Order

```bash
# 1. Start API
# 2. Wait 5s, start vLLM
# 3. Wait for vLLM to load (check: curl http://localhost:9001/health)
# 4. Start environment
# 5. Start trainer
```

---

##  Shared vLLM Mode (Advanced)

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
python environments/gsm8k_server.py serve \
    --openai.base_url "http://localhost:9001" \
    --openai.model_name "NousResearch/Hermes-3-Llama-3.1-8B" \
    --openai.server_type vllm
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

### 1. Always Use `--enforce-eager` with Shared Weights

**Why:** CUDA graphs "bake" weights at compile time. Without eager mode, vLLM won't see weight updates!

```bash
# WRONG - weight updates won't be visible to inference
python vllm_api_server.py --model $MODEL

# CORRECT - disables CUDA graphs
python vllm_api_server.py --model $MODEL --enforce-eager
```

### 2. Use `--openai.server_type vllm` for Training

The gsm8k environment needs logprobs for GRPO. Only `server_type=vllm` uses the `/generate` endpoint which returns logprobs.

```bash
# CORRECT - gets logprobs for training
--openai.server_type vllm

# WRONG for training - no logprobs
--openai.server_type openai
```

### 3. KL Coefficient and Clipping Are Essential

Without these, training will collapse (reward hacking):

```bash
--kl-coef 0.1      # Prevents policy from drifting too far
--clip-eps 0.2     # Limits update magnitude
```

**Symptoms of missing KL/clipping:**
- Accuracy drops dramatically (e.g., 59% → 7%)
- Loss goes to very negative values
- Model outputs become repetitive/degenerate

### 4. Memory Budgeting for Large Models

| Model Size | GPU Memory | Recommended Settings |
|------------|------------|----------------------|
| 8B | 80GB | `--gpu-memory-utilization 0.5` |
| 14B | 80GB | `--gpu-memory-utilization 0.45`, `--batch-size 2` |
| 24B | 192GB (B200) | `--gpu-memory-utilization 0.30`, `--optimizer adafactor` |

### 5. Start with Small Batch Sizes

```bash
# Start conservative, increase if no OOM
--batch-size 2 --gradient-accumulation-steps 8  # Effective batch = 16
```

### 6. Optimizer Selection

The trainer supports multiple optimizer options to trade off between speed, memory, and precision:

| Optimizer | GPU Memory for States | Speed | Precision | Dependencies |
|-----------|----------------------|-------|-----------|--------------|
| `adamw` (default) | ~32GB (for 8B model) | Fastest | Full FP32 | None |
| `adamw_8bit` | ~8GB | Fast | 8-bit quantized | `bitsandbytes` |
| `adafactor` | ~8GB | Fast | Full (no momentum) | `transformers` |
| `adamw_cpu` | ~0GB (on CPU) | ~2x slower | Full FP32 | None |

**Usage:**
```bash
# Standard AdamW (default)
--optimizer adamw

# 8-bit AdamW - recommended for memory-constrained setups
--optimizer adamw_8bit

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


### Q: Why isn't vLLM seeing my weight updates?

**A:** CUDA graphs are caching the old weights. Add `--enforce-eager`:

```bash
python vllm_api_server.py --model $MODEL --enforce-eager
```



### Q: How do I debug logprob alignment issues?

**A:** Look for these log messages:
```
[WARNING] ref_logprobs at generated positions avg 0.85 (should be negative!)
```

This means inference logprobs aren't being passed correctly. Check that:
1. Environment uses `--openai.server_type vllm`
2. vLLM returns logprobs (check `/generate` response)

### Q: Why does vLLM v1 engine fail with CUDA fork errors?

**A:** vLLM v1 uses multiprocessing that conflicts with CUDA initialization. We default to v0 engine:

```python
# vllm_api_server.py automatically sets:
os.environ.setdefault("VLLM_USE_V1", "0")
```


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

### Training is slow / no batches

1. Check vLLM is running: `curl http://localhost:9001/health`
2. Check API is running: `curl http://localhost:8002/info`
3. Check environment is connected and generating rollouts

---

## 📊 Monitoring Training

### Key Metrics to Watch

| Metric | Healthy Range | Problem If... |
|--------|---------------|---------------|
| `mean_ratio` | 0.8 - 1.2 | Far from 1.0 = policy changed too much |
| `mean_kl` | 0.01 - 0.1 | > 0.5 = policy drifting |
| `clipped_fraction` | < 0.3 | > 0.5 = learning rate too high |
| `loss` | Gradually decreasing | Exploding or very negative |

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
| `--model-name` | (required) | HuggingFace model ID |
| `--weight-bridge-mode` | `none` | `shared_vllm`, `lora_only`, or `none` |
| `--training-steps` | 10 | Number of training steps |
| `--batch-size` | 2 | Micro-batch size |
| `--gradient-accumulation-steps` | 1 | Effective batch = batch × accum |

### GRPO Hyperparameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--kl-coef` | 0.1 | KL penalty strength (higher = more conservative) |
| `--clip-eps` | 0.2 | PPO clipping range [1-ε, 1+ε] |
| `--learning-rate` | 1e-6 | Learning rate |

### LoRA Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--lora-r` | 16 | LoRA rank |
| `--lora-alpha` | 32 | LoRA scaling factor |
| `--lora-dropout` | 0.05 | LoRA dropout |

### vLLM Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--vllm-port` | 9001 | vLLM server port |
| `--vllm-config-path` | auto | Path to bridge config (shared mode) |
| `--gpu-memory-utilization` | 0.9 | vLLM GPU memory fraction |

---

## Module Documentation

| Module | Purpose |
|--------|---------|
| `grpo.py` | CLI entry point, dispatches to training modes |
| `run.py` | Unified launcher for shared_vllm mode |
| `cli.py` | Single source of truth for all CLI arguments |
| `config.py` | `TrainingConfig` Pydantic model |
| `api.py` | Communication with Atropos API |
| `data.py` | Batch preprocessing, logprob extraction |
| `model.py` | Model loading, CUDA IPC attachment, tensor mapping |
| `training.py` | GRPO loss computation |
| `trainers.py` | Mode-specific training loops |
| `vllm_api_server.py` | Streamlined vLLM server for training |
| `vllm_manager.py` | vLLM process lifecycle management |
| `checkpointing.py` | Save/load checkpoints and adapters |
