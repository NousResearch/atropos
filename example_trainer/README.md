# GRPO Example Trainer

This directory contains an example script (`grpo.py`) demonstrating how to integrate a custom training loop with the Atropos API for reinforcement learning using the GRPO (Group Relative Policy Optimization) algorithm.

## Training Modes

The trainer supports three weight synchronization modes:

| Mode | Description | Sync Latency | Best For |
|------|-------------|--------------|----------|
| **Legacy** (`none`) | Save checkpoints, restart vLLM | ~30-60 seconds | Simple setups, debugging |
| **Single-Copy** (`shared_vllm`) | Direct CUDA IPC - ONE model copy! | 0 ms | Production, memory efficiency |
| **LoRA** (`lora_only`) | Train adapters, hot-swap | ~1-5 seconds | Memory-constrained, fast iteration |

---

## Model Compatibility

This training pipeline works with models that meet the following requirements:

### Required Compatibility

| Component | Requirement |
|-----------|-------------|
| **HuggingFace** | Must support `AutoModelForCausalLM` |
| **vLLM** | Must be in [vLLM's supported model list](https://docs.vllm.ai/en/latest/models/supported_models.html) |
| **Architecture** | Decoder-only (causal language model) |

### ✅ Compatible Model Families

- **Qwen** (Qwen2, Qwen2.5)
- **Llama** (Llama-2, Llama-3, Llama-3.1)
- **Mistral** (Mistral, Mixtral)
- **Phi** (Phi-2, Phi-3)
- **Gemma** (Gemma, Gemma-2)
- **DeepSeek** (DeepSeek-Coder, DeepSeek-V2)
- **Yi** (Yi, Yi-1.5)
- **StarCoder** (StarCoder2)

### ❌ Not Compatible

| Type | Reason |
|------|--------|
| Encoder-only (BERT, RoBERTa) | No causal language modeling head |
| Encoder-decoder (T5, BART) | Different architecture, not supported by vLLM |
| Non-HuggingFace models | Requires `AutoModelForCausalLM.from_pretrained()` |

### Single-Copy Mode Constraints

| Constraint | Reason |
|------------|--------|
| `tensor-parallel-size` must be 1 | Multi-GPU tensor parallelism not yet supported for IPC |
| Model must fit on single GPU | No model sharding in single-copy mode |
| Trainer and vLLM on same GPU(s) | CUDA IPC requires same device |

> **Tip**: For models too large for a single GPU, use **LoRA mode** (`--weight-bridge-mode lora_only`) instead.

---

## Quick Start with GSM8k (Single-Copy Mode)

This is the **recommended** production setup for maximum training throughput and memory efficiency.

### Prerequisites

```bash
# Install dependencies
pip install -r example_trainer/requirements.txt

# Install GSM8k environment dependencies
pip install datasets latex2sympy2_extended math_verify
```

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SINGLE-COPY TRAINING ARCHITECTURE                         │
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────────┐  │
│  │ GSM8k Env   │───▶│ Atropos API │◀───│ GRPO Trainer                    │  │
│  │ (problems)  │    │ (batching)  │    │ - Attached to vLLM's tensors    │  │
│  └─────────────┘    └─────────────┘    │ - optimizer.step() updates both │  │
│         │                              └─────────────────────────────────┘  │
│         │                                              │                     │
│         │                                              │ CUDA IPC            │
│         │                                              │ (same memory!)      │
│         ▼                                              ▼                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │              vLLM Inference Server (GPU 0)                           │    │
│  │         - Model weights in GPU memory                                │    │
│  │         - Trainer sees same tensors via IPC                         │    │
│  │         - Generates rollouts for scoring                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### How Single-Copy Mode Works

```
┌────────────────────────────────────────────────────────────┐
│                    SAME GPU(s)                             │
│                                                            │
│     ┌──────────────────────────────────────────────────┐  │
│     │         SHARED MODEL TENSORS                      │  │
│     │      (only ONE copy in GPU memory!)               │  │
│     └──────────────────────────────────────────────────┘  │
│              ▲                           ▲                 │
│              │ Reads/Writes              │ Reads           │
│     ┌────────┴───────┐          ┌────────┴───────┐        │
│     │    Trainer     │          │     vLLM       │        │
│     │  (gradients)   │          │  (inference)   │        │
│     └────────────────┘          └────────────────┘        │
│              │                                             │
│              │ optimizer.step()                            │
│              │ (updates shared tensors in-place)           │
│              ▼                                             │
│     vLLM immediately sees new weights!                     │
└────────────────────────────────────────────────────────────┘
```

- **Memory**: 1x model size (truly shared via CUDA IPC!)
- **Sync Latency**: 0ms (same memory, no copy needed)
- **Requirement**: Trainer and vLLM on SAME GPU(s)

---

### Step-by-Step Guide

**IMPORTANT: GPU Allocation**
- vLLM and Trainer run on the SAME GPU(s)
- Use `tensor-parallel-size 1` for single-copy mode (TP>1 not yet supported)

---

#### Step 1: Kill Any Existing Processes

```bash
pkill -9 -u $USER -f "vllm|grpo|python|run-api" 2>/dev/null; sleep 3
```

#### Step 2: Setup Directory

```bash
cd ~/atropos_stuff/atropos
rm -f vllm_bridge_config.json vllm.log trainer.log api.log gsm8k.log
```

#### Step 3: Set Environment Variables

```bash
export VLLM_ENABLE_SHARED_WEIGHTS=1
export VLLM_SKIP_WEIGHT_DAEMON=1
export NUM_INFERENCE_NODES=0
export LOGDIR=.
```

#### Step 4: Start vLLM Server

```bash
CUDA_VISIBLE_DEVICES=0 python -u example_trainer/vllm_api_server.py \
    --model Qwen/Qwen2.5-14B-Instruct \
    --tensor-parallel-size 1 \
    --port 9001 \
    > vllm.log 2>&1 &
echo "vLLM starting on GPU 0..."
```

#### Step 5: Wait for vLLM to Load

```bash
tail -f vllm.log
```

Wait until you see: `Uvicorn running on http://0.0.0.0:9001`

Then press **Ctrl+C** to stop tailing.

#### Step 6: Verify IPC Handles Exported

```bash
grep -E "IPC|Exported|single_copy" vllm.log
```

You should see:
```
[vLLM Patch] Exported X IPC handles for single-copy mode
[vLLM Patch] ✓ Exported 339 params to vllm_bridge_config.json
```

#### Step 7: Start an Environment (THE EXAMPLE HERE IS GSM8K in this case)

```bash
python environments/gsm8k_server.py serve \
    --slurm False \
    --openai.model_name Qwen/Qwen2.5-14B-Instruct \
    --openai.base_url http://localhost:9001/v1 \
    --openai.server_type vllm \
    --openai.api_key x \
    --env.tokenizer_name Qwen/Qwen2.5-14B-Instruct \
    --env.use_wandb False \
    > gsm8k.log 2>&1 &
echo "GSM8K environment started"
sleep 10
```

#### Step 8: Start Trainer (Same GPU as vLLM!)

```bash
CUDA_VISIBLE_DEVICES=0 LOGDIR=. python -u example_trainer/grpo.py \
    --model-name Qwen/Qwen2.5-14B-Instruct \
    --weight-bridge-mode shared_vllm \
    --training-steps 100 \
    2>&1 | tee trainer.log
```

#### Step 9: Monitor Training

```bash
tail -f trainer.log
```

You should see:
```
[Setup] ✓ Attached 195 tensors to vLLM's shared memory
[Setup] ✓ Single-copy mode active - using vLLM's tensors directly!
[2/2] Starting training for 100 steps
Step 1/100
  [SINGLE-COPY] Weights updated in-place - step 1
```

---

### Quick Copy-Paste (All-in-One)

```bash
# Kill everything and setup
pkill -9 -u $USER -f "vllm|grpo|python" 2>/dev/null; sleep 3
cd ~/atropos_stuff/atropos
rm -f vllm_bridge_config.json *.log

# Environment variables
export VLLM_ENABLE_SHARED_WEIGHTS=1 VLLM_SKIP_WEIGHT_DAEMON=1 NUM_INFERENCE_NODES=0 LOGDIR=.

# Start vLLM
CUDA_VISIBLE_DEVICES=0 python -u example_trainer/vllm_api_server.py \
    --model Qwen/Qwen2.5-14B-Instruct --tensor-parallel-size 1 --port 9001 > vllm.log 2>&1 &
echo "Waiting 90s for vLLM..."; sleep 90

# Start GSM8k environment
python environments/gsm8k_server.py serve --slurm False \
    --openai.model_name Qwen/Qwen2.5-14B-Instruct \
    --openai.base_url http://localhost:9001/v1 \
    --openai.server_type vllm --openai.api_key x \
    --env.tokenizer_name Qwen/Qwen2.5-14B-Instruct \
    --env.use_wandb False > gsm8k.log 2>&1 &
sleep 10

# Start trainer (same GPU!)
CUDA_VISIBLE_DEVICES=0 LOGDIR=. python -u example_trainer/grpo.py \
    --model-name Qwen/Qwen2.5-14B-Instruct \
    --weight-bridge-mode shared_vllm \
    --training-steps 100 \
    2>&1 | tee trainer.log
```

---

## How Each Mode Works (Data Flow Diagrams)

### Single-Copy Mode (`--weight-bridge-mode shared_vllm`) ⭐ RECOMMENDED

**The Magic**: Trainer and vLLM share the EXACT SAME GPU memory via CUDA IPC.

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                     SINGLE-COPY MODE - COMPLETE DATA FLOW                           │
│                                                                                     │
│  STEP 1: GSM8k sends problem                                                        │
│  ┌──────────────────┐                                                               │
│  │   GSM8k Server   │──── "What is 15 × 7?" ────▶┌──────────────────┐              │
│  │  (Environment)   │                            │   Atropos API    │              │
│  └──────────────────┘                            │   (Batching)     │              │
│                                                  └────────┬─────────┘              │
│                                                           │                         │
│  STEP 2: Atropos forwards to vLLM                         │                         │
│                                                           ▼                         │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │                              GPU MEMORY                                       │  │
│  │                                                                               │  │
│  │  ┌────────────────────────────────────────────────────────────────────────┐  │  │
│  │  │              MODEL WEIGHTS (ONE COPY - SHARED!)                         │  │  │
│  │  │                                                                         │  │  │
│  │  │   embed_tokens.weight, layers.*.qkv_proj, ..., lm_head.weight          │  │  │
│  │  │                      (address: 0x7f8a12340000)                          │  │  │
│  │  └────────────────────────────────────────────────────────────────────────┘  │  │
│  │           ▲                                              ▲                    │  │
│  │           │ STEP 3: READ                                 │ STEP 6: WRITE      │  │
│  │           │ (generate tokens)                            │ (optimizer.step)   │  │
│  │  ┌────────┴────────┐                           ┌─────────┴─────────┐         │  │
│  │  │   vLLM Server   │                           │     Trainer       │         │  │
│  │  │                 │                           │    (grpo.py)      │         │  │
│  │  │  Generates:     │                           │                   │         │  │
│  │  │  "15 × 7 = 105" │                           │ STEP 5: Compute   │         │  │
│  │  │                 │                           │ GRPO loss &       │         │  │
│  │  └────────┬────────┘                           │ gradients         │         │  │
│  │           │                                    └─────────▲─────────┘         │  │
│  └───────────┼──────────────────────────────────────────────┼────────────────────┘  │
│              │                                              │                       │
│              │ STEP 4: Return completion                    │                       │
│              ▼                                              │                       │
│  ┌──────────────────┐                                       │                       │
│  │   GSM8k Server   │───────────────────────────────────────┘                       │
│  │   (Scoring)      │                                                               │
│  │                  │  Scores: "15 × 7 = 105" ✓ reward=1.0                         │
│  │                  │          "15 × 7 = 100" ✗ reward=0.0                         │
│  └──────────────────┘                                                               │
│                                                                                     │
│  STEP 7: IMMEDIATE UPDATE                                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │  After optimizer.step(), vLLM's NEXT inference uses the NEW weights!         │   │
│  │  NO SYNC NEEDED - it's the same memory!                                      │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

**Key Points:**
- ✅ ONE copy of weights in GPU memory
- ✅ 0ms sync latency (same memory!)
- ✅ Memory efficient (~1x model size)
- ⚠️ Requires same GPU for trainer and vLLM

---

### LoRA Mode (`--weight-bridge-mode lora_only`)

**The Idea**: Freeze base model, only train small adapter layers. Hot-swap adapters into vLLM.

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                        LORA MODE - COMPLETE DATA FLOW                               │
│                                                                                     │
│  STEP 1: GSM8k sends problem                                                        │
│  ┌──────────────────┐                                                               │
│  │   GSM8k Server   │──── "What is 15 × 7?" ────▶┌──────────────────┐              │
│  │  (Environment)   │                            │   Atropos API    │              │
│  └──────────────────┘                            └────────┬─────────┘              │
│                                                           │                         │
│  STEP 2: Forward to vLLM                                  ▼                         │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │                         vLLM GPU MEMORY                                       │  │
│  │  ┌────────────────────────────────────────────────────────────────────────┐  │  │
│  │  │  BASE MODEL (frozen, ~6GB)                                              │  │  │
│  │  │  + LORA ADAPTER A (current, ~50MB)                                      │  │  │
│  │  └────────────────────────────────────────────────────────────────────────┘  │  │
│  │           │                                                                   │  │
│  │           │ STEP 3: Inference with base + adapter A                          │  │
│  │           ▼                                                                   │  │
│  │  ┌────────────────────┐                                                       │  │
│  │  │   vLLM Server      │ ──── "15 × 7 = 105" ────▶                            │  │
│  │  └────────────────────┘                                                       │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                     │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │                       TRAINER GPU MEMORY (separate!)                          │  │
│  │  ┌────────────────────────────────────────────────────────────────────────┐  │  │
│  │  │  BASE MODEL (frozen, ~6GB)                                              │  │  │
│  │  │  + LORA ADAPTER B (training, ~50MB) ◀── gradients flow here only!       │  │  │
│  │  └────────────────────────────────────────────────────────────────────────┘  │  │
│  │           │                                                                   │  │
│  │           │ STEP 4-5: Receive rollout, compute loss, update adapter B        │  │
│  │           ▼                                                                   │  │
│  │  ┌────────────────────┐                                                       │  │
│  │  │     Trainer        │                                                       │  │
│  │  │    (grpo.py)       │                                                       │  │
│  │  └────────┬───────────┘                                                       │  │
│  └───────────┼──────────────────────────────────────────────────────────────────┘  │
│              │                                                                      │
│              │ STEP 6: Every N steps, save adapter B to disk                       │
│              ▼                                                                      │
│  ┌──────────────────┐     STEP 7: POST /lora/load      ┌──────────────────┐        │
│  │  adapter_step_N/ │ ─────────────────────────────────▶│   vLLM Server    │        │
│  │  (50MB on disk)  │                                   │  Swaps A → B     │        │
│  └──────────────────┘                                   └──────────────────┘        │
│                                                                                     │
│  STEP 8: Next inference uses NEW adapter B                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │  Sync latency: 1-5 seconds (save to disk + HTTP load)                        │   │
│  │  Memory: 2x base model + adapters                                            │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

**Key Points:**
- ✅ Small adapter files (~50MB vs ~28GB)
- ✅ Works on separate GPUs
- ✅ Easy to switch between adapters
- ⚠️ 1-5 second sync latency
- ⚠️ 2x base model memory (trainer + vLLM)

---

### Legacy Mode (`--weight-bridge-mode none`)

**The Simple Approach**: Save full checkpoints, restart vLLM to load new weights.

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                       LEGACY MODE - COMPLETE DATA FLOW                              │
│                                                                                     │
│  STEP 1: GSM8k sends problem                                                        │
│  ┌──────────────────┐                                                               │
│  │   GSM8k Server   │──── "What is 15 × 7?" ────▶┌──────────────────┐              │
│  │  (Environment)   │                            │   Atropos API    │              │
│  └──────────────────┘                            └────────┬─────────┘              │
│                                                           │                         │
│  STEP 2: Forward to vLLM                                  ▼                         │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │                         vLLM GPU MEMORY                                       │  │
│  │  ┌────────────────────────────────────────────────────────────────────────┐  │  │
│  │  │  FULL MODEL - Version 1 (~28GB)                                         │  │  │
│  │  └────────────────────────────────────────────────────────────────────────┘  │  │
│  │           │                                                                   │  │
│  │           │ STEP 3: Inference                                                │  │
│  │           ▼                                                                   │  │
│  │  ┌────────────────────┐                                                       │  │
│  │  │   vLLM Server      │ ──── "15 × 7 = 105" ────▶                            │  │
│  │  └────────────────────┘                                                       │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                     │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │                       TRAINER GPU MEMORY (separate!)                          │  │
│  │  ┌────────────────────────────────────────────────────────────────────────┐  │  │
│  │  │  FULL MODEL - Version 2 (~28GB + gradients + optimizer)                 │  │  │
│  │  └────────────────────────────────────────────────────────────────────────┘  │  │
│  │           │                                                                   │  │
│  │           │ STEP 4-5: Receive rollout, compute loss, update weights          │  │
│  │           ▼                                                                   │  │
│  │  ┌────────────────────┐                                                       │  │
│  │  │     Trainer        │                                                       │  │
│  │  │    (grpo.py)       │                                                       │  │
│  │  └────────┬───────────┘                                                       │  │
│  └───────────┼──────────────────────────────────────────────────────────────────┘  │
│              │                                                                      │
│              │ STEP 6: Every N steps, save FULL checkpoint to disk (~28GB)         │
│              ▼                                                                      │
│  ┌──────────────────┐                                                               │
│  │  checkpoint/     │                                                               │
│  │  step_N/         │ (28GB on disk!)                                              │
│  │  - model.safetensors                                                            │
│  │  - config.json                                                                  │
│  └────────┬─────────┘                                                               │
│           │                                                                         │
│           │ STEP 7: RESTART vLLM with new checkpoint                               │
│           │                                                                         │
│           │  ┌─────────────────────────────────────────────────────────────────┐   │
│           │  │  1. Kill vLLM process                                            │   │
│           │  │  2. Start new vLLM with --model checkpoint/step_N/               │   │
│           │  │  3. Wait for model to load (~30-60 seconds)                      │   │
│           │  │  4. Resume training                                              │   │
│           │  └─────────────────────────────────────────────────────────────────┘   │
│           ▼                                                                         │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │                         vLLM GPU MEMORY (restarted)                           │  │
│  │  ┌────────────────────────────────────────────────────────────────────────┐  │  │
│  │  │  FULL MODEL - Version 2 (loaded from checkpoint)                        │  │  │
│  │  └────────────────────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                     │
│  STEP 8: Next inference uses updated model                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │  Sync latency: 30-60 seconds (save + restart + reload)                       │   │
│  │  Memory: 2x full model                                                       │   │
│  │  Disk: 28GB per checkpoint                                                   │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
```
---

## Mode Comparison Summary

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         MODE COMPARISON AT A GLANCE                              │
├────────────────┬───────────────┬────────────────┬────────────────────────────────┤
│                │ SINGLE-COPY   │ LORA           │ LEGACY                         │
├────────────────┼───────────────┼────────────────┼────────────────────────────────┤
│ Sync Latency   │ 0 ms ⚡       │ 1-5 sec        │ 30-60 sec                      │
│ GPU Memory     │ 1x model      │ 2x model       │ 2x model                       │
│ Disk Space     │ 28GB/ckpt     │ 50MB/adapter   │ 28GB/ckpt                      │
│ Complexity     │ Medium        │ Medium         │ Simple                         │
│ Same GPU?      │ Required      │ Optional       │ Optional                       │
│ Best For       │ Production    │ Experiments    │ Debugging                      │
└────────────────┴───────────────┴────────────────┴────────────────────────────────┘
```

---

## Alternative Mode Commands

### Legacy Mode (Checkpoint + Restart)

For simple setups or debugging. Saves checkpoints and restarts vLLM to load new weights.

```bash
python example_trainer/grpo.py \
    --model-name Qwen/Qwen2.5-3B-Instruct \
    --weight-bridge-mode none \
    --training-steps 100 \
    --vllm-restart-interval 10 \
    --batch-size 2 \
    --lr 1e-5
```

### LoRA Mode (Adapter Training)

Trains only adapter weights. Small checkpoints, lower memory.

```bash
python example_trainer/grpo.py \
    --model-name Qwen/Qwen2.5-3B-Instruct \
    --weight-bridge-mode lora_only \
    --lora-r 16 \
    --lora-alpha 32 \
    --training-steps 100 \
    --batch-size 2 \
    --lr 1e-4
```

---

## Configuration Reference

### Environment Variables

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `VLLM_ENABLE_SHARED_WEIGHTS` | Yes (single-copy) | Enable vLLM patching for IPC | `1` |
| `VLLM_SKIP_WEIGHT_DAEMON` | Yes (single-copy) | Skip NCCL daemon (not needed) | `1` |
| `NUM_INFERENCE_NODES` | Yes | Number of vLLM nodes (0 = local) | `0` |
| `LOGDIR` | Recommended | Directory for vllm_bridge_config.json | `.` |
| `CUDA_VISIBLE_DEVICES` | Recommended | GPU allocation | `0` |

### Trainer CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model-name` | (required) | HuggingFace model ID |
| `--weight-bridge-mode` | `none` | `none`, `shared_vllm`, or `lora_only` |
| `--single-copy` | `false` | Enable TRUE single-copy mode via CUDA IPC |
| `--vllm-config-path` | (auto-detect) | Explicit path to `vllm_bridge_config.json` |
| `--vllm-port` | `9001` | vLLM server port |
| `--training-steps` | `10` | Total optimization steps |
| `--batch-size` | `2` | Micro-batch size |
| `--lr` | `1e-5` | Learning rate |
| `--save-path` | `trained_model_checkpoints` | Checkpoint directory |

### vLLM Server Options

| Option | Description |
|--------|-------------|
| `--model` | HuggingFace model ID |
| `--tensor-parallel-size` | Number of GPUs (use 1 for single-copy) |
| `--port` | Server port (default: 9001) |
| `--dtype` | Model dtype (`bfloat16`, `float16`, `auto`) |
| `--gpu-memory-utilization` | Fraction of GPU memory for KV cache (default: 0.9) |

---

## The vLLM Bridge Config (vllm_bridge_config.json)

The `vllm_bridge_config.json` file is the critical communication mechanism between the vLLM inference server and the GRPO trainer in single-copy mode. Understanding this file is essential for debugging and advanced configurations.

### What It Is

When you start vLLM with `VLLM_ENABLE_SHARED_WEIGHTS=1`, the patched `GPUModelRunner` exports CUDA IPC (Inter-Process Communication) handles for all model tensors. These handles allow another process (the trainer) to access the exact same GPU memory—no copying required.

### Why It's Important

1. **True Single-Copy Architecture**: Instead of loading the model twice (once for training, once for inference), both processes share the same tensors in GPU memory.

2. **Zero-Latency Weight Updates**: When `optimizer.step()` modifies the weights, vLLM immediately sees the changes—no serialization, no network transfer, no disk I/O.

3. **Memory Efficiency**: For a 7B model (~14GB in bf16), you save ~14GB of GPU memory compared to having two separate copies.

### File Location

The trainer searches for `vllm_bridge_config.json` in this order:

1. **Explicit path** (if `--vllm-config-path` is provided)
2. **`$LOGDIR/vllm_bridge_config.json`** (if `LOGDIR` env var is set)
3. **`./vllm_bridge_config.json`** (current directory)
4. **`/tmp/atropos_bridge/vllm_bridge_config.json`** (default fallback)

**Tip**: To avoid "Config not found" errors, always set `LOGDIR`:

```bash
export LOGDIR=.
```

### File Contents

The JSON file contains everything needed to reconstruct tensor references in another process:

```json
{
  "model": "Qwen/Qwen2.5-3B-Instruct",
  "tp_degree": 1,
  "dp_shard_degree": 1,

  "param_names": [
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.qkv_proj.weight",
    ...
  ],

  "param_mappings": {
    "model.embed_tokens.weight": {
      "vllm_name": "model.embed_tokens.weight",
      "shape": [152064, 2048],
      "dtype": "torch.bfloat16",
      "device": "cuda:0"
    },
    ...
  },

  "ipc_handles": {
    "model.embed_tokens.weight": {
      "device_index": 0,
      "ipc_handle_b64": "AmPA0pN...",
      "storage_size": 623902720,
      "storage_offset": 0,
      "ref_counter_handle_b64": "Y2JY...",
      "ref_counter_offset": 0,
      "event_handle_b64": "wRIs...",
      "event_sync_required": true,
      "shape": [152064, 2048],
      "dtype": "torch.bfloat16"
    },
    ...
  },

  "shared_weights_enabled": true,
  "single_copy_enabled": true,
  "num_params": 255
}
```

#### Field Descriptions

| Field | Description |
|-------|-------------|
| `model` | HuggingFace model identifier |
| `tp_degree` | Tensor parallel degree (must be 1 for single-copy) |
| `param_names` | List of all parameter names in the model |
| `param_mappings` | Shape, dtype, and device info for each parameter |
| `ipc_handles` | CUDA IPC handles for reconstructing shared tensors |
| `ipc_handle_b64` | The actual CUDA IPC handle (base64-encoded bytes) |
| `ref_counter_handle_b64` | Reference counter for CUDA memory (base64) |
| `event_handle_b64` | CUDA event handle for synchronization (base64) |
| `storage_size` | Size of the underlying storage in bytes |

### How the Trainer Uses It

1. **Load Config**: Trainer reads `vllm_bridge_config.json`
2. **Create Shell Model**: Uses `AutoModelForCausalLM.from_config()` with meta tensors (no memory allocation)
3. **Attach IPC Handles**: For each parameter, reconstructs the tensor using `torch.UntypedStorage._new_shared_cuda()` with the IPC handles
4. **Verify Shapes**: Ensures trainer's model architecture matches vLLM's sharding

```python
# Simplified version of what happens internally:
for name, ipc_info in config["ipc_handles"].items():
    # Decode IPC handle from base64
    ipc_handle = base64.b64decode(ipc_info["ipc_handle_b64"])

    # Reconstruct storage from IPC handle
    storage = torch.UntypedStorage._new_shared_cuda(
        device_index, ipc_handle, storage_size, ...
    )

    # Create tensor from shared storage
    tensor = torch.tensor(storage).view(shape).to(dtype)

    # Replace model parameter with shared tensor
    model.get_parameter(name).data = tensor
```

### Specifying the Config Path Explicitly

If auto-detection isn't working (e.g., in complex cluster setups), you can specify the path explicitly:

```bash
# If vLLM writes config to a non-standard location:
python -u example_trainer/grpo.py \
    --model-name Qwen/Qwen2.5-3B-Instruct \
    --weight-bridge-mode shared_vllm \
    --single-copy \
    --vllm-config-path /shared/nfs/vllm_bridge_config.json \
    --training-steps 50
```

### Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| "Could not find vllm_bridge_config.json" | vLLM didn't export config | Check `VLLM_ENABLE_SHARED_WEIGHTS=1` was set BEFORE starting vLLM |
| Config exists but has empty `ipc_handles` | Patch didn't run | Ensure vLLM is using our custom `vllm_api_server.py` |
| "tuple of 8 items expected" | IPC handle format mismatch | Update to latest code (handles all 8 CUDA IPC tuple components) |
| "size mismatch" errors | Tensor parallel mismatch | Use `tensor-parallel-size 1` for single-copy mode |

---

## FAQ & Troubleshooting

### Q: I get "Could not find vllm_bridge_config.json"

**A:** vLLM didn't export the IPC handles. Check:

1. `VLLM_ENABLE_SHARED_WEIGHTS=1` was set **before** starting vLLM
2. `LOGDIR` is set to a valid, writable directory
3. Look for export messages in vllm.log:
```bash
grep "Exported" vllm.log
```

If the file exists but in a different location, specify it explicitly:
```bash
python grpo.py ... --vllm-config-path /path/to/vllm_bridge_config.json
```

---

### Q: I get "CUDA out of memory" when starting the trainer

**A:** For single-copy mode, trainer and vLLM MUST be on the same GPU(s). Check:

```bash
# Both should use the same CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=0 python ... vllm_api_server.py ...
CUDA_VISIBLE_DEVICES=0 python ... grpo.py ...
```

---

### Q: Trainer crashes with "Cannot copy out of meta tensor"

**A:** Some model buffers (like rotary embeddings) weren't initialized. This is a known issue being fixed. Update to the latest code.

---

### Q: Single-copy mode doesn't work with tensor-parallel > 1

**A:** Currently, single-copy mode only works with `tensor-parallel-size 1`. For larger models that need tensor parallelism, use a single GPU with a smaller model, or wait for multi-GPU single-copy support.

---

### Q: How do I check GPU memory usage?

**A:**
```bash
nvidia-smi
```

For single-copy mode with Qwen2.5-14B:
- GPU 0: ~28GB (shared between vLLM and trainer)

---

### Q: How do I stop all processes?

**A:**
```bash
pkill -9 -u $USER -f "vllm|grpo|python|run-api"
```

---

## Files in This Directory

| File | Description |
|------|-------------|
| `grpo.py` | Main trainer script with all modes |
| `vllm_api_server.py` | Custom vLLM server with shared memory patches |
| `vllm_patching/` | vLLM patches for CUDA IPC support |
| `requirements.txt` | Python dependencies |
| `README.md` | This documentation |

### vllm_patching/ Directory

| File | Description |
|------|-------------|
| `__init__.py` | Module exports and patch application |
| `patched_gpu_runner.py` | Patches GPUModelRunner to export CUDA IPC handles |

---

## Performance Comparison

| Mode | Sync Latency | Memory (14B model) | Best For |
|------|--------------|-------------------|----------|
| **Legacy** | 30-60s | 2x model | Debugging |
| **Single-Copy** | 0ms | 1x model (shared!) | Production |
| **LoRA** | 5-10s | 1x model + adapters | Memory-constrained |

---

## Checkpoint Locations

| Mode | Location | Size |
|------|----------|------|
| Legacy | `trained_model_checkpoints/step_N/` | ~28GB (14B model) |
| Single-Copy | `trained_model_checkpoints/step_N/` | ~28GB |
| LoRA | `trained_model_checkpoints/adapter_step_N/` | ~50MB |

---

## Feature Availability Matrix

### What's Available 

| Feature | Status | Notes |
|---------|--------|-------|
| **Single-Copy Mode** |  Working | True shared memory via CUDA IPC |
| **LoRA Mode** |  Working | Hot-swap adapters without restart |
| **Legacy Mode** |  Working | Checkpoint-based, restart vLLM |
| **Qwen Models** |  Working | Qwen2, Qwen2.5 (0.5B - 72B) |
| **Llama Models** |  Working | Llama-2, Llama-3, Llama-3.1 |
| **Mistral Models** |  Working | Mistral-7B, Mixtral |
| **Single GPU** |  Working | All modes supported |
| **bfloat16/float16** |  Working | Configurable via `--dtype` |
| **Gradient Checkpointing** |  Available | Reduces memory usage |
| **Wandb Logging** |  Working | Via `--use-wandb` flag |
| **Custom Environments** |  Working | Extend `BaseEnv` class |

### What's NOT Available 

| Feature | Mode | Status | Reason / Workaround |
|---------|------|--------|---------------------|
| **Multi-GPU (TP > 1)** | Single-Copy |  Not Supported | CUDA IPC handles are per-device; sharding complicates sharing |
| **Multi-GPU (TP > 1)** | LoRA |  Supported | vLLM handles TP, trainer only swaps adapters |
| **Multi-GPU (TP > 1)** | Legacy |  Supported | Standard vLLM with TP supported |
| **Pipeline Parallel** | Single-Copy |  Not Supported | Would need cross-device IPC |
| **Pipeline Parallel** | LoRA/Legacy |  Via vLLM | Use `--pipeline-parallel-size` flag |
| **Data Parallel** | Single-Copy |  Not Supported | Shared tensors can't be safely updated by multiple trainers |
| **Data Parallel** | LoRA/Legacy |  Manual | Run multiple trainer instances (see docs below) |
| **Multi-Node** | Single-Copy |  Not Supported | CUDA IPC is single-node only |
| **Multi-Node** | LoRA/Legacy |  Via vLLM | vLLM supports distributed inference |
| **DeepSpeed/FSDP** | All |  Not Integrated | Would require custom integration with trainer |
| **Quantized Models** | Single-Copy |  Not Supported | IPC handles may not work with quantized tensors |
| **Quantized Models** | LoRA/Legacy |  Supported | Standard vLLM quantization (GPTQ, AWQ, etc.) |
| **Encoder-Decoder** | All |  Not Supported | Architecture not supported by vLLM |

### Multi-GPU Support Summary

| Mode | Tensor Parallel | Pipeline Parallel | Data Parallel |
|------|-----------------|-------------------|---------------|
| **Single-Copy** |  TP=1 only |  Not Supported |  Not Supported |
| **LoRA** |  Supported |  Via vLLM |  Multiple Trainers |
| **Legacy** |  Supported |  Via vLLM |  Multiple Trainers |

> **Key Point**: The multi-GPU limitation is **ONLY for single-copy mode** due to CUDA IPC constraints. 
> LoRA and Legacy modes work with standard vLLM which fully supports tensor parallelism.

#### Pipeline Parallel (PP)

vLLM supports pipeline parallelism via `--pipeline-parallel-size`. For LoRA/Legacy modes:

```bash
# LoRA/Legacy with Pipeline Parallel (2 GPUs for PP)
python -u example_trainer/vllm_api_server.py \
    --model Qwen/Qwen2.5-14B-Instruct \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 2 \
    --port 9001
```

**Note**: PP requires the model to be split across GPUs by layers. Performance may vary.

#### Data Parallel (DP)

Data parallelism means running **multiple trainer instances** against the same vLLM server. Each trainer processes different batches:

```bash
# Terminal 1: First trainer instance
CUDA_VISIBLE_DEVICES=4 python -u example_trainer/grpo.py \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --weight-bridge-mode lora_only \
    --trainer-rank 0 \
    --world-size 2 \
    > trainer_0.log 2>&1 &

# Terminal 2: Second trainer instance
CUDA_VISIBLE_DEVICES=5 python -u example_trainer/grpo.py \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --weight-bridge-mode lora_only \
    --trainer-rank 1 \
    --world-size 2 \
    > trainer_1.log 2>&1 &
```

**Note**: DP requires gradient synchronization between trainers. Currently, trainers operate independently - true distributed DP would need additional coordination.

### GPU Support

| GPU Type | Single-Copy | LoRA | Legacy | Notes |
|----------|-------------|------|--------|-------|
| **NVIDIA A100** | YES | YES | YES | Recommended |
| **NVIDIA H100** | YES | YES | YES | Recommended |
| **NVIDIA B200** | YES | YES | YES | Recommended |
| **NVIDIA RTX 4090** | YES | YES | YES | Consumer, works well |
| **NVIDIA RTX 3090** | YES | YES | YES | Consumer, works well |
| **NVIDIA V100** | ? | YES | YES | Old, may have IPC issues |

### Memory Requirements (Approximate)

| Model Size | Single-Copy | LoRA | Legacy |
|------------|-------------|------|--------|
| 0.5B - 1B | 4-6 GB | 4-6 GB | 8-12 GB |
| 3B | 8-12 GB | 8-12 GB | 16-24 GB |
| 7B | 16-20 GB | 16-20 GB | 32-40 GB |
| 14B | 32-40 GB | 32-40 GB | 64-80 GB |
| 32B | 70-80 GB | 70-80 GB | 140+ GB |
| 70B+ | Single GPU impossible | 80+ GB | 160+ GB |

> **Note**: Single-copy mode uses ~50% less memory than legacy because there's only ONE model copy.

---

## Known Limitations

### Single-Copy Mode Specific

| Limitation | Reason | Workaround |
|------------|--------|------------|
| **Same GPU Required** | CUDA IPC only works within same physical device | Use same `CUDA_VISIBLE_DEVICES` for trainer and vLLM |
| **TP=1 Only** | Trainer expects unsharded model; IPC per-device | Use LoRA mode for TP > 1 |
| **Custom Server Required** | Standard `vllm serve` doesn't export IPC handles | Use `vllm_api_server.py` |
| **Single Node Only** | CUDA IPC is node-local | Use LoRA/Legacy for multi-node |

### LoRA Mode Specific

| Limitation | Reason | Workaround |
|------------|--------|------------|
| **~5s Swap Latency** | Adapter weights need to be loaded | Acceptable for most use cases |
| **vLLM LoRA Support Required** | Model must support LoRA in vLLM | Check vLLM documentation |

### General Limitations

| Limitation | Reason | Workaround |
|------------|--------|------------|
| **GSM8k Needs `server_type=vllm`** | Default `openai` type lacks state tracking | Use `--openai.server_type vllm` |
| **Decoder-Only Models Only** | vLLM architecture constraint | Use different framework for encoder-decoder |
| **Custom vLLM Server Required** | Standard `vllm serve` lacks IPC patches | Use `vllm_api_server.py` for all modes |

---

## Future Work

### High Priority 

| Feature | Description |
|---------|-------------|
| **Multi-GPU Single-Copy** | Support `tensor-parallel-size > 1` with sharded IPC |
| **Automatic Server Type Detection** | Auto-detect correct `server_type` for environments |
| **Checkpoint Resume** | Resume training from checkpoints seamlessly |

### Medium Priority 

| Feature | Description | Difficulty |
|---------|-------------|------------|
| **DeepSpeed Integration** | ZeRO optimization for larger models | Hard |
| **Quantization Support** | Test and document GPTQ/AWQ in single-copy | Medium |
| **Multi-Node Training** | Distributed training across nodes | Hard |
| **Streaming Weights** | Stream weight updates instead of full sync | Medium |
| **Mixed Precision Training** | Support fp8/int8 training | Medium |


## Contributing

We welcome contributions! Priority areas:

1. **Multi-GPU single-copy support** - The biggest missing feature
2. **Better documentation** - More examples, tutorials
3. **Environment implementations** - New RL environments
4. **Bug fixes** - Especially edge cases in IPC handling

See the main repository CONTRIBUTING.md for guidelines.
