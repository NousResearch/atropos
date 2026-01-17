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

#### Step 7: Start GSM8K Environment

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

## Alternative Modes

### Mode 1: Legacy (Checkpoint + Restart)

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

### Mode 2: LoRA Adapters

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

---

## FAQ & Troubleshooting

### Q: I get "Could not find vllm_bridge_config.json"

**A:** vLLM didn't export the IPC handles. Check:

1. `VLLM_ENABLE_SHARED_WEIGHTS=1` was set **before** starting vLLM
2. Look for export messages in vllm.log:
```bash
grep "Exported" vllm.log
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
| `patched_gpu_runner.py` | Patches GPUModelRunner to export IPC handles |
| `distributed_utils.py` | Distributed training utilities |

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
