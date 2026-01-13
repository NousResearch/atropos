# GRPO Example Trainer

This directory contains an example script (`grpo.py`) demonstrating how to integrate a custom training loop with the Atropos API for reinforcement learning using the GRPO (Group Relative Policy Optimization) algorithm.

## Training Modes

The trainer supports three weight synchronization modes:

| Mode | Description | Sync Latency | Best For |
|------|-------------|--------------|----------|
| **Legacy** (`none`) | Save checkpoints, restart vLLM | ~30-60 seconds | Simple setups, debugging |
| **Shared vLLM** (`shared_vllm`) | Direct shared memory updates via NCCL | ~0 ms | Production, maximum throughput |
| **LoRA** (`lora_only`) | Train adapters, hot-swap | ~1-5 seconds | Memory-constrained, fast iteration |

---

## Quick Start with GSM8k (Shared vLLM Mode)

This is the **recommended** production setup for maximum training throughput.

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
│                    SHARED VLLM TRAINING ARCHITECTURE                        │
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────────┐ │
│  │ GSM8k Env   │───▶│ Atropos API │◀───│ GRPO Trainer (GPU 2)            │ │
│  │ (problems)  │    │ (batching)  │    │ - Loads model for training      │ │
│  └─────────────┘    └─────────────┘    │ - Broadcasts weights via NCCL   │ │
│         │                              └─────────────────────────────────┘ │
│         │                                              │                    │
│         │                                              │ NCCL Broadcast     │
│         ▼                                              ▼                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              vLLM Inference Server (GPUs 0-1)                        │   │
│  │         - Model weights in shared memory                             │   │
│  │         - Weight updater threads receive NCCL updates               │   │
│  │         - Generates rollouts for scoring                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Step-by-Step Guide (Tested & Working)

**IMPORTANT: GPU Allocation**
- vLLM runs on GPUs 0-1 (tensor-parallel)
- Trainer runs on GPU 2 (separate to avoid OOM)

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
export NUM_INFERENCE_NODES=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500
```

#### Step 4: Start Atropos API

```bash
python -m atroposlib.cli.run_api > api.log 2>&1 &
echo "Atropos API started"
sleep 3
```

#### Step 5: Start GSM8K Environment

```bash
python environments/gsm8k_server.py > gsm8k.log 2>&1 &
echo "GSM8K environment started"
sleep 3
```

#### Step 6: Start vLLM Server on GPUs 0-1

```bash
CUDA_VISIBLE_DEVICES=0,1 python -u example_trainer/vllm_api_server.py \
    --model Qwen/Qwen2.5-14B-Instruct \
    --tensor-parallel-size 2 \
    --port 9001 \
    --dtype bfloat16 \
    > vllm.log 2>&1 &
echo "vLLM starting on GPUs 0,1..."
```

#### Step 7: Wait for vLLM to Load

```bash
tail -f vllm.log
```

Wait until you see: `Uvicorn running on http://0.0.0.0:9001`

Then press **Ctrl+C** to stop tailing.

#### Step 8: Verify Shared Memory Setup

```bash
grep -E "thread|updater|Exported|Shared memory" vllm.log
```

You should see:
```
[vLLM Patch] ✓ Shared memory setup complete!
[vLLM Patch] ✓ Weight updater thread started (name: WeightUpdater_TP0)
[vLLM Patch] ✓ Weight updater thread started (name: WeightUpdater_TP1)
```

#### Step 9: Start Trainer on GPU 2

```bash
CUDA_VISIBLE_DEVICES=2 python -u example_trainer/grpo.py \
    --model-name Qwen/Qwen2.5-14B-Instruct \
    --weight-bridge-mode shared_vllm \
    --vllm-port 9001 \
    --lr 1e-6 \
    --batch-size 4 \
    --training-steps 100 \
    --use-shared-memory \
    2>&1 | tee trainer.log
```

#### Step 10: Monitor Training

```bash
tail -f trainer.log
```

You should see:
```
[Bridge] ✓ Gloo group created
[Bridge] ✓ NCCL group created
[Bridge] ✓ All ranks synchronized and ready
[Bridge] Mapped 195/339 params from vLLM to trainer
Step 1/100
```

---

### Quick Copy-Paste (All-in-One)

```bash
# Kill everything and setup
pkill -9 -u $USER -f "vllm|grpo|python|run-api" 2>/dev/null; sleep 3
cd ~/atropos_stuff/atropos
rm -f vllm_bridge_config.json vllm.log trainer.log api.log gsm8k.log

# Environment variables
export VLLM_ENABLE_SHARED_WEIGHTS=1 NUM_INFERENCE_NODES=0 MASTER_ADDR=localhost MASTER_PORT=29500

# Start services
python -m atroposlib.cli.run_api > api.log 2>&1 &
sleep 3
python environments/gsm8k_server.py > gsm8k.log 2>&1 &
sleep 3
CUDA_VISIBLE_DEVICES=0,1 python -u example_trainer/vllm_api_server.py --model Qwen/Qwen2.5-14B-Instruct --tensor-parallel-size 2 --port 9001 --dtype bfloat16 > vllm.log 2>&1 &

echo "Waiting for vLLM to load... (check: tail -f vllm.log)"
echo "Once ready, run the trainer command below:"
echo ""
echo "CUDA_VISIBLE_DEVICES=2 python -u example_trainer/grpo.py --model-name Qwen/Qwen2.5-14B-Instruct --weight-bridge-mode shared_vllm --vllm-port 9001 --lr 1e-6 --batch-size 4 --training-steps 100 --use-shared-memory 2>&1 | tee trainer.log"
```

---

## How Shared vLLM Mode Works

### The Problem
Traditional RL training requires syncing model weights between the trainer and inference server. This is slow:
- Save checkpoint → Load into vLLM → Restart server = **30-60 seconds per sync**

### Two Solutions Available

#### Option 1: Broadcast Mode (`--use-shared-memory`)
Two copies of the model, but instant NCCL sync. Use when trainer is on **different GPUs**.

```
Trainer (GPU 2)              NCCL               vLLM Workers (GPUs 0-1)
     │                         │                        │
     │ optimizer.step()        │                        │
     │ ─────────────────────────────────────────────►   │
     │   broadcast_weights()   │                        │ Thread receives
     │                         │                        │ weights via NCCL
     │                         │                        │ Copies to shared
     │                         │                        │ memory tensors
     │                         │                        │
     │ Next training step      │                        │ Ready for inference
```

- **Memory**: 2x model size (trainer copy + vLLM copy)
- **Sync Latency**: ~0ms (NCCL broadcast)
- **GPU Layout**: Trainer on different GPUs than vLLM

#### Option 2: Single-Copy Mode (`--single-copy`) ⭐ RECOMMENDED
TRUE shared memory - only ONE copy of the model! Use when trainer is on **same GPUs**.

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
- **GPU Layout**: Trainer on SAME GPUs as vLLM (required!)

### When to Use Which

| Mode | Memory | Sync | Use When |
|------|--------|------|----------|
| **Broadcast** (`--use-shared-memory`) | 2x model | ~0ms NCCL | Trainer on different GPUs |
| **Single-Copy** (`--single-copy`) | 1x model | 0ms | Trainer on same GPUs, memory constrained |

### Single-Copy Mode Usage

```bash
# vLLM and Trainer on SAME GPUs (0,1)
CUDA_VISIBLE_DEVICES=0,1 python -u example_trainer/vllm_api_server.py \
    --model Qwen/Qwen2.5-14B-Instruct \
    --tensor-parallel-size 2 \
    --port 9001 \
    > vllm.log 2>&1 &

# Wait for vLLM to load...

# Trainer also on GPUs 0,1 - shares vLLM's tensors!
CUDA_VISIBLE_DEVICES=0,1 python -u example_trainer/grpo.py \
    --model-name Qwen/Qwen2.5-14B-Instruct \
    --weight-bridge-mode shared_vllm \
    --single-copy \
    --training-steps 100 \
    2>&1 | tee trainer.log
```

---

## Alternative Modes

### Mode 1: Legacy (Checkpoint + Restart)

For simple setups or debugging. Saves checkpoints and can restart vLLM.

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
| `VLLM_ENABLE_SHARED_WEIGHTS` | Yes (shared mode) | Enable vLLM patching | `1` |
| `NUM_INFERENCE_NODES` | Yes | Number of vLLM nodes (0 = local) | `0` |
| `MASTER_ADDR` | Yes | Rendezvous address | `localhost` |
| `MASTER_PORT` | Yes | Rendezvous port | `29500` |
| `CUDA_VISIBLE_DEVICES` | Recommended | GPU allocation | `0,1` or `2` |

### Trainer CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model-name` | (required) | HuggingFace model ID |
| `--weight-bridge-mode` | `none` | `none`, `shared_vllm`, or `lora_only` |
| `--use-shared-memory` | `False` | Enable NCCL weight broadcasting |
| `--vllm-port` | `9001` | vLLM server port |
| `--training-steps` | `10` | Total optimization steps |
| `--batch-size` | `2` | Micro-batch size |
| `--lr` | `1e-5` | Learning rate |
| `--save-path` | `trained_model_checkpoints` | Checkpoint directory |

### vLLM Server Options

| Option | Description |
|--------|-------------|
| `--model` | HuggingFace model ID |
| `--tensor-parallel-size` | Number of GPUs for tensor parallelism |
| `--port` | Server port (default: 9001) |
| `--dtype` | Model dtype (`bfloat16`, `float16`, `auto`) |

---

## FAQ & Troubleshooting

### Q: The trainer is stuck at "Creating Gloo process group..."

**A:** This means the trainer is waiting for the vLLM weight updater threads to connect. Check if the threads started:

```bash
grep -E "thread|updater|ERROR" vllm.log
```

You should see:
```
[vLLM Patch] ✓ Weight updater thread started (name: WeightUpdater_TP0)
[vLLM Patch] ✓ Weight updater thread started (name: WeightUpdater_TP1)
```

If not, ensure `VLLM_ENABLE_SHARED_WEIGHTS=1` was set **before** starting vLLM.

---

### Q: I get "CUDA out of memory" when starting the trainer

**A:** The trainer is trying to load the model on the same GPUs as vLLM. Use separate GPUs:

```bash
# vLLM on GPUs 0-1
CUDA_VISIBLE_DEVICES=0,1 python -u example_trainer/vllm_api_server.py ...

# Trainer on GPU 2
CUDA_VISIBLE_DEVICES=2 python -u example_trainer/grpo.py ...
```

---

### Q: I see "daemonic processes are not allowed to have children"

**A:** This was a bug in older versions. The fix uses **threads** instead of **processes** for the weight updater. Make sure you have the latest `patched_gpu_runner.py`.

---

### Q: The `vllm_bridge_config.json` shows `param_mappings: {}`

**A:** The vLLM patches didn't run. Check:

1. `VLLM_ENABLE_SHARED_WEIGHTS=1` was set before starting vLLM
2. Look for `[vLLM Patch] ✓ Exported X params` in vllm.log

```bash
grep "Exported" vllm.log
```

---

### Q: How do I verify the NCCL connection is working?

**A:** Check the trainer log for these messages:

```
[Bridge] ✓ Gloo group created
[Bridge] ✓ NCCL group created
[Bridge] ✓ All ranks synchronized and ready
```

---

### Q: What's the difference between Gloo and NCCL?

**A:** 
- **Gloo**: CPU-based coordination protocol. Used for synchronization barriers.
- **NCCL**: GPU-based high-speed protocol. Used for broadcasting weight tensors.

Both are needed: Gloo for coordination, NCCL for fast tensor transfers.

---

### Q: How do I check GPU memory usage?

**A:**
```bash
nvidia-smi
```

Expected for Qwen2.5-14B with shared mode:
- GPUs 0-1: ~168GB each (vLLM workers)
- GPU 2: ~29GB (trainer)

---

### Q: How do I stop all processes?

**A:**
```bash
pkill -9 -u $USER -f "vllm|grpo|python|run-api"
```

---

### Q: The training is slow / not progressing

**A:** Check if all services are running:

```bash
ps aux | grep -E "(run_api|vllm|grpo|gsm8k)" | grep $USER
```

Check logs for errors:
```bash
tail -20 api.log
tail -20 gsm8k.log
tail -20 vllm.log
tail -20 trainer.log
```

---

### Q: How do I use a smaller model for testing?

**A:** Use Qwen2.5-3B-Instruct with single GPU:

```bash
# vLLM on GPU 0
CUDA_VISIBLE_DEVICES=0 python -u example_trainer/vllm_api_server.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --port 9001 \
    > vllm.log 2>&1 &

# Trainer on GPU 1
CUDA_VISIBLE_DEVICES=1 python -u example_trainer/grpo.py \
    --model-name Qwen/Qwen2.5-3B-Instruct \
    --weight-bridge-mode shared_vllm \
    --use-shared-memory \
    --training-steps 10 \
    2>&1 | tee trainer.log
```

---

## Files in This Directory

| File | Description |
|------|-------------|
| `grpo.py` | Main trainer script with all modes |
| `vllm_api_server.py` | Custom vLLM server with shared memory patches |
| `vllm_weight_bridge.py` | NCCL bridge for weight synchronization |
| `vllm_patching/` | vLLM patches for shared memory support |
| `requirements.txt` | Python dependencies |
| `README.md` | This documentation |

### vllm_patching/ Directory

| File | Description |
|------|-------------|
| `__init__.py` | Module exports |
| `patched_gpu_runner.py` | Patches GPUModelRunner for shared memory |
| `weight_updater.py` | Thread that receives NCCL weight broadcasts |
| `distributed_utils.py` | Process group initialization helpers |

---

## Performance Comparison

| Mode | Sync Latency | Memory (14B model) | Best For |
|------|--------------|-------------------|----------|
| **Legacy** | 30-60s | 2x model | Debugging |
| **Shared vLLM** | ~0ms | 1x model (shared) + trainer | Production |
| **LoRA** | 5-10s | 1x model + adapters | Memory-constrained |

---

## Checkpoint Locations

| Mode | Location | Size |
|------|----------|------|
| Legacy | `trained_model_checkpoints/step_N/` | ~28GB (14B model) |
| Shared vLLM | `trained_model_checkpoints/step_N/` | ~28GB |
| LoRA | `trained_model_checkpoints/adapter_step_N/` | ~50MB |

---

## Example Training Runs

### Quick Test (3B model, LoRA)
```bash
python example_trainer/grpo.py \
    --model-name Qwen/Qwen2.5-3B-Instruct \
    --weight-bridge-mode lora_only \
    --training-steps 5 \
    --batch-size 1
```

### Production (14B model, Shared vLLM)
```bash
# See Step-by-Step Guide above
CUDA_VISIBLE_DEVICES=2 python -u example_trainer/grpo.py \
    --model-name Qwen/Qwen2.5-14B-Instruct \
    --weight-bridge-mode shared_vllm \
    --use-shared-memory \
    --training-steps 1000 \
    --batch-size 4 \
    --lr 1e-6
```

### Multi-GPU Training (70B model)
```bash
# vLLM on GPUs 0-3 (tensor parallel 4)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u example_trainer/vllm_api_server.py \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 \
    --port 9001 \
    > vllm.log 2>&1 &

# Trainer on GPUs 4-5
CUDA_VISIBLE_DEVICES=4,5 python -u example_trainer/grpo.py \
    --model-name Qwen/Qwen2.5-72B-Instruct \
    --weight-bridge-mode shared_vllm \
    --use-shared-memory \
    --training-steps 100 \
    2>&1 | tee trainer.log
```
