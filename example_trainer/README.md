# GRPO Example Trainer

This directory contains an example script (`grpo.py`) demonstrating how to integrate a custom training loop with the Atropos API for reinforcement learning using the GRPO (Group Relative Policy Optimization) algorithm.

## Training Modes

The trainer supports three weight synchronization modes:

| Mode | Description | Sync Latency | Best For |
|------|-------------|--------------|----------|
| **Legacy** (`none`) | Save checkpoints, restart vLLM | ~30-60 seconds | Simple setups, debugging |
| **Shared vLLM** (`shared_vllm`) | Direct shared memory updates | ~0 ms | Production, maximum throughput |
| **LoRA** (`lora_only`) | Train adapters, hot-swap | ~1-5 seconds | Memory-constrained, fast iteration |

---

## Quick Start with GSM8k

### Prerequisites

```bash
# Install dependencies
pip install -r example_trainer/requirements.txt

# Install GSM8k environment dependencies
pip install datasets latex2sympy2_extended math_verify
```

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Training Setup                          │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │ GSM8k Env   │───▶│ Atropos API │◀───│ GRPO Trainer        │ │
│  │ (problems)  │    │ (batching)  │    │ (optimization)      │ │
│  └─────────────┘    └─────────────┘    └─────────────────────┘ │
│         │                                        │              │
│         │                                        │              │
│         ▼                                        ▼              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              vLLM Inference Server                       │   │
│  │         (generates rollouts for scoring)                 │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Mode 1: Legacy (Checkpoint + Restart)

This is the simplest mode. The trainer periodically saves checkpoints and restarts vLLM.

### Step-by-Step Guide

**Terminal 1: Start the Atropos API**
```bash
cd atropos
run-api
```

**Terminal 2: Start the GSM8k Environment**
```bash
cd atropos
python environments/gsm8k_server.py serve --slurm False
```

**Terminal 3: Start the GRPO Trainer**
```bash
cd atropos
python example_trainer/grpo.py \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --weight-bridge-mode none \
  --training-steps 100 \
  --vllm-restart-interval 10 \
  --batch-size 2 \
  --gradient-accumulation-steps 16 \
  --lr 1e-5 \
  --use-wandb \
  --wandb-project gsm8k-grpo
```

### What Happens

1. Trainer loads `Qwen/Qwen2.5-3B-Instruct` into GPU memory
2. Trainer launches vLLM server on port 9001
3. GSM8k env sends problems → vLLM generates solutions → scores sent to API
4. Trainer fetches scored batches from API, computes GRPO loss, updates weights
5. Every 10 steps: save checkpoint → kill vLLM → restart vLLM with new weights
6. Repeat until done

### Pros & Cons

+ Simple, works out of the box  
+ Easy to debug  
- 30-60 second sync latency per restart  
- 2x GPU memory (trainer + vLLM both load model)

---

## Mode 2: Shared vLLM Bridge (In-Place Updates)

This mode shares GPU tensors between trainer and vLLM. Updates happen instantly.

### Step-by-Step Guide

**Terminal 1: Start the Atropos API**
```bash
cd atropos
run-api
```

**Terminal 2: Set up environment variables and start vLLM with bridge support**
```bash
cd atropos
export LOGDIR=/tmp/atropos_bridge
export NUM_INFERENCE_NODES=0  # Single-node local mode
export MASTER_ADDR=localhost
export MASTER_PORT=26756

mkdir -p $LOGDIR

# Start the custom vLLM server with bridge endpoints
python example_trainer/vllm_api_server.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --port 9001 \
  --gpu-memory-utilization 0.45
```

**Terminal 3: Start the GSM8k Environment**
```bash
cd atropos
python environments/gsm8k_server.py serve --slurm False
```

**Terminal 4: Start the GRPO Trainer in shared mode**
```bash
cd atropos
export LOGDIR=/tmp/atropos_bridge
export NUM_INFERENCE_NODES=0
export MASTER_ADDR=localhost
export MASTER_PORT=26756

python example_trainer/grpo.py \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --weight-bridge-mode shared_vllm \
  --trainer-rank 0 \
  --world-size 1 \
  --num-inference-nodes 0 \
  --training-steps 100 \
  --batch-size 2 \
  --gradient-accumulation-steps 16 \
  --lr 1e-5 \
  --use-wandb \
  --wandb-project gsm8k-grpo-shared
```

### What Happens

1. vLLM server starts, writes parameter mapping to `$LOGDIR/vllm_bridge_config.json`
2. Trainer reads mapping, joins NCCL process group with vLLM
3. Trainer's model parameters point to vLLM's GPU tensors (shared memory)
4. Training loop:
   - Forward pass uses shared weights
   - `optimizer.step()` modifies shared tensors in-place
   - `bridge.notify_update()` signals vLLM (optional coordination)
   - vLLM immediately uses new weights for next inference
5. No restarts needed!

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `LOGDIR` | Directory for bridge coordination files | `/tmp/atropos_bridge` |
| `NUM_INFERENCE_NODES` | Number of vLLM nodes (0 = local) | `0` |
| `MASTER_ADDR` | Rendezvous address | `localhost` |
| `MASTER_PORT` | Rendezvous port | `26756` |

### Pros & Cons

+ ~0ms sync latency (instant updates)  
+ 1x GPU memory (shared tensors)  
+ Maximum training throughput  
- More complex setup  
- Requires compatible vLLM version

---

## Mode 3: LoRA Adapters (Hot-Swap)

This mode trains only LoRA adapter weights. Much smaller checkpoints, faster iteration.

### Step-by-Step Guide

**Terminal 1: Start the Atropos API**
```bash
cd atropos
run-api
```

**Terminal 2: Start the GSM8k Environment**
```bash
cd atropos
python environments/gsm8k_server.py serve --slurm False
```

**Terminal 3: Start the GRPO Trainer in LoRA mode**
```bash
cd atropos
python example_trainer/grpo.py \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --weight-bridge-mode lora_only \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --lora-target-modules q_proj v_proj \
  --training-steps 100 \
  --vllm-restart-interval 20 \
  --batch-size 2 \
  --gradient-accumulation-steps 16 \
  --lr 1e-4 \
  --use-wandb \
  --wandb-project gsm8k-grpo-lora
```

### What Happens

1. Trainer loads base model, wraps with LoRA adapters (PEFT)
2. Only adapter parameters are trainable (~0.1% of total)
3. Training loop updates adapter weights only
4. Every N steps: save adapter checkpoint (small, ~10-50MB)
5. vLLM can hot-swap adapters via `/lora/load` endpoint

### LoRA Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `--lora-r` | 16 | Rank of low-rank matrices |
| `--lora-alpha` | 32 | Scaling factor (typically 2x rank) |
| `--lora-dropout` | 0.05 | Dropout for regularization |
| `--lora-target-modules` | `q_proj v_proj` | Which layers to adapt |

### Common Target Module Combinations

```bash
# Minimal (fastest training)
--lora-target-modules q_proj v_proj

# Attention only
--lora-target-modules q_proj k_proj v_proj o_proj

# Full (most expressive)
--lora-target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
```

### Pros & Cons

+ Much faster training (fewer parameters)  
+ Tiny checkpoints (~10-50MB vs ~6GB)  
+ Can hot-swap adapters without full restart  
+ Lower GPU memory (base model frozen)  
- Less expressive than full fine-tuning  
- May need higher learning rate

---

## Configuration Reference

### All CLI Options

```bash
python example_trainer/grpo.py --help
```

### Core Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model-name` | (required) | HuggingFace model ID |
| `--lr` | `1e-5` | Learning rate |
| `--training-steps` | `10` | Total optimization steps |
| `--batch-size` | `2` | Micro-batch size |
| `--gradient-accumulation-steps` | `32` | Gradient accumulation |
| `--seq-len` | `2048` | Max sequence length |
| `--save-path` | `trained_model_checkpoints` | Checkpoint directory |

### vLLM Options

| Option | Default | Description |
|--------|---------|-------------|
| `--vllm-port` | `9001` | vLLM server port |
| `--vllm-restart-interval` | `3` | Steps between syncs |

### Weight Bridge Options

| Option | Default | Description |
|--------|---------|-------------|
| `--weight-bridge-mode` | `none` | `none`, `shared_vllm`, or `lora_only` |
| `--trainer-rank` | `0` | Distributed rank |
| `--world-size` | `1` | Total processes |
| `--init-method` | `env://` | PyTorch distributed init |
| `--num-inference-nodes` | `0` | Number of vLLM nodes |

### Logging Options

| Option | Default | Description |
|--------|---------|-------------|
| `--use-wandb` | `False` | Enable W&B logging |
| `--wandb-project` | `None` | W&B project name |
| `--wandb-group` | `None` | W&B group name |

---

## Troubleshooting

### "CUDA out of memory"

Try reducing:
```bash
--batch-size 1 \
--gradient-accumulation-steps 64 \
--seq-len 1024
```

Or use LoRA mode which uses less memory.

### "Connection refused" to Atropos API

Make sure the API is running:
```bash
run-api  # In a separate terminal
```

### vLLM fails to start

Check if port 9001 is in use:
```bash
lsof -i :9001
```

Kill existing processes or use a different port:
```bash
--vllm-port 9002
```

### Bridge mode: "Parameter mapping file not found"

Ensure `$LOGDIR` is set and vLLM server is running:
```bash
export LOGDIR=/tmp/atropos_bridge
ls $LOGDIR/vllm_bridge_config.json
```

### LoRA mode: "PEFT library not available"

Install PEFT:
```bash
pip install peft
```

---

## Files in This Directory

| File | Description |
|------|-------------|
| `grpo.py` | Main trainer script with all modes |
| `vllm_api_server.py` | Custom vLLM server with bridge endpoints |
| `vllm_weight_bridge.py` | Shared memory bridge implementation |
| `requirements.txt` | Python dependencies |
| `README.md` | This documentation |

---

## Example Runs

### Quick Test (Legacy Mode)
```bash
# Minimal test to verify setup works
python example_trainer/grpo.py \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --training-steps 5 \
  --batch-size 1 \
  --gradient-accumulation-steps 4
```

### Full GSM8k Training (LoRA Mode)
```bash
# Recommended for single-GPU training
python example_trainer/grpo.py \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --weight-bridge-mode lora_only \
  --lora-r 32 \
  --lora-alpha 64 \
  --training-steps 500 \
  --batch-size 2 \
  --gradient-accumulation-steps 32 \
  --lr 5e-5 \
  --use-wandb \
  --wandb-project gsm8k-lora
```

### Production (Shared vLLM Mode)
```bash
# Maximum throughput setup
export LOGDIR=/tmp/atropos_bridge
export NUM_INFERENCE_NODES=0

python example_trainer/grpo.py \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --weight-bridge-mode shared_vllm \
  --training-steps 1000 \
  --batch-size 4 \
  --gradient-accumulation-steps 16 \
  --lr 1e-5 \
  --use-wandb \
  --wandb-project gsm8k-shared
```
