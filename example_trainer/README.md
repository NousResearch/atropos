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

### IMPORTANT: Startup Order (Same for ALL Modes!)

All three training modes use the **same startup order**:

```
1. Atropos API          (python -m atroposlib.cli.run_api)
       ↓
2. vLLM Server          (python example_trainer/vllm_api_server.py)
       ↓ wait 60-90s for model load
3. GRPO Trainer         (python example_trainer/grpo.py)
       ↓
4. GSM8k Environment    (python environments/gsm8k_server.py serve ...)
```

**Why this order?**
- The API must be running before the trainer or environment tries to connect
- vLLM must be loaded before GSM8k tries to generate rollouts
- The trainer must be running before GSM8k sends scored batches
- GSM8k is started last because it immediately begins generating work

### GSM8k CLI Arguments (Required for All Modes)

When starting the GSM8k environment, always include these arguments:

```bash
python environments/gsm8k_server.py serve \
  --slurm False \
  --openai.model_name Qwen/Qwen2.5-3B-Instruct \
  --openai.base_url http://localhost:9001/v1 \
  --openai.server_type vllm \
  --env.tokenizer_name Qwen/Qwen2.5-3B-Instruct
```

| Argument | Required | Description |
|----------|----------|-------------|
| `--slurm False` | Yes | Disable SLURM mode for local runs |
| `--openai.model_name` | Yes | Model name (must match vLLM) |
| `--openai.base_url` | Yes | vLLM server URL with `/v1` suffix |
| `--openai.server_type vllm` | **Yes** | Must be `vllm` for `/generate` endpoint |
| `--env.tokenizer_name` | Yes | Tokenizer for environment |

**Note:** `--openai.server_type vllm` is required because only the `VLLMServer` class supports `tokens_and_logprobs_completion` which GSM8k needs.

---

## Mode 1: Legacy (Checkpoint + Restart)

This mode saves checkpoints periodically and can restart vLLM with updated weights.

### Startup Order (Same for All Modes!)

```
┌────────────────────────────────────────────────────────────────┐
│  1. Atropos API       →  Coordinates environments + trainer   │
│  2. vLLM Server       →  Serves inference requests            │
│  3. GRPO Trainer      →  Trains model, fetches batches        │
│  4. GSM8k Environment →  Generates problems, scores rollouts  │
└────────────────────────────────────────────────────────────────┘
```

### Step-by-Step Guide

**Step 1: Start the Atropos API**
```bash
cd atropos
python -m atroposlib.cli.run_api > api.log 2>&1 &
sleep 5
```

**Step 2: Start the vLLM Server**
```bash
cd atropos
python example_trainer/vllm_api_server.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --port 9001 \
  --gpu-memory-utilization 0.30 \
  > vllm.log 2>&1 &
sleep 90  # Wait for model to load

# Verify vLLM is ready
curl -s http://localhost:9001/health && echo "vLLM ready!"
```

**Step 3: Start the GRPO Trainer**
```bash
cd atropos
python example_trainer/grpo.py \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --weight-bridge-mode none \
  --training-steps 100 \
  --vllm-restart-interval 10 \
  --vllm-port 9001 \
  --vllm-gpu-memory-utilization 0.30 \
  --batch-size 2 \
  --gradient-accumulation-steps 16 \
  --lr 1e-5 \
  --save-path checkpoints_legacy \
  --use-wandb \
  --wandb-project gsm8k-grpo \
  > trainer.log 2>&1 &
sleep 10
```

**Step 4: Start the GSM8k Environment**
```bash
cd atropos
python environments/gsm8k_server.py serve \
  --slurm False \
  --openai.model_name Qwen/Qwen2.5-3B-Instruct \
  --openai.base_url http://localhost:9001/v1 \
  --openai.server_type vllm \
  --env.tokenizer_name Qwen/Qwen2.5-3B-Instruct \
  > gsm8k.log 2>&1 &
```

**Monitor Training:**
```bash
tail -f trainer.log
```

### Quick Copy-Paste (All-in-One)

```bash
cd atropos && \
pkill -f "grpo.py|vllm|gsm8k|run_api" 2>/dev/null; sleep 3 && \
python -m atroposlib.cli.run_api > api.log 2>&1 & sleep 5 && \
python example_trainer/vllm_api_server.py --model Qwen/Qwen2.5-3B-Instruct --port 9001 --gpu-memory-utilization 0.30 > vllm.log 2>&1 & sleep 90 && \
python example_trainer/grpo.py --model-name Qwen/Qwen2.5-3B-Instruct --weight-bridge-mode none --training-steps 100 --vllm-port 9001 --vllm-gpu-memory-utilization 0.30 --batch-size 2 --gradient-accumulation-steps 16 --lr 1e-5 --save-path checkpoints_legacy --use-wandb --wandb-project gsm8k-grpo > trainer.log 2>&1 & sleep 10 && \
python environments/gsm8k_server.py serve --slurm False --openai.model_name Qwen/Qwen2.5-3B-Instruct --openai.base_url http://localhost:9001/v1 --openai.server_type vllm --env.tokenizer_name Qwen/Qwen2.5-3B-Instruct > gsm8k.log 2>&1 & \
tail -f trainer.log
```

### What Happens

1. vLLM server starts and loads `Qwen/Qwen2.5-3B-Instruct`
2. Trainer loads its own copy of the model for training
3. GSM8k env sends problems → vLLM generates solutions → scores sent to API
4. Trainer fetches scored batches from API, computes GRPO loss, updates weights
5. Every N steps: save checkpoint (weights stay in sync via external vLLM)
6. Repeat until done

### Pros & Cons

+ Simple conceptually  
+ Easy to debug  
+ Uses custom vLLM server with full endpoint support  
- 2x GPU memory (trainer + vLLM both load model)  
- Requires external vLLM to be running

---

## Mode 2: Shared vLLM Bridge (In-Place Updates)

This mode uses an HTTP-based notification system. The trainer notifies vLLM after weight updates.

### Step-by-Step Guide

**Step 1: Start the Atropos API**
```bash
cd atropos
python -m atroposlib.cli.run_api > api.log 2>&1 &
sleep 5
```

**Step 2: Start the vLLM Server with Bridge Support**
```bash
cd atropos
export LOGDIR=/tmp/atropos_bridge
export NUM_INFERENCE_NODES=0
mkdir -p $LOGDIR

python example_trainer/vllm_api_server.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --port 9001 \
  --gpu-memory-utilization 0.30 \
  > vllm.log 2>&1 &
sleep 90

curl -s http://localhost:9001/health && echo "vLLM ready!"
```

**Step 3: Start the GRPO Trainer in Shared Mode**
```bash
cd atropos
export LOGDIR=/tmp/atropos_bridge
export NUM_INFERENCE_NODES=0

python example_trainer/grpo.py \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --weight-bridge-mode shared_vllm \
  --num-inference-nodes 0 \
  --training-steps 100 \
  --vllm-port 9001 \
  --batch-size 2 \
  --gradient-accumulation-steps 16 \
  --lr 1e-5 \
  --save-path checkpoints_shared \
  --use-wandb \
  --wandb-project gsm8k-grpo-shared \
  > trainer.log 2>&1 &
sleep 10
```

**Step 4: Start the GSM8k Environment**
```bash
cd atropos
python environments/gsm8k_server.py serve \
  --slurm False \
  --openai.model_name Qwen/Qwen2.5-3B-Instruct \
  --openai.base_url http://localhost:9001/v1 \
  --openai.server_type vllm \
  --env.tokenizer_name Qwen/Qwen2.5-3B-Instruct \
  > gsm8k.log 2>&1 &
```

**Monitor Training:**
```bash
tail -f trainer.log
```

### Quick Copy-Paste (All-in-One)

```bash
cd atropos && \
pkill -f "grpo.py|vllm|gsm8k|run_api" 2>/dev/null; sleep 3 && \
export LOGDIR=/tmp/atropos_bridge && export NUM_INFERENCE_NODES=0 && mkdir -p $LOGDIR && \
python -m atroposlib.cli.run_api > api.log 2>&1 & sleep 5 && \
python example_trainer/vllm_api_server.py --model Qwen/Qwen2.5-3B-Instruct --port 9001 --gpu-memory-utilization 0.30 > vllm.log 2>&1 & sleep 90 && \
python example_trainer/grpo.py --model-name Qwen/Qwen2.5-3B-Instruct --weight-bridge-mode shared_vllm --num-inference-nodes 0 --training-steps 100 --vllm-port 9001 --batch-size 2 --gradient-accumulation-steps 16 --lr 1e-5 --save-path checkpoints_shared --use-wandb --wandb-project gsm8k-grpo-shared > trainer.log 2>&1 & sleep 10 && \
python environments/gsm8k_server.py serve --slurm False --openai.model_name Qwen/Qwen2.5-3B-Instruct --openai.base_url http://localhost:9001/v1 --openai.server_type vllm --env.tokenizer_name Qwen/Qwen2.5-3B-Instruct > gsm8k.log 2>&1 & \
tail -f trainer.log
```

### What Happens (Local Mode - num_inference_nodes=0)

1. vLLM server starts on port 9001
2. Trainer initializes bridge in LOCAL MODE (HTTP-based, no NCCL)
3. Trainer loads its own model copy and trains normally
4. After each `optimizer.step()`:
   - `bridge.notify_update()` sends HTTP POST to vLLM
   - Periodic checkpoint saves sync weights to disk
5. Much simpler than distributed mode!

### What Happens (Distributed Mode - num_inference_nodes>0)

1. vLLM server starts, writes parameter mapping to `$LOGDIR/vllm_bridge_config.json`
2. Trainer reads mapping, joins NCCL process group with vLLM
3. Trainer's model parameters point to vLLM's GPU tensors (shared memory)
4. Training loop:
   - Forward pass uses shared weights
   - `optimizer.step()` modifies shared tensors in-place
   - `bridge.notify_update()` broadcasts via Gloo
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

**Step 1: Start the Atropos API**
```bash
cd atropos
python -m atroposlib.cli.run_api > api.log 2>&1 &
sleep 5
```

**Step 2: Start the vLLM Server (Required for LoRA Hot-Swap)**
```bash
cd atropos
python example_trainer/vllm_api_server.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --port 9001 \
  --gpu-memory-utilization 0.30 \
  > vllm.log 2>&1 &
sleep 90

curl -s http://localhost:9001/health && echo "vLLM ready!"
```

**Step 3: Start the GRPO Trainer in LoRA Mode**
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
  --vllm-port 9001 \
  --batch-size 2 \
  --gradient-accumulation-steps 16 \
  --lr 1e-4 \
  --save-path checkpoints_lora \
  --use-wandb \
  --wandb-project gsm8k-grpo-lora \
  > trainer.log 2>&1 &
sleep 10
```

**Step 4: Start the GSM8k Environment**
```bash
cd atropos
python environments/gsm8k_server.py serve \
  --slurm False \
  --openai.model_name Qwen/Qwen2.5-3B-Instruct \
  --openai.base_url http://localhost:9001/v1 \
  --openai.server_type vllm \
  --env.tokenizer_name Qwen/Qwen2.5-3B-Instruct \
  > gsm8k.log 2>&1 &
```

**Monitor Training:**
```bash
tail -f trainer.log
```

### Quick Copy-Paste (All-in-One)

```bash
cd atropos && \
pkill -f "grpo.py|vllm|gsm8k|run_api" 2>/dev/null; sleep 3 && \
python -m atroposlib.cli.run_api > api.log 2>&1 & sleep 5 && \
python example_trainer/vllm_api_server.py --model Qwen/Qwen2.5-3B-Instruct --port 9001 --gpu-memory-utilization 0.30 > vllm.log 2>&1 & sleep 90 && \
python example_trainer/grpo.py --model-name Qwen/Qwen2.5-3B-Instruct --weight-bridge-mode lora_only --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 --training-steps 100 --vllm-port 9001 --batch-size 2 --gradient-accumulation-steps 16 --lr 1e-4 --save-path checkpoints_lora --use-wandb --wandb-project gsm8k-grpo-lora > trainer.log 2>&1 & sleep 10 && \
python environments/gsm8k_server.py serve --slurm False --openai.model_name Qwen/Qwen2.5-3B-Instruct --openai.base_url http://localhost:9001/v1 --openai.server_type vllm --env.tokenizer_name Qwen/Qwen2.5-3B-Instruct > gsm8k.log 2>&1 & \
tail -f trainer.log
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

## Shutdown / Cleanup

### Stop All Processes

```bash
# Graceful shutdown
pkill -f "gsm8k_server"
sleep 2
pkill -f "grpo.py"
sleep 2
pkill -f "vllm_api_server"
sleep 2
pkill -f "run_api"

echo "All processes stopped"
```

### Check Running Processes

```bash
ps aux | grep -E "(grpo|vllm|gsm8k|run_api)" | grep -v grep
```

### Check GPU Usage

```bash
nvidia-smi
```

---

## Troubleshooting

### "CUDA out of memory"

Try reducing:
```bash
--batch-size 1 \
--gradient-accumulation-steps 64 \
--seq-len 1024 \
--vllm-gpu-memory-utilization 0.25
```

Or use LoRA mode which uses less memory.

### "Connection refused" to Atropos API

Make sure the API is running:
```bash
python -m atroposlib.cli.run_api
```

### vLLM fails to start

Check if port 9001 is in use:
```bash
lsof -i :9001
```

Kill existing processes or use a different port:
```bash
pkill -f "vllm_api_server"
# or use different port:
--vllm-port 9002
```

### "NotImplementedError" or "404 Not Found" on `/generate`

This means you're using the wrong server type. Make sure:

1. You started `vllm_api_server.py` (NOT standard `vllm serve`)
2. GSM8k uses `--openai.server_type vllm` (NOT `openai`)

```bash
# CORRECT
python example_trainer/vllm_api_server.py --model ... --port 9001
python environments/gsm8k_server.py serve --openai.server_type vllm ...

# WRONG - standard vLLM doesn't have /generate endpoint
python -m vllm.entrypoints.openai.api_server --model ... --port 9001
```

### "Free memory on device is less than desired GPU memory utilization"

Lower the vLLM memory utilization:

```bash
python example_trainer/vllm_api_server.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --port 9001 \
  --gpu-memory-utilization 0.25  # Lower this
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

### No trajectories collected / Workers timing out

Check that all services are running in the correct order:
```bash
# Check processes
ps aux | grep -E "(run_api|vllm|grpo|gsm8k)" | grep -v grep

# Check vLLM health
curl http://localhost:9001/health

# Check API health  
curl http://localhost:8000/health
```

If vLLM isn't ready, wait longer before starting GSM8k.

---

## Checkpoint Locations

### Where Are Trained Models Saved?

| Mode | Location | Contents |
|------|----------|----------|
| **Legacy** | `trained_model_checkpoints/step_N/` | Full model + tokenizer |
| **Legacy** | `trained_model_checkpoints/final_model/` | Final checkpoint |
| **Shared vLLM** | `trained_model_checkpoints/step_N/` | Full model + tokenizer |
| **LoRA** | `trained_model_checkpoints/adapter_step_N/` | LoRA adapters only (~10-50MB) |
| **LoRA** | `trained_model_checkpoints/final_adapter/` | Final adapter |

### Customizing Save Path

```bash
python example_trainer/grpo.py \
  --save-path /path/to/my/checkpoints \
  ...
```

### Loading Checkpoints for Inference

```python
# Full model (Legacy/Shared modes)
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("trained_model_checkpoints/final_model")
tokenizer = AutoTokenizer.from_pretrained("trained_model_checkpoints/final_model")

# LoRA adapter
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = PeftModel.from_pretrained(base_model, "trained_model_checkpoints/final_adapter")
```

---

## vLLM Server Requirements

When using `--openai.server_type vllm` or the shared_vllm bridge, your vLLM server must expose these endpoints:

### Required Endpoints

| Endpoint | Method | Purpose | Used By |
|----------|--------|---------|---------|
| `/health` | GET | Health check | All modes |
| `/generate` | POST | Native generation with token IDs + logprobs | VLLMServer class |

### Required `/generate` Request Format

The vLLM server must handle **both** prompt formats:

```json
// String prompt (simple)
{
  "prompt": "Hello, world!",
  "max_tokens": 100,
  "temperature": 1.0,
  "logprobs": 1
}

// Token ID prompt (used by atroposlib)
{
  "prompt": {"prompt_token_ids": [1, 2, 3, 4, 5]},
  "max_tokens": 100,
  "temperature": 1.0,
  "logprobs": 1
}
```

### Required `/generate` Response Format

```json
{
  "text": ["generated text here"],
  "prompt": "original prompt",
  "finish_reasons": ["stop"],
  "logprobs": [
    [
      [{"12345": -0.5}],
      [{"67890": -1.2}]
    ]
  ],
  "prompt_token_ids": [1, 2, 3, 4, 5],
  "token_ids": [[12345, 67890, ...]]
}
```

The `logprobs` field format: `List[List[List[Dict[token_id, logprob]]]]`
- Outer list: per completion (n samples)
- Middle list: per token in completion
- Inner list: contains single dict `{token_id: logprob}`

### Optional Bridge Endpoints (for shared_vllm mode)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/bridge/info` | GET | Get bridge status |
| `/bridge/notify_update` | POST | Receive weight update notifications |
| `/bridge/state_dict_info` | GET | Get model parameter mappings |

### Optional LoRA Endpoints (for lora_only mode)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/lora/status` | GET | Get active LoRA adapter |
| `/lora/load` | POST | Load new LoRA adapter |
| `/lora/unload` | POST | Unload current adapter |

### Using Standard vLLM vs Custom Server

| Server | Supports `/generate` with logprobs | Supports bridge | Supports LoRA hot-swap |
|--------|-----------------------------------|-----------------|------------------------|
| `vllm serve ...` | ❌ No | ❌ No | ❌ No |
| `vllm_api_server.py` | ✅ Yes | ✅ Yes | ✅ Yes |

**Use `example_trainer/vllm_api_server.py` for full feature support.**

---

## Benchmarking Speed & Memory

### Memory Usage Comparison

```bash
# Run this during training to monitor GPU memory
watch -n 1 nvidia-smi
```

**Expected Memory Usage (Qwen2.5-3B-Instruct):**

| Mode | Trainer GPU | vLLM GPU | Total |
|------|------------|----------|-------|
| **Legacy** | ~8GB | ~8GB | ~16GB (2x model) |
| **Shared vLLM** | ~8GB (shared) | ~8GB (shared) | ~8GB (1x model) |
| **LoRA** | ~10GB (frozen base) | ~8GB | ~18GB |

### Speed Benchmarking

Add these measurements to your training script or use the built-in wandb logging:

```python
import time
import torch

# Track step times
step_times = []
sync_times = []

for step in range(training_steps):
    # Measure training step time
    step_start = time.time()
    # ... training code ...
    step_time = time.time() - step_start
    step_times.append(step_time)
    
    # Measure sync time (Legacy mode only)
    if step % vllm_restart_interval == 0:
        sync_start = time.time()
        # ... checkpoint + restart vLLM ...
        sync_time = time.time() - sync_start
        sync_times.append(sync_time)

# Print summary
print(f"Avg step time: {sum(step_times)/len(step_times):.2f}s")
print(f"Avg sync time: {sum(sync_times)/len(sync_times):.2f}s" if sync_times else "No syncs")
```

### Benchmark Script

Create a benchmark comparing modes:

```bash
#!/bin/bash
# benchmark_modes.sh

MODEL="Qwen/Qwen2.5-3B-Instruct"
STEPS=50
BATCH=2
ACCUM=16

echo "=== Benchmarking Legacy Mode ==="
time python example_trainer/grpo.py \
  --model-name $MODEL \
  --weight-bridge-mode none \
  --training-steps $STEPS \
  --batch-size $BATCH \
  --gradient-accumulation-steps $ACCUM \
  --vllm-restart-interval 10 \
  2>&1 | tee benchmark_legacy.log

echo "=== Benchmarking Shared vLLM Mode ==="
export LOGDIR=/tmp/bench_shared
export NUM_INFERENCE_NODES=0
mkdir -p $LOGDIR

# Start vLLM first
python example_trainer/vllm_api_server.py \
  --model $MODEL --port 9001 --gpu-memory-utilization 0.45 &
VLLM_PID=$!
sleep 60  # Wait for vLLM to load

time python example_trainer/grpo.py \
  --model-name $MODEL \
  --weight-bridge-mode shared_vllm \
  --training-steps $STEPS \
  --batch-size $BATCH \
  --gradient-accumulation-steps $ACCUM \
  --num-inference-nodes 0 \
  2>&1 | tee benchmark_shared.log

kill $VLLM_PID

echo "=== Benchmarking LoRA Mode ==="
time python example_trainer/grpo.py \
  --model-name $MODEL \
  --weight-bridge-mode lora_only \
  --training-steps $STEPS \
  --batch-size $BATCH \
  --gradient-accumulation-steps $ACCUM \
  --lora-r 16 \
  --vllm-restart-interval 25 \
  2>&1 | tee benchmark_lora.log

echo "=== Summary ==="
echo "Check benchmark_*.log for detailed timing"
```

### Expected Benchmark Results

| Metric | Legacy | Shared vLLM | LoRA |
|--------|--------|-------------|------|
| **Step time** | ~2-5s | ~2-5s | ~1-3s |
| **Sync overhead** | ~30-60s every N steps | ~0ms | ~5-10s every N steps |
| **Total time (50 steps, sync every 10)** | ~15-20 min | ~3-5 min | ~5-8 min |
| **Peak GPU memory** | ~16GB | ~8GB | ~10GB |
| **Checkpoint size** | ~6GB | ~6GB | ~50MB |

### WandB Metrics to Watch

If using `--use-wandb`, these metrics are logged:

| Metric | Description |
|--------|-------------|
| `train/loss` | GRPO loss |
| `train/grad_norm` | Gradient norm |
| `train/pos_logp` | Log prob of positive examples |
| `train/neg_logp` | Log prob of negative examples |
| `train/step_time` | Time per training step |
| `train/sync_time` | Time for weight sync (legacy/lora) |

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
