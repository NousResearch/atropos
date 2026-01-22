# GRPO Trainer

A modular training framework for fine-tuning language models with **Group Relative Policy Optimization (GRPO)**, designed to work with the Atropos environment system.

## 📁 Module Structure

```
example_trainer/
├── grpo.py              # CLI entry point (dispatches to trainers)
├── config.py            # TrainingConfig dataclass
├── api.py               # Atropos API communication
├── data.py              # Data fetching & preprocessing
├── model.py             # Model loading & CUDA IPC shared memory
├── training.py          # Loss computation & training step
├── checkpointing.py     # Save models & LoRA adapters
├── vllm_manager.py      # vLLM process management
├── trainers.py          # Training mode implementations
├── cli.py               # CLI argument parsing
├── vllm_api_server.py   # Custom vLLM server with IPC support
├── vllm_patching/       # B200/Blackwell GPU patches
│   └── patched_gpu_runner.py
└── scripts/             # Helper scripts
    ├── run_comparison.sh
    ├── run_concurrent_tests.sh
    ├── test_lora_mode.sh
    └── test_single_copy_mode.sh
```

---

## 🔄 Full System Architecture

The Atropos training system consists of 4 components that must run together:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       ATROPOS TRAINING SYSTEM                               │
└─────────────────────────────────────────────────────────────────────────────┘

 ┌─────────────┐      ┌──────────────────┐      ┌─────────────────┐
 │    vLLM     │◄────►│  Environment     │─────►│   run-api       │
 │   Server    │      │  (gsm8k_server)  │      │  (Trajectory    │
 │  (Inference)│      │  (Process Env)   │      │   Handler API)  │
 └─────────────┘      └──────────────────┘      └────────┬────────┘
       ▲                                                  │
       │                                                  │
       │              ┌───────────────────────────────────┘
       │              │
       │              ▼
       │        ┌─────────────┐
       └────────│   GRPO      │
                │   Trainer   │
                │  (grpo.py)  │
                └─────────────┘

Data Flow:
1. run-api      : Central API that receives trajectories and serves batches
2. Environment  : Generates prompts, calls vLLM, scores responses → sends to run-api
3. Trainer      : Fetches batches from run-api → trains model → updates weights
4. vLLM         : Serves inference for environment (and gets weight updates)
```

### Components Explained

| Component | Command | Port | Purpose |
|-----------|---------|------|---------|
| **run-api** | `run-api` | 8000 | Central trajectory handler API |
| **Environment** | `gsm8k_server.py serve` | (internal) | Generates rollouts, scores them |
| **vLLM** | `vllm_api_server.py` | 9001 | Model inference |
| **Trainer** | `grpo.py` | (client) | Fetches batches, trains model |

---

## 🎯 Three Training Modes

| Mode | Description | vLLM Setup | Best For |
|------|-------------|------------|----------|
| **Legacy** (`none`) | Trainer manages vLLM, restarts with new checkpoints | Auto-managed | Simple setup, different GPUs |
| **Shared vLLM** (`shared_vllm`) | Single-copy mode via CUDA IPC - no model duplication! | External, `VLLM_ENABLE_SHARED_WEIGHTS=1` | Same GPU, max efficiency |
| **LoRA** (`lora_only`) | Train adapters only, hot-swap in vLLM | External, `--enable-lora` | Fast training, small checkpoints |

---

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install torch transformers peft vllm wandb requests tenacity pydantic

# Set environment variables
export LOGDIR=/tmp/atropos_test
export MODEL=Qwen/Qwen2.5-3B-Instruct
mkdir -p $LOGDIR
```

---

## 📖 Detailed Usage for Each Mode

### Mode 1: Legacy (Checkpoint + Restart)

The simplest mode. Trainer manages vLLM internally.

```bash
# Terminal 1: Start the central API server (handles trajectories)
run-api --port 8000

# Terminal 2: Start the environment server (generates rollouts)
python -u environments/gsm8k_server.py serve \
    --env.tokenizer_name $MODEL \
    --env.use_wandb=False \
    --openai.model_name $MODEL \
    --openai.base_url http://localhost:9001/v1 \
    --openai.server_type vllm

# Terminal 3: Run training (trainer will launch its own vLLM)
CUDA_VISIBLE_DEVICES=0 python -m example_trainer.grpo \
    --model-name $MODEL \
    --weight-bridge-mode none \
    --vllm-port 9001 \
    --atropos-url http://localhost:8000 \
    --training-steps 20 \
    --batch-size 2 \
    --save-path $LOGDIR/checkpoints_legacy \
    --benchmark
```

### Mode 2: Shared vLLM (Single-Copy CUDA IPC)

Zero model duplication - trainer and vLLM share the exact same GPU memory!

```bash
# Terminal 1: Start the central API server
run-api --port 8000

# Terminal 2: Start vLLM with shared weights enabled
VLLM_ENABLE_SHARED_WEIGHTS=1 LOGDIR=$LOGDIR \
CUDA_VISIBLE_DEVICES=0 python example_trainer/vllm_api_server.py \
    --model $MODEL \
    --port 9001 \
    --gpu-memory-utilization 0.45

# Terminal 3: Start the environment server
python -u environments/gsm8k_server.py serve \
    --env.tokenizer_name $MODEL \
    --env.use_wandb=False \
    --openai.model_name $MODEL \
    --openai.base_url http://localhost:9001/v1 \
    --openai.server_type vllm

# Terminal 4: Run training (attaches to vLLM's tensors)
CUDA_VISIBLE_DEVICES=0 python -m example_trainer.grpo \
    --model-name $MODEL \
    --weight-bridge-mode shared_vllm \
    --vllm-port 9001 \
    --vllm-config-path $LOGDIR/vllm_bridge_config.json \
    --atropos-url http://localhost:8000 \
    --training-steps 20 \
    --batch-size 2 \
    --save-path $LOGDIR/checkpoints_shared \
    --benchmark
```

### Mode 3: LoRA (Adapter Training)

Fast training with hot-swappable adapters.

```bash
# Terminal 1: Start the central API server
run-api --port 8000

# Terminal 2: Start vLLM with LoRA support
CUDA_VISIBLE_DEVICES=0 python example_trainer/vllm_api_server.py \
    --model $MODEL \
    --port 9001 \
    --gpu-memory-utilization 0.45 \
    --enable-lora \
    --max-lora-rank 32 \
    --enforce-eager

# Terminal 3: Start the environment server
python -u environments/gsm8k_server.py serve \
    --env.tokenizer_name $MODEL \
    --env.use_wandb=False \
    --openai.model_name $MODEL \
    --openai.base_url http://localhost:9001/v1 \
    --openai.server_type vllm

# Terminal 4: Run LoRA training
CUDA_VISIBLE_DEVICES=1 python -m example_trainer.grpo \
    --model-name $MODEL \
    --weight-bridge-mode lora_only \
    --vllm-port 9001 \
    --atropos-url http://localhost:8000 \
    --lora-r 16 \
    --lora-alpha 32 \
    --training-steps 20 \
    --batch-size 2 \
    --save-path $LOGDIR/checkpoints_lora \
    --benchmark
```

---

## 🔬 Run All 3 Modes in Parallel (8-GPU Comparison)

Use this setup to compare training efficiency across all three modes on a single 8-GPU node.

### GPU & Port Allocation

| Mode | GPUs | vLLM Port | API Port | Env Port |
|------|------|-----------|----------|----------|
| Legacy | 0-1 | 9001 | 8001 | (internal) |
| Shared vLLM | 2-3 | 9002 | 8002 | (internal) |
| LoRA | 4-5 | 9003 | 8003 | (internal) |

### Quick Start: Use the Comparison Script

```bash
cd /path/to/atropos

# Run comparison with default 50 steps
./example_trainer/scripts/run_comparison.sh

# Or specify steps
./example_trainer/scripts/run_comparison.sh 100
```

### Manual Parallel Execution

If you prefer to run each mode manually in separate terminal sessions:

```bash
# Setup
export MODEL="Qwen/Qwen2.5-3B-Instruct"
export LOGDIR=/tmp/atropos_test
mkdir -p $LOGDIR

# =============================================
# LEGACY MODE (Terminals 1-3)
# =============================================

# Terminal 1: API server for legacy
run-api --port 8001

# Terminal 2: Environment server
python -u environments/gsm8k_server.py serve \
    --env.tokenizer_name $MODEL \
    --env.use_wandb=False \
    --openai.model_name $MODEL \
    --openai.base_url http://localhost:9001/v1 \
    --openai.server_type vllm \
    --server.port 8001

# Terminal 3: Trainer (manages its own vLLM)
CUDA_VISIBLE_DEVICES=0,1 python -m example_trainer.grpo \
    --model-name $MODEL \
    --weight-bridge-mode none \
    --vllm-port 9001 \
    --atropos-url http://localhost:8001 \
    --training-steps 50 \
    --save-path $LOGDIR/checkpoints_legacy \
    --benchmark

# =============================================
# SHARED VLLM MODE (Terminals 4-7)
# =============================================

# Terminal 4: API server for shared
run-api --port 8002

# Terminal 5: vLLM server with shared weights
VLLM_ENABLE_SHARED_WEIGHTS=1 LOGDIR=$LOGDIR \
CUDA_VISIBLE_DEVICES=2 python example_trainer/vllm_api_server.py \
    --model $MODEL --port 9002 --gpu-memory-utilization 0.45

# Terminal 6: Environment server
python -u environments/gsm8k_server.py serve \
    --env.tokenizer_name $MODEL \
    --env.use_wandb=False \
    --openai.model_name $MODEL \
    --openai.base_url http://localhost:9002/v1 \
    --openai.server_type vllm \
    --server.port 8002

# Terminal 7: Trainer (attaches to vLLM)
CUDA_VISIBLE_DEVICES=2 python -m example_trainer.grpo \
    --model-name $MODEL \
    --weight-bridge-mode shared_vllm \
    --vllm-port 9002 \
    --vllm-config-path $LOGDIR/vllm_bridge_config.json \
    --atropos-url http://localhost:8002 \
    --training-steps 50 \
    --save-path $LOGDIR/checkpoints_shared \
    --benchmark

# =============================================
# LORA MODE (Terminals 8-11)
# =============================================

# Terminal 8: API server for LoRA
run-api --port 8003

# Terminal 9: vLLM server with LoRA
CUDA_VISIBLE_DEVICES=4 python example_trainer/vllm_api_server.py \
    --model $MODEL --port 9003 --gpu-memory-utilization 0.45 \
    --enable-lora --max-lora-rank 32 --enforce-eager

# Terminal 10: Environment server
python -u environments/gsm8k_server.py serve \
    --env.tokenizer_name $MODEL \
    --env.use_wandb=False \
    --openai.model_name $MODEL \
    --openai.base_url http://localhost:9003/v1 \
    --openai.server_type vllm \
    --server.port 8003

# Terminal 11: Trainer
CUDA_VISIBLE_DEVICES=5 python -m example_trainer.grpo \
    --model-name $MODEL \
    --weight-bridge-mode lora_only \
    --vllm-port 9003 \
    --atropos-url http://localhost:8003 \
    --lora-r 16 --lora-alpha 32 \
    --training-steps 50 \
    --save-path $LOGDIR/checkpoints_lora \
    --benchmark
```

---

## 📊 Understanding the Benchmark Output

Each trainer outputs a benchmark summary at the end:

```
======================================================================
BENCHMARK SUMMARY (shared_vllm)
======================================================================
  Total training time:     168.65s (2.81 min)
  Total steps:             50
  
  TIMING BREAKDOWN:
    Avg step time:         11.95s
    Total step time:       59.76s
    Avg sync time:         0.00s (x0 syncs)   <-- No syncs in shared mode!
    Total sync time:       0.00s
    Avg data fetch time:   10.90s
    Total data fetch time: 54.52s
  
  MEMORY:
    Peak GPU memory:       31.44 GB
    Avg GPU memory:        18.88 GB
======================================================================
```

**Key metrics to compare:**

| Metric | Legacy | Shared vLLM | LoRA |
|--------|--------|-------------|------|
| **Sync time** | High (restart vLLM) | 0 (in-place update) | Low (adapter swap) |
| **GPU memory** | 2x model | 1x model | 1x + adapter |
| **Step time** | ~10-15s | ~10-15s | ~5-10s |
| **Checkpoint size** | ~6GB | ~6GB | ~50MB |

---

## 🛠 CLI Reference

```bash
python -m example_trainer.grpo --help
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-name` | (required) | HuggingFace model ID |
| `--weight-bridge-mode` | `none` | `none`, `shared_vllm`, or `lora_only` |
| `--training-steps` | 10 | Number of training steps |
| `--batch-size` | 2 | Batch size |
| `--vllm-port` | 9001 | vLLM server port |
| `--atropos-url` | `http://localhost:8000` | Atropos API server URL |
| `--save-path` | `trained_model_checkpoints` | Checkpoint directory |
| `--benchmark` | false | Show timing stats |
| `--debug-loading` | false | Verbose model loading output |

### LoRA-specific Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--lora-r` | 16 | LoRA rank |
| `--lora-alpha` | 32 | LoRA alpha (scaling) |
| `--lora-dropout` | 0.05 | LoRA dropout |
| `--lora-target-modules` | `q_proj v_proj` | Modules to apply LoRA |

### Single-Copy Mode Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--single-copy` | false | Enable CUDA IPC mode |
| `--vllm-config-path` | auto-detect | Path to `vllm_bridge_config.json` |

---

## 🐛 Troubleshooting

### "Atropos API not reachable"
```bash
# Make sure run-api is running
run-api --port 8000
```

### "vLLM server not running" (LoRA mode)
```bash
# LoRA mode requires external vLLM with --enable-lora
python example_trainer/vllm_api_server.py \
    --model $MODEL --port 9001 --enable-lora --enforce-eager
```

### "Could not find vllm_bridge_config.json" (Shared mode)
```bash
# Make sure vLLM was started with VLLM_ENABLE_SHARED_WEIGHTS=1 and LOGDIR set
VLLM_ENABLE_SHARED_WEIGHTS=1 LOGDIR=/tmp/atropos python example_trainer/vllm_api_server.py ...
```

### "Triton compilation error" on B200/Blackwell GPUs
The patched vLLM server (`vllm_api_server.py`) automatically applies B200 fixes. If using standard vLLM, add `--enforce-eager`.

### Port already in use
```bash
# Kill existing processes
pkill -f "run-api"
pkill -f "vllm_api_server.py"
pkill -f "gsm8k_server.py"
```

### No batches available / trainer hangs
```bash
# Ensure the environment server is connected to the correct API and vLLM
# Check that vLLM is running and environment can reach it
curl http://localhost:9001/health
curl http://localhost:8000/info
```

---

## 📚 Module Documentation

### `config.py`
Contains `TrainingConfig` - all training parameters as a Pydantic model.

### `api.py`
- `check_atropos_api()` - Wait for run-api server
- `register_trainer()` - Register with Atropos
- `get_batch()` - Fetch training batch from run-api

### `data.py`
- `pad_data_to_good_offset()` - Pad sequences to GPU-friendly lengths
- `get_data()` - Fetch and preprocess batches

### `model.py`
- `load_model_and_tokenizer()` - Load model based on mode
- `_attach_to_vllm_shared_tensors()` - CUDA IPC attachment
- `_create_vllm_to_hf_mapping()` - Handle QKV/Gate-Up fusion

### `training.py`
- `compute_grpo_loss()` - GRPO loss computation
- `run_training_step()` - Single step with gradient accumulation
- `log_metrics()` - Console and WandB logging
- `finalize_training()` - Cleanup and summary

### `checkpointing.py`
- `save_checkpoint()` - Save full model
- `save_lora_checkpoint()` - Save LoRA adapter only

### `vllm_manager.py`
- `launch_vllm_server()` - Start vLLM process
- `terminate_vllm_process()` - Stop vLLM
- `hotswap_lora_adapter()` - Hot-swap LoRA in vLLM

### `trainers.py`
- `train_legacy()` - Checkpoint + restart mode
- `train_shared_vllm()` - Single-copy CUDA IPC mode
- `train_lora()` - Adapter training mode

### `cli.py`
- `parse_args()` - Argparse setup
- `config_from_args()` - Convert args to TrainingConfig

---

## 📝 License

MIT License
