#!/bin/bash
# ==============================================================================
# GRPO Training Mode Comparison Script
# ==============================================================================
# Runs all three training modes (Legacy, Shared vLLM, LoRA) in parallel
# on an 8-GPU node for comparison.
#
# GPU Allocation:
#   - GPUs 0-1: Legacy mode (trainer manages vLLM)
#   - GPUs 2-3: Shared vLLM mode (CUDA IPC single-copy)
#   - GPUs 4-5: LoRA mode (adapter training)
#   - GPUs 6-7: Reserved
#
# Port Allocation:
#   - Legacy:      API 8001, vLLM 9001
#   - Shared vLLM: API 8002, vLLM 9002
#   - LoRA:        API 8003, vLLM 9003
#
# Usage:
#   ./run_comparison.sh                    # Default 50 steps, logs to ./comparison_<timestamp>
#   ./run_comparison.sh 100                # 100 steps
#   LOGDIR=/my/path ./run_comparison.sh    # Custom log directory
#
# ==============================================================================
# OUTPUT DIRECTORY STRUCTURE ($LOGDIR):
# ==============================================================================
#
#   $LOGDIR/
#   ├── api_legacy.log          # run-api server log (port 8001)
#   ├── api_shared.log          # run-api server log (port 8002)
#   ├── api_lora.log            # run-api server log (port 8003)
#   ├── env_legacy.log          # gsm8k environment log
#   ├── env_shared.log          # gsm8k environment log
#   ├── env_lora.log            # gsm8k environment log
#   ├── vllm_shared.log         # vLLM server log (shared mode)
#   ├── vllm_lora.log           # vLLM server log (lora mode)
#   ├── trainer_legacy.log      # GRPO trainer log (MAIN OUTPUT)
#   ├── trainer_shared.log      # GRPO trainer log (MAIN OUTPUT)
#   ├── trainer_lora.log        # GRPO trainer log (MAIN OUTPUT)
#   ├── vllm_bridge_config_shared.json  # CUDA IPC config (shared mode)
#   ├── vllm_bridge_config_lora.json    # CUDA IPC config (lora mode)
#   ├── pids.txt                # Process IDs for cleanup
#   ├── checkpoints_legacy/     # Model checkpoints
#   │   ├── step_3/
#   │   ├── step_6/
#   │   └── final_model/
#   ├── checkpoints_shared/     # Model checkpoints
#   │   └── final_model/
#   └── checkpoints_lora/       # LoRA adapter checkpoints
#       ├── adapter_step_3/
#       ├── adapter_step_6/
#       ├── final_adapter/
#       └── tokenizer/
#
# ==============================================================================

set -e

# Configuration
export MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
export TRAINING_STEPS="${1:-50}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export LOGDIR="${LOGDIR:-./comparison_$(date +%Y%m%d_%H%M%S)}"

mkdir -p $LOGDIR

# ==============================================================================
# Helper function: Wait for vLLM to be ready
# ==============================================================================
wait_for_vllm() {
    local port=$1
    local name=$2
    local max_attempts=${3:-60}  # Default 60 attempts (5 minutes with 5s sleep)
    local attempt=1
    
    echo "  Waiting for vLLM ($name) on port $port..."
    while [ $attempt -le $max_attempts ]; do
        if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
            echo "  ✓ vLLM ($name) is ready after ~$((attempt * 5))s"
            return 0
        fi
        echo "    Attempt $attempt/$max_attempts - vLLM not ready yet..."
        sleep 5
        attempt=$((attempt + 1))
    done
    
    echo "  ✗ vLLM ($name) failed to start after $((max_attempts * 5))s"
    return 1
}

# ==============================================================================
# Helper function: Wait for API server to be ready
# ==============================================================================
wait_for_api() {
    local port=$1
    local name=$2
    local max_attempts=${3:-20}
    local attempt=1
    
    echo "  Waiting for API ($name) on port $port..."
    while [ $attempt -le $max_attempts ]; do
        if curl -s "http://localhost:$port/info" > /dev/null 2>&1; then
            echo "  ✓ API ($name) is ready"
            return 0
        fi
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo "  ✗ API ($name) failed to start"
    return 1
}

echo "=============================================="
echo "GRPO Training Mode Comparison"
echo "=============================================="
echo "Model:        $MODEL"
echo "Steps:        $TRAINING_STEPS"
echo "Batch size:   $BATCH_SIZE"
echo "Log dir:      $LOGDIR"
echo "=============================================="
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up processes..."
    if [ -f "$LOGDIR/pids.txt" ]; then
        source $LOGDIR/pids.txt
        kill $LEGACY_TRAINER_PID $LEGACY_ENV_PID $LEGACY_API_PID 2>/dev/null || true
        kill $SHARED_TRAINER_PID $SHARED_VLLM_PID $SHARED_ENV_PID $SHARED_API_PID 2>/dev/null || true
        kill $LORA_TRAINER_PID $LORA_VLLM_PID $LORA_ENV_PID $LORA_API_PID 2>/dev/null || true
    fi
    pkill -f "gsm8k_server.py.*800[123]" 2>/dev/null || true
    pkill -f "run_api.*800[123]" 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT

# Kill any existing processes on our ports
echo "Killing any existing processes on ports 8001-8003, 9001-9003..."
pkill -f "vllm_api_server.py.*900[123]" 2>/dev/null || true
pkill -f "gsm8k_server.py.*800[123]" 2>/dev/null || true
pkill -f "run_api.*800[123]" 2>/dev/null || true
sleep 2

# ==============================================================================
# MODE 1: LEGACY (GPUs 0-1, API 8001, vLLM 9001)
# ==============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[1/3] LEGACY MODE (GPUs 0-1, API:8001, vLLM:9001)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Start run-api server for Legacy
echo "  Starting run-api server..."
run-api --port 8001 > $LOGDIR/api_legacy.log 2>&1 &
LEGACY_API_PID=$!
echo "  ✓ run-api started (PID: $LEGACY_API_PID, port 8001)"
wait_for_api 8001 "legacy" || { echo "Failed to start legacy API"; exit 1; }

# In legacy mode: trainer launches vLLM internally, so start trainer FIRST
echo "  Starting trainer (will launch internal vLLM on port 9001)..."
CUDA_VISIBLE_DEVICES=0,1 python -m example_trainer.grpo \
    --model-name $MODEL \
    --weight-bridge-mode none \
    --vllm-port 9001 \
    --atropos-url http://localhost:8001 \
    --training-steps $TRAINING_STEPS \
    --batch-size $BATCH_SIZE \
    --save-path $LOGDIR/checkpoints_legacy \
    --benchmark \
    > $LOGDIR/trainer_legacy.log 2>&1 &
LEGACY_TRAINER_PID=$!
echo "  ✓ Trainer started (PID: $LEGACY_TRAINER_PID)"

# Wait for trainer's internal vLLM to be ready
wait_for_vllm 9001 "legacy (internal)" || { echo "Legacy vLLM failed to start"; exit 1; }

# NOW start environment server (after vLLM is ready)
echo "  Starting environment server..."
python environments/gsm8k_server.py serve \
    --slurm.num_gpus 0 \
    --env.tokenizer_name $MODEL \
    --openai.base_url http://localhost:9001/v1 \
    --server.port 8001 \
    > $LOGDIR/env_legacy.log 2>&1 &
LEGACY_ENV_PID=$!
echo "  ✓ Environment server started (PID: $LEGACY_ENV_PID)"
sleep 3

# ==============================================================================
# MODE 2: SHARED_VLLM (GPUs 2-3, API 8002, vLLM 9002)
# ==============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[2/3] SHARED_VLLM MODE (GPUs 2-3, API:8002, vLLM:9002)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Start run-api server for Shared
echo "  Starting run-api server..."
run-api --port 8002 > $LOGDIR/api_shared.log 2>&1 &
SHARED_API_PID=$!
echo "  ✓ run-api started (PID: $SHARED_API_PID, port 8002)"
wait_for_api 8002 "shared" || { echo "Failed to start shared API"; exit 1; }

# Start vLLM with shared weights (use separate config path)
echo "  Starting vLLM with shared weights..."
VLLM_ENABLE_SHARED_WEIGHTS=1 VLLM_BRIDGE_CONFIG_PATH=$LOGDIR/vllm_bridge_config_shared.json \
CUDA_VISIBLE_DEVICES=2 python example_trainer/vllm_api_server.py \
    --model $MODEL \
    --port 9002 \
    --gpu-memory-utilization 0.45 \
    > $LOGDIR/vllm_shared.log 2>&1 &
SHARED_VLLM_PID=$!
echo "  ✓ vLLM started (PID: $SHARED_VLLM_PID, port 9002)"
wait_for_vllm 9002 "shared" || { echo "Failed to start shared vLLM"; exit 1; }

# Start environment server for Shared
echo "  Starting environment server..."
python environments/gsm8k_server.py serve \
    --slurm.num_gpus 0 \
    --env.tokenizer_name $MODEL \
    --openai.base_url http://localhost:9002/v1 \
    --server.port 8002 \
    > $LOGDIR/env_shared.log 2>&1 &
SHARED_ENV_PID=$!
echo "  ✓ Environment server started (PID: $SHARED_ENV_PID)"
sleep 5

# Start Shared vLLM trainer
echo "  Starting trainer..."
CUDA_VISIBLE_DEVICES=2 python -m example_trainer.grpo \
    --model-name $MODEL \
    --weight-bridge-mode shared_vllm \
    --vllm-port 9002 \
    --vllm-config-path $LOGDIR/vllm_bridge_config_shared.json \
    --atropos-url http://localhost:8002 \
    --training-steps $TRAINING_STEPS \
    --batch-size $BATCH_SIZE \
    --save-path $LOGDIR/checkpoints_shared \
    --benchmark \
    > $LOGDIR/trainer_shared.log 2>&1 &
SHARED_TRAINER_PID=$!
echo "  ✓ Trainer started (PID: $SHARED_TRAINER_PID)"

# ==============================================================================
# MODE 3: LORA (GPUs 4-5, API 8003, vLLM 9003)
# ==============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[3/3] LORA MODE (GPUs 4-5, API:8003, vLLM:9003)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Start run-api server for LoRA
echo "  Starting run-api server..."
run-api --port 8003 > $LOGDIR/api_lora.log 2>&1 &
LORA_API_PID=$!
echo "  ✓ run-api started (PID: $LORA_API_PID, port 8003)"
wait_for_api 8003 "lora" || { echo "Failed to start lora API"; exit 1; }

# Start vLLM with LoRA support (use separate config path)
echo "  Starting vLLM with LoRA support..."
VLLM_BRIDGE_CONFIG_PATH=$LOGDIR/vllm_bridge_config_lora.json \
CUDA_VISIBLE_DEVICES=4 python example_trainer/vllm_api_server.py \
    --model $MODEL \
    --port 9003 \
    --gpu-memory-utilization 0.45 \
    --enable-lora \
    --max-lora-rank 32 \
    --enforce-eager \
    > $LOGDIR/vllm_lora.log 2>&1 &
LORA_VLLM_PID=$!
echo "  ✓ vLLM started (PID: $LORA_VLLM_PID, port 9003)"
wait_for_vllm 9003 "lora" || { echo "Failed to start lora vLLM"; exit 1; }

# Start environment server for LoRA
echo "  Starting environment server..."
python environments/gsm8k_server.py serve \
    --slurm.num_gpus 0 \
    --env.tokenizer_name $MODEL \
    --openai.base_url http://localhost:9003/v1 \
    --server.port 8003 \
    > $LOGDIR/env_lora.log 2>&1 &
LORA_ENV_PID=$!
echo "  ✓ Environment server started (PID: $LORA_ENV_PID)"
sleep 5

# Start LoRA trainer
echo "  Starting trainer..."
CUDA_VISIBLE_DEVICES=5 python -m example_trainer.grpo \
    --model-name $MODEL \
    --weight-bridge-mode lora_only \
    --vllm-port 9003 \
    --atropos-url http://localhost:8003 \
    --lora-r 16 \
    --lora-alpha 32 \
    --training-steps $TRAINING_STEPS \
    --batch-size $BATCH_SIZE \
    --save-path $LOGDIR/checkpoints_lora \
    --benchmark \
    > $LOGDIR/trainer_lora.log 2>&1 &
LORA_TRAINER_PID=$!
echo "  ✓ Trainer started (PID: $LORA_TRAINER_PID)"

# ==============================================================================
# Save PIDs and Monitor
# ==============================================================================
cat > $LOGDIR/pids.txt << EOF
LEGACY_TRAINER_PID=$LEGACY_TRAINER_PID
LEGACY_ENV_PID=$LEGACY_ENV_PID
LEGACY_API_PID=$LEGACY_API_PID
SHARED_TRAINER_PID=$SHARED_TRAINER_PID
SHARED_VLLM_PID=$SHARED_VLLM_PID
SHARED_ENV_PID=$SHARED_ENV_PID
SHARED_API_PID=$SHARED_API_PID
LORA_TRAINER_PID=$LORA_TRAINER_PID
LORA_VLLM_PID=$LORA_VLLM_PID
LORA_ENV_PID=$LORA_ENV_PID
LORA_API_PID=$LORA_API_PID
EOF

echo ""
echo "=============================================="
echo "All components started!"
echo "=============================================="
echo ""
echo "📂 Log directory: $LOGDIR"
echo ""
echo "📊 Monitor progress:"
echo "  tail -f $LOGDIR/trainer_legacy.log"
echo "  tail -f $LOGDIR/trainer_shared.log"
echo "  tail -f $LOGDIR/trainer_lora.log"
echo ""
echo "🔍 Or watch all at once:"
echo "  tail -f $LOGDIR/trainer_*.log"
echo ""
echo "📋 Check API servers:"
echo "  curl http://localhost:8001/info"
echo "  curl http://localhost:8002/info"
echo "  curl http://localhost:8003/info"
echo ""
echo "📝 Process IDs saved to: $LOGDIR/pids.txt"
echo ""
echo "Waiting for all trainers to complete..."
echo "(This may take a while depending on training steps)"
echo ""

# Wait for trainers to complete
wait $LEGACY_TRAINER_PID 2>/dev/null && echo "  ✓ Legacy trainer completed" || echo "  ✗ Legacy trainer failed"
wait $SHARED_TRAINER_PID 2>/dev/null && echo "  ✓ Shared vLLM trainer completed" || echo "  ✗ Shared vLLM trainer failed"
wait $LORA_TRAINER_PID 2>/dev/null && echo "  ✓ LoRA trainer completed" || echo "  ✗ LoRA trainer failed"

# ==============================================================================
# Print Results
# ==============================================================================
echo ""
echo "=============================================="
echo "COMPARISON RESULTS"
echo "=============================================="
echo ""

echo "📊 LEGACY MODE:"
echo "─────────────────────────────────────────────"
grep -A 15 "BENCHMARK SUMMARY" $LOGDIR/trainer_legacy.log 2>/dev/null || echo "  (check $LOGDIR/trainer_legacy.log)"
echo ""

echo "📊 SHARED VLLM MODE:"
echo "─────────────────────────────────────────────"
grep -A 15 "BENCHMARK SUMMARY" $LOGDIR/trainer_shared.log 2>/dev/null || echo "  (check $LOGDIR/trainer_shared.log)"
echo ""

echo "📊 LORA MODE:"
echo "─────────────────────────────────────────────"
grep -A 15 "BENCHMARK SUMMARY" $LOGDIR/trainer_lora.log 2>/dev/null || echo "  (check $LOGDIR/trainer_lora.log)"
echo ""

echo "=============================================="
echo "📁 ALL OUTPUT SAVED TO: $LOGDIR"
echo "=============================================="
echo ""
echo "📋 LOG FILES:"
echo "  Trainers (main output):"
echo "    $LOGDIR/trainer_legacy.log"
echo "    $LOGDIR/trainer_shared.log"
echo "    $LOGDIR/trainer_lora.log"
echo ""
echo "  vLLM servers:"
echo "    $LOGDIR/vllm_shared.log"
echo "    $LOGDIR/vllm_lora.log"
echo ""
echo "  Environment servers:"
echo "    $LOGDIR/env_legacy.log"
echo "    $LOGDIR/env_shared.log"
echo "    $LOGDIR/env_lora.log"
echo ""
echo "  API servers:"
echo "    $LOGDIR/api_legacy.log"
echo "    $LOGDIR/api_shared.log"
echo "    $LOGDIR/api_lora.log"
echo ""
echo "💾 CHECKPOINTS:"
echo "  Legacy:      $LOGDIR/checkpoints_legacy/final_model/"
echo "  Shared vLLM: $LOGDIR/checkpoints_shared/final_model/"
echo "  LoRA:        $LOGDIR/checkpoints_lora/final_adapter/"
echo ""
echo "🔧 OTHER:"
echo "  Process IDs: $LOGDIR/pids.txt"
echo "  IPC Config (shared): $LOGDIR/vllm_bridge_config_shared.json"
echo "  IPC Config (lora):   $LOGDIR/vllm_bridge_config_lora.json"
echo "=============================================="
echo ""
echo "To re-run or inspect later:"
echo "  export LOGDIR=$LOGDIR"
echo "  tail -f \$LOGDIR/trainer_*.log"
echo ""
echo "Done!"
