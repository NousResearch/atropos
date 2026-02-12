#!/bin/bash
# =============================================================================
# All Training Modes Comparison on Math Zero (32k context)
# =============================================================================
#
# Compares all 3 training modes on math_server_zero environment:
#   - GPU 0: shared_vllm (CUDA IPC, zero-copy weight updates)
#   - GPU 1: lora_only   (--enforce-eager, ~13 TPS, slow)
#   - GPU 2: lora_restart (no --enforce-eager, ~108 TPS, fast)
#
# All at 32k context length for proper math reasoning.
#
# Usage:
#   ./scripts/compare_all_modes_math_zero.sh [MODEL] [STEPS]
#
# Example:
#   ./scripts/compare_all_modes_math_zero.sh Qwen/Qwen3-4B-Instruct-2507 30
#
# =============================================================================

set -e

MODEL="${1:-Qwen/Qwen3-4B-Instruct-2507}"
TRAINING_STEPS="${2:-30}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
USE_WANDB="${USE_WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-math-zero-mode-comparison}"

# Port allocation (separate ports for each mode)
# shared_vllm: API 8001, vLLM 9001
# lora_only:   API 8002, vLLM 9002
# lora_restart: API 8003, vLLM 9003

SHARED_API_PORT=8001
SHARED_VLLM_PORT=9001
SHARED_GPU=0

LORA_ONLY_API_PORT=8002
LORA_ONLY_VLLM_PORT=9002
LORA_ONLY_GPU=1

LORA_RESTART_API_PORT=8003
LORA_RESTART_VLLM_PORT=9003
LORA_RESTART_GPU=2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINER_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$TRAINER_DIR")"

LOG_DIR="${REPO_DIR}/math_zero_comparison_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Math Zero Mode Comparison (32k Context)"
echo "============================================================"
echo "Model: $MODEL"
echo "Steps: $TRAINING_STEPS"
echo "Batch: $BATCH_SIZE"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "Wandb: $USE_WANDB (project: $WANDB_PROJECT)"
echo ""
echo "GPU Allocation:"
echo "  GPU $SHARED_GPU: shared_vllm (ports $SHARED_API_PORT, $SHARED_VLLM_PORT)"
echo "  GPU $LORA_ONLY_GPU: lora_only (ports $LORA_ONLY_API_PORT, $LORA_ONLY_VLLM_PORT)"
echo "  GPU $LORA_RESTART_GPU: lora_restart (ports $LORA_RESTART_API_PORT, $LORA_RESTART_VLLM_PORT)"
echo ""
echo "Log Dir: $LOG_DIR"
echo "============================================================"
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up all processes..."
    pkill -9 -f "vllm_api_server" 2>/dev/null || true
    pkill -9 -f "math_server_zero" 2>/dev/null || true
    pkill -9 -f "run-api" 2>/dev/null || true
    pkill -9 -f "grpo" 2>/dev/null || true
    pkill -9 -f "vllm.*EngineCore" 2>/dev/null || true
    for port in $SHARED_API_PORT $SHARED_VLLM_PORT $LORA_ONLY_API_PORT $LORA_ONLY_VLLM_PORT $LORA_RESTART_API_PORT $LORA_RESTART_VLLM_PORT; do
        fuser -k ${port}/tcp 2>/dev/null || true
    done
    sleep 2
}
trap cleanup EXIT

# Initial cleanup
cleanup

# Clear triton cache for clean start
rm -rf ~/.triton/cache 2>/dev/null || true

cd "$REPO_DIR"

# =============================================================================
# Helper functions
# =============================================================================

wait_for_health() {
    local port=$1
    local name=$2
    local max_attempts=${3:-120}
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
            echo "  ✓ $name ready (port $port)"
            return 0
        fi
        sleep 5
        attempt=$((attempt + 1))
    done
    echo "  ✗ $name failed to start (port $port)"
    return 1
}

wait_for_api() {
    local port=$1
    local name=$2
    local max_attempts=${3:-30}
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "http://localhost:$port/info" > /dev/null 2>&1; then
            echo "  ✓ $name ready (port $port)"
            return 0
        fi
        sleep 2
        attempt=$((attempt + 1))
    done
    echo "  ✗ $name failed to start (port $port)"
    return 1
}

# =============================================================================
# START ALL THREE MODES IN PARALLEL
# =============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Starting all three modes in parallel..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Pre-create checkpoint directories
mkdir -p "$LOG_DIR/checkpoints_shared"
mkdir -p "$LOG_DIR/checkpoints_lora_only"
mkdir -p "$LOG_DIR/checkpoints_lora_restart"

# -----------------------------------------------------------------------------
# MODE 1: SHARED_VLLM (GPU 0)
# -----------------------------------------------------------------------------
echo ""
echo "[SHARED_VLLM] Starting on GPU $SHARED_GPU..."

# Start run-api for shared_vllm
run-api --port $SHARED_API_PORT > "$LOG_DIR/api_shared.log" 2>&1 &

# Start vLLM with shared weights
echo "[SHARED_VLLM] Starting vLLM with shared weights..."
VLLM_ENABLE_SHARED_WEIGHTS=1 VLLM_BRIDGE_CONFIG_PATH=$LOG_DIR/vllm_bridge_config_shared.json \
CUDA_VISIBLE_DEVICES=$SHARED_GPU python -u example_trainer/vllm_api_server.py \
    --model "$MODEL" \
    --port $SHARED_VLLM_PORT \
    --gpu-memory-utilization 0.85 \
    --max-model-len $MAX_MODEL_LEN \
    > "$LOG_DIR/vllm_shared.log" 2>&1 &

# -----------------------------------------------------------------------------
# MODE 2: LORA_ONLY (GPU 1)
# -----------------------------------------------------------------------------
echo ""
echo "[LORA_ONLY] Starting on GPU $LORA_ONLY_GPU..."

# Start run-api for lora_only
run-api --port $LORA_ONLY_API_PORT > "$LOG_DIR/api_lora_only.log" 2>&1 &

# Start vLLM with --enforce-eager for lora_only
echo "[LORA_ONLY] Starting vLLM with --enable-lora --enforce-eager..."
CUDA_VISIBLE_DEVICES=$LORA_ONLY_GPU python -u example_trainer/vllm_api_server.py \
    --model "$MODEL" \
    --port $LORA_ONLY_VLLM_PORT \
    --gpu-memory-utilization 0.85 \
    --max-model-len $MAX_MODEL_LEN \
    --enable-lora \
    --max-lora-rank 64 \
    --enforce-eager \
    > "$LOG_DIR/vllm_lora_only.log" 2>&1 &

# -----------------------------------------------------------------------------
# MODE 3: LORA_RESTART (GPU 2) - Trainer manages vLLM internally
# -----------------------------------------------------------------------------
echo ""
echo "[LORA_RESTART] Starting on GPU $LORA_RESTART_GPU..."

# Start run-api for lora_restart
run-api --port $LORA_RESTART_API_PORT > "$LOG_DIR/api_lora_restart.log" 2>&1 &

# =============================================================================
# WAIT FOR INFRASTRUCTURE
# =============================================================================
echo ""
echo "Waiting for infrastructure to be ready..."
echo "  (vLLM at 32k context takes ~2-5 minutes to start)"

wait_for_api $SHARED_API_PORT "shared_vllm API" || exit 1
wait_for_api $LORA_ONLY_API_PORT "lora_only API" || exit 1
wait_for_api $LORA_RESTART_API_PORT "lora_restart API" || exit 1

wait_for_health $SHARED_VLLM_PORT "shared_vllm vLLM" 180 || exit 1
wait_for_health $LORA_ONLY_VLLM_PORT "lora_only vLLM" 180 || exit 1

# =============================================================================
# START ENVIRONMENTS AND TRAINERS
# =============================================================================
echo ""
echo "Starting environments and trainers..."

# Record start time
START_TIME=$(date +%s)

# Build wandb args
WANDB_ARGS=""
if [ "$USE_WANDB" = "true" ]; then
    WANDB_ARGS="--use-wandb --wandb-project $WANDB_PROJECT"
fi

# -----------------------------------------------------------------------------
# SHARED_VLLM: Start environment and trainer
# -----------------------------------------------------------------------------
echo ""
echo "[SHARED_VLLM] Starting math_server_zero environment..."
MATH_ENV_MODEL="$MODEL" \
MATH_ENV_ROLLOUT_URL="http://localhost:${SHARED_API_PORT}" \
MATH_ENV_VLLM_URL="http://localhost:${SHARED_VLLM_PORT}/v1" \
MATH_ENV_WANDB_NAME="shared-vllm-env" \
MATH_ENV_MAX_TOKENS=$MAX_MODEL_LEN \
MATH_ENV_WORKER_TIMEOUT=1800 \
python -u environments/math_server_zero.py serve \
    --slurm false \
    2>&1 | tee "$LOG_DIR/env_shared.log" &
SHARED_ENV_PID=$!

echo "[SHARED_VLLM] Starting trainer..."
CUDA_VISIBLE_DEVICES=$SHARED_GPU python -m example_trainer.grpo \
    --model-name "$MODEL" \
    --weight-bridge-mode shared_vllm \
    --vllm-port $SHARED_VLLM_PORT \
    --vllm-config-path "$LOG_DIR/vllm_bridge_config_shared.json" \
    --atropos-url "http://localhost:${SHARED_API_PORT}" \
    --batch-size $BATCH_SIZE \
    --training-steps $TRAINING_STEPS \
    --max-model-len $MAX_MODEL_LEN \
    --save-path "$LOG_DIR/checkpoints_shared" \
    $WANDB_ARGS --wandb-group "shared-vllm" \
    --benchmark \
    2>&1 | tee "$LOG_DIR/trainer_shared.log" &
SHARED_TRAINER_PID=$!

# -----------------------------------------------------------------------------
# LORA_ONLY: Start environment and trainer
# -----------------------------------------------------------------------------
echo ""
echo "[LORA_ONLY] Starting math_server_zero environment..."
MATH_ENV_MODEL="$MODEL" \
MATH_ENV_ROLLOUT_URL="http://localhost:${LORA_ONLY_API_PORT}" \
MATH_ENV_VLLM_URL="http://localhost:${LORA_ONLY_VLLM_PORT}/v1" \
MATH_ENV_WANDB_NAME="lora-only-env" \
MATH_ENV_MAX_TOKENS=$MAX_MODEL_LEN \
MATH_ENV_WORKER_TIMEOUT=1800 \
python -u environments/math_server_zero.py serve \
    --slurm false \
    2>&1 | tee "$LOG_DIR/env_lora_only.log" &
LORA_ONLY_ENV_PID=$!

echo "[LORA_ONLY] Starting trainer..."
CUDA_VISIBLE_DEVICES=$LORA_ONLY_GPU python -m example_trainer.grpo \
    --model-name "$MODEL" \
    --weight-bridge-mode lora_only \
    --vllm-port $LORA_ONLY_VLLM_PORT \
    --atropos-url "http://localhost:${LORA_ONLY_API_PORT}" \
    --batch-size $BATCH_SIZE \
    --training-steps $TRAINING_STEPS \
    --max-model-len $MAX_MODEL_LEN \
    --lora-r 16 \
    --lora-alpha 32 \
    --vllm-restart-interval 5 \
    --save-path "$LOG_DIR/checkpoints_lora_only" \
    $WANDB_ARGS --wandb-group "lora-only" \
    --benchmark \
    2>&1 | tee "$LOG_DIR/trainer_lora_only.log" &
LORA_ONLY_TRAINER_PID=$!

# -----------------------------------------------------------------------------
# LORA_RESTART: Start trainer (it manages vLLM internally)
# -----------------------------------------------------------------------------
echo ""
echo "[LORA_RESTART] Starting trainer (manages vLLM internally)..."
CUDA_VISIBLE_DEVICES=$LORA_RESTART_GPU python -m example_trainer.grpo \
    --model-name "$MODEL" \
    --weight-bridge-mode lora_restart \
    --vllm-port $LORA_RESTART_VLLM_PORT \
    --vllm-gpu-memory-utilization 0.85 \
    --atropos-url "http://localhost:${LORA_RESTART_API_PORT}" \
    --batch-size $BATCH_SIZE \
    --training-steps $TRAINING_STEPS \
    --max-model-len $MAX_MODEL_LEN \
    --lora-r 16 \
    --lora-alpha 32 \
    --vllm-restart-interval 5 \
    --save-path "$LOG_DIR/checkpoints_lora_restart" \
    $WANDB_ARGS --wandb-group "lora-restart" \
    --benchmark \
    2>&1 | tee "$LOG_DIR/trainer_lora_restart.log" &
LORA_RESTART_TRAINER_PID=$!

# Wait for lora_restart's internal vLLM to start
echo "[LORA_RESTART] Waiting for internal vLLM to start..."
echo "  NOTE: vLLM at 32k context with CUDA graphs takes 2-5 min"
sleep 60
wait_for_health $LORA_RESTART_VLLM_PORT "lora_restart internal vLLM" 300 || {
    echo "  Failed - check logs:"
    tail -50 "$LOG_DIR/trainer_lora_restart.log"
    exit 1
}

# Start environment for lora_restart
echo "[LORA_RESTART] Starting math_server_zero environment..."
MATH_ENV_MODEL="$MODEL" \
MATH_ENV_ROLLOUT_URL="http://localhost:${LORA_RESTART_API_PORT}" \
MATH_ENV_VLLM_URL="http://localhost:${LORA_RESTART_VLLM_PORT}/v1" \
MATH_ENV_WANDB_NAME="lora-restart-env" \
MATH_ENV_MAX_TOKENS=$MAX_MODEL_LEN \
MATH_ENV_WORKER_TIMEOUT=1800 \
python -u environments/math_server_zero.py serve \
    --slurm false \
    2>&1 | tee "$LOG_DIR/env_lora_restart.log" &
LORA_RESTART_ENV_PID=$!

# =============================================================================
# WAIT FOR ALL TRAINERS TO COMPLETE
# =============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "All three trainers running in parallel. Waiting for completion..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 WANDB: https://wandb.ai (project: $WANDB_PROJECT)"
echo ""
echo "📋 MONITOR LOGS (in another terminal):"
echo ""
echo "  # Trainer logs:"
echo "  tail -f $LOG_DIR/trainer_shared.log"
echo "  tail -f $LOG_DIR/trainer_lora_only.log"
echo "  tail -f $LOG_DIR/trainer_lora_restart.log"
echo ""
echo "  # Environment logs:"
echo "  tail -f $LOG_DIR/env_shared.log"
echo "  tail -f $LOG_DIR/env_lora_only.log"
echo "  tail -f $LOG_DIR/env_lora_restart.log"
echo ""
echo "  # vLLM logs:"
echo "  tail -f $LOG_DIR/vllm_shared.log"
echo "  tail -f $LOG_DIR/vllm_lora_only.log"
echo "  tail -f $LOG_DIR/checkpoints_lora_restart/vllm_restart_*.log"
echo ""

# Wait for trainers
SHARED_EXIT=0
LORA_ONLY_EXIT=0
LORA_RESTART_EXIT=0

wait $SHARED_TRAINER_PID || SHARED_EXIT=$?
SHARED_END=$(date +%s)
SHARED_TIME=$((SHARED_END - START_TIME))
echo "  ✓ shared_vllm finished in ${SHARED_TIME}s (exit: $SHARED_EXIT)"

wait $LORA_ONLY_TRAINER_PID || LORA_ONLY_EXIT=$?
LORA_ONLY_END=$(date +%s)
LORA_ONLY_TIME=$((LORA_ONLY_END - START_TIME))
echo "  ✓ lora_only finished in ${LORA_ONLY_TIME}s (exit: $LORA_ONLY_EXIT)"

wait $LORA_RESTART_TRAINER_PID || LORA_RESTART_EXIT=$?
LORA_RESTART_END=$(date +%s)
LORA_RESTART_TIME=$((LORA_RESTART_END - START_TIME))
echo "  ✓ lora_restart finished in ${LORA_RESTART_TIME}s (exit: $LORA_RESTART_EXIT)"

# =============================================================================
# RESULTS
# =============================================================================
echo ""
echo "============================================================"
echo "COMPARISON RESULTS (Math Zero @ 32k Context)"
echo "============================================================"
echo ""
echo "Training Steps: $TRAINING_STEPS"
echo "Batch Size: $BATCH_SIZE"
echo "Max Context: $MAX_MODEL_LEN"
echo ""
echo "┌─────────────────┬──────┬──────────────┬────────────────────────────────┐"
echo "│ Mode            │ GPU  │ Total Time   │ Notes                          │"
echo "├─────────────────┼──────┼──────────────┼────────────────────────────────┤"
printf "│ shared_vllm     │  %d   │ %10ss │ CUDA IPC zero-copy (~172 TPS)  │\n" "$SHARED_GPU" "$SHARED_TIME"
printf "│ lora_only       │  %d   │ %10ss │ --enforce-eager (~13 TPS)      │\n" "$LORA_ONLY_GPU" "$LORA_ONLY_TIME"
printf "│ lora_restart    │  %d   │ %10ss │ no --enforce-eager (~108 TPS)  │\n" "$LORA_RESTART_GPU" "$LORA_RESTART_TIME"
echo "└─────────────────┴──────┴──────────────┴────────────────────────────────┘"
echo ""

# Calculate speedups
if [ $LORA_ONLY_TIME -gt 0 ] && [ $LORA_RESTART_TIME -gt 0 ]; then
    RESTART_SPEEDUP=$(echo "scale=2; $LORA_ONLY_TIME / $LORA_RESTART_TIME" | bc)
    echo "lora_restart vs lora_only speedup: ${RESTART_SPEEDUP}x"
fi
if [ $LORA_ONLY_TIME -gt 0 ] && [ $SHARED_TIME -gt 0 ]; then
    SHARED_SPEEDUP=$(echo "scale=2; $LORA_ONLY_TIME / $SHARED_TIME" | bc)
    echo "shared_vllm vs lora_only speedup: ${SHARED_SPEEDUP}x"
fi

echo ""
echo "📊 BENCHMARK DETAILS:"
echo ""
echo "━━━ shared_vllm (GPU $SHARED_GPU) ━━━"
grep -A 15 "BENCHMARK SUMMARY" "$LOG_DIR/trainer_shared.log" 2>/dev/null || echo "  (check $LOG_DIR/trainer_shared.log)"
echo ""
echo "━━━ lora_only (GPU $LORA_ONLY_GPU) ━━━"
grep -A 15 "BENCHMARK SUMMARY" "$LOG_DIR/trainer_lora_only.log" 2>/dev/null || echo "  (check $LOG_DIR/trainer_lora_only.log)"
echo ""
echo "━━━ lora_restart (GPU $LORA_RESTART_GPU) ━━━"
grep -A 15 "BENCHMARK SUMMARY" "$LOG_DIR/trainer_lora_restart.log" 2>/dev/null || echo "  (check $LOG_DIR/trainer_lora_restart.log)"

echo ""
echo "============================================================"
echo "📁 All logs saved to: $LOG_DIR"
echo "============================================================"
echo ""
echo "Log files:"
echo "  Trainers:"
echo "    $LOG_DIR/trainer_shared.log"
echo "    $LOG_DIR/trainer_lora_only.log"
echo "    $LOG_DIR/trainer_lora_restart.log"
echo ""
echo "  Environments:"
echo "    $LOG_DIR/env_shared.log"
echo "    $LOG_DIR/env_lora_only.log"
echo "    $LOG_DIR/env_lora_restart.log"
echo ""
echo "  vLLM:"
echo "    $LOG_DIR/vllm_shared.log"
echo "    $LOG_DIR/vllm_lora_only.log"
echo "    $LOG_DIR/checkpoints_lora_restart/vllm_restart_*.log"
echo ""
