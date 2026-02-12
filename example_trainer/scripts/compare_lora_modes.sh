#!/bin/bash
# =============================================================================
# LoRA Mode Comparison: lora_only vs lora_restart (PARALLEL)
# =============================================================================
#
# Runs both modes IN PARALLEL on separate GPUs for fair comparison:
#   - GPU 0: lora_only    (--enforce-eager, ~13 TPS)
#   - GPU 1: lora_restart (no --enforce-eager, ~108 TPS)
#
# Usage:
#   ./scripts/compare_lora_modes.sh [MODEL] [STEPS]
#
# Example:
#   ./scripts/compare_lora_modes.sh Qwen/Qwen3-4B-Instruct-2507 20
#
# =============================================================================

set -e

MODEL="${1:-Qwen/Qwen3-4B-Instruct-2507}"
TRAINING_STEPS="${2:-20}"
BATCH_SIZE="${BATCH_SIZE:-2}"
USE_WANDB="${USE_WANDB:-true}"  # Set USE_WANDB=false to disable
WANDB_PROJECT="${WANDB_PROJECT:-lora-mode-comparison}"

# Port allocation (separate ports for each mode)
LORA_ONLY_VLLM_PORT=9001
LORA_ONLY_API_PORT=8001

LORA_RESTART_VLLM_PORT=9002
LORA_RESTART_API_PORT=8002

# GPU allocation
LORA_ONLY_GPU=0
LORA_RESTART_GPU=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINER_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$TRAINER_DIR")"

LOG_DIR="${REPO_DIR}/lora_comparison_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "LoRA Mode Comparison: lora_only vs lora_restart (PARALLEL)"
echo "============================================================"
echo "Model: $MODEL"
echo "Steps: $TRAINING_STEPS"
echo "Batch: $BATCH_SIZE"
echo "Wandb: $USE_WANDB (project: $WANDB_PROJECT)"
echo ""
echo "GPU Allocation:"
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
    pkill -u $USER -f "vllm_api_server" 2>/dev/null || true
    pkill -u $USER -f "gsm8k_server" 2>/dev/null || true
    pkill -u $USER -f "run-api" 2>/dev/null || true
    pkill -u $USER -f "grpo" 2>/dev/null || true
    for port in $LORA_ONLY_VLLM_PORT $LORA_ONLY_API_PORT $LORA_RESTART_VLLM_PORT $LORA_RESTART_API_PORT; do
        fuser -k ${port}/tcp 2>/dev/null || true
    done
    sleep 2
}
trap cleanup EXIT

# Initial cleanup
cleanup

cd "$REPO_DIR"

# =============================================================================
# Helper functions
# =============================================================================

wait_for_health() {
    local port=$1
    local name=$2
    local max_attempts=${3:-60}
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
# START BOTH MODES IN PARALLEL
# =============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Starting both modes in parallel..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# -----------------------------------------------------------------------------
# LORA_ONLY (GPU 0)
# -----------------------------------------------------------------------------
echo ""
echo "[LORA_ONLY] Starting on GPU $LORA_ONLY_GPU..."

# Start run-api for lora_only
run-api --port $LORA_ONLY_API_PORT > "$LOG_DIR/api_lora_only.log" 2>&1 &
LORA_ONLY_API_PID=$!

# Start vLLM with --enforce-eager for lora_only
echo "[LORA_ONLY] Starting vLLM with --enable-lora --enforce-eager..."
CUDA_VISIBLE_DEVICES=$LORA_ONLY_GPU python -u example_trainer/vllm_api_server.py \
    --model "$MODEL" \
    --port $LORA_ONLY_VLLM_PORT \
    --gpu-memory-utilization 0.3 \
    --enable-lora \
    --max-lora-rank 64 \
    --enforce-eager \
    > "$LOG_DIR/vllm_lora_only.log" 2>&1 &
LORA_ONLY_VLLM_PID=$!

# -----------------------------------------------------------------------------
# LORA_RESTART (GPU 1) - Trainer manages vLLM internally
# -----------------------------------------------------------------------------
echo ""
echo "[LORA_RESTART] Starting on GPU $LORA_RESTART_GPU..."

# Pre-create checkpoint directory so vLLM can write its log there
mkdir -p "$LOG_DIR/checkpoints_lora_restart"

# Start run-api for lora_restart
run-api --port $LORA_RESTART_API_PORT > "$LOG_DIR/api_lora_restart.log" 2>&1 &
LORA_RESTART_API_PID=$!

# =============================================================================
# WAIT FOR INFRASTRUCTURE
# =============================================================================
echo ""
echo "Waiting for infrastructure to be ready..."

wait_for_api $LORA_ONLY_API_PORT "lora_only API" || exit 1
wait_for_api $LORA_RESTART_API_PORT "lora_restart API" || exit 1
wait_for_health $LORA_ONLY_VLLM_PORT "lora_only vLLM" 90 || exit 1

# =============================================================================
# START ENVIRONMENTS AND TRAINERS
# =============================================================================
echo ""
echo "Starting environments and trainers..."

# Record start time
START_TIME=$(date +%s)

# -----------------------------------------------------------------------------
# LORA_ONLY: Start environment and trainer
# -----------------------------------------------------------------------------
echo ""
echo "[LORA_ONLY] Starting GSM8k environment..."
python -u environments/gsm8k_server.py serve \
    --env.tokenizer_name "$MODEL" \
    --env.use_wandb=$USE_WANDB \
    --env.wandb_name "lora-only-env" \
    --env.rollout_server_url "http://localhost:${LORA_ONLY_API_PORT}" \
    --openai.model_name "$MODEL" \
    --openai.base_url "http://localhost:${LORA_ONLY_VLLM_PORT}/v1" \
    --openai.server_type vllm \
    --slurm false \
    2>&1 | tee "$LOG_DIR/env_lora_only.log" &
LORA_ONLY_ENV_PID=$!

echo "[LORA_ONLY] Starting trainer..."

# Build wandb args
WANDB_ARGS=""
if [ "$USE_WANDB" = "true" ]; then
    WANDB_ARGS="--use-wandb --wandb-project $WANDB_PROJECT --wandb-group lora-only"
fi

CUDA_VISIBLE_DEVICES=$LORA_ONLY_GPU python -m example_trainer.grpo \
    --model-name "$MODEL" \
    --weight-bridge-mode lora_only \
    --vllm-port $LORA_ONLY_VLLM_PORT \
    --atropos-url "http://localhost:${LORA_ONLY_API_PORT}" \
    --batch-size $BATCH_SIZE \
    --training-steps $TRAINING_STEPS \
    --lora-r 16 \
    --lora-alpha 32 \
    --vllm-restart-interval 5 \
    --save-path "$LOG_DIR/checkpoints_lora_only" \
    $WANDB_ARGS \
    --benchmark \
    2>&1 | tee "$LOG_DIR/trainer_lora_only.log" &
LORA_ONLY_TRAINER_PID=$!

# -----------------------------------------------------------------------------
# LORA_RESTART: Start trainer (it manages vLLM internally)
# -----------------------------------------------------------------------------
echo ""
echo "[LORA_RESTART] Starting trainer (manages vLLM internally)..."

# Build wandb args for lora_restart
WANDB_ARGS_RESTART=""
if [ "$USE_WANDB" = "true" ]; then
    WANDB_ARGS_RESTART="--use-wandb --wandb-project $WANDB_PROJECT --wandb-group lora-restart"
fi

CUDA_VISIBLE_DEVICES=$LORA_RESTART_GPU python -m example_trainer.grpo \
    --model-name "$MODEL" \
    --weight-bridge-mode lora_restart \
    --vllm-port $LORA_RESTART_VLLM_PORT \
    --vllm-gpu-memory-utilization 0.3 \
    --atropos-url "http://localhost:${LORA_RESTART_API_PORT}" \
    --batch-size $BATCH_SIZE \
    --training-steps $TRAINING_STEPS \
    --lora-r 16 \
    --lora-alpha 32 \
    --vllm-restart-interval 5 \
    --save-path "$LOG_DIR/checkpoints_lora_restart" \
    $WANDB_ARGS_RESTART \
    --benchmark \
    2>&1 | tee "$LOG_DIR/trainer_lora_restart.log" &
LORA_RESTART_TRAINER_PID=$!

# Wait for lora_restart's internal vLLM to start
# NOTE: Without --enforce-eager, vLLM compiles CUDA graphs which takes 1-3 minutes!
echo "[LORA_RESTART] Waiting for internal vLLM to start..."
echo "  NOTE: vLLM without --enforce-eager compiles CUDA graphs on startup (1-3 min)"
echo "  Check progress: tail -f $LOG_DIR/checkpoints_lora_restart/vllm_internal.log"
sleep 30  # Give more time for model loading before checking health
wait_for_health $LORA_RESTART_VLLM_PORT "lora_restart internal vLLM" 180 || {
    echo "  Failed - check logs:"
    echo "  Trainer log:"
    tail -30 "$LOG_DIR/trainer_lora_restart.log"
    echo ""
    echo "  vLLM internal log (if exists):"
    tail -50 "$LOG_DIR/checkpoints_lora_restart/vllm_internal.log" 2>/dev/null || echo "  (not found)"
    exit 1
}

# Start GSM8k environment for lora_restart
echo "[LORA_RESTART] Starting GSM8k environment..."
python -u environments/gsm8k_server.py serve \
    --env.tokenizer_name "$MODEL" \
    --env.use_wandb=$USE_WANDB \
    --env.wandb_name "lora-restart-env" \
    --env.rollout_server_url "http://localhost:${LORA_RESTART_API_PORT}" \
    --openai.model_name "$MODEL" \
    --openai.base_url "http://localhost:${LORA_RESTART_VLLM_PORT}/v1" \
    --openai.server_type vllm \
    --slurm false \
    2>&1 | tee "$LOG_DIR/env_lora_restart.log" &
LORA_RESTART_ENV_PID=$!

# =============================================================================
# WAIT FOR BOTH TRAINERS TO COMPLETE
# =============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Both trainers running in parallel. Waiting for completion..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 WANDB: https://wandb.ai (project: $WANDB_PROJECT)"
echo ""
echo "📋 MONITOR LOGS (in another terminal):"
echo ""
echo "  # Trainer logs (main output):"
echo "  tail -f $LOG_DIR/trainer_lora_only.log"
echo "  tail -f $LOG_DIR/trainer_lora_restart.log"
echo ""
echo "  # Environment logs (rollouts, scores):"
echo "  tail -f $LOG_DIR/env_lora_only.log"
echo "  tail -f $LOG_DIR/env_lora_restart.log"
echo ""
echo "  # vLLM logs:"
echo "  tail -f $LOG_DIR/vllm_lora_only.log"
echo "  tail -f $LOG_DIR/checkpoints_lora_restart/vllm_internal.log"
echo ""
echo "  # All logs at once:"
echo "  tail -f $LOG_DIR/*.log"
echo ""

# Wait for trainers
LORA_ONLY_EXIT=0
LORA_RESTART_EXIT=0

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
echo "COMPARISON RESULTS (Parallel Execution)"
echo "============================================================"
echo ""
echo "Training Steps: $TRAINING_STEPS"
echo "Batch Size: $BATCH_SIZE"
echo ""
echo "┌─────────────────┬──────┬──────────────┬────────────────────────────┐"
echo "│ Mode            │ GPU  │ Total Time   │ Notes                      │"
echo "├─────────────────┼──────┼──────────────┼────────────────────────────┤"
printf "│ lora_only       │  %d   │ %10ss │ --enforce-eager (~13 TPS)  │\n" "$LORA_ONLY_GPU" "$LORA_ONLY_TIME"
printf "│ lora_restart    │  %d   │ %10ss │ no --enforce-eager (~108 TPS)│\n" "$LORA_RESTART_GPU" "$LORA_RESTART_TIME"
echo "└─────────────────┴──────┴──────────────┴────────────────────────────┘"
echo ""

if [ $LORA_ONLY_TIME -gt 0 ] && [ $LORA_RESTART_TIME -gt 0 ]; then
    SPEEDUP=$(echo "scale=2; $LORA_ONLY_TIME / $LORA_RESTART_TIME" | bc)
    echo "Speedup: ${SPEEDUP}x (lora_restart vs lora_only)"
fi

echo ""
echo "📊 BENCHMARK DETAILS:"
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
echo "  $LOG_DIR/trainer_lora_only.log"
echo "  $LOG_DIR/trainer_lora_restart.log"
echo "  $LOG_DIR/vllm_lora_only.log"
echo "  $LOG_DIR/env_lora_only.log"
echo "  $LOG_DIR/env_lora_restart.log"
echo ""
