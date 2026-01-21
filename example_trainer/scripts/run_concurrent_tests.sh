#!/bin/bash
# =============================================================================
# Concurrent GSM8k Training Test Script
# =============================================================================
# 
# This script runs BOTH LoRA and Single-Copy modes concurrently on an 8-GPU node:
#   - GPUs 0-1: LoRA mode (vLLM on GPU 0, trainer on GPU 1)
#   - GPUs 4-5: Single-Copy mode (vLLM+trainer share GPU 4)
#
# Usage:
#   ./scripts/run_concurrent_tests.sh [MODEL] [STEPS]
#
# Example:
#   ./scripts/run_concurrent_tests.sh Qwen/Qwen2.5-3B-Instruct 100
#
# =============================================================================

set -e

# Configuration
MODEL="${1:-Qwen/Qwen2.5-3B-Instruct}"
TRAINING_STEPS="${2:-100}"
BATCH_SIZE=4
LORA_SAVE_INTERVAL=20

# Ports (separate for each mode)
LORA_VLLM_PORT=9001
LORA_GSM8K_PORT=8001

SINGLE_COPY_VLLM_PORT=9002
SINGLE_COPY_GSM8K_PORT=8002

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINER_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$TRAINER_DIR")"

LOG_DIR="${REPO_DIR}/test_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

LORA_CHECKPOINT_DIR="${LOG_DIR}/lora_checkpoints"
SINGLE_COPY_CHECKPOINT_DIR="${LOG_DIR}/single_copy_checkpoints"
mkdir -p "$LORA_CHECKPOINT_DIR" "$SINGLE_COPY_CHECKPOINT_DIR"

echo "============================================================"
echo "Concurrent GSM8k Training Test"
echo "============================================================"
echo "Model: $MODEL"
echo "Training Steps: $TRAINING_STEPS"
echo "Batch Size: $BATCH_SIZE"
echo "Log Directory: $LOG_DIR"
echo ""
echo "LoRA Mode:        GPUs 0-1, ports ${LORA_VLLM_PORT}/${LORA_GSM8K_PORT}"
echo "Single-Copy Mode: GPU 4,   ports ${SINGLE_COPY_VLLM_PORT}/${SINGLE_COPY_GSM8K_PORT}"
echo "============================================================"

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up processes..."
    pkill -u $USER -f "vllm_api_server.*port.${LORA_VLLM_PORT}" 2>/dev/null || true
    pkill -u $USER -f "vllm_api_server.*port.${SINGLE_COPY_VLLM_PORT}" 2>/dev/null || true
    pkill -u $USER -f "gsm8k_server.*${LORA_GSM8K_PORT}" 2>/dev/null || true
    pkill -u $USER -f "gsm8k_server.*${SINGLE_COPY_GSM8K_PORT}" 2>/dev/null || true
    pkill -u $USER -f "grpo.py.*lora_only" 2>/dev/null || true
    pkill -u $USER -f "grpo.py.*shared_vllm" 2>/dev/null || true
    echo "Cleanup complete."
}

trap cleanup EXIT

# Kill any existing processes
cleanup

# Clear Triton cache (for LoRA B200 compatibility)
rm -rf ~/.triton/cache

cd "$REPO_DIR"

echo ""
echo "[1/6] Starting LoRA vLLM server (GPUs 0)..."
CUDA_VISIBLE_DEVICES=0 \
VLLM_ENABLE_SHARED_WEIGHTS=1 \
python -u example_trainer/vllm_api_server.py \
    --model "$MODEL" \
    --tensor-parallel-size 1 \
    --port $LORA_VLLM_PORT \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.7 \
    --enable-lora \
    --max-loras 2 \
    --max-lora-rank 64 \
    --enforce-eager \
    > "${LOG_DIR}/lora_vllm.log" 2>&1 &
LORA_VLLM_PID=$!
echo "  PID: $LORA_VLLM_PID"

echo ""
echo "[2/6] Starting Single-Copy vLLM server (GPU 4)..."
CUDA_VISIBLE_DEVICES=4 \
VLLM_ENABLE_SHARED_WEIGHTS=1 \
LOGDIR="$SINGLE_COPY_CHECKPOINT_DIR" \
python -u example_trainer/vllm_api_server.py \
    --model "$MODEL" \
    --tensor-parallel-size 1 \
    --port $SINGLE_COPY_VLLM_PORT \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.5 \
    > "${LOG_DIR}/single_copy_vllm.log" 2>&1 &
SINGLE_COPY_VLLM_PID=$!
echo "  PID: $SINGLE_COPY_VLLM_PID"

echo ""
echo "Waiting for vLLM servers to initialize (60s)..."
sleep 60

# Verify servers are running
echo ""
echo "Verifying vLLM servers..."

if curl -s "http://localhost:${LORA_VLLM_PORT}/health" > /dev/null; then
    echo "  ✓ LoRA vLLM server healthy"
else
    echo "  ✗ LoRA vLLM server failed to start"
    cat "${LOG_DIR}/lora_vllm.log" | tail -50
    exit 1
fi

if curl -s "http://localhost:${SINGLE_COPY_VLLM_PORT}/health" > /dev/null; then
    echo "  ✓ Single-Copy vLLM server healthy"
else
    echo "  ✗ Single-Copy vLLM server failed to start"
    cat "${LOG_DIR}/single_copy_vllm.log" | tail -50
    exit 1
fi

echo ""
echo "[3/6] Starting LoRA GSM8k environment..."
python -u environments/gsm8k_server.py serve \
    --env.tokenizer_name "$MODEL" \
    --env.use_wandb=False \
    --openai.model_name "$MODEL" \
    --openai.base_url "http://localhost:${LORA_VLLM_PORT}/v1" \
    --openai.server_type vllm \
    --server.port $LORA_GSM8K_PORT \
    > "${LOG_DIR}/lora_gsm8k.log" 2>&1 &
LORA_GSM8K_PID=$!
echo "  PID: $LORA_GSM8K_PID"

echo ""
echo "[4/6] Starting Single-Copy GSM8k environment..."
python -u environments/gsm8k_server.py serve \
    --env.tokenizer_name "$MODEL" \
    --env.use_wandb=False \
    --openai.model_name "$MODEL" \
    --openai.base_url "http://localhost:${SINGLE_COPY_VLLM_PORT}/v1" \
    --openai.server_type vllm \
    --server.port $SINGLE_COPY_GSM8K_PORT \
    > "${LOG_DIR}/single_copy_gsm8k.log" 2>&1 &
SINGLE_COPY_GSM8K_PID=$!
echo "  PID: $SINGLE_COPY_GSM8K_PID"

echo ""
echo "Waiting for GSM8k environments to initialize (15s)..."
sleep 15

echo ""
echo "[5/6] Starting LoRA trainer (GPU 1)..."
CUDA_VISIBLE_DEVICES=1 \
python -u example_trainer/grpo.py \
    --model-name "$MODEL" \
    --weight-bridge-mode lora_only \
    --vllm-port $LORA_VLLM_PORT \
    --atropos-url "http://localhost:${LORA_GSM8K_PORT}" \
    --batch-size $BATCH_SIZE \
    --training-steps $TRAINING_STEPS \
    --vllm-restart-interval $LORA_SAVE_INTERVAL \
    --save-path "$LORA_CHECKPOINT_DIR" \
    --benchmark \
    > "${LOG_DIR}/lora_trainer.log" 2>&1 &
LORA_TRAINER_PID=$!
echo "  PID: $LORA_TRAINER_PID"

echo ""
echo "[6/6] Starting Single-Copy trainer (GPU 4 - shared with vLLM)..."
CUDA_VISIBLE_DEVICES=4 \
python -u example_trainer/grpo.py \
    --model-name "$MODEL" \
    --weight-bridge-mode shared_vllm \
    --vllm-port $SINGLE_COPY_VLLM_PORT \
    --atropos-url "http://localhost:${SINGLE_COPY_GSM8K_PORT}" \
    --batch-size $BATCH_SIZE \
    --training-steps $TRAINING_STEPS \
    --save-path "$SINGLE_COPY_CHECKPOINT_DIR" \
    --vllm-config-path "${SINGLE_COPY_CHECKPOINT_DIR}/vllm_bridge_config.json" \
    --benchmark \
    > "${LOG_DIR}/single_copy_trainer.log" 2>&1 &
SINGLE_COPY_TRAINER_PID=$!
echo "  PID: $SINGLE_COPY_TRAINER_PID"

echo ""
echo "============================================================"
echo "Both trainers started!"
echo ""
echo "Monitor logs:"
echo "  tail -f ${LOG_DIR}/lora_trainer.log"
echo "  tail -f ${LOG_DIR}/single_copy_trainer.log"
echo ""
echo "Or watch both:"
echo "  tail -f ${LOG_DIR}/*.log"
echo ""
echo "Waiting for training to complete..."
echo "============================================================"

# Wait for both trainers to complete
wait $LORA_TRAINER_PID
LORA_EXIT=$?

wait $SINGLE_COPY_TRAINER_PID
SINGLE_COPY_EXIT=$?

echo ""
echo "============================================================"
echo "TRAINING COMPLETE"
echo "============================================================"
echo "LoRA Trainer Exit Code: $LORA_EXIT"
echo "Single-Copy Trainer Exit Code: $SINGLE_COPY_EXIT"
echo ""
echo "Results saved to: $LOG_DIR"
echo ""
echo "Checkpoints:"
echo "  LoRA: $LORA_CHECKPOINT_DIR"
echo "  Single-Copy: $SINGLE_COPY_CHECKPOINT_DIR"
echo "============================================================"

# Generate summary
echo ""
echo "=== LoRA Training Summary ===" | tee "${LOG_DIR}/summary.txt"
grep -E "Step|Loss|Accuracy" "${LOG_DIR}/lora_trainer.log" | tail -20 | tee -a "${LOG_DIR}/summary.txt"

echo "" | tee -a "${LOG_DIR}/summary.txt"
echo "=== Single-Copy Training Summary ===" | tee -a "${LOG_DIR}/summary.txt"
grep -E "Step|Loss|Accuracy" "${LOG_DIR}/single_copy_trainer.log" | tail -20 | tee -a "${LOG_DIR}/summary.txt"

