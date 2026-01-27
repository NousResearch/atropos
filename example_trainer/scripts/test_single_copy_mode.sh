#!/bin/bash
# =============================================================================
# Single-Copy Mode GSM8k Training Test
# =============================================================================
#
# Tests the single-copy (shared_vllm) training pipeline with GSM8k environment.
# vLLM and trainer share the SAME GPU memory - true single-copy architecture.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./scripts/test_single_copy_mode.sh [MODEL] [STEPS]
#
# Note: Single-copy mode requires tensor-parallel-size=1
#
# =============================================================================

set -e

MODEL="${1:-Qwen/Qwen2.5-3B-Instruct}"
TRAINING_STEPS="${2:-50}"
BATCH_SIZE=4

VLLM_PORT=9002
GSM8K_PORT=8002

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINER_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$TRAINER_DIR")"

LOG_DIR="${REPO_DIR}/single_copy_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Single-Copy Mode GSM8k Training Test"
echo "============================================================"
echo "Model: $MODEL"
echo "Steps: $TRAINING_STEPS"
echo "Log Dir: $LOG_DIR"
echo ""
echo "NOTE: vLLM and trainer share the SAME GPU memory!"
echo "      Weight updates are INSTANT (no copying)."
echo "============================================================"

cleanup() {
    echo "Cleaning up..."
    pkill -u $USER -f "vllm_api_server.*port.*${VLLM_PORT}" 2>/dev/null || true
    pkill -u $USER -f "gsm8k_server.*${GSM8K_PORT}" 2>/dev/null || true
    pkill -u $USER -f "grpo.py.*shared_vllm" 2>/dev/null || true
}
trap cleanup EXIT
cleanup

cd "$REPO_DIR"

echo ""
echo "[1/4] Starting vLLM with shared memory enabled..."
# NOTE: --enforce-eager is REQUIRED for single-copy mode!
# Without it, CUDA graphs freeze weights and updates won't be visible to inference.
VLLM_ENABLE_SHARED_WEIGHTS=1 \
LOGDIR="$LOG_DIR" \
python -u example_trainer/vllm_api_server.py \
    --model "$MODEL" \
    --tensor-parallel-size 1 \
    --port $VLLM_PORT \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.5 \
    --enforce-eager \
    > "${LOG_DIR}/vllm.log" 2>&1 &

echo "Waiting for vLLM (45s)..."
sleep 45

curl -s "http://localhost:${VLLM_PORT}/health" && echo " ✓ vLLM ready" || { echo " ✗ vLLM failed"; exit 1; }

# Verify IPC handles are exported
if [ -f "${LOG_DIR}/vllm_bridge_config.json" ]; then
    echo " ✓ vllm_bridge_config.json created"
    PARAM_COUNT=$(jq '.ipc_handles | keys | length' "${LOG_DIR}/vllm_bridge_config.json" 2>/dev/null || echo "0")
    echo "   Exported parameters: $PARAM_COUNT"
else
    echo " ✗ vllm_bridge_config.json not found - shared memory may not work"
fi

echo ""
echo "[2/4] Starting GSM8k environment..."
python -u environments/gsm8k_server.py serve \
    --env.tokenizer_name "$MODEL" \
    --env.use_wandb=False \
    --env.rollout_server_url "http://localhost:${GSM8K_PORT}" \
    --openai.model_name "$MODEL" \
    --openai.base_url "http://localhost:${VLLM_PORT}/v1" \
    --openai.server_type vllm \
    --slurm false \
    > "${LOG_DIR}/gsm8k.log" 2>&1 &

echo "Waiting for GSM8k (10s)..."
sleep 10

echo ""
echo "[3/4] Baseline test (before training)..."
curl -s -X POST "http://localhost:${VLLM_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL"'",
        "messages": [{"role": "user", "content": "What is 123 + 456?"}],
        "max_tokens": 100,
        "temperature": 0.1
    }' | jq '.choices[0].message.content' | tee "${LOG_DIR}/baseline_response.txt"

echo ""
echo "[4/4] Starting Single-Copy trainer..."
echo "The trainer will attach to vLLM's GPU memory via CUDA IPC."
echo ""

python -u example_trainer/grpo.py \
    --model-name "$MODEL" \
    --weight-bridge-mode shared_vllm \
    --vllm-port $VLLM_PORT \
    --atropos-url "http://localhost:${GSM8K_PORT}" \
    --batch-size $BATCH_SIZE \
    --training-steps $TRAINING_STEPS \
    --save-path "$LOG_DIR/checkpoints" \
    --vllm-config-path "${LOG_DIR}/vllm_bridge_config.json" \
    --benchmark \
    --debug-loading \
    2>&1 | tee "${LOG_DIR}/trainer.log"

echo ""
echo "============================================================"
echo "Training Complete!"
echo "============================================================"
echo "Logs: $LOG_DIR"
echo ""
echo "Key Metrics:"
grep -E "Attached|fused|Step.*Loss" "${LOG_DIR}/trainer.log" | tail -20
echo "============================================================"

# Post-training test
echo ""
echo "Post-training test (weights are already updated in vLLM):"
curl -s -X POST "http://localhost:${VLLM_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL"'",
        "messages": [{"role": "user", "content": "What is 123 + 456?"}],
        "max_tokens": 100,
        "temperature": 0.1
    }' | jq '.choices[0].message.content' | tee "${LOG_DIR}/trained_response.txt"

