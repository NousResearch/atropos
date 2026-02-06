#!/bin/bash
# =============================================================================
# LoRA Mode GSM8k Training Test
# =============================================================================
#
# Tests the LoRA training pipeline with GSM8k environment.
# Uses separate GPUs for vLLM and trainer.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1 ./scripts/test_lora_mode.sh [MODEL] [STEPS]
#
# =============================================================================

set -e

MODEL="${1:-Qwen/Qwen2.5-3B-Instruct}"
TRAINING_STEPS="${2:-50}"
BATCH_SIZE=4
SAVE_INTERVAL=10

VLLM_PORT=9001
GSM8K_PORT=8001

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINER_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$TRAINER_DIR")"

LOG_DIR="${REPO_DIR}/lora_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "LoRA Mode GSM8k Training Test"
echo "============================================================"
echo "Model: $MODEL"
echo "Steps: $TRAINING_STEPS"
echo "Log Dir: $LOG_DIR"
echo "============================================================"

cleanup() {
    echo "Cleaning up..."
    pkill -u $USER -f "vllm_api_server.*port.*${VLLM_PORT}" 2>/dev/null || true
    pkill -u $USER -f "gsm8k_server" 2>/dev/null || true
    pkill -u $USER -f "grpo.py" 2>/dev/null || true
}
trap cleanup EXIT
cleanup

# Clear Triton cache for B200 compatibility
rm -rf ~/.triton/cache

cd "$REPO_DIR"

echo ""
echo "[1/4] Starting vLLM with LoRA support..."
VLLM_ENABLE_SHARED_WEIGHTS=1 \
python -u example_trainer/vllm_api_server.py \
    --model "$MODEL" \
    --tensor-parallel-size 1 \
    --port $VLLM_PORT \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.6 \
    --enable-lora \
    --max-loras 2 \
    --max-lora-rank 64 \
    --enforce-eager \
    > "${LOG_DIR}/vllm.log" 2>&1 &

echo "Waiting for vLLM (45s)..."
sleep 45

curl -s "http://localhost:${VLLM_PORT}/health" && echo " ✓ vLLM ready" || { echo " ✗ vLLM failed"; exit 1; }

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
curl -s -X POST "http://localhost:${VLLM_PORT}/generate" \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "<|im_start|>user\nWhat is 123 + 456?<|im_end|>\n<|im_start|>assistant\n",
        "max_tokens": 100,
        "temperature": 0.1
    }' | jq '.text[0]' | tee "${LOG_DIR}/baseline_response.txt"

echo ""
echo "[4/4] Starting LoRA trainer..."
python -u example_trainer/grpo.py \
    --model-name "$MODEL" \
    --weight-bridge-mode lora_only \
    --vllm-port $VLLM_PORT \
    --atropos-url "http://localhost:${GSM8K_PORT}" \
    --batch-size $BATCH_SIZE \
    --training-steps $TRAINING_STEPS \
    --vllm-restart-interval $SAVE_INTERVAL \
    --save-path "$LOG_DIR/checkpoints" \
    --benchmark \
    2>&1 | tee "${LOG_DIR}/trainer.log"

echo ""
echo "============================================================"
echo "Training Complete!"
echo "Logs: $LOG_DIR"
echo "Checkpoints: $LOG_DIR/checkpoints"
echo "============================================================"

# Post-training test
if [ -d "$LOG_DIR/checkpoints" ]; then
    LATEST_ADAPTER=$(ls -td "$LOG_DIR/checkpoints/adapter_"* 2>/dev/null | head -1)
    if [ -n "$LATEST_ADAPTER" ]; then
        echo ""
        echo "Post-training test with adapter: $LATEST_ADAPTER"

        curl -s -X POST "http://localhost:${VLLM_PORT}/lora/load" \
            -H "Content-Type: application/json" \
            -d '{"adapter_path": "'"$LATEST_ADAPTER"'"}' | jq

        echo ""
        echo "Response after training:"
        curl -s -X POST "http://localhost:${VLLM_PORT}/generate" \
            -H "Content-Type: application/json" \
            -d '{
                "prompt": "<|im_start|>user\nWhat is 123 + 456?<|im_end|>\n<|im_start|>assistant\n",
                "max_tokens": 100,
                "temperature": 0.1
            }' | jq '.text[0]' | tee "${LOG_DIR}/trained_response.txt"
    fi
fi
