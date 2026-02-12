#!/bin/bash
# Quick test for lora_restart mode - just 10 steps with 2 restarts
set -e

MODEL="${1:-Qwen/Qwen3-4B-Instruct-2507}"
STEPS="${2:-10}"
GPU="${3:-0}"
PORT_API=8099
PORT_VLLM=9099
MAX_LEN="${MAX_LEN:-8192}"  # Use 8k for quick test, set MAX_LEN=32768 for full test

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="./lora_restart_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "LORA_RESTART Quick Test"
echo "=============================================="
echo "Model: $MODEL"
echo "Steps: $STEPS"
echo "GPU: $GPU"
echo "Max Length: $MAX_LEN"
echo "Log dir: $LOG_DIR"
echo "=============================================="

# Cleanup
cleanup() {
    echo "Cleaning up..."
    pkill -9 -f "port $PORT_VLLM" 2>/dev/null || true
    pkill -9 -f "port $PORT_API" 2>/dev/null || true
    fuser -k ${PORT_API}/tcp 2>/dev/null || true
    fuser -k ${PORT_VLLM}/tcp 2>/dev/null || true
}
trap cleanup EXIT

# Kill any existing processes
cleanup
sleep 2

# Start API server
echo ""
echo "[1/3] Starting API server on port $PORT_API..."
run-api --port $PORT_API > "$LOG_DIR/api.log" 2>&1 &
API_PID=$!
sleep 3

# Check API is up
if ! curl -s "http://localhost:$PORT_API/info" > /dev/null; then
    echo "ERROR: API server failed to start"
    cat "$LOG_DIR/api.log"
    exit 1
fi
echo "  ✓ API server ready"

# Start environment (GSM8K for simplicity)
echo ""
echo "[2/3] Starting GSM8K environment..."
python -u environments/gsm8k_server.py serve \
    --env.tokenizer_name "$MODEL" \
    --env.use_wandb=False \
    --env.rollout_server_url "http://localhost:$PORT_API" \
    --openai.model_name "$MODEL" \
    --openai.base_url "http://localhost:$PORT_VLLM/v1" \
    --openai.server_type vllm \
    --slurm false \
    > "$LOG_DIR/env.log" 2>&1 &
ENV_PID=$!
echo "  ✓ Environment started (PID: $ENV_PID)"
sleep 5

# Start trainer
echo ""
echo "[3/3] Starting LORA_RESTART trainer..."
echo "  (This will launch vLLM internally and restart every 5 steps)"
echo ""

CUDA_VISIBLE_DEVICES=$GPU python -m example_trainer.grpo \
    --model-name "$MODEL" \
    --weight-bridge-mode lora_restart \
    --vllm-port $PORT_VLLM \
    --vllm-gpu-memory-utilization 0.20 \
    --atropos-url "http://localhost:$PORT_API" \
    --batch-size 2 \
    --training-steps $STEPS \
    --max-model-len $MAX_LEN \
    --seq-len $MAX_LEN \
    --lora-r 16 \
    --lora-alpha 32 \
    --vllm-restart-interval 5 \
    --save-path "$LOG_DIR/checkpoints" \
    --benchmark \
    2>&1 | tee "$LOG_DIR/trainer.log"

echo ""
echo "=============================================="
echo "Test complete! Logs in: $LOG_DIR"
echo "=============================================="
