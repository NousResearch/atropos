#!/bin/bash
# Quick test script for lora_restart mode
# Tests that the mode works and compares timing

set -e

MODEL="Qwen/Qwen3-4B-Instruct-2507"
STEPS=10
RESTART_INTERVAL=3

echo "=============================================="
echo "Testing lora_restart mode"
echo "=============================================="
echo "Model: $MODEL"
echo "Steps: $STEPS"
echo "Restart interval: $RESTART_INTERVAL"
echo "=============================================="

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Create log directory
LOGDIR="./lora_restart_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
echo "Logs: $LOGDIR"

# Cleanup function
cleanup() {
    echo "Cleaning up..."
    pkill -f "gsm8k_server.py" 2>/dev/null || true
    pkill -f "run-api" 2>/dev/null || true
    pkill -f "vllm_api_server.py" 2>/dev/null || true
    # Kill by port
    for port in 8000 9001; do
        fuser -k ${port}/tcp 2>/dev/null || true
    done
}
trap cleanup EXIT

# Kill any existing processes
cleanup
sleep 2

# Start API server
echo ""
echo "[1/3] Starting Atropos API..."
run-api --port 8000 > "$LOGDIR/api.log" 2>&1 &
API_PID=$!
sleep 3

# Check API is up
if ! curl -s "http://localhost:8000/info" > /dev/null 2>&1; then
    echo "ERROR: API server failed to start"
    cat "$LOGDIR/api.log"
    exit 1
fi
echo "  ✓ API running (PID: $API_PID)"

# Start trainer (lora_restart manages vLLM internally)
echo ""
echo "[2/3] Starting lora_restart trainer..."
echo "  (This will launch vLLM internally)"

START_TIME=$(date +%s)

CUDA_VISIBLE_DEVICES=0 python -m example_trainer.grpo \
    --model-name "$MODEL" \
    --weight-bridge-mode lora_restart \
    --vllm-port 9001 \
    --atropos-url http://localhost:8000 \
    --lora-r 16 \
    --lora-alpha 32 \
    --training-steps $STEPS \
    --vllm-restart-interval $RESTART_INTERVAL \
    --save-path "$LOGDIR/checkpoints" \
    --benchmark \
    > "$LOGDIR/trainer.log" 2>&1 &
TRAINER_PID=$!

# Wait for vLLM to start (trainer launches it)
echo "  Waiting for trainer to launch vLLM..."
sleep 30

# Start environment (needs to wait for vLLM)
echo ""
echo "[3/3] Starting GSM8K environment..."
python -u environments/gsm8k_server.py serve \
    --env.tokenizer_name "$MODEL" \
    --env.rollout_server_url "http://localhost:8000" \
    --env.max_token_length 2048 \
    --env.use_wandb=False \
    --openai.model_name "$MODEL" \
    --openai.base_url "http://localhost:9001/v1" \
    --openai.server_type vllm \
    --slurm false \
    > "$LOGDIR/env.log" 2>&1 &
ENV_PID=$!
sleep 5
echo "  ✓ Environment running (PID: $ENV_PID)"

# Wait for trainer to complete
echo ""
echo "Waiting for training to complete..."
echo "(Check progress: tail -f $LOGDIR/trainer.log)"

wait $TRAINER_PID
TRAINER_EXIT=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=============================================="
echo "TEST RESULTS"
echo "=============================================="

if [ $TRAINER_EXIT -eq 0 ]; then
    echo "✓ Training completed successfully!"
    echo "  Time: ${ELAPSED}s"
    echo ""
    echo "Checkpoints:"
    ls -la "$LOGDIR/checkpoints/" 2>/dev/null || echo "  (no checkpoints found)"
    echo ""
    echo "Benchmark summary:"
    grep -A 20 "BENCHMARK SUMMARY" "$LOGDIR/trainer.log" 2>/dev/null || echo "  (no benchmark found)"
else
    echo "✗ Training FAILED (exit code: $TRAINER_EXIT)"
    echo ""
    echo "Last 50 lines of trainer log:"
    tail -50 "$LOGDIR/trainer.log"
fi

echo ""
echo "=============================================="
echo "Log files saved to: $LOGDIR"
echo "=============================================="
