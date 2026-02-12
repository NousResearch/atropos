#!/bin/bash
# ============================================================================
# Compare lora_restart vs lora_only performance
# ============================================================================
# Runs both modes in parallel with separate APIs/environments/ports
# All commands run in background (single terminal)
# Results uploaded to W&B
#
# Usage:
#   ./compare_lora_modes.sh [steps]
#   ./compare_lora_modes.sh 30      # 30 steps (default)
#   ./compare_lora_modes.sh 10      # Quick 10-step test
# ============================================================================

set -e

# Configuration
MODEL="Qwen/Qwen3-4B-Instruct-2507"
STEPS="${1:-30}"
RESTART_INTERVAL=3
WANDB_PROJECT="lora-mode-comparison"

# Port allocation
# lora_restart: API 8001, vLLM 9001
# lora_only:    API 8002, vLLM 9002

echo "============================================================================"
echo "LoRA Mode Comparison: lora_restart vs lora_only"
echo "============================================================================"
echo "Model:            $MODEL"
echo "Steps:            $STEPS"
echo "Restart interval: $RESTART_INTERVAL"
echo "W&B project:      $WANDB_PROJECT"
echo ""
echo "Port allocation:"
echo "  lora_restart: API=8001, vLLM=9001, GPU=0"
echo "  lora_only:    API=8002, vLLM=9002, GPU=1"
echo "============================================================================"

# Get script directory and repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
echo "Working directory: $(pwd)"

# Create log directory
LOGDIR="./lora_comparison_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
echo "Log directory: $LOGDIR"

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up all processes..."
    
    # Kill by name
    pkill -f "gsm8k_server.py" 2>/dev/null || true
    pkill -f "run-api" 2>/dev/null || true
    pkill -f "vllm_api_server.py" 2>/dev/null || true
    pkill -f "example_trainer.grpo" 2>/dev/null || true
    
    # Kill by port
    for port in 8001 8002 9001 9002; do
        fuser -k ${port}/tcp 2>/dev/null || true
    done
    
    echo "Cleanup complete."
}
trap cleanup EXIT

# Initial cleanup
echo ""
echo "Killing any existing processes on ports 8001, 8002, 9001, 9002..."
cleanup
sleep 3

# ============================================================================
# MODE 1: lora_restart (GPU 0, ports 8001/9001)
# ============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[1/2] LORA_RESTART MODE (GPU 0, API:8001, vLLM:9001)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Start API for lora_restart
echo "  Starting API server (port 8001)..."
run-api --port 8001 > "$LOGDIR/api_restart.log" 2>&1 &
RESTART_API_PID=$!
sleep 3

# Check API is up
if curl -s "http://localhost:8001/info" > /dev/null 2>&1; then
    echo "  ✓ API running (PID: $RESTART_API_PID)"
else
    echo "  ✗ API failed to start"
    cat "$LOGDIR/api_restart.log"
    exit 1
fi

# Start trainer (lora_restart manages vLLM internally)
echo "  Starting lora_restart trainer (will launch vLLM on port 9001)..."
CUDA_VISIBLE_DEVICES=0 python -m example_trainer.grpo \
    --model-name "$MODEL" \
    --weight-bridge-mode lora_restart \
    --vllm-port 9001 \
    --atropos-url http://localhost:8001 \
    --lora-r 16 \
    --lora-alpha 32 \
    --training-steps $STEPS \
    --vllm-restart-interval $RESTART_INTERVAL \
    --save-path "$LOGDIR/checkpoints_restart" \
    --use-wandb \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-group "comparison-$(date +%Y%m%d)" \
    --benchmark \
    > "$LOGDIR/trainer_restart.log" 2>&1 &
RESTART_TRAINER_PID=$!
echo "  ✓ Trainer started (PID: $RESTART_TRAINER_PID)"

# Wait for vLLM to be ready (trainer launches it)
echo "  Waiting for vLLM to start (port 9001)..."
for i in {1..60}; do
    if curl -s "http://localhost:9001/health" > /dev/null 2>&1; then
        echo "  ✓ vLLM ready after ~${i}s"
        break
    fi
    sleep 2
done

# Start environment for lora_restart
echo "  Starting environment..."
python -u environments/gsm8k_server.py serve \
    --env.tokenizer_name "$MODEL" \
    --env.rollout_server_url "http://localhost:8001" \
    --env.max_token_length 2048 \
    --env.use_wandb=True \
    --env.wandb_name "lora-restart-env" \
    --openai.model_name "$MODEL" \
    --openai.base_url "http://localhost:9001/v1" \
    --openai.server_type vllm \
    --slurm false \
    > "$LOGDIR/env_restart.log" 2>&1 &
RESTART_ENV_PID=$!
echo "  ✓ Environment started (PID: $RESTART_ENV_PID)"

# ============================================================================
# MODE 2: lora_only (GPU 1, ports 8002/9002)
# ============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[2/2] LORA_ONLY MODE (GPU 1, API:8002, vLLM:9002)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Start API for lora_only
echo "  Starting API server (port 8002)..."
run-api --port 8002 > "$LOGDIR/api_only.log" 2>&1 &
ONLY_API_PID=$!
sleep 3

# Check API is up
if curl -s "http://localhost:8002/info" > /dev/null 2>&1; then
    echo "  ✓ API running (PID: $ONLY_API_PID)"
else
    echo "  ✗ API failed to start"
    cat "$LOGDIR/api_only.log"
    exit 1
fi

# Start vLLM for lora_only (external, with --enforce-eager)
echo "  Starting vLLM with --enable-lora --enforce-eager (port 9002)..."
CUDA_VISIBLE_DEVICES=1 python example_trainer/vllm_api_server.py \
    --model "$MODEL" \
    --port 9002 \
    --gpu-memory-utilization 0.45 \
    --enable-lora \
    --max-lora-rank 32 \
    --enforce-eager \
    > "$LOGDIR/vllm_only.log" 2>&1 &
ONLY_VLLM_PID=$!
echo "  ✓ vLLM started (PID: $ONLY_VLLM_PID)"

# Wait for vLLM to be ready
echo "  Waiting for vLLM to start (port 9002)..."
for i in {1..90}; do
    if curl -s "http://localhost:9002/health" > /dev/null 2>&1; then
        echo "  ✓ vLLM ready after ~${i}s"
        break
    fi
    sleep 2
done

# Start environment for lora_only
echo "  Starting environment..."
python -u environments/gsm8k_server.py serve \
    --env.tokenizer_name "$MODEL" \
    --env.rollout_server_url "http://localhost:8002" \
    --env.max_token_length 2048 \
    --env.use_wandb=True \
    --env.wandb_name "lora-only-env" \
    --openai.model_name "$MODEL" \
    --openai.base_url "http://localhost:9002/v1" \
    --openai.server_type vllm \
    --slurm false \
    > "$LOGDIR/env_only.log" 2>&1 &
ONLY_ENV_PID=$!
echo "  ✓ Environment started (PID: $ONLY_ENV_PID)"

# Start trainer for lora_only
echo "  Starting lora_only trainer..."
CUDA_VISIBLE_DEVICES=1 python -m example_trainer.grpo \
    --model-name "$MODEL" \
    --weight-bridge-mode lora_only \
    --vllm-port 9002 \
    --atropos-url http://localhost:8002 \
    --lora-r 16 \
    --lora-alpha 32 \
    --training-steps $STEPS \
    --save-path "$LOGDIR/checkpoints_only" \
    --use-wandb \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-group "comparison-$(date +%Y%m%d)" \
    --benchmark \
    > "$LOGDIR/trainer_only.log" 2>&1 &
ONLY_TRAINER_PID=$!
echo "  ✓ Trainer started (PID: $ONLY_TRAINER_PID)"

# ============================================================================
# Save PIDs and monitor
# ============================================================================
cat > "$LOGDIR/pids.txt" << EOF
RESTART_API_PID=$RESTART_API_PID
RESTART_TRAINER_PID=$RESTART_TRAINER_PID
RESTART_ENV_PID=$RESTART_ENV_PID
ONLY_API_PID=$ONLY_API_PID
ONLY_VLLM_PID=$ONLY_VLLM_PID
ONLY_ENV_PID=$ONLY_ENV_PID
ONLY_TRAINER_PID=$ONLY_TRAINER_PID
EOF

echo ""
echo "============================================================================"
echo "All components started!"
echo "============================================================================"
echo ""
echo "📊 Monitor progress:"
echo "  tail -f $LOGDIR/trainer_restart.log   # lora_restart"
echo "  tail -f $LOGDIR/trainer_only.log      # lora_only"
echo ""
echo "🔍 Watch both:"
echo "  tail -f $LOGDIR/trainer_*.log"
echo ""
echo "📈 W&B Dashboard:"
echo "  https://wandb.ai/$WANDB_PROJECT"
echo ""
echo "Waiting for trainers to complete..."
echo "(lora_restart should finish MUCH faster than lora_only)"
echo ""

# Wait for trainers
RESTART_STATUS="running"
ONLY_STATUS="running"

while [ "$RESTART_STATUS" = "running" ] || [ "$ONLY_STATUS" = "running" ]; do
    sleep 30
    
    # Check lora_restart
    if [ "$RESTART_STATUS" = "running" ]; then
        if ! kill -0 $RESTART_TRAINER_PID 2>/dev/null; then
            wait $RESTART_TRAINER_PID 2>/dev/null && RESTART_STATUS="completed" || RESTART_STATUS="failed"
            echo "  lora_restart: $RESTART_STATUS"
        fi
    fi
    
    # Check lora_only
    if [ "$ONLY_STATUS" = "running" ]; then
        if ! kill -0 $ONLY_TRAINER_PID 2>/dev/null; then
            wait $ONLY_TRAINER_PID 2>/dev/null && ONLY_STATUS="completed" || ONLY_STATUS="failed"
            echo "  lora_only: $ONLY_STATUS"
        fi
    fi
    
    # Show status
    if [ "$RESTART_STATUS" = "running" ] || [ "$ONLY_STATUS" = "running" ]; then
        echo "  [$(date +%H:%M:%S)] lora_restart: $RESTART_STATUS, lora_only: $ONLY_STATUS"
    fi
done

# ============================================================================
# Print results
# ============================================================================
echo ""
echo "============================================================================"
echo "COMPARISON RESULTS"
echo "============================================================================"

echo ""
echo "📊 LORA_RESTART (CUDA graphs, vLLM restarts):"
echo "─────────────────────────────────────────────────"
grep -A 20 "BENCHMARK SUMMARY" "$LOGDIR/trainer_restart.log" 2>/dev/null || echo "  (check $LOGDIR/trainer_restart.log)"

echo ""
echo "📊 LORA_ONLY (--enforce-eager, hot-swap):"
echo "─────────────────────────────────────────────────"
grep -A 20 "BENCHMARK SUMMARY" "$LOGDIR/trainer_only.log" 2>/dev/null || echo "  (check $LOGDIR/trainer_only.log)"

echo ""
echo "============================================================================"
echo "📁 LOGS SAVED TO: $LOGDIR"
echo "============================================================================"
echo ""
echo "Log files:"
echo "  $LOGDIR/trainer_restart.log  # lora_restart trainer"
echo "  $LOGDIR/trainer_only.log     # lora_only trainer"
echo "  $LOGDIR/vllm_only.log        # lora_only vLLM"
echo "  $LOGDIR/env_restart.log      # lora_restart environment"
echo "  $LOGDIR/env_only.log         # lora_only environment"
echo ""
echo "Checkpoints:"
echo "  $LOGDIR/checkpoints_restart/"
echo "  $LOGDIR/checkpoints_only/"
echo ""
echo "W&B runs should be visible at:"
echo "  https://wandb.ai/$WANDB_PROJECT"
echo ""
echo "============================================================================"
echo "Done!"
