#!/bin/bash
# Launch SGLang server with nohup for non-blocking execution

# Default configuration
MODEL_NAME="${MODEL_NAME:-NousResearch/DeepHermes-3-Llama-3-8B-Preview}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
TENSOR_PARALLEL="${TP:-4}"
LOG_FILE="${LOG_FILE:-/tmp/sglang_local.log}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --port)
            SGLANG_PORT="$2"
            shift 2
            ;;
        --tp)
            TENSOR_PARALLEL="$2"
            shift 2
            ;;
        --log)
            LOG_FILE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --model MODEL_NAME    Model to serve (default: NousResearch/DeepHermes-3-Llama-3-8B-Preview)"
            echo "  --port PORT          Port to serve on (default: 30000)"
            echo "  --tp NUM            Tensor parallel size (default: 4)"
            echo "  --log FILE          Log file path (default: /tmp/sglang_local.log)"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "SGLang Non-Blocking Server Launcher"
echo "========================================"
echo "Model: ${MODEL_NAME}"
echo "Port: ${SGLANG_PORT}"
echo "Tensor Parallel: ${TENSOR_PARALLEL}"
echo "Log file: ${LOG_FILE}"
echo "========================================"

# Check if SGLang is already running on the port
if lsof -Pi :${SGLANG_PORT} -sTCP:LISTEN -t >/dev/null ; then
    echo "ERROR: Port ${SGLANG_PORT} is already in use!"
    echo "Please stop the existing server or use a different port with --port"
    exit 1
fi

# Kill any existing SGLang process on this port
pkill -f "sglang.launch_server.*--port ${SGLANG_PORT}" 2>/dev/null || true

# Set higher ulimit for async IO
ulimit -n 32000

# Launch SGLang server with nohup
echo "Launching SGLang server in background..."
source /home/maxpaperclips/sglang/.venv/bin/activate

nohup python -m sglang.launch_server \
    --model-path ${MODEL_NAME} \
    --host 0.0.0.0 \
    --port ${SGLANG_PORT} \
    --tp ${TENSOR_PARALLEL} \
    --disable-outlines-disk-cache \
    --grammar-backend xgrammar \
    > ${LOG_FILE} 2>&1 &

SGLANG_PID=$!
echo "SGLang server launched with PID ${SGLANG_PID}"

# Deactivate virtual environment
deactivate

# Quick check (non-blocking) - just wait a few seconds
echo "Waiting a moment for server startup..."
sleep 5

# Save PID to file for easy stopping later
echo ${SGLANG_PID} > /tmp/sglang_${SGLANG_PORT}.pid

echo ""
echo "Server launched! Check status with:"
echo "  curl http://localhost:${SGLANG_PORT}/health"
echo ""
echo "Monitor logs with:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "Stop server with:"
echo "  kill ${SGLANG_PID}"
echo "  # or: kill \$(cat /tmp/sglang_${SGLANG_PORT}.pid)"
echo ""