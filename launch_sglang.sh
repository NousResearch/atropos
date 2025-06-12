#!/bin/bash
# Launch SGLang server for local testing (not via SLURM)

# Default configuration
MODEL_NAME="${MODEL_NAME:-NousResearch/DeepHermes-3-Mistral-24B-Preview}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
TENSOR_PARALLEL="${TP:-8}"
LOG_FILE="${LOG_FILE:-/tmp/sglang_local.log}"

# Function to cleanup on exit
cleanup() {
    echo "Cleaning up..."
    if [ ! -z "$SGLANG_PID" ]; then
        echo "Killing SGLang server (PID: $SGLANG_PID)..."
        kill $SGLANG_PID 2>/dev/null || true
        # Also kill any child processes
        pkill -P $SGLANG_PID 2>/dev/null || true
    fi
}

# Set trap to cleanup on exit
trap cleanup EXIT INT TERM

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
            echo "  --model MODEL_NAME    Model to serve (default: NousResearch/DeepHermes-3-Mistral-24B-Preview)"
            echo "  --port PORT          Port to serve on (default: 30000)"
            echo "  --tp NUM            Tensor parallel size (default: 8)"
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
echo "SGLang Local Server Launcher"
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

# Set higher ulimit for async IO
ulimit -n 32000

# Launch SGLang server
echo "Launching SGLang server..."
source /home/maxpaperclips/sglang/.venv/bin/activate

python -m sglang.launch_server \
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

# Wait for server to be ready
echo "Waiting for SGLang server to be ready..."
MAX_RETRIES=60
for i in $(seq 1 $MAX_RETRIES); do
    if curl -s http://localhost:${SGLANG_PORT}/health > /dev/null 2>&1; then
        echo "✓ SGLang server is ready!"
        echo ""
        echo "Server is running at: http://localhost:${SGLANG_PORT}"
        echo "API endpoint: http://localhost:${SGLANG_PORT}/v1"
        echo "Log file: ${LOG_FILE}"
        echo ""
        echo "To monitor logs: tail -f ${LOG_FILE}"
        echo "To stop server: kill ${SGLANG_PID} or press Ctrl+C"
        echo ""
        
        # Keep the script running until interrupted
        echo "Press Ctrl+C to stop the server..."
        wait $SGLANG_PID
        exit 0
    fi
    
    # Check if process died
    if ! kill -0 $SGLANG_PID 2>/dev/null; then
        echo "✗ SGLang server failed to start!"
        echo "Check the log file for errors: ${LOG_FILE}"
        echo "Last 20 lines of log:"
        tail -20 ${LOG_FILE}
        exit 1
    fi
    
    echo "Waiting for SGLang... (attempt $i/$MAX_RETRIES)"
    sleep 2
done

echo "✗ SGLang server failed to become ready after ${MAX_RETRIES} attempts"
echo "Check the log file: ${LOG_FILE}"
exit 1