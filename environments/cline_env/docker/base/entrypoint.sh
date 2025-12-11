#!/bin/bash
# Entrypoint for Cline CLI Docker container

set -e

echo "[cline] Starting Cline container"
echo "[cline] Cline version: $(cline version 2>/dev/null | head -3 || echo 'checking...')"

# Configure API provider based on environment variables
if [ -n "$ANTHROPIC_API_KEY" ]; then
    echo "[cline] Configuring Anthropic API..."
    MODEL="${CLINE_MODEL:-claude-sonnet-4-5-20250929}"
    cline auth -p anthropic -k "$ANTHROPIC_API_KEY" -m "$MODEL" --output-format plain 2>/dev/null || true
    echo "[cline] Configured with model: $MODEL"
elif [ -n "$OPENAI_API_KEY" ] && [ -n "$OPENAI_BASE_URL" ]; then
    # For custom model endpoints (vLLM, etc.) during RL training
    echo "[cline] Configuring OpenAI-compatible API..."
    MODEL="${CLINE_MODEL:-gpt-4}"
    cline auth -p openai-compatible -k "$OPENAI_API_KEY" -m "$MODEL" -b "$OPENAI_BASE_URL" --output-format plain 2>/dev/null || true
    echo "[cline] Configured with model: $MODEL at $OPENAI_BASE_URL"
elif [ -n "$OPENAI_API_KEY" ]; then
    echo "[cline] Configuring OpenAI API..."
    MODEL="${CLINE_MODEL:-gpt-4o}"
    cline auth -p openai-native -k "$OPENAI_API_KEY" -m "$MODEL" --output-format plain 2>/dev/null || true
    echo "[cline] Configured with model: $MODEL"
fi

# If CLINE_TASK is set, run the task in YOLO mode
if [ -n "$CLINE_TASK" ]; then
    echo "[cline] Running task: $CLINE_TASK"
    cd /workspace
    
    # Run in YOLO + oneshot mode (fully autonomous)
    # -y = yolo mode (auto-approve)
    # -o = oneshot mode (full autonomy)
    cline "$CLINE_TASK" -y -o --output-format plain
    
    # Find the latest task directory and output the trajectory path
    LATEST_TASK=$(ls -td ~/.cline/data/tasks/*/ 2>/dev/null | head -1)
    if [ -n "$LATEST_TASK" ]; then
        echo "[cline] Trajectory saved to: $LATEST_TASK"
        echo "[cline] Files in trajectory:"
        ls -la "$LATEST_TASK"
        
        # Copy trajectory to /output if mounted
        if [ -d "/output" ]; then
            TASK_ID=$(basename "$LATEST_TASK")
            cp -r "$LATEST_TASK" "/output/$TASK_ID"
            echo "[cline] Trajectory copied to /output/$TASK_ID"
        fi
    fi
else
    # No task specified, run whatever command was passed
    exec "$@"
fi
