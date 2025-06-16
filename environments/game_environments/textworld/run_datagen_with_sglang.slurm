#!/bin/bash
#SBATCH --job-name=textworld_datagen
#SBATCH --output=/home/maxpaperclips/atropos/environments/game_environments/textworld/logs/%j.out
#SBATCH --error=/home/maxpaperclips/atropos/environments/game_environments/textworld/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --requeue

set -e

# Job info
echo "Starting TextWorld data generation job: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Date: $(date)"

# Create log directory for this job
LOG_DIR="/home/maxpaperclips/atropos/environments/game_environments/textworld/logs/$SLURM_JOB_ID"
mkdir -p $LOG_DIR

# Set up environment
cd /home/maxpaperclips/atropos

# Model configuration - using 8B model for better format compliance
MODEL_NAME="NousResearch/DeepHermes-3-Llama-3-8B-Preview"
PORT=30000

# Kill any existing SGLang servers on this port
echo "Cleaning up any existing SGLang servers..."
pkill -f "sglang.launch_server" || true
sleep 5

# Launch SGLang server with tensor parallelism across 4 GPUs (8B model needs less)
echo "Launching SGLang server on port $PORT with model $MODEL_NAME..."
cd /home/maxpaperclips/sglang
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup uv run python -m sglang.launch_server \
    --model-path $MODEL_NAME \
    --port $PORT \
    --tp 4 \
    --grammar-backend xgrammar \
    --host 0.0.0.0 \
    --log-level info \
    --trust-remote-code \
    --disable-outlines-disk-cache \
    --mem-fraction-static 0.85 \
    --max-running-requests 32 \
    --watchdog-timeout 600 \
    --schedule-conservativeness 1.5 \
    --chunked-prefill-size 8192 \
    > $LOG_DIR/sglang.log 2>&1 &

SGLANG_PID=$!
echo "SGLang server PID: $SGLANG_PID"

# Wait for SGLang server to be ready
echo "Waiting for SGLang server to start..."
for i in {1..120}; do
    if curl -s http://localhost:$PORT/health > /dev/null; then
        echo "SGLang server is ready!"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "ERROR: SGLang server failed to start after 2 minutes"
        cat $LOG_DIR/sglang.log
        exit 1
    fi
    sleep 1
done

# Configure output file with job ID
OUTPUT_FILE="data/textworld_deephermes8b_dataset_${SLURM_JOB_ID}.jsonl"

# Run Atropos data generation using process command
echo "Starting Atropos TextWorld data generation with 4 workers..."
cd /home/maxpaperclips/atropos
uv run python -m environments.game_environments.textworld.textworld_env process \
    --config environments/game_environments/textworld/config_process.yaml \
    --env.total_steps 10000 \
    --env.group_size 16 \
    --env.max_num_workers 4 \
    --env.data_path_to_save_groups "$OUTPUT_FILE" \
    --slurm true \
    --openai.model_name "$MODEL_NAME" \
    --openai.base_url "http://localhost:$PORT/v1" \
    --openai.api_key "dummy"

# Capture exit code
EXIT_CODE=$?

# Kill SGLang server
echo "Shutting down SGLang server..."
kill $SGLANG_PID || true
sleep 5
pkill -f "sglang.launch_server" || true

# Report results
if [ $EXIT_CODE -eq 0 ]; then
    echo "Data generation completed successfully!"
    echo "Output saved to: $OUTPUT_FILE"
    echo "Job completed at: $(date)"
    
    # Count results
    if [ -f "$OUTPUT_FILE" ]; then
        NUM_LINES=$(wc -l < "$OUTPUT_FILE")
        echo "Generated $NUM_LINES data points"
    fi
else
    echo "Data generation failed with exit code: $EXIT_CODE"
    echo "Check logs at: $LOG_DIR/sglang.log"
    exit $EXIT_CODE
fi