#!/bin/bash
#SBATCH --job-name=intern_bootcamp_datagen
#SBATCH --output=/home/maxpaperclips/atropos/environments/intern_bootcamp/logs/%j.out
#SBATCH --error=/home/maxpaperclips/atropos/environments/intern_bootcamp/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --requeue

set -e

# Job info
echo "Starting InternBootcamp data generation job: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Date: $(date)"

# Create log directory for this job
LOG_DIR="/home/maxpaperclips/atropos/environments/intern_bootcamp/logs/$SLURM_JOB_ID"
mkdir -p $LOG_DIR

# Set up environment
cd /home/maxpaperclips/atropos

# Model configuration
MODEL_NAME="NousResearch/DeepHermes-3-Mistral-24B-Preview"
PORT=8088

# Kill any existing SGLang servers on this port
echo "Cleaning up any existing SGLang servers..."
pkill -f "sglang.launch_server" || true
sleep 5

# Launch SGLang server with tensor parallelism across all 8 GPUs
echo "Launching SGLang server on port $PORT with model $MODEL_NAME..."
cd /home/maxpaperclips/sglang
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup uv run python -m sglang.launch_server \
    --model-path $MODEL_NAME \
    --port $PORT \
    --tp 8 \
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
OUTPUT_FILE="data/intern_bootcamp_deephermes24b_sglang_dataset_${SLURM_JOB_ID}.jsonl"

# Run Atropos data generation using process command
echo "Starting Atropos data generation with 4 workers..."
cd /home/maxpaperclips/atropos
uv run python -m environments.intern_bootcamp.intern_bootcamp_env process \
    --config environments/intern_bootcamp/config_process.yaml \
    --env.total_steps 100000 \
    --env.group_size 64 \
    --env.temperature 0.7 \
    --env.top_p 0.9 \
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