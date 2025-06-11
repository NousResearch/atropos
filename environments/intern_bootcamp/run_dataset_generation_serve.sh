#!/bin/bash

# InternBootcamp Dataset Generation Script - Serve Mode
# This script runs the intern bootcamp environment using serve mode with run-api
# for more robust data generation with DeepHermes-3-Mistral-24B-Preview

set -e  # Exit on any error

# Export the API key for this session
export HERMES_API_KEY="sk-CRs4gcGL5Jai3ojQ2BKxxA"

echo "Starting InternBootcamp dataset generation using serve mode..."
echo "Configuration: RandomTask mode with rejection sampling"
echo "Model: DeepHermes-3-Mistral-24B-Preview"
echo "Output: data/intern_bootcamp_deephermes24b_dataset_local.jsonl"
echo

# Ensure we're in the right directory
cd /home/maxpaperclips/atropos

# Create data and logs directories
mkdir -p data
mkdir -p environments/intern_bootcamp/logs

# Define output files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
API_LOG="environments/intern_bootcamp/logs/${TIMESTAMP}_api.log"
ENV_LOG="environments/intern_bootcamp/logs/${TIMESTAMP}_env.log"
SFT_OUTPUT="data/intern_bootcamp_deephermes24b_dataset_${TIMESTAMP}.jsonl"

echo "API Log: ${API_LOG}"
echo "Environment Log: ${ENV_LOG}"
echo "Output: ${SFT_OUTPUT}"
echo

# Function to cleanup processes on exit
cleanup() {
    echo "Cleaning up..."
    if [ ! -z "$ENV_PID" ]; then
        kill $ENV_PID 2>/dev/null || true
    fi
    if [ ! -z "$API_PID" ]; then
        kill $API_PID 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Start the central API server in background
echo "Starting run-api server..."
uv run run-api > "${API_LOG}" 2>&1 &
API_PID=$!
echo "API server started with PID: ${API_PID}"

# Wait for API to be ready
echo "Waiting for API server to be ready..."
sleep 5

# Check if API is responding
for i in {1..12}; do
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
        echo "API server is ready!"
        break
    fi
    if [ $i -eq 12 ]; then
        echo "API server failed to start after 2 minutes"
        exit 1
    fi
    echo "Waiting for API... (attempt $i/12)"
    sleep 10
done

# Start the intern_bootcamp environment server
echo "Starting InternBootcamp environment server..."
uv run python -m environments.intern_bootcamp.intern_bootcamp_env serve \
    --openai.base_url https://inference-api.nousresearch.com/v1 \
    --openai.api_key "${HERMES_API_KEY}" \
    --openai.model_name DeepHermes-3-Mistral-24B-Preview \
    --env.max_token_length 14000 \
    --env.group_size 16 \
    --env.total_steps 100 \
    --env.steps_per_eval 1000 \
    --env.dump_rollouts True \
    --env.task_name RandomTask \
    --env.temperature 0.7 \
    --env.top_p 0.9 \
    --env.tokenizer_name DeepHermes-3-Mistral-24B-Preview \
    --env.use_wandb true \
    --env.wandb_name "intern_bootcamp_deephermes24b_dataset_generation_local_${TIMESTAMP}" \
    --slurm False > "${ENV_LOG}" 2>&1 &
ENV_PID=$!
echo "Environment server started with PID: ${ENV_PID}"

# Wait for environment to start processing
echo "Waiting for environment server to start processing..."
sleep 30

# Collect the data using atropos-sft-gen (smaller amount for local testing)
echo "Starting data collection with atropos-sft-gen..."
uv run atropos-sft-gen \
    --group-size 16 \
    --max-token-len 14000 \
    --num-seqs-to-save 1600 \
    --save-top-n-per-group 8 \
    "${SFT_OUTPUT}"

echo "Dataset generation completed!"
echo "Output: ${SFT_OUTPUT}"
echo "API Log: ${API_LOG}"
echo "Environment Log: ${ENV_LOG}"

# Run the filtering script
echo "Filtering dataset for high-quality responses..."
uv run python environments/intern_bootcamp/filter_dataset.py \
    "${SFT_OUTPUT}" \
    --format sft \
    --min-score 0.0 \
    --verbose

echo "All tasks completed successfully!"
echo
echo "Summary:"
echo "- Generated 100 problem groups (16 responses each = 1,600 total responses)"
echo "- Used RandomTask to sample from all available InternBootcamp tasks"
echo "- Rejection sampling with atropos-sft-gen (top 8 per group)"
echo "- Full conversations saved for SFT training"
echo "- Using DeepHermes-3-Mistral-24B-Preview for in-distribution data generation"