#!/bin/bash
#SBATCH --job-name=intern_bootcamp_dataset_serve
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/maxpaperclips/atropos/environments/intern_bootcamp/logs/%j.out
#SBATCH --error=/home/maxpaperclips/atropos/environments/intern_bootcamp/logs/%j.err

# Load environment variables from .env file and bashrc
source /home/maxpaperclips/.bashrc
if [ -f /home/maxpaperclips/atropos/.env ]; then
    source /home/maxpaperclips/atropos/.env
fi

# Change to project directory
cd /home/maxpaperclips/atropos

# Create data and logs directories
mkdir -p data
mkdir -p environments/intern_bootcamp/logs

# Define output files
API_LOG="environments/intern_bootcamp/logs/${SLURM_JOB_ID}_api.log"
ENV_LOG="environments/intern_bootcamp/logs/${SLURM_JOB_ID}_env.log"
SFT_OUTPUT="data/intern_bootcamp_deephermes24b_dataset_${SLURM_JOB_ID}.jsonl"

echo "Starting InternBootcamp dataset generation using serve mode"
echo "Job ID: ${SLURM_JOB_ID}"
echo "API Log: ${API_LOG}"
echo "Environment Log: ${ENV_LOG}"
echo "Output: ${SFT_OUTPUT}"
echo

# Start the central API server in background
echo "Starting run-api server..."
uv run run-api > "${API_LOG}" 2>&1 &
API_PID=$!
echo "API server started with PID: ${API_PID}"

# Wait for API to be ready
echo "Waiting for API server to be ready..."
sleep 10

# Check if API is responding
for i in {1..30}; do
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
        echo "API server is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "API server failed to start after 5 minutes"
        kill $API_PID 2>/dev/null
        exit 1
    fi
    echo "Waiting for API... (attempt $i/30)"
    sleep 10
done

# Start the intern_bootcamp environment server
echo "Starting InternBootcamp environment server..."
uv run python -m environments.intern_bootcamp.intern_bootcamp_env serve \
    --openai.base_url https://inference-api.nousresearch.com/v1 \
    --openai.api_key "${HERMES_API_KEY}" \
    --openai.model_name DeepHermes-3-Llama-3-8B-Preview \
    --env.max_token_length 14000 \
    --env.group_size 64 \
    --env.total_steps 50000 \
    --env.steps_per_eval 1000 \
    --env.dump_rollouts True \
    --env.task_name RandomTask \
    --env.temperature 0.7 \
    --env.top_p 0.9 \
    --env.tokenizer_name NousResearch/DeepHermes-3-Mistral-24B-Preview \
    --env.use_wandb true \
    --env.wandb_name "intern_bootcamp_deephermes24b_dataset_generation_${SLURM_JOB_ID}" \
    --slurm False > "${ENV_LOG}" 2>&1 &
ENV_PID=$!
echo "Environment server started with PID: ${ENV_PID}"

# Wait for environment to start processing
echo "Waiting for environment server to start processing..."
sleep 30

# Collect the data using atropos-sft-gen
echo "Starting data collection with atropos-sft-gen..."
uv run atropos-sft-gen \
    --group-size 64 \
    --max-token-len 14000 \
    --num-seqs-to-save 3200000 \
    --save-top-n-per-group 32 \
    "${SFT_OUTPUT}"

# Stop the environment and API servers
echo "Stopping servers..."
kill $ENV_PID 2>/dev/null
kill $API_PID 2>/dev/null

# Wait for processes to terminate
wait $ENV_PID 2>/dev/null
wait $API_PID 2>/dev/null

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