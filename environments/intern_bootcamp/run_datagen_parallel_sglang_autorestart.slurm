#!/bin/bash
#SBATCH --job-name=intern-bootcamp-parallel-sglang
#SBATCH --output=environments/intern_bootcamp/logs/%j.out
#SBATCH --error=environments/intern_bootcamp/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gpus=8
#SBATCH --cpus-per-task=64
#SBATCH --time=7-00:00:00
#SBATCH --requeue                    # Allow job to be requeued after node failure
#SBATCH --open-mode=append           # Append to log files on restart

# Dataset generation with local SGLang serving DeepHermes-3-Mistral-24B
# This version uses the new parallel processing feature and auto-restart

# Don't use set -e so the job continues on errors
# set -e

# Function to cleanup on exit
cleanup() {
    echo "Cleaning up..."
    # Kill SGLang server if it exists
    if [ ! -z "$SGLANG_PID" ]; then
        kill $SGLANG_PID 2>/dev/null || true
    fi
}

# Set trap to cleanup on exit
trap cleanup EXIT

# Load environment variables if .env file exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Create logs directory if it doesn't exist
mkdir -p environments/intern_bootcamp/logs/${SLURM_JOB_ID}

# Set ulimit higher for async IO
ulimit -n 32000

LOGDIR="$(pwd)/environments/intern_bootcamp/logs/${SLURM_JOB_ID}"

# Configuration
MODEL_NAME="NousResearch/DeepHermes-3-Mistral-24B-Preview"
SGLANG_PORT=8000
OUTPUT_FILE="data/intern_bootcamp_deephermes24b_parallel_sglang_dataset_${SLURM_JOB_ID}.jsonl"
MAX_WORKERS=4  # Number of parallel groups to process
TOTAL_STEPS=10000
GROUP_SIZE=64  # Responses per group

# Check if this is a restart
if [ -f "${OUTPUT_FILE}" ]; then
    EXISTING_LINES=$(wc -l < "${OUTPUT_FILE}")
    echo "RESTART DETECTED: Found existing file with ${EXISTING_LINES} lines"
    echo "Will continue from where we left off..."
    # Adjust total steps to account for already completed work
    TOTAL_STEPS=$((TOTAL_STEPS - EXISTING_LINES))
    if [ $TOTAL_STEPS -le 0 ]; then
        echo "Job already completed!"
        exit 0
    fi
else
    echo "Starting fresh InternBootcamp PARALLEL dataset generation with local SGLang"
fi

echo "Model: ${MODEL_NAME}"
echo "Output file: ${OUTPUT_FILE}"
echo "Max parallel groups: ${MAX_WORKERS}"
echo "Remaining steps: ${TOTAL_STEPS}"
echo "Group size: ${GROUP_SIZE}"

# Step 1: Launch SGLang server
echo "Launching SGLang server on port ${SGLANG_PORT}..."

# Define SGLang environment path
SGLANG_ENV="/home/maxpaperclips/sglang/.venv"

# Activate SGLang virtual environment
source ${SGLANG_ENV}/bin/activate

python3 -m sglang.launch_server \
    --model-path ${MODEL_NAME} \
    --host 0.0.0.0 \
    --port ${SGLANG_PORT} \
    --tp 8 \
    --disable-outlines-disk-cache \
    --grammar-backend xgrammar
    > ${LOGDIR}/sglang.log 2>&1 &

SGLANG_PID=$!
echo "SGLang server started with PID ${SGLANG_PID}"

# Deactivate SGLang environment
deactivate

# Wait for SGLang to be ready
echo "Waiting for SGLang server to be ready..."
sleep 60  # Give it time to load the model

# Check if server is responsive
MAX_RETRIES=30
for i in $(seq 1 $MAX_RETRIES); do
    if curl -s http://localhost:${SGLANG_PORT}/health > /dev/null 2>&1; then
        echo "SGLang server is ready!"
        break
    fi
    echo "Waiting for SGLang... (attempt $i/$MAX_RETRIES)"
    sleep 10
done

# Step 2: Create a modified config file to use local SGLang endpoint
CONFIG_FILE="${LOGDIR}/config_sglang.yaml"
cat > ${CONFIG_FILE} << EOF
---
# Configuration for InternBootcamp PARALLEL dataset generation using local SGLang
# This uses the new parallel processing feature for improved throughput

env:
  # Task configuration - RandomTask selects from all available bootcamp tasks
  task_name: "RandomTask"
  task_params: {}
  
  # Reward configuration for scoring
  correct_reward: 1.0
  incorrect_reward: -0.5
  format_bonus: 0.2
  
  # Generation parameters
  group_size: ${GROUP_SIZE}  # Generate responses per problem for rejection sampling
  batch_size: -1  # Disable batching for direct file writing
  total_steps: ${TOTAL_STEPS}  # Process problems
  max_num_workers: ${MAX_WORKERS}  # Number of parallel groups to process
  use_parallel_processing: true  # Enable new parallel processing mode
  
  # Reasoning requirements
  require_reasoning: true
  min_reasoning_length: 50
  
  # Sampling parameters for DeepHermes
  temperature: 0.7
  top_p: 0.9
  max_token_length: 24576
  
  # Output configuration
  data_path_to_save_groups: "${OUTPUT_FILE}"
  ensure_scores_are_not_same: false  # Allow rejection sampling
  include_messages: true  # Save full conversations
  strip_tokens_and_masks: true  # Save space by removing tokens/masks
  
  # Tokenizer for processing (use same model as generation model)
  tokenizer_name: "${MODEL_NAME}"
  
  # WandB tracking
  use_wandb: true
  wandb_name: "intern_bootcamp_deephermes24b_parallel_sglang_generation"
  
  # Enable direct rollout dumping to datadumps directory
  dump_rollouts: true
  min_score_to_save: 0.0  # Only save responses with score >= 0.0
  do_send_to_api: false  # Disable API sending for direct file writing

# OpenAI-compatible API configuration for local SGLang
openai:
  model_name: "${MODEL_NAME}"
  base_url: "http://localhost:${SGLANG_PORT}/v1"
  api_key: "dummy"  # SGLang doesn't need a real API key
  
# Server configuration for serve mode
slurm: true
testing: false

EOF

# Step 3: Run the data generation with parallel processing
echo "Starting PARALLEL data generation..."
echo "This will process up to ${MAX_WORKERS} groups in parallel (${MAX_WORKERS} x ${GROUP_SIZE} = $((MAX_WORKERS * GROUP_SIZE)) concurrent requests)"
cd /home/maxpaperclips/atropos

# Run process mode with error handling
# The || true ensures the script continues even if the Python script fails
uv run python -m environments.intern_bootcamp.intern_bootcamp_env process \
    --config ${CONFIG_FILE} \
    --env.use_parallel_processing true \
    --env.max_num_workers ${MAX_WORKERS} \
    --env.total_steps ${TOTAL_STEPS} \
    --env.group_size ${GROUP_SIZE} \
    --env.strip_tokens_and_masks true || {
    echo "Data generation encountered an error, but continuing..."
    echo "Error code: $?"
}

echo "Dataset generation phase complete!"

# Kill SGLang server
echo "Shutting down SGLang server..."
kill ${SGLANG_PID} 2>/dev/null || true

echo "Output saved to: ${OUTPUT_FILE}"

# Show statistics
echo "Total lines generated:"
wc -l ${OUTPUT_FILE} 2>/dev/null || echo "0"

echo "File size:"
ls -lh ${OUTPUT_FILE} 2>/dev/null || echo "No file generated"

# Check if we should resubmit for more data
if [ -f "${OUTPUT_FILE}" ]; then
    FINAL_LINES=$(wc -l < "${OUTPUT_FILE}")
    if [ $FINAL_LINES -lt 10000 ]; then
        echo "Only generated ${FINAL_LINES} lines out of 10000 target"
        echo "Consider resubmitting job to continue data collection"
        # Uncomment below to auto-resubmit
        # sbatch $0
    else
        echo "Target reached! Generated ${FINAL_LINES} lines"
    fi
fi

echo "Job completed at $(date)"
