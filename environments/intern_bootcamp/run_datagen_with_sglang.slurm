#!/bin/bash
#SBATCH --job-name=intern-bootcamp-sglang-datagen
#SBATCH --output=environments/intern_bootcamp/logs/%j.out
#SBATCH --error=environments/intern_bootcamp/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gpus=8
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00

# Dataset generation with local SGLang serving DeepHermes-3-Mistral-24B

set -e

# Create logs directory if it doesn't exist
mkdir -p environments/intern_bootcamp/logs/${SLURM_JOB_ID}

# Set ulimit higher for async IO
ulimit -n 32000

LOGDIR="$(pwd)/environments/intern_bootcamp/logs/${SLURM_JOB_ID}"

# Configuration
MODEL_NAME="NousResearch/DeepHermes-3-Mistral-24B-Preview"
SGLANG_PORT=30000
OUTPUT_FILE="data/intern_bootcamp_deephermes24b_sglang_dataset_${SLURM_JOB_ID}.jsonl"
MAX_WORKERS=32
TOTAL_STEPS=10000

echo "Starting InternBootcamp dataset generation with local SGLang"
echo "Model: ${MODEL_NAME}"
echo "Output file: ${OUTPUT_FILE}"
echo "Max parallel workers: ${MAX_WORKERS}"
echo "Total steps: ${TOTAL_STEPS}"

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
# Configuration for InternBootcamp dataset generation using local SGLang
# Modified from config_process.yaml to use local endpoint

env:
  # Task configuration - RandomTask selects from all available bootcamp tasks
  task_name: "RandomTask"
  task_params: {}
  
  # Reward configuration for scoring
  correct_reward: 1.0
  incorrect_reward: -0.5
  format_bonus: 0.2
  
  # Generation parameters
  group_size: 64  # Generate 64 responses per problem for rejection sampling
  batch_size: -1  # Disable batching for direct file writing
  total_steps: ${TOTAL_STEPS}  # Process problems
  max_num_workers: ${MAX_WORKERS}  # Number of parallel workers
  
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
  
  # Tokenizer for processing (use same model as generation model)
  tokenizer_name: "${MODEL_NAME}"
  
  # WandB tracking
  use_wandb: true
  wandb_name: "intern_bootcamp_deephermes24b_sglang_dataset_generation"
  
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

# Step 3: Run the data generation
echo "Starting data generation..."
cd /home/maxpaperclips/atropos

# Run process mode with direct file writing (no API server needed)
uv run python -m environments.intern_bootcamp.intern_bootcamp_env process \
    --config ${CONFIG_FILE} \
    --env.data_path_to_save_groups "${OUTPUT_FILE}" \
    --env.batch_size -1 \
    --env.max_num_workers ${MAX_WORKERS} \
    --env.total_steps ${TOTAL_STEPS} \
    --env.group_size 64 \
    --env.ensure_scores_are_not_same false \
    --env.dump_rollouts true \
    --env.use_wandb false \
    --env.do_send_to_api false \
    --env.rollout_server_url ""

echo "Dataset generation complete!"

# Kill SGLang server
echo "Shutting down SGLang server..."
kill ${SGLANG_PID} || true

echo "Output saved to: ${OUTPUT_FILE}"

# Show statistics
echo "Total lines generated:"
wc -l ${OUTPUT_FILE}

echo "File size:"
ls -lh ${OUTPUT_FILE}

# Filter for SFT format (score >= 0.0)
echo "Filtering dataset for SFT..."
SFT_OUTPUT="${OUTPUT_FILE%.jsonl}_sft.jsonl"
uv run python environments/intern_bootcamp/filter_dataset.py \
    ${OUTPUT_FILE} \
    --format sft \
    --min-score 0.0 \
    --output ${SFT_OUTPUT} \
    --verbose

echo "Filtered SFT dataset saved to: ${SFT_OUTPUT}"
echo "Job completed at $(date)"