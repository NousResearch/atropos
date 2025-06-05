#!/bin/bash
#SBATCH --job-name=intern_bootcamp_dataset_multinode
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
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

echo "Starting InternBootcamp dataset generation using multi-node approach"
echo "Job ID: ${SLURM_JOB_ID}"
echo "API Log: ${API_LOG}"
echo "Environment Log: ${ENV_LOG}"
echo "Output: ${SFT_OUTPUT}"
echo "Nodes: $(scontrol show hostnames $SLURM_JOB_NODELIST)"
echo

# Get node names
NODE1=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
NODE2=$(scontrol show hostnames $SLURM_JOB_NODELIST | tail -n1)

echo "Node 1 (API + Environment): $NODE1"
echo "Node 2 (Data Collection): $NODE2"

# Start API server and environment on node 1
echo "Starting API server and environment on node 1..."
srun --nodes=1 --ntasks=1 --nodelist=$NODE1 bash -c "
    cd /home/maxpaperclips/atropos
    source /home/maxpaperclips/.bashrc
    if [ -f /home/maxpaperclips/atropos/.env ]; then
        source /home/maxpaperclips/atropos/.env
    fi
    
    # Start API server
    echo 'Starting API server on $NODE1'
    uv run run-api > '${API_LOG}' 2>&1 &
    API_PID=\$!
    echo 'API server started with PID: \$API_PID'
    
    # Wait for API to be ready
    sleep 15
    for i in {1..30}; do
        if curl -s http://localhost:8000/health >/dev/null 2>&1; then
            echo 'API server is ready on $NODE1!'
            break
        fi
        if [ \$i -eq 30 ]; then
            echo 'API server failed to start on $NODE1'
            exit 1
        fi
        echo 'Waiting for API... (attempt \$i/30)'
        sleep 10
    done
    
    # Start environment server
    echo 'Starting environment server on $NODE1'
    uv run python -m environments.intern_bootcamp.intern_bootcamp_env serve \
        --openai.base_url https://inference-api.nousresearch.com/v1 \
        --openai.api_key '${HERMES_API_KEY}' \
        --openai.model_name DeepHermes-3-Mistral-24B-Preview \
        --env.max_token_length 24576 \
        --env.group_size 64 \
        --env.batch_size 512 \
        --env.total_steps 50000 \
        --env.steps_per_eval 1000 \
        --env.task_name RandomTask \
        --env.temperature 0.7 \
        --env.top_p 0.9 \
        --env.tokenizer_name NousResearch/DeepHermes-3-Mistral-24B-Preview \
        --env.use_wandb true \
        --env.wandb_name 'intern_bootcamp_deephermes24b_dataset_generation_${SLURM_JOB_ID}' \
        --env.ensure_scores_are_not_same false \
        --slurm False > '${ENV_LOG}' 2>&1 &
    ENV_PID=\$!
    echo 'Environment server started with PID: \$ENV_PID'
    
    # Keep servers running until data collection is done
    wait \$ENV_PID
" &
SERVER_JOB_PID=$!

# Wait for servers to be ready
echo "Waiting for servers to start..."
sleep 60

# Start data collection on node 2
echo "Starting data collection on node 2..."
srun --nodes=1 --ntasks=1 --nodelist=$NODE2 bash -c "
    cd /home/maxpaperclips/atropos
    source /home/maxpaperclips/.bashrc
    if [ -f /home/maxpaperclips/atropos/.env ]; then
        source /home/maxpaperclips/atropos/.env
    fi
    
    echo 'Starting atropos-sft-gen on $NODE2'
    echo 'Connecting to API server on $NODE1:8000'
    
    # Set the API server URL to point to node 1
    export ATROPOS_API_URL=http://$NODE1:8000
    
    uv run atropos-sft-gen \
        --group-size 64 \
        --max-token-len 24576 \
        --num-seqs-to-save 3200000 \
        --save-top-n-per-group 32 \
        --api-url http://$NODE1:8000 \
        '${SFT_OUTPUT}'
"

echo "Data collection completed!"

# Stop the server job
echo "Stopping servers..."
kill $SERVER_JOB_PID 2>/dev/null
wait $SERVER_JOB_PID 2>/dev/null

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