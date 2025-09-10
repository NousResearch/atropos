#!/bin/bash
#SBATCH --job-name=factorio_genrm_parallel
#SBATCH --output=/home/maxpaperclips/atropos/logs/%j.out
#SBATCH --error=/home/maxpaperclips/atropos/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --exclusive
#SBATCH --gpus-per-task=0
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00

mkdir -p /home/maxpaperclips/atropos/logs/$SLURM_JOB_ID
ulimit -n 32000
export LOGDIR="/home/maxpaperclips/atropos/logs/${SLURM_JOB_ID}"

echo "SLURM nodes: $SLURM_JOB_NODELIST"
echo "Starting parallel Factorio GenRM data generation with Hermes-405B judge at $(date)"
echo "Launching 10 parallel processes for 2000 total episodes (200 per process)"

# Number of Factorio containers to run (one per process)
FACTORIO_INSTANCES=10

# Config for GenRM data generation
export MODEL_NAME="Hermes-4-405B"
export PYTHON_SCRIPT="/home/maxpaperclips/atropos/environments/game_environments/factorio_env/factorio_env_genrm.py"
export HERMES_API_KEY="sk-CRs4gcGL5Jai3ojQ2BKxxA"

# Each process attempts 200 episodes
# NOTE: data_path_to_save_groups is set to /dev/null to discard automatic token dumps
export PYTHON_ARGS="process --openai.model_name=Hermes-4-405B --openai.base_url=https://inference-api.nousresearch.com/v1 --openai.api_key=${HERMES_API_KEY} --env.tokenizer_name=NousResearch/Hermes-4-Qwen3-14B-1-e3 --env.total_steps=200 --env.group_size=4 --env.max_steps_per_episode=250 --env.max_num_workers=1 --env.action_judge_num_judges=1 --env.progress_judge_num_judges=1 --env.outcome_judge_num_judges=1 --env.data_path_to_save_groups=/dev/null"

# Output file that all processes will append to
export OUTPUT_FILE="/home/maxpaperclips/atropos/data/factorio_genrm_hermes405b.jsonl"

# HF auth/cache (if needed for tokenizers)
export HF_TOKEN=${HF_TOKEN:-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}
export HF_HOME="/home/maxpaperclips/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/maxpaperclips/.cache/huggingface"
mkdir -p $HF_HOME

# Paths
export API_ENV="/home/maxpaperclips/atropos/.venv"
export PYTHONPATH=/home/maxpaperclips/atropos:$PYTHONPATH
export WANDB_PROJECT="factorio-genrm-data-generation"

# Get node information
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
export head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Head node: $head_node_ip"
echo "Model: $MODEL_NAME"
echo "Judge: Hermes-4-405B"
echo "Output file: $OUTPUT_FILE"
echo "Episodes per process: 200"
echo "Total target episodes: 2000"
echo "Max steps per episode: 250"
echo "Group size: 4"

# Cleanup function to stop all Factorio containers on exit
cleanup() {
    echo "Cleaning up Factorio containers..."
    for ((i=0; i<$FACTORIO_INSTANCES; i++)); do
        sudo docker stop factorio_$i 2>/dev/null && sudo docker rm factorio_$i 2>/dev/null
    done
    echo "Cleanup completed"
}
trap cleanup EXIT

# Build Factorio Docker image if it doesn't exist
echo "Checking for Factorio Docker image..."
if ! sudo docker images | grep -q "^factorio "; then
    echo "Building Factorio Docker image..."
    FLE_DIR="/home/maxpaperclips/atropos/environments/game_environments/factorio_env/fle"
    if [ -d "${FLE_DIR}/fle/cluster/docker" ]; then
        cd ${FLE_DIR}/fle/cluster/docker
        sudo docker build -t factorio .
        cd /home/maxpaperclips/atropos
    else
        echo "ERROR: Factorio Docker build directory not found at ${FLE_DIR}/fle/cluster/docker"
        exit 1
    fi
else
    echo "Factorio Docker image already exists"
fi

# Clean up any existing Factorio containers from previous runs
echo "Cleaning up any existing Factorio containers..."
for ((i=0; i<$FACTORIO_INSTANCES; i++)); do
    sudo docker stop factorio_$i 2>/dev/null && sudo docker rm factorio_$i 2>/dev/null
done

# Start Factorio containers for each process
echo "Starting $FACTORIO_INSTANCES Factorio container(s)..."
for ((i=0; i<$FACTORIO_INSTANCES; i++)); do
    TCP_PORT=$((27000 + i))
    UDP_PORT=$((34197 + i))
    echo "Starting Factorio container $i on ports TCP:$TCP_PORT UDP:$UDP_PORT"
    
    sudo docker run -d --name factorio_$i \
        -p $TCP_PORT:27015 \
        -p $UDP_PORT:34197/udp \
        -v /home/maxpaperclips/atropos/environments/game_environments/factorio_env/fle/fle/cluster/scenarios:/opt/factorio/scenarios \
        factorio default_lab_scenario
    
    if [ $? -eq 0 ]; then
        echo "Container factorio_$i started successfully"
    else
        echo "ERROR: Failed to start container factorio_$i"
        exit 1
    fi
done

# Wait for containers to initialize
echo "Waiting for Factorio containers to initialize..."
sleep 30

# Verify containers are running
echo "Verifying Factorio containers:"
RUNNING_CONTAINERS=$(sudo docker ps | grep factorio | wc -l)
echo "Running containers: $RUNNING_CONTAINERS/$FACTORIO_INSTANCES"

if [ $RUNNING_CONTAINERS -lt $FACTORIO_INSTANCES ]; then
    echo "ERROR: Only $RUNNING_CONTAINERS of $FACTORIO_INSTANCES containers are running!"
    exit 1
fi

# Test connectivity to all containers
echo "Testing container connectivity..."
for ((i=0; i<$FACTORIO_INSTANCES; i++)); do
    TCP_PORT=$((27000 + i))
    if timeout 5 bash -c "</dev/tcp/localhost/$TCP_PORT"; then
        echo "Container $i (port $TCP_PORT) is accessible"
    else
        echo "ERROR: Container $i (port $TCP_PORT) is not accessible"
        exit 1
    fi
done

echo "=== Factorio GenRM Parallel Data Generation Configuration ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Processes: 10"
echo "Containers: $FACTORIO_INSTANCES"
echo "Episodes per process: 200"
echo "Total episodes target: 2000"
echo "Max steps per episode: 250"
echo "================================================"

# Launch 10 parallel processes
srun -l --export=ALL /home/maxpaperclips/atropos/launch_factorio_genrm_parallel.sh

# Wait for all processes to complete
wait

# Consolidate results
echo "=== Data Collection Summary ==="
if [ -f "$OUTPUT_FILE" ]; then
    TOTAL_EPISODES=$(wc -l < "$OUTPUT_FILE")
    FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
    echo "Output file: $OUTPUT_FILE"
    echo "Total episodes collected: $TOTAL_EPISODES"
    echo "File size: $FILE_SIZE"
    
    # Show sample of collected data
    echo "Sample trajectories:"
    head -3 "$OUTPUT_FILE" | python3 -c "
import sys, json
for line in sys.stdin:
    try:
        data = json.loads(line)
        task = data.get('task_name', 'unknown')
        steps = len(data.get('steps', []))
        outcome = data.get('episode_outcome', 0)
        print(f'  Task: {task}, Steps: {steps}, Success: {outcome > 0}')
    except:
        pass
" 2>/dev/null || echo "Could not parse sample data"
else
    echo "WARNING: No output file generated at $OUTPUT_FILE"
fi

echo "=== Job Summary ==="
echo "End time: $(date)"
echo "Duration: $((SECONDS/60)) minutes"
echo "================"

# Cleanup happens automatically via trap