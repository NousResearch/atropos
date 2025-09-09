#!/bin/bash
#SBATCH --job-name=factorio_genrm_data
#SBATCH --output=/home/maxpaperclips/atropos/logs/factorio_genrm_%j.out
#SBATCH --error=/home/maxpaperclips/atropos/logs/factorio_genrm_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=batch

# Configuration - easily adjustable
FACTORIO_INSTANCES=1  # Number of Factorio containers to run (1 for sequential processing)

# Export API key for Hermes-405B judge calls
export HERMES_API_KEY="sk-CRs4gcGL5Jai3ojQ2BKxxA"

# Change to project directory
cd /home/maxpaperclips/atropos

# Create logs directory if it doesn't exist
mkdir -p logs

# Set ulimit for many Docker containers and async I/O
ulimit -n 32000

# Cleanup function to stop Factorio containers on exit
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

# Start Factorio containers
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

# Set up environment variables
export PYTHONPATH=/home/maxpaperclips/atropos:$PYTHONPATH
export MODEL_NAME="NousResearch/Hermes-4-Qwen3-14B-1-e3"
export WANDB_PROJECT="factorio-genrm-data-generation"

# Log configuration
echo "=== Factorio GenRM Data Generation Configuration ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Model: $MODEL_NAME"
echo "API Key set: $([ -z \"$HERMES_API_KEY\" ] && echo 'NO' || echo 'YES')"
echo "Output directory: data/"
echo "Expected output: factorio_genrm_hermes405b.jsonl"
echo "Containers running: $RUNNING_CONTAINERS"
echo "================================================"

# Run Factorio GenRM data generation
echo "Starting Factorio GenRM data generation..."

# Use uv to run the environment in process mode for data generation
# Using the standard CLI argument format: --env.* for environment config, --openai.* for API config
uv run python environments/game_environments/factorio_env/factorio_env_genrm.py process \
    --env.total_steps=1000 \
    --env.max_num_workers=1 \
    --env.group_size=4 \
    --env.max_steps_per_episode=4 \
    --env.data_path_to_save_groups=/home/maxpaperclips/atropos/data/factorio_genrm_hermes405b.jsonl \
    --openai.model_name=Hermes-4-405B \
    --openai.base_url=https://inference-api.nousresearch.com/v1 \
    --openai.api_key=$HERMES_API_KEY

PYTHON_EXIT_CODE=$?

if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo "Data generation completed successfully!"
    echo "Checking output file..."
    
    OUTPUT_FILE="/home/maxpaperclips/atropos/data/factorio_genrm_hermes405b.jsonl"
    if [ -f "$OUTPUT_FILE" ]; then
        TRAJECTORY_COUNT=$(wc -l < "$OUTPUT_FILE")
        FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
        echo "Output file: $OUTPUT_FILE"
        echo "Trajectories collected: $TRAJECTORY_COUNT"
        echo "File size: $FILE_SIZE"
        
        # Show sample of collected data
        echo "Sample trajectories:"
        head -3 "$OUTPUT_FILE" | jq -r '.metadata.trajectory_id + " (task: " + .metadata.task_name + ", steps: " + (.metadata.steps|tostring) + ")"' 2>/dev/null || echo "Could not parse sample data"
    else
        echo "WARNING: Expected output file not found at $OUTPUT_FILE"
    fi
else
    echo "ERROR: Data generation failed with exit code: $PYTHON_EXIT_CODE"
fi

echo "=== Job Summary ==="
echo "End time: $(date)"
echo "Duration: $((SECONDS/60)) minutes"
echo "Exit code: $PYTHON_EXIT_CODE"
echo "================"

# Cleanup happens automatically via trap
exit $PYTHON_EXIT_CODE