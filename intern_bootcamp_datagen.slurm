#!/bin/bash
#SBATCH --job-name=intern_bootcamp_datagen
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=12:00:00

# Create logs directory if it doesn't exist
mkdir -p logs/$SLURM_JOB_ID

# Set ulimit higher for async io
ulimit -n 32000

# Export log directory
export LOGDIR="$(pwd)/logs/${SLURM_JOB_ID}"

echo "Starting InternBootcamp data generation on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Log directory: $LOGDIR"

# Basic configuration
export MODEL_NAME="deepseek-ai/DeepSeek-R1"  # High quality reasoning model
export ATROPOS_ENV="${HOME}/atropos/.venv"
export SGLANG_ENV="${HOME}/sglang/.venv"

# Create data directory if it doesn't exist
mkdir -p ${HOME}/atropos/data

# Step 1: Start the Atropos API server (non-blocking)
echo "Starting Atropos API server..."
source ${ATROPOS_ENV}/bin/activate
cd ${HOME}/atropos
run-api > ${LOGDIR}/api.log 2>&1 &
API_PID=$!
echo "API server started with PID: $API_PID"
deactivate

# Give API server time to start
sleep 10

# Step 2: Start SGLang with data and tensor parallelism
echo "Starting SGLang server with DP=4, TP=2..."
source ${SGLANG_ENV}/bin/activate

# Kill any existing SGLang processes
pkill -f sglang.launch_server || true
sleep 5

# Launch SGLang with tensor parallelism
cd ${HOME}/sglang
nohup python3 -m sglang.launch_server \
    --model ${MODEL_NAME} \
    --tp 8 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 9000 \
    --disable-outlines-disk-cache \
    --grammar-backend xgrammar \
    --attention-backend triton \
    > ${LOGDIR}/sglang.log 2>${LOGDIR}/sglang.err &
SGLANG_PID=$!
echo "SGLang server started with PID: $SGLANG_PID"

# Wait for SGLang to be ready
echo "Waiting for SGLang to initialize..."
for i in {1..60}; do
    if curl -s http://localhost:9000/v1/models > /dev/null; then
        echo "SGLang is ready!"
        break
    fi
    echo "Waiting for SGLang... ($i/60)"
    sleep 5
done

deactivate

# Step 3: Start a fake trainer to allow environment to start
echo "Starting fake trainer..."
source ${ATROPOS_ENV}/bin/activate

cd ${HOME}/atropos
python fake_trainer.py > ${LOGDIR}/fake_trainer.log 2>&1 &
TRAINER_PID=$!
echo "Fake trainer started with PID: $TRAINER_PID"

# Give the fake trainer time to register
sleep 10

# Step 4: Start the InternBootcamp environment server
echo "Starting InternBootcamp environment server..."
python -m environments.intern_bootcamp.intern_bootcamp_env serve \
    --config environments/intern_bootcamp/config_serve.yaml \
    --env.use_wandb false \
    --slurm false \
    > ${LOGDIR}/intern_bootcamp.log 2>&1 &
ENV_PID=$!
echo "InternBootcamp environment started with PID: $ENV_PID"

# Monitor the services
echo "All services started. Monitoring..."
echo "Logs are in: $LOGDIR"
echo "Data will be saved to: ${HOME}/atropos/data/"

# Wait for the environment server to complete or until job time limit
wait $ENV_PID

# Cleanup
echo "InternBootcamp environment completed. Cleaning up..."
kill $SGLANG_PID $API_PID $TRAINER_PID 2>/dev/null || true

echo "Data generation completed!"
echo "Check logs in: $LOGDIR"
echo "Data files in: ${HOME}/atropos/data/"