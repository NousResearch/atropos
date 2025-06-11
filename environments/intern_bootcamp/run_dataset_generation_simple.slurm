#!/bin/bash
#SBATCH --job-name=intern-bootcamp-datagen
#SBATCH --output=environments/intern_bootcamp/logs/%j.out
#SBATCH --error=environments/intern_bootcamp/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

# Simple dataset generation using serve mode with direct file writing

set -e

cd /home/maxpaperclips/atropos

# Configuration
OUTPUT_FILE="data/intern_bootcamp_deephermes24b_dataset_${SLURM_JOB_ID}.jsonl"
MAX_WORKERS=32  # Number of parallel tasks
TOTAL_STEPS=10000

echo "Starting InternBootcamp dataset generation job ${SLURM_JOB_ID}"
echo "Output file: ${OUTPUT_FILE}"
echo "Max parallel workers: ${MAX_WORKERS}"
echo "Total steps: ${TOTAL_STEPS}"

# Run serve mode with direct file writing (no API server needed)
uv run python -m environments.intern_bootcamp.intern_bootcamp_env serve \
    --config environments/intern_bootcamp/config_process.yaml \
    --env.data_path_to_save_groups "${OUTPUT_FILE}" \
    --env.batch_size -1 \
    --env.max_num_workers ${MAX_WORKERS} \
    --env.total_steps ${TOTAL_STEPS} \
    --env.group_size 64 \
    --env.ensure_scores_are_not_same false \
    --env.dump_rollouts true \
    --env.use_wandb false \
    --env.do_send_to_api false

echo "Dataset generation complete!"
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