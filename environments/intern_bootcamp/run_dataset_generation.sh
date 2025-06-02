#!/bin/bash

# InternBootcamp Dataset Generation Script
# This script runs the intern bootcamp environment in process mode to generate
# a dataset using rejection sampling with DeepHermes-3-Mistral-24B-Preview

set -e  # Exit on any error

echo "Starting InternBootcamp dataset generation with DeepHermes-3-Mistral-24B-Preview..."
echo "Configuration: RandomTask mode with rejection sampling"
echo "Output: data/intern_bootcamp_deephermes24b_dataset.jsonl"
echo

# Ensure we're in the right directory
cd /home/maxpaperclips/atropos

# API key is configured in the YAML file, no environment variable needed

# Create data directory if it doesn't exist
mkdir -p data

# Run the intern bootcamp environment in process mode
python -m environments.intern_bootcamp.intern_bootcamp_env process \
    --config environments/intern_bootcamp/config_process.yaml \
    --env--total_steps 1000 \
    --env--group_size 16 \
    --env--temperature 0.7 \
    --env--top_p 0.9 \
    --openai--model_name "DeepHermes-3-Mistral-24B-Preview"

echo
echo "Dataset generation completed!"
echo "Output saved to: data/intern_bootcamp_deephermes24b_dataset.jsonl"
echo
echo "Summary:"
echo "- Generated 1000 problem groups (16 responses each = 16,000 total responses)"
echo "- Used RandomTask to sample from all available InternBootcamp tasks"
echo "- Rejection sampling enabled (ensure_scores_are_not_same=false)"
echo "- Full conversations saved for SFT training"
echo "- Using DeepHermes-3-Mistral-24B-Preview for in-distribution data generation"
echo
echo "To use this dataset for SFT training, filter for high-scoring responses:"
echo "jq 'select(.scores[] > 0.5)' data/intern_bootcamp_deephermes24b_dataset.jsonl > data/intern_bootcamp_filtered.jsonl"