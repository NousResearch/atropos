#!/bin/bash
#SBATCH --job-name=diplomacy_memory
#SBATCH --output=/home/maxpaperclips/atropos/logs/diplomacy_memory_%j.out
#SBATCH --error=/home/maxpaperclips/atropos/logs/diplomacy_memory_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=batch

# Export API key
export HERMES_API_KEY="sk-CRs4gcGL5Jai3ojQ2BKxxA"

# Change to project directory
cd /home/maxpaperclips/atropos

# Create logs directory if it doesn't exist
mkdir -p logs

# Total interactions in the dataset
TOTAL_INTERACTIONS=1178

echo "Starting memory generation for Diplomacy trajectory"
echo "Total interactions to process: $TOTAL_INTERACTIONS"
echo "Output will be saved to data/diplomacy_with_memory.jsonl"
echo "Script will automatically resume from checkpoints if interrupted"

# Run the memory generation script
# The Python script handles checkpoint detection and resuming internally
uv run python scripts/generate_diplomacy_memory.py \
    --start 0 \
    --limit $TOTAL_INTERACTIONS

MEMORY_EXIT_CODE=$?

if [ $MEMORY_EXIT_CODE -eq 0 ]; then
    echo "Memory generation complete!"
    echo "Starting thinking block generation..."
    
    # Generate thinking blocks using the memory-enhanced data
    uv run python scripts/add_diplomacy_thinking.py \
        --start 0 \
        --limit $TOTAL_INTERACTIONS
    
    if [ $? -eq 0 ]; then
        echo "Thinking block generation complete!"
        echo "Final output saved to data/diplomacy_final_transformed.jsonl"
    else
        echo "Error during thinking block generation"
        exit 1
    fi
else
    echo "Error during memory generation (exit code: $MEMORY_EXIT_CODE)"
    exit $MEMORY_EXIT_CODE
fi