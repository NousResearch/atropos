#!/bin/bash

# Test script for SmolaGents Environment
# This script tests the environment including dataset download

set -e  # Exit on error

echo "=========================================="
echo "SmolaGents Environment Test Script"
echo "=========================================="
echo ""

# Step 1: Check for and download GAIA dataset if needed
echo "Step 1: Checking GAIA dataset..."
if [ ! -d "data/gaia/2023/validation" ]; then
    echo "GAIA dataset not found. Downloading..."
    python -m environments.smolagents_integration.download_gaia --output-dir data/gaia
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to download GAIA dataset"
        exit 1
    fi
    echo "✓ Dataset downloaded successfully"
else
    echo "✓ Dataset already exists"
fi
echo ""

# Step 2: Check for OpenAI API key
echo "Step 2: Checking OpenAI API configuration..."
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY environment variable not set"
    echo "Please set it with: export OPENAI_API_KEY='your-api-key'"
    exit 1
fi
echo "✓ OpenAI API key found"
echo ""

# Step 3: Run a test with the smolagents environment
echo "Step 3: Running test with smolagents environment..."
echo "This will process a small batch of examples from the GAIA validation set"
echo ""

# Configuration
OUTPUT_FILE="test_output_$(date +%Y%m%d_%H%M%S).jsonl"
GROUP_SIZE=2  # Small group size for testing
MAX_STEPS=8   # Fewer steps for faster testing

echo "Configuration:"
echo "  - Model: gpt-4o-mini (OpenAI)"
echo "  - Output file: $OUTPUT_FILE"
echo "  - Group size: $GROUP_SIZE"
echo "  - Max agent steps: $MAX_STEPS"
echo ""

# Run the process command with test settings (using -m to run as module)
python -m environments.smolagents_integration.smolagents_env process \
    --env.data_path_to_save_groups "$OUTPUT_FILE" \
    --env.group_size $GROUP_SIZE \
    --env.max_steps $MAX_STEPS \
    --env.batch_size 1 \
    --env.total_steps 1 \
    --env.use_wandb false \
    --env.debug_scoring true \
    --openai.model_name "gpt-4o-mini" \
    --openai.base_url "https://api.openai.com/v1" \
    --openai.api_key "$OPENAI_API_KEY"

# Check if test completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Test completed successfully!"
    echo "=========================================="
    echo ""
    echo "Output files created:"
    echo "  - $OUTPUT_FILE (JSONL data)"
    echo "  - ${OUTPUT_FILE%.jsonl}.html (HTML visualization)"
    echo ""
    echo "You can open the HTML file in a browser to view the results:"
    echo "  firefox ${OUTPUT_FILE%.jsonl}.html"
    echo "  # or"
    echo "  google-chrome ${OUTPUT_FILE%.jsonl}.html"
else
    echo ""
    echo "ERROR: Test failed!"
    exit 1
fi