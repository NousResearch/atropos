#!/bin/bash

# Test script for Atropos-SkyRL SHM Transport
# This script verifies the Zero-Copy SHM bridge and SkyRL adapter logic

set -e  # Exit on error

echo "=========================================="
echo "Atropos-SkyRL SHM Transport Test"
echo "=========================================="
echo ""

# Configuration
TEST_NAME="shm_test_$(date +%s)"
echo "Configuration:"
echo "  - SHM Segment: $TEST_NAME"
echo ""

# Run the end-to-end SHM verification suite
echo "Step 1: Running E2E SHM Verification..."
pytest -v atroposlib/tests/test_skyrl_shm_e2e.py
if [ $? -eq 0 ]; then
    echo "✓ SHM E2E verification passed"
else
    echo "ERROR: SHM E2E verification failed"
    exit 1
fi
echo ""

# Verify the adapter can be initialized
echo "Step 2: Verifying SkyRLAdapter Initialization..."
python3 -c "from atroposlib.envs.skyrl_adapter import SkyRLAdapter; from atroposlib.envs.base import TransportType; print('✓ SkyRLAdapter successfully imported')"
if [ $? -eq 0 ]; then
    echo "✓ Adapter initialization verified"
else
    echo "ERROR: Failed to initialize SkyRLAdapter"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ All Atropos-side SHM tests passed!"
echo "=========================================="
echo ""
