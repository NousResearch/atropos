#!/usr/bin/env python3
"""
Gradual test to identify where FactorioInstance connections break.
Start with 1 container, then gradually increase.
"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fle"))

import fle
from fle.env import FactorioInstance

def test_single_connection(port):
    """Test a single connection to a container."""
    print(f"\nTesting connection to port {port}...")
    start = time.time()
    
    try:
        # Simple connection like our working test
        instance = FactorioInstance(
            address='localhost',
            tcp_port=port,
            fast=True,
            num_agents=1
        )
        elapsed = time.time() - start
        print(f"✅ Connected to port {port} in {elapsed:.2f}s")
        
        # Clean up
        del instance
        return True
        
    except Exception as e:
        elapsed = time.time() - start
        print(f"❌ Failed to connect to port {port} after {elapsed:.2f}s: {e}")
        return False


def main():
    """Test connections to N containers."""
    # Get number of containers to test from command line
    if len(sys.argv) > 1:
        num_containers = int(sys.argv[1])
    else:
        num_containers = 1
    
    print(f"Testing connections to {num_containers} container(s)")
    print("=" * 50)
    
    successful = 0
    failed = 0
    
    # Test each container sequentially
    for i in range(num_containers):
        port = 27000 + i
        if test_single_connection(port):
            successful += 1
        else:
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Results: {successful} successful, {failed} failed out of {num_containers}")
    
    if successful == num_containers:
        print("✅ All connections successful!")
        return 0
    else:
        print(f"⚠️  {failed} connections failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())