#!/usr/bin/env python3
"""
Test parallel connections to multiple Factorio containers.
This simulates what happens in the actual training environment.
"""

import sys
import os
import time
import asyncio
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fle"))

import fle
from fle.env import FactorioInstance


async def connect_async(port):
    """Connect to a container asynchronously."""
    print(f"[{time.time():.2f}] Starting connection to port {port}...")
    start = time.time()
    
    try:
        # Create instance directly (no executor)
        instance = FactorioInstance(
            address='localhost',
            tcp_port=port,
            fast=True,
            num_agents=1
        )
        elapsed = time.time() - start
        print(f"[{time.time():.2f}] ✅ Connected to port {port} in {elapsed:.2f}s")
        
        # Clean up
        del instance
        return True
        
    except Exception as e:
        elapsed = time.time() - start
        print(f"[{time.time():.2f}] ❌ Failed port {port} after {elapsed:.2f}s: {e}")
        return False


async def test_parallel(num_containers):
    """Test parallel connections to N containers."""
    print(f"\nTesting PARALLEL connections to {num_containers} containers")
    print("=" * 50)
    
    start = time.time()
    
    # Create tasks for parallel connection
    tasks = []
    for i in range(num_containers):
        port = 27000 + i
        tasks.append(connect_async(port))
    
    # Run all connections in parallel
    print(f"[{time.time():.2f}] Starting {num_containers} parallel connections...")
    results = await asyncio.gather(*tasks)
    
    elapsed = time.time() - start
    successful = sum(1 for r in results if r)
    
    print(f"\n[{time.time():.2f}] Completed in {elapsed:.2f}s")
    print(f"Results: {successful} successful, {num_containers - successful} failed")
    
    return successful == num_containers


def test_sequential(num_containers):
    """Test sequential connections for comparison."""
    print(f"\nTesting SEQUENTIAL connections to {num_containers} containers")
    print("=" * 50)
    
    start = time.time()
    successful = 0
    
    for i in range(num_containers):
        port = 27000 + i
        print(f"[{time.time():.2f}] Connecting to port {port}...")
        
        try:
            instance = FactorioInstance(
                address='localhost',
                tcp_port=port,
                fast=True,
                num_agents=1
            )
            print(f"[{time.time():.2f}] ✅ Connected to port {port}")
            del instance
            successful += 1
        except Exception as e:
            print(f"[{time.time():.2f}] ❌ Failed port {port}: {e}")
    
    elapsed = time.time() - start
    print(f"\n[{time.time():.2f}] Completed in {elapsed:.2f}s")
    print(f"Results: {successful} successful, {num_containers - successful} failed")
    
    return successful == num_containers


async def main():
    """Test both sequential and parallel connections."""
    # Get number of containers from command line
    if len(sys.argv) > 1:
        num_containers = int(sys.argv[1])
    else:
        num_containers = 4
    
    print(f"Testing connections to {num_containers} container(s)")
    print(f"Start time: {time.time():.2f}")
    
    # Test sequential first (known to work)
    seq_success = test_sequential(num_containers)
    
    # Test parallel
    par_success = await test_parallel(num_containers)
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Sequential: {'✅ SUCCESS' if seq_success else '❌ FAILED'}")
    print(f"Parallel:   {'✅ SUCCESS' if par_success else '❌ FAILED'}")
    
    if not par_success and seq_success:
        print("\n⚠️  Parallel connections fail while sequential works!")
        print("This is the root cause of the training environment hanging.")
        return 1
    elif par_success and seq_success:
        print("\n✅ Both sequential and parallel connections work!")
        return 0
    else:
        print("\n❌ Connection issues detected")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))