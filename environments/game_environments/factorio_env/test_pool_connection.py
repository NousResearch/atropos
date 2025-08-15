#!/usr/bin/env python3
"""
Test script to simulate container pool connection with async executor.
This mimics how factorio_env_minimal.py connects to containers.
"""

import asyncio
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fle"))

import fle
from fle.env import FactorioInstance


async def connect_simple_async(port):
    """Connect to container directly in async context (no executor)."""
    print(f"[{time.time():.2f}] Attempting connection to port {port} WITHOUT executor...")
    
    try:
        # Direct creation without executor - much simpler!
        instance = FactorioInstance(
            address="localhost",
            tcp_port=port,
            fast=True,
            num_agents=1
        )
        print(f"[{time.time():.2f}] ✅ Connected to port {port} directly in async")
        del instance
        return True
    except Exception as e:
        print(f"[{time.time():.2f}] ❌ Failed on port {port} directly in async: {e}")
        return False


def connect_direct(port):
    """Connect to container directly (like test scripts do)."""
    print(f"[{time.time():.2f}] Attempting connection to port {port} DIRECTLY...")
    
    try:
        instance = FactorioInstance(
            address="localhost",
            tcp_port=port,
            fast=True,
            num_agents=1
        )
        print(f"[{time.time():.2f}] ✅ Connected to port {port} directly")
        del instance
        return True
    except Exception as e:
        print(f"[{time.time():.2f}] ❌ Failed on port {port} directly: {e}")
        return False


async def test_parallel_connections():
    """Test connecting to multiple containers in parallel WITHOUT executor."""
    print("\n" + "="*60)
    print("Testing PARALLEL connections WITHOUT executor (simplified)...")
    print("="*60)
    
    ports = list(range(27000, 27016))  # Test first 16 containers
    start = time.time()
    
    # Create tasks for parallel connection (no executor)
    tasks = [connect_simple_async(port) for port in ports]
    results = await asyncio.gather(*tasks)
    
    elapsed = time.time() - start
    successful = sum(1 for r in results if r)
    print(f"\nParallel without executor: {successful}/{len(ports)} successful in {elapsed:.2f}s")
    
    return successful


def test_sequential_connections():
    """Test connecting to containers sequentially."""
    print("\n" + "="*60)
    print("Testing SEQUENTIAL direct connections (like test scripts)...")
    print("="*60)
    
    ports = list(range(27000, 27016))  # Test first 16 containers
    start = time.time()
    successful = 0
    
    for port in ports:
        if connect_direct(port):
            successful += 1
    
    elapsed = time.time() - start
    print(f"\nSequential direct: {successful}/{len(ports)} successful in {elapsed:.2f}s")
    
    return successful


async def test_mixed_approach():
    """Test a mixed approach - direct creation but in async context."""
    print("\n" + "="*60)
    print("Testing MIXED approach (direct in async)...")
    print("="*60)
    
    async def connect_mixed(port):
        print(f"[{time.time():.2f}] Mixed approach for port {port}...")
        try:
            # Direct creation without executor
            instance = FactorioInstance(
                address="localhost",
                tcp_port=port,
                fast=True,
                num_agents=1
            )
            print(f"[{time.time():.2f}] ✅ Mixed connected to port {port}")
            del instance
            return True
        except Exception as e:
            print(f"[{time.time():.2f}] ❌ Mixed failed on port {port}: {e}")
            return False
    
    ports = list(range(27000, 27016))
    start = time.time()
    
    tasks = [connect_mixed(port) for port in ports]
    results = await asyncio.gather(*tasks)
    
    elapsed = time.time() - start
    successful = sum(1 for r in results if r)
    print(f"\nMixed approach: {successful}/{len(ports)} successful in {elapsed:.2f}s")
    
    return successful


async def main():
    print("Container Pool Connection Test")
    print("==============================")
    print("This tests different connection methods to the Factorio container pool.")
    print(f"Start time: {time.time():.2f}")
    
    # Test sequential first (known to work)
    seq_success = test_sequential_connections()
    
    if seq_success == 0:
        print("\n❌ No containers accessible! Make sure they're running.")
        return 1
    
    # Test parallel WITHOUT executor (simplified approach)
    par_success = await test_parallel_connections()
    
    # Test mixed approach
    mix_success = await test_mixed_approach()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Sequential direct: {seq_success} successful")
    print(f"Parallel without executor: {par_success} successful")
    print(f"Mixed approach: {mix_success} successful")
    
    if par_success == seq_success:
        print("\n✅ Simplified parallel approach works perfectly!")
        print("The async executor was the problem.")
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))