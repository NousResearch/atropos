#!/usr/bin/env python3
"""
Test script to verify multiple Factorio containers are accessible.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fle"))

import fle
from fle.env import FactorioInstance
import time

def test_container(port):
    """Test connection to a single container."""
    print(f"\nTesting container on port {port}...")
    try:
        instance = FactorioInstance(
            address="localhost",
            tcp_port=port,
            fast=True,
            num_agents=1
        )
        print(f"✅ Successfully connected to container on port {port}")
        # FactorioInstance doesn't have a close method
        del instance
        return True
    except Exception as e:
        print(f"❌ Failed to connect to container on port {port}: {e}")
        return False

def main():
    print("Testing Factorio multi-container setup...")
    print("=" * 50)
    
    # Test ports 27000-27007 (8 containers)
    ports = list(range(27000, 27008))
    results = []
    
    for port in ports:
        success = test_container(port)
        results.append((port, success))
        time.sleep(0.5)  # Small delay between connections
    
    print("\n" + "=" * 50)
    print("Summary:")
    successful = sum(1 for _, success in results if success)
    print(f"Connected to {successful}/{len(ports)} containers")
    
    if successful == len(ports):
        print("✅ All containers are accessible!")
    elif successful > 0:
        print("⚠️  Some containers are not accessible. Check if they're all running.")
    else:
        print("❌ No containers are accessible. Start them with:")
        print("   sudo docker run -d --name factorio_0 -p 27000:27015 -p 34197:34197/udp \\")
        print("     -v /home/maxpaperclips/atropos/environments/game_environments/factorio_env/fle/fle/cluster/scenarios:/opt/factorio/scenarios \\")
        print("     factorio default_lab_scenario")
        print("   (repeat for ports 27001-27007)")
    
    return 0 if successful == len(ports) else 1

if __name__ == "__main__":
    sys.exit(main())