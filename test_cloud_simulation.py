"""
Test script for cloud-deployed society simulation
"""

import json
import time

import requests

# Your deployed Cloud Run service URL
BASE_URL = "https://society-simulation-643533604146.us-central1.run.app"


def test_cloud_simulation():
    """Test the cloud simulation API"""

    print("🧪 Testing Cloud Society Simulation")
    print("=" * 50)

    # Test 1: Health check
    print("\n1. 🏥 Health Check...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
        return

    # Test 2: Start simulation
    print("\n2. 🚀 Starting Simulation...")
    sim_params = {"agents": 100, "steps": 20, "workers": 4}

    try:
        response = requests.post(f"{BASE_URL}/simulation/run", json=sim_params)

        if response.status_code == 200:
            sim_data = response.json()
            sim_id = sim_data["simulation_id"]
            print(f"   ✅ Simulation started: {sim_id}")
            print(f"   Parameters: {sim_data['parameters']}")

            # Test 3: Monitor simulation
            print(f"\n3. 📊 Monitoring Simulation {sim_id}...")

            max_checks = 60  # Check for up to 5 minutes
            for i in range(max_checks):
                try:
                    status_response = requests.get(
                        f"{BASE_URL}/simulation/status/{sim_id}"
                    )
                    if status_response.status_code == 200:
                        status = status_response.json()
                        print(f"   Check {i+1}: {status['status']}")

                        if status["status"] == "completed":
                            print("   ✅ Simulation completed!")
                            if "duration" in status:
                                print(f"   Duration: {status['duration']:.2f} seconds")
                            if "results_url" in status:
                                print(f"   Results: {status['results_url']}")
                            break
                        elif status["status"] == "failed":
                            print(
                                f"   ❌ Simulation failed: {status.get('error', 'Unknown error')}"
                            )
                            break
                        elif status["status"] == "timeout":
                            print("   ⏰ Simulation timed out")
                            break

                    time.sleep(5)  # Wait 5 seconds between checks

                except Exception as e:
                    print(f"   ❌ Status check failed: {e}")
                    break
            else:
                print("   ⚠️ Simulation monitoring timed out")

        else:
            print(f"   ❌ Failed to start simulation: {response.status_code}")
            print(f"   Error: {response.text}")

    except Exception as e:
        print(f"   ❌ Simulation request failed: {e}")

    # Test 4: List all simulations
    print("\n4. 📝 Listing All Simulations...")
    try:
        response = requests.get(f"{BASE_URL}/simulation/list")
        if response.status_code == 200:
            data = response.json()
            print(f"   Total simulations: {data['total']}")
            for sim_id, sim_info in list(data["simulations"].items())[-3:]:
                print(f"   {sim_id}: {sim_info.get('status', 'unknown')}")
        else:
            print(f"   ❌ Failed to list simulations: {response.status_code}")
    except Exception as e:
        print(f"   ❌ List request failed: {e}")

    # Test 5: Benchmark
    print("\n5. 🏃 Running Benchmark...")
    try:
        response = requests.post(f"{BASE_URL}/simulation/benchmark")
        if response.status_code == 200:
            data = response.json()
            print("   ✅ Benchmark completed")
            print(f"   Status: {data['status']}")
            # Parse benchmark results
            output = data.get("output", "")
            if "Benchmark Results:" in output:
                benchmark_section = output.split("Benchmark Results:")[1]
                print("   Results:")
                for line in benchmark_section.split("\n")[:10]:
                    if line.strip():
                        print(f"     {line.strip()}")
        else:
            print(f"   ❌ Benchmark failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Benchmark request failed: {e}")

    print("\n🎯 Cloud Testing Complete!")
    print(f"\n🌐 Your Society Simulation is running at: {BASE_URL}")
    print("You can now run simulations with thousands of agents in the cloud!")


if __name__ == "__main__":
    # Check if URL is set
    if "xxxxx" in BASE_URL:
        print("⚠️  Please update BASE_URL in this script with your actual Cloud Run URL")
        print("   You'll get this URL after running: ./deploy_society_cloud.sh")
    else:
        test_cloud_simulation()
