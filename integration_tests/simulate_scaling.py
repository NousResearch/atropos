import subprocess
import time
import requests
import os
import signal
import sys

def main():
    print("🚀 Starting Atropos Elasticity Simulation...")
    
    # 1. Start Atropos Server
    server_process = subprocess.Popen(
        ["run-api", "--port", "8008"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT
    )
    time.sleep(3)
    server_url = "http://localhost:8008"
    
    # 2. Register Trainer
    # This must happen before /register-env will work on server
    print("Registering Trainer (BatchSize=32)...")
    requests.post(f"{server_url}/register", json={
        "wandb_group": "sim",
        "wandb_project": "sim",
        "batch_size": 32,
        "max_token_len": 512,
        "checkpoint_dir": "/tmp",
        "save_checkpoint_interval": 10,
        "starting_step": 0,
        "num_steps": 100
    })
    
    # 3. Start Orchestrator
    print("Starting Orchestrator (Monitoring RP=1.0)...")
    # Command for dummy actor: "python integration_tests/dummy_actor.py --server http://localhost:8008"
    orchestrator_process = subprocess.Popen(
        [
            "python", "-m", "atroposlib.cli.orchestrate",
            "--server-url", server_url,
            "--env-command", f"python integration_tests/dummy_actor.py --server {server_url}",
            "--min-actors", "1",
            "--max-actors", "10",
            "--poll-interval", "5",
            "--cooldown", "10" # Short cooldown for simulation
        ],
        cwd=".",
        # Use shell if needed for complex commands, but list is safer
    )
    
    time.sleep(10)
    
    def get_status():
        try:
            return requests.get(f"{server_url}/global-status").json()
        except:
            return None

    try:
        # Pass 1: "Quiet Mode"
        print("\n--- TEST 1: Baseline Scaling (Goal: 1 Actor) ---")
        status = get_status()
        print(f"Queue Size: {status['queue_size']}, Actors: {status['num_connected_envs']}")
        
        # Pass 2: "Burst Mode"
        print("\n--- TEST 2: Burst Scaling (Goal: Target RP > 1) ---")
        # RP = (Queue / BatchSize). To get RP=5 with batch_size=32, we need 160 rollouts.
        print("Pushing 160 dummy rollouts to increase Rollout Pressure...")
        fake_rollout = {
            "tokens": [[1]*10],
            "masks": [[1]*10],
            "scores": [1.0],
            "env_id": 0 # This might cause some warnings if not registered, but we just need it in queue
        }
        # scored_data_list is faster
        requests.post(f"{server_url}/scored_data_list", json=[fake_rollout]*160)
        
        print("Waiting for Orchestrator to react (approx 15s)...")
        for _ in range(4):
            time.sleep(5)
            status = get_status()
            print(f"Queue Size: {status['queue_size']}, Actors: {status['num_connected_envs']} (RP: {status['queue_size']/status['batch_size']:.2f})")
            if status['num_connected_envs'] > 1:
                print("✅ SCALING UP DETECTED.")
        
        # Pass 3: "Emptying Mode"
        print("\n--- TEST 3: Drain Scaling (Goal: Scale Back Down) ---")
        print("Cleaning server queue...")
        requests.get(f"{server_url}/reset_data")
        # Re-register trainer
        requests.post(f"{server_url}/register", json={
            "wandb_group": "sim", "wandb_project": "sim", "batch_size": 32, "max_token_len": 512, "checkpoint_dir": "/tmp", "save_checkpoint_interval": 10, "starting_step": 0, "num_steps": 100
        })

        print("Waiting for Orchestrator to react (approx 20s)...")
        for _ in range(5):
            time.sleep(5)
            status = get_status()
            print(f"Queue Size: {status['queue_size']}, Actors: {status['num_connected_envs']}")
            if status['num_connected_envs'] < 4:
                 print("✅ SCALING DOWN DETECTED.")

    finally:
        print("\nCleaning up simulation...")
        orchestrator_process.terminate()
        server_process.terminate()
        os.system("pkill -f dummy_actor") # Kill any orphans
        print("Done.")

if __name__ == "__main__":
    main()
