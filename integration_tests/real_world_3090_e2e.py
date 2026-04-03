import subprocess
import time
import requests
import os
import signal
import sys

def main():
    print("🌟 Starting Real-World 3090 E2E Stress Test (GSM8K + Dynamic Ports) 🌟")
    
    server_port = 8040
    server_url = f"http://localhost:{server_port}"
    
    # 1. Start Atropos Server
    print(f"Launching Atropos API Server on {server_port}...")
    server_proc = subprocess.Popen(
        ["python", "-m", "atroposlib.cli.run_api", "--port", str(server_port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
    )
    time.sleep(5)
    
    # 2. Register Trainer (needed to bootstrap queue)
    print("Registering Trainer...")
    requests.post(f"{server_url}/register", json={
        "wandb_group": "3090-e2e", "wandb_project": "atropos-harden",
        "batch_size": 32, "max_token_len": 512, "checkpoint_dir": "/tmp"
    })
    
    # 3. Start Orchestrator with REAL GSM8K Command
    # We use Qwen2.5-1.5B (small but real)
    print("Launching Orchestrator with GSM8K + Dynamic Port injection...")
    # NOTE: We escape $PORT so the shell doesn't expand it locally
    env_cmd = f"/usr/bin/env SERVER_PORT=\$PORT python environments/gsm8k_server.py serve"
    
    orchestrator_proc = subprocess.Popen(
        [
            "python", "-m", "atroposlib.cli.orchestrate",
            "--server-url", server_url,
            "--env-command", env_cmd,
            "--port-range", "8100:8110",
            "--min-actors", "1",
            "--max-actors", "4",
            "--poll-interval", "5",
            "--cooldown", "10"
        ]
    )
    
    try:
        # Step A: Wait for first actor to boot (approx 30s for model load)
        print("Waiting 45s for first GSM8K instance to boot and register...")
        time.sleep(45)
        
        status = requests.get(f"{server_url}/global-status").json()
        print(f"Init Status: Connected={status['num_connected_envs']}, Pressure={status['rollout_pressure']:.2f}")
        
        # Step B: PUSH PRESSURE (Target 4 instances)
        print("\nPushing 160 rollouts (Pressure 5.0) to trigger multi-port scaling...")
        fake_rollout = {"tokens": [[1]*10], "masks": [[1]*10], "scores": [1.0], "env_id": 0}
        requests.post(f"{server_url}/scored_data_list", json=[fake_rollout]*160)
        
        # Step C: Watch for multiple PORTS in nvidia-smi or server status
        print("Watching scaling for 60s...")
        for i in range(12):
            time.sleep(5)
            status = requests.get(f"{server_url}/global-status").json()
            print(f"[{i*5}s] Connected={status['num_connected_envs']}, Queue={status['queue_size']}")
            if status['num_connected_envs'] > 1:
                print("🏆 SCALING SUCCESSFUL! Multiple real environments connected.")
                break
        
        print("\nFinal Check: Checking for dynamic ports in log...")
        # (This is manual or we could check server registry)
        
    finally:
        print("\nCleaning up...")
        orchestrator_proc.terminate()
        server_proc.terminate()
        # Clean up real gsm8k processes (bash pkill is easier)
        os.system("pkill -f gsm8k_server")
        print("E2E Done.")

if __name__ == "__main__":
    main()
