#!/usr/bin/env python3
"""
Local Blackjack training launcher.

Usage:
    python -m environments.game_environments.gymnasium.launch_local_blackjack_run

This script does:
  1) Starts the Trajectory Handler API server via uvicorn
  2) Launches the BlackjackEnv in local serve mode using blackjack_local config
  3) Imports and runs the example trainer (GRPO) directly

Requirements:
  - Run from project root so example_trainer is on PYTHONPATH
  - example_trainer/ is a valid Python package (with __init__.py)
"""
import atexit
import os
import signal
import subprocess
import sys
import time
import traceback

# Ensure project root is on PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import trainer via standard module import
try:
    from example_trainer.grpo import TrainingConfig, train
except ImportError as e:
    print(f"Error importing example_trainer.grpo: {e}")
    print(
        "Ensure you're running from project root and that example_trainer/ is a package."
    )
    sys.exit(1)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
API_HOST = "127.0.0.1"
API_PORT = 8000

# Trainer assumes the BlackjackEnv server handles LLM connection based on its config
# We don't need VLLM host/port here unless trainer *also* needs it directly.

# Define model/tokenizer - should ideally match what blackjack_local.yaml points to
# if consistency is needed, but trainer might use a different one for reference.
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct" # Placeholder, adjust if needed
TOKENIZER_NAME = MODEL_NAME

TRAINER_CONFIG = {
    "model_name": MODEL_NAME,
    "training_steps": 20,       # Short run for local test
    "batch_size": 2,            # Small batch size
    "gradient_accumulation_steps": 2,
    "seq_len": 512,             # Blackjack interactions are short
    "use_wandb": False,
    "wandb_project": "",
    "wandb_group": "",
    "save_path": "./trained_blackjack_model_local_test", # Distinct save path
    # vllm specific args removed, assuming env server handles inference connection
}

# Flags for launching BlackjackEnv serve
# We primarily need to tell it which config file to use
BLACKJACK_ENV_SCRIPT = "environments/game_environments/gymnasium/blackjack_env.py"
BLACKJACK_CONFIG_NAME = "blackjack_local" # Tells env to load blackjack_local.yaml

# Track background processes for cleanup
processes = []


def cleanup_processes():
    print("\nCleaning up background processes...")
    # Terminate in reverse order of creation
    for p in reversed(processes):
        if p.poll() is None: # Check if process is still running
            print(f"Terminating PID {p.pid}...")
            # Send SIGTERM first for graceful shutdown
            p.terminate()
            try:
                # Wait a bit for graceful termination
                p.wait(timeout=5)
                print(f"PID {p.pid} terminated gracefully.")
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't terminate
                print(f"PID {p.pid} did not terminate gracefully; killing.")
                p.kill()
                p.wait() # Wait for kill confirmation
                print(f"PID {p.pid} killed.")
        else:
            print(f"PID {p.pid} already exited with code {p.poll()}.")
    print("Cleanup complete.")


# Register cleanup function to run on exit
atexit.register(cleanup_processes)


def handle_signal(sig, frame):
    print(f"\nSignal {sig} received; initiating cleanup and exit.")
    # atexit handler will run automatically
    sys.exit(0)


# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, handle_signal)  # Ctrl+C
signal.signal(signal.SIGTERM, handle_signal) # Kill command


def main():
    # 1) Start the Trajectory Handler API Server
    print("--- Starting Trajectory Handler API Server ---")
    # Use uv run if available
    api_cmd = [
        "uv", "run", "uvicorn",
        "atroposlib.api.server:app",
        "--host", API_HOST,
        "--port", str(API_PORT),
    ]
    print(f"$ {' '.join(api_cmd)}")
    # Start process, ensuring std streams are handled if needed for debug
    api_proc = subprocess.Popen(api_cmd, stdout=sys.stdout, stderr=sys.stderr)
    processes.append(api_proc)
    print(f"API server started with PID {api_proc.pid}. Waiting briefly...")
    time.sleep(5) # Increased wait time for server startup

    # Check if API server started correctly
    if api_proc.poll() is not None:
         print(f"API server failed to start (exit code {api_proc.poll()}). Exiting.")
         sys.exit(1)
    print("API server appears to be running.")


    # 2) Start the Blackjack Environment Server
    print("\n--- Starting Blackjack Environment Server ---")
    env_cmd = [
        "uv", "run", "python",
        BLACKJACK_ENV_SCRIPT,
        "serve",
        "--config", BLACKJACK_CONFIG_NAME,
        # Add other BASE flags if needed and NOT handled by config, e.g.:
        # "--max_num_workers", "2",
        "--rollout_server_url", f"http://{API_HOST}:{API_PORT}", # Essential
        # Flags like model_name, base_url, rewards etc. should come from blackjack_local.yaml
    ]
    print(f"$ {' '.join(env_cmd)}")
    env_proc = subprocess.Popen(env_cmd, stdout=sys.stdout, stderr=sys.stderr)
    processes.append(env_proc)
    print(f"Blackjack Env server started with PID {env_proc.pid}. Waiting briefly...")
    time.sleep(5) # Increased wait time

    # Check if env server started correctly
    if env_proc.poll() is not None:
         print(f"Blackjack Env server failed to start (exit code {env_proc.poll()}). Exiting.")
         sys.exit(1)
    print("Blackjack Env server appears to be running.")


    # 3) Run the Example Trainer (GRPO)
    print("\n--- Running Example Trainer (GRPO) ---")
    try:
        config = TrainingConfig(**TRAINER_CONFIG)
        print(f"Trainer Config: {config}")
        train(config)
        print("--- Training run finished successfully (or ended naturally) ---")
    except Exception:
        print("--- Error during training run: ---")
        traceback.print_exc()
        print("---------------------------------")
    finally:
        # Explicitly call cleanup, though atexit should also handle it
        print("Training finished or errored. Initiating cleanup.")
        # Cleanup is handled by atexit


if __name__ == "__main__":
    main() 