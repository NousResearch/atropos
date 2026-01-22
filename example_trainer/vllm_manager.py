"""
vLLM process management for GRPO trainer.

Handles launching, monitoring, and terminating vLLM server processes
for legacy mode training.
"""

import atexit
import os
import subprocess
import time
from typing import Optional

import requests

from .config import TrainingConfig


# Global variable to keep track of the vLLM process
_vllm_process: Optional[subprocess.Popen] = None


def cleanup_vllm():
    """Cleanup function to terminate vLLM on exit."""
    global _vllm_process
    if _vllm_process:
        print("\nTerminating vLLM process...")
        _vllm_process.terminate()
        try:
            _vllm_process.wait(timeout=5)
            print("vLLM process terminated.")
        except subprocess.TimeoutExpired:
            print("vLLM process did not terminate gracefully, killing.")
            _vllm_process.kill()
            _vllm_process.wait()
            print("vLLM process killed.")
        _vllm_process = None


# Register cleanup on module load
atexit.register(cleanup_vllm)


def launch_vllm_server(
    config: TrainingConfig,
    model_path: str,
) -> Optional[subprocess.Popen]:
    """
    Launch a vLLM server process using our custom vllm_api_server.py.

    Uses the custom server instead of standard vLLM because:
    - Standard vLLM only has /v1/completions (OpenAI-compatible)
    - Our custom server has /generate endpoint needed by VLLMServer class
    - This allows proper tokens_and_logprobs_completion support

    Args:
        config: Training configuration
        model_path: Path to model checkpoint

    Returns:
        Popen process object, or None if launch failed
    """
    global _vllm_process

    # Use our custom vllm_api_server.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    custom_server_path = os.path.join(script_dir, "vllm_api_server.py")

    vllm_command = [
        "python",
        custom_server_path,
        "--model",
        model_path,
        "--port",
        str(config.vllm_port),
        "--gpu-memory-utilization",
        str(config.vllm_gpu_memory_utilization),
    ]
    
    # Add served-model-name if using checkpoint path
    if model_path != config.model_name:
        vllm_command.extend(["--served-model-name", config.model_name])

    print(f"  Launching vLLM: {' '.join(vllm_command)}")

    try:
        proc = subprocess.Popen(vllm_command)
        print(f"  vLLM launched with PID: {proc.pid}")

        # Check for immediate startup errors
        try:
            proc.communicate(timeout=2)
            if proc.returncode is not None and proc.returncode != 0:
                print("  WARNING: vLLM failed to start")
                return None
        except subprocess.TimeoutExpired:
            print("  vLLM process started (check logs for details)")

        _vllm_process = proc
        return proc

    except FileNotFoundError:
        print("  ERROR: vLLM not found. Is it installed?")
        return None
    except Exception as e:
        print(f"  ERROR launching vLLM: {e}")
        return None


def terminate_vllm_process() -> None:
    """Terminate the running vLLM process if any."""
    global _vllm_process

    if _vllm_process is None:
        return

    print("  Terminating vLLM process...")
    _vllm_process.terminate()
    try:
        _vllm_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        print("  vLLM did not terminate gracefully, killing...")
        _vllm_process.kill()
        _vllm_process.wait()
    _vllm_process = None


def check_vllm_process_health() -> None:
    """Check if vLLM process terminated unexpectedly."""
    global _vllm_process

    if _vllm_process is not None and _vllm_process.poll() is not None:
        print(f"  WARNING: vLLM terminated unexpectedly (code: {_vllm_process.returncode})")
        _vllm_process = None


def get_vllm_process() -> Optional[subprocess.Popen]:
    """Get the current vLLM process."""
    return _vllm_process


def set_vllm_process(proc: Optional[subprocess.Popen]) -> None:
    """Set the vLLM process (for external management)."""
    global _vllm_process
    _vllm_process = proc


def check_vllm_health(port: int) -> bool:
    """
    Check if vLLM server is healthy and responding.

    Args:
        port: Port the vLLM server is running on

    Returns:
        True if server is healthy
    """
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def wait_for_vllm_ready(port: int, timeout: float = 120.0) -> bool:
    """
    Wait for vLLM server to be ready.

    Args:
        port: Port the vLLM server is running on
        timeout: Maximum time to wait in seconds

    Returns:
        True if server is ready, False if timeout
    """
    print(f"  Waiting for vLLM to be ready (port {port})...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        if check_vllm_health(port):
            print("  vLLM is ready!")
            return True
        time.sleep(2)

    print(f"  WARNING: vLLM not ready after {timeout}s")
    return False


def hotswap_lora_adapter(
    adapter_name: str,
    adapter_path: str,
    port: int,
) -> bool:
    """
    Hot-swap a LoRA adapter on a running vLLM server.

    Uses the vLLM /v1/load_lora_adapter endpoint to load a new adapter
    without restarting the server.

    Args:
        adapter_name: Name to identify the adapter
        adapter_path: Path to the adapter checkpoint
        port: vLLM server port

    Returns:
        True if hot-swap succeeded
    """
    try:
        # Use vLLM's native LoRA loading endpoint
        response = requests.post(
            f"http://localhost:{port}/v1/load_lora_adapter",
            json={
                "lora_name": adapter_name,
                "lora_path": adapter_path,
            },
            timeout=30,
        )

        if response.status_code == 200:
            print(f"  [LORA] ✓ Hot-swapped adapter: {adapter_name} ({adapter_path})")
            return True
        else:
            print(f"  [LORA] ✗ Hot-swap failed: {response.status_code} - {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"  [LORA] ✗ Cannot connect to vLLM at port {port}")
        return False
    except Exception as e:
        print(f"  [LORA] ✗ Error during hot-swap: {e}")
        return False

