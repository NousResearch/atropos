"""
vLLM process management for GRPO trainer.

Handles launching, monitoring, and terminating vLLM server processes
for legacy mode training.
"""

import atexit
import os
import signal
import socket
import subprocess
import time
from typing import Optional

import requests

from .config import TrainingConfig

# Global variable to keep track of the vLLM process
_vllm_process: Optional[subprocess.Popen] = None


def is_port_in_use(port: int) -> bool:
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


# Substrings we expect to see in /proc/<pid>/cmdline for a process Atropos
# is allowed to terminate. Anything else listening on the requested port is
# treated as a foreign process — Atropos refuses to kill it. Keeps the
# manager safe on shared multi-tenant clusters where another tenant might
# happen to bind the same port. See issue #460.
_OWNED_PROCESS_KEYWORDS = ("vllm", "atropos", "torchrun", "python")


def _read_proc_cmdline(pid: int) -> str:
    """Return /proc/<pid>/cmdline joined on spaces, or '' on any failure.

    Returns '' for non-Linux platforms, processes that disappeared between
    discovery and inspection, or permission errors.
    """
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read()
    except (FileNotFoundError, PermissionError, IsADirectoryError, OSError):
        return ""
    return raw.replace(b"\x00", b" ").decode(errors="replace").strip()


def _is_atropos_owned(pid: int) -> bool:
    """Return True iff `pid`'s cmdline contains one of the owned keywords.

    On non-Linux (no /proc) returns False — better to fail safe and skip the
    kill than to terminate a process Atropos cannot identify.
    """
    cmdline = _read_proc_cmdline(pid).lower()
    if not cmdline:
        return False
    return any(kw in cmdline for kw in _OWNED_PROCESS_KEYWORDS)


def kill_process_on_port(port: int, timeout: float = 5.0) -> bool:
    """
    Kill any **Atropos-owned** process using the specified port.

    A process is considered Atropos-owned only if its `/proc/<pid>/cmdline`
    contains one of the keywords in `_OWNED_PROCESS_KEYWORDS`. If a
    different process (e.g. a database, an SSH session, a monitoring agent)
    happens to be on the port, this function refuses to kill it and
    surfaces a port-collision warning instead.

    Returns True if no process was running, if all owned processes were
    successfully killed, or if the port was freed by other means. Returns
    False if the port is still bound by either a stubborn owned process or
    a foreign process.
    """
    if not is_port_in_use(port):
        return True

    print(f"  Port {port} is in use, attempting to kill existing process...")

    try:
        # Try to find and kill the process using lsof (Linux/Mac)
        result = subprocess.run(
            ["lsof", "-t", "-i", f":{port}"], capture_output=True, text=True, timeout=5
        )
        if result.stdout.strip():
            raw_pids = result.stdout.strip().split("\n")
            owned_pids: list[str] = []
            foreign_pids: list[str] = []
            for raw in raw_pids:
                try:
                    pid_int = int(raw)
                except ValueError:
                    continue
                if _is_atropos_owned(pid_int):
                    owned_pids.append(raw)
                else:
                    foreign_pids.append(raw)

            if foreign_pids:
                cmds = ", ".join(
                    f"pid={p} cmdline='{_read_proc_cmdline(int(p))[:80]}'"
                    for p in foreign_pids
                )
                print(
                    f"  REFUSING to kill {len(foreign_pids)} foreign process(es) on "
                    f"port {port}: {cmds}"
                )
                print(
                    f"  Atropos only kills processes whose cmdline contains one of "
                    f"{_OWNED_PROCESS_KEYWORDS}. Free the port manually and retry."
                )

            if not owned_pids:
                # Nothing we own; do not touch foreign processes.
                return False

            print(f"  Killing {len(owned_pids)} Atropos-owned processes on port {port}...")
            for pid in owned_pids:
                try:
                    os.kill(int(pid), signal.SIGTERM)
                except (ProcessLookupError, ValueError):
                    pass

            # Wait for port to be free
            start = time.time()
            while time.time() - start < timeout:
                if not is_port_in_use(port):
                    print(f"  Port {port} is now free")
                    return True
                time.sleep(0.5)

            # Force kill if still running — owned PIDs only
            killed_count = 0
            for pid in owned_pids:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                    killed_count += 1
                except (ProcessLookupError, ValueError):
                    pass
            if killed_count > 0:
                print(f"  Force killed {killed_count} stubborn processes")

            time.sleep(1)
            return not is_port_in_use(port)
    except FileNotFoundError:
        # lsof not available — `fuser -k` is even more dangerous (it kills
        # ANY process on the port) so skip it. The user can free the port
        # manually; we do not silently terminate processes we cannot
        # identify.
        print(
            f"  WARNING: lsof not available; cannot identify processes on port {port}. "
            f"Refusing to fall back to `fuser -k` because it would kill foreign "
            f"processes too. Free the port manually and retry."
        )
        return False
    except subprocess.TimeoutExpired:
        pass

    print(f"  WARNING: Could not kill process on port {port}")
    return False


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
    - Streamlined API: Only /generate endpoint (provides logprobs)
    - Weight bridge support: /bridge/* endpoints for shared memory mode
    - LoRA hot-swap: /lora/* endpoints for adapter loading/unloading

    Args:
        config: Training configuration
        model_path: Path to model checkpoint

    Returns:
        Popen process object, or None if launch failed
    """
    global _vllm_process

    # Check if port is in use and try to kill existing process
    if is_port_in_use(config.vllm_port):
        print(f"  WARNING: Port {config.vllm_port} is already in use!")
        if not kill_process_on_port(config.vllm_port):
            print(
                f"  ERROR: Could not free port {config.vllm_port}. Please manually kill the process."
            )
            print(f"    Try: lsof -i :{config.vllm_port} | grep LISTEN")
            print(f"    Or:  pkill -f 'vllm.*{config.vllm_port}'")
            return None
        print(f"  Successfully freed port {config.vllm_port}")

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
        print(
            f"  WARNING: vLLM terminated unexpectedly (code: {_vllm_process.returncode})"
        )
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
            print(
                f"  [LORA] ✗ Hot-swap failed: {response.status_code} - {response.text}"
            )
            return False

    except requests.exceptions.ConnectionError:
        print(f"  [LORA] ✗ Cannot connect to vLLM at port {port}")
        return False
    except Exception as e:
        print(f"  [LORA] ✗ Error during hot-swap: {e}")
        return False
