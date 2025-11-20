import logging
import os
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _wait_for_port(host: str, port: int, timeout: float = 600.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=5.0):
                return
        except OSError:
            time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for {host}:{port} to accept connections")


@dataclass
class WorkerHandle:
    protobus_address: str
    workspace_root: Path
    cline_src_dir: Path
    process: subprocess.Popen


class LocalWorkerManager:
    """Starts and stops local Cline workers via bootstrap_cline_worker.sh.

    For now this is specialized to the Rust/Ratatui example profile:
    - Clones/updates the NousResearch Cline fork.
    - Bootstraps the Ratatui workspace.
    - Starts the standalone gRPC server on a fixed port.
    """

    def __init__(self, protobus_port: int = 46040, hostbridge_port: int = 46041) -> None:
        self.protobus_port = protobus_port
        self.hostbridge_port = hostbridge_port

    def start_for_profile(self, profile: str) -> WorkerHandle:
        if profile == "rust_ratatui":
            return self._start_rust_ratatui_worker()
        raise ValueError(f"Unsupported worker profile: {profile}")

    def _start_rust_ratatui_worker(self) -> WorkerHandle:
        base_dir = Path(__file__).resolve().parent
        cline_dev_dir = base_dir / "cline_dev"
        bootstrap_script = cline_dev_dir / "bootstrap_cline_worker.sh"
        task_bootstrap_script = (
            cline_dev_dir / "examples" / "ratatui_vertical_gauge" / "bootstrap.sh"
        )

        if not bootstrap_script.exists():
            raise FileNotFoundError(f"bootstrap_cline_worker.sh not found at {bootstrap_script}")
        if not task_bootstrap_script.exists():
            raise FileNotFoundError(
                f"Ratatuí bootstrap script not found at {task_bootstrap_script}"
            )

        workspace_root = Path(os.getenv("CLINE_RUST_WORKSPACE", "/tmp/ratatui-workspace"))
        cline_src_dir = Path(os.getenv("CLINE_RUST_CLONE", "/tmp/nous-cline-worker"))
        workspace_root.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env.update(
            {
                "CLINE_SRC_DIR": str(cline_src_dir),
                "WORKSPACE_ROOT": str(workspace_root),
                "TASK_BOOTSTRAP_SCRIPT": str(task_bootstrap_script),
                "PROTOBUS_PORT": str(self.protobus_port),
                "HOSTBRIDGE_PORT": str(self.hostbridge_port),
            }
        )

        logger.info("Starting local Cline worker with profile rust_ratatui")
        process = subprocess.Popen(
            [str(bootstrap_script)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        def _log_stream(proc: subprocess.Popen) -> None:
            if not proc.stdout:
                return
            for line in proc.stdout:
                logger.info("[worker] %s", line.rstrip())

        threading.Thread(target=_log_stream, args=(process,), daemon=True).start()

        _wait_for_port("127.0.0.1", self.protobus_port, timeout=600.0)

        return WorkerHandle(
            protobus_address=f"127.0.0.1:{self.protobus_port}",
            workspace_root=workspace_root,
            cline_src_dir=cline_src_dir,
            process=process,
        )

    def stop(self, handle: WorkerHandle, timeout: float = 20.0) -> None:
        proc = handle.process
        if proc.poll() is not None:
            return
        logger.info("Stopping local Cline worker")
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning("Worker did not terminate gracefully; killing")
            proc.kill()

