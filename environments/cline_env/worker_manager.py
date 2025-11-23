import logging
import os
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from .profile_registry import PROFILE_REGISTRY, ProfileConfig

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

    - Clones/updates the NousResearch Cline fork.
    - Bootstraps the target workspace.
    - Starts the standalone gRPC server on a fixed port.
    """

    def __init__(
        self,
        protobus_port: int = 46040,
        hostbridge_port: int = 46041,
        profiles: Optional[Dict[str, ProfileConfig]] = None,
    ) -> None:
        self.protobus_port = protobus_port
        self.hostbridge_port = hostbridge_port
        self.profile_registry = profiles or PROFILE_REGISTRY
        self.bootstrap_script = Path(__file__).resolve().parent / "cline_dev" / "bootstrap_cline_worker.sh"

    def start_for_profile(self, profile_key: str, task_env: Dict[str, str]) -> WorkerHandle:
        config = self.profile_registry.get(profile_key)
        if not config:
            raise ValueError(f"Unsupported worker profile: {profile_key}")
        if not config.profile_dir.exists():
            raise FileNotFoundError(f"Nix profile for '{profile_key}' not found at {config.profile_dir}")
        if not config.bootstrap_script.exists():
            raise FileNotFoundError(
                f"Bootstrap script for profile '{profile_key}' missing: {config.bootstrap_script}"
            )

        base_dir = Path(__file__).resolve().parent
        profile_env = os.environ.copy()
        profile_env.update(task_env)
        profile_env.setdefault("TASK_BOOTSTRAP_SCRIPT", str(config.bootstrap_script))
        profile_env.setdefault("CLINE_PROFILE_KEY", profile_key)

        workspace_root = Path(profile_env.get("WORKSPACE_ROOT", "/tmp/cline-workspace"))
        workspace_root.mkdir(parents=True, exist_ok=True)

        cline_src_dir = Path(profile_env.get("CLINE_SRC_DIR", "/tmp/nous-cline-worker"))

        profile_env.update(
            {
                "CLINE_SRC_DIR": str(cline_src_dir),
                "WORKSPACE_ROOT": str(workspace_root),
                "PROTOBUS_PORT": str(self.protobus_port),
                "HOSTBRIDGE_PORT": str(self.hostbridge_port),
            }
        )

        logger.info("Starting local Cline worker for profile %s", profile_key)
        cmd = [
            "nix",
            "develop",
            str(config.profile_dir),
            "--command",
            str(self.bootstrap_script),
        ]
        process = subprocess.Popen(
            cmd,
            env=profile_env,
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
