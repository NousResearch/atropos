import json
import logging
import os
import socket
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Protocol

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
    process: Optional[subprocess.Popen] = None
    # For Nomad workers
    nomad_job_id: Optional[str] = None
    nomad_allocation_id: Optional[str] = None


class WorkerManager(Protocol):
    """Protocol for worker managers - can be Local or Nomad."""

    def start_for_profile(self, profile_key: str, task_env: Dict[str, str]) -> WorkerHandle:
        ...

    def stop(self, handle: WorkerHandle, timeout: float = 20.0) -> None:
        ...


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
        if proc is None or proc.poll() is not None:
            return
        logger.info("Stopping local Cline worker")
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning("Worker did not terminate gracefully; killing")
            proc.kill()


class NomadWorkerManager:
    """Starts and stops Cline workers via Nomad job scheduler.

    Uses a parameterized HCL job template to spin up workers with the correct
    Nix profile and task environment.
    """

    def __init__(
        self,
        protobus_port: int = 46040,
        hostbridge_port: int = 46041,
        profiles: Optional[Dict[str, ProfileConfig]] = None,
        nomad_address: str = "http://127.0.0.1:4646",
    ) -> None:
        self.protobus_port = protobus_port
        self.hostbridge_port = hostbridge_port
        self.profile_registry = profiles or PROFILE_REGISTRY
        self.nomad_address = nomad_address
        self.atropos_root = Path(__file__).resolve().parent.parent.parent
        self.job_hcl = self.atropos_root / "environments" / "cline_env" / "cline_dev" / "nomad_worker_job.hcl"

    def _run_nomad_cmd(self, args: list) -> str:
        env = os.environ.copy()
        env["NOMAD_ADDR"] = self.nomad_address
        logger.debug("Running Nomad command: %s", " ".join(args))
        result = subprocess.run(args, env=env, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Nomad command {' '.join(args)} failed: {result.stderr.strip()}")
        return result.stdout

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

        workspace_root = Path(task_env.get("WORKSPACE_ROOT", tempfile.mkdtemp(prefix=f"cline-{profile_key}-")))
        workspace_root.mkdir(parents=True, exist_ok=True)

        cline_src_dir = Path(task_env.get("CLINE_SRC_DIR", "/tmp/nous-cline-worker"))

        # Prepare Nomad job variables
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")

        if not anthropic_key:
            raise RuntimeError("ANTHROPIC_API_KEY must be set for Nomad worker")

        job_vars = {
            "anthropic_api_key": anthropic_key,
            "anthropic_model": anthropic_model,
            "profile_key": profile_key,
            "bootstrap_script": str(config.bootstrap_script),
            "workspace_root": str(workspace_root),
            "cline_src_dir": str(cline_src_dir),
            "task_env_json": json.dumps(task_env),
            "job_name": f"cline-{profile_key}",
            "protobus_port": str(self.protobus_port),
            "hostbridge_port": str(self.hostbridge_port),
            "profile_dir": str(config.profile_dir),
            "atropos_root": str(self.atropos_root),
        }

        # Submit Nomad job
        args = ["nomad", "job", "run"]
        for key, value in job_vars.items():
            args.extend(["-var", f"{key}={value}"])
        args.append(str(self.job_hcl))

        logger.info("Submitting Nomad job for profile %s", profile_key)
        output = self._run_nomad_cmd(args)
        logger.info("Nomad job submitted: %s", output[:500])

        # Parse job ID and allocation ID from output
        job_id = f"cline-{profile_key}"
        allocation_id = None
        for line in output.splitlines():
            if "Allocation" in line and "created" in line:
                cleaned = line.replace('"', "")
                parts = cleaned.split()
                if "Allocation" in parts:
                    idx = parts.index("Allocation")
                    if idx + 1 < len(parts):
                        allocation_id = parts[idx + 1]
                        break

        # Wait for allocation to be running
        if allocation_id:
            deadline = time.time() + 300.0
            while time.time() < deadline:
                try:
                    status_output = self._run_nomad_cmd(["nomad", "alloc", "status", "-json", allocation_id])
                    status_data = json.loads(status_output)
                    client_status = status_data.get("ClientStatus")
                    if client_status == "running":
                        logger.info("Nomad allocation %s is running", allocation_id)
                        break
                    if client_status in {"complete", "failed", "lost"}:
                        raise RuntimeError(f"Allocation {allocation_id} entered terminal state: {client_status}")
                except Exception as e:
                    logger.warning("Error checking allocation status: %s", e)
                time.sleep(2.0)
            else:
                raise TimeoutError("Timed out waiting for Nomad allocation to reach running state")

        # Wait for protobus port to be ready
        _wait_for_port("127.0.0.1", self.protobus_port, timeout=600.0)

        return WorkerHandle(
            protobus_address=f"127.0.0.1:{self.protobus_port}",
            workspace_root=workspace_root,
            cline_src_dir=cline_src_dir,
            process=None,
            nomad_job_id=job_id,
            nomad_allocation_id=allocation_id,
        )

    def stop(self, handle: WorkerHandle, timeout: float = 20.0) -> None:
        if not handle.nomad_job_id:
            return
        try:
            logger.info("Stopping Nomad job %s", handle.nomad_job_id)
            self._run_nomad_cmd(["nomad", "job", "stop", "-purge", handle.nomad_job_id])
            logger.info("Nomad job %s stopped", handle.nomad_job_id)
        except Exception as e:
            logger.warning("Failed to stop Nomad job %s: %s", handle.nomad_job_id, e)


def get_worker_manager(
    use_nomad: bool = True,
    protobus_port: int = 46040,
    hostbridge_port: int = 46041,
    nomad_address: str = "http://127.0.0.1:4646",
) -> WorkerManager:
    """Factory function to get the appropriate worker manager.
    
    Args:
        use_nomad: If True (default), use NomadWorkerManager. If False, use LocalWorkerManager.
        protobus_port: Port for gRPC protobus service.
        hostbridge_port: Port for host bridge service.
        nomad_address: Address of Nomad server (only used if use_nomad=True).
    
    Returns:
        WorkerManager instance (either NomadWorkerManager or LocalWorkerManager).
    """
    if use_nomad:
        return NomadWorkerManager(
            protobus_port=protobus_port,
            hostbridge_port=hostbridge_port,
            nomad_address=nomad_address,
        )
    else:
        return LocalWorkerManager(
            protobus_port=protobus_port,
            hostbridge_port=hostbridge_port,
        )
