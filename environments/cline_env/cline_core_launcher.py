import os
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ClineCoreConfig:
    cline_root: Path
    protobus_port: int = 26040
    hostbridge_port: int = 26041
    workspace_dir: Optional[Path] = None
    use_coverage: bool = False


class ClineCoreProcess:
    def __init__(self, config: ClineCoreConfig):
        self.config = config
        self.process: Optional[subprocess.Popen] = None

    def start(self, timeout: float = 60.0) -> None:
        if self.process is not None:
            raise RuntimeError("Cline core process is already running")

        cline_root = self.config.cline_root
        if not cline_root.exists():
            raise FileNotFoundError(f"Cline root directory not found: {cline_root}")

        env = os.environ.copy()
        env["PROTOBUS_PORT"] = str(self.config.protobus_port)
        env["HOSTBRIDGE_PORT"] = str(self.config.hostbridge_port)
        env.setdefault("E2E_TEST", "true")
        env.setdefault("CLINE_ENVIRONMENT", "local")
        # Disable banners and remote config for standalone / RL usage
        env.setdefault("CLINE_DISABLE_BANNERS", "true")
        env.setdefault("CLINE_DISABLE_REMOTE_CONFIG", "true")

        if self.config.workspace_dir is not None:
            env["WORKSPACE_DIR"] = str(self.config.workspace_dir)

        if self.config.use_coverage:
            env["USE_C8"] = "true"

        cmd = ["npx", "tsx", "scripts/test-standalone-core-api-server.ts"]

        self.process = subprocess.Popen(
            cmd,
            cwd=str(cline_root),
            env=env,
        )

        self._wait_for_port(timeout)

    def _wait_for_port(self, timeout: float) -> None:
        deadline = time.time() + timeout
        addr = ("127.0.0.1", self.config.protobus_port)

        while time.time() < deadline:
            if self.process and self.process.poll() is not None:
                raise RuntimeError(
                    f"Cline core process exited early with code {self.process.returncode}"
                )
            try:
                with socket.create_connection(addr, timeout=1.0):
                    return
            except OSError:
                time.sleep(0.2)

        raise TimeoutError(
            f"Timed out waiting for Cline core to listen on {addr[0]}:{addr[1]}"
        )

    def stop(self, timeout: float = 10.0) -> None:
        if self.process is None:
            return

        self.process.terminate()
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
        finally:
            self.process = None
