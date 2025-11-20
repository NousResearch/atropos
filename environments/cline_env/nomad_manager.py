import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class NomadJobError(Exception):
    pass


class NomadManager:
    def __init__(
        self,
        job_hcl: Path,
        nomad_address: str = "http://127.0.0.1:4646",
        job_name: Optional[str] = None,
    ) -> None:
        self.job_hcl = job_hcl
        self.nomad_address = nomad_address
        self.job_name = job_name or job_hcl.stem
        self.job_id: Optional[str] = None
        self.allocation_id: Optional[str] = None

    def _run_cmd(self, args):
        env = os.environ.copy()
        env["NOMAD_ADDR"] = self.nomad_address
        logger.debug("Running command: %s", " ".join(args))
        result = subprocess.run(args, env=env, capture_output=True, text=True)
        if result.returncode != 0:
            raise NomadJobError(f"Command {' '.join(args)} failed: {result.stderr.strip()}")
        return result.stdout

    def submit(self, job_vars: Optional[Dict[str, str]] = None) -> str:
        args = ["nomad", "job", "run"]
        if job_vars:
            for key, value in job_vars.items():
                args.extend(["-var", f"{key}={value}"])
        args.append(str(self.job_hcl))
        output = self._run_cmd(args)
        logger.info("Nomad job run output: %s", output)
        for line in output.splitlines():
            if "Allocation" in line and "created" in line:
                cleaned = line.replace('"', "")
                parts = cleaned.split()
                if "Allocation" in parts:
                    idx = parts.index("Allocation")
                    if idx + 1 < len(parts):
                        self.allocation_id = parts[idx + 1]
            if line.startswith("==> Monitoring evaluation"):
                parts = line.split()
                if len(parts) >= 5:
                    self.job_id = parts[4].rstrip('.')
                    break
        if not self.job_id:
            self.job_id = self.job_name
        return self.job_id

    def _infer_job_id(self) -> Optional[str]:
        try:
            output = self._run_cmd(["nomad", "job", "status", "-json", self.job_name])
            data = json.loads(output)
            return data.get("ID")
        except Exception:
            return None

    def wait_for_allocation(self, timeout: float = 300.0) -> str:
        if not self.allocation_id:
            raise NomadJobError("Allocation ID unknown; check job submission output")

        deadline = time.time() + timeout
        while time.time() < deadline:
            output = self._run_cmd(["nomad", "alloc", "status", "-json", self.allocation_id])
            data = json.loads(output)
            status = data.get("ClientStatus")
            if status == "running":
                return self.allocation_id
            if status in {"complete", "failed", "lost"}:
                raise NomadJobError(f"Allocation {self.allocation_id} entered terminal state: {status}")
            time.sleep(2.0)
        raise TimeoutError("Timed out waiting for allocation to reach running state")

    def stop_job(self) -> None:
        if not self.job_id:
            return
        try:
            self._run_cmd(["nomad", "job", "stop", "-purge", self.job_id])
            logger.info("Nomad job %s stopped", self.job_id)
        except NomadJobError as exc:
            logger.warning("Failed to stop job %s: %s", self.job_id, exc)
        self.job_id = None
