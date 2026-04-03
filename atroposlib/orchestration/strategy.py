import logging
import os
import shlex
import signal
import socket
import subprocess
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import psutil

logger = logging.getLogger(__name__)


class ScalingStrategy(ABC):
    """
    Abstract interface for scaling environment actors.
    """

    @abstractmethod
    def set_instance_count(self, target_count: int, **kwargs):
        pass

    @abstractmethod
    def get_current_count(self) -> int:
        pass

    @abstractmethod
    def get_draining_count(self) -> int:
        pass

    @abstractmethod
    def cleanup(self):
        pass


class LocalActor(ScalingStrategy):
    """
    Manages local environment server processes via subprocess.
    Supports dynamic port injection via '$PORT' placeholder.
    Now supports GPU isolation via '--gpus-per-actor'.
    """

    def __init__(
        self, command: List[str], cwd: str = ".", port_range: str = "8001:8020"
    ):
        self.command = command
        self.cwd = cwd
        self.processes: List[subprocess.Popen] = []
        self.draining_processes: List[subprocess.Popen] = []
        # Store launch timestamps to handle boot timeouts/CrashLoops
        self.launch_timestamps: Dict[int, float] = {}
        # Count failures within short windows to detect crash loops
        self.failure_history: List[float] = []

        # Port management
        try:
            start, end = map(int, port_range.split(":"))
            self.free_ports = list(range(start, end + 1))
        except:
            logger.warning(
                f"Invalid port range '{port_range}'. Using default 8001:8020"
            )
            self.free_ports = list(range(8001, 8021))

        self.pid_to_port: Dict[int, int] = {}

        # GPU management
        self.pid_to_gpus: Dict[int, List[int]] = {}
        self.gpu_pool: List[int] = self._discover_gpus()
        self.available_gpus: List[int] = list(self.gpu_pool)

        # Adopt any existing processes on startup
        self._adopt_existing_processes()

    def _discover_gpus(self) -> List[int]:
        """Discovery of available GPU IDs via nvidia-smi."""
        try:
            out = subprocess.check_output(["nvidia-smi", "-L"]).decode()
            return [
                int(line.split(":")[0].split()[-1]) for line in out.strip().split("\n")
            ]
        except:
            logger.warning(
                "No GPUs discovered via nvidia-smi. Running in CPU-only mode."
            )
            return []

    def _check_gpu_health(self, gpu_id: int) -> bool:
        """Resource Cordoning: Check if a GPU is thermally throttled."""
        try:
            cmd = [
                "nvidia-smi",
                "-i",
                str(gpu_id),
                "--query-gpu=clocks_throttle_reasons.active",
                "--format=csv,noheader,nounits",
            ]
            reason = subprocess.check_output(cmd).decode().strip()
            # 0x0 or 0x1 (idle) are fine. Anything else is a hardware-level throttle/error.
            return reason in ["0x0000000000000000", "0x0000000000000001"]
        except:
            return True

    def _adopt_existing_processes(self):
        """Find and manage existing processes that match the environment command."""
        search_cmd = " ".join(self.command).replace("$PORT", "")
        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmdline = " ".join(proc.info["cmdline"] or [])
                if search_cmd in cmdline and proc.pid != os.getpid():
                    if any(p.pid == proc.pid for p in self.processes):
                        continue

                    logger.info(f"LocalActor: Adopting existing process {proc.pid}")

                    class AdoptedProcess:
                        def __init__(self, pid):
                            self.pid = pid

                        def poll(self):
                            try:
                                p = psutil.Process(self.pid)
                                return None if p.is_running() else 0
                            except:
                                return 0

                        def wait(self, timeout=None):
                            try:
                                return psutil.Process(self.pid).wait(timeout)
                            except:
                                return 0

                        def terminate(self):
                            try:
                                os.killpg(os.getpgid(self.pid), signal.SIGTERM)
                            except:
                                pass

                        def kill(self):
                            try:
                                os.killpg(os.getpgid(self.pid), signal.SIGKILL)
                            except:
                                pass

                    self.processes.append(AdoptedProcess(proc.pid))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    def set_instance_count(self, target_count: int, gpus_per_actor: int = 1):
        current_count = self.get_current_count()

        if target_count > current_count:
            to_add = target_count - current_count
            logger.info(
                f"LocalActor: Scaling UP by {to_add} (Gpus/Actor: {gpus_per_actor})"
            )
            for _ in range(to_add):
                # Backoff protection
                now = time.time()
                if len([f for f in self.failure_history if now - f < 60]) >= 3:
                    logger.error("LocalActor: CrashLoopBackOff. Scaling halted.")
                    break

                # Resource availability check
                if not self.free_ports:
                    logger.error("LocalActor: Out of Port capacity.")
                    break

                # If we have GPUs, we enforce isolation. If not (CPU node), we ignore.
                assigned_gpus = []
                if self.gpu_pool:
                    if len(self.available_gpus) < gpus_per_actor:
                        logger.error("LocalActor: Out of GPU capacity.")
                        break

                    # GPU Cordoning: Verify healthy silicon
                    while len(assigned_gpus) < gpus_per_actor and self.available_gpus:
                        gid = self.available_gpus.pop(0)
                        if self._check_gpu_health(gid):
                            assigned_gpus.append(gid)
                        else:
                            logger.critical(
                                f"LocalActor: CORDONING GPU {gid} due to hardware throttle!"
                            )

                    if len(assigned_gpus) < gpus_per_actor:
                        logger.error("LocalActor: Could not find enough healthy GPUs.")
                        self.available_gpus.extend(assigned_gpus)
                        break

                port = self.free_ports.pop(0)
                # Socket pre-flight
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    if s.connect_ex(("localhost", port)) == 0:
                        logger.warning(f"LocalActor: Port {port} hijacked. Skipping.")
                        self.available_gpus.extend(assigned_gpus)
                        continue

                # Process isolation launch
                instance_command = [c.replace("$PORT", str(port)) for c in self.command]
                env = os.environ.copy()
                if assigned_gpus:
                    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, assigned_gpus))

                proc = subprocess.Popen(
                    instance_command,
                    cwd=self.cwd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setpgrp,
                    env=env,
                )
                self.processes.append(proc)
                self.launch_timestamps[proc.pid] = time.time()
                self.pid_to_port[proc.pid] = port
                self.pid_to_gpus[proc.pid] = assigned_gpus
                logger.info(
                    f"LocalActor: Launched PID {proc.pid} on port {port} (GPUs: {assigned_gpus})"
                )

        elif target_count < current_count:
            to_remove = current_count - target_count
            logger.info(f"LocalActor: Scaling DOWN by {to_remove}")
            for _ in range(to_remove):
                proc = self.processes.pop()
                pid = proc.pid
                if pid in self.launch_timestamps:
                    del self.launch_timestamps[pid]
                if pid in self.pid_to_port:
                    self.free_ports.append(self.pid_to_port.pop(pid))
                if pid in self.pid_to_gpus:
                    self.available_gpus.extend(self.pid_to_gpus.pop(pid))

                try:
                    logger.info(
                        f"LocalActor: Moving PID {pid} to drain mode (SIGUSR1)..."
                    )
                    os.killpg(os.getpgid(pid), signal.SIGUSR1)
                    self.draining_processes.append(proc)
                except:
                    try:
                        os.killpg(os.getpgid(pid), signal.SIGKILL)
                    except:
                        pass

    def get_current_count(self) -> int:
        new_processes = []
        for p in self.processes:
            if p.poll() is None:
                new_processes.append(p)
            else:
                pid = p.pid
                launch_time = self.launch_timestamps.get(pid, 0)
                if time.time() - launch_time < 10:
                    logger.warning(
                        f"LocalActor: PID {pid} died rapidly. Recording failure."
                    )
                    self.failure_history.append(time.time())

                if pid in self.launch_timestamps:
                    del self.launch_timestamps[pid]
                if pid in self.pid_to_port:
                    self.free_ports.append(self.pid_to_port.pop(pid))
                if pid in self.pid_to_gpus:
                    self.available_gpus.extend(self.pid_to_gpus.pop(pid))
        self.processes = new_processes

        still_draining = []
        for p in self.draining_processes:
            if p.poll() is not None:
                logger.debug(f"LocalActor: Draining finished for {p.pid}")
            else:
                still_draining.append(p)
        self.draining_processes = still_draining

        return len(self.processes)

    def get_draining_count(self) -> int:
        return len(self.draining_processes)

    def cleanup(self):
        logger.info("LocalActor: Cleaning up all managed processes...")
        for proc in self.processes + self.draining_processes:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=1)
            except:
                pass
        self.processes = []
        self.draining_processes = []
        self.launch_timestamps = {}
        self.pid_to_port = {}
        self.pid_to_gpus = {}
        self.available_gpus = list(self.gpu_pool)


class RemoteActor(ScalingStrategy):
    """
    Manages environment servers on a remote host via SSH.
    Requirement: Passwordless SSH access to the target host.
    """

    def __init__(self, host: str, command: List[str], port_range: str = "8001:8020"):
        self.host = host
        self.command = command
        # Remote scaling uses PIDs on the remote machine
        self.remote_pids: List[int] = []
        self.draining_pids: List[int] = []

        try:
            start, end = map(int, port_range.split(":"))
            self.free_ports = list(range(start, end + 1))
        except:
            self.free_ports = list(range(8001, 8021))

        self.pid_to_port: Dict[int, int] = {}

    def _ssh_exec(self, cmd: str) -> str:
        full_cmd = ["ssh", "-o", "BatchMode=yes", self.host, cmd]
        return (
            subprocess.check_output(full_cmd, stderr=subprocess.STDOUT).decode().strip()
        )

    def set_instance_count(self, target_count: int, **kwargs):
        current_count = self.get_current_count()

        if target_count > current_count:
            to_add = target_count - current_count
            logger.info(f"RemoteActor({self.host}): Scaling UP by {to_add}")
            for _ in range(to_add):
                if not self.free_ports:
                    break
                port = self.free_ports.pop(0)
                cmd_str = " ".join(
                    [c.replace("$PORT", str(port)) for c in self.command]
                )
                # Launch in background on remote
                launch_str = f"nohup {cmd_str} > /dev/null 2>&1 & echo $!"
                pid = int(self._ssh_exec(launch_str))
                self.remote_pids.append(pid)
                self.pid_to_port[pid] = port
                logger.debug(
                    f"RemoteActor({self.host}): Launched PID {pid} on port {port}"
                )

        elif target_count < current_count:
            to_remove = current_count - target_count
            logger.info(f"RemoteActor({self.host}): Scaling DOWN by {to_remove}")
            for _ in range(to_remove):
                pid = self.remote_pids.pop()
                if pid in self.pid_to_port:
                    self.free_ports.append(self.pid_to_port.pop(pid))
                try:
                    logger.info(
                        f"RemoteActor({self.host}): Draining PID {pid} (SIGUSR1)"
                    )
                    self._ssh_exec(f"kill -USR1 {pid}")
                    self.draining_pids.append(pid)
                except:
                    pass

    def get_current_count(self) -> int:
        alive_pids = []
        if self.remote_pids:
            pids_str = " ".join(map(str, self.remote_pids))
            try:
                out = self._ssh_exec(f"ps -p {pids_str} -o pid=")
                alive_pids = [int(p) for p in out.split()]
            except:
                pass

        for p in self.remote_pids:
            if p not in alive_pids and p in self.pid_to_port:
                self.free_ports.append(self.pid_to_port.pop(p))
        self.remote_pids = alive_pids

        if self.draining_pids:
            dpids_str = " ".join(map(str, self.draining_pids))
            try:
                dout = self._ssh_exec(f"ps -p {dpids_str} -o pid=")
                still_draining = [int(p) for p in dout.split()]
                self.draining_pids = still_draining
            except:
                self.draining_pids = []

        return len(self.remote_pids)

    def get_draining_count(self) -> int:
        return len(self.draining_pids)

    def cleanup(self):
        logger.info(f"RemoteActor({self.host}): Emergency cleanup of all PIDs")
        all_pids = self.remote_pids + self.draining_pids
        if all_pids:
            pids_str = " ".join(map(str, all_pids))
            try:
                self._ssh_exec(f"kill -9 {pids_str}")
            except:
                pass
        self.remote_pids = []
        self.draining_pids = []
