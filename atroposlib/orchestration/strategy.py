import subprocess
import os
import signal
import logging
from abc import ABC, abstractmethod
from typing import List, Dict

logger = logging.getLogger(__name__)

class ScalingStrategy(ABC):
    """
    Abstract interface for scaling environment actors.
    """
    @abstractmethod
    def set_instance_count(self, target_count: int):
        pass

    @abstractmethod
    def get_current_count(self) -> int:
        pass

    @abstractmethod
    def cleanup(self):
        pass

class LocalActor(ScalingStrategy):
    """
    Manages local environment server processes via subprocess.
    """
    def __init__(self, command: List[str], cwd: str = "."):
        self.command = command
        self.cwd = cwd
        self.processes: List[subprocess.Popen] = []

    def set_instance_count(self, target_count: int):
        current_count = len(self.processes)
        
        if target_count > current_count:
            # Scale UP
            to_add = target_count - current_count
            logger.info(f"LocalActor: Scaling UP by {to_add} (Total: {target_count})")
            for _ in range(to_add):
                proc = subprocess.Popen(
                    self.command,
                    cwd=self.cwd,
                    stdout=subprocess.DEVNULL, # Should probably be configurable
                    stderr=subprocess.STDOUT
                )
                self.processes.append(proc)
        
        elif target_count < current_count:
            # Scale DOWN
            to_remove = current_count - target_count
            logger.info(f"LocalActor: Scaling DOWN by {to_remove} (Total: {target_count})")
            for _ in range(to_remove):
                proc = self.processes.pop()
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

    def get_current_count(self) -> int:
        # Filter out dead processes
        self.processes = [p for p in self.processes if p.poll() is None]
        return len(self.processes)

    def cleanup(self):
        logger.info("LocalActor: Cleaning up all managed processes...")
        for proc in self.processes:
            proc.terminate()
        for proc in self.processes:
            try:
                proc.wait(timeout=2)
            except:
                proc.kill()
        self.processes = []
