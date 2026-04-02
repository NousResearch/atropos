import time
import requests
from typing import Optional, Dict, Any
from dataclasses import dataclass
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

@dataclass
class WorkloadMetrics:
    current_step: int
    queue_size: int
    total_rollouts: int
    unallocated_fraction: float
    num_envs: int
    batch_size: int
    timestamp: float

    @property
    def rollout_pressure(self) -> float:
        """
        Calculates the "Rollout Pressure" (RP). 
        RP = (Queue Size / Batch Size).
        If RP > 1.0, the trainer is starving.
        """
        if self.batch_size <= 0:
            return 0.0
        return self.queue_size / self.batch_size

class MetricsCollector:
    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip("/")
        self.last_metrics: Optional[WorkloadMetrics] = None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException,)),
        reraise=False
    )
    def poll(self) -> Optional[WorkloadMetrics]:
        """
        Polls the Atropos server for global metrics with retries.
        """
        try:
            response = requests.get(f"{self.server_url}/global-status", timeout=5)
            response.raise_for_status()
            data = response.json()
            
            metrics = WorkloadMetrics(
                current_step=data["current_step"],
                queue_size=data["queue_size"],
                total_rollouts=data["total_rollouts_processed"],
                unallocated_fraction=data["unallocated_fraction"],
                num_envs=data["num_connected_envs"],
                batch_size=data["batch_size"],
                timestamp=time.time()
            )
            self.last_metrics = metrics
            return metrics
        except Exception as e:
            logger.error(f"Failed to poll metrics from {self.server_url}: {e}")
            return None
