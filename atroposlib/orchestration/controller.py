import logging
import math
from typing import Optional, Dict, Any, List
from .metrics import WorkloadMetrics

logger = logging.getLogger(__name__)

class ScalingController:
    """
    Decides the "Desired Actor Count" based on workload metrics.
    Uses a dampened calculation with hysteresis to avoid flapping.
    """
    def __init__(
        self,
        min_actors: int = 1,
        max_actors: int = 20,
        target_pressure: float = 1.0,
        scaling_threshold: float = 0.2, # ±20%
        cooldown_seconds: int = 60,
        max_step_change: int = 4
    ):
        self.min_actors = min_actors
        self.max_actors = max_actors
        self.target_pressure = target_pressure
        self.scaling_threshold = scaling_threshold
        self.cooldown_seconds = cooldown_seconds
        self.max_step_change = max_step_change
        
        self.last_action_timestamp = 0
        self.current_desired = min_actors

    def calculate_desired(self, metrics: WorkloadMetrics, current_actors: int) -> int:
        """
        Decides the next target for the number of environment actors.
        """
        now = metrics.timestamp
        pressure = metrics.rollout_pressure
        
        # 1. Check cooldown
        if now - self.last_action_timestamp < self.cooldown_seconds:
            remaining = int(self.cooldown_seconds - (now - self.last_action_timestamp))
            logger.debug(f"Controller: In cooldown ({remaining}s remaining). Holding at {self.current_desired} actors.")
            return self.current_desired

        # 2. Sensitivity check (Hysteresis)
        # If work is roughly satisfying target, don't change anything.
        if abs(pressure - self.target_pressure) < self.scaling_threshold:
            logger.debug(f"Controller: Pressure {pressure:.2f} within threshold of {self.target_pressure}. No action.")
            return self.current_desired

        # 3. Calculate target
        # Target = Current * (Current_Pressure / Ideal_Pressure)
        raw_target = math.ceil(current_actors * (pressure / self.target_pressure))
        
        # 4. Apply step constraints (Rate Limiting)
        diff = raw_target - current_actors
        if abs(diff) > self.max_step_change:
            logger.info(f"Controller: Step change {diff} exceeds max_step_change ({self.max_step_change}). Capping.")
            raw_target = current_actors + (self.max_step_change if diff > 0 else -self.max_step_change)

        # 5. Apply world bounds
        final_target = max(self.min_actors, min(self.max_actors, raw_target))
        
        if final_target != current_actors:
            self.last_action_timestamp = now
            self.current_desired = final_target
            direction = "UP" if final_target > current_actors else "DOWN"
            logger.info(f"Controller DECISION: Scale {direction} {current_actors} -> {final_target} (Pressure: {pressure:.2f})")
        
        return final_target
