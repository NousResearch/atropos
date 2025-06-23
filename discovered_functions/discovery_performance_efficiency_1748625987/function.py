import numpy as np

def vr_optimization_function(gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type):
    """
    Optimizes VR performance focusing on GPU utilization.

    Args:
        gpu_util (float): GPU utilization percentage (0-100).
        vram_usage (float): VRAM usage in GB.
        cpu_util (float): CPU utilization percentage (0-100).
        scene_complexity (float): Scene complexity score (higher is more complex).
        duration (float): Frame render duration in seconds.
        gpu_type (float): GPU type (numerical representation, e.g., 2.0 for RTX 2080, etc.).


    Returns:
        float: Optimized performance score between 0 and 1 (higher is better). 
               Returns -1 if input validation fails.

    Raises:
        TypeError: if any input is not a number.
        ValueError: if any input is outside the expected range.

    """

    # Input validation
    inputs = [gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type]
    if not all(isinstance(i, (int, float)) for i in inputs):
        raise TypeError("All inputs must be numbers.")
    if not (0 <= gpu_util <= 100 and 0 <= cpu_util <= 100 and vram_usage >=0 and scene_complexity >=0 and duration > 0):
        raise ValueError("Invalid input range. GPU and CPU utilization must be between 0 and 100, VRAM, Scene Complexity and Duration must be non-negative and Duration must be positive")


    #Feature Normalization (Min-Max scaling)
    gpu_util_norm = gpu_util / 100.0  # Normalize GPU utilization to 0-1
    cpu_util_norm = cpu_util / 100.0  # Normalize CPU utilization to 0-1

    #Avoid division by zero
    if duration == 0:
      duration = 1e-9 #Assigning a very small value to avoid division by zero

    #Optimization logic prioritizing GPU utilization and penalizing high VRAM/CPU usage & long duration
    #Weight adjustments can be tuned based on specific priorities
    gpu_weight = 0.6  
    vram_weight = 0.2
    cpu_weight = 0.1
    duration_weight = 0.1

    optimized_score = (gpu_weight * gpu_util_norm) - (vram_weight * (vram_usage / 16)) - (cpu_weight * cpu_util_norm) - (duration_weight * duration)


    #Clamp the score between 0 and 1
    optimized_score = np.clip(optimized_score, 0, 1)

    return optimized_score