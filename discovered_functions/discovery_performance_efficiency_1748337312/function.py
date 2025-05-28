import numpy as np

def vr_optimization_function(gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type):
    """
    Optimizes VR performance score based on GPU utilization.

    Args:
        gpu_util (float): GPU utilization percentage (0-100).
        vram_usage (float): VRAM usage in GB.
        cpu_util (float): CPU utilization percentage (0-100).
        scene_complexity (float): Scene complexity score (higher is more complex).
        duration (float): Frame duration in seconds.
        gpu_type (float): GPU type (categorical, needs to be encoded numerically).

    Returns:
        float: Optimized performance score between 0 and 1 (higher is better). 
               Returns -1 if input validation fails.

    Raises:
        TypeError: if input types are invalid.
        ValueError: if input values are out of range.
    """

    # Input validation
    if not all(isinstance(x, (int, float)) for x in [gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type]):
        raise TypeError("All inputs must be numeric.")

    if not (0 <= gpu_util <= 100 and 0 <= cpu_util <= 100 and vram_usage >=0 and scene_complexity >= 0 and duration > 0):
        raise ValueError("Invalid input range.  Check GPU utilization, CPU utilization, VRAM usage, scene complexity, and duration")


    # Feature normalization (min-max scaling)
    gpu_util_norm = gpu_util / 100.0  
    cpu_util_norm = cpu_util / 100.0
    vram_usage_norm = vram_usage / 16 #assuming a maximum of 16GB VRAM is reasonable, adjust if necessary

    #Prioritize GPU utilization
    gpu_weight = 0.6

    # Optimization logic (weighted average focusing on GPU utilization)
    performance_score = gpu_weight * gpu_util_norm + (1 - gpu_weight) * (1 - (cpu_util_norm + vram_usage_norm + scene_complexity * 0.2 + duration * 0.1)) #Weight duration and scene complexity lower

    # Handle potential negative scores due to normalization and weighting
    performance_score = max(0, min(1, performance_score))

    return performance_score