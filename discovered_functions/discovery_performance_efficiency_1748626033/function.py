import numpy as np

def vr_optimization_function(gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type):
    """
    Optimizes VR performance score focusing on GPU utilization.

    Args:
        gpu_util (float): GPU utilization percentage (0-100).
        vram_usage (float): VRAM usage in GB.
        cpu_util (float): CPU utilization percentage (0-100).
        scene_complexity (float): Scene complexity score (higher is more complex).
        duration (float): Frame duration in seconds.
        gpu_type (float): GPU type (numerical representation, e.g., 1,2,3...).

    Returns:
        float: Optimized performance score between 0 and 1 (higher is better). 
               Returns None if input is invalid.

    Raises:
        TypeError: if any input is not a number.
        ValueError: if any input is out of range.

    """
    # Input validation
    if not all(isinstance(x, (int, float)) for x in [gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type]):
        raise TypeError("All inputs must be numbers.")

    if not (0 <= gpu_util <= 100 and 0 <= cpu_util <= 100 and 0 <= scene_complexity and 0 <= duration and gpu_type >0):
        raise ValueError("Invalid input range. Check gpu_util, cpu_util, scene_complexity, duration, and gpu_type values.")

    # Feature normalization (min-max scaling)
    gpu_util_norm = gpu_util / 100.0  
    cpu_util_norm = cpu_util / 100.0
    vram_usage_norm = vram_usage / 16 #assuming a max VRAM of 16GB as a reasonable upper bound. Adjust as needed.


    # Optimization logic focusing on GPU utilization.  Penalize high VRAM and CPU usage.
    # We weight GPU utilization more heavily.

    gpu_weight = 0.6  
    vram_weight = 0.2
    cpu_weight = 0.2

    #Inverse duration to reward shorter frame times.  Handle potential division by zero.
    inv_duration = 1.0 / (duration + 1e-9) if duration > 0 else 0 # add a small value to avoid division by zero

    performance_score = (gpu_weight * gpu_util_norm) + (inv_duration * gpu_weight * 0.5) - (vram_weight * vram_usage_norm) - (cpu_weight * cpu_util_norm)


    #Clip the score between 0 and 1
    performance_score = np.clip(performance_score, 0, 1)

    return performance_score