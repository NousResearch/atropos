import numpy as np

def vr_optimization_function(gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type):
    """
    Optimizes VR performance score based on GPU utilization.  Higher score indicates better optimization.

    Args:
        gpu_util (float): GPU utilization percentage (0-100).
        vram_usage (float): VRAM usage in GB.
        cpu_util (float): CPU utilization percentage (0-100).
        scene_complexity (float): Scene complexity score (higher is more complex).
        duration (float): Frame rendering duration in seconds.
        gpu_type (float): GPU type (e.g., 2.0 for RTX 2080, 3.5 for RTX 3090, higher is better).

    Returns:
        float: Optimized performance score between 0 and 1 (inclusive). Returns -1 if input is invalid.

    Raises:
        TypeError: if any input is not a number.
        ValueError: if any input is outside the acceptable range.

    """

    # Input validation
    if not all(isinstance(x, (int, float)) for x in [gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type]):
        raise TypeError("All inputs must be numbers.")

    if not (0 <= gpu_util <= 100 and 0 <= cpu_util <= 100 and vram_usage >= 0 and scene_complexity >= 0 and duration > 0 and gpu_type > 0):
        raise ValueError("Invalid input range. Check your input values.")


    # Feature normalization (min-max scaling)
    gpu_util_norm = gpu_util / 100  #Scale GPU utilization to 0-1
    cpu_util_norm = cpu_util / 100  #Scale CPU utilization to 0-1

    #Consider VRAM usage penalty - higher VRAM usage reduces score.  Adjust scaling as needed
    vram_penalty = 1 / (1 + vram_usage)

    #Scene complexity impact - higher complexity reduces score.  Adjust scaling as needed.
    complexity_penalty = 1 / (1 + scene_complexity)

    #GPU type boost - higher GPU type increases score
    gpu_type_boost = np.clip(gpu_type / 4.0, 0, 1) #Clip to prevent overshooting


    # Optimization logic: Prioritize GPU utilization, penalize high CPU, VRAM and scene complexity, boost based on GPU type.
    #Adjust weights as needed to tune the optimization
    performance_score = 0.5 * gpu_util_norm + 0.2 * (1 - cpu_util_norm) + 0.1 * vram_penalty + 0.1 * complexity_penalty + 0.1 * gpu_type_boost

    #Handle duration - shorter duration is better.  Adjust scaling as needed.
    duration_weight = 0.1
    performance_score -= duration_weight * min(duration, 1) # penalize if duration exceeds 1 second


    # Ensure score is within 0-1 range
    return np.clip(performance_score, 0, 1)