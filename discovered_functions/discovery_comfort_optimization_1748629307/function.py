import numpy as np

def vr_optimization_function(gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type):
    """
    Optimizes VR performance for maximum comfort score.

    Args:
        gpu_util (float): GPU utilization percentage (0-100).
        vram_usage (float): VRAM usage in GB.
        cpu_util (float): CPU utilization percentage (0-100).
        scene_complexity (float): Scene complexity score (higher is more complex).
        duration (float): Duration of VR experience in seconds.
        gpu_type (float): GPU type (categorical, needs mapping for numerical processing).

    Returns:
        float: Optimized VR comfort score between 0 and 1 (inclusive).  Returns -1 if input is invalid.

    Raises:
        TypeError: if input is not a number.
        ValueError: if input values are out of range.

    """

    # Input validation
    if not all(isinstance(x, (int, float)) for x in [gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type]):
        raise TypeError("All inputs must be numbers.")

    if not (0 <= gpu_util <= 100 and 0 <= cpu_util <= 100 and scene_complexity >=0 and duration > 0 and gpu_type >= 0):
        raise ValueError("Invalid input range. GPU/CPU utilization (0-100), Scene Complexity (>=0), Duration (>0), GPU Type (>=0).")


    # Feature normalization (min-max scaling)
    gpu_util_norm = gpu_util / 100.0  
    cpu_util_norm = cpu_util / 100.0
    #Assuming VRAM usage is capped, otherwise needs range analysis based on hardware.
    vram_usage_norm = vram_usage / 8.0 #Example cap of 8GB. Adjust as needed for realistic cap.
    
    #Handle GPU type, assuming it is a categorical variable represented numerically
    #Replace with your mapping logic if necessary
    gpu_type_norm = gpu_type / 3.5 #Example mapping - adjust based on the scale of gpu_type

    #Mathematical optimization logic (weighted average with penalties)
    #Weights are adjusted based on perceived impact.  Adjust these based on your data.
    comfort_score = 0.3 * (1 - gpu_util_norm) + 0.2 * (1 - cpu_util_norm) + 0.2 * (1 - vram_usage_norm) + 0.2 * (1 / (1 + scene_complexity)) + 0.1 * (1/duration)


    #Penalty for high GPU type (Assuming higher values represent less efficient GPUs).
    comfort_score -= 0.1* gpu_type_norm

    #Clip the score to be between 0 and 1
    comfort_score = np.clip(comfort_score, 0, 1)


    return comfort_score