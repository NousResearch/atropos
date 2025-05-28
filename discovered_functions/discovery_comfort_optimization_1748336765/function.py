import numpy as np

def vr_optimization_function(gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type):
    """
    Optimizes VR performance for maximum comfort score.

    Args:
        gpu_util (float): GPU utilization (0-100).
        vram_usage (float): VRAM usage in GB.
        cpu_util (float): CPU utilization (0-100).
        scene_complexity (float): Scene complexity score (higher is more complex).
        duration (float): Duration of VR experience in minutes.
        gpu_type (float): GPU type (e.g., 2.0 for RTX 2080, 3.5 for RTX 3090, higher is better).


    Returns:
        float: Optimized VR comfort score between 0 and 1 (inclusive). Returns -1 if input is invalid.

    Raises:
        TypeError: if any input is not a number.
        ValueError: if any input is out of range.

    """

    # Input validation
    if not all(isinstance(x, (int, float)) for x in [gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type]):
        raise TypeError("All inputs must be numbers.")

    if not (0 <= gpu_util <= 100 and 0 <= cpu_util <= 100 and vram_usage >= 0 and scene_complexity >= 0 and duration >= 0 and gpu_type > 0):
        raise ValueError("Inputs are out of range. Check gpu_util, cpu_util, vram_usage, scene_complexity, duration, and gpu_type.")


    # Feature normalization (min-max scaling)
    gpu_util_norm = gpu_util / 100.0
    cpu_util_norm = cpu_util / 100.0
    # Assuming a reasonable maximum VRAM usage of 24GB. Adjust as needed based on your system
    vram_usage_norm = min(vram_usage / 24.0,1.0) #Cap at 1.0 to avoid disproportionate effect

    #Considering scene complexity inversely proportional to comfort.  Higher complexity = lower comfort
    scene_complexity_norm = 1.0 / (1.0 + scene_complexity)  


    # Duration needs careful consideration.  Longer duration can reduce comfort due to fatigue
    duration_norm = 1.0 / (1.0 + duration)


    #Weighted average based on importance. Adjust weights based on your requirements.
    gpu_weight = 0.3  #Higher weight for GPU due to rendering
    cpu_weight = 0.2
    vram_weight = 0.15
    scene_weight = 0.2
    duration_weight = 0.1
    gpu_type_weight = 0.05


    comfort_score = (gpu_weight * (1 - gpu_util_norm) + cpu_weight * (1 - cpu_util_norm) + vram_weight * (1-vram_usage_norm) +
                    scene_weight * scene_complexity_norm + duration_weight * duration_norm + gpu_type_weight * (gpu_type/5.0) ) #Normalize gpu_type


    #Ensure score is within [0,1]
    comfort_score = np.clip(comfort_score, 0, 1)

    return comfort_score