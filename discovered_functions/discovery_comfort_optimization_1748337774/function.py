import numpy as np

def vr_optimization_function(gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type):
    """
    Optimizes VR performance for maximum comfort score.

    Args:
        gpu_util (float): GPU utilization (%).
        vram_usage (float): VRAM usage (GB).
        cpu_util (float): CPU utilization (%).
        scene_complexity (float): Scene complexity score (higher is more complex).
        duration (float): Duration of VR experience (seconds).
        gpu_type (float): GPU type (numerical representation, higher is better).


    Returns:
        float: Optimized performance score between 0 and 1 (inclusive).  Returns -1 if input is invalid.

    Raises:
        TypeError: If any input is not a number.
        ValueError: If any input is outside the expected range.
    """

    # Input validation
    if not all(isinstance(x, (int, float)) for x in [gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type]):
        raise TypeError("All inputs must be numbers.")

    if not (0 <= gpu_util <= 100 and 0 <= cpu_util <= 100 and vram_usage >=0 and scene_complexity >=0 and duration > 0 and gpu_type > 0):
        raise ValueError("Inputs are out of range. Check gpu_util, cpu_util, vram_usage, scene_complexity, duration and gpu_type values.")


    # Feature normalization (min-max scaling)
    gpu_util_norm = gpu_util / 100.0
    cpu_util_norm = cpu_util / 100.0
    # Assuming VRAM usage has an upper bound of 16GB for normalization
    vram_usage_norm = min(vram_usage / 16.0,1.0) #Cap at 1.0 to avoid values >1
    # Normalize scene complexity (assuming a reasonable max complexity of 10)
    scene_complexity_norm = min(scene_complexity / 10.0,1.0) #Cap at 1.0
    duration_norm = 1.0 / (1 + duration) #Inverse relationship: shorter duration is better


    # Optimization logic (weighted average with penalties)

    # Weights reflect importance; adjust based on your specific needs.
    weights = np.array([0.25, 0.2, 0.2, 0.15, 0.1, 0.1]) #GPU,VRAM,CPU,Scene,Duration,GPUtype

    #Penalties for high utilization
    gpu_util_penalty = 1 - (1-gpu_util_norm)**2 #Steeper penalty for higher GPU utilization.
    cpu_util_penalty = 1 - (1-cpu_util_norm)**2

    normalized_features = np.array([1-gpu_util_penalty, 1-vram_usage_norm, 1-cpu_util_penalty, 1-scene_complexity_norm, duration_norm, gpu_type/5.0]) #Assuming a max gpu type of 5.

    #Weighted score
    weighted_score = np.sum(normalized_features * weights)

    #Ensure score is within 0-1 range
    optimized_score = max(0, min(1, weighted_score))

    return optimized_score