import numpy as np

def vr_optimization_function(gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type):
    """
    Optimizes VR performance focusing on GPU utilization.

    Args:
        gpu_util (float): GPU utilization percentage (0-100).
        vram_usage (float): VRAM usage in GB.
        cpu_util (float): CPU utilization percentage (0-100).
        scene_complexity (float): Scene complexity score (higher is more complex).
        duration (float): Frame rendering duration in seconds.
        gpu_type (float): GPU type (numerical representation, e.g., 2.0 for RTX 2080).


    Returns:
        float: Optimized performance score between 0 and 1 (higher is better).  Returns -1 if input is invalid.

    Raises:
        TypeError: if input is not a number.
        ValueError: if input is outside the acceptable range.

    """

    # Input validation
    input_features = [gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type]
    if not all(isinstance(i, (int, float)) for i in input_features):
        raise TypeError("All input features must be numbers.")
    if not 0 <= gpu_util <= 100 or not 0 <= cpu_util <= 100 or not scene_complexity >=0 or not duration > 0 or not gpu_type >0 :
        raise ValueError("Invalid input range. gpu_util and cpu_util should be between 0 and 100. scene_complexity and duration should be positive.")


    # Feature normalization (min-max scaling)
    gpu_util_norm = gpu_util / 100.0
    cpu_util_norm = cpu_util / 100.0
    vram_usage_norm = vram_usage / 16  # Assuming a max VRAM of 16GB as a reasonable upper bound. Adjust as needed.
    scene_complexity_norm = scene_complexity / 10 #Assumes a reasonable upper bound of 10 for scene complexity. Adjust as needed.
    duration_norm = 1/(duration + 0.001) #inverse duration, shorter is better. adding small constant to prevent division by zero


    # Optimization logic (weighted average focusing on GPU utilization)
    weights = np.array([0.6, 0.1, 0.1, 0.1, 0.1]) #weighting GPU utilization higher
    normalized_features = np.array([gpu_util_norm, vram_usage_norm, cpu_util_norm, scene_complexity_norm, duration_norm])

    performance_score = np.sum(weights * normalized_features)


    #Ensure score is within 0-1 range
    performance_score = max(0, min(1, performance_score))


    return performance_score