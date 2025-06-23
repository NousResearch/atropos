import numpy as np

def vr_optimization_function(gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type):
    """
    Optimizes VR performance for frame time consistency and comfort.

    Args:
        gpu_util (float): GPU utilization percentage (0-100).
        vram_usage (float): VRAM usage in GB.
        cpu_util (float): CPU utilization percentage (0-100).
        scene_complexity (float): Scene complexity score (higher is more complex).
        duration (float): Frame time in seconds.
        gpu_type (float): GPU type (numerical representation, e.g., 2.0 for RTX 2080).


    Returns:
        float: Optimized performance score between 0 and 1 (inclusive).  Returns -1 if input is invalid.

    Raises:
        TypeError: If any input is not a number.
        ValueError: If any input is outside the expected range.

    """

    # Input validation
    if not all(isinstance(x, (int, float)) for x in [gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type]):
        raise TypeError("All inputs must be numbers.")

    if not (0 <= gpu_util <= 100 and 0 <= cpu_util <= 100 and vram_usage >=0 and scene_complexity >= 0 and duration > 0):
        raise ValueError("Invalid input range. GPU/CPU utilization (0-100%), VRAM usage >=0, Scene complexity >=0, duration > 0")


    # Feature normalization (min-max scaling)
    gpu_util_norm = gpu_util / 100.0
    cpu_util_norm = cpu_util / 100.0
    vram_usage_norm = np.clip(vram_usage / 16, 0,1) # Assuming 16GB as a reasonable upper bound for VRAM. Adjust if needed.

    #scene_complexity_norm = scene_complexity / 10 #Assuming a max complexity of 10. Adjust if needed.
    scene_complexity_norm = 1/(1+scene_complexity) #Inverse relationship: higher complexity means lower score


    #This is a simplification, consider more sophisticated weighting based on VR hardware and application specifics.
    duration_norm = 1/(1+duration) # shorter duration is better.

    # Optimization logic (weighted average focusing on frame time consistency and resource utilization)

    weights = np.array([0.3, 0.2, 0.2, 0.1, 0.2]) # Adjust weights based on priorities. Frame time (duration) is heavily weighted.

    normalized_features = np.array([1 - gpu_util_norm, 1 - cpu_util_norm, 1- vram_usage_norm, scene_complexity_norm, duration_norm])


    performance_score = np.dot(normalized_features, weights)


    return np.clip(performance_score, 0, 1) #Ensure score is within [0,1] range.