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
        gpu_type (float): GPU type (e.g., 1.0, 2.0, 3.0 representing different tiers).

    Returns:
        float: Optimized performance score between 0 and 1 (higher is better). 
               Returns -1 if input validation fails.

    Raises:
        TypeError: if input data types are not numeric.
        ValueError: if input values are out of range.
    """

    # Input validation
    if not all(isinstance(i, (int, float)) for i in [gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type]):
        raise TypeError("All inputs must be numeric.")

    if not (0 <= gpu_util <= 100 and 0 <= cpu_util <= 100 and vram_usage >= 0 and scene_complexity >= 0 and duration > 0 and gpu_type > 0):
        raise ValueError("Input values are out of range.")


    # Feature normalization (min-max scaling) to prevent features with larger magnitudes from dominating.
    gpu_util_norm = gpu_util / 100.0  
    cpu_util_norm = cpu_util / 100.0
    vram_usage_norm = vram_usage / 16 #Assuming a maximum of 16GB VRAM as a reasonable upper bound. Adjust as needed.
    scene_complexity_norm = scene_complexity / 10 # Assuming a max complexity score of 10, adjust as needed.
    duration_norm = 1 / (duration + 0.001) #Reciprocal to reward shorter frame times. Added a small value to prevent division by zero.


    # Optimization logic: Weighted average focusing on frame time and resource utilization.
    # Weights are adjusted based on importance; Frame time consistency is prioritized.
    weights = np.array([0.4, 0.2, 0.2, 0.1, 0.1])  #Weights for duration, gpu_util, cpu_util, vram, scene complexity respectively

    normalized_features = np.array([duration_norm, gpu_util_norm, cpu_util_norm, vram_usage_norm, scene_complexity_norm])
    
    #GPU type influence: penalize lower end cards more for high utilization
    gpu_type_penalty = 1 - (gpu_type/5) #Assumes a maximum GPU type of 5. Adjust accordingly.
    gpu_type_penalty = max(0, min(1,gpu_type_penalty)) #Clamp between 0 and 1

    optimized_score = np.dot(weights, 1-normalized_features) * (1-gpu_type_penalty)


    #Ensure score is within [0,1]
    return max(0,min(1, optimized_score))