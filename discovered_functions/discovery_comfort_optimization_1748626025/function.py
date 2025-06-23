import numpy as np

def vr_optimization_function(gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type):
    """
    Optimizes VR performance for maximum comfort score.

    Args:
        gpu_util (float): GPU utilization percentage (0-100).
        vram_usage (float): VRAM usage in GB.
        cpu_util (float): CPU utilization percentage (0-100).
        scene_complexity (float): Scene complexity score (higher is more complex).
        duration (float): Frame duration in seconds.
        gpu_type (float): GPU type (numerical encoding, e.g., 1,2,3...).


    Returns:
        float: Optimized VR comfort score (0-1, higher is better).  Returns -1 if input is invalid.

    Raises:
        TypeError: if any input is not a number.
        ValueError: if any input is outside the expected range.

    """
    # Input validation
    if not all(isinstance(x, (int, float)) for x in [gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type]):
        raise TypeError("All inputs must be numbers.")

    if not (0 <= gpu_util <= 100 and 0 <= cpu_util <= 100 and vram_usage >=0 and scene_complexity >=0 and duration > 0 and gpu_type >0):
        raise ValueError("Inputs are outside the valid range.")


    # Feature normalization (min-max scaling)
    gpu_util_norm = gpu_util / 100.0
    cpu_util_norm = cpu_util / 100.0
    #Assuming VRAM has a reasonable upper bound. Adjust 16 as needed based on your data.
    vram_usage_norm = vram_usage / 16.0 #Normalize to a maximum of 16GB VRAM
    duration_norm = 1.0/(1+duration) #Inverse duration, shorter is better

    #Optimization logic (weighted average focusing on minimizing resource usage and maximizing frame rate)
    #Weights are adjusted based on importance.  Experimentation is key here.
    weights = np.array([0.2, 0.3, 0.2, 0.1, 0.2]) #weights for gpu,vram,cpu, scene complexity, duration
    normalized_features = np.array([gpu_util_norm, vram_usage_norm, cpu_util_norm, scene_complexity/10.0, duration_norm]) #Normalize scene complexity as needed
    comfort_score = 1 - np.sum(weights * normalized_features) # Higher score implies better performance

    #GPU type impact (example: better GPU gives a boost)
    gpu_type_boost = min(1, gpu_type/3) #Example boost for higher GPU types. Adjust as needed.
    comfort_score = comfort_score * (1 + 0.1 * gpu_type_boost)  #Apply a small boost based on GPU type


    # Ensure score is within 0-1 range
    comfort_score = max(0, min(1, comfort_score))

    return comfort_score