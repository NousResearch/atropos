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
        gpu_type (float): GPU type (numerical representation, e.g., 2.0 for RTX 2080).


    Returns:
        float: Optimized performance score between 0 and 1 (higher is better). 
               Returns None if input is invalid.
    """

    # Input validation
    if not all(isinstance(x, (int, float)) for x in [gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type]):
        print("Error: All inputs must be numeric.")
        return None
    if not 0 <= gpu_util <= 100 or not 0 <= cpu_util <= 100:
        print("Error: GPU and CPU utilization must be between 0 and 100.")
        return None
    if vram_usage < 0 or scene_complexity < 0 or duration < 0 or gpu_type < 0:
        print("Error: VRAM usage, scene complexity, duration and GPU type cannot be negative.")
        return None


    # Feature normalization (min-max scaling)
    gpu_util_norm = gpu_util / 100  #Normalize to 0-1 range
    cpu_util_norm = cpu_util / 100  #Normalize to 0-1 range

    #Define weights based on importance (adjust as needed based on domain expertise)
    weights = {
        'gpu_util': 0.3,  #higher weight for GPU utilization
        'vram_usage': 0.2,
        'cpu_util': 0.2,
        'scene_complexity': 0.1, #Lower weight to avoid penalizing complex scenes too heavily
        'duration': 0.1,      #Lower weight for duration
        'gpu_type': 0.1      # Moderate weight for GPU type
    }

    # Optimization logic (weighted average with penalties for high resource usage)

    # Penalty functions for high resource usage.  These are examples, adjust as needed.
    gpu_penalty = np.exp(gpu_util_norm - 0.8) -1 if gpu_util_norm > 0.8 else 0  #Increased penalty above 80%
    cpu_penalty = np.exp(cpu_util_norm - 0.7) -1 if cpu_util_norm > 0.7 else 0 #Increased penalty above 70%
    vram_penalty =  np.exp(vram_usage/10) -1 if vram_usage > 6 else 0 #Increased penalty above 6GB


    weighted_score = (weights['gpu_util'] * (1 - gpu_penalty) * (1- gpu_util_norm) + #Prioritize lower GPU usage
                      weights['cpu_util'] * (1 - cpu_penalty) * (1 - cpu_util_norm) + #Prioritize lower CPU usage
                      weights['vram_usage'] * (1 - vram_penalty) * (1 - (vram_usage/10)) + #Prioritize lower VRAM usage
                      weights['scene_complexity'] * (1 / (1 + scene_complexity)) + #Inverse relationship with complexity
                      weights['duration'] * (1 / (1 + duration)) + #Inverse relationship with duration
                      weights['gpu_type'] * (gpu_type / 4) ) # Higher GPU type means better performance

    # Ensure score is within the range [0, 1]
    optimized_score = np.clip(weighted_score, 0, 1)

    return optimized_score