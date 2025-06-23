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
        gpu_type (float): GPU type (numeric representation, e.g., 1 for RTX 3080, 2 for RTX 4090).


    Returns:
        float: Optimized performance score between 0 and 1 (higher is better). 
               Returns -1 if input validation fails.

    Raises:
        TypeError: If any input is not a number.
        ValueError: If any input is out of range.

    """

    # Input validation
    if not all(isinstance(x, (int, float)) for x in [gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type]):
        raise TypeError("All inputs must be numbers.")

    if not (0 <= gpu_util <= 100 and 0 <= cpu_util <= 100 and vram_usage >= 0 and scene_complexity >= 0 and duration > 0 and gpu_type >0):
        raise ValueError("Input values are out of range.")


    #Feature Normalization (min-max scaling)
    gpu_util_norm = gpu_util / 100  
    cpu_util_norm = cpu_util / 100
    
    # Assuming a reasonable VRAM maximum for normalization. Adjust as needed based on your target hardware
    max_vram = 24 #GB
    vram_usage_norm = vram_usage / max_vram

    #Normalize scene complexity.  Assumes a reasonable range; adjust if needed.
    max_scene_complexity = 10 # Adjust this based on your scene complexity scoring system.
    scene_complexity_norm = scene_complexity/max_scene_complexity

    #duration normalization (inverse relationship: lower is better)
    #We assume a target frametime; adjust based on the desired frame rate (e.g., 1/90 for 90fps)
    target_frame_time = 1/90 #seconds
    duration_norm =  np.exp(-(duration - target_frame_time)**2/(2*(target_frame_time/3)**2)) #Gaussian weighting


    #Optimization Logic (weighted average emphasizing frame time and resource usage)
    weights = np.array([0.4, 0.2, 0.2, 0.1, 0.1]) #weights for duration, gpu, cpu, vram, scene complexity

    normalized_features = np.array([duration_norm, 1-gpu_util_norm, 1-cpu_util_norm, 1-vram_usage_norm, 1-scene_complexity_norm ])

    performance_score = np.dot(normalized_features, weights)

    #Handle potential score out of bounds due to numerical error
    performance_score = np.clip(performance_score, 0, 1)


    return performance_score