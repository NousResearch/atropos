import numpy as np

def vr_optimization_function(gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type):
    """
    Optimizes VR performance to maximize comfort scores.

    Args:
        gpu_util (float): GPU utilization percentage (0-100).
        vram_usage (float): VRAM usage in GB.
        cpu_util (float): CPU utilization percentage (0-100).
        scene_complexity (float): Scene complexity score (higher is more complex).
        duration (float): Frame duration in seconds.
        gpu_type (float): GPU type (numerical representation, e.g., 1 for GTX 1660, 2 for RTX 2070, etc.).

    Returns:
        float: Optimized performance score between 0 and 1 (higher is better).  Returns -1 if input is invalid.
    """

    # Input validation
    if not all(isinstance(x, (int, float)) for x in [gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type]):
        print("Error: Input values must be numeric.")
        return -1
    if not all(0 <= x <= 100 for x in [gpu_util, cpu_util]) :
        print("Error: GPU and CPU utilization must be between 0 and 100.")
        return -1
    if not all (x >= 0 for x in [vram_usage, scene_complexity, duration, gpu_type]):
        print("Error: VRAM usage, scene complexity, duration, and GPU type must be non-negative.")
        return -1

    # Feature normalization (min-max scaling)
    gpu_util_norm = gpu_util / 100  # Scale GPU utilization to 0-1
    cpu_util_norm = cpu_util / 100  # Scale CPU utilization to 0-1
    
    #Assuming reasonable upper bounds for other features based on typical VR usage. Adjust if necessary.
    vram_usage_norm = min(vram_usage / 16,1) #Scale VRAM to 0-1 assuming a maximum of 16GB is acceptable
    scene_complexity_norm = min(scene_complexity / 10, 1) #Scale scene complexity to 0-1, assuming 10 is a high complexity
    duration_norm = min(duration / 0.05,1) #Scale duration to 0-1, assuming 0.05 seconds as a comfortable frame duration
    gpu_type_norm = gpu_type / 10 #Normalize GPU type; needs context dependent upper bound


    #Mathematical optimization logic (weighted average with penalties)
    #Weights are adjusted based on perceived importance for VR comfort. Adjust as needed.
    comfort_score = 0.3 * (1 - gpu_util_norm) + 0.2 * (1 - vram_usage_norm) + 0.2 * (1 - cpu_util_norm) + 0.2 * (1 - scene_complexity_norm) + 0.1 * (1 - duration_norm)


    # Penalty for high GPU utilization and high VRAM usage (combined)
    resource_penalty = min(1, (gpu_util_norm + vram_usage_norm) * 0.5) #This increases the penalty if both GPU and VRAM are high
    comfort_score -= resource_penalty * 0.1


    #Consider GPU type - higher value means better GPU (this is arbitrary and needs modification based on actual GPU types and performance).
    comfort_score += gpu_type_norm * 0.05

    # Ensure score is within 0-1 range
    comfort_score = max(0, min(1, comfort_score))

    return comfort_score