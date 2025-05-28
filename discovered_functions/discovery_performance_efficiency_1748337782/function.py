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
        gpu_type (float): GPU type (numerical representation, higher is better).

    Returns:
        float: Optimized performance score between 0 and 1 (higher is better). 
               Returns -1 if input validation fails.

    Raises:
        TypeError: if any input is not a number.
        ValueError: if any input is out of range.

    """

    # Input validation
    try:
        if not all(isinstance(x, (int, float)) for x in [gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type]):
            raise TypeError("All inputs must be numbers.")
        if not 0 <= gpu_util <= 100 or not 0 <= cpu_util <= 100 or not scene_complexity >=0 or not duration > 0 or not gpu_type >=0:
            raise ValueError("Invalid input range.  Check GPU utilization, CPU utilization, scene complexity, and duration.")

    except (TypeError, ValueError) as e:
        print(f"Error: {e}")
        return -1


    # Feature normalization (min-max scaling)
    gpu_util_norm = gpu_util / 100  #already normalized
    vram_usage_norm = vram_usage / 16  # Assuming a maximum of 16GB VRAM for normalization. Adjust as needed.
    cpu_util_norm = cpu_util / 100 #already normalized
    #Assuming scene complexity is already scaled appropriately. No normalization needed.
    scene_complexity_norm = scene_complexity
    duration_norm = 1/(1+duration) #inverse scaling for duration, lower duration is better

    gpu_type_norm = gpu_type / 5 # Assuming a maximum GPU type score of 5. Adjust as needed.

    # Optimization logic (weights can be adjusted based on priorities)
    gpu_weight = 0.5  # Higher weight given to GPU utilization
    vram_weight = 0.1
    cpu_weight = 0.1
    scene_weight = 0.2  
    duration_weight = 0.1
    gpu_type_weight = 0.1


    optimized_score = (gpu_weight * gpu_util_norm) + (vram_weight * (1 - vram_usage_norm)) + (cpu_weight * (1-cpu_util_norm)) + (scene_weight * (1/(1+scene_complexity_norm))) + (duration_weight * duration_norm) + (gpu_type_weight * gpu_type_norm)


    #Ensure score is within 0-1 range
    optimized_score = np.clip(optimized_score, 0, 1)

    return optimized_score