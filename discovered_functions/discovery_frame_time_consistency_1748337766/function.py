import numpy as np

def vr_optimization_function(gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type):
    """
    Optimizes VR performance for frame time consistency and comfort.

    Args:
        gpu_util (float): GPU utilization percentage (0-100).
        vram_usage (float): VRAM usage in GB.
        cpu_util (float): CPU utilization percentage (0-100).
        scene_complexity (float): Scene complexity score (higher is more complex).
        duration (float): Frame time in milliseconds.
        gpu_type (float): GPU type (numerical representation, higher is better).


    Returns:
        float: Optimized performance score between 0 and 1 (higher is better). 
               Returns -1 if input validation fails.

    Raises:
        TypeError: If any input is not a number.
        ValueError: If any input is outside the acceptable range.

    """
    # Input validation
    if not all(isinstance(x, (int, float)) for x in [gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type]):
        raise TypeError("All inputs must be numbers.")

    if not (0 <= gpu_util <= 100 and 0 <= cpu_util <= 100 and duration > 0 and scene_complexity >=0 and vram_usage >=0 and gpu_type > 0):
        raise ValueError("Invalid input range. Check GPU utilization, CPU utilization, duration, scene complexity, VRAM usage and GPU type.")


    # Feature normalization (min-max scaling)
    gpu_util_norm = gpu_util / 100.0
    cpu_util_norm = cpu_util / 100.0
    # Assuming a reasonable upper bound for VRAM usage (adjust as needed)
    vram_usage_norm = min(vram_usage / 16.0, 1.0) #Normalizes to a max of 16GB
    # scene complexity normalization depends on the expected range, adjust accordingly.
    scene_complexity_norm = min(scene_complexity / 10.0,1.0) #Normalized to a max complexity of 10
    # Assuming a target frame time of 16ms.  Adjust as needed for your target.
    duration_norm = 1.0 / (1 + (duration / 16.0))  #Inversely proportional to frame time.


    # Optimization logic (weighted average focusing on frame time consistency and resource utilization)

    #Weights adjusted based on importance for VR comfort. Frame time is most crucial.
    weight_duration = 0.5
    weight_gpu = 0.2
    weight_cpu = 0.2
    weight_vram = 0.05
    weight_scene = 0.05


    optimized_score = (weight_duration * duration_norm + 
                       weight_gpu * (1 - gpu_util_norm) +  #Lower GPU usage is better
                       weight_cpu * (1 - cpu_util_norm) + #Lower CPU usage is better
                       weight_vram * (1 - vram_usage_norm) + #Lower VRAM usage is better
                       weight_scene * (1/(1+scene_complexity_norm)) #Lower scene complexity is better
                      )

    # Ensure score is within [0, 1] range.
    optimized_score = np.clip(optimized_score, 0, 1)


    return optimized_score