import numpy as np

def vr_optimization_function(gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type):
    """
    Optimizes VR performance for maximum comfort score.

    Args:
        gpu_util (float): GPU utilization (%).
        vram_usage (float): VRAM usage (GB).
        cpu_util (float): CPU utilization (%).
        scene_complexity (float): Scene complexity score (higher is more complex).
        duration (float): Duration of the VR experience (seconds).
        gpu_type (float): GPU type (numerical representation, e.g., 1 for GTX 1660, 2 for RTX 3070, etc.).


    Returns:
        float: Optimized VR comfort score between 0 and 1 (inclusive). Returns -1 if input is invalid.

    Raises:
        TypeError: if any input is not a number.
        ValueError: if any input is outside the acceptable range.

    """

    # Input validation
    if not all(isinstance(x, (int, float)) for x in [gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type]):
        raise TypeError("All inputs must be numbers.")

    if not (0 <= gpu_util <= 100 and 0 <= cpu_util <= 100 and vram_usage >= 0 and scene_complexity >= 0 and duration > 0 and gpu_type > 0):
        raise ValueError("Invalid input range. Check your input values.")


    # Feature normalization (min-max scaling)
    gpu_util_norm = gpu_util / 100  #Scale GPU and CPU utilization to 0-1 range.
    cpu_util_norm = cpu_util / 100

    #  Assume reasonable max values for other features based on typical VR usage. Adjust these values if needed based on your specific dataset and hardware.
    max_vram = 24  #GB
    max_scene_complexity = 5
    max_duration = 10 # minutes, converted to seconds

    vram_usage_norm = vram_usage / max_vram
    scene_complexity_norm = scene_complexity / max_scene_complexity
    duration_norm = duration / (max_duration * 60) #scaling duration to be within 0-1.


    #Mathematical optimization logic (weighted average with penalties)
    #Weights are assigned based on domain knowledge – adjust them based on your needs
    weight_gpu = 0.25
    weight_cpu = 0.15
    weight_vram = 0.2
    weight_scene = 0.2
    weight_duration = 0.1
    weight_gpu_type = 0.1 #Higher GPU type should ideally improve score.

    #Penalties for high utilization.  These penalty functions are examples;  you may need to experiment with different functions
    gpu_penalty = 1 - np.exp(-gpu_util_norm**2)  #Increasing penalty as utilization rises exponentially.
    cpu_penalty = 1 - np.exp(-cpu_util_norm**2)
    vram_penalty = vram_usage_norm **2 #Quadratic penalty for high VRAM usage

    # Calculate the weighted score – a higher score indicates better performance and comfort
    comfort_score = (1 - gpu_penalty) * weight_gpu + \
                    (1 - cpu_penalty) * weight_cpu + \
                    (1- vram_penalty) * weight_vram + \
                    (1 - scene_complexity_norm) * weight_scene + \
                    (1 - duration_norm) * weight_duration + \
                    (gpu_type/10) * weight_gpu_type #Scaling gpu_type to be within a reasonable range

    #Ensure the score is within the 0-1 range.
    comfort_score = max(0, min(1, comfort_score))

    return comfort_score