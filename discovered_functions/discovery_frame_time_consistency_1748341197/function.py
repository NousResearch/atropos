import numpy as np


def vr_optimization_function(
    gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type
):
    """
    Optimizes VR performance for frame time consistency, aiming for a score between 0 and 1 (higher is better).

    Args:
        gpu_util (float): GPU utilization percentage (0-100).
        vram_usage (float): VRAM usage in GB.
        cpu_util (float): CPU utilization percentage (0-100).
        scene_complexity (float): Scene complexity score (higher is more complex).
        duration (float): Frame duration in seconds.
        gpu_type (float): GPU type (e.g., 2.0 for a specific model, higher number for better GPUs).


    Returns:
        float: Optimized performance score between 0 and 1. Returns -1 if input is invalid.
    """

    # Input validation
    if not all(
        isinstance(x, (int, float))
        for x in [gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type]
    ):
        print("Error: Input values must be numeric.")
        return -1
    if not (
        0 <= gpu_util <= 100
        and 0 <= cpu_util <= 100
        and scene_complexity >= 0
        and duration > 0
        and gpu_type > 0
    ):
        print("Error: Invalid input ranges.")
        return -1

    # Feature normalization (min-max scaling)

    gpu_util_norm = gpu_util / 100.0  # Normalize to 0-1 range.
    cpu_util_norm = cpu_util / 100.0  # Normalize to 0-1 range.

    #  Assume VRAM and Scene Complexity have reasonable upper bounds for normalization, adjust these if needed.
    vram_max = 24  # Example:  adjust based on your expected VRAM range
    scene_complexity_max = 10  # Example: adjust based on your scene complexity scoring
    vram_usage_norm = vram_usage / vram_max
    scene_complexity_norm = scene_complexity / scene_complexity_max

    # GPU type weighting - higher is better
    gpu_type_weight = np.clip(gpu_type / 5, 0, 1)  # example scaling: cap at 1

    # Optimization logic (weighted average focusing on frame time consistency and resource utilization)
    # Prioritize lower duration and lower resource utilization.  GPU is given more importance.
    frame_time_weight = 0.5  # weight on Frame time.
    resource_weight = 0.5  # weight on resource usage.

    optimized_score = (
        frame_time_weight * (1 / (duration + 0.0001))
        + resource_weight
        * (
            1
            - (gpu_util_norm + cpu_util_norm + vram_usage_norm + scene_complexity_norm)
            / 4
        )
        * gpu_type_weight
    )  # Avoid division by zero.

    # Ensure score is within 0-1 range
    optimized_score = np.clip(optimized_score, 0, 1)

    return optimized_score
