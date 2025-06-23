import numpy as np


def vr_optimization_function(
    gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type
):
    """
    Optimizes VR performance score based on GPU utilization.  Higher score indicates better performance.

    Args:
        gpu_util (float): GPU utilization percentage (0-100).
        vram_usage (float): VRAM usage in GB.
        cpu_util (float): CPU utilization percentage (0-100).
        scene_complexity (float): Scene complexity score (higher is more complex).
        duration (float): Frame rendering time in seconds.
        gpu_type (float): GPU type (e.g., 1.0, 2.0, 3.5 representing different GPUs).


    Returns:
        float: Optimized performance score between 0 and 1 (inclusive). Returns -1 if input is invalid.
    """

    # Input validation
    if not all(
        isinstance(x, (int, float))
        for x in [gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type]
    ):
        print("Error: Invalid input type. All inputs must be numeric.")
        return -1
    if (
        not 0 <= gpu_util <= 100
        or not 0 <= cpu_util <= 100
        or not scene_complexity >= 0
        or not duration > 0
        or not gpu_type > 0
    ):
        print(
            "Error: Invalid input range. Check GPU utilization, CPU utilization, scene complexity, duration, and GPU type."
        )
        return -1

    # Feature normalization (min-max scaling)

    gpu_util_norm = gpu_util / 100.0  # Normalize GPU utilization to 0-1 range
    cpu_util_norm = cpu_util / 100.0  # Normalize CPU utilization to 0-1 range

    # Assuming VRAM usage is generally lower than 16GB. Adjust upper limit as needed based on your system's VRAM.
    vram_usage_norm = vram_usage / 16.0  # Normalize VRAM usage. Adjust 16.0 as needed.
    scene_complexity_norm = (
        scene_complexity / 5
    )  # Normalize Scene Complexity. Adjust 5 based on your maximum Scene Complexity Value.
    duration_norm = 1 / (
        duration + 0.001
    )  # Inverse duration. Smaller duration is better, so we invert. Adding 0.001 to avoid division by zero.
    gpu_type_norm = gpu_type / 5  # Assuming maximum GPU Type 5, adjust as needed

    # Optimization logic: prioritizing GPU utilization and minimizing duration and VRAM usage.
    # Weighting factors adjust the importance of each feature. Adjust weights as needed based on priorities.

    gpu_weight = 0.6  # GPU utilization is the most important factor
    duration_weight = 0.2  # Minimizing rendering time is important
    vram_weight = 0.1  # VRAM Usage should be minimized
    cpu_weight = 0.05  # CPU usage is less critical in this optimization scenario
    scene_complexity_weight = (
        0.05  # Scene complexity should have a smaller effect than other factors
    )

    optimized_score = (
        (gpu_weight * gpu_util_norm)
        + (duration_weight * duration_norm)
        - (vram_weight * vram_usage_norm)
        - (cpu_weight * cpu_util_norm)
        - (scene_complexity_weight * scene_complexity_norm)
    )

    # Ensure the score stays within the 0-1 range.
    optimized_score = np.clip(optimized_score, 0, 1)

    return optimized_score
