import numpy as np


def vr_optimization_function(
    gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type
):
    """
    Optimizes VR performance score based on GPU utilization.

    Args:
        gpu_util (float): GPU utilization percentage (0-100).
        vram_usage (float): VRAM usage in GB.
        cpu_util (float): CPU utilization percentage (0-100).
        scene_complexity (float): Scene complexity score (higher is more complex).
        duration (float): Frame rendering duration in seconds.
        gpu_type (float): GPU type (higher is better).  Assume a numerical representation.

    Returns:
        float: Optimized performance score between 0 and 1 (higher is better).
               Returns -1 if input validation fails.

    Raises:
        TypeError: if any input is not a number.
        ValueError: if any input is out of range.

    """

    # Input validation
    if not all(
        isinstance(i, (int, float))
        for i in [gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type]
    ):
        raise TypeError("All inputs must be numbers.")
    if not (
        0 <= gpu_util <= 100
        and 0 <= cpu_util <= 100
        and scene_complexity >= 0
        and duration > 0
        and gpu_type > 0
    ):  # Allow vram to be arbitrarily large
        raise ValueError(
            "Invalid input range.  Check GPU utilization, CPU utilization, Scene Complexity and Duration."
        )

    # Feature normalization (min-max scaling)
    gpu_util_norm = gpu_util / 100.0  # Normalize to 0-1 range
    cpu_util_norm = cpu_util / 100.0
    # Assuming higher scene complexity reduces performance, hence 1 - scene_complexity
    scene_complexity_norm = 1.0 / (
        1 + scene_complexity
    )  # Inverse scaling to penalize high complexity.
    # duration is inversely proportional to performance
    duration_norm = 1 / (1 + duration)

    # Consider GPU type as a weight
    gpu_type_weight = gpu_type / (
        gpu_type + 5
    )  # example weighting scheme. Adjust as needed.

    # Optimization Logic: prioritize GPU utilization, then minimize VRAM and CPU usage, duration, and maximize GPU type
    # Weights can be adjusted based on priorities.
    performance_score = (
        0.5 * gpu_util_norm
        + 0.15 * (1 - vram_usage / 20.0)
        + 0.15  # Assumes a reasonable maximum VRAM of 20GB. Adjust as needed.
        * (1 - cpu_util_norm)
        + 0.1 * duration_norm
        + 0.1 * scene_complexity_norm
        + 0.1 * gpu_type_weight
    )

    # Clip the score between 0 and 1
    return np.clip(performance_score, 0, 1)
