import numpy as np


def vr_optimization_function(
    gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type
):
    """
    Optimizes VR performance for maximum comfort score.

    Args:
        gpu_util (float): GPU utilization percentage (0-100).
        vram_usage (float): VRAM usage in GB.
        cpu_util (float): CPU utilization percentage (0-100).
        scene_complexity (float): Scene complexity score (higher is more complex).
        duration (float): Duration of VR experience in seconds.
        gpu_type (float): GPU type (e.g., 2.0 for mid-range, 3.5 for high-end).

    Returns:
        float: Optimized performance score between 0 and 1 (higher is better).
               Returns -1 if input validation fails.

    Raises:
        TypeError: if any input is not a number.
        ValueError: if any input is outside the acceptable range.

    """

    # Input validation
    if not all(
        isinstance(x, (int, float))
        for x in [gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type]
    ):
        raise TypeError("All inputs must be numbers.")
    if not (
        0 <= gpu_util <= 100
        and 0 <= cpu_util <= 100
        and vram_usage >= 0
        and scene_complexity >= 0
        and duration > 0
        and gpu_type > 0
    ):
        raise ValueError("Invalid input values. Check ranges for each parameter.")

    # Feature normalization (min-max scaling) -  Improves robustness across different hardware/scenes
    gpu_util_norm = gpu_util / 100.0
    cpu_util_norm = cpu_util / 100.0
    vram_usage_norm = (
        vram_usage / 16
    )  # Assuming a maximum of 16GB VRAM as a reasonable upper bound. Adjust as needed.

    # Consider GPU type impact - higher is better
    gpu_type_weight = (
        1 + (gpu_type - 2) / 2
    )  # Weighs GPU type impact (mid-range as baseline)

    # Optimization logic (weighted average with penalties for high utilization and complexity)
    # Weights are chosen empirically - adjust based on experimentation and VR system specific priorities
    comfort_score = (
        0.3 * (1 - gpu_util_norm)
        + 0.25 * (1 - cpu_util_norm)  # Penalize high GPU utilization
        + 0.2 * (1 - vram_usage_norm)  # Penalize high CPU utilization
        + 0.15 * (1 / (1 + scene_complexity))  # Penalize high VRAM usage
        + 0.1  # Inverse relationship with scene complexity
        * (1 / duration)
        * gpu_type_weight  # Reward shorter duration and better GPU type
    )

    # Clamp the score to 0-1 range
    return max(0, min(1, comfort_score))
