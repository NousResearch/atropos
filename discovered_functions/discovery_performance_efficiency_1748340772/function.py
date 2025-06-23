import numpy as np


def vr_optimization_function(
    gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type
):
    """
    Optimizes VR performance focusing on GPU utilization.

    Args:
        gpu_util (float): GPU utilization percentage (0-100).
        vram_usage (float): VRAM usage in GB.
        cpu_util (float): CPU utilization percentage (0-100).
        scene_complexity (float): Scene complexity score (higher is more complex).
        duration (float): Frame rendering duration in seconds.
        gpu_type (float): GPU type (higher is better, e.g., based on a ranking system).

    Returns:
        float: Optimized performance score between 0 and 1 (higher is better).
               Returns -1 if input validation fails.

    Raises:
        TypeError: if any input is not a number.
        ValueError: if any input is out of the expected range.

    """

    # Input validation
    if not all(
        isinstance(x, (int, float))
        for x in [gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type]
    ):
        raise TypeError("All inputs must be numbers.")
    if not 0 <= gpu_util <= 100:
        raise ValueError("GPU utilization must be between 0 and 100.")
    if not vram_usage >= 0:  # VRAM usage can be arbitrarily large.
        raise ValueError("VRAM usage must be non-negative.")
    if not 0 <= cpu_util <= 100:
        raise ValueError("CPU utilization must be between 0 and 100.")
    if not scene_complexity >= 0:  # Scene complexity can be arbitrarily large
        raise ValueError("Scene complexity must be non-negative.")
    if not duration > 0:
        raise ValueError("Duration must be positive.")
    if not gpu_type >= 0:  # GPU type ranking can be arbitrarily large
        raise ValueError("GPU type must be non-negative.")

    # Feature normalization (min-max scaling)
    gpu_util_norm = gpu_util / 100.0
    cpu_util_norm = cpu_util / 100.0

    # Consider VRAM usage as a penalty. High VRAM usage negatively impacts performance, especially when focusing on GPU utilization.

    vram_penalty = 1 / (
        1 + vram_usage
    )  # inverse relationship - higher usage, lower penalty

    # Optimize for GPU utilization, penalizing high CPU usage, high scene complexity, and long duration.
    # GPU type acts as a positive factor.
    optimization_score = (
        (gpu_util_norm * 0.6)
        + (vram_penalty * 0.2)
        - (cpu_util_norm * 0.1)
        - (scene_complexity * 0.05)
        - (duration * 0.05)
        + (gpu_type * 0.1)
    )

    # Ensure score is within 0 and 1 range
    optimization_score = max(0, min(1, optimization_score))

    return optimization_score
