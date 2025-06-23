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
        duration (float): Frame time in seconds.
        gpu_type (float): GPU type (numerical representation, e.g., 1 for GTX 1660, 2 for RTX 3070, etc.).

    Returns:
        float: Optimized performance score between 0 and 1 (higher is better). Returns -1 if input is invalid.

    Raises:
        TypeError: if any input is not a number.
        ValueError: if any input is outside the expected range.

    """

    # Input validation
    if not all(
        isinstance(x, (int, float))
        for x in [gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type]
    ):
        raise TypeError("All inputs must be numbers.")
    if not 0 <= gpu_util <= 100:
        raise ValueError("GPU utilization must be between 0 and 100.")
    if not vram_usage >= 0:  # VRAM usage can be arbitrarily high.
        raise ValueError("VRAM usage must be non-negative.")
    if not 0 <= cpu_util <= 100:
        raise ValueError("CPU utilization must be between 0 and 100.")
    if not scene_complexity >= 0:  # Scene complexity can be arbitrarily high
        raise ValueError("Scene complexity must be non-negative.")
    if not duration > 0:
        raise ValueError("Duration must be positive.")
    if not gpu_type > 0:  # GPU type should be a positive identifier.
        raise ValueError("GPU type must be positive.")

    # Feature normalization (min-max scaling)
    gpu_util_norm = gpu_util / 100.0
    cpu_util_norm = cpu_util / 100.0
    # Assuming vram_usage is in GB, a reasonable upper bound needs to be established for normalization.  16GB is used here as an example. Adjust as needed.
    vram_usage_norm = np.clip(
        vram_usage / 16.0, 0, 1
    )  # Clip to prevent values greater than 1.
    # Assuming scene complexity and duration are already somewhat normalized, no further scaling might be needed. Adjust as needed depending on your data.

    # Optimization logic (weighted average focusing on GPU utilization)
    # Weights are adjusted to prioritize GPU utilization.  Adjust weights as needed based on priorities.
    gpu_weight = 0.6
    vram_weight = 0.1
    cpu_weight = 0.1
    scene_weight = 0.1
    duration_weight = 0.1  # Lower weight because a slightly longer frame time is less critical than high GPU utilization.

    optimized_score = (
        (gpu_weight * gpu_util_norm)
        - (vram_weight * vram_usage_norm)
        - (cpu_weight * cpu_util_norm)
        - (scene_weight * scene_complexity)
        - (duration_weight * duration)
    )

    # Handle potential negative scores.  This adjustment prevents negative scores. Adjust as needed.
    optimized_score = max(0, optimized_score)

    # Ensure score is within 0-1 range
    optimized_score = np.clip(optimized_score, 0, 1)

    return optimized_score
