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
        duration (float): Frame time duration in milliseconds.
        gpu_type (float): GPU type (numerical representation, e.g., 2.0 for RTX 2080, 3.5 for RTX 3090).


    Returns:
        float: Optimized performance score between 0 and 1. Returns -1 if input is invalid.

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
        raise ValueError(
            "Invalid input range. Check GPU utilization, CPU utilization, VRAM usage, scene complexity, and duration."
        )

    # Feature Normalization (min-max scaling) to handle different scales and units.
    gpu_util_norm = gpu_util / 100
    cpu_util_norm = cpu_util / 100
    vram_usage_norm = (
        vram_usage / 16
    )  # Assuming a max VRAM of 16GB as a reasonable upper bound. Adjust if needed.
    # Assuming scene complexity and gpu_type are already somewhat normalized by their nature
    duration_norm = 1 / (1 + duration)  # Inverse scaling for duration; lower is better

    # Optimization Logic:  Weighted average focusing on frame time consistency and resource usage
    # Weights are adjusted based on importance.  These can be tuned based on specific needs.
    weight_duration = 0.5  # Frame time is crucial for VR comfort
    weight_gpu = 0.2  # GPU utilization should be high but not maxed out
    weight_cpu = 0.15  # CPU impact on frame time
    weight_vram = 0.1  # VRAM usage
    weight_complexity = 0.05  # Scene complexity impact

    # Penalize high GPU usage above a threshold (e.g., 90%) to prevent overheating and instability.
    gpu_penalty = (
        max(0, gpu_util_norm - 0.9) ** 2
    )  # Quadratic penalty for values above 90%

    optimized_score = (
        weight_duration * duration_norm
        + weight_gpu * (gpu_util_norm - gpu_penalty)
        + weight_cpu * (1 - cpu_util_norm)
        + weight_vram * (1 - vram_usage_norm)
        + weight_complexity * (1 / (1 + scene_complexity))
    )  # Inverse scaling for complexity

    # Ensure score is within 0-1 range.  Clipping avoids potential floating-point errors.
    return np.clip(optimized_score, 0, 1)
