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
        duration (float): Frame time duration in seconds.
        gpu_type (float): GPU type (numerical representation, e.g., 2.0 for RTX 2080,  etc.).

    Returns:
        float: Optimized performance score between 0 and 1. Returns -1 if input is invalid.
    """

    # Input validation
    if not all(
        isinstance(arg, (int, float))
        for arg in [
            gpu_util,
            vram_usage,
            cpu_util,
            scene_complexity,
            duration,
            gpu_type,
        ]
    ):
        print("Error: All inputs must be numeric.")
        return -1
    if (
        not 0 <= gpu_util <= 100
        or not 0 <= cpu_util <= 100
        or not duration > 0
        or not scene_complexity >= 0
    ):
        print(
            "Error: Invalid input range for gpu_util, cpu_util, duration or scene_complexity."
        )
        return -1
    if vram_usage < 0:
        print("Error: Invalid input range for vram_usage.")
        return -1

    # Feature normalization (min-max scaling)
    gpu_util_norm = gpu_util / 100.0
    cpu_util_norm = cpu_util / 100.0
    vram_usage_norm = (
        vram_usage / 8
    )  # Assuming a maximum of 8GB VRAM as a reasonable upper bound. Adjust if needed.

    # Optimization logic: Weighted average focusing on frame time and resource utilization balance.
    # Lower duration and resource utilization are preferred.  Weights are adjusted based on importance.

    weight_duration = 0.5  # High weight for frame time consistency
    weight_gpu = 0.2  # Moderate weight for GPU usage
    weight_cpu = 0.2  # Moderate weight for CPU usage
    weight_vram = 0.1  # Lower weight for VRAM usage (relative to GPU/CPU)

    # Penalty for high scene complexity
    complexity_penalty = 1.0 / (
        1 + scene_complexity
    )  # Decreases as complexity increases

    # Inverse duration to prioritize lower frame times
    duration_score = 1 / duration if duration > 0 else 0

    optimized_score = (
        weight_duration * duration_score
        + weight_gpu * (1 - gpu_util_norm)
        + weight_cpu * (1 - cpu_util_norm)
        + weight_vram * (1 - vram_usage_norm)
    ) * complexity_penalty

    # Ensure score is within 0-1 range
    optimized_score = np.clip(optimized_score, 0, 1)

    return optimized_score
