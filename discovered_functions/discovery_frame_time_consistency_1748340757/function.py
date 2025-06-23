import numpy as np


def vr_optimization_function(
    gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type
):
    """
    Optimizes VR performance for frame time consistency to enhance user comfort.

    Args:
        gpu_util (float): GPU utilization percentage (0-100).
        vram_usage (float): VRAM usage in GB.
        cpu_util (float): CPU utilization percentage (0-100).
        scene_complexity (float): Scene complexity score (higher is more complex).
        duration (float): Frame time in seconds.
        gpu_type (float): GPU type (e.g., 2.0 for RTX 2080, 3.5 for RTX 3080).  Higher is better.


    Returns:
        float: Optimized performance score between 0 and 1 (inclusive). Higher score indicates better performance and smoother VR experience. Returns -1 if input is invalid.

    Raises:
        TypeError: if input types are not numerical.
        ValueError: if input values are out of range.

    """
    # Input validation
    if not all(
        isinstance(x, (int, float))
        for x in [gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type]
    ):
        raise TypeError("All inputs must be numerical.")
    if not (
        0 <= gpu_util <= 100
        and 0 <= cpu_util <= 100
        and vram_usage >= 0
        and scene_complexity >= 0
        and duration > 0
        and gpu_type > 0
    ):
        raise ValueError(
            "Invalid input range. Check values for gpu_util, cpu_util, vram_usage, scene_complexity, duration, and gpu_type."
        )

    # Feature normalization (min-max scaling)
    gpu_util_norm = gpu_util / 100  # Normalize to 0-1
    cpu_util_norm = cpu_util / 100  # Normalize to 0-1

    # Avoid division by zero if scene complexity or duration are 0.  Add a small constant to prevent this.
    scene_complexity_norm = scene_complexity / (
        scene_complexity + 1e-9
    )  # Normalize and prevent divide by zero
    duration_norm = 1 / (
        duration + 1e-9
    )  # Inverse duration: shorter frame times are better

    # Weighting factors based on importance (adjust as needed)
    weight_gpu = 0.3
    weight_cpu = 0.2
    weight_vram = 0.1
    weight_scene = 0.2
    weight_duration = 0.2
    weight_gpu_type = 0.1

    # Optimization logic (weighted average with emphasis on frame time and resource efficiency)
    performance_score = (
        weight_gpu * (1 - gpu_util_norm)
        + weight_cpu * (1 - cpu_util_norm)  # Lower GPU usage is better
        + weight_vram * (1 - min(vram_usage / 16, 1))  # Lower CPU usage is better
        + weight_scene  # VRAM usage capped at 16GB, normalized to 0-1. Lower is better
        * (1 / (scene_complexity_norm + 1e-9))
        + weight_duration  # Lower scene complexity is better. Avoid division by zero
        * duration_norm
        + weight_gpu_type * (gpu_type / 5)  # Shorter frame times are better.
    )  # GPU type score normalized by max type

    # Ensure the score is within 0-1
    performance_score = np.clip(performance_score, 0, 1)

    return performance_score
