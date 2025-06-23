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
        duration (float): Duration of the VR experience in seconds.
        gpu_type (float): GPU type (numerical representation, e.g., 2.0 for RTX 2080, etc.).

    Returns:
        float: Optimized VR comfort score between 0 and 1 (inclusive).  Returns -1 if input is invalid.

    Raises:
        TypeError: If any input is not a number.
        ValueError: If any input is out of range.

    """

    # Input validation
    if not all(
        isinstance(x, (int, float))
        for x in [gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type]
    ):
        raise TypeError("All inputs must be numbers.")
    if not 0 <= gpu_util <= 100:
        raise ValueError("GPU utilization must be between 0 and 100.")
    if not vram_usage >= 0:  # VRAM usage can be arbitrarily high but not negative.
        raise ValueError("VRAM usage must be non-negative.")
    if not 0 <= cpu_util <= 100:
        raise ValueError("CPU utilization must be between 0 and 100.")
    if (
        not scene_complexity >= 0
    ):  # Scene complexity can be arbitrarily high but not negative.
        raise ValueError("Scene complexity must be non-negative.")
    if not duration > 0:
        raise ValueError("Duration must be positive.")
    # GPU type validation is application specific and omitted for generality.

    # Feature normalization (min-max scaling)
    gpu_util_norm = gpu_util / 100  # Scale GPU utilization to 0-1
    cpu_util_norm = cpu_util / 100  # Scale CPU utilization to 0-1

    # considering a hypothetical maximum VRAM usage and Scene complexity for normalization.  Adjust as needed based on your data.
    max_vram = 16  # Example
    max_scene_complexity = 5  # Example

    vram_usage_norm = vram_usage / max_vram
    scene_complexity_norm = scene_complexity / max_scene_complexity

    # Mathematical optimization logic (example using weighted average)
    # Weights are adjusted based on the importance of each feature for VR comfort.  This needs careful tuning based on your data analysis and experiements.
    weights = np.array(
        [0.25, 0.2, 0.2, 0.15, 0.1, 0.1]
    )  # weights for gpu_util_norm, vram_usage_norm, cpu_util_norm, scene_complexity_norm, duration, gpu_type respectively.

    normalized_features = np.array(
        [
            gpu_util_norm,
            vram_usage_norm,
            cpu_util_norm,
            scene_complexity_norm,
            1 / duration,
            1 / gpu_type,
        ]
    )  # Duration and gpu_type are inversely proportional to comfort in this example. Adjust as needed.

    comfort_score = np.dot(weights, normalized_features)

    # clip to 0-1 range
    comfort_score = np.clip(comfort_score, 0, 1)

    return comfort_score
