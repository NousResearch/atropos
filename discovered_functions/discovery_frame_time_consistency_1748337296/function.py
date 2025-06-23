import numpy as np


def vr_optimization_function(
    gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type
):
    """
    Optimizes VR performance score based on input features.  Higher score indicates better frame time consistency.

    Args:
        gpu_util (float): GPU utilization percentage (0-100).
        vram_usage (float): VRAM usage in GB.
        cpu_util (float): CPU utilization percentage (0-100).
        scene_complexity (float): Scene complexity score (higher is more complex).
        duration (float): Frame time duration in seconds.
        gpu_type (float): GPU type (numeric representation, e.g., 2.0 for RTX 2080,  higher is better).


    Returns:
        float: Optimized performance score between 0 and 1 (inclusive). Returns -1 if input is invalid.

    Raises:
        TypeError: if input types are not numeric.
        ValueError: if input values are out of range.

    """

    # Input validation
    if not all(
        isinstance(x, (int, float))
        for x in [gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type]
    ):
        raise TypeError("All inputs must be numeric.")

    if not (
        0 <= gpu_util <= 100
        and 0 <= cpu_util <= 100
        and vram_usage >= 0
        and scene_complexity >= 0
        and duration > 0
        and gpu_type > 0
    ):
        raise ValueError(
            "Invalid input range. Check values for gpu_util, cpu_util, vram_usage, scene_complexity, duration and gpu_type."
        )

    # Feature normalization (min-max scaling)
    gpu_util_norm = gpu_util / 100.0  # Scale GPU utilization to 0-1
    cpu_util_norm = cpu_util / 100.0  # Scale CPU utilization to 0-1

    # Considering that higher scene complexity and VRAM usage can affect performance negatively, we invert their normalization.
    # A lower normalized value for these indicates better performance in this context.
    vram_usage_norm = 1.0 / (
        1 + vram_usage
    )  # inverse scaling of VRAM to handle large values and avoid division by zero

    # We assume there is an upper limit for scene complexity that would result in poor performance
    scene_complexity_norm = 1.0 / (1 + scene_complexity)  # inverse scaling

    # Duration is inversely proportional to performance; shorter is better
    duration_norm = 1.0 / (1 + duration)  # inverse scaling

    # GPU type is assumed to be already scaled relative to a baseline
    gpu_type_norm = (
        gpu_type / 10.0
    )  # Adjust the divisor (10 here) based on your expected range of gpu_type values

    # Optimization logic (weighted average, emphasizing duration and GPU type for VR comfort)
    weights = np.array(
        [0.1, 0.1, 0.1, 0.2, 0.3, 0.2]
    )  # Assign weights based on importance.  Duration and GPU type are given higher weight

    normalized_features = np.array(
        [
            gpu_util_norm,
            vram_usage_norm,
            cpu_util_norm,
            scene_complexity_norm,
            duration_norm,
            gpu_type_norm,
        ]
    )

    performance_score = np.dot(normalized_features, weights)

    # Ensure score is within the 0-1 range
    return np.clip(performance_score, 0, 1)
