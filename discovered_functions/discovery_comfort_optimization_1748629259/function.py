import numpy as np


def vr_optimization_function(
    gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type
):
    """
    Predicts VR comfort score based on performance features.

    Args:
        gpu_util (float): GPU utilization (%).
        vram_usage (float): VRAM usage (GB).
        cpu_util (float): CPU utilization (%).
        scene_complexity (float): Scene complexity score (higher is more complex).
        duration (float): Frame duration (seconds).
        gpu_type (float): GPU type (numerical encoding, higher is better).


    Returns:
        float: Optimized VR comfort score between 0 and 1 (inclusive).
               Returns -1 if input validation fails.

    Raises:
        TypeError: if any input is not a number.
        ValueError: if any input is out of range.

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
        and 0 <= scene_complexity <= 10
        and duration > 0
        and gpu_type > 0
    ):  # Adjust ranges as needed
        raise ValueError(
            "Invalid input range. Check gpu_util, cpu_util, scene_complexity, duration and gpu_type."
        )

    # Feature normalization (min-max scaling)
    gpu_util_norm = gpu_util / 100
    cpu_util_norm = cpu_util / 100
    vram_usage_norm = (
        vram_usage / 8
    )  # Assuming a reasonable max VRAM of 8GB. Adjust as needed.
    scene_complexity_norm = (
        scene_complexity / 10
    )  # Assuming a reasonable max scene complexity of 10. Adjust as needed.
    duration_norm = 1 / (
        1 + duration
    )  # Inverse scaling to penalize higher frame durations

    # Optimization Logic (weighted average with penalties)

    # Weights are adjusted based on their importance in VR comfort. Adjust as needed based on your data and requirements.
    comfort_score = (
        0.3 * (1 - gpu_util_norm)
        + 0.2 * (1 - cpu_util_norm)
        + 0.2 * (1 - vram_usage_norm)
        + 0.2 * (1 - scene_complexity_norm)
        + 0.1 * duration_norm
        + 0.1 * (gpu_type / 5)
    )  # Assuming a max gpu_type of 5

    # Ensure score is within 0-1 range
    comfort_score = np.clip(comfort_score, 0, 1)

    return comfort_score
