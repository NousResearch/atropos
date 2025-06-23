import numpy as np


def vr_optimization_function(
    gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type
):
    """
    Optimizes VR performance to maximize comfort score.

    Args:
        gpu_util (float): GPU utilization percentage (0-100).
        vram_usage (float): VRAM usage in GB.
        cpu_util (float): CPU utilization percentage (0-100).
        scene_complexity (float): Scene complexity score (higher is more complex).
        duration (float): Duration of VR experience in seconds.
        gpu_type (float): GPU type (e.g., 2.0 for RTX 2080, 3.5 for RTX 3080).

    Returns:
        float: Optimized VR comfort score between 0 and 1 (higher is better).
               Returns -1 if input is invalid.

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

    if not (
        0 <= gpu_util <= 100
        and 0 <= cpu_util <= 100
        and scene_complexity >= 0
        and duration > 0
        and gpu_type > 0
    ):
        raise ValueError("Invalid input range.")

    # Feature normalization (min-max scaling)
    gpu_util_norm = gpu_util / 100.0
    cpu_util_norm = cpu_util / 100.0
    # Assuming VRAM usage is capped at 24GB for this example. Adjust as needed based on your data
    vram_usage_norm = vram_usage / 24.0

    # Optimization Logic (weighted average with penalties for high resource usage and complexity)

    # Weights are adjusted based on importance. Experiment to find optimal values for your data.
    gpu_weight = 0.3
    cpu_weight = 0.2
    vram_weight = 0.2
    complexity_weight = 0.2
    gpu_type_weight = 0.1

    # Penalties are applied to high resource usage and complexity, reducing the comfort score.
    gpu_penalty = 1 - np.exp(
        -((gpu_util_norm - 0.8) ** 2)
    )  # Penalty for high GPU usage, centered around 80%
    cpu_penalty = 1 - np.exp(
        -((cpu_util_norm - 0.7) ** 2)
    )  # Penalty for high CPU usage, centered around 70%
    vram_penalty = 1 - np.exp(
        -((vram_usage_norm - 0.6) ** 2)
    )  # Penalty for high VRAM usage, centered around 6GB
    complexity_penalty = 1 - np.exp(
        -((scene_complexity) ** 2)
    )  # Penalty for high scene complexity

    comfort_score = (
        gpu_weight * (1 - gpu_penalty) * (1 - gpu_util_norm)
        + cpu_weight * (1 - cpu_penalty) * (1 - cpu_util_norm)
        + vram_weight * (1 - vram_penalty) * (1 - vram_usage_norm)
        + complexity_weight * (1 - complexity_penalty) * (1 / (scene_complexity + 1))
        + gpu_type_weight * (gpu_type / 5.0)
    )  # Reward for better GPU type

    # Ensure the score is within 0-1 range
    comfort_score = max(0, min(1, comfort_score))
    return comfort_score
