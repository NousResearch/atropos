import numpy as np


def vr_optimization_function(
    gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type
):
    """
    Optimizes VR performance for frame time consistency, aiming for a score between 0 and 1.

    Args:
        gpu_util (float): GPU utilization percentage (0-100).
        vram_usage (float): VRAM usage in GB.
        cpu_util (float): CPU utilization percentage (0-100).
        scene_complexity (float): Scene complexity score (higher is more complex).
        duration (float): Frame duration in seconds.
        gpu_type (float): GPU type (numerical representation, e.g., 1 for GTX 1080, 2 for RTX 2080 etc.).

    Returns:
        float: Optimized performance score (0-1, higher is better). Returns -1 if input is invalid.
    """

    # Input validation
    try:
        if not all(
            isinstance(i, (int, float))
            for i in [
                gpu_util,
                vram_usage,
                cpu_util,
                scene_complexity,
                duration,
                gpu_type,
            ]
        ):
            raise ValueError("All inputs must be numeric.")
        if not 0 <= gpu_util <= 100 or not 0 <= cpu_util <= 100:
            raise ValueError("GPU and CPU utilization must be between 0 and 100.")
        if vram_usage < 0 or scene_complexity < 0 or duration < 0 or gpu_type < 0:
            raise ValueError(
                "VRAM usage, scene complexity, duration, and GPU type cannot be negative."
            )

    except ValueError as e:
        print(f"Error: {e}")
        return -1

    # Feature normalization (min-max scaling)
    gpu_util_norm = gpu_util / 100  # Normalize to 0-1
    cpu_util_norm = cpu_util / 100  # Normalize to 0-1

    #  We assume scene complexity is already relatively scaled; we'll just cap it
    scene_complexity_norm = min(scene_complexity, 5) / 5  # Cap at 5 and normalize

    # GPU type influence -  assuming higher number indicates better performance
    gpu_type_influence = gpu_type / (
        gpu_type + 1
    )  # Example influence function. Adjust as needed based on your GPU type mapping.

    # Frame time consistency penalty
    frame_time_consistency = np.exp(
        -duration
    )  # Exponentially penalize longer durations

    # Optimization logic (weighted average) - adjust weights as needed
    weights = np.array(
        [0.25, 0.15, 0.25, 0.2, 0.15, 0.15]
    )  # Weights for gpu,vram,cpu,scene,duration,gpu_type respectively

    normalized_features = np.array(
        [
            gpu_util_norm,
            vram_usage,
            cpu_util_norm,
            scene_complexity_norm,
            frame_time_consistency,
            gpu_type_influence,
        ]
    )

    # Avoid division by zero if all weights are zero.
    if np.sum(weights) == 0:
        return 0

    performance_score = np.sum(weights * normalized_features) / np.sum(weights)

    # Ensure score is within 0-1 range
    performance_score = max(0, min(1, performance_score))

    return performance_score
