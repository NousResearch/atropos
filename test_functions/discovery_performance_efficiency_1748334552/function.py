# Auto-generated VR optimization function
# Type: polynomial
# Fitness: 0.0000

import numpy as np


def vr_optimization_function(features):
    """
    VR optimization function discovered through evolutionary algorithm

    Args:
        features: Array of input features [gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type]

    Returns:
        Optimized output value
    """

    # Feature weights
    weights = [
        -1.7520140247728957,
        1.16865982623378,
        -0.22217354108158904,
        -1.3510415026080107,
        1.0467079942883317,
        -1.802607352540241,
    ]
    x = np.dot(features, weights[: len(features)])

    # Polynomial function
    result = (
        -0.9717716282884401 * x**3
        + 0.173355474869477 * x**2
        + 0.6194853611721087 * x
        + 0.6066102847243802
    )

    return np.clip(result, -1e6, 1e6)


# Example usage:
# result = vr_optimization_function([gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type])
