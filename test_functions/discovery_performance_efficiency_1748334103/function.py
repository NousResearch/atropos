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
        -1.7195188866848152,
        0.5966045106178974,
        -0.35159253635382814,
        0.6851269947344978,
        0.005824026968042961,
        -0.936694707724218,
    ]
    x = np.dot(features, weights[: len(features)])

    # Polynomial function
    result = (
        0.32905126833841103 * x**3
        + 0.31997985379646066 * x**2
        + -0.42457118753421463 * x
        + -0.619282581095919
    )

    return np.clip(result, -1e6, 1e6)


# Example usage:
# result = vr_optimization_function([gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type])
