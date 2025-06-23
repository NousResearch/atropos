# Auto-generated VR optimization function
# Type: polynomial
# Fitness: -1000000000000.0000

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
        1.0710808781499481,
        -1.8448277658423797,
        1.9080829410602669,
        0.45169059539560985,
        1.1695381170163461,
        -0.6405332132459884,
    ]
    x = np.dot(features, weights[: len(features)])

    # Polynomial function
    result = (
        -0.6245155955259436 * x**3
        + -0.39898316653022414 * x**2
        + -0.0598710608014108 * x
        + -0.9163816979346255
    )

    return np.clip(result, -1e6, 1e6)


# Example usage:
# result = vr_optimization_function([gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type])
