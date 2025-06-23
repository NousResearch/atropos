# Auto-generated VR optimization function
# Type: polynomial
# Fitness: 1.0000

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
        -1.393250392523104,
        -1.747013980171813,
        0.29367536387141424,
        -1.2592429370784872,
        -1.0605106506622612,
        1.416422009223389,
    ]
    x = np.dot(features, weights[: len(features)])

    # Polynomial function
    result = (
        -0.40058381914502794 * x**3
        + -0.9509513716365221 * x**2
        + -0.7109474544938195 * x
        + -0.9952346741502942
    )

    return np.clip(result, -1e6, 1e6)


# Example usage:
# result = vr_optimization_function([gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type])
