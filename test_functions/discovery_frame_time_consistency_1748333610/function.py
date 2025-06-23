# Auto-generated VR optimization function
# Type: exponential
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
        0.33362111469905087,
        -0.35622882413439805,
        0.36682118830739885,
        -0.486961366673617,
        0.3693682895232908,
        -0.727378481764628,
    ]
    x = np.dot(features, weights[: len(features)])

    # Exponential function
    exp_arg = np.clip(1.647440711203808 * x, -50, 50)
    result = np.exp(exp_arg) + 0.81567028378001

    return np.clip(result, -1e6, 1e6)


# Example usage:
# result = vr_optimization_function([gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type])
