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
        -1.574184100758179,
        -1.763115912438158,
        1.876132292821092,
        -0.5090195101726027,
        1.4778460993867202,
        -0.9922529823544854,
    ]
    x = np.dot(features, weights[: len(features)])

    # Polynomial function
    result = (
        -0.2753064383992698 * x**3
        + -0.17152659722418062 * x**2
        + -0.33475629720338285 * x
        + -0.6531441807400002
    )

    return np.clip(result, -1e6, 1e6)


# Example usage:
# result = vr_optimization_function([gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type])
