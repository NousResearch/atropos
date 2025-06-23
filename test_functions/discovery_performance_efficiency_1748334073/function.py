# Auto-generated VR optimization function
# Type: linear
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
        0.8790465522632731,
        -1.3006541396594828,
        -1.2083172763579877,
        -0.38259623577751833,
        1.0135228857542846,
        -0.9665032368745976,
    ]
    x = np.dot(features, weights[: len(features)])

    # Linear function
    result = x + 5.596096452060522

    return np.clip(result, -1e6, 1e6)


# Example usage:
# result = vr_optimization_function([gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type])
