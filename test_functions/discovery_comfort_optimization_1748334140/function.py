# Auto-generated VR optimization function
# Type: linear
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
        1.3604547077741675,
        1.8424516528438568,
        0.6104377712623656,
        -0.12289056651387975,
        -0.5328208745650165,
        -0.9952628074642718,
    ]
    x = np.dot(features, weights[: len(features)])

    # Linear function
    result = x + -2.681506391163639

    return np.clip(result, -1e6, 1e6)


# Example usage:
# result = vr_optimization_function([gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type])
