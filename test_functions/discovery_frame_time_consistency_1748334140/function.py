
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
    weights = [0.3900248918585145, -0.5796707947235356, 0.754544614182979, -0.7064317495703227, 0.007866426664854265, -0.7949336218180736]
    x = np.dot(features, weights[:len(features)])
    
    # Exponential function
    exp_arg = np.clip(1.7996454781413975 * x, -50, 50)
    result = np.exp(exp_arg) + -4.52296505206495
    
    return np.clip(result, -1e6, 1e6)

# Example usage:
# result = vr_optimization_function([gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type])
