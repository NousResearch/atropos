
# Auto-generated VR optimization function
# Type: exponential
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
    weights = [0.872504312794262, -0.08148408219278203, 0.3889184436787363, -0.2515395891407839, -0.37522044326428206, -0.8206766666950813]
    x = np.dot(features, weights[:len(features)])
    
    # Exponential function
    exp_arg = np.clip(1.2392278688033156 * x, -50, 50)
    result = np.exp(exp_arg) + 3.4077324626898413
    
    return np.clip(result, -1e6, 1e6)

# Example usage:
# result = vr_optimization_function([gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type])
