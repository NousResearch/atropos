
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
    weights = [0.8853990025025649, 0.8356038890455189, -0.9642515498007673, -0.07744606350500605, -0.8524545510791419, -0.42670378074486504]
    x = np.dot(features, weights[:len(features)])
    
    # Exponential function
    exp_arg = np.clip(1.2844119380859922 * x, -50, 50)
    result = np.exp(exp_arg) + 1.0399254586264748
    
    return np.clip(result, -1e6, 1e6)

# Example usage:
# result = vr_optimization_function([gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type])
