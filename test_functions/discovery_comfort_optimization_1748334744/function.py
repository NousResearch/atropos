
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
    weights = [-0.39990804013178716, -0.39950604266626444, 1.0444668064689826, -0.19027565334905017, -0.6392601281820158, 0.425559389789254]
    x = np.dot(features, weights[:len(features)])
    
    # Linear function
    result = x + 8.090493841963017
    
    return np.clip(result, -1e6, 1e6)

# Example usage:
# result = vr_optimization_function([gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type])
