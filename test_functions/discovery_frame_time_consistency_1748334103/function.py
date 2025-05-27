
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
    weights = [-1.8114649811324908, -1.3086795348859765, -0.8650676240529309, 1.763793882772906, 1.3707410472517054, -1.622888549987879]
    x = np.dot(features, weights[:len(features)])
    
    # Polynomial function
    result = (-0.8502331021135503 * x**3 + -0.9833742946674568 * x**2 + 
              0.10602537099553033 * x + -0.9811133893241266)
    
    return np.clip(result, -1e6, 1e6)

# Example usage:
# result = vr_optimization_function([gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type])
