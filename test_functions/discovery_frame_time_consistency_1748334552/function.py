
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
    weights = [-1.917072430836797, 0.5812591962324349, -1.1309543837219072, 1.2238122713480792, 0.1920436493062465, -0.9705819502551347]
    x = np.dot(features, weights[:len(features)])
    
    # Polynomial function
    result = (-0.7906629893807404 * x**3 + 0.3755862321766821 * x**2 + 
              0.40269241439798686 * x + -0.9952283492454468)
    
    return np.clip(result, -1e6, 1e6)

# Example usage:
# result = vr_optimization_function([gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type])
