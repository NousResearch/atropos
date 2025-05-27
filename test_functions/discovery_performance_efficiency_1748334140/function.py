
# Auto-generated VR optimization function
# Type: polynomial
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
    weights = [-1.6347276729162536, 1.4200187880342678, 1.9375364312162322, 1.1950634564599287, -0.5175712559086416, -1.687801191027622]
    x = np.dot(features, weights[:len(features)])
    
    # Polynomial function
    result = (0.8467876979546551 * x**3 + -0.8562089546162246 * x**2 + 
              0.706035086389377 * x + -0.23631529485870173)
    
    return np.clip(result, -1e6, 1e6)

# Example usage:
# result = vr_optimization_function([gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type])
