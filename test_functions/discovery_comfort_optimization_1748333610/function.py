
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
    weights = [1.2716888790333067, -1.7073850491211942, -0.7916505393920463, -0.709683771857534, 1.2207661119277446, 1.936998803804086]
    x = np.dot(features, weights[:len(features)])
    
    # Polynomial function
    result = (0.05064980095505489 * x**3 + 0.5713302593166167 * x**2 + 
              -0.25427243433309177 * x + -0.16615243073529573)
    
    return np.clip(result, -1e6, 1e6)

# Example usage:
# result = vr_optimization_function([gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type])
