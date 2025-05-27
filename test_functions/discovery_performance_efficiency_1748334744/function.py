
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
    weights = [1.0825393984413925, 1.676140911337467, 1.4883577783155588, 1.3838010085004604, 0.41890578079777985, 1.1157493851577867]
    x = np.dot(features, weights[:len(features)])
    
    # Polynomial function
    result = (-0.2772220168568287 * x**3 + 0.781927603830201 * x**2 + 
              0.5555335800780907 * x + 0.7745885122001905)
    
    return np.clip(result, -1e6, 1e6)

# Example usage:
# result = vr_optimization_function([gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type])
