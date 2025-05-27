
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
    weights = [1.2364612694963877, 1.7662878033410694, -1.2396305530471388, -0.30302795917101966, 0.3995972322916579, -0.017669142102945568]
    x = np.dot(features, weights[:len(features)])
    
    # Linear function
    result = x + 3.46881794794424
    
    return np.clip(result, -1e6, 1e6)

# Example usage:
# result = vr_optimization_function([gpu_util, vram_usage, cpu_util, scene_complexity, duration, gpu_type])
