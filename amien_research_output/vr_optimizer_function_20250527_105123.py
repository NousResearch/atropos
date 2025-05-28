import numpy as np

def vr_comfort_optimizer(features):
    """
    Optimizes VR comfort score based on performance features.

    Args:
        features: A list of 6 normalized features (0-1): 
                  [gpu_utilization, vram_usage, cpu_utilization, 
                   scene_complexity, duration, app_type].

    Returns:
        An optimized comfort score (0.0 to 1.0). Returns -1 if input is invalid.
    """
    if not isinstance(features, list) or len(features) != 6:
        return -1
    for feature in features:
        if not 0 <= feature <= 1:
            return -1

    gpu_util, vram, cpu_util, scene_comp, duration, app_type = features

    #Weighting based on observed importance in VR comfort
    comfort_score = 0.35 * (1 - gpu_util) + 0.25 * (1 - vram) + 0.15 * (1 - cpu_util) + \
                    0.1 * (1 - scene_comp) + 0.1 * (1 - duration) + 0.05 * (1 - app_type)

    # Non-linear adjustment for interaction effects
    comfort_score *= (1 - 0.1 * (gpu_util * vram * cpu_util))

    #Ensure score remains within 0-1 range
    comfort_score = np.clip(comfort_score, 0.0, 1.0)

    return comfort_score