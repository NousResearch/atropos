import numpy as np

def vr_comfort_optimizer(features):
    """
    Optimizes VR comfort score based on performance features.

    Args:
        features: A list of 6 normalized features (0-1): 
                  [gpu_utilization, vram_usage, cpu_utilization, scene_complexity, duration, app_type].

    Returns:
        Optimized comfort score (0.0 to 1.0). Returns -1 if input is invalid.
    """
    try:
        gpu_util, vram, cpu_util, scene_comp, duration, app_type = features
        if not all(0 <= x <= 1 for x in features):
            return -1

        #Simulate a discovered pattern; Replace with a more sophisticated model if needed.
        comfort_score = 1 - (0.3 * gpu_util + 0.2 * vram + 0.15 * cpu_util + 0.2 * scene_comp + 0.05 * duration)  + 0.1 * app_type

        return max(0.0, min(1.0, comfort_score)) #Ensure score within 0-1 range

    except (ValueError, TypeError):
        return -1