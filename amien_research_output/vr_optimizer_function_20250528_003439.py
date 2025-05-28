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

        #  Weighting factors based on presumed impact on comfort.  These could be tuned.
        weights = np.array([0.3, 0.2, 0.15, 0.2, 0.05, 0.1])

        #  Inverse relationship for utilization and complexity.  Higher utilization and complexity reduce comfort.
        utilization_impact = 1 - (gpu_util + vram + cpu_util + scene_comp)/4
        
        #Application type impact (example: 0 for less demanding apps, 1 for more demanding)
        app_impact = 1- app_type
        
        # Duration impact (longer duration slightly reduces comfort, but not overly significant)
        duration_impact = 1 - duration * 0.05


        # Combined weighted score.  More complex interactions could be added here.
        comfort_score = np.dot(np.array([utilization_impact, vram, cpu_util, scene_comp, duration_impact, app_impact]), weights)

        # Clamp score to [0, 1] range.
        return np.clip(comfort_score, 0.0, 1.0)

    except (ValueError, TypeError):
        return -1