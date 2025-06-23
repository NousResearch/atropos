import numpy as np


def vr_comfort_optimizer(features):
    """
    Optimizes VR comfort score based on performance features.

    Args:
        features: A list of 6 normalized features (0-1):
                  [gpu_utilization, vram_usage, cpu_utilization, scene_complexity, duration, app_type].

    Returns:
        Optimized comfort score (0.0 to 1.0). Returns None if input is invalid.
    """
    try:
        gpu_util, vram, cpu_util, scene_comp, duration, app_type = features
        if not all(0 <= x <= 1 for x in features):
            return None

        # Weighting based on observed impact in VR comfort
        weights = np.array([0.3, 0.2, 0.15, 0.2, 0.1, 0.05])
        feature_array = np.array(features)

        # Nonlinear transformation to capture diminishing returns/thresholds
        transformed_features = np.exp(-((1 - feature_array) ** 2)) * feature_array

        weighted_sum = np.sum(transformed_features * weights)

        # Comfort score adjustment based on app type (example: 0 = game, 1 = productivity)
        app_type_adjustment = 0.05 if app_type > 0.5 else -0.05
        final_score = min(max(weighted_sum + app_type_adjustment, 0.0), 1.0)

        return final_score

    except (ValueError, TypeError):
        return None
