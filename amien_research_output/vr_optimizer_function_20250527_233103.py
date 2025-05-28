import numpy as np

def vr_comfort_optimizer(features):
    """
    Optimizes VR comfort score based on performance features.

    Args:
        features: A list of 6 normalized features (0-1): 
                  [gpu_utilization, vram_usage, cpu_utilization, scene_complexity, duration, app_type].

    Returns:
        An optimized comfort score (0.0 to 1.0). Returns None if input is invalid.
    """
    try:
        features = np.array(features)
        if features.shape != (6,):
            return None
        if not np.all((features >= 0) & (features <= 1)):
            return None

        #  Weighting based on observed importance (adjust weights as needed based on data analysis)
        weights = np.array([0.3, 0.2, 0.15, 0.2, -0.05, 0.1]) # Negative weight for duration suggests longer is worse

        # Applying a non-linear transformation to capture potential interactions. 
        transformed_features = np.tanh(features)

        # Weighted sum to obtain a comfort score
        weighted_sum = np.dot(transformed_features, weights)

        # Sigmoid activation function to ensure the score is between 0 and 1
        comfort_score = 1 / (1 + np.exp(-weighted_sum))

        return comfort_score

    except (ValueError, TypeError):
        return None