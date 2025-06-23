import numpy as np


def vr_comfort_optimizer(features):
    """
    Optimizes VR comfort score based on performance features.

    Args:
        features: A list of 6 normalized features (0-1):
                  [gpu_utilization, vram_usage, cpu_utilization, scene_complexity, duration, app_type].

    Returns:
        An optimized comfort score (0.0 to 1.0). Returns an error message if input is invalid.
    """
    try:
        gpu_util, vram, cpu_util, scene_comp, duration, app_type = features
        if not all(0 <= x <= 1 for x in features):
            raise ValueError("Features must be normalized between 0 and 1.")

        # Weighting factors based on assumed importance for comfort.  These could be tuned further.
        weights = np.array([0.3, 0.2, 0.15, 0.2, 0.05, 0.1])

        # Weighted sum reflecting the impact of each feature.  Higher values indicate higher load.
        weighted_sum = np.sum(np.array(features) * weights)

        # Nonlinear transformation to map weighted sum to comfort score.  Experimentation is key here.
        comfort_score = 1 / (1 + np.exp(-(5 * (weighted_sum - 0.5))))

        return max(
            0.0, min(1.0, comfort_score)
        )  # Ensure score remains within 0-1 range.

    except ValueError as e:
        return f"Error: {e}"
    except TypeError:
        return "Error: Input must be a list of 6 numbers."
    except Exception as e:
        return f"An unexpected error occurred: {e}"
