import numpy as np


def get_std_min_max_avg(name: str, data: list, metrics_dict: dict) -> dict:
    """
    Calculate the standard deviation, minimum, maximum, and average of a list of numbers.
    Adds it to the metrics dict for logging.

    Args:
        name: The base name for the metrics keys.
        data: A list of numbers to compute statistics from.
        metrics_dict: Dictionary to add the computed metrics to.

    Returns:
        The updated metrics dictionary with added statistics (mean, std, max, min).
    """
    metrics_dict[f"{name}_mean"] = np.mean(data)
    metrics_dict[f"{name}_std"] = np.std(data)
    metrics_dict[f"{name}_max"] = np.max(data)
    metrics_dict[f"{name}_min"] = np.min(data)
    return metrics_dict
