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
    arr = np.asarray(data, dtype=float)
    arr = arr[np.isfinite(arr)]  # drop NaN/Inf values

    if arr.size == 0:
        # Avoid crashes and keep metric keys stable
        metrics_dict[f"{name}_mean"] = float("nan")
        metrics_dict[f"{name}_std"] = float("nan")
        metrics_dict[f"{name}_max"] = float("nan")
        metrics_dict[f"{name}_min"] = float("nan")
        return metrics_dict

    metrics_dict[f"{name}_mean"] = float(np.mean(arr))
    metrics_dict[f"{name}_std"] = float(np.std(arr))
    metrics_dict[f"{name}_max"] = float(np.max(arr))
    metrics_dict[f"{name}_min"] = float(np.min(arr))
    return metrics_dict
