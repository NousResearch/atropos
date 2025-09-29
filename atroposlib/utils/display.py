"""Display utilities for formatting evaluation results and tables."""

from typing import Dict


def display_metrics_table(
    task_name: str, metrics: Dict, start_time: float, end_time: float
) -> None:
    """
    Display evaluation results in a formatted markdown table.

    Args:
        task_name: Name of the evaluation task
        metrics: Dictionary of metric names to values
        start_time: Start time of evaluation (unix timestamp)
        end_time: End time of evaluation (unix timestamp)
    """
    print(f"\nEvaluation Results: {task_name}")

    # Prepare data for column width calculation
    clean_metric_names = [
        metric_name.replace("eval/", "").replace("_", " ")
        for metric_name in metrics.keys()
    ]
    formatted_values = [f"{value:.4f}" for value in metrics.values()]

    # Calculate dynamic column widths
    col_groups = max(len(task_name), len("Groups"))
    col_version = max(len("1"), len("Version"))
    col_filter = max(len("none"), len("Filter"))
    col_nshot = max(len(""), len("n-shot"))
    col_metric = max(max(len(name) for name in clean_metric_names), len("Metric"))
    col_dir = 3  # Fixed for directional indicators
    col_value = max(max(len(val) for val in formatted_values), len("Value"))
    col_pm = 3  # Fixed for "±" symbol
    col_stderr = max(len("0.0000"), len("Stderr"))

    # Header
    print(
        f"|{'Groups':<{col_groups}}|{'Version':<{col_version}}|{'Filter':<{col_filter}}|{'n-shot':<{col_nshot}}|{'Metric':<{col_metric}}|{'':^{col_dir}}|{'Value':>{col_value}}|{'':^{col_pm}}|{'Stderr':>{col_stderr}}|"  # noqa: E501
    )

    # Separator
    print(
        f"|{'-' * col_groups}|{'-' * col_version}|{'-' * col_filter}|{'-' * col_nshot}|{'-' * col_metric}|{'-' * col_dir}|{'-' * (col_value-1)}:|{'-' * col_pm}|{'-' * (col_stderr-1)}:|"  # noqa: E501
    )

    # Data rows
    for metric_name, metric_value in metrics.items():
        clean_metric_name = metric_name.replace("eval/", "").replace("_", " ")
        direction = "↑" if "correct" in metric_name or "acc" in metric_name else ""

        print(
            f"|{task_name:<{col_groups}}|{1:<{col_version}}|{'none':<{col_filter}}|{'':<{col_nshot}}|{clean_metric_name:<{col_metric}}|{direction:^{col_dir}}|{metric_value:>{col_value}.4f}|{'±':^{col_pm}}|{'0.0000':>{col_stderr}}|"  # noqa: E501
        )

    print(f"Evaluation completed in {end_time - start_time:.2f} seconds")
