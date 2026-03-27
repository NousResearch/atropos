"""
Numerical verification utilities for RL reward and advantage computation.

Provides test harness functions for validating numerical correctness across
inference and training, including:
- Reward function determinism verification
- Advantage computation stability checks (NaN, Inf, magnitude)
- Floating-point precision comparison (FP32 vs FP16 vs BF16)
- Score distribution analysis (collapse/explosion detection)

Designed to be used in environment test suites and CI pipelines.

Usage:
    from atroposlib.utils.numerical_verification import (
        verify_reward_determinism,
        verify_advantage_stability,
        compare_fp_precision,
        verify_score_distribution,
    )

    # In a test
    assert verify_reward_determinism(my_reward_fn, test_inputs, n_runs=5)
    report = verify_advantage_stability(advantages)
    assert report.is_stable
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StabilityReport:
    """Report from numerical stability checks."""

    is_stable: bool
    has_nan: bool = False
    has_inf: bool = False
    max_magnitude: float = 0.0
    mean_magnitude: float = 0.0
    std: float = 0.0
    issues: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary of the report."""
        if self.is_stable:
            return (
                f"STABLE: max_mag={self.max_magnitude:.6f}, "
                f"mean_mag={self.mean_magnitude:.6f}, std={self.std:.6f}"
            )
        return f"UNSTABLE: {'; '.join(self.issues)}"


@dataclass
class PrecisionReport:
    """Report from floating-point precision comparison."""

    max_divergence: float
    mean_divergence: float
    divergent_fraction: float
    reference_precision: str
    compared_precisions: List[str] = field(default_factory=list)
    per_precision: Dict[str, float] = field(default_factory=dict)

    @property
    def is_acceptable(self) -> bool:
        """Whether divergence is within acceptable bounds."""
        return self.max_divergence < 0.01 and self.divergent_fraction < 0.05

    def summary(self) -> str:
        """Human-readable summary."""
        status = "PASS" if self.is_acceptable else "FAIL"
        return (
            f"{status}: max_div={self.max_divergence:.6f}, "
            f"mean_div={self.mean_divergence:.6f}, "
            f"divergent_frac={self.divergent_fraction:.4f}"
        )


@dataclass
class DistributionReport:
    """Report from score distribution analysis."""

    is_healthy: bool
    mean: float
    std: float
    min_val: float
    max_val: float
    range_val: float
    is_collapsed: bool = False
    is_exploded: bool = False
    is_biased: bool = False
    issues: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary."""
        if self.is_healthy:
            return (
                f"HEALTHY: mean={self.mean:.4f}, std={self.std:.4f}, "
                f"range=[{self.min_val:.4f}, {self.max_val:.4f}]"
            )
        return f"UNHEALTHY: {'; '.join(self.issues)}"


def verify_reward_determinism(
    reward_fn: Callable,
    inputs: List[Any],
    n_runs: int = 5,
    atol: float = 1e-10,
    **kwargs,
) -> bool:
    """
    Verify that a reward function produces identical outputs across N runs.

    Non-deterministic reward functions can cause training instability and
    make debugging extremely difficult. This check ensures reproducibility.

    Args:
        reward_fn: Callable that takes (inputs, **kwargs) and returns List[float].
        inputs: Input data to pass to the reward function.
        n_runs: Number of times to run the function. Default: 5.
        atol: Absolute tolerance for floating-point comparison. Default: 1e-10.
        **kwargs: Additional keyword arguments passed to reward_fn.

    Returns:
        True if all runs produce identical results (within tolerance).
    """
    if n_runs < 2:
        logger.warning("verify_reward_determinism: n_runs < 2 is trivially true")
        return True

    results = []
    for i in range(n_runs):
        try:
            scores = reward_fn(inputs, **kwargs)
            results.append(scores)
        except Exception as e:
            logger.error("Run %d failed with error: %s", i, e)
            return False

    # Compare all runs to the first
    reference = results[0]
    for i, run_scores in enumerate(results[1:], start=1):
        if len(run_scores) != len(reference):
            logger.error(
                "Run %d returned %d scores, expected %d",
                i,
                len(run_scores),
                len(reference),
            )
            return False

        for j, (ref, score) in enumerate(zip(reference, run_scores)):
            if abs(ref - score) > atol:
                logger.error(
                    "Non-determinism at index %d: run 0 = %.10f, run %d = %.10f "
                    "(diff = %.2e)",
                    j,
                    ref,
                    i,
                    score,
                    abs(ref - score),
                )
                return False

    return True


def verify_advantage_stability(
    advantages: Union[List[float], np.ndarray],
    max_magnitude: float = 100.0,
    warn_threshold: float = 10.0,
) -> StabilityReport:
    """
    Check advantage values for numerical stability issues.

    Detects NaN, Inf, and excessive magnitude which indicate problems
    in the reward → advantage pipeline (e.g., division by zero in
    normalization, exploding rewards, or dtype overflow).

    Args:
        advantages: Advantage values to check.
        max_magnitude: Maximum acceptable absolute value. Default: 100.0.
        warn_threshold: Threshold for warning-level magnitude. Default: 10.0.

    Returns:
        StabilityReport with detailed diagnostics.
    """
    arr = np.asarray(advantages, dtype=np.float64)
    issues = []

    has_nan = bool(np.any(np.isnan(arr)))
    has_inf = bool(np.any(np.isinf(arr)))

    if has_nan:
        nan_count = int(np.sum(np.isnan(arr)))
        issues.append(f"Contains {nan_count} NaN values ({nan_count}/{len(arr)})")

    if has_inf:
        inf_count = int(np.sum(np.isinf(arr)))
        issues.append(f"Contains {inf_count} Inf values ({inf_count}/{len(arr)})")

    # Filter out NaN/Inf for magnitude analysis
    finite = arr[np.isfinite(arr)]

    if len(finite) == 0:
        return StabilityReport(
            is_stable=False,
            has_nan=has_nan,
            has_inf=has_inf,
            max_magnitude=float("inf"),
            mean_magnitude=float("inf"),
            std=float("inf"),
            issues=issues or ["All values are NaN or Inf"],
        )

    abs_values = np.abs(finite)
    max_mag = float(np.max(abs_values))
    mean_mag = float(np.mean(abs_values))
    std_val = float(np.std(finite))

    if max_mag > max_magnitude:
        issues.append(
            f"Max magnitude {max_mag:.4f} exceeds limit {max_magnitude:.4f}"
        )

    if max_mag > warn_threshold and max_mag <= max_magnitude:
        logger.warning(
            "Advantage magnitudes are large (max=%.4f). "
            "Consider checking reward normalization.",
            max_mag,
        )

    is_stable = not has_nan and not has_inf and max_mag <= max_magnitude

    return StabilityReport(
        is_stable=is_stable,
        has_nan=has_nan,
        has_inf=has_inf,
        max_magnitude=max_mag,
        mean_magnitude=mean_mag,
        std=std_val,
        issues=issues,
    )


def compare_fp_precision(
    fn: Callable,
    inputs: List[Any],
    precisions: Optional[List[str]] = None,
    reference: str = "float64",
    atol: float = 1e-3,
    **kwargs,
) -> PrecisionReport:
    """
    Compare function outputs across different floating-point precisions.

    Runs the function with inputs cast to different precision levels and
    measures the divergence from a high-precision reference. This detects
    precision-sensitive computations that may break under mixed-precision
    training.

    Args:
        fn: Function that takes a list of float values and returns a list of floats.
        inputs: Input values to test.
        precisions: List of numpy dtype strings to test.
                    Default: ["float32", "float16", "bfloat16"].
        reference: Reference precision for comparison. Default: "float64".
        atol: Absolute tolerance for "divergent" classification. Default: 1e-3.
        **kwargs: Additional arguments passed to fn.

    Returns:
        PrecisionReport with divergence statistics.
    """
    if precisions is None:
        precisions = ["float32", "float16"]
        # Only add bfloat16 if numpy supports it
        try:
            np.dtype("bfloat16")
            precisions.append("bfloat16")
        except TypeError:
            pass

    # Compute reference output at high precision
    ref_inputs = [float(x) for x in inputs]
    try:
        ref_outputs = fn(ref_inputs, **kwargs)
        ref_arr = np.array(ref_outputs, dtype=np.float64)
    except Exception as e:
        return PrecisionReport(
            max_divergence=float("inf"),
            mean_divergence=float("inf"),
            divergent_fraction=1.0,
            reference_precision=reference,
            compared_precisions=precisions,
            per_precision={p: float("inf") for p in precisions},
        )

    per_precision = {}
    all_divergences = []

    for prec in precisions:
        try:
            # Cast inputs to target precision and back to float
            prec_arr = np.array(inputs, dtype=prec)
            prec_inputs = prec_arr.astype(np.float64).tolist()

            prec_outputs = fn(prec_inputs, **kwargs)
            prec_arr_out = np.array(prec_outputs, dtype=np.float64)

            # Compute per-element divergence
            divergences = np.abs(ref_arr - prec_arr_out)
            max_div = float(np.max(divergences))
            per_precision[prec] = max_div
            all_divergences.extend(divergences.tolist())

        except Exception as e:
            logger.warning("Precision %s failed: %s", prec, e)
            per_precision[prec] = float("inf")
            all_divergences.extend([float("inf")] * len(ref_arr))

    if not all_divergences:
        return PrecisionReport(
            max_divergence=0.0,
            mean_divergence=0.0,
            divergent_fraction=0.0,
            reference_precision=reference,
            compared_precisions=precisions,
            per_precision=per_precision,
        )

    div_arr = np.array(all_divergences)
    finite_div = div_arr[np.isfinite(div_arr)]

    return PrecisionReport(
        max_divergence=float(np.max(div_arr)) if len(div_arr) > 0 else 0.0,
        mean_divergence=float(np.mean(finite_div)) if len(finite_div) > 0 else 0.0,
        divergent_fraction=float(np.mean(div_arr > atol)),
        reference_precision=reference,
        compared_precisions=precisions,
        per_precision=per_precision,
    )


def verify_score_distribution(
    scores: Union[List[float], np.ndarray],
    expected_range: Tuple[float, float] = (-1.0, 1.0),
    collapse_threshold: float = 1e-6,
    explosion_threshold: float = 100.0,
    bias_threshold: float = 0.9,
) -> DistributionReport:
    """
    Verify that a batch of reward scores has a healthy distribution.

    Detects common failure modes:
    - Reward collapse: all scores converge to the same value (std ~ 0)
    - Reward explosion: scores exceed expected range by large factors
    - Reward bias: nearly all scores at one extreme of the range

    Args:
        scores: Reward scores to analyze.
        expected_range: Expected (min, max) range for scores. Default: (-1, 1).
        collapse_threshold: Std below this = collapsed. Default: 1e-6.
        explosion_threshold: Max absolute value above this = exploded. Default: 100.
        bias_threshold: Fraction of scores at one extreme = biased. Default: 0.9.

    Returns:
        DistributionReport with health diagnostics.
    """
    arr = np.asarray(scores, dtype=np.float64)

    if len(arr) == 0:
        return DistributionReport(
            is_healthy=True,
            mean=0.0,
            std=0.0,
            min_val=0.0,
            max_val=0.0,
            range_val=0.0,
        )

    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    range_val = max_val - min_val

    issues = []
    is_collapsed = False
    is_exploded = False
    is_biased = False

    # Check for collapse
    if std_val < collapse_threshold and len(arr) > 1:
        is_collapsed = True
        issues.append(
            f"Reward collapse detected: std={std_val:.2e} < {collapse_threshold:.2e}"
        )

    # Check for explosion
    if abs(max_val) > explosion_threshold or abs(min_val) > explosion_threshold:
        is_exploded = True
        issues.append(
            f"Reward explosion: range [{min_val:.4f}, {max_val:.4f}] "
            f"exceeds threshold {explosion_threshold}"
        )

    # Check for bias (nearly all at one extreme)
    exp_min, exp_max = expected_range
    exp_mid = (exp_min + exp_max) / 2
    if len(arr) > 1:
        at_min = float(np.mean(arr <= exp_min + (exp_max - exp_min) * 0.1))
        at_max = float(np.mean(arr >= exp_max - (exp_max - exp_min) * 0.1))
        if at_min > bias_threshold:
            is_biased = True
            issues.append(
                f"Reward bias: {at_min:.1%} of scores at/near minimum ({exp_min})"
            )
        elif at_max > bias_threshold:
            is_biased = True
            issues.append(
                f"Reward bias: {at_max:.1%} of scores at/near maximum ({exp_max})"
            )

    is_healthy = not is_collapsed and not is_exploded and not is_biased

    return DistributionReport(
        is_healthy=is_healthy,
        mean=mean_val,
        std=std_val,
        min_val=min_val,
        max_val=max_val,
        range_val=range_val,
        is_collapsed=is_collapsed,
        is_exploded=is_exploded,
        is_biased=is_biased,
        issues=issues,
    )
