"""
Online reward normalization for multi-environment RL training stability.

Implements Welford's online algorithm for running mean/variance computation,
enabling z-score and min-max normalization of reward signals without needing
to store all historical values.

This is critical for multi-environment training where different environments
produce rewards on different scales (e.g., GSM8K gives {-1, 1} while
tool-use environments give continuous [0, 1] scores).

Usage:
    normalizer = RewardNormalizer(mode="zscore", clip=5.0)

    # During training loop
    scores = [0.5, -0.3, 0.8, 1.0]
    normalized = normalizer.normalize(scores)

    # Checkpointing
    state = normalizer.state_dict()
    normalizer.load_state_dict(state)
"""

import logging
import math
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class WelfordAccumulator:
    """
    Welford's online algorithm for computing running mean and variance.

    Numerically stable single-pass algorithm that avoids catastrophic
    cancellation. Maintains count, mean, and M2 (sum of squared deviations)
    to compute variance on demand.

    Reference: Welford, B. P. (1962). "Note on a method for calculating
    corrected sums of squares and products". Technometrics. 4(3): 419-420.
    """

    def __init__(self):
        self.count: int = 0
        self.mean: float = 0.0
        self._m2: float = 0.0
        self._min: float = float("inf")
        self._max: float = float("-inf")

    def update(self, value: float) -> None:
        """Update running statistics with a new value."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self._m2 += delta * delta2
        self._min = min(self._min, value)
        self._max = max(self._max, value)

    def update_batch(self, values: List[float]) -> None:
        """Update running statistics with a batch of values."""
        for v in values:
            self.update(v)

    @property
    def variance(self) -> float:
        """Population variance of all observed values."""
        if self.count < 2:
            return 0.0
        return self._m2 / self.count

    @property
    def std(self) -> float:
        """Population standard deviation of all observed values."""
        return math.sqrt(self.variance)

    @property
    def min_val(self) -> float:
        """Minimum observed value."""
        return self._min if self.count > 0 else 0.0

    @property
    def max_val(self) -> float:
        """Maximum observed value."""
        return self._max if self.count > 0 else 0.0

    def state_dict(self) -> Dict[str, Any]:
        """Serialize state for checkpointing."""
        return {
            "count": self.count,
            "mean": self.mean,
            "m2": self._m2,
            "min": self._min,
            "max": self._max,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore state from checkpoint."""
        self.count = state["count"]
        self.mean = state["mean"]
        self._m2 = state["m2"]
        self._min = state["min"]
        self._max = state["max"]


class RewardNormalizer:
    """
    Reward normalization for stable multi-environment RL training.

    Supports two normalization modes:
    - "zscore": Standardize to zero mean, unit variance using running stats
    - "minmax": Scale to [0, 1] range using observed min/max

    Both modes use Welford's online algorithm so no historical data storage
    is required. Optional reward clipping prevents extreme values from
    destabilizing training.

    Args:
        mode: Normalization mode. One of "zscore", "minmax", or "none".
        clip: Maximum absolute value after normalization. Set to 0 or None
              to disable clipping. Default: 5.0.
        warmup: Minimum number of samples before normalization activates.
                During warmup, raw scores are returned (optionally clipped).
                Default: 10.
        eps: Small constant for numerical stability in division. Default: 1e-8.
    """

    VALID_MODES = {"zscore", "minmax", "none"}

    def __init__(
        self,
        mode: str = "zscore",
        clip: Optional[float] = 5.0,
        warmup: int = 10,
        eps: float = 1e-8,
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid normalization mode '{mode}'. "
                f"Must be one of: {self.VALID_MODES}"
            )

        self.mode = mode
        self.clip = clip if clip and clip > 0 else None
        self.warmup = max(0, warmup)
        self.eps = eps
        self._accumulator = WelfordAccumulator()

    @property
    def count(self) -> int:
        """Number of samples observed."""
        return self._accumulator.count

    @property
    def mean(self) -> float:
        """Running mean of observed values."""
        return self._accumulator.mean

    @property
    def std(self) -> float:
        """Running standard deviation of observed values."""
        return self._accumulator.std

    @property
    def is_warmed_up(self) -> bool:
        """Whether enough samples have been observed for normalization."""
        return self._accumulator.count >= self.warmup

    def normalize(self, scores: List[float]) -> List[float]:
        """
        Normalize a batch of reward scores.

        Updates running statistics with the new scores, then applies
        normalization. During warmup, raw scores are returned (with
        optional clipping).

        Args:
            scores: Raw reward scores to normalize.

        Returns:
            Normalized (and optionally clipped) scores.
        """
        if not scores:
            return []

        if self.mode == "none":
            return list(scores)

        # Update running statistics
        self._accumulator.update_batch(scores)

        # During warmup, return raw scores (optionally clipped)
        if not self.is_warmed_up:
            logger.debug(
                "Reward normalizer warmup: %d/%d samples",
                self._accumulator.count,
                self.warmup,
            )
            return self._clip(list(scores))

        # Apply normalization
        if self.mode == "zscore":
            normalized = self._zscore(scores)
        elif self.mode == "minmax":
            normalized = self._minmax(scores)
        else:
            normalized = list(scores)

        return self._clip(normalized)

    def _zscore(self, scores: List[float]) -> List[float]:
        """Z-score normalize: (x - mean) / std."""
        mean = self._accumulator.mean
        std = self._accumulator.std
        if std < self.eps:
            # All values nearly identical -- return zeros
            return [0.0] * len(scores)
        return [(s - mean) / (std + self.eps) for s in scores]

    def _minmax(self, scores: List[float]) -> List[float]:
        """Min-max normalize to [0, 1] range."""
        min_val = self._accumulator.min_val
        max_val = self._accumulator.max_val
        range_val = max_val - min_val
        if range_val < self.eps:
            return [0.5] * len(scores)
        return [(s - min_val) / (range_val + self.eps) for s in scores]

    def _clip(self, scores: List[float]) -> List[float]:
        """Clip scores to [-clip, clip] range."""
        if self.clip is None:
            return scores
        return [max(-self.clip, min(self.clip, s)) for s in scores]

    def metrics_dict(self) -> Dict[str, float]:
        """
        Return current normalization statistics for WandB logging.

        Returns:
            Dictionary with keys suitable for wandb.log().
        """
        metrics = {
            "reward_norm/count": float(self._accumulator.count),
            "reward_norm/mean": self._accumulator.mean,
            "reward_norm/std": self._accumulator.std,
            "reward_norm/min": self._accumulator.min_val,
            "reward_norm/max": self._accumulator.max_val,
        }
        return metrics

    def state_dict(self) -> Dict[str, Any]:
        """Serialize full state for checkpointing."""
        return {
            "mode": self.mode,
            "clip": self.clip,
            "warmup": self.warmup,
            "eps": self.eps,
            "accumulator": self._accumulator.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore state from checkpoint."""
        self.mode = state["mode"]
        self.clip = state["clip"]
        self.warmup = state["warmup"]
        self.eps = state["eps"]
        self._accumulator.load_state_dict(state["accumulator"])
