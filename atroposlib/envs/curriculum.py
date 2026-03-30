from __future__ import annotations
"""
Curriculum learning scheduler for sample-efficient RL training.

Implements automatic difficulty-based sampling for environments, tracking
per-item difficulty from reward signals and adjusting sampling probabilities
to focus training on appropriately challenging examples.

Strategies:
- uniform: No curriculum (baseline, default)
- easy_first: Oversample easy items early, anneal to uniform
- competence_based: Sample items at the competence frontier (reward ~ 0.5),
  following Platanios et al. 2019 (https://arxiv.org/abs/1904.03746)

Usage:
    scheduler = CurriculumScheduler(
        strategy="competence_based",
        n_bins=5,
        temperature=1.0,
    )

    # After scoring an item
    scheduler.update("item_key_123", reward_score=0.7)

    # When selecting next item
    target_bin = scheduler.sample_bin(current_step=50, total_steps=1000)
"""

import logging
import math
import random
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CurriculumStrategy(str, Enum):
    """Available curriculum learning strategies."""

    UNIFORM = "uniform"
    EASY_FIRST = "easy_first"
    COMPETENCE_BASED = "competence_based"


class CurriculumScheduler:
    """
    Curriculum learning scheduler that tracks item difficulty and provides
    difficulty-aware sampling.

    Maintains an exponential moving average (EMA) of reward scores per item
    to estimate difficulty. Items are binned by difficulty quantile, and the
    sampling strategy determines which bins are preferred at each stage of
    training.

    Args:
        strategy: Sampling strategy. One of "uniform", "easy_first",
                  "competence_based".
        n_bins: Number of difficulty bins. Default: 5.
        temperature: Controls sampling sharpness. Higher = more uniform,
                     lower = more concentrated on target bin. Default: 1.0.
        ema_alpha: EMA smoothing factor for difficulty scores. Higher values
                   give more weight to recent rewards. Default: 0.3.
        competence_threshold: For competence_based strategy, the target
                              reward level considered "at frontier". Default: 0.5.
    """

    def __init__(
        self,
        strategy: str = "uniform",
        n_bins: int = 5,
        temperature: float = 1.0,
        ema_alpha: float = 0.3,
        competence_threshold: float = 0.5,
    ):
        # Validate strategy
        try:
            self._strategy = CurriculumStrategy(strategy)
        except ValueError:
            valid = [s.value for s in CurriculumStrategy]
            raise ValueError(
                f"Invalid curriculum strategy '{strategy}'. Must be one of: {valid}"
            )

        if n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {n_bins}")

        self.n_bins = n_bins
        self.temperature = max(0.01, temperature)
        self.ema_alpha = max(0.0, min(1.0, ema_alpha))
        self.competence_threshold = competence_threshold

        # Per-item difficulty tracking: key -> (ema_score, count)
        self._item_scores: Dict[str, Tuple[float, int]] = {}

        # Bin boundaries (recomputed periodically)
        self._bin_boundaries: List[float] = []
        self._last_rebin_count: int = 0
        self._rebin_interval: int = 50  # Recompute bins every N updates

    @property
    def strategy(self) -> str:
        """Current strategy name."""
        return self._strategy.value

    @property
    def n_items_tracked(self) -> int:
        """Number of unique items being tracked."""
        return len(self._item_scores)

    def update(self, item_key: str, score: float) -> None:
        """
        Update difficulty estimate for an item based on its reward score.

        Uses exponential moving average so recent performance has more
        influence than historical.

        Args:
            item_key: Unique identifier for the item (e.g., dataset index).
            score: Reward score achieved on this item. Higher = easier.
        """
        if item_key in self._item_scores:
            old_ema, count = self._item_scores[item_key]
            new_ema = self.ema_alpha * score + (1 - self.ema_alpha) * old_ema
            self._item_scores[item_key] = (new_ema, count + 1)
        else:
            self._item_scores[item_key] = (score, 1)

        # Periodically recompute bin boundaries
        total_updates = sum(c for _, c in self._item_scores.values())
        if total_updates - self._last_rebin_count >= self._rebin_interval:
            self._recompute_bins()
            self._last_rebin_count = total_updates

    def update_batch(self, item_key: str, scores: List[float]) -> None:
        """
        Update difficulty estimate with multiple scores (e.g., from group_size).

        Args:
            item_key: Unique identifier for the item.
            scores: List of reward scores from the group rollout.
        """
        if not scores:
            return
        avg_score = sum(scores) / len(scores)
        self.update(item_key, avg_score)

    def get_item_difficulty(self, item_key: str) -> Optional[float]:
        """
        Get the current difficulty estimate for an item.

        Returns:
            EMA reward score (higher = easier), or None if item not tracked.
        """
        if item_key not in self._item_scores:
            return None
        return self._item_scores[item_key][0]

    def get_item_bin(self, item_key: str) -> int:
        """
        Get the difficulty bin for an item.

        Args:
            item_key: Unique identifier for the item.

        Returns:
            Bin index (0 = easiest, n_bins-1 = hardest).
            Returns middle bin if item is not tracked.
        """
        difficulty = self.get_item_difficulty(item_key)
        if difficulty is None:
            return self.n_bins // 2  # Default to middle bin

        if not self._bin_boundaries:
            self._recompute_bins()

        # Bin assignment: higher score = lower bin index (easier)
        # We invert so bin 0 = easiest (highest reward)
        for i, boundary in enumerate(self._bin_boundaries):
            if difficulty >= boundary:
                return i
        return self.n_bins - 1

    def sample_bin(self, current_step: int = 0, total_steps: int = 1000) -> int:
        """
        Sample a target difficulty bin based on the curriculum strategy.

        Args:
            current_step: Current training step (for annealing strategies).
            total_steps: Total training steps planned.

        Returns:
            Target bin index to sample from (0 = easiest, n_bins-1 = hardest).
        """
        if self._strategy == CurriculumStrategy.UNIFORM:
            return random.randint(0, self.n_bins - 1)

        # Compute bin probabilities
        probs = self._compute_bin_probabilities(current_step, total_steps)

        # Temperature-scaled sampling
        if self.temperature != 1.0:
            log_probs = [math.log(max(p, 1e-10)) / self.temperature for p in probs]
            max_lp = max(log_probs)
            exp_probs = [math.exp(lp - max_lp) for lp in log_probs]
            total = sum(exp_probs)
            probs = [p / total for p in exp_probs]

        # Weighted random choice
        return random.choices(range(self.n_bins), weights=probs, k=1)[0]

    def _compute_bin_probabilities(
        self, current_step: int, total_steps: int
    ) -> List[float]:
        """Compute sampling probabilities for each bin."""
        progress = min(1.0, max(0.0, current_step / max(1, total_steps)))

        if self._strategy == CurriculumStrategy.EASY_FIRST:
            return self._easy_first_probs(progress)
        elif self._strategy == CurriculumStrategy.COMPETENCE_BASED:
            return self._competence_based_probs(progress)
        else:
            # Uniform fallback
            return [1.0 / self.n_bins] * self.n_bins

    def _easy_first_probs(self, progress: float) -> List[float]:
        """
        Easy-first: linearly anneal from easy-biased to uniform.

        At progress=0: strongly prefer easy items (bin 0).
        At progress=1: uniform sampling across all bins.
        """
        probs = []
        for i in range(self.n_bins):
            # Base: uniform
            uniform_prob = 1.0 / self.n_bins
            # Bias: exponential decay favoring low bins (easy)
            easy_bias = math.exp(-2.0 * i / max(1, self.n_bins - 1))
            # Anneal from biased to uniform
            prob = (1.0 - progress) * easy_bias + progress * uniform_prob
            probs.append(prob)

        # Normalize
        total = sum(probs)
        return [p / total for p in probs]

    def _competence_based_probs(self, progress: float) -> List[float]:
        """
        Competence-based: sample items near the competence frontier.

        The frontier moves from easy to hard as training progresses.
        Items where expected reward ~ competence_threshold are preferred.
        """
        # Competence level increases with training progress
        # Maps to which bin is at the frontier
        frontier_bin = progress * (self.n_bins - 1)

        probs = []
        for i in range(self.n_bins):
            # Gaussian-like probability centered on frontier bin
            distance = abs(i - frontier_bin)
            prob = math.exp(-0.5 * (distance ** 2))
            probs.append(prob)

        total = sum(probs)
        return [p / total for p in probs]

    def _recompute_bins(self) -> None:
        """Recompute bin boundaries based on current difficulty quantiles."""
        if not self._item_scores:
            self._bin_boundaries = []
            return

        # Sort scores descending (highest reward = easiest = bin 0)
        scores = sorted(
            [ema for ema, _ in self._item_scores.values()], reverse=True
        )

        if len(scores) < self.n_bins:
            # Not enough items to properly bin, use equal spacing
            min_s = min(scores)
            max_s = max(scores)
            if max_s == min_s:
                self._bin_boundaries = [min_s] * self.n_bins
            else:
                step = (max_s - min_s) / self.n_bins
                self._bin_boundaries = [
                    max_s - i * step for i in range(self.n_bins)
                ]
            return

        # Quantile-based boundaries
        boundaries = []
        for i in range(self.n_bins):
            idx = int(i * len(scores) / self.n_bins)
            idx = min(idx, len(scores) - 1)
            boundaries.append(scores[idx])
        self._bin_boundaries = boundaries

    def metrics_dict(self) -> Dict[str, float]:
        """
        Return curriculum stats for WandB logging.

        Returns:
            Dictionary with keys suitable for wandb.log().
        """
        if not self._item_scores:
            return {
                "curriculum/items_tracked": 0,
                "curriculum/strategy": 0,  # Can't log strings to wandb
            }

        scores = [ema for ema, _ in self._item_scores.values()]
        counts = [c for _, c in self._item_scores.values()]

        metrics = {
            "curriculum/items_tracked": float(len(scores)),
            "curriculum/mean_difficulty": sum(scores) / len(scores),
            "curriculum/min_difficulty": min(scores),
            "curriculum/max_difficulty": max(scores),
            "curriculum/total_updates": float(sum(counts)),
        }

        # Bin distribution
        if self._bin_boundaries:
            bin_counts = [0] * self.n_bins
            for key in self._item_scores:
                bin_idx = self.get_item_bin(key)
                bin_counts[bin_idx] += 1
            for i, count in enumerate(bin_counts):
                metrics[f"curriculum/bin_{i}_count"] = float(count)

        return metrics

    def state_dict(self) -> Dict[str, Any]:
        """Serialize state for checkpointing."""
        return {
            "strategy": self._strategy.value,
            "n_bins": self.n_bins,
            "temperature": self.temperature,
            "ema_alpha": self.ema_alpha,
            "competence_threshold": self.competence_threshold,
            "item_scores": dict(self._item_scores),
            "bin_boundaries": self._bin_boundaries,
            "last_rebin_count": self._last_rebin_count,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore state from checkpoint."""
        self._strategy = CurriculumStrategy(state["strategy"])
        self.n_bins = state["n_bins"]
        self.temperature = state["temperature"]
        self.ema_alpha = state["ema_alpha"]
        self.competence_threshold = state["competence_threshold"]
        self._item_scores = {
            k: tuple(v) for k, v in state["item_scores"].items()
        }
        self._bin_boundaries = state["bin_boundaries"]
        self._last_rebin_count = state["last_rebin_count"]
