"""
Ensemble reward function with robust aggregation and inter-rater reliability.

Extends the CombinedReward pattern with:
- Multiple aggregation strategies (mean, median, min, majority_vote)
- Inter-rater reliability metrics (Krippendorff's alpha)
- Disagreement tracking for reward hacking detection

Usage:
    reward_fn = registry.create("ensemble", rewards=["accuracy", "format"], strategy="median")
    scores = reward_fn(completions, **kwargs)

    # Access reliability metrics
    alpha = reward_fn.last_reliability_alpha
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .registry import registry
from .reward_function import RewardFunction

logger = logging.getLogger(__name__)


def _krippendorff_alpha(ratings_matrix: np.ndarray) -> float:
    """
    Compute Krippendorff's alpha for inter-rater reliability.

    Uses the interval/ratio metric (squared differences).

    Args:
        ratings_matrix: Shape (n_raters, n_items). NaN values indicate
                        missing ratings and are excluded from computation.

    Returns:
        Alpha value in [-1, 1]. 1 = perfect agreement, 0 = chance agreement,
        negative = systematic disagreement.
    """
    n_raters, n_items = ratings_matrix.shape

    if n_raters < 2 or n_items < 2:
        return float("nan")

    # Build coincidence matrix approach using pairwise disagreements
    # For each item, compute observed disagreement across all rater pairs
    observed_disagreement = 0.0
    total_pairs = 0

    for item_idx in range(n_items):
        values = ratings_matrix[:, item_idx]
        valid = values[~np.isnan(values)]
        n_valid = len(valid)
        if n_valid < 2:
            continue

        # Sum of squared differences for all pairs within this item
        for i in range(n_valid):
            for j in range(i + 1, n_valid):
                observed_disagreement += (valid[i] - valid[j]) ** 2
                total_pairs += 1

    if total_pairs == 0:
        return float("nan")

    observed_disagreement /= total_pairs

    # Expected disagreement: pairwise differences across ALL values
    all_valid = ratings_matrix[~np.isnan(ratings_matrix)]
    n_all = len(all_valid)
    if n_all < 2:
        return float("nan")

    expected_disagreement = 0.0
    expected_pairs = 0
    for i in range(n_all):
        for j in range(i + 1, n_all):
            expected_disagreement += (all_valid[i] - all_valid[j]) ** 2
            expected_pairs += 1

    if expected_pairs == 0:
        return float("nan")

    expected_disagreement /= expected_pairs

    if expected_disagreement == 0.0:
        # All raters gave identical scores -- perfect agreement
        return 1.0

    alpha = 1.0 - (observed_disagreement / expected_disagreement)
    return float(alpha)


@registry.register
class EnsembleReward(RewardFunction):
    """
    Ensemble reward function that aggregates multiple reward functions
    with robust strategies and inter-rater reliability tracking.

    Compared to CombinedReward, this adds:
    - Median and min (conservative) aggregation for robustness
    - Majority vote for binary reward environments
    - Krippendorff's alpha inter-rater reliability metric
    - Per-item disagreement tracking for reward hacking detection

    Strategies:
        - "mean": Weighted average (same as CombinedReward)
        - "median": Median across reward functions (robust to outliers)
        - "min": Conservative -- use the minimum score (prevents reward hacking)
        - "majority_vote": For binary rewards -- majority wins (ties -> positive)
    """

    def __init__(
        self,
        rewards: List[Union[str, Dict]],
        strategy: str = "mean",
        weight: float = 1.0,
        track_disagreement: bool = True,
        **kwargs,
    ):
        """
        Initialize the ensemble reward function.

        Args:
            rewards: List of reward function names or config dicts.
                     Resolved via RewardRegistry.
            strategy: Aggregation strategy. One of: "mean", "median",
                      "min", "majority_vote".
            weight: Weight for this ensemble when used inside another
                    CombinedReward.
            track_disagreement: If True, track per-item reward variance
                                for disagreement analysis.
            **kwargs: Additional parameters passed to RewardFunction.
        """
        super().__init__(weight=weight, **kwargs)

        valid_strategies = {"mean", "median", "min", "majority_vote"}
        if strategy not in valid_strategies:
            raise ValueError(
                f"Invalid strategy '{strategy}'. Must be one of: {valid_strategies}"
            )

        self.strategy = strategy
        self.track_disagreement = track_disagreement
        self.reward_functions: List[RewardFunction] = []

        # Initialize sub-reward functions via registry
        for reward_config in rewards:
            self.reward_functions.append(registry.create(reward_config))

        if len(self.reward_functions) < 2:
            warnings.warn(
                "EnsembleReward initialized with fewer than 2 reward functions. "
                "Inter-rater reliability metrics will not be meaningful.",
                stacklevel=2,
            )

        # State for reliability tracking
        self.last_reliability_alpha: float = float("nan")
        self.last_disagreement_scores: Optional[List[float]] = None
        self._all_sub_rewards: Optional[List[List[float]]] = None

    @property
    def name(self) -> str:
        sub_names = ",".join(r.name for r in self.reward_functions)
        return f"ensemble_{self.strategy}({sub_names})"

    def set_wandb_logger(self, wandb_logger):
        """Propagate WandB logger to all sub-reward functions."""
        super().set_wandb_logger(wandb_logger)
        for reward_fn in self.reward_functions:
            reward_fn.set_wandb_logger(wandb_logger)

    def compute(self, completions: List[Any], **kwargs) -> List[float]:
        """
        Compute ensemble reward scores.

        Calls all sub-reward functions, aggregates by strategy,
        and computes reliability metrics.

        Args:
            completions: List of completions to evaluate.
            **kwargs: Additional context passed to sub-rewards.

        Returns:
            Aggregated reward scores, one per completion.
        """
        if not completions:
            return []

        n_completions = len(completions)

        # Collect all sub-reward scores
        all_rewards: List[List[float]] = []
        for reward_fn in self.reward_functions:
            try:
                scores = reward_fn.compute(completions, **kwargs)
                if len(scores) != n_completions:
                    logger.warning(
                        "Reward function %s returned %d scores for %d completions. "
                        "Padding/truncating.",
                        reward_fn.name,
                        len(scores),
                        n_completions,
                    )
                    # Pad or truncate
                    if len(scores) < n_completions:
                        scores = scores + [0.0] * (n_completions - len(scores))
                    else:
                        scores = scores[:n_completions]
                all_rewards.append(scores)
            except Exception as e:
                logger.error("Error in reward function %s: %s", reward_fn.name, e)
                all_rewards.append([0.0] * n_completions)

        self._all_sub_rewards = all_rewards

        if not all_rewards:
            return [0.0] * n_completions

        # Convert to numpy for efficient aggregation
        # Shape: (n_reward_fns, n_completions)
        reward_matrix = np.array(all_rewards, dtype=np.float64)

        # Aggregate by strategy
        if self.strategy == "mean":
            aggregated = np.mean(reward_matrix, axis=0)
        elif self.strategy == "median":
            aggregated = np.median(reward_matrix, axis=0)
        elif self.strategy == "min":
            aggregated = np.min(reward_matrix, axis=0)
        elif self.strategy == "majority_vote":
            # Treat positive as vote for 1, non-positive as vote for 0
            votes = (reward_matrix > 0).astype(np.float64)
            vote_fractions = np.mean(votes, axis=0)
            # Majority wins; ties (0.5) go to positive
            aggregated = np.where(vote_fractions >= 0.5, 1.0, 0.0)
        else:
            # Should not reach here due to __init__ validation
            aggregated = np.mean(reward_matrix, axis=0)

        # Compute reliability metrics
        self._compute_reliability_metrics(reward_matrix)

        # Track per-item disagreement
        if self.track_disagreement:
            self.last_disagreement_scores = np.var(
                reward_matrix, axis=0
            ).tolist()

        return aggregated.tolist()

    def _compute_reliability_metrics(self, reward_matrix: np.ndarray):
        """
        Compute and store inter-rater reliability metrics.

        Args:
            reward_matrix: Shape (n_raters, n_items)
        """
        n_raters, n_items = reward_matrix.shape

        if n_raters < 2 or n_items < 2:
            self.last_reliability_alpha = float("nan")
            return

        self.last_reliability_alpha = _krippendorff_alpha(reward_matrix)

    def reliability_metrics(self) -> Dict[str, float]:
        """
        Return the latest inter-rater reliability metrics.

        Returns:
            Dictionary with reliability statistics:
            - alpha: Krippendorff's alpha
            - mean_disagreement: Average per-item variance across raters
            - max_disagreement: Maximum per-item variance (worst agreement)
        """
        metrics = {
            "alpha": self.last_reliability_alpha,
        }

        if self.last_disagreement_scores is not None:
            scores = self.last_disagreement_scores
            metrics["mean_disagreement"] = (
                sum(scores) / len(scores) if scores else 0.0
            )
            metrics["max_disagreement"] = max(scores) if scores else 0.0

        return metrics

    def log_metrics(self, raw_rewards: List[float], weighted_rewards: List[float]):
        """Log ensemble-specific metrics alongside standard reward metrics."""
        super().log_metrics(raw_rewards, weighted_rewards)

        if not self.wandb_logger:
            return

        reliability = self.reliability_metrics()
        wandb_metrics = {}

        if not np.isnan(reliability.get("alpha", float("nan"))):
            wandb_metrics[f"reward/{self.name}/reliability_alpha"] = reliability[
                "alpha"
            ]

        if "mean_disagreement" in reliability:
            wandb_metrics[f"reward/{self.name}/mean_disagreement"] = reliability[
                "mean_disagreement"
            ]
            wandb_metrics[f"reward/{self.name}/max_disagreement"] = reliability[
                "max_disagreement"
            ]

        if wandb_metrics:
            self.wandb_logger.log(wandb_metrics)
