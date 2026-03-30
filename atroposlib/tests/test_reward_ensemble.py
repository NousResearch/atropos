"""
Tests for EnsembleReward -- reward aggregation with inter-rater reliability.

Tests cover:
- All aggregation strategies (mean, median, min, majority_vote)
- Krippendorff's alpha computation (perfect/no agreement)
- Disagreement tracking
- Registry integration
- Edge cases (empty completions, single reward function)
"""

import math
from typing import Any, List

import numpy as np
import pytest

from atroposlib.envs.reward_fns.ensemble_reward import (
    EnsembleReward,
    _krippendorff_alpha,
)
from atroposlib.envs.reward_fns.registry import RewardRegistry
from atroposlib.envs.reward_fns.reward_function import RewardFunction

# ---------------------------------------------------------------------------
# Test fixtures -- simple reward functions for composing ensembles
# ---------------------------------------------------------------------------


class ConstantReward(RewardFunction):
    """Returns a fixed score for every completion."""

    def __init__(self, value: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self._value = value

    def compute(self, completions: List[Any], **kwargs) -> List[float]:
        return [self._value] * len(completions)


class LengthReward(RewardFunction):
    """Scores by string length (for testing divergent reward signals)."""

    def compute(self, completions: List[Any], **kwargs) -> List[float]:
        return [float(len(self.get_content(c))) for c in completions]


class BinaryReward(RewardFunction):
    """Returns 1.0 if completion contains 'good', else 0.0."""

    def compute(self, completions: List[Any], **kwargs) -> List[float]:
        return [
            1.0 if "good" in self.get_content(c).lower() else 0.0 for c in completions
        ]


def _make_ensemble(strategy, reward_functions):
    """Helper to construct an EnsembleReward without going through registry."""
    ensemble = EnsembleReward.__new__(EnsembleReward)
    ensemble.weight = 1.0
    ensemble.strategy = strategy
    ensemble.track_disagreement = True
    ensemble.reward_functions = reward_functions
    ensemble.wandb_logger = None
    ensemble._name = None
    ensemble.config = {}
    ensemble.last_reliability_alpha = float("nan")
    ensemble.last_disagreement_scores = None
    ensemble._all_sub_rewards = None
    return ensemble


@pytest.fixture
def test_registry():
    """Create a clean registry with test reward functions."""
    reg = RewardRegistry()
    reg.register(name="constant")(ConstantReward)
    reg.register(name="length")(LengthReward)
    reg.register(name="binary")(BinaryReward)
    return reg


@pytest.fixture
def completions():
    """Sample completions for testing."""
    return ["short", "a medium length string", "good answer here"]


# ---------------------------------------------------------------------------
# Aggregation strategy tests
# ---------------------------------------------------------------------------


class TestMeanAggregation:
    def test_mean_of_identical_scores(self, completions):
        ensemble = _make_ensemble(
            "mean",
            [
                ConstantReward(value=2.0),
                ConstantReward(value=2.0),
            ],
        )
        scores = ensemble.compute(completions)
        assert len(scores) == 3
        assert all(math.isclose(s, 2.0, rel_tol=1e-9) for s in scores)

    def test_mean_of_different_scores(self, completions):
        ensemble = _make_ensemble(
            "mean",
            [
                ConstantReward(value=1.0),
                ConstantReward(value=3.0),
            ],
        )
        scores = ensemble.compute(completions)
        assert all(math.isclose(s, 2.0, rel_tol=1e-9) for s in scores)


class TestMedianAggregation:
    def test_median_rejects_outlier(self, completions):
        """Median should be robust to a single outlier reward function."""
        ensemble = _make_ensemble(
            "median",
            [
                ConstantReward(value=1.0),
                ConstantReward(value=1.0),
                ConstantReward(value=100.0),
            ],
        )
        scores = ensemble.compute(completions)
        assert all(math.isclose(s, 1.0, rel_tol=1e-9) for s in scores)


class TestMinAggregation:
    def test_min_is_conservative(self, completions):
        ensemble = _make_ensemble(
            "min",
            [
                ConstantReward(value=0.5),
                ConstantReward(value=0.8),
                ConstantReward(value=1.0),
            ],
        )
        scores = ensemble.compute(completions)
        assert all(math.isclose(s, 0.5, rel_tol=1e-9) for s in scores)


class TestMajorityVoteAggregation:
    def test_majority_positive(self, completions):
        ensemble = _make_ensemble(
            "majority_vote",
            [
                ConstantReward(value=1.0),
                ConstantReward(value=1.0),
                ConstantReward(value=-1.0),
            ],
        )
        scores = ensemble.compute(completions)
        assert all(math.isclose(s, 1.0) for s in scores)

    def test_majority_negative(self, completions):
        ensemble = _make_ensemble(
            "majority_vote",
            [
                ConstantReward(value=-1.0),
                ConstantReward(value=-1.0),
                ConstantReward(value=1.0),
            ],
        )
        scores = ensemble.compute(completions)
        assert all(math.isclose(s, 0.0) for s in scores)

    def test_tie_goes_positive(self, completions):
        ensemble = _make_ensemble(
            "majority_vote",
            [
                ConstantReward(value=1.0),
                ConstantReward(value=-1.0),
            ],
        )
        scores = ensemble.compute(completions)
        assert all(math.isclose(s, 1.0) for s in scores)


# ---------------------------------------------------------------------------
# Inter-rater reliability tests
# ---------------------------------------------------------------------------


class TestKrippendorffAlpha:
    def test_perfect_agreement(self):
        ratings = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
            ]
        )
        alpha = _krippendorff_alpha(ratings)
        assert math.isclose(alpha, 1.0, rel_tol=1e-9)

    def test_no_agreement(self):
        ratings = np.array(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
            ]
        )
        alpha = _krippendorff_alpha(ratings)
        assert alpha < 0.0

    def test_random_agreement(self):
        np.random.seed(42)
        ratings = np.random.rand(5, 100)
        alpha = _krippendorff_alpha(ratings)
        assert abs(alpha) < 0.3

    def test_insufficient_data(self):
        alpha = _krippendorff_alpha(np.array([[1.0, 2.0, 3.0]]))
        assert math.isnan(alpha)

        alpha = _krippendorff_alpha(np.array([[1.0], [2.0]]))
        assert math.isnan(alpha)


class TestReliabilityMetrics:
    def test_reliability_computed_after_scoring(self, completions):
        ensemble = _make_ensemble(
            "mean",
            [
                ConstantReward(value=1.0),
                ConstantReward(value=1.0),
            ],
        )
        ensemble.compute(completions)
        metrics = ensemble.reliability_metrics()
        assert "alpha" in metrics
        assert math.isclose(metrics["alpha"], 1.0, rel_tol=1e-9)

    def test_disagreement_tracked(self, completions):
        ensemble = _make_ensemble(
            "mean",
            [
                ConstantReward(value=0.0),
                ConstantReward(value=10.0),
            ],
        )
        ensemble.compute(completions)
        assert ensemble.last_disagreement_scores is not None
        assert len(ensemble.last_disagreement_scores) == len(completions)
        # Variance of [0.0, 10.0] = 25.0
        assert all(
            math.isclose(d, 25.0, rel_tol=1e-9)
            for d in ensemble.last_disagreement_scores
        )


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    def test_create_via_registry(self, test_registry):
        # EnsembleReward.__init__ resolves sub-rewards via the global registry,
        # so we must register our test fixtures there too.
        from atroposlib.envs.reward_fns.registry import registry as global_registry

        global_registry.register(name="test_constant")(ConstantReward)
        test_registry.register(name="ensemble")(EnsembleReward)

        ensemble = test_registry.create(
            {
                "type": "ensemble",
                "rewards": ["test_constant", "test_constant"],
                "strategy": "median",
            }
        )
        assert isinstance(ensemble, EnsembleReward)
        assert ensemble.strategy == "median"
        assert len(ensemble.reward_functions) == 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_completions(self):
        ensemble = _make_ensemble("mean", [ConstantReward(value=1.0)])
        scores = ensemble.compute([])
        assert scores == []

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Invalid strategy"):
            EnsembleReward(rewards=[], strategy="nonexistent")

    def test_name_format(self):
        ensemble = _make_ensemble(
            "median",
            [
                ConstantReward(value=1.0),
                LengthReward(),
            ],
        )
        name = ensemble.name
        assert "ensemble_median" in name
        assert "constantreward" in name
        assert "lengthreward" in name
