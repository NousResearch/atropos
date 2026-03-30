"""
Tests for CurriculumScheduler -- difficulty-based sampling for RL training.

Tests cover:
- Uniform passthrough (default behavior unchanged)
- Easy-first annealing
- Competence-based frontier sampling
- EMA difficulty updates
- Bin assignment with quantile boundaries
- Metrics and state persistence
- Edge cases
"""

import math
import random

import pytest

from atroposlib.envs.curriculum import CurriculumScheduler, CurriculumStrategy


# ---------------------------------------------------------------------------
# Strategy tests
# ---------------------------------------------------------------------------


class TestUniformStrategy:
    def test_uniform_returns_valid_bins(self):
        scheduler = CurriculumScheduler(strategy="uniform", n_bins=5)
        bins = [scheduler.sample_bin(step, 1000) for step in range(100)]
        assert all(0 <= b < 5 for b in bins)

    def test_uniform_covers_all_bins(self):
        """Uniform should eventually sample from every bin."""
        random.seed(42)
        scheduler = CurriculumScheduler(strategy="uniform", n_bins=5)
        bins = set()
        for _ in range(200):
            bins.add(scheduler.sample_bin(0, 1000))
        assert bins == {0, 1, 2, 3, 4}


class TestEasyFirstStrategy:
    def test_early_training_prefers_easy(self):
        """At step 0, easy_first should strongly prefer low bins (easy)."""
        random.seed(42)
        scheduler = CurriculumScheduler(
            strategy="easy_first", n_bins=5, temperature=0.5
        )
        bins = [scheduler.sample_bin(0, 1000) for _ in range(200)]
        easy_count = sum(1 for b in bins if b <= 1)
        hard_count = sum(1 for b in bins if b >= 3)
        # Early training should have more easy than hard
        assert easy_count > hard_count

    def test_late_training_approaches_uniform(self):
        """Near the end (step~total), easy_first should be roughly uniform."""
        random.seed(42)
        scheduler = CurriculumScheduler(
            strategy="easy_first", n_bins=5, temperature=1.0
        )
        probs = scheduler._easy_first_probs(progress=1.0)
        # At progress=1.0, all probs should be near 1/n_bins
        for p in probs:
            assert abs(p - 0.2) < 0.05


class TestCompetenceBasedStrategy:
    def test_competence_frontier_moves(self):
        """The frontier should shift from easy to hard as training progresses."""
        scheduler = CurriculumScheduler(
            strategy="competence_based", n_bins=5, temperature=0.5
        )

        # Early training: frontier at easy bins
        random.seed(42)
        early_bins = [scheduler.sample_bin(0, 1000) for _ in range(200)]
        early_mean = sum(early_bins) / len(early_bins)

        # Late training: frontier at hard bins
        late_bins = [scheduler.sample_bin(900, 1000) for _ in range(200)]
        late_mean = sum(late_bins) / len(late_bins)

        # Late mean should be higher (harder bins)
        assert late_mean > early_mean

    def test_mid_training_prefers_middle(self):
        """At 50% progress, competence_based should prefer middle bins."""
        random.seed(42)
        scheduler = CurriculumScheduler(
            strategy="competence_based", n_bins=5, temperature=0.5
        )
        bins = [scheduler.sample_bin(500, 1000) for _ in range(300)]
        mid_count = sum(1 for b in bins if 1 <= b <= 3)
        edge_count = sum(1 for b in bins if b == 0 or b == 4)
        assert mid_count > edge_count


# ---------------------------------------------------------------------------
# EMA difficulty tracking tests
# ---------------------------------------------------------------------------


class TestDifficultyTracking:
    def test_ema_update(self):
        scheduler = CurriculumScheduler(strategy="uniform", ema_alpha=0.5)
        scheduler.update("item_1", 1.0)
        assert math.isclose(scheduler.get_item_difficulty("item_1"), 1.0)

        scheduler.update("item_1", 0.0)
        # EMA: 0.5 * 0.0 + 0.5 * 1.0 = 0.5
        assert math.isclose(scheduler.get_item_difficulty("item_1"), 0.5)

    def test_batch_update(self):
        scheduler = CurriculumScheduler(strategy="uniform")
        scheduler.update_batch("item_1", [0.8, 0.6, 1.0])
        # Should use average: 0.8
        diff = scheduler.get_item_difficulty("item_1")
        assert diff is not None
        assert math.isclose(diff, 0.8)

    def test_untracked_item_returns_none(self):
        scheduler = CurriculumScheduler(strategy="uniform")
        assert scheduler.get_item_difficulty("nonexistent") is None

    def test_multiple_items_tracked(self):
        scheduler = CurriculumScheduler(strategy="uniform")
        scheduler.update("easy", 0.9)
        scheduler.update("hard", 0.1)
        scheduler.update("medium", 0.5)

        assert scheduler.n_items_tracked == 3
        assert scheduler.get_item_difficulty("easy") > scheduler.get_item_difficulty("hard")


# ---------------------------------------------------------------------------
# Bin assignment tests
# ---------------------------------------------------------------------------


class TestBinAssignment:
    def test_easy_item_gets_low_bin(self):
        scheduler = CurriculumScheduler(strategy="uniform", n_bins=5)
        # Create items spanning the difficulty range
        for i in range(100):
            scheduler.update(f"item_{i}", i / 100.0)

        # High score = easy = low bin
        easy_bin = scheduler.get_item_bin("item_95")
        hard_bin = scheduler.get_item_bin("item_5")
        assert easy_bin < hard_bin

    def test_untracked_gets_middle_bin(self):
        scheduler = CurriculumScheduler(strategy="uniform", n_bins=5)
        assert scheduler.get_item_bin("unknown") == 2  # n_bins // 2

    def test_single_bin(self):
        scheduler = CurriculumScheduler(strategy="uniform", n_bins=1)
        scheduler.update("item", 0.5)
        assert scheduler.get_item_bin("item") == 0


# ---------------------------------------------------------------------------
# Metrics and state tests
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_metrics_dict_empty(self):
        scheduler = CurriculumScheduler(strategy="uniform")
        metrics = scheduler.metrics_dict()
        assert "curriculum/items_tracked" in metrics
        assert metrics["curriculum/items_tracked"] == 0

    def test_metrics_dict_populated(self):
        scheduler = CurriculumScheduler(strategy="uniform", n_bins=3)
        for i in range(60):  # Enough to trigger rebinning
            scheduler.update(f"item_{i}", i / 60.0)

        metrics = scheduler.metrics_dict()
        assert metrics["curriculum/items_tracked"] == 60
        assert "curriculum/mean_difficulty" in metrics
        assert "curriculum/min_difficulty" in metrics
        assert "curriculum/max_difficulty" in metrics
        assert "curriculum/total_updates" in metrics


class TestStatePersistence:
    def test_save_load_roundtrip(self):
        scheduler = CurriculumScheduler(
            strategy="competence_based", n_bins=3, temperature=0.8
        )
        for i in range(20):
            scheduler.update(f"item_{i}", i / 20.0)

        state = scheduler.state_dict()

        scheduler2 = CurriculumScheduler(strategy="uniform")
        scheduler2.load_state_dict(state)

        assert scheduler2.strategy == "competence_based"
        assert scheduler2.n_bins == 3
        assert math.isclose(scheduler2.temperature, 0.8)
        assert scheduler2.n_items_tracked == 20

        # Difficulty scores should match
        for i in range(20):
            key = f"item_{i}"
            d1 = scheduler.get_item_difficulty(key)
            d2 = scheduler2.get_item_difficulty(key)
            assert math.isclose(d1, d2)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Invalid curriculum strategy"):
            CurriculumScheduler(strategy="invalid")

    def test_invalid_n_bins_raises(self):
        with pytest.raises(ValueError, match="n_bins must be >= 1"):
            CurriculumScheduler(n_bins=0)

    def test_temperature_floor(self):
        scheduler = CurriculumScheduler(temperature=0.001)
        assert scheduler.temperature >= 0.01

    def test_ema_alpha_clamped(self):
        scheduler = CurriculumScheduler(ema_alpha=2.0)
        assert scheduler.ema_alpha <= 1.0

        scheduler2 = CurriculumScheduler(ema_alpha=-1.0)
        assert scheduler2.ema_alpha >= 0.0

    def test_empty_batch_update(self):
        scheduler = CurriculumScheduler(strategy="uniform")
        scheduler.update_batch("item", [])
        assert scheduler.n_items_tracked == 0

    def test_strategy_enum_values(self):
        assert CurriculumStrategy.UNIFORM.value == "uniform"
        assert CurriculumStrategy.EASY_FIRST.value == "easy_first"
        assert CurriculumStrategy.COMPETENCE_BASED.value == "competence_based"
