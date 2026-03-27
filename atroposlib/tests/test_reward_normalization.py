"""
Tests for RewardNormalizer -- online reward normalization with Welford's algorithm.

Tests cover:
- Welford's accumulator numerical accuracy vs numpy
- Z-score normalization
- Min-max normalization
- Clipping behavior
- Warmup period
- State save/load roundtrip
- Edge cases (empty input, constant values, mode validation)
"""

import math
from typing import List

import numpy as np
import pytest

from atroposlib.envs.reward_normalization import RewardNormalizer, WelfordAccumulator


# ---------------------------------------------------------------------------
# WelfordAccumulator tests
# ---------------------------------------------------------------------------


class TestWelfordAccumulator:
    def test_single_value(self):
        acc = WelfordAccumulator()
        acc.update(5.0)
        assert acc.count == 1
        assert math.isclose(acc.mean, 5.0)
        assert math.isclose(acc.variance, 0.0)

    def test_matches_numpy(self):
        """Welford's running stats should match numpy's batch computation."""
        np.random.seed(42)
        values = np.random.randn(1000).tolist()

        acc = WelfordAccumulator()
        acc.update_batch(values)

        expected_mean = np.mean(values)
        expected_var = np.var(values)  # population variance

        assert math.isclose(acc.mean, expected_mean, rel_tol=1e-9)
        assert math.isclose(acc.variance, expected_var, rel_tol=1e-6)

    def test_min_max_tracking(self):
        acc = WelfordAccumulator()
        acc.update_batch([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
        assert math.isclose(acc.min_val, 1.0)
        assert math.isclose(acc.max_val, 9.0)

    def test_state_roundtrip(self):
        acc = WelfordAccumulator()
        acc.update_batch([1.0, 2.0, 3.0, 4.0, 5.0])
        state = acc.state_dict()

        acc2 = WelfordAccumulator()
        acc2.load_state_dict(state)

        assert acc2.count == acc.count
        assert math.isclose(acc2.mean, acc.mean)
        assert math.isclose(acc2.variance, acc.variance)
        assert math.isclose(acc2.min_val, acc.min_val)
        assert math.isclose(acc2.max_val, acc.max_val)

    def test_empty_accumulator(self):
        acc = WelfordAccumulator()
        assert acc.count == 0
        assert math.isclose(acc.mean, 0.0)
        assert math.isclose(acc.variance, 0.0)
        assert math.isclose(acc.std, 0.0)


# ---------------------------------------------------------------------------
# RewardNormalizer z-score tests
# ---------------------------------------------------------------------------


class TestZScoreNormalization:
    def test_zscore_centers_around_zero(self):
        normalizer = RewardNormalizer(mode="zscore", clip=None, warmup=0)
        # Feed enough data to establish stats
        normalizer.normalize([1.0, 2.0, 3.0, 4.0, 5.0] * 10)
        # Now normalize a new batch
        result = normalizer.normalize([3.0])  # mean should be ~3.0
        assert abs(result[0]) < 0.1  # Should be near 0

    def test_zscore_output_scale(self):
        normalizer = RewardNormalizer(mode="zscore", clip=None, warmup=0)
        # Standard normal-ish data
        np.random.seed(42)
        data = np.random.randn(500).tolist()
        normalizer.normalize(data)

        # Normalize the same data again
        result = normalizer.normalize(data)
        # After normalization, std should be approximately 1.0
        result_std = np.std(result)
        assert 0.8 < result_std < 1.2

    def test_zscore_constant_values(self):
        """Constant values should normalize to 0."""
        normalizer = RewardNormalizer(mode="zscore", clip=None, warmup=0)
        result = normalizer.normalize([5.0, 5.0, 5.0, 5.0, 5.0])
        assert all(math.isclose(s, 0.0) for s in result)


# ---------------------------------------------------------------------------
# RewardNormalizer min-max tests
# ---------------------------------------------------------------------------


class TestMinMaxNormalization:
    def test_minmax_scales_to_unit_range(self):
        normalizer = RewardNormalizer(mode="minmax", clip=None, warmup=0)
        normalizer.normalize([0.0, 10.0])  # Establish min=0, max=10
        result = normalizer.normalize([0.0, 5.0, 10.0])
        assert math.isclose(result[0], 0.0, abs_tol=1e-6)
        assert math.isclose(result[1], 0.5, abs_tol=1e-3)
        assert math.isclose(result[2], 1.0, abs_tol=1e-6)

    def test_minmax_constant_returns_half(self):
        normalizer = RewardNormalizer(mode="minmax", clip=None, warmup=0)
        result = normalizer.normalize([3.0, 3.0, 3.0])
        assert all(math.isclose(s, 0.5) for s in result)


# ---------------------------------------------------------------------------
# Clipping tests
# ---------------------------------------------------------------------------


class TestClipping:
    def test_clip_bounds(self):
        normalizer = RewardNormalizer(mode="zscore", clip=2.0, warmup=0)
        # Feed data with a big outlier
        normalizer.normalize([0.0] * 100)
        result = normalizer.normalize([1000.0])
        assert result[0] <= 2.0

    def test_no_clip_when_disabled(self):
        normalizer = RewardNormalizer(mode="zscore", clip=None, warmup=0)
        normalizer.normalize([0.0] * 100)
        result = normalizer.normalize([1000.0])
        assert result[0] > 2.0  # Should NOT be clipped

    def test_negative_clip_disabled(self):
        normalizer = RewardNormalizer(mode="zscore", clip=-1.0, warmup=0)
        assert normalizer.clip is None


# ---------------------------------------------------------------------------
# Warmup tests
# ---------------------------------------------------------------------------


class TestWarmup:
    def test_warmup_returns_raw(self):
        normalizer = RewardNormalizer(mode="zscore", clip=None, warmup=10)
        # During warmup, should return raw scores
        result = normalizer.normalize([5.0, 10.0])
        assert math.isclose(result[0], 5.0)
        assert math.isclose(result[1], 10.0)

    def test_warmup_transition(self):
        normalizer = RewardNormalizer(mode="zscore", clip=None, warmup=5)
        # Feed 3 values (under warmup)
        r1 = normalizer.normalize([1.0, 2.0, 3.0])
        assert not normalizer.is_warmed_up
        # Raw values during warmup
        assert math.isclose(r1[0], 1.0)

        # Feed 3 more (now at 6, above warmup)
        r2 = normalizer.normalize([4.0, 5.0, 6.0])
        assert normalizer.is_warmed_up
        # Should be normalized now (not raw)
        assert not math.isclose(r2[0], 4.0)


# ---------------------------------------------------------------------------
# State persistence tests
# ---------------------------------------------------------------------------


class TestStatePersistence:
    def test_save_load_roundtrip(self):
        normalizer = RewardNormalizer(mode="zscore", clip=3.0, warmup=5)
        normalizer.normalize([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

        state = normalizer.state_dict()

        normalizer2 = RewardNormalizer()
        normalizer2.load_state_dict(state)

        assert normalizer2.mode == "zscore"
        assert normalizer2.clip == 3.0
        assert normalizer2.warmup == 5
        assert normalizer2.count == normalizer.count
        assert math.isclose(normalizer2.mean, normalizer.mean)
        assert math.isclose(normalizer2.std, normalizer.std)

    def test_loaded_normalizer_continues(self):
        """A loaded normalizer should produce same results as the original."""
        normalizer = RewardNormalizer(mode="zscore", clip=5.0, warmup=0)
        normalizer.normalize([1.0, 2.0, 3.0, 4.0, 5.0] * 10)
        state = normalizer.state_dict()

        normalizer2 = RewardNormalizer()
        normalizer2.load_state_dict(state)

        test_data = [2.5, 3.5, 4.5]
        r1 = normalizer.normalize(test_data)
        r2 = normalizer2.normalize(test_data)

        # Results won't be identical because normalize also updates stats,
        # but they should be very close for the first call after loading
        for a, b in zip(r1, r2):
            assert math.isclose(a, b, rel_tol=1e-3)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_input(self):
        normalizer = RewardNormalizer(mode="zscore")
        assert normalizer.normalize([]) == []

    def test_none_mode_passthrough(self):
        normalizer = RewardNormalizer(mode="none")
        scores = [1.0, 2.0, 3.0]
        assert normalizer.normalize(scores) == scores

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid normalization mode"):
            RewardNormalizer(mode="invalid")

    def test_metrics_dict_keys(self):
        normalizer = RewardNormalizer(mode="zscore", warmup=0)
        normalizer.normalize([1.0, 2.0, 3.0])
        metrics = normalizer.metrics_dict()
        assert "reward_norm/count" in metrics
        assert "reward_norm/mean" in metrics
        assert "reward_norm/std" in metrics
        assert "reward_norm/min" in metrics
        assert "reward_norm/max" in metrics
