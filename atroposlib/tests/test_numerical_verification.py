"""
Tests for numerical verification utilities.

Tests cover:
- Reward determinism checks
- Advantage stability (NaN, Inf, magnitude detection)
- Floating-point precision comparison
- Score distribution analysis (collapse, explosion, bias)
- Integration with existing advantage computation
"""

import math


import numpy as np

from atroposlib.utils.numerical_verification import (
    compare_fp_precision,
    verify_advantage_stability,
    verify_reward_determinism,
    verify_score_distribution,
)


# ---------------------------------------------------------------------------
# Reward determinism tests
# ---------------------------------------------------------------------------


class TestRewardDeterminism:
    def test_deterministic_function_passes(self):
        def constant_reward(inputs, **kwargs):
            return [1.0] * len(inputs)

        assert verify_reward_determinism(constant_reward, ["a", "b", "c"])

    def test_nondeterministic_function_fails(self):
        call_count = [0]

        def noisy_reward(inputs, **kwargs):
            call_count[0] += 1
            return [float(call_count[0]) + i for i in range(len(inputs))]

        assert not verify_reward_determinism(noisy_reward, ["a", "b"])

    def test_math_operations_are_deterministic(self):
        def math_reward(inputs, **kwargs):
            return [math.sqrt(abs(hash(str(x)))) for x in inputs]

        assert verify_reward_determinism(math_reward, [1, 2, 3, 4, 5])

    def test_length_mismatch_fails(self):
        call_count = [0]

        def varying_length(inputs, **kwargs):
            call_count[0] += 1
            if call_count[0] > 1:
                return [1.0]
            return [1.0] * len(inputs)

        assert not verify_reward_determinism(varying_length, ["a", "b"])

    def test_single_run_trivially_true(self):
        def any_fn(inputs, **kwargs):
            return [0.0]

        assert verify_reward_determinism(any_fn, ["a"], n_runs=1)

    def test_failing_function_returns_false(self):
        def error_fn(inputs, **kwargs):
            raise RuntimeError("boom")

        assert not verify_reward_determinism(error_fn, ["a"])


# ---------------------------------------------------------------------------
# Advantage stability tests
# ---------------------------------------------------------------------------


class TestAdvantageStability:
    def test_stable_advantages(self):
        advantages = [0.1, -0.2, 0.3, -0.1, 0.05]
        report = verify_advantage_stability(advantages)
        assert report.is_stable
        assert not report.has_nan
        assert not report.has_inf

    def test_nan_detected(self):
        advantages = [0.1, float("nan"), 0.3]
        report = verify_advantage_stability(advantages)
        assert not report.is_stable
        assert report.has_nan
        assert "NaN" in report.issues[0]

    def test_inf_detected(self):
        advantages = [0.1, float("inf"), -float("inf")]
        report = verify_advantage_stability(advantages)
        assert not report.is_stable
        assert report.has_inf

    def test_excessive_magnitude(self):
        advantages = [0.1, 200.0, -0.3]
        report = verify_advantage_stability(advantages, max_magnitude=100.0)
        assert not report.is_stable
        assert report.max_magnitude > 100.0

    def test_all_nan_inf(self):
        advantages = [float("nan"), float("inf")]
        report = verify_advantage_stability(advantages)
        assert not report.is_stable

    def test_numpy_array_input(self):
        advantages = np.array([0.1, -0.2, 0.3])
        report = verify_advantage_stability(advantages)
        assert report.is_stable

    def test_summary_format(self):
        report = verify_advantage_stability([0.1, 0.2])
        assert "STABLE" in report.summary()

        report = verify_advantage_stability([float("nan")])
        assert "UNSTABLE" in report.summary()


# ---------------------------------------------------------------------------
# Precision comparison tests
# ---------------------------------------------------------------------------


class TestFPPrecisionComparison:
    def test_identity_function_passes(self):
        def identity(inputs, **kwargs):
            return inputs

        report = compare_fp_precision(identity, [1.0, 2.0, 3.0])
        assert report.is_acceptable

    def test_precision_sensitive_function(self):
        def sensitive(inputs, **kwargs):
            # Return values directly -- precision differences come from
            # input casting: float16 can't represent these precisely
            return [x * 1.0001 for x in inputs]

        report = compare_fp_precision(
            sensitive,
            [1000.001, 2048.123, 4000.567],  # Values float16 can't represent exactly
            precisions=["float32", "float16"],
        )
        # float16 truncates these values, causing output divergence
        assert report.per_precision.get("float16", 0) > 0

    def test_report_structure(self):
        def identity(inputs, **kwargs):
            return inputs

        report = compare_fp_precision(
            identity, [1.0, 2.0], precisions=["float32"]
        )
        assert report.reference_precision == "float64"
        assert "float32" in report.compared_precisions
        assert isinstance(report.max_divergence, float)
        assert isinstance(report.mean_divergence, float)
        assert isinstance(report.divergent_fraction, float)

    def test_summary_format(self):
        def identity(inputs, **kwargs):
            return inputs

        report = compare_fp_precision(identity, [1.0])
        summary = report.summary()
        assert "PASS" in summary or "FAIL" in summary


# ---------------------------------------------------------------------------
# Score distribution tests
# ---------------------------------------------------------------------------


class TestScoreDistribution:
    def test_healthy_distribution(self):
        np.random.seed(42)
        scores = np.random.uniform(-0.5, 0.5, 100).tolist()
        report = verify_score_distribution(scores)
        assert report.is_healthy
        assert not report.is_collapsed
        assert not report.is_exploded
        assert not report.is_biased

    def test_collapsed_distribution(self):
        scores = [0.5] * 100
        report = verify_score_distribution(scores)
        assert not report.is_healthy
        assert report.is_collapsed
        assert "collapse" in report.issues[0].lower()

    def test_exploded_distribution(self):
        scores = [0.0, 0.1, 500.0, -300.0]
        report = verify_score_distribution(scores, explosion_threshold=100.0)
        assert not report.is_healthy
        assert report.is_exploded

    def test_biased_distribution(self):
        # 95% of scores at maximum
        scores = [1.0] * 95 + [0.0] * 5
        report = verify_score_distribution(
            scores, expected_range=(-1.0, 1.0), bias_threshold=0.9
        )
        assert not report.is_healthy
        assert report.is_biased

    def test_empty_scores(self):
        report = verify_score_distribution([])
        assert report.is_healthy

    def test_summary_format(self):
        report = verify_score_distribution([0.1, 0.2, 0.3])
        assert "HEALTHY" in report.summary()

        report = verify_score_distribution([0.5] * 100)
        assert "UNHEALTHY" in report.summary()


# ---------------------------------------------------------------------------
# Integration with existing advantage computation
# ---------------------------------------------------------------------------


class TestIntegrationWithAdvantages:
    def test_grpo_style_advantages(self):
        """Test with GRPO-style advantages (scores - mean(scores))."""
        scores = [0.2, 0.8, 0.5, 0.3, 0.7]
        mean_score = sum(scores) / len(scores)
        advantages = [s - mean_score for s in scores]

        report = verify_advantage_stability(advantages)
        assert report.is_stable

        dist_report = verify_score_distribution(scores, expected_range=(0.0, 1.0))
        assert dist_report.is_healthy

    def test_degenerate_grpo_scores(self):
        """All identical scores should produce collapsed advantages."""
        scores = [0.5] * 10
        mean_score = sum(scores) / len(scores)
        advantages = [s - mean_score for s in scores]

        report = verify_advantage_stability(advantages)
        assert report.is_stable  # All zeros is technically stable

        dist = verify_score_distribution(scores)
        assert dist.is_collapsed  # But score distribution is collapsed
