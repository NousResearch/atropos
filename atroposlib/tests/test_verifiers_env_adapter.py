import pytest

from atroposlib.envs.verifiers_env import (
    normalize_vf_env_id,
    reward_scales,
    weighted_sum,
)


def test_normalize_vf_env_id_slash_to_hyphen():
    assert normalize_vf_env_id("will/wordle") == "wordle"


def test_normalize_vf_env_id_idempotent():
    assert normalize_vf_env_id("retry-updated") == "retry-updated"


def test_normalize_vf_env_id_strips_whitespace():
    assert normalize_vf_env_id("  will/wordle  ") == "wordle"


def test_normalize_vf_env_id_strips_version_suffix():
    assert normalize_vf_env_id("will/wordle@latest") == "wordle"


def test_normalize_vf_env_id_empty_raises():
    with pytest.raises(ValueError):
        normalize_vf_env_id("")


def test_reward_scales_normalizes_weights():
    assert reward_scales([1.0, 2.0, 1.0]) == [0.25, 0.5, 0.25]


def test_reward_scales_falls_back_when_total_nonpositive():
    assert reward_scales([0.0, 0.0]) == [0.5, 0.5]


def test_weighted_sum_matches_manual_aggregation():
    scales = reward_scales([1.0, 2.0])
    score = weighted_sum([0.5, 1.0], scales)
    assert score == pytest.approx((0.5 * (1.0 / 3.0)) + (1.0 * (2.0 / 3.0)))
