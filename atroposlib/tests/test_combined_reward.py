"""Tests that CombinedReward applies each sub-reward's weight."""

from atroposlib.envs.reward_fns.combined_reward import CombinedReward
from atroposlib.envs.reward_fns.reward_function import RewardFunction


class _Const(RewardFunction):
    def __init__(self, value, weight):
        super().__init__(weight=weight)
        self._value = value

    def compute(self, completions, **kwargs):
        return [self._value] * len(completions)


def _combined(normalization):
    # Construct directly with controlled sub-rewards (values 1.0 @ w=1 and
    # 0.5 @ w=3) instead of going through the registry.
    cr = CombinedReward.__new__(CombinedReward)
    RewardFunction.__init__(cr, weight=1.0)
    cr.normalization = normalization
    cr.reward_functions = [_Const(1.0, 1.0), _Const(0.5, 3.0)]
    return cr


def test_combined_reward_weighted_sum_none():
    # normalization="none" -> weighted sum: 1*1.0 + 3*0.5 = 2.5
    assert _combined("none").compute(["x"]) == [2.5]


def test_combined_reward_weighted_average_sum():
    # normalization="sum" -> weighted average: (1*1.0 + 3*0.5) / (1 + 3) = 0.625
    assert _combined("sum").compute(["x"]) == [0.625]
