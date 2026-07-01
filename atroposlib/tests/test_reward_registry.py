"""Tests for the reward-function registry's lazy loader."""

from atroposlib.envs.reward_fns import registry


def test_create_resolves_correct_class_in_multiclass_module():
    """``r1_reward.py`` defines AccuracyXReward, FormatReasoningReward and
    R1Reward. ``create("r1")`` must return R1Reward, not the alphabetically
    first subclass in the module.
    """
    reward = registry.create("r1")
    assert type(reward).__name__ == "R1Reward"


def test_create_resolves_reward_suffix_alias():
    """The ``<name>_reward`` alias resolves to the same class."""
    reward = registry.create("r1_reward")
    assert type(reward).__name__ == "R1Reward"


def test_create_single_class_module_still_works():
    """A module that defines a single reward class keeps working."""
    reward = registry.create("reasoning_steps")
    assert type(reward).__name__ == "ReasoningStepsReward"
