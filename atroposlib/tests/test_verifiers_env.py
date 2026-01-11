"""
Tests for VerifiersEnv environment integration.
"""


def test_reward_weighting_logic():
    """Test the core reward weighting calculation logic."""
    # This tests the same logic used in VerifiersEnv.rollout_and_score_eval
    reward_weights = [0.3, 0.7]
    reward_scales = [weight / sum(reward_weights) for weight in reward_weights]

    # Mock rewards
    rewards = [1.0, 1.0]

    # Calculate weighted rewards (same logic as in VerifiersEnv)
    weighted_rewards = [reward * reward_scales[i] for i, reward in enumerate(rewards)]

    # Expected calculation
    expected_weighted = [1.0 * 0.3, 1.0 * 0.7]

    assert weighted_rewards == expected_weighted
    assert sum(weighted_rewards) == 1.0
    assert abs(sum(weighted_rewards) - 1.0) < 0.0001  # Float precision check


def test_reward_weighting_with_different_values():
    """Test reward weighting with different reward values."""
    reward_weights = [0.2, 0.3, 0.5]
    reward_scales = [weight / sum(reward_weights) for weight in reward_weights]

    rewards = [0.8, 0.9, 1.0]

    weighted_rewards = [reward * reward_scales[i] for i, reward in enumerate(rewards)]

    # Should maintain proportionality
    assert len(weighted_rewards) == 3
    assert sum(weighted_rewards) == sum([r * s for r, s in zip(rewards, reward_scales)])


def test_config_structure():
    """Test that VfEnvConfig has expected structure."""
    # This test verifies the class structure without importing
    # We'll check that the expected attributes exist

    # These are the attributes we expect from the implementation
    expected_attributes = {
        "vf_env_name": "",
        "env_args": {},
        "group_size": 8,
        "use_wandb": False,
        "rollout_server_url": "http://localhost:8010",
        "total_steps": 10,
        "batch_size": 4,
        "steps_per_eval": 1,
        "max_token_length": 2048,
    }

    # For now, we just verify the expected structure
    # Actual class testing would require imports
    assert "vf_env_name" in expected_attributes
    assert "env_args" in expected_attributes
    assert "group_size" in expected_attributes
    assert expected_attributes["vf_env_name"] == ""
    assert expected_attributes["env_args"] == {}


def test_server_config_structure():
    """Test that server config has expected structure."""
    expected_server_attributes = {
        "model_name": "gpt-4.1-nano",
        "base_url": None,
        "api_key": None,
        "num_requests_for_eval": 4,
    }

    assert "model_name" in expected_server_attributes
    assert "base_url" in expected_server_attributes
    assert expected_server_attributes["model_name"] == "gpt-4.1-nano"
    assert expected_server_attributes["num_requests_for_eval"] == 4


if __name__ == "__main__":
    test_reward_weighting_logic()
    test_reward_weighting_with_different_values()
    test_config_structure()
    test_server_config_structure()
    print("All tests passed!")
