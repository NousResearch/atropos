#!/usr/bin/env python3
"""
Example configuration for training the checkers environment.
This shows how to set up a training run for the checkers environment.
"""

from checkers_env import CheckersEnvConfig, OpponentType, Player

from atroposlib.envs.base import APIServerConfig


# Example training configuration
def get_checkers_config() -> tuple:
    """Get example configuration for checkers training"""

    # Environment configuration
    env_config = CheckersEnvConfig(
        opponent_type=OpponentType.RANDOM,  # Start with random opponent
        max_episode_turns=100,  # Maximum moves per game
        eval_episodes=50,  # Number of evaluation episodes
        thinking_enabled=True,  # Enable <think> tags for reasoning
        temperature=0.7,  # LLM sampling temperature
        ai_plays_as=Player.RED,  # AI plays as red pieces
        # Base environment configuration
        num_items=1000,  # Number of training episodes
        concurrent_limit=10,  # Concurrent episodes
        server_timeout=300,  # Server timeout in seconds
        reward_fn="checkers_reward",  # Custom reward function name
        # Evaluation settings
        eval_freq=100,  # Evaluate every 100 episodes
        eval_on_last_n_items=50,  # Evaluate on last 50 episodes
        items_per_save=100,  # Save checkpoint every 100 episodes
    )

    # API server configuration (adjust for your setup)
    server_configs = [
        APIServerConfig(
            model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",  # Or your preferred model
            base_url="http://localhost:9001/v1",  # Or your API endpoint
            api_key="x",  # Your API key
            num_max_requests_at_once=50,
            num_requests_for_eval=64,
            timeout=300,
        )
    ]

    return env_config, server_configs


# Example usage
if __name__ == "__main__":
    print("Checkers Environment Configuration Example")
    print("=" * 50)

    env_config, server_configs = get_checkers_config()

    print("Environment Configuration:")
    print(f"  Opponent: {env_config.opponent_type.value}")
    print(f"  AI plays as: {env_config.ai_plays_as.name}")
    print(f"  Thinking enabled: {env_config.thinking_enabled}")
    print(f"  Max episode turns: {env_config.max_episode_turns}")
    print(f"  Evaluation episodes: {env_config.eval_episodes}")
    print(f"  Temperature: {env_config.temperature}")

    print("\nServer Configuration:")
    for i, config in enumerate(server_configs):
        print(f"  Server {i+1}: {config.model_name}")
        print(f"    Base URL: {config.base_url}")
        print(f"    Max requests: {config.num_max_requests_at_once}")

    print("\nTo use this configuration:")
    print("1. Update the API key and model settings")
    print("2. Create a training script that uses these configs")
    print("3. Run: python train_checkers.py")
