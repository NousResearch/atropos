#!/usr/bin/env python3
"""
Local test server for the minimal Diplomacy environment.

This script runs the full AI_Diplomacy game with real OpenAI models
to test the AtroposClient proxy integration.
"""

import asyncio
import logging
import os

from dotenv import load_dotenv

from atroposlib.envs.base import APIServerConfig, EvalHandlingEnum
from environments.game_environments.diplomacy_environment.diplomacy_env_minimal import (
    DiplomacyEnvMinimal,
    DiplomacyEnvMinimalConfig,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Run Diplomacy games for testing the minimal environment."""
    logger.info("Starting Diplomacy minimal environment local test runner")

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error(
            "OPENAI_API_KEY not found. Please set it in your environment or .env file"
        )
        return

    # Configure environment - using real OpenAI model
    env_config = DiplomacyEnvMinimalConfig(
        tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
        group_size=2,  # Run 2 parallel games
        use_wandb=False,
        wandb_name="diplomacy_minimal_local_test",
        max_num_workers=1,
        rollout_server_url="http://localhost:8000",
        total_steps=1,
        batch_size=1,
        steps_per_eval=0,
        max_token_length=4096,
        inference_weight=1.0,
        data_path_to_save_groups=None,
        eval_handling=EvalHandlingEnum.NONE,
        eval_limit_ratio=0.0,
        max_game_turns=5,  # Short games for testing
        training_power="FRANCE",  # Which power we're training
        include_messages=True,  # Include messages for debugging
        eval_episodes=0,
        start_diplomacy_server=True,  # Let the env start the server
        save_game_logs=True,
        game_logs_dir="./test_game_logs",
    )

    # Configure server - using 4 servers to match SLURM setup
    # For local testing, we'll simulate this with the same OpenAI endpoint
    server_configs = [
        APIServerConfig(
            model_name="gpt-4.1",  # Using the OpenAI model directly
            base_url="https://api.openai.com/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
            num_requests_for_eval=0,
        ),
        APIServerConfig(
            model_name="gpt-4.1",
            base_url="https://api.openai.com/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
            num_requests_for_eval=0,
        ),
        APIServerConfig(
            model_name="gpt-4.1",
            base_url="https://api.openai.com/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
            num_requests_for_eval=0,
        ),
        APIServerConfig(
            model_name="gpt-4.1",
            base_url="https://api.openai.com/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
            num_requests_for_eval=0,
        ),
    ]

    logger.info("Using OpenAI gpt-4.1 for Diplomacy test")
    logger.debug(f"Env Config: {env_config}")
    logger.debug(f"Server Configs: {server_configs}")

    try:
        env = DiplomacyEnvMinimal(
            config=env_config,
            server_configs=server_configs,
            slurm=False,
            testing=False,
        )
    except Exception as e:
        logger.exception(f"Failed to initialize DiplomacyEnvMinimal: {e}")
        return

    logger.info("Running test games")
    try:
        await env.setup()

        # Get number of episodes from command line or default
        import sys

        num_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 3

        # Track statistics
        episode_results = []

        for episode_num in range(num_episodes):
            logger.info(f"\n===== Episode {episode_num + 1}/{num_episodes} =====")

            item = await env.get_next_item()
            logger.info(f"Game ID: {item['game_id']}, Seed: {item['seed']}")

            # Collect trajectories (will run group_size parallel games)
            scored_data_group, _ = await env.collect_trajectories(item)

            if scored_data_group and scored_data_group["scores"]:
                avg_score = sum(scored_data_group["scores"]) / len(scored_data_group["scores"])
                logger.info(f"Collected {len(scored_data_group['scores'])} trajectories with average score: {avg_score:.2f}")

                # Get game outcomes from buffer
                if env.game_outcomes_buffer:
                    latest_outcomes = env.game_outcomes_buffer[-env.config.group_size :]
                    for i, outcome in enumerate(latest_outcomes):
                        logger.info(
                            f"  Game {i}: Score={outcome['score']:.2f}, "
                            f"Winner={outcome['winner']}, "
                            f"Turns={outcome['turns']}, "
                            f"Centers={outcome['final_centers'].get(env.config.training_power, 0)}"
                        )

                episode_results.append(
                    {
                        "episode": episode_num + 1,
                        "score": avg_score,
                        "outcomes": latest_outcomes if env.game_outcomes_buffer else [],
                    }
                )
            else:
                logger.error("Failed to collect trajectory")
                episode_results.append(
                    {
                        "episode": episode_num + 1,
                        "score": 0.0,
                        "outcomes": [],
                    }
                )

        # Print overall statistics
        logger.info("\n" + "=" * 60)
        logger.info("OVERALL RESULTS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total episodes: {num_episodes}")
        logger.info(f"Group size: {env.config.group_size} games per episode")
        logger.info(f"Training power: {env.config.training_power}")

        # Calculate statistics
        if episode_results:
            avg_score = sum(ep["score"] for ep in episode_results) / len(
                episode_results
            )
            logger.info(f"\nAverage trajectory score: {avg_score:.2f}")

            # Count wins
            total_games = 0
            wins = 0
            for ep in episode_results:
                for outcome in ep["outcomes"]:
                    total_games += 1
                    if outcome["winner"] == env.config.training_power:
                        wins += 1

            if total_games > 0:
                logger.info(
                    f"Win rate: {wins}/{total_games} ({100*wins/total_games:.1f}%)"
                )

                # Average supply centers
                total_centers = sum(
                    outcome["final_centers"].get(env.config.training_power, 0)
                    for ep in episode_results
                    for outcome in ep["outcomes"]
                )
                avg_centers = total_centers / total_games
                logger.info(f"Average final supply centers: {avg_centers:.1f}")

        logger.info("\nTest completed successfully!")

    except Exception as e:
        logger.exception(f"An error occurred during test: {e}")


if __name__ == "__main__":
    asyncio.run(main())
