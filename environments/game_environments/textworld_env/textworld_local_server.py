#!/usr/bin/env python3
"""
Local test server for the minimalist TextWorld environment.
"""

import asyncio
import logging
import os

from dotenv import load_dotenv

from atroposlib.envs.base import APIServerConfig
from environments.game_environments.textworld_env.textworld_env import (
    TextWorldEnv,
    TextWorldEnvConfig,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    """Run a complete TextWorld episode for testing the minimalist environment."""
    logger.info("Starting Minimalist TextWorld Environment Test")

    # Configure environment
    env_config = TextWorldEnvConfig(
        tokenizer_name="NousResearch/Hermes-3-Llama-3.1-8B",
        group_size=4,  # Small group for testing
        use_wandb=False,
        wandb_name="textworld-minimal-test",
        max_num_workers=1,
        total_steps=1,
        batch_size=1,
        max_token_length=8192,
        max_steps=15,  # Max steps per episode
        include_messages=True,  # Include messages for debugging
        # Use registry for variety
        use_registry=True,
        registry_mode="challenge",
        registry_generation_ratio=0.0,  # Only challenges for testing
        registry_difficulty="easy",  # Start with easy games
    )

    # Configure server - adjust to your local setup
    server_configs = [
        APIServerConfig(
            model_name="gpt-4-mini",  # Or your local model
            base_url="https://api.openai.com/v1",  # Or your local server
            api_key=os.getenv("OPENAI_API_KEY", "x"),
            num_requests_for_eval=0,
            timeout=120,
        )
    ]

    logger.info("Configuration:")
    logger.info(f"  Group size: {env_config.group_size}")
    logger.info(f"  Max steps: {env_config.max_steps}")
    logger.info(f"  Max tokens: {env_config.max_token_length}")
    logger.info(f"  Registry mode: {env_config.registry_mode}")

    try:
        # Initialize environment
        env = TextWorldEnv(
            config=env_config,
            server_configs=server_configs,
            slurm=False,
            testing=False,
        )

        await env.setup()
        logger.info("Environment setup complete")

        # Get game configuration
        item = await env.get_next_item()
        logger.info(f"Game configuration: {item}")

        # Collect trajectories
        logger.info(f"Collecting {env_config.group_size} parallel trajectories...")
        sdg, _ = await env.collect_trajectories(item)

        # Display results
        logger.info("\n" + "=" * 60)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 60)

        if sdg and sdg.scored_data_items:
            logger.info(
                f"Successfully collected {len(sdg.scored_data_items)} trajectories"
            )

            # Summary statistics
            scores = [item.scores for item in sdg.scored_data_items]
            won_count = sum(
                1
                for item in sdg.scored_data_items
                if item.metadata and item.metadata.get("won", False)
            )

            logger.info("\nScore Statistics:")
            logger.info(f"  Average score: {sum(scores) / len(scores):.2f}")
            logger.info(f"  Min score: {min(scores)}")
            logger.info(f"  Max score: {max(scores)}")
            logger.info(f"  Games won: {won_count}/{len(scores)}")

            # Show a sample trajectory
            logger.info("\n" + "-" * 60)
            logger.info("SAMPLE TRAJECTORY (First collected)")
            logger.info("-" * 60)

            sample = sdg.scored_data_items[0]
            if sample.messages:
                for i, msg in enumerate(sample.messages):
                    role = msg["role"]
                    content = msg["content"]

                    if role == "system":
                        logger.info(f"\n[SYSTEM]\n{content[:200]}...")
                    elif role == "user":
                        logger.info(f"\n[GAME STATE]\n{content}")
                    elif role == "assistant":
                        logger.info(f"\n[ACTION] {content}")

            logger.info(f"\n[FINAL SCORE] {sample.scores}")
            logger.info(f"[METADATA] {sample.metadata}")

        else:
            logger.error("No trajectories collected!")

    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)
    finally:
        await env.cleanup()
        logger.info("\nTest complete!")


if __name__ == "__main__":
    asyncio.run(main())
