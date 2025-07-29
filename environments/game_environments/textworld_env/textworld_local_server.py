#!/usr/bin/env python3
"""
Local test server for the minimalist TextWorld environment.
"""

import asyncio
import logging
import os
import random

from dotenv import load_dotenv

from atroposlib.envs.base import APIServerConfig, EvalHandlingEnum
from environments.game_environments.textworld_env.textworld_env import (
    TextWorldEnv,
    TextWorldEnvConfig,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Run a complete TextWorld episode for testing the minimalist environment."""
    logger.info("Starting TextWorld (No Thinking) environment local debug runner")

    # Configure environment - matching blackjack_no_thinking settings
    env_config = TextWorldEnvConfig(
        tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
        group_size=1,
        use_wandb=False,
        wandb_name="textworld_no_thinking_local_debug",
        max_num_workers=1,
        rollout_server_url="http://localhost:8000",
        total_steps=1,
        batch_size=1,
        steps_per_eval=0,
        max_token_length=1024,
        inference_weight=1.0,
        data_path_to_save_groups=None,
        eval_handling=EvalHandlingEnum.NONE,
        eval_limit_ratio=0.0,
        max_steps=10,  # Max steps per episode
        include_messages=True,  # Include messages for debugging
        eval_episodes=0,
    )

    # Configure server - using same model as blackjack example
    server_configs = [
        APIServerConfig(
            model_name="gpt-4.1-nano",
            base_url="https://api.openai.com/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
            num_requests_for_eval=0,
        )
    ]

    logger.info("Using hardcoded debug configuration for No Thinking TextWorld.")
    logger.debug(f"Env Config: {env_config}")
    logger.debug(f"Server Configs: {server_configs}")

    try:
        env = TextWorldEnv(
            config=env_config,
            server_configs=server_configs,
            slurm=False,
            testing=False,
        )
    except Exception as e:
        logger.exception(f"Failed to initialize TextWorldEnv: {e}")
        return

    logger.info("Running a single trajectory directly using collect_trajectories")
    try:
        await env.setup()
        
        # Get a random game
        item = await env.get_next_item()
        logger.info(f"Using game: {item}")

        # Collect trajectories (group_size=1 so just one trajectory)
        sdg, _ = await env.collect_trajectories(item)

        if sdg and sdg.scored_data_items and len(sdg.scored_data_items) > 0:
            scored_data_item = sdg.scored_data_items[0]
            logger.info(
                f"Trajectory collection complete. Score: {scored_data_item.scores}"
            )
            if env_config.include_messages and scored_data_item.messages:
                logger.info("Collected Messages:")
                for i, msg in enumerate(scored_data_item.messages):
                    logger.info(
                        f"  {i}. Role: {msg['role']}, Content: '{str(msg['content'])[:150]}...'"
                    )
            logger.info(
                f"Tokens ({len(scored_data_item.tokens)}): {str(scored_data_item.tokens)[:100]}..."
            )
            logger.info(
                f"Masks ({len(scored_data_item.masks)}): {str(scored_data_item.masks)[:100]}..."
            )

            # Episode summary
            if scored_data_item.metadata:
                logger.info("\n========== Episode Summary ==========")
                logger.info(f"Game: {item.get('challenge_name', 'unknown')}")
                logger.info(
                    f"Final Environment reward (Score): {scored_data_item.scores:.2f}"
                )
                outcome_str = "Loss"
                if scored_data_item.metadata.get("won"):
                    outcome_str = "Win"
                elif scored_data_item.scores > 0:
                    outcome_str = "Partial Success"
                logger.info(f"Game Outcome: {outcome_str}")
                logger.info(f"Moves: {scored_data_item.metadata.get('moves', 0)}")
                logger.info("=======================================")
        else:
            logger.error("Trajectory collection did not return a ScoredDataItem.")

    except Exception as e:
        logger.exception(
            f"An error occurred during trajectory collection or summary: {e}"
        )


if __name__ == "__main__":
    asyncio.run(main())
