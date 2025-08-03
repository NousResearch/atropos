#!/usr/bin/env python3
"""
Local test server for the minimalist TextWorld environment.
"""

import asyncio
import logging
import os

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
    """Run multiple TextWorld episodes for testing the minimalist environment."""
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
        max_token_length=32768,
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
            model_name="gpt-4.1",
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

    logger.info("Running 20 episodes across all challenges")
    try:
        await env.setup()

        # Test each challenge type
        import sys

        # Check if specific challenge requested
        challenge_to_test = sys.argv[1] if len(sys.argv) > 1 else None
        num_episodes = 20 if not challenge_to_test else 1

        # Track statistics
        episode_results = []
        challenge_counts = {
            "tw-simple": 0,
            "tw-cooking": 0,
            "tw-coin_collector": 0,
            "tw-treasure_hunter": 0,
        }

        for episode_num in range(num_episodes):
            if challenge_to_test:
                # Override config to test specific challenge
                env.config.challenge_names = [challenge_to_test]

            item = await env.get_next_item()
            challenge_name = item["challenge_name"]
            challenge_counts[challenge_name] += 1
            logger.info(f"\n===== Episode {episode_num + 1}/{num_episodes} =====")
            logger.info(f"Using game: {item}")

            # Collect trajectories (group_size=1 so just one trajectory)
            sdg, _ = await env.collect_trajectories(item)

            scored_data_item = None
            if (
                sdg
                and hasattr(sdg, "scored_data_items")
                and sdg.scored_data_items
                and len(sdg.scored_data_items) > 0
            ):
                scored_data_item = sdg.scored_data_items[0]
            elif (
                sdg
                and isinstance(sdg, dict)
                and "scored_data_items" in sdg
                and len(sdg["scored_data_items"]) > 0
            ):
                scored_data_item = sdg["scored_data_items"][0]

            if scored_data_item:
                # Handle both object and dict access patterns
                scores = (
                    scored_data_item.scores
                    if hasattr(scored_data_item, "scores")
                    else scored_data_item.get("scores")
                )
                metadata = (
                    scored_data_item.metadata
                    if hasattr(scored_data_item, "metadata")
                    else scored_data_item.get("metadata", {})
                )

                # Log brief summary
                outcome_str = "Loss"
                if metadata.get("won"):
                    outcome_str = "Win"
                elif scores > 0:
                    outcome_str = "Partial Success"

                moves = metadata.get("moves", 0)
                logger.info(
                    f"Result: {outcome_str}, Score: {scores:.2f}, Moves: {moves}"
                )

                # Collect statistics
                episode_results.append(
                    {
                        "episode": episode_num + 1,
                        "challenge": challenge_name,
                        "score": scores,
                        "won": metadata.get("won", False),
                        "moves": moves,
                        "difficulty": item.get("settings", {}),
                    }
                )
            else:
                logger.error("Trajectory collection did not return a ScoredDataItem.")
                episode_results.append(
                    {
                        "episode": episode_num + 1,
                        "challenge": challenge_name,
                        "score": 0.0,
                        "won": False,
                        "moves": 0,
                        "difficulty": item.get("settings", {}),
                    }
                )

        # Print overall statistics
        logger.info("\n" + "=" * 60)
        logger.info("OVERALL RESULTS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total episodes: {num_episodes}")
        logger.info(f"Challenge distribution: {challenge_counts}")

        # Calculate win rates per challenge
        for challenge in challenge_counts:
            challenge_episodes = [
                ep for ep in episode_results if ep["challenge"] == challenge
            ]
            if challenge_episodes:
                wins = sum(1 for ep in challenge_episodes if ep["won"])
                avg_score = sum(ep["score"] for ep in challenge_episodes) / len(
                    challenge_episodes
                )
                avg_moves = sum(ep["moves"] for ep in challenge_episodes) / len(
                    challenge_episodes
                )
                logger.info(f"\n{challenge}:")
                logger.info(f"  Episodes: {len(challenge_episodes)}")
                logger.info(
                    f"  Win rate: {wins}/{len(challenge_episodes)} ({100*wins/len(challenge_episodes):.1f}%)"
                )
                logger.info(f"  Avg score: {avg_score:.2f}")
                logger.info(f"  Avg moves: {avg_moves:.1f}")

        # Overall stats
        total_wins = sum(1 for ep in episode_results if ep["won"])
        total_avg_score = (
            sum(ep["score"] for ep in episode_results) / len(episode_results)
            if episode_results
            else 0
        )
        logger.info(
            f"\nOverall win rate: {total_wins}/{len(episode_results)} ({100*total_wins/len(episode_results):.1f}%)"
        )
        logger.info(f"Overall avg score: {total_avg_score:.2f}")

    except Exception as e:
        logger.exception(
            f"An error occurred during trajectory collection or summary: {e}"
        )


if __name__ == "__main__":
    asyncio.run(main())
