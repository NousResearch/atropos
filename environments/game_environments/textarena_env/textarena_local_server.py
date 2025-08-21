#!/usr/bin/env python3
"""
Local test server for the minimal TextArena environment.

This script runs TextArena games with OpenRouter models to test the integration.
"""

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

from atroposlib.envs.base import APIServerConfig, EvalHandlingEnum
from environments.game_environments.textarena_env.textarena_env_minimal import (
    TextArenaEnvMinimal,
    TextArenaEnvMinimalConfig,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Run TextArena games for testing the minimal environment."""
    logger.info("Starting TextArena minimal environment local test runner")

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    model_name = "qwen/qwen3-235b-a22b-2507"

    # CLI: [game_filter] [specific_game] [num_episodes]
    game_filter = sys.argv[1] if len(sys.argv) > 1 else "all"

    specific_game = None
    if len(sys.argv) > 2 and sys.argv[2].strip():
        specific_game = sys.argv[2].strip()

    env_config = TextArenaEnvMinimalConfig(
        tokenizer_name="NousResearch/Hermes-4-Qwen3-14B-1-e3",
        group_size=2,
        use_wandb=False,
        wandb_name="textarena_minimal_local_test",
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
        max_steps=5,
        include_messages=True,
        game_filter=game_filter,
        specific_game=specific_game,
        opponent_temperature=0.7,
        training_player_index=0,
    )

    server_configs = [
        APIServerConfig(
            model_name=model_name,
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
            num_requests_for_eval=0,
        ),
        APIServerConfig(
            model_name=model_name,
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
            num_requests_for_eval=0,
        ),
        APIServerConfig(
            model_name=model_name,
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
            num_requests_for_eval=0,
        ),
        APIServerConfig(
            model_name=model_name,
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
            num_requests_for_eval=0,
        ),
    ]

    logger.warning(f"Testing TextArena with game filter: {game_filter}")
    if specific_game:
        logger.warning(f"Testing specific game: {specific_game}")
    logger.warning(f"Using OpenRouter {model_name} for all agents")

    try:
        env = TextArenaEnvMinimal(
            config=env_config,
            server_configs=server_configs,
            slurm=False,
            testing=False,
        )
    except Exception as e:
        logger.exception(f"Failed to initialize TextArenaEnvMinimal: {e}")
        return

    logger.warning("Running test games")
    try:
        await env.setup()

        num_episodes = 3
        if len(sys.argv) > 3:
            try:
                num_episodes = int(sys.argv[3])
            except ValueError:
                pass

        # Track statistics
        episode_results = []
        games_tested = set()

        for episode_num in range(num_episodes):
            logger.info(f"\n{'='*60}")
            logger.info(f"Episode {episode_num + 1}/{num_episodes}")
            logger.info(f"{'='*60}")

            item = await env.get_next_item()
            env_id = item["env_id"]
            num_players = item["num_players"]
            game_type = item["game_type"]

            games_tested.add(env_id)

            logger.info(f"Game: {env_id}")
            logger.info(f"Type: {game_type} ({num_players} players)")
            logger.info(f"Training player index: {env.config.training_player_index}")

            scored_data_groups, _ = await env.collect_trajectories(item)
            
            if scored_data_groups:
                # Aggregate stats across all player groups
                all_scores = []
                total_tokens = 0
                
                for player_idx, sdg in enumerate(scored_data_groups):
                    if sdg and sdg["scores"]:
                        all_scores.extend(sdg["scores"])
                        total_tokens += sum(len(t) for t in sdg["tokens"]) if sdg["tokens"] else 0
                        logger.info(
                            f"Player {player_idx} SDG: "
                            f"items={len(sdg['scores'])}, "
                            f"scores={sdg['scores']}"
                        )
                
                if all_scores:
                    avg_score = sum(all_scores) / len(all_scores)
                    logger.info(f"\nResults for {env_id}:")
                    logger.info(f"  Player groups collected: {len(scored_data_groups)}")
                    logger.info(f"  Total trajectories: {len(all_scores)}")
                    logger.info(f"  Average score: {avg_score:.2f}")
                    logger.info(f"  Score range: [{min(all_scores):.2f}, {max(all_scores):.2f}]")
                    logger.info(f"  Total tokens: {total_tokens}")

                    if env.episode_outcomes_buffer:
                        recent_outcomes = env.episode_outcomes_buffer[
                            -env.config.group_size :
                        ]
                        wins = sum(1 for o in recent_outcomes if o > 0)
                        losses = sum(1 for o in recent_outcomes if o < 0)
                        draws = sum(1 for o in recent_outcomes if o == 0)

                        logger.info(
                            f"  Training player outcomes: {wins} wins, {losses} losses, {draws} draws"
                        )

                    episode_results.append(
                        {
                            "episode": episode_num + 1,
                            "game": env_id,
                            "game_type": game_type,
                            "num_players": num_players,
                            "avg_score": avg_score,
                            "num_trajectories": len(all_scores),
                        }
                    )
                else:
                    logger.warning(f"No scores collected for {env_id}")
                    episode_results.append(
                        {
                            "episode": episode_num + 1,
                            "game": env_id,
                            "game_type": game_type,
                            "num_players": num_players,
                            "avg_score": 0.0,
                            "num_trajectories": 0,
                        }
                    )
            else:
                logger.error(f"Failed to collect trajectories for {env_id}")
                episode_results.append(
                    {
                        "episode": episode_num + 1,
                        "game": env_id,
                        "game_type": game_type,
                        "num_players": num_players,
                        "avg_score": 0.0,
                        "num_trajectories": 0,
                    }
                )

        logger.info("\n" + "=" * 60)
        logger.info("OVERALL TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total episodes: {num_episodes}")
        logger.info(f"Group size: {env.config.group_size} games per episode")
        logger.info(f"Unique games tested: {len(games_tested)}")
        logger.info(f"Games: {', '.join(sorted(games_tested))}")

        game_type_stats = {}
        for result in episode_results:
            game_type = result["game_type"]
            if game_type not in game_type_stats:
                game_type_stats[game_type] = {
                    "episodes": 0,
                    "total_score": 0.0,
                    "games": set(),
                }
            game_type_stats[game_type]["episodes"] += 1
            game_type_stats[game_type]["total_score"] += result["avg_score"]
            game_type_stats[game_type]["games"].add(result["game"])

        logger.info("\nStatistics by game type:")
        for game_type, stats in game_type_stats.items():
            avg_score = (
                stats["total_score"] / stats["episodes"] if stats["episodes"] > 0 else 0
            )
            logger.info(
                f"  {game_type}: {stats['episodes']} episodes, "
                f"avg score {avg_score:.2f}, "
                f"{len(stats['games'])} unique games"
            )

        if episode_results:
            overall_avg_score = sum(r["avg_score"] for r in episode_results) / len(
                episode_results
            )
            logger.info(f"\nOverall average score: {overall_avg_score:.2f}")

            successful_episodes = sum(
                1 for r in episode_results if r["num_trajectories"] > 0
            )
            logger.info(f"Successful episodes: {successful_episodes}/{num_episodes}")

        logger.info("\nTest completed successfully!")

    except Exception as e:
        logger.exception(f"An error occurred during test: {e}")


if __name__ == "__main__":
    asyncio.run(main())
