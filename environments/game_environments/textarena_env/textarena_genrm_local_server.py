#!/usr/bin/env python3
"""
Local test runner for TextArena GenRM environment.

Uses Hermes-4-405B (Nous inference API) for both policy generation and judging.

CLI usage:
  python textarena_genrm_local_server.py [game_filter] [specific_game] [num_episodes]
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from atroposlib.envs.base import APIServerConfig, EvalHandlingEnum
from environments.game_environments.textarena_env.textarena_env_genrm import (
    TextArenaEnvGenRM,
    TextArenaEnvGenRMConfig,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    logger.info("Starting TextArena GenRM local runner")

    hermes_api_key = os.getenv("HERMES_API_KEY") or os.getenv("NOUS_API_KEY")
    if not hermes_api_key:
        logger.error("HERMES_API_KEY (or NOUS_API_KEY) must be set in .env for Hermes-4-405B")
        sys.exit(1)

    # Model used for policy generation and judge
    model_name = "Hermes-4-405B"

    # CLI args
    game_filter = sys.argv[1] if len(sys.argv) > 1 else "all"
    specific_game = sys.argv[2].strip() if len(sys.argv) > 2 and sys.argv[2].strip() else None
    try:
        num_episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    except ValueError:
        num_episodes = 2

    output_path = os.getenv("TEXTARENA_GENRM_OUTPUT", "data/textarena_genrm_local.jsonl")
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

    env_config = TextArenaEnvGenRMConfig(
        tokenizer_name=os.getenv("TOKENIZER_NAME", "NousResearch/Hermes-4-Qwen3-14B-1-e3"),
        group_size=int(os.getenv("GENRM_GROUP_SIZE", "4")),
        use_wandb=False,
        wandb_name="textarena_genrm_local",
        max_num_workers=1,
        rollout_server_url="http://localhost:8000",
        total_steps=1,
        batch_size=1,
        steps_per_eval=0,
        max_token_length=int(os.getenv("GENRM_MAX_TOKENS", "8192")),
        inference_weight=1.0,
        data_path_to_save_groups=None,
        eval_handling=EvalHandlingEnum.NONE,
        eval_limit_ratio=0.0,
        max_steps=int(os.getenv("GENRM_MAX_STEPS", "10")),
        include_messages=True,
        game_filter=game_filter,
        specific_game=specific_game,
        judge_model_name=os.getenv("JUDGE_MODEL_NAME", "Hermes-4-405B"),
        judge_group_size=int(os.getenv("JUDGE_GROUP_SIZE", "3")),
        trajectories_output_path=output_path,
    )

    server_configs = [
        APIServerConfig(
            model_name=model_name,
            base_url="https://inference-api.nousresearch.com/v1",
            api_key=hermes_api_key,
            num_requests_for_eval=0,
        )
    ]

    logger.warning(f"Testing TextArena GenRM with filter: {game_filter}")
    if specific_game:
        logger.warning(f"Specific game: {specific_game}")
    logger.warning(f"Policy and judge model: {model_name}")

    env = TextArenaEnvGenRM(
        config=env_config,
        server_configs=server_configs,
        slurm=False,
        testing=False,
    )

    await env.setup()

    episode_summaries = []
    games_tested = set()
    for ep in range(num_episodes):
        logger.info(f"\n{'='*60}\nEpisode {ep+1}/{num_episodes}\n{'='*60}")
        item = await env.get_next_item()
        env_id = item["env_id"]
        num_players = item["num_players"]
        game_type = item.get("game_type", "unknown")
        games_tested.add(env_id)
        logger.info(f"Game: {env_id} | Type: {game_type} | Players: {num_players}")

        sdgs, _ = await env.collect_trajectories(item)
        total_alternatives = sum(len(g["scores"]) for g in sdgs) if sdgs else 0
        avg_score = (
            sum(sum(g["scores"]) for g in sdgs) / total_alternatives if total_alternatives > 0 else 0.0
        )
        logger.info(
            f"Collected {len(sdgs)} step-groups across all players; total alternatives {total_alternatives}; avg score {avg_score:.3f}"
        )
        episode_summaries.append(
            {
                "episode": ep + 1,
                "game": env_id,
                "game_type": game_type,
                "num_players": num_players,
                "num_step_groups": len(sdgs),
                "avg_score": avg_score,
            }
        )

    logger.info("\n" + "=" * 60)
    logger.info("TextArena GenRM local run summary")
    logger.info("=" * 60)
    logger.info(f"Episodes: {len(episode_summaries)} | Unique games: {len(games_tested)}")
    if episode_summaries:
        mean_score = sum(s["avg_score"] for s in episode_summaries) / len(episode_summaries)
        logger.info(f"Mean avg_score across episodes: {mean_score:.3f}")
    logger.info(f"Trajectories saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
