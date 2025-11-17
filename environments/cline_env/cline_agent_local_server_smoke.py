import asyncio
import logging
import os
from typing import Optional

from dotenv import load_dotenv

from atroposlib.envs.base import APIServerConfig, EvalHandlingEnum, ScoredDataItem
from environments.cline_env.cline_agent_env import (
    ClineAgentEnv,
    ClineAgentEnvConfig,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    logger.info("Starting ClineAgentEnv local debug runner")

    env_config = ClineAgentEnvConfig(
        tokenizer_name="gpt2",
        group_size=1,
        use_wandb=False,
        wandb_name="cline_agent_env_local_debug",
        max_num_workers=1,
        rollout_server_url="http://localhost:8000",
        total_steps=1,
        batch_size=1,
        steps_per_eval=0,
        max_token_length=int(os.getenv("MAX_TOKENS", "4096")),
        inference_weight=1.0,
        data_path_to_save_groups=None,
        eval_handling=EvalHandlingEnum.NONE,
        eval_limit_ratio=0.0,
        env_name="cline_agent_env",
        max_episode_turns=1,
        eval_episodes=0,
        scoring_function="dataset_target",
    )

    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4.1")
    anthropic_base_url = os.getenv(
        "ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"
    )

    if not anthropic_api_key:
        logger.error("ANTHROPIC_API_KEY is not set in the environment.")
        return

    server_configs = [
        APIServerConfig(
            model_name=anthropic_model,
            base_url=anthropic_base_url,
            api_key=anthropic_api_key,
            num_requests_for_eval=0,
        )
    ]

    logger.info("Using local debug configuration for ClineAgentEnv.")
    logger.debug("Env Config: %s", env_config)
    logger.debug("Server Configs: %s", server_configs)

    try:
        env = ClineAgentEnv(
            config=env_config,
            server_configs=server_configs,
            slurm=False,
            testing=False,
        )
    except Exception as exc:
        logger.exception("Failed to initialize ClineAgentEnv: %s", exc)
        return

    logger.info("Running a single trajectory directly using collect_trajectory")
    try:
        await env.setup()
        item_for_env = await env.get_next_item()
        logger.info(
            "Sampled item: instance_id=%s, model_name=%s",
            item_for_env.get("instance_id"),
            item_for_env.get("model_name"),
        )

        result_tuple = await env.collect_trajectory(item_for_env)

        scored_data_item: Optional[ScoredDataItem] = None
        if result_tuple and result_tuple[0]:
            scored_data_item = result_tuple[0]
            logger.info(
                "Trajectory collection complete. Score: %s",
                scored_data_item.get("scores"),
            )
            if env_config.include_messages and scored_data_item.get("messages"):
                logger.info("Collected Messages:")
                for i, msg in enumerate(scored_data_item["messages"]):
                    logger.info(
                        "  %d. Role: %s, Content: '%s...'",
                        i,
                        msg["role"],
                        str(msg["content"])[:150],
                    )
            logger.info(
                "Tokens (%d): %s...",
                len(scored_data_item.get("tokens", [])),
                str(scored_data_item.get("tokens"))[:100],
            )
            logger.info(
                "Masks (%d): %s...",
                len(scored_data_item.get("masks", [])),
                str(scored_data_item.get("masks"))[:100],
            )
        else:
            logger.error("Trajectory collection did not return a ScoredDataItem.")

    except Exception as exc:
        logger.exception(
            "An error occurred during trajectory collection: %s", exc
        )


if __name__ == "__main__":
    asyncio.run(main())
