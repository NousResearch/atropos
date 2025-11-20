import asyncio
import logging
import os

from dotenv import load_dotenv

from atroposlib.envs.base import APIServerConfig
from environments.cline_env.cline_agent_env import ClineAgentEnv, ClineAgentEnvConfig

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main() -> None:
    logger.info("Starting ClineAgentEnv Rust + Cline worker smoke test")

    env_config = ClineAgentEnvConfig(
        tokenizer_name=os.getenv("TOKENIZER_NAME", "gpt2"),
        group_size=1,
        use_wandb=False,
        rollout_server_url="http://localhost:8000",
        max_token_length=int(os.getenv("MAX_TOKENS", "4096")),
        wandb_name="cline_agent_env_rust_smoke",
        steps_per_eval=0,
        max_episode_turns=1,
        eval_episodes=1,
        scoring_function="dataset_target",
        allowed_languages=["Rust"],
        use_cline_worker=True,
    )

    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
    anthropic_base_url = os.getenv(
        "ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"
    )

    if not anthropic_api_key:
        logger.error("ANTHROPIC_API_KEY is not set; cannot run smoke test.")
        return

    server_configs = [
        APIServerConfig(
            model_name=anthropic_model,
            base_url=anthropic_base_url,
            api_key=anthropic_api_key,
            num_requests_for_eval=0,
        )
    ]

    logger.info("Initializing ClineAgentEnv with Rust-only filtering and Cline worker enabled")
    env = ClineAgentEnv(
        config=env_config,
        server_configs=server_configs,
        slurm=False,
        testing=True,
    )

    await env.setup()
    item = await env.get_next_item()
    logger.info(
        "Sampled dataset item: id=%s language=%s",
        item.get("instance_id"),
        item.get("language"),
    )

    scored, _ = await env.collect_trajectory(item)
    if scored is None:
        logger.error("Smoke test failed: collect_trajectory returned None")
        return

    logger.info("Smoke trajectory score: %s", scored.get("scores"))

    overrides = scored.get("overrides") or {}
    cline_meta = overrides.get("cline_metadata")
    if cline_meta:
        logger.info("Cline metadata: %s", cline_meta)
    else:
        logger.warning("No Cline metadata found in overrides; check worker integration.")


if __name__ == "__main__":
    asyncio.run(main())

