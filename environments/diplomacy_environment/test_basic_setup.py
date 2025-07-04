#!/usr/bin/env python3
"""
Basic test to verify Diplomacy environment setup works.
Tests the environment without running a full game.
"""

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from atroposlib.envs.base import APIServerConfig
from environments.diplomacy_environment.diplomacy_env_no_thinking import (
    DiplomacyEnvNoThinking,
    DiplomacyEnvNoThinkingConfig,
    PowerConfig,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_basic_setup():
    """Test basic environment setup without running a full game."""
    logger.info("Testing Diplomacy environment basic setup...")

    # Minimal config
    env_config = DiplomacyEnvNoThinkingConfig(
        tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
        max_token_length=1024,
        group_size=1,
        use_wandb=False,
        max_num_workers=1,
        rollout_server_url="http://localhost:8000",
        total_steps=1,
        batch_size=1,
        steps_per_eval=0,
        # Disable web server for quick test
        launch_web_server=False,
        # Simple power config
        powers_config={
            "FRANCE": PowerConfig(type="llm", model="gpt-4o-mini", is_training=True),
            "ENGLAND": PowerConfig(type="llm", model="gpt-4o-mini", is_training=False),
            "GERMANY": PowerConfig(type="llm", model="gpt-4o-mini", is_training=False),
            "ITALY": PowerConfig(type="llm", model="gpt-4o-mini", is_training=False),
            "AUSTRIA": PowerConfig(type="llm", model="gpt-4o-mini", is_training=False),
            "RUSSIA": PowerConfig(type="llm", model="gpt-4o-mini", is_training=False),
            "TURKEY": PowerConfig(type="llm", model="gpt-4o-mini", is_training=False),
        },
        # Disable game logs for test
        save_game_logs=False,
    )

    server_configs = [
        APIServerConfig(
            model_name="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
            num_requests_for_eval=0,
        )
    ]

    try:
        # Initialize environment
        logger.info("Initializing environment...")
        env = DiplomacyEnvNoThinking(
            config=env_config,
            server_configs=server_configs,
            slurm=False,
            testing=False,
        )
        logger.info("✅ Environment initialized successfully")

        # Test setup
        logger.info("Running setup...")
        await env.setup()
        logger.info("✅ Setup completed successfully")

        # Test get_next_item
        logger.info("Testing get_next_item...")
        item = await env.get_next_item()
        logger.info(f"✅ Got item: {item}")

        # Test training powers
        logger.info("Testing _get_training_powers...")
        training_powers = env._get_training_powers()
        logger.info(f"✅ Training powers: {training_powers}")

        # Test model configs
        logger.info("Testing _create_agent_configs...")
        models = env._create_agent_configs()
        logger.info(f"✅ Model configs: {models}")

        logger.info("\n🎉 All basic tests passed!")

    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
    finally:
        # Clean up if web server was started
        if hasattr(env, "web_server_process") and env.web_server_process:
            env.web_server_process.terminate()
            env.web_server_process.wait()


if __name__ == "__main__":
    print("\n🧪 DIPLOMACY ENVIRONMENT BASIC TEST 🧪\n")
    print("This will test the environment setup without running a full game.\n")

    asyncio.run(test_basic_setup())
