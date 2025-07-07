#!/usr/bin/env python3
"""
Test script to verify Diplomacy environment works with sglang server.
Make sure to run the sglang server first with: ./run_server.sh
"""

import asyncio
import logging
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from atroposlib.envs.base import APIServerConfig
from environments.diplomacy_environment.diplomacy_env_no_thinking import (
    DiplomacyEnvNoThinking,
    DiplomacyEnvNoThinkingConfig,
    PowerConfig,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_sglang_setup():
    """Test Diplomacy environment with sglang server."""
    logger.info("Testing Diplomacy environment with sglang server...")
    logger.info("Make sure sglang server is running on http://localhost:8000")
    
    # Config pointing to sglang server
    env_config = DiplomacyEnvNoThinkingConfig(
        tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
        max_token_length=2048,
        group_size=1,
        use_wandb=False,
        max_num_workers=1,
        rollout_server_url="http://localhost:8000",
        total_steps=1,
        batch_size=1,
        steps_per_eval=0,
        launch_web_server=False,
        # Use the same model for all powers
        powers_config={
            "FRANCE": PowerConfig(
                type="llm", 
                model="NousResearch/DeepHermes-3-Llama-3-8B-Preview", 
                is_training=True
            ),
            "ENGLAND": PowerConfig(
                type="llm", 
                model="NousResearch/DeepHermes-3-Llama-3-8B-Preview", 
                is_training=False
            ),
            "GERMANY": PowerConfig(
                type="llm", 
                model="NousResearch/DeepHermes-3-Llama-3-8B-Preview", 
                is_training=False
            ),
            "ITALY": PowerConfig(
                type="llm", 
                model="NousResearch/DeepHermes-3-Llama-3-8B-Preview", 
                is_training=False
            ),
            "AUSTRIA": PowerConfig(
                type="llm", 
                model="NousResearch/DeepHermes-3-Llama-3-8B-Preview", 
                is_training=False
            ),
            "RUSSIA": PowerConfig(
                type="llm", 
                model="NousResearch/DeepHermes-3-Llama-3-8B-Preview", 
                is_training=False
            ),
            "TURKEY": PowerConfig(
                type="llm", 
                model="NousResearch/DeepHermes-3-Llama-3-8B-Preview", 
                is_training=False
            ),
        },
        save_game_logs=True,
        game_log_dir="./test_sglang_logs",
        max_game_years=1,  # Just test 1 year
    )

    # Server config for sglang
    server_configs = [
        APIServerConfig(
            model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            base_url="http://localhost:8000/v1",  # sglang uses OpenAI-compatible API
            api_key="dummy",  # sglang doesn't need a real API key
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
        logger.info("✅ Environment initialized")

        # Test setup
        logger.info("Running setup...")
        await env.setup()
        logger.info("✅ Setup completed")

        # Test get_next_item
        logger.info("Getting next item...")
        item = await env.get_next_item()
        logger.info(f"✅ Got item: {item}")

        # Test a simple API call to verify sglang connection
        logger.info("Testing sglang server connection...")
        test_messages = [
            {"role": "system", "content": "You are playing Diplomacy as France."},
            {"role": "user", "content": "What are your initial thoughts about the game?"}
        ]
        
        response = await env.server.chat_completion(
            messages=test_messages,
            model="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            max_tokens=100,
            temperature=0.7,
        )
        
        logger.info(f"✅ Got response from sglang: {response.choices[0].message.content[:100]}...")

        # Run a quick game test
        logger.info("\nRunning a quick game test...")
        trajectories, backlog = await env.collect_trajectories(item)
        
        if trajectories:
            logger.info(f"✅ Collected {len(trajectories)} trajectories")
            logger.info(f"First trajectory: {trajectories[0].get('task_type', 'unknown')}")
        else:
            logger.info("⚠️  No trajectories collected (this might be normal for a quick test)")

        logger.info("\n✅ All tests passed! Diplomacy environment works with sglang.")

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(test_sglang_setup())