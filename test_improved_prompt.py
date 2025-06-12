#!/usr/bin/env python3
"""Test TextWorld environment prompting"""

import asyncio
import logging
from dotenv import load_dotenv

from atroposlib.envs.base import APIServerConfig
from environments.game_environments.textworld.textworld_env import TextWorldEnv, TextWorldEnvConfig

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def test_textworld_prompt():
    """Test TextWorld with its built-in prompting."""
    logger.info("Testing TextWorld environment prompt...")

    server_config = APIServerConfig(
        model_name="NousResearch/DeepHermes-3-Mistral-24B-Preview",
        base_url="http://localhost:30000/v1",
        api_key="dummy",
        num_requests_for_eval=0,
        server_type="openai",
    )

    env_config = TextWorldEnvConfig(
        tokenizer_name="NousResearch/DeepHermes-3-Mistral-24B-Preview",
        max_token_length=8192,
        max_steps=5,
        use_registry=True,
        registry_mode="challenge",
        challenge_name="tw-simple",
        debug_mode=True,
        default_server_config=server_config,
        group_size=1,
    )

    try:
        env = TextWorldEnv(config=env_config, server_configs=[server_config], slurm=False)
        await env.setup()
        
        # Log the actual system prompt being used
        logger.info("ACTUAL SYSTEM PROMPT FROM ENVIRONMENT:")
        logger.info("=" * 80)
        logger.info(env.agent.config.system_prompt)
        logger.info("=" * 80)
        
        item = await env.get_next_item()
        if not item:
            logger.error("Failed to create episode")
            return
            
        episode_state = item["episode_state"]
        
        # Generate response
        responses = await env.agent.generate_action(episode_state.initial_formatted_obs, n=1)
        
        if responses:
            response_text = responses[0]["action_text"]
            logger.info("\nRESPONSE:")
            logger.info("=" * 80)
            logger.info(response_text)
            logger.info("=" * 80)
            
            # Also log the exact messages sent
            logger.info("\nMESSAGES SENT TO MODEL:")
            logger.info("=" * 80)
            # Get the last messages from agent's log
            if hasattr(env.agent, 'game_log') and env.agent.game_log.get('turn'):
                last_turn = env.agent.game_log['turn'][-1]
                if 'messages_sent' in last_turn:
                    for msg in last_turn['messages_sent']:
                        logger.info(f"Role: {msg['role']}")
                        logger.info(f"Content: {msg['content'][:200]}...")
                        logger.info("-" * 40)
            logger.info("=" * 80)
            
            # Test parsing
            action, prediction = env._parse_action_with_prediction(response_text)
            logger.info(f"\nParsing result:")
            logger.info(f"Action: {action}")
            logger.info(f"Prediction: {prediction[:100] if prediction else None}...")
            
            if action:
                logger.info("✓ Successfully parsed action!")
            else:
                logger.info("✗ Failed to parse action")
                
    finally:
        await env.cleanup()


if __name__ == "__main__":
    asyncio.run(test_textworld_prompt())