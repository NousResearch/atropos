#!/usr/bin/env python3
"""
Test script to debug AtroposClient proxy registration
"""

import asyncio
import logging
import os
import sys

import dotenv

dotenv.load_dotenv()

# Add AI_Diplomacy to path
sys.path.append(os.path.join(os.path.dirname(__file__), "AI_Diplomacy"))

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


async def test_proxy_registration():
    """Test the proxy registration in isolation"""

    # Test 1: Import and check AI_Diplomacy's client system
    logger.info("=== Test 1: Importing AI_Diplomacy clients ===")
    try:
        from ai_diplomacy import clients

        logger.info("Successfully imported clients module")
        logger.info(f"load_model_client function: {clients.load_model_client}")
    except Exception as e:
        logger.error(f"Failed to import clients: {e}")
        return

    # Test 2: Try to load a standard model
    logger.info("\n=== Test 2: Loading standard OpenAI model ===")
    try:
        standard_client = clients.load_model_client("gpt-4o-mini")
        logger.info(f"Loaded standard client: {type(standard_client)}")
    except Exception as e:
        logger.error(f"Failed to load standard model: {e}")

    # Test 3: Import our AtroposClient
    logger.info("\n=== Test 3: Importing AtroposClientMinimal ===")
    try:
        from atropos_client_minimal import register_atropos_models

        logger.info("Successfully imported AtroposClientMinimal")
    except Exception as e:
        logger.error(f"Failed to import AtroposClientMinimal: {e}")
        return

    # Test 4: Register with a mock server config
    logger.info("\n=== Test 4: Registering AtroposClient ===")
    try:
        # Create a mock server config
        server_config = {
            "model_name": "gpt-4.1",
            "base_url": "https://api.openai.com/v1",
            "api_key": os.getenv("OPENAI_API_KEY", "dummy-key"),
        }

        # Register the models
        register_atropos_models(server_config)
        logger.info("Successfully registered AtroposClient")

        # Check if the registration worked
        original_load = clients.load_model_client
        logger.info(f"load_model_client after registration: {original_load}")

    except Exception as e:
        logger.error(f"Failed to register AtroposClient: {e}", exc_info=True)
        return

    # Test 5: Try to load an atropos model
    logger.info("\n=== Test 5: Loading atropos model ===")
    try:
        atropos_client = clients.load_model_client("atropos-training-policy")
        logger.info(f"Loaded atropos client: {type(atropos_client)}")
        logger.info(f"Client model_name: {atropos_client.model_name}")
        logger.info(
            f"Client server_url: {getattr(atropos_client, 'server_url', 'N/A')}"
        )
    except Exception as e:
        logger.error(f"Failed to load atropos model: {e}", exc_info=True)

    # Test 6: Test a simple generate_response
    logger.info("\n=== Test 6: Testing generate_response ===")
    try:
        if "atropos_client" in locals():
            response = await atropos_client.generate_response("Test prompt")
            logger.info(f"Response: {response[:100]}...")
    except Exception as e:
        logger.error(f"Failed to generate response: {e}", exc_info=True)


async def test_diplomacy_specific():
    """Test with actual Diplomacy prompts"""
    logger.info("\n=== Testing Diplomacy-Specific Prompts ===")

    from ai_diplomacy import clients

    # Test loading our atropos model
    try:
        client = clients.load_model_client("atropos-training-policy")
        logger.info(f"Loaded client: {type(client)}")

        # Test a typical Diplomacy initialization prompt
        test_prompt = """You are FRANCE. The game is beginning in Spring 1901.

Your starting position:
- Army in Paris
- Army in Marseilles
- Fleet in Brest

What are your initial strategic goals and relationships with other powers?

Respond in JSON format:
{
    "initial_goals": ["goal1", "goal2", "goal3"],
    "initial_relationships": {
        "AUSTRIA": "Neutral",
        "ENGLAND": "Friendly",
        ...
    }
}"""

        logger.info("Testing initialization prompt...")
        response = await client.generate_response(test_prompt, temperature=0.7)
        logger.info(f"Response length: {len(response)} chars")
        logger.info(f"Response preview: {response[:200]}...")

        # Test an orders prompt
        orders_prompt = """You are FRANCE in Spring 1901. Submit your orders.

Current units:
- Army in Paris
- Army in Marseilles
- Fleet in Brest

Respond in JSON format:
{
    "orders": {
        "PAR": "PAR H",
        "MAR": "MAR - SPA",
        "BRE": "BRE - MAO"
    }
}"""

        logger.info("\nTesting orders prompt...")
        response = await client.generate_response(orders_prompt, temperature=0.0)
        logger.info(f"Orders response: {response[:200]}...")

    except Exception as e:
        logger.error(f"Diplomacy-specific test failed: {e}", exc_info=True)


async def test_env_registration():
    """Test registration as done in the environment"""
    logger.info("\n=== Testing Environment Registration Pattern ===")

    # Simulate what the environment does
    from atroposlib.envs.base import APIServerConfig

    server_config = APIServerConfig(
        model_name="gpt-4.1",
        base_url="https://api.openai.com/v1",
        api_key=os.getenv("OPENAI_API_KEY", "dummy-key"),
    )

    # Test the registration with APIServerConfig
    try:
        from atropos_client_minimal import register_atropos_models

        # The environment passes server_configs[0] which is an APIServerConfig object
        # But our function expects a dict
        logger.info(f"Server config type: {type(server_config)}")
        logger.info(f"Server config attributes: {vars(server_config)}")

        # This is likely the issue - we need to convert APIServerConfig to dict
        server_config_dict = {
            "model_name": server_config.model_name,
            "base_url": server_config.base_url,
            "api_key": server_config.api_key,
        }

        register_atropos_models(server_config_dict)
        logger.info("Registration succeeded with dict conversion")

    except Exception as e:
        logger.error(f"Registration failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(test_proxy_registration())
    asyncio.run(test_env_registration())
    asyncio.run(test_diplomacy_specific())
