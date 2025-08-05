#!/usr/bin/env python3
"""
Local test server for the minimal Diplomacy environment with mock SGLang.

This version simulates SGLang responses for local testing without actual servers.
"""

import asyncio
import json
import logging
import os
from typing import Dict

from dotenv import load_dotenv

from atroposlib.envs.base import APIServerConfig, EvalHandlingEnum
from environments.game_environments.diplomacy_environment.diplomacy_env_minimal import (
    DiplomacyEnvMinimal,
    DiplomacyEnvMinimalConfig,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockDiplomacyEnv(DiplomacyEnvMinimal):
    """Mock version that simulates SGLang responses."""
    
    async def _handle_policy_request(self, request):
        """Handle policy request with mock responses instead of SGLang."""
        try:
            logger.info(f"Mock handling request {request.request_id} for {request.power} in game {request.game_id}")
            
            # Generate a mock response based on the prompt
            response_text = self._generate_mock_response(request.prompt, request.power)
            
            # Create response
            from queue_manager import PolicyResponse
            policy_response = PolicyResponse(
                request_id=request.request_id,
                response=response_text,
                metadata={
                    "power": request.power,
                    "phase": request.phase,
                    "mock": True
                }
            )
            
            # Put response back on queue
            await self.queue_manager.put_response(request.game_id, policy_response)
            logger.info(f"Sent mock response for request {request.request_id}")
            
        except Exception as e:
            logger.error(f"Error handling policy request: {e}")
            # Send error response
            from queue_manager import PolicyResponse
            error_response = PolicyResponse(
                request_id=request.request_id,
                response="Error: Failed to generate response",
                metadata={"error": str(e)}
            )
            await self.queue_manager.put_response(request.game_id, error_response)
    
    def _generate_mock_response(self, prompt: str, power: str) -> str:
        """Generate a mock response based on the prompt type."""
        prompt_lower = prompt.lower()
        
        if "orders" in prompt_lower:
            # Mock orders response
            orders = self._get_mock_orders(power)
            explanations = {
                "general": f"Mock orders for {power} - focusing on defense and expansion",
                "per_unit": {unit: f"Moving {unit} strategically" for unit in orders}
            }
            return json.dumps({
                "orders": orders,
                "explanations": explanations
            })
        elif "message" in prompt_lower or "negotiate" in prompt_lower:
            # Mock negotiation response
            return json.dumps({
                "messages": {
                    "ENGLAND": f"Greetings from {power}. Let's work together.",
                    "GERMANY": f"{power} proposes cooperation against common threats."
                },
                "explanations": {
                    "general": f"Mock diplomatic messages from {power}"
                }
            })
        elif "goals" in prompt_lower or "strategy" in prompt_lower:
            # Mock strategy response
            return json.dumps({
                "initial_goals": [
                    f"Secure {power}'s home centers",
                    "Form strategic alliances",
                    "Expand carefully"
                ],
                "initial_relationships": {
                    p: "Neutral" for p in ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]
                    if p != power
                }
            })
        else:
            # Generic response
            return f"Mock response from {power}: Acknowledged. Proceeding with diplomatic strategy."
    
    def _get_mock_orders(self, power: str) -> Dict[str, str]:
        """Get mock orders for a power."""
        # Simple mock orders for Spring 1901
        mock_orders = {
            "FRANCE": {
                "A PAR": "BUR",
                "A MAR": "SPA",
                "F BRE": "MAO"
            },
            "ENGLAND": {
                "F LON": "NTH",
                "F EDI": "NWG", 
                "A LVP": "YOR"
            },
            "GERMANY": {
                "A BER": "KIE",
                "A MUN": "RUH",
                "F KIE": "DEN"
            },
            "ITALY": {
                "A VEN": "TYR",
                "A ROM": "VEN",
                "F NAP": "ION"
            },
            "AUSTRIA": {
                "A VIE": "GAL",
                "A BUD": "SER",
                "F TRI": "ALB"
            },
            "RUSSIA": {
                "A MOS": "UKR",
                "A WAR": "GAL",
                "F SEV": "BLA",
                "F STP/SC": "BOT"
            },
            "TURKEY": {
                "A CON": "BUL",
                "A SMY": "ARM",
                "F ANK": "BLA"
            }
        }
        return mock_orders.get(power, {})


async def main():
    """Run Diplomacy games for testing the minimal environment."""
    logger.info("Starting Diplomacy minimal environment mock test runner")

    # Configure environment
    env_config = DiplomacyEnvMinimalConfig(
        tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
        group_size=1,  # Just 1 game for quick test
        use_wandb=False,
        wandb_name="diplomacy_minimal_mock_test",
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
        max_game_turns=2,  # Very short games for testing
        training_power="FRANCE",
        include_messages=True,
        eval_episodes=0,
        start_diplomacy_server=False,  # Don't start real server
        save_game_logs=True,
        game_logs_dir="./test_game_logs_mock",
    )

    # Mock server configs (not actually used)
    server_configs = [
        APIServerConfig(
            model_name="mock-model",
            base_url="http://localhost:9004/v1",
            api_key="mock",
            num_requests_for_eval=0,
        )
        for _ in range(4)
    ]

    logger.info("Using mock responses for Diplomacy test")

    try:
        env = MockDiplomacyEnv(
            config=env_config,
            server_configs=server_configs,
            slurm=False,
            testing=True,
        )
    except Exception as e:
        logger.exception(f"Failed to initialize MockDiplomacyEnv: {e}")
        return

    logger.info("Running test games with mock responses")
    try:
        await env.setup()

        # Get number of episodes from command line or default
        import sys
        num_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 1

        # Track statistics
        episode_results = []

        for episode_num in range(num_episodes):
            logger.info(f"\n===== Episode {episode_num + 1}/{num_episodes} =====")

            item = await env.get_next_item()
            logger.info(f"Game ID: {item['game_id']}, Seed: {item['seed']}")

            # Collect trajectories
            scored_data_group, _ = await env.collect_trajectories(item)

            if scored_data_group and scored_data_group["scores"]:
                avg_score = sum(scored_data_group["scores"]) / len(scored_data_group["scores"])
                logger.info(f"Collected {len(scored_data_group['scores'])} trajectories with average score: {avg_score:.2f}")

                # Check interactions
                if env.active_games:
                    for game_id, game_info in env.active_games.items():
                        for model_id, client in game_info.get("clients", {}).items():
                            if client.current_power == env.config.training_power:
                                logger.info(f"Client for {client.current_power} collected {len(client.interactions)} interactions")
                                for i, interaction in enumerate(client.interactions[:3]):  # Show first 3
                                    logger.info(f"  Interaction {i+1}: {interaction['task_type']} - {interaction['prompt'][:50]}...")

        logger.info("\n✅ Mock test completed successfully!")

    except Exception as e:
        logger.exception(f"Error during test execution: {e}")
    finally:
        # Cleanup
        if hasattr(env, "game_server_process") and env.game_server_process:
            env.game_server_process.terminate()
            env.game_server_process.wait()


if __name__ == "__main__":
    asyncio.run(main())