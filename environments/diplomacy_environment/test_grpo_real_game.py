#!/usr/bin/env python3
"""Test GRPO environment with a real game using actual LLM models."""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent / "AI_Diplomacy"))

from diplomacy_env_grpo import DiplomacyEnvGRPO, DiplomacyEnvGRPOConfig

from atroposlib.envs.base import APIServerConfig

# Enable debug logging to see AtroposClient GRPO messages
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


async def test_real_game():
    """Test GRPO environment with a real game using OpenAI API."""
    print("\n=== Testing GRPO with Real Game (OpenAI API) ===\n")
    print("Creating environment config...")

    try:
        # Create environment config
        config = DiplomacyEnvGRPOConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=4,
            training_power="FRANCE",
            max_game_years=1,  # Just 1 year for quick test
            negotiation_rounds=0,  # Skip negotiations to speed up
            game_log_dir="./test_grpo_logs",
            use_wandb=False,
            temperature=0.7,
            opponent_models={
                "AUSTRIA": "gpt-4.1-nano",
                "ENGLAND": "gpt-4.1-nano",
                "GERMANY": "gpt-4.1-nano",
                "ITALY": "gpt-4.1-nano",
                "RUSSIA": "gpt-4.1-nano",
                "TURKEY": "gpt-4.1-nano",
            },
        )

        # Server config for the training policy - use OpenAI directly
        server_configs = [
            APIServerConfig(
                model_name="gpt-4.1-nano",
                base_url="https://api.openai.com/v1",
                api_key=os.getenv("OPENAI_API_KEY"),
                num_requests_for_eval=0,
            )
        ]

        # Create environment
        env = DiplomacyEnvGRPO(config, server_configs, testing=False)
        await env.setup()

        print("✓ Environment created and setup")

        # Get next item
        item = await env.get_next_item()
        print(f"\nRunning game: {item['game_id']}")
        print(f"Training power: {item['training_power']}")

        # Ensure we have OpenAI API key
        if "OPENAI_API_KEY" not in os.environ:
            logger.warning("OPENAI_API_KEY not set, setting a placeholder")
            os.environ["OPENAI_API_KEY"] = "test-key"

        # Collect trajectories
        print("\nCollecting trajectories (this will take a few minutes)...")
        start_time = time.time()

        trajectories, backlog = await env.collect_trajectories(item)

        elapsed = time.time() - start_time
        print(f"\nTrajectory collection complete in {elapsed:.1f} seconds!")

        # Analyze results
        if trajectories:
            print(f"\nCollected {len(trajectories)} decision points")

            # Show first few decisions
            for i, group in enumerate(trajectories[:3]):
                overrides = group.get("group_overrides", {})
                print(f"\nDecision {i+1}:")
                print(f"  Power: {overrides.get('power')}")
                print(f"  Task type: {overrides.get('task_type')}")
                print(f"  Phase: {overrides.get('phase')}")
                print(f"  Alternatives: {len(group['tokens'])}")
                print(f"  Token counts: {[len(t) for t in group['tokens']]}")
                print(f"  Final scores: {[f'{s:.3f}' for s in group['scores']]}")

            if len(trajectories) > 3:
                print(f"\n... and {len(trajectories) - 3} more decisions")

            # Check credit assignment worked
            print("\nCredit Assignment Check:")
            if len(trajectories) >= 2:
                first_scores = trajectories[0]["scores"]
                last_scores = trajectories[-1]["scores"]
                print(
                    f"First decision average score: {sum(first_scores)/len(first_scores):.3f}"
                )
                print(
                    f"Last decision average score: {sum(last_scores)/len(last_scores):.3f}"
                )
                print("(First should be higher due to discounted returns)")
        else:
            print("\nNo trajectories collected - check logs for errors")

        # Check if game log was created
        game_file = os.path.join(config.game_log_dir, f"{item['game_id']}.json")
        if os.path.exists(game_file):
            print(f"\n✓ Game log saved to: {game_file}")
            with open(game_file, "r") as f:
                game_data = json.load(f)
            print(f"  Phases played: {len(game_data.get('phases', []))}")
            print(
                f"  Final year: {game_data.get('phases', [{}])[-1].get('name', 'unknown')}"
            )

        print("\n=== Test Complete ===")

    except Exception as e:
        logger.error(f"Test error: {e}", exc_info=True)
        raise


async def main():
    """Run the test."""
    try:
        await test_real_game()

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
