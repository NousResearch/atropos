#!/usr/bin/env python3
"""Test LaTRo rewards implementation with OpenAI API."""

import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from diplomacy_env_grpo import DiplomacyEnvGRPO, DiplomacyEnvGRPOConfig

from atroposlib.envs.base import APIServerConfig

# Enable debug logging to see LaTRo scores
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_latro_rewards():
    """Test that LaTRo rewards are computed correctly."""
    print("\n=== Testing LaTRo Rewards ===\n")

    # Create minimal config
    config = DiplomacyEnvGRPOConfig(
        tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
        group_size=4,  # Sample 4 alternatives
        training_power="FRANCE",
        max_game_years=1,  # Just test one year
        negotiation_rounds=0,  # Skip negotiations
        game_log_dir="./test_latro_logs",
        use_wandb=False,
        temperature=0.7,
        use_latro_rewards=True,  # Enable LaTRo rewards
        opponent_models={
            p: "mistral-small3.1:latest"
            for p in ["AUSTRIA", "ENGLAND", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]
        },
    )

    # Server config - Use Ollama with mistral-small
    server_configs = [
        APIServerConfig(
            model_name="mistral-small3.1:latest",
            base_url="http://localhost:11434",  # Ollama's OpenAI-compatible endpoint (no /v1)
            api_key="dummy",  # Ollama doesn't need an API key
            num_requests_for_eval=0,
        )
    ]

    # Create environment
    env = DiplomacyEnvGRPO(config, server_configs, testing=False)
    await env.setup()

    print("✓ Environment created with LaTRo rewards enabled")

    # Get item
    item = await env.get_next_item()
    print(f"✓ Running game: {item['game_id']}")

    # Run trajectories
    print("\nCollecting trajectories (watch for LaTRo score logs)...")
    trajectories, _ = await env.collect_trajectories(item)

    print(f"\n✓ Collected {len(trajectories)} decision points")

    if trajectories:
        # Analyze scores
        print("\nScore Analysis:")
        for i, group in enumerate(trajectories[:3]):
            print(f"\nDecision {i+1}:")
            scores = group.get("scores", [])
            print(f"  Number of alternatives: {len(scores)}")
            print(f"  Scores: {[f'{s:.3f}' for s in scores]}")
            print(f"  Score range: [{min(scores):.3f}, {max(scores):.3f}]")
            print(f"  Selected: index with score {max(scores):.3f}")

            # Check if scores show variation (indicating LaTRo is working)
            score_std = np.std(scores) if len(scores) > 1 else 0
            print(f"  Score std dev: {score_std:.3f}")
            if score_std > 0.1:
                print(
                    "  ✓ Good score variation - LaTRo rewards are differentiating responses"
                )
            else:
                print("  ⚠️  Low score variation - responses may be too similar")


async def main():
    try:
        # Check API key
        if not os.getenv("OPENAI_API_KEY"):
            print("ERROR: OPENAI_API_KEY not set!")
            return

        import numpy as np  # Import here so we can use it

        await test_latro_rewards()

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
