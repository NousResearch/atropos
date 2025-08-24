#!/usr/bin/env python3
"""
Training script example for the checkers environment.
This shows how to set up and run training with the checkers environment.
"""

import asyncio
import logging

from checkers_env import CheckersEnv, CheckersEnvConfig, OpponentType, Player

from atroposlib.envs.base import APIServerConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Main training function"""
    print("🏁 Starting Checkers Environment Training")
    print("=" * 50)

    # Configuration
    env_config = CheckersEnvConfig(
        opponent_type=OpponentType.RANDOM,
        ai_plays_as=Player.RED,
        max_episode_turns=50,  # Shorter games for demo
        eval_episodes=5,  # Fewer episodes for demo
        thinking_enabled=True,
        temperature=0.7,
        tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
        group_size=4,  # Small batch for demo
        use_wandb=False,  # Disable wandb for demo
        max_token_length=4096,
    )

    server_configs = [
        APIServerConfig(
            model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            base_url="http://localhost:9001/v1",
            api_key="x",
            num_max_requests_at_once=10,
            num_requests_for_eval=32,
            timeout=300,
        )
    ]

    # Create environment
    env = CheckersEnv(
        config=env_config, server_configs=server_configs, slurm=False, testing=False
    )

    try:
        # Setup
        await env.setup()
        logger.info("Environment setup complete")

        # Run a few training episodes
        print("\n🎮 Running training episodes...")
        for episode in range(3):  # Just a few for demo
            print(f"\nEpisode {episode + 1}/3")

            # Get next item
            item = await env.get_next_item()
            logger.info(f"Generated item with seed: {item['seed']}")

            # Collect trajectory
            try:
                scored_item, _ = await env.collect_trajectory(item)
                if scored_item:
                    print("  ✓ Episode completed!")
                    print(f"    Score: {scored_item.score:.3f}")
                    print(f"    Winner: {scored_item.extra_info['winner']}")
                    print(f"    Moves: {scored_item.extra_info['move_count']}")
                    print(f"    Valid moves: {scored_item.extra_info['valid_moves']}")
                    print(
                        f"    Invalid moves: {scored_item.extra_info['invalid_moves']}"
                    )
                else:
                    print("  ❌ Episode failed")
            except Exception as e:
                print(f"  ❌ Episode error: {e}")
                logger.error(f"Episode {episode + 1} failed: {e}")

        # Run evaluation (optional)
        print("\n📊 Running evaluation...")
        try:
            await env.evaluate()

            # Print evaluation results
            if env.eval_metrics_custom:
                print("Evaluation results:")
                for metric_name, metric_value in env.eval_metrics_custom:
                    print(f"  {metric_name}: {metric_value:.3f}")
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            logger.error(f"Evaluation failed: {e}")

        print("\n🎉 Training demo completed!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"❌ Training failed: {e}")
        print("\nThis is expected if you don't have a local LLM server running.")
        print("To run actual training, you'll need:")
        print("1. A running LLM server (e.g., vLLM, SGLang, or OpenAI API)")
        print("2. Correct server configuration in the script")
        print("3. Appropriate model and API keys")


if __name__ == "__main__":
    print("Checkers Training Example")
    print("This demonstrates how to train an LLM to play checkers.")
    print("Note: Requires a running LLM server for actual training.")
    print()

    asyncio.run(main())
