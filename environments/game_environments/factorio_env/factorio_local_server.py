#!/usr/bin/env python3
"""
Local test server for the minimal Factorio environment.

This script runs Factorio episodes using OpenRouter's gpt-oss-120b model
to test the FLE integration with Atropos.
"""

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

from atroposlib.envs.base import APIServerConfig, EvalHandlingEnum
from environments.game_environments.factorio_env.factorio_env_minimal import (
    FactorioEnv,
    FactorioEnvConfig,
)

load_dotenv()

# Set debug logging to see what's happening
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# Reduce noise from httpx
logging.getLogger("httpx").setLevel(logging.INFO)

model_name = "DeepHermes-3-Mistral-24B-Preview-q8"

async def main():
    """Run Factorio episodes for testing the minimal environment."""
    logger.info("Starting Factorio minimal environment local test runner")

    # Configure environment
    env_config = FactorioEnvConfig(
        tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
        group_size=1,  # Single trajectory for testing
        use_wandb=False,
        wandb_name="factorio_minimal_local_test",
        max_num_workers=1,
        rollout_server_url="http://localhost:8000",
        total_steps=1,
        batch_size=1,
        steps_per_eval=0,
        max_token_length=32768,
        inference_weight=1.0,
        data_path_to_save_groups=None,
        eval_handling=EvalHandlingEnum.NONE,
        eval_limit_ratio=0.0,
        max_steps_per_episode=30,  # Short episodes for testing
        include_messages=True,  # Include messages for debugging
        eval_episodes=0,
        # Factorio specific settings
        task_names=["iron_ore_throughput"],  # Start with simplest task
        enable_self_planning=True,  # Test self-planning capability
        factorio_host="localhost",
        factorio_tcp_port=27000,
        factorio_fast_mode=False,  # Run at normal speed for testing
    )

    # Configure server - using local LLM server
    server_configs = [
        APIServerConfig(
            model_name=model_name,  # Using the local llama-server model
            base_url="http://127.0.0.1:8080/v1",
            api_key="dummy",  # Local server doesn't need an API key
            num_requests_for_eval=0,
        ),
    ]

    logger.info(f"Using local LLM server with {model_name} for Factorio test")
    logger.debug(f"Env Config: {env_config}")
    logger.debug(f"Server Configs: {server_configs}")

    try:
        env = FactorioEnv(
            config=env_config,
            server_configs=server_configs,
            slurm=False,
            testing=False,
        )
    except Exception as e:
        logger.exception(f"Failed to initialize FactorioEnv: {e}")
        return

    logger.info("Running test episodes")
    try:
        await env.setup()

        # Get number of episodes from command line or default
        num_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 3

        # Allow specific task selection from command line
        if len(sys.argv) > 2:
            task_name = sys.argv[2]
            if task_name in ["iron_ore_throughput", "copper_ore_throughput", 
                           "iron_plate_throughput", "automation_science_pack_throughput"]:
                env.config.task_names = [task_name]
                logger.info(f"Testing specific task: {task_name}")

        # Track statistics
        episode_results = []
        task_completions = {}
        total_rewards = []
        total_steps = []

        for episode_num in range(num_episodes):
            logger.info(f"\n{'='*60}")
            logger.info(f"Episode {episode_num + 1}/{num_episodes}")
            logger.info(f"{'='*60}")

            item = await env.get_next_item()
            task_name = item["task_name"]
            logger.info(f"Task: {task_name}")
            logger.info(f"Seed: {item['seed']}")
            logger.info(f"Episode ID: {item['episode_id']}")

            # Collect trajectories
            scored_data_group, _ = await env.collect_trajectories(item)

            if scored_data_group and scored_data_group["scores"]:
                # Extract metrics
                scores = scored_data_group["scores"]
                avg_score = sum(scores) / len(scores) if scores else 0
                
                logger.info(f"\nResults:")
                logger.info(f"  Trajectories collected: {len(scores)}")
                logger.info(f"  Average score: {avg_score:.2f}")
                
                # Get detailed metrics from episode buffers
                if env.episode_outcomes_buffer:
                    task_completed = env.episode_outcomes_buffer[-1]
                    logger.info(f"  Task completed: {task_completed}")
                    
                    # Track completion by task type
                    if task_name not in task_completions:
                        task_completions[task_name] = []
                    task_completions[task_name].append(task_completed)
                
                if env.episode_rewards_buffer:
                    episode_reward = env.episode_rewards_buffer[-1]
                    total_rewards.append(episode_reward)
                    logger.info(f"  Total reward: {episode_reward:.2f}")
                
                if env.episode_steps_buffer:
                    episode_steps = env.episode_steps_buffer[-1]
                    total_steps.append(episode_steps)
                    logger.info(f"  Steps taken: {episode_steps}")
                
                # Show messages if debugging
                if env.config.include_messages and scored_data_group.get("messages"):
                    messages = scored_data_group["messages"][0] if scored_data_group["messages"] else []
                    if messages and len(messages) > 2:
                        logger.info("\n  Sample interaction:")
                        # Show first goal-setting if present
                        for msg in messages[:10]:
                            if msg["role"] == "assistant" and "update_goals" in msg["content"]:
                                logger.info(f"    Agent goals: {msg['content'][:200]}...")
                                break
                
                episode_results.append({
                    "episode": episode_num + 1,
                    "task": task_name,
                    "score": avg_score,
                    "completed": task_completed if env.episode_outcomes_buffer else False,
                    "reward": episode_reward if env.episode_rewards_buffer else 0,
                    "steps": episode_steps if env.episode_steps_buffer else 0,
                })
            else:
                logger.error("Failed to collect trajectory")
                episode_results.append({
                    "episode": episode_num + 1,
                    "task": task_name,
                    "score": 0.0,
                    "completed": False,
                    "reward": 0,
                    "steps": 0,
                })

        # Print overall statistics
        logger.info("\n" + "="*60)
        logger.info("OVERALL RESULTS SUMMARY")
        logger.info("="*60)
        logger.info(f"Total episodes: {num_episodes}")
        logger.info(f"Group size: {env.config.group_size} trajectory per episode")
        
        # Task completion rates
        logger.info("\nTask Completion Rates:")
        for task, completions in task_completions.items():
            if completions:
                rate = sum(completions) / len(completions) * 100
                logger.info(f"  {task}: {rate:.1f}% ({sum(completions)}/{len(completions)})")
        
        # Overall metrics
        if total_rewards:
            logger.info(f"\nAverage reward: {sum(total_rewards)/len(total_rewards):.2f}")
            logger.info(f"Max reward: {max(total_rewards):.2f}")
            logger.info(f"Min reward: {min(total_rewards):.2f}")
        
        if total_steps:
            logger.info(f"\nAverage steps: {sum(total_steps)/len(total_steps):.1f}")
            logger.info(f"Min steps: {min(total_steps)}")
            logger.info(f"Max steps: {max(total_steps)}")
        
        # Episode details
        logger.info("\nEpisode Details:")
        for result in episode_results:
            status = "✓" if result["completed"] else "✗"
            logger.info(
                f"  Episode {result['episode']}: {result['task']} - "
                f"{status} Score: {result['score']:.2f}, "
                f"Reward: {result['reward']:.2f}, "
                f"Steps: {result['steps']}"
            )

    except Exception as e:
        logger.exception(f"Error during test execution: {e}")
    finally:
        logger.info("\nTest completed")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Usage: python factorio_local_server.py [num_episodes] [task_name]")
        print("\nnum_episodes: Number of episodes to run (default: 3)")
        print("task_name: Specific task to test (optional)")
        print("  Options: iron_ore_throughput, copper_ore_throughput,")
        print("           iron_plate_throughput, automation_science_pack_throughput")
        sys.exit(0)
    
    asyncio.run(main())