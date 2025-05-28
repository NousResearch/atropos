#!/usr/bin/env python3
"""
Local testing script for InternBootcamp environment
"""

import asyncio
import logging
import os

from dotenv import load_dotenv

from atroposlib.envs.base import APIServerConfig, EvalHandlingEnum
from environments.intern_bootcamp.intern_bootcamp_env import (
    InternBootcampEnv,
    InternBootcampEnvConfig,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    logger.info("Starting InternBootcamp environment local test runner")

    # Test configuration - using Game24 as an example
    env_config = InternBootcampEnvConfig(
        tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",  # Using proper tokenizer
        group_size=2,  # Small group for testing
        use_wandb=False,
        wandb_name="intern_bootcamp_local_test",
        max_num_workers=1,
        rollout_server_url="http://localhost:8000",
        total_steps=1,
        batch_size=2,
        steps_per_eval=0,
        max_token_length=1024,
        inference_weight=1.0,
        data_path_to_save_groups=None,
        eval_handling=EvalHandlingEnum.NONE,
        eval_limit_ratio=0.0,
        # InternBootcamp specific settings
        task_name="Game24bootcamp",
        task_params={
            "num_numbers": 4,
            "range_max": 20,  # Smaller range for easier problems
            "target_max": 30,
        },
        correct_reward=1.0,
        incorrect_reward=-0.5,
        format_bonus=0.2,
        require_reasoning=True,
        min_reasoning_length=20,
        temperature=0.7,
        top_p=0.9,
    )

    server_configs = [
        APIServerConfig(
            model_name="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
            num_requests_for_eval=0,
        )
    ]

    logger.info("Using test configuration for Game24bootcamp")
    logger.debug(f"Env Config: {env_config}")
    logger.debug(f"Server Configs: {server_configs}")

    try:
        env = InternBootcampEnv(
            config=env_config, server_configs=server_configs, slurm=False
        )
    except Exception as e:
        logger.exception(f"Failed to initialize InternBootcampEnv: {e}")
        return

    logger.info("Running test trajectories")
    try:
        await env.setup()

        # Test 1: Generate and solve a single problem
        logger.info("\n========== Test 1: Single Problem ==========")
        item = await env.get_next_item()
        prompt_tuple, metadata = item

        logger.info("Generated problem:")
        logger.info(f"  Task: {metadata['task_name']}")
        logger.info(f"  Identity: {metadata['identity']}")
        logger.info(f"  Prompt: {metadata['raw_prompt'][:200]}...")

        # Collect trajectories
        trajectories, backlog = await env.collect_trajectories(item)
        logger.info(f"Collected {len(trajectories)} trajectories")

        # Score the trajectories
        scored_data = await env.score(trajectories)
        logger.info(f"Scored {len(scored_data['scores'])} responses")

        for i, (messages, metadata, response) in enumerate(trajectories):
            logger.info(f"\n--- Response {i+1} ---")
            logger.info(f"Model response: {response[:200]}...")
            logger.info(f"Score: {scored_data['scores'][i]}")

        # Test 2: Run evaluation (reduced for testing)
        logger.info("\n========== Test 2: Evaluation (3 problems) ==========")

        async def quick_evaluate(*args, **kwargs):
            logger.info("Starting quick evaluation with 3 problems")
            eval_tasks = []
            for i in range(3):  # Only 3 problems for testing
                logger.info(f"Starting evaluation problem {i+1}/3")
                eval_tasks.append(env.evaluate_single_problem())

            results = await asyncio.gather(*eval_tasks)

            # Calculate metrics
            correct_count = sum(1 for is_correct, _ in results if is_correct)
            format_count = sum(1 for _, has_format in results if has_format)
            total_count = len(results)

            accuracy = correct_count / total_count if total_count > 0 else 0
            format_rate = format_count / total_count if total_count > 0 else 0

            logger.info(
                f"Quick evaluation complete: accuracy={accuracy:.2%}, format_rate={format_rate:.2%}"
            )

            return [(f"eval/{env.current_task_name}_accuracy", accuracy)]

        env.evaluate = quick_evaluate
        await env.evaluate()

        # Test 3: Test different bootcamp tasks
        logger.info("\n========== Test 3: Testing Other Bootcamps ==========")
        test_tasks = ["Sudokubootcamp", "Mazebootcamp", "Cipherbootcamp"]

        for task_name in test_tasks:
            try:
                # Create a new config for each task
                test_config = InternBootcampEnvConfig(
                    **env_config.model_dump(),
                    task_name=task_name,
                    task_params={},  # Use default parameters
                )

                test_env = InternBootcampEnv(
                    config=test_config,
                    server_configs=server_configs,
                    slurm=False,
                    testing=True,
                )

                await test_env.setup()
                item = await test_env.get_next_item()
                _, metadata = item

                logger.info(f"\n{task_name}:")
                logger.info(f"  Generated problem: {metadata['identity']}")
                logger.info(f"  Prompt preview: {metadata['raw_prompt'][:100]}...")

            except Exception as e:
                logger.error(f"Failed to test {task_name}: {e}")

        logger.info("\n========== Test Complete ==========")

    except Exception as e:
        logger.exception(f"An error occurred during testing: {e}")


if __name__ == "__main__":
    asyncio.run(main())
