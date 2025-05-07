#!/usr/bin/env python3
import argparse
import asyncio
import logging
import os

from dotenv import load_dotenv

# Only need DatasetEnv now, as it handles config init
from environments.dataset_environment.dataset_env import DatasetEnv
# from atroposlib.envs.base import OpenaiConfig # Not needed here
# from atroposlib.envs.reward_fns import registry # Not needed here

load_dotenv()

logging.basicConfig(level=logging.INFO) # Keep base level INFO for the script itself
logger = logging.getLogger(__name__)
# Note: DatasetEnv will set its own logger level based on debug_mode in its config


def parse_arguments():
    parser = argparse.ArgumentParser(description="Dataset environment local server")
    parser.add_argument(
        "--config",
        type=str,
        default="dataset_local",
        help=(
            "Configuration file name (without .yaml extension, relative to "
            "environments/dataset_environment/configs/), or full path to a YAML file."
        ),
    )
    return parser.parse_args()


async def main():
    logger.info("Starting Dataset environment local server")

    # Parse command line arguments
    args = parse_arguments()

    # Determine config name/path for config_init
    config_input = args.config
    if not os.path.isabs(config_input) and not config_input.endswith(".yaml"):
        # Assume it's a name relative to the dataset env's config dir
        # (config_init handles joining with its own base dir and adding .yaml)
        config_name_or_path = config_input
        logger.info(f"Using relative config name: {config_name_or_path}")
    else:
        # It's likely an absolute path or path relative to cwd
        config_name_or_path = os.path.abspath(config_input)
        logger.info(f"Using absolute config path: {config_name_or_path}")

    # Use the environment's config_init method to load configurations
    try:
        env_config, server_configs = DatasetEnv.config_init(config_name_or_path)
        logger.info("Configuration loaded successfully via DatasetEnv.config_init")
        logger.debug(f"Loaded Env Config: {env_config}")
        logger.debug(f"Loaded Server Configs: {server_configs}")
    except Exception as e:
        logger.exception(f"Failed to load configuration using DatasetEnv.config_init: {e}")
        return # Cannot proceed without config


    # Create the environment using loaded configs
    logger.info("Creating dataset environment...")
    try:
        env = DatasetEnv(
            config=env_config,
            server_configs=server_configs,
            slurm=False, # Explicitly false for local testing
            testing=False # Explicitly false unless needed for harness
        )
    except Exception as e:
        logger.exception(f"Failed to initialize DatasetEnv: {e}")
        return

    # Setup the environment directly
    try:
        await env.setup()
        logger.info("Environment setup complete")
    except Exception as setup_error:
        logger.error(f"Error during environment setup: {setup_error}")
        logger.exception(setup_error)
        return

    # --- Start Test Run --- #
    logger.info("\n=== Starting Local Test Run ===")
    test_items_count = 5 # Number of dataset items to test
    successful_runs = 0

    for i in range(test_items_count):
        logger.info(f"\n--- Running Test Item {i+1}/{test_items_count} ---")
        try:
            # Get a sample item from the dataset
            item = await env.get_next_item()
            if not item or not item[0]:
                logger.warning("Failed to get a valid item from the environment.")
                continue

            prompt, answer, ground_truth = item
            user_content = dict(prompt[0])["content"]
            logger.info(
                f"Prompt: {user_content[:200]}..." if user_content else "(Empty Prompt)"
            )
            if answer:
                logger.info(
                    f"Answer: {answer[:200]}..." if answer else "(Empty Answer)"
                )
            if ground_truth:
                logger.info(
                    f"Ground Truth: {ground_truth[:200]}..."
                    if ground_truth
                    else "(Empty Ground Truth)"
                )

            # Collect trajectories (using group_size from config)
            logger.info(
                f"Collecting {env.config.group_size} trajectories for this item..."
            )
            # Use the correct config attribute name (group_size)
            trajectories_data, backlog = await env.collect_trajectories(item)

            if not trajectories_data:
                logger.warning("No trajectories were collected.")
                continue

            logger.info(f"Collected {len(trajectories_data)} trajectories.")
            # Log first trajectory message content for inspection
            if trajectories_data[0] and isinstance(trajectories_data[0], list):
                first_response = "(Empty or invalid trajectory format)"
                assistant_msgs = [
                    m
                    for m in trajectories_data[0]
                    if isinstance(m, dict) and m.get("role") == "assistant"
                ]
                if assistant_msgs:
                    first_response = assistant_msgs[-1].get("content", "(No content)")
                logger.info(f"First Response Content: {first_response[:300]}...")
            else:
                 logger.warning(f"First trajectory data is empty or not a list: {trajectories_data[0]}")


            # Score the collected trajectories
            logger.info("Scoring trajectories...")
            scored_data = await env.score(trajectories_data)

            # Print scores
            if scored_data and "scores" in scored_data:
                scores_list = scored_data["scores"]
                if scores_list:
                     logger.info(f"Scores: {scores_list}")
                     logger.info(f"  Avg Score: {sum(scores_list)/len(scores_list):.4f}")
                     successful_runs += 1
                else:
                     logger.warning("Scores list is empty in scored_data.")
            else:
                logger.warning("No scores available in the scored data for this item.")

        except Exception as run_error:
            logger.error(f"Error during test item {i+1}")
            logger.exception(run_error) # Log full traceback
            # Optionally continue to the next item or break
            # break

    logger.info(
        f"\n=== Local Test Run Complete ({successful_runs}/{test_items_count} items processed successfully) ==="
    )


if __name__ == "__main__":
    asyncio.run(main())
