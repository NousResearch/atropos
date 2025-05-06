import asyncio
import logging
import random
import os
import argparse

from dotenv import load_dotenv

# Use config_init from the environment itself
from environments.game_environments.gymnasium.blackjack_env import BlackjackEnv
# from trajectoryhandler.envs.base import OpenaiConfig # No longer needed here
# from trajectoryhandler.utils.config_handler import ConfigHandler # No longer needed here

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Blackjack environment local server")
    parser.add_argument(
        "--config",
        type=str,
        default="blackjack_local", # Default to the local config
        help="Configuration file name (without .yaml extension, relative to envs/gymnasium/configs), or full path to a YAML file.",
    )
    return parser.parse_args()


async def main():
    logger.info("Starting Blackjack environment server")

    args = parse_arguments()

    # Determine the config name/path for config_init
    # config_init expects the name relative to its own configs dir, or an absolute path
    config_input = args.config
    if not os.path.isabs(config_input) and not config_input.endswith(".yaml"):
        # Assume it's a name relative to the blackjack env's config dir
        config_name_or_path = config_input
        logger.info(f"Using relative config name: {config_name_or_path}")
    else:
        # It's likely an absolute path or path relative to cwd
        config_name_or_path = os.path.abspath(config_input)
        logger.info(f"Using absolute config path: {config_name_or_path}")

    # Use the environment's config_init method to load configurations
    try:
        config, server_configs = BlackjackEnv.config_init(config_name_or_path)
        logger.info("Configuration loaded successfully via BlackjackEnv.config_init")
        logger.debug(f"Loaded Env Config: {config}")
        logger.debug(f"Loaded Server Configs: {server_configs}")
    except Exception as e:
        logger.exception(f"Failed to load configuration using BlackjackEnv.config_init: {e}")
        return # Cannot proceed without config

    # Create and set up the environment using the loaded configs
    try:
        env = BlackjackEnv(
            config=config,
            server_configs=server_configs,
            slurm=False, # Explicitly false for local testing
        )
    except Exception as e:
        logger.exception(f"Failed to initialize BlackjackEnv: {e}")
        return

    # Run a single trajectory directly
    logger.info("Running a single trajectory directly")
    try:
        await env.setup() # Setup the server connection etc.
        seed = random.randint(0, 1000000)
        logger.info(f"Using seed: {seed}")

        # Make sure the episode exists before collecting
        # This also initializes the message history correctly
        _ = env._get_or_create_episode(seed)

        result = await env.collect_trajectory(seed, interactive=False) # interactive=False is typical for direct run
        logger.info(f"Trajectory collection complete with {len(result)} steps.")

        # Get episode state for summary (should exist now)
        if seed in env.episodes:
            episode_state = env.episodes[seed]

            # Print a final summary
            logger.info("\n========== Episode Summary ==========")
            logger.info(f"Seed: {seed}")
            logger.info(f"Total steps taken: {len(episode_state.actions)}")
            logger.info(f"Final Environment reward: {episode_state.total_env_reward:.2f}")
            logger.info(f"Final Format reward: {episode_state.total_format_reward:.2f}")
            logger.info(f"Final Combined reward: {episode_state.total_combined_reward:.2f}")
            # Verify calculation based on final totals and weights
            logger.info(
                f"Combined Calculation Check: ({config.environment_reward_weight:.2f} * {episode_state.total_env_reward:.2f}) + "
                f"({config.format_reward_weight:.2f} * {episode_state.total_format_reward:.2f}) = "
                f"{(config.environment_reward_weight * episode_state.total_env_reward) + (config.format_reward_weight * episode_state.total_format_reward):.2f}"
            )

            accuracy = episode_state.num_correct_actions / max(1, episode_state.num_total_actions)
            logger.info(f"Action accuracy (valid format): {episode_state.num_correct_actions}/{episode_state.num_total_actions} ({accuracy:.2%})")
            logger.info("=======================================")
        else:
            logger.error(f"Could not find episode state for seed {seed} after running trajectory.")

    except Exception as e:
        logger.exception(f"An error occurred during trajectory collection or summary: {e}")


if __name__ == "__main__":
    asyncio.run(main())
