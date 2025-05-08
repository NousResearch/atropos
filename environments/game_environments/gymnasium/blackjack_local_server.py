import asyncio
import logging
import random
import os
import argparse

from dotenv import load_dotenv

# Use config_init from the environment itself
from environments.game_environments.gymnasium.blackjack_env import BlackjackEnv

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

        result_trajectory = await env.collect_trajectory(seed)
        logger.info(f"Trajectory collection complete with {len(result_trajectory)} steps.")

        episode_summary = None
        if env.completed_episode_metrics_buffer:
            # Assume the last entry in the buffer corresponds to the trajectory just run
            episode_summary = env.completed_episode_metrics_buffer[-1]
            # Optionally, clear the buffer if this script is only for single runs
            # env.completed_episode_metrics_buffer.clear()

        if episode_summary and episode_summary.get("seed") == seed:
            # Print a final summary
            logger.info("\n========== Episode Summary ==========")
            logger.info(f"Seed: {episode_summary['seed']}")
            logger.info(f"Total steps taken: {episode_summary['num_steps']}")
            logger.info(f"Final Environment reward: {episode_summary['total_reward']:.2f}")

            game_outcome_val = episode_summary.get('game_outcome', 0)
            outcome_str = "Draw"
            if game_outcome_val == 1:
                outcome_str = "Win"
            elif game_outcome_val == -1:
                outcome_str = "Loss"
            logger.info(f"Game Outcome: {outcome_str} (Reward: {episode_summary['total_reward']:.0f})")

            # Calculate and log action accuracy based on EpisodeState fields
            if episode_summary['num_total_actions'] > 0:
                accuracy = episode_summary['num_correct_actions'] / episode_summary['num_total_actions']
                logger.info(f"Action accuracy (valid tool calls): {episode_summary['num_correct_actions']}/{episode_summary['num_total_actions']} ({accuracy:.2%})")
            else:
                logger.info("Action accuracy (valid tool calls): No tool calls attempted or recorded.")
            logger.info("=======================================")
        else:
            logger.error(f"Could not get episode summary for seed {seed} from metrics buffer or seed mismatch.")

    except Exception as e:
        logger.exception(f"An error occurred during trajectory collection or summary: {e}")


if __name__ == "__main__":
    asyncio.run(main())
