#!/usr/bin/env python3
import argparse
import asyncio
import logging
import os
import random

from dotenv import load_dotenv

from environments.game_environments.textarena.hangman_env import HangmanOnlineEnv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""Local debugging script example - useful to test the environment locally"""


def parse_arguments():
    parser = argparse.ArgumentParser(description="Hangman environment local server")
    parser.add_argument(
        "--config",
        type=str,
        default="hangman_local.yaml",
        help="Configuration file name in the configs directory",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debugging mode to show model inputs/outputs",
    )
    return parser.parse_args()


async def main():
    logger.info("=" * 80)
    logger.info("Starting Hangman environment server")
    logger.info("=" * 80)
    args = parse_arguments()

    # Set debug level if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled - showing detailed model I/O")

    logger.info(f"OpenAI API key: {os.environ['OPENAI_API_KEY']}")

    # Get the configuration from HangmanOnlineEnv.config_init()
    env_config, server_configs = HangmanOnlineEnv.config_init(args.config)

    # Check if a specific config file was requested and it was loaded successfully
    if args.config and hasattr(env_config, "group_size"):
        logger.info(f"Using configuration from {args.config}")
    else:
        logger.warning(f"Could not load {args.config}, using default configuration")

    # Log configuration details in debug mode
    if args.debug:
        logger.debug("-" * 80)
        logger.debug("Environment Configuration:")
        for key, value in vars(env_config).items():
            logger.debug(f"  {key}: {value}")
        logger.debug("Server Configurations:")
        for i, sc in enumerate(server_configs):
            logger.debug(f"  Server {i+1}:")
            for key, value in vars(sc).items():
                if key != "api_key":  # Don't log the full API key
                    logger.debug(f"    {key}: {value}")
        logger.debug("-" * 80)

    # Create environment with the configured settings
    env = HangmanOnlineEnv(
        config=env_config,
        server_configs=server_configs,
        slurm=False,
        debug_mode=args.debug,  # Pass debug flag to environment
    )

    logger.info("Running a single trajectory directly")
    await env.setup()
    seed = random.randint(0, 1000000)
    logger.info(f"Using seed: {seed}")
    logger.info("-" * 80)
    result = await env.collect_trajectory(seed)
    logger.info("-" * 80)
    logger.info(f"Episode complete with trajectory of {len(result)} steps")

    episode_state = env._get_or_create_episode(seed)

    if hasattr(episode_state, "total_env_reward") and hasattr(
        episode_state, "total_format_reward"
    ):
        logger.info("\n" + "=" * 50)
        logger.info("========== Episode Summary ==========")
        logger.info("=" * 50)
        total_steps = len(result) if result else 0
        logger.info(f"Total steps: {total_steps}")
        step_count = (
            len(episode_state.step_rewards)
            if hasattr(episode_state, "step_rewards")
            else 0
        )
        if step_count > 0:
            logger.info(f"Steps with rewards: {step_count}")

        logger.info(f"Environment reward: {episode_state.total_env_reward:.2f}")
        logger.info(f"Format reward: {episode_state.total_format_reward:.2f}")

        if hasattr(episode_state, "total_combined_reward"):
            logger.info(f"Combined reward: {episode_state.total_combined_reward:.2f}")
            if hasattr(env_config, "environment_reward_weight") and hasattr(
                env_config, "format_reward_weight"
            ):
                combined_calc = (
                    env_config.environment_reward_weight
                    * episode_state.total_env_reward
                ) + (
                    env_config.format_reward_weight * episode_state.total_format_reward
                )
                logger.info(
                    f"Combined reward calculation: "
                    f"({env_config.environment_reward_weight:.2f} * {episode_state.total_env_reward:.2f}) + "
                    f"({env_config.format_reward_weight:.2f} * {episode_state.total_format_reward:.2f}) = "
                    f"{combined_calc:.2f}"
                )

        if hasattr(episode_state, "num_correct_actions") and hasattr(
            episode_state, "num_total_actions"
        ):
            accuracy = episode_state.num_correct_actions / max(
                1, episode_state.num_total_actions
            )
            logger.info(
                f"Action accuracy: {episode_state.num_correct_actions}/"
                f"{episode_state.num_total_actions} ({accuracy:.2%})"
            )

        if hasattr(episode_state, "games_won") and hasattr(
            episode_state, "episodes_run"
        ):
            win_rate = episode_state.games_won / max(1, episode_state.episodes_run)
            logger.info(
                f"Win rate: {episode_state.games_won}/{episode_state.episodes_run} ({win_rate:.2%})"
            )

        logger.info("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
