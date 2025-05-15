#!/usr/bin/env python3
"""
Local server script for testing the TextWorldEnv environment.

Generates a single TextWorld game based on config, runs one episode,
and prints trajectory information.
"""

import argparse
import asyncio
import logging
import os
import sys
from typing import List

from dotenv import load_dotenv

# Update imports to match blackjack_local_server.py
from environments.game_environments.textworld.textworld_env import TextWorldEnv, TextWorldEnvConfig
from trajectoryhandler.envs.base import OpenaiConfig # Import OpenaiConfig
from trajectoryhandler.utils.config_handler import ConfigHandler # Import ConfigHandler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run a test episode for TextWorldEnv.")
    parser.add_argument(
        "--config",
        type=str,
        default="textworld",
        help="Configuration file name (without .yaml extension or path for configs/envs/ directory, or full path)."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging."
    )
    return parser.parse_args()

async def run_test_episode(args):
    """Loads config using ConfigHandler, creates env, runs one episode."""
    env = None # Initialize env to None for finally block
    try:
        # Initialize config handler and determine config path
        config_handler = ConfigHandler()
        if os.path.isabs(args.config) or "/" in args.config or args.config.endswith(".yaml"):
            config_path = args.config
        else:
            config_path = os.path.join(config_handler.config_dir, f"envs/{args.config}.yaml")

        logger.info(f"Loading configuration from: {config_path}")
        try:
            # Load raw config dictionary
            import yaml
            with open(config_path, "r") as f:
                raw_config = yaml.safe_load(f) or {}
            logger.info(f"Loaded raw configuration successfully")
        except FileNotFoundError:
            logger.error(f"Configuration file not found at: {config_path}")
            return
        except Exception as e:
            logger.error(f"Error loading config: {e}", exc_info=True)
            return

        # Extract TextWorld specific config section (assuming it's nested under 'textworld')
        # If your YAML structure is different (e.g., list under 'environments'), adjust access accordingly.
        tw_config_dict = raw_config.get('textworld', raw_config.get('environments', [{}])[0]) 
        if not tw_config_dict or tw_config_dict.get('env_name', '') != 'TextWorld':
             # Attempt to find it in a list if the first attempt failed
             found = False
             if isinstance(raw_config.get('environments'), list):
                 for cfg_item in raw_config['environments']:
                      if isinstance(cfg_item, dict) and (cfg_item.get('env_name') == 'TextWorld' or cfg_item.get('type') == 'TextWorldEnvConfig'):
                           tw_config_dict = cfg_item
                           found = True
                           break
             if not found:
                  logger.error("Could not find TextWorld configuration section in the YAML.")
                  print(f"Ensure your {config_path} has a dictionary for TextWorld, perhaps under an 'environments' list.")
                  return
        
        # Create TextWorldEnvConfig instance
        env_config = TextWorldEnvConfig(
            # BaseEnv parameters (extract from root or defaults)
            tokenizer_name=raw_config.get("tokenizer_name", "NousResearch/DeepHermes-3-Llama-3-8B-Preview"),
            group_size=tw_config_dict.get("group_size", 3),
            use_wandb=raw_config.get("use_wandb", False),
            max_num_workers=raw_config.get("max_num_workers", 1),
            rollout_server_url=raw_config.get("rollout_server_url", "http://localhost:8000"),
            total_steps=raw_config.get("total_steps", 1),
            batch_size=raw_config.get("batch_size", 1),
            steps_per_eval=raw_config.get("steps_per_eval", 20),
            max_token_length=tw_config_dict.get("max_token_length", 300),
            wandb_name=raw_config.get("wandb_name", "textworld_test"),
            ensure_scores_are_not_same=raw_config.get("ensure_scores_are_not_same", True),

            # TextWorldEnv specific parameters
            env_name=tw_config_dict.get("env_name", "TextWorld"),
            temperature=tw_config_dict.get("temperature", 0.7),
            top_p=tw_config_dict.get("top_p", 0.9),
            max_steps=tw_config_dict.get("max_steps", 50),
            challenge_name=tw_config_dict.get("challenge_name", "tw-simple"),
            challenge_rewards=tw_config_dict.get("challenge_rewards", "balanced"),
            challenge_goal=tw_config_dict.get("challenge_goal", "brief"),
            challenge_test_mode=tw_config_dict.get("challenge_test_mode", False),
            nb_rooms=tw_config_dict.get("nb_rooms", 5),
            nb_objects=tw_config_dict.get("nb_objects", 10),
            quest_min_length=tw_config_dict.get("quest_min_length", 3),
            quest_max_length=tw_config_dict.get("quest_max_length", 3),
            quest_max_depth=tw_config_dict.get("quest_max_depth", 3),
            grammar_theme=tw_config_dict.get("grammar_theme", "house"),
            grammar_include_adj=tw_config_dict.get("grammar_include_adj", True),
            game_seed=tw_config_dict.get("game_seed", None),
            thinking_active=tw_config_dict.get("thinking_active", True),
            thinking_prefill=tw_config_dict.get("thinking_prefill", "<think>\n"),
            reward_functions=tw_config_dict.get("reward_functions", ["tool_calling"]),
            format_reward_weight=tw_config_dict.get("format_reward_weight", 0.3),
            environment_reward_weight=tw_config_dict.get("environment_reward_weight", 0.7),
            invalid_action_penalty=tw_config_dict.get("invalid_action_penalty", -0.1)
        )

        # Extract Server configurations (simplified, matching Blackjack)
        server_configs_raw = raw_config.get("servers", {}).get("openai_servers", [])
        server_configs: List[OpenaiConfig] = []
        
        if not server_configs_raw:
            logger.warning("No OpenAI server configurations found in YAML. Creating default from env vars.")
            server_configs.append(
                OpenaiConfig(
                    model_name=os.environ.get("OPENAI_MODEL", "gpt-4.1-nano"), # Default model
                    base_url=os.environ.get("OPENAI_API_BASE"), # Default to None if env var not set
                    api_key=os.environ.get("OPENAI_API_KEY"), # Default to None if env var not set
                    # Add other necessary fields with defaults if needed, e.g., num_requests_for_eval
                    num_requests_for_eval=64
                )
            )
        else:
            for sc_raw in server_configs_raw:
                 api_key = sc_raw.get("api_key", os.environ.get("OPENAI_API_KEY"))
                 base_url = sc_raw.get("base_url") # Get directly, will be None if not present
                 model_name = sc_raw.get("model_name", "gpt-4.1-nano") # Default model if not in config
                 
                 logger.debug(f"Creating OpenaiConfig with: model_name='{model_name}', base_url='{base_url}'")
                 server_configs.append(
                    OpenaiConfig(
                        api_key=api_key,
                        base_url=base_url, 
                        model_name=model_name, 
                        # Add other fields directly from sc_raw with defaults if needed
                        num_requests_for_eval=sc_raw.get("num_requests_for_eval", 64)
                    )
                )
        
        # Ensure at least one server config exists after processing
        if not server_configs:
             logger.error("Failed to configure any OpenAI server. Check YAML and .env variables.")
             return

        # Instantiate the environment
        # Pass testing=False as per user correction
        env = TextWorldEnv(config=env_config, server_configs=server_configs, slurm=False, testing=False)
        logger.info("TextWorldEnv instantiated.")

        # Setup the environment
        await env.setup()
        logger.info("Environment setup complete.")

        # Get parameters for the next game/episode
        item = await env.get_next_item()
        if not item or not all(k in item for k in ["challenge_name", "challenge_settings"]):
             logger.error("Failed to get valid game settings from get_next_item.")
             return # Cleanup happens in finally
             
        logger.info(f"Generated game settings: Challenge='{item['challenge_name']}', Settings={item['challenge_settings']}")

        # Run a single episode
        logger.info("Starting trajectory collection...")
        trajectory_data, _ = await env.collect_trajectories(item)

        if trajectory_data:
            logger.info(f"Trajectory collection complete. Collected {len(trajectory_data)} steps.")
            # Optional: Print details of the trajectory
            for i, step_data in enumerate(trajectory_data):
                # Log the type of step_data for debugging
                logger.debug(f"Type of step_data at index {i}: {type(step_data)}") 
                # Check if it has the attribute before accessing
                if hasattr(step_data, 'parsed_action'):
                     logger.info(f" Step {i+1}: Best Action: {step_data.parsed_action}")
                else:
                     logger.warning(f" Step {i+1}: step_data is missing 'parsed_action'. Data: {step_data}")
                     
                if args.debug:
                    # Assuming step_data might be a dict if not ScoredDataGroup
                    scores = getattr(step_data, 'scores', step_data.get('scores', [])) if isinstance(step_data, dict) else getattr(step_data, 'scores', [])
                    messages = getattr(step_data, 'messages', step_data.get('messages', [])) if isinstance(step_data, dict) else getattr(step_data, 'messages', [])
                    
                    logger.debug(f"  Scores: {scores}")
                    if len(messages) >= 2:
                         logger.debug(f"  Agent Response: {messages[-2].get('content', 'N/A') if isinstance(messages[-2], dict) else messages[-2]}")
                         logger.debug(f"  Env Observation: {messages[-1].get('content', 'N/A') if isinstance(messages[-1], dict) else messages[-1]}")
                    # Log full step messages if needed at DEBUG
                    # logger.debug(f"  Full Messages: {messages}")
        else:
            logger.warning("Trajectory collection finished, but no data was returned. Game generation or interaction might have failed.")

    except Exception as e:
        logger.error(f"An error occurred during the test run: {e}", exc_info=True)
    finally:
        if env: # Check if env was successfully initialized
            logger.info("Cleaning up environment...")
            await env.cleanup()
            logger.info("Environment cleanup complete.")

if __name__ == "__main__":
    # Load environment variables from .env file if it exists
    try:
        load_dotenv()
        logger.info(".env file loaded if present.")
    except ImportError:
        logger.info("python-dotenv not installed, skipping .env file loading.")
    except Exception as e:
        logger.warning(f"Error loading .env file: {e}")

    args = parse_arguments()

    # Set logging level based on debug flag
    log_level = logging.DEBUG if args.debug else logging.INFO
    # Use force=True to override basicConfig if already called (e.g., by imports)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
    # Optionally set specific logger levels more granularly
    logging.getLogger("trajectoryhandler").setLevel(log_level) 
    logging.getLogger("environments").setLevel(log_level)
    logging.getLogger("textworld").setLevel(logging.WARNING) # Keep TextWorld lib less noisy

    logger.info(f"Starting TextWorldEnv local test with config: {args.config}")
    asyncio.run(run_test_episode(args))
    logger.info("TextWorldEnv local test finished.")
