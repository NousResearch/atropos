#!/usr/bin/env python3
"""Utilities for generating TextWorld games, including specific challenges."""

import logging
import os
import random
import textworld
import textworld.challenges
from textworld import GameOptions
from textworld.generator import QuestGenerationError, compile_game
from textworld.generator.text_grammar import MissingTextGrammar
from typing import Dict, Any, Optional, Tuple

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_FOLDER = "environments/game_environments/textworld/tw_generated_games"

def generate_textworld_game(
    challenge_name: str,
    settings: Dict[str, Any],
    options: Optional[GameOptions] = None,
    output_folder: str = DEFAULT_OUTPUT_FOLDER,
    filename_prefix: Optional[str] = None
) -> Tuple[Optional[str], Optional[Any]]:
    """
    Generates and compiles a TextWorld game based on a challenge name and settings.

    Args:
        challenge_name: The name of the TextWorld challenge (e.g., 'tw-simple').
        settings: Dictionary containing challenge-specific settings (e.g., seed, rewards, goal).
                  Must include 'seed'.
        options: An optional pre-configured GameOptions object. If None, a default one is created.
                 Key settings like 'seeds' and 'path' will be overwritten.
        output_folder: The directory to save the compiled game file.
        filename_prefix: Optional prefix for the game filename.

    Returns:
        A tuple containing (game_file_path, game_object) if successful, else (None, None).
    """
    try:
        if not options:
            options = GameOptions()

        seed = settings.get('seed')
        if seed is None:
            seed = random.randint(0, 65535)
            settings['seed'] = seed # Ensure seed is in settings
        
        options.seeds = seed # Set seed in options as well
        options.file_ext = options.file_ext or ".z8" # Default to z8 if not set

        # --- Prepare Output Path --- 
        os.makedirs(output_folder, exist_ok=True)
        prefix = filename_prefix or challenge_name
        game_filename = f"{prefix}_seed{seed}.z8"
        options.path = os.path.join(output_folder, game_filename)

        logger.info(f"Attempting to generate challenge: {challenge_name}")
        logger.info(f"Challenge settings: {settings}")
        logger.info(f"Output path: {options.path}")

        # --- Generation & Compilation --- 
        if challenge_name not in textworld.challenges.CHALLENGES:
            raise ValueError(f"Unknown challenge: {challenge_name}")
        
        _, make_challenge_game, _ = textworld.challenges.CHALLENGES[challenge_name]
        
        # 1. Make the game object
        logger.debug("Calling challenge make function...")
        game = make_challenge_game(settings=settings, options=options)
        if not game:
            raise RuntimeError("Challenge make function did not return a game object.")
        logger.debug("Game object created successfully.")

        # 2. Compile the game
        logger.debug("Calling compile_game...")
        game_file = compile_game(game, options)

        if game_file and os.path.exists(game_file):
            logger.info(f"Game generated and compiled successfully!")
            logger.info(f"Game file: {game_file}")
            logger.info(f"Game Object Info (UUID): {game.metadata.get('uuid', 'N/A')}")
            return game_file, game
        else:
            logger.error(f"compile_game failed to produce file at {options.path}")
            return None, None

    except (QuestGenerationError, MissingTextGrammar, ValueError, Exception) as e:
        logger.error(f"Error during challenge game generation ('{challenge_name}'): {e}", exc_info=True)
        return None, None

# Example usage / test function (can be run directly)
if __name__ == "__main__":
    logger.info("--- Testing TextWorld Generation Utility (tw-simple) ---")
    
    test_settings = {
        'seed': random.randint(0, 65535),
        'rewards': 'balanced',
        'goal': 'brief',
        'test': False
    }
    
    game_file, game_obj = generate_textworld_game("tw-simple", test_settings)
    
    if game_file:
        logger.info(f"SUCCESS: Game generated and saved to {game_file}")
    else:
        logger.error("FAILURE: Challenge game generation failed.")
    logger.info("---------------------------------------------------------")
