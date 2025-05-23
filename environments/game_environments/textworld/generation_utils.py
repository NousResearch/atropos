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
    Generate and compile a TextWorld game based on a challenge name and settings.

    Args:
        challenge_name: The name of the TextWorld challenge (e.g., 'tw-simple').
        settings: Dictionary containing challenge-specific settings (must include 'seed').
        options: Optional pre-configured GameOptions object.
        output_folder: Directory to save the compiled game file.
        filename_prefix: Optional prefix for the game filename.

    Returns:
        Tuple of (game_file_path, game_object) if successful, else (None, None).
    """
    try:
        if not options:
            options = GameOptions()

        seed = settings.get('seed')
        if seed is None:
            seed = random.randint(0, 65535)
            settings['seed'] = seed
        
        options.seeds = seed
        options.file_ext = options.file_ext or ".z8"

        # Prepare output path
        os.makedirs(output_folder, exist_ok=True)
        prefix = filename_prefix or challenge_name
        game_filename = f"{prefix}_seed{seed}.z8"
        options.path = os.path.join(output_folder, game_filename)

        # Generate game
        if challenge_name not in textworld.challenges.CHALLENGES:
            raise ValueError(f"Unknown challenge: {challenge_name}")
        
        _, make_challenge_game, _ = textworld.challenges.CHALLENGES[challenge_name]
        
        game = make_challenge_game(settings=settings, options=options)
        if not game:
            raise RuntimeError("Challenge make function did not return a game object.")

        # Compile game
        game_file = compile_game(game, options)

        if game_file and os.path.exists(game_file):
            return game_file, game
        else:
            logger.error(f"compile_game failed to produce file at {options.path}")
            return None, None

    except (QuestGenerationError, MissingTextGrammar, ValueError, Exception) as e:
        logger.error(f"Error during challenge game generation ('{challenge_name}'): {e}")
        return None, None


if __name__ == "__main__":
    """Test the generation utility."""
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
