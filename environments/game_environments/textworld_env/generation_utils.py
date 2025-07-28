#!/usr/bin/env python3
"""Utilities for generating TextWorld games, including specific challenges."""

import logging
import os
import random
from typing import Any, Dict, Optional, Tuple

import textworld
import textworld.challenges
from textworld import GameOptions
from textworld.generator import QuestGenerationError, compile_game
from textworld.generator.text_grammar import MissingTextGrammar

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_FOLDER = "/home/maxpaperclips/atropos/environments/game_environments/textworld_env/tw_generated_games"


def compile_game_with_retry(game, options, max_retries=5, cleanup_on_error=True):
    """
    Compile a TextWorld game with automatic retry on cache conflicts.

    If compilation fails due to cache conflicts (existing game with same ID),
    this function will automatically try with a new seed.

    Args:
        game: The game object to compile
        options: GameOptions object
        max_retries: Maximum number of retry attempts
        cleanup_on_error: Whether to clean up partial files on error

    Returns:
        Compiled game file path or None if all retries failed
    """
    original_path = options.path

    for attempt in range(max_retries):
        try:
            # Try to compile
            game_file = compile_game(game, options)
            if game_file and os.path.exists(game_file):
                return game_file
            else:
                logger.warning(
                    f"Compile attempt {attempt + 1} failed: no file produced"
                )

        except AssertionError as e:
            if "same id have different structures" in str(e):
                # Cache conflict - try with a new seed
                logger.info(
                    f"Cache conflict detected on attempt {attempt + 1}, retrying with new seed"
                )

                # Clean up the conflicting file if it exists
                if cleanup_on_error and os.path.exists(options.path):
                    try:
                        os.remove(options.path)
                        # Also remove .ni file if it exists
                        ni_path = options.path.replace(".z8", ".ni")
                        if os.path.exists(ni_path):
                            os.remove(ni_path)
                    except OSError as oe:
                        logger.warning(f"Failed to clean up conflicting file: {oe}")

                # Generate new seed and path
                new_seed = random.randint(0, 65535)
                options.seeds = new_seed

                # Update the path with new seed
                base_path = original_path.rsplit("_seed", 1)[0]
                options.path = f"{base_path}_seed{new_seed}.z8"

                logger.info(f"Retrying with new seed {new_seed}")
            else:
                # Some other assertion error
                logger.error(f"Compile failed with assertion: {e}")
                if cleanup_on_error:
                    _cleanup_game_files(options.path)
                return None

        except Exception as e:
            logger.error(f"Compile failed with error: {e}")
            if cleanup_on_error:
                _cleanup_game_files(options.path)
            return None

    logger.error(f"Failed to compile after {max_retries} attempts")
    return None


def _cleanup_game_files(game_path):
    """Clean up game files (.z8 and .ni)."""
    if not game_path:
        return

    for ext in [".z8", ".ni"]:
        file_path = (
            game_path.replace(".z8", ext)
            if game_path.endswith(".z8")
            else game_path + ext
        )
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass


def generate_textworld_game(
    challenge_name: str,
    settings: Dict[str, Any],
    options: Optional[GameOptions] = None,
    output_folder: str = DEFAULT_OUTPUT_FOLDER,
    filename_prefix: Optional[str] = None,
    cleanup_on_error: bool = True,
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

        seed = settings.get("seed")
        if seed is None:
            seed = random.randint(0, 65535)
            settings["seed"] = seed

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

        # Compile game with retry on cache conflicts
        game_file = compile_game_with_retry(
            game, options, cleanup_on_error=cleanup_on_error
        )

        if game_file and os.path.exists(game_file):
            # Update seed in settings if it changed during retry
            settings["seed"] = options.seeds
            return game_file, game
        else:
            logger.error("compile_game_with_retry failed to produce file")
            return None, None

    except (QuestGenerationError, MissingTextGrammar, ValueError, Exception) as e:
        logger.error(
            f"Error during challenge game generation ('{challenge_name}'): {e}"
        )
        # Clean up partial files if requested
        if (
            cleanup_on_error
            and options
            and options.path
            and os.path.exists(options.path)
        ):
            try:
                os.remove(options.path)
            except OSError:
                pass
        return None, None


if __name__ == "__main__":
    """Test the generation utility."""
    test_settings = {
        "seed": random.randint(0, 65535),
        "rewards": "balanced",
        "goal": "brief",
        "test": False,
    }

    game_file, game_obj = generate_textworld_game("tw-simple", test_settings)

    if game_file:
        logger.info(f"SUCCESS: Game generated and saved to {game_file}")
    else:
        logger.error("FAILURE: Challenge game generation failed.")
