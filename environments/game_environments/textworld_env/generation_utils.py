#!/usr/bin/env python3
"""Utilities for generating TextWorld games from pre-built challenges."""

import logging
from typing import Any, Dict, Optional, Tuple

import textworld
import textworld.challenges

logger = logging.getLogger(__name__)


def generate_textworld_game(
    challenge_name: str,
    settings: Dict[str, Any],
    output_folder: Optional[str] = None,
    filename_prefix: Optional[str] = None,
) -> Tuple[Optional[str], Optional[Any]]:
    """
    Generate a TextWorld game using pre-built challenges.

    Args:
        challenge_name: Name of the challenge (e.g., "tw-simple", "tw-cooking")
        settings: Challenge-specific settings
        output_folder: Where to save the game (if needed)
        filename_prefix: Prefix for the game file (if saved)

    Returns:
        Tuple of (game_file_path, game_object)
    """
    try:
        # Generate game based on challenge type
        if challenge_name == "tw-simple":
            game_file = textworld.challenges.simple.make(**settings)[0]
        elif challenge_name == "tw-cooking":
            game_file = textworld.challenges.cooking.make(**settings)[0]
        elif challenge_name == "tw-coin_collector":
            game_file = textworld.challenges.coin_collector.make(**settings)[0]
        elif challenge_name == "tw-treasure_hunter":
            game_file = textworld.challenges.treasure_hunter.make(**settings)[0]
        else:
            logger.error(f"Unknown challenge: {challenge_name}")
            return None, None

        logger.info(f"Generated {challenge_name} game with settings: {settings}")
        return game_file, None

    except Exception as e:
        logger.error(f"Error generating {challenge_name} game: {e}")
        return None, None
