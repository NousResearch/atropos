#!/usr/bin/env python3
"""
TextWorld Challenge Registry

Provides a simple registry for the pre-built TextWorld challenges.
"""

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TextWorldChallengeRegistry:
    """Registry for pre-built TextWorld challenges."""

    # Pre-built challenges with their settings ranges for randomization
    CHALLENGES = {
        "tw-simple": {
            "rewards": ["sparse", "balanced", "dense"],
            "goal": ["detailed", "brief", "none"],
            "test": [False],
        },
        "tw-cooking": {
            "recipe": [1, 2, 3],  # Number of ingredients in recipe
            "take": [1, 2, 3],  # Number of ingredients to find
            "cook": [False, True],  # Whether ingredients need cooking
            "open": [False, True],  # Whether containers/doors need opening
            "drop": [False, True],  # Whether inventory has limited capacity
            "go": [1, 6, 9, 12],  # Number of locations
        },
        "tw-coin_collector": {
            "level": [1, 2, 3, 4, 5],  # Difficulty levels
        },
        "tw-treasure_hunter": {
            "level": [1, 2, 3, 4, 5],  # Difficulty levels
        },
    }

    # All available challenge names
    ALL_CHALLENGES = list(CHALLENGES.keys())

    def __init__(self, seed: Optional[int] = None):
        self._challenges = self.CHALLENGES.copy()
        self.rng = random.Random(seed)

    def list_challenges(self) -> List[str]:
        """List all available pre-built challenges."""
        return list(self._challenges.keys())

    def get_random_challenge(
        self, randomize_settings: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """Get a random challenge with optionally randomized settings.

        Args:
            randomize_settings: Whether to randomize settings from available options

        Returns:
            Tuple of (challenge_name, settings_dict)
        """
        challenge_name = self.rng.choice(self.list_challenges())
        return self.get_challenge(challenge_name, randomize_settings)

    def get_challenge(
        self, name: str, randomize_settings: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """Get challenge name and settings (optionally randomized).

        Args:
            name: Challenge name
            randomize_settings: Whether to randomize settings from available options

        Returns:
            Tuple of (challenge_name, settings_dict)
        """
        if name not in self._challenges:
            raise ValueError(
                f"Unknown challenge: {name}. Available: {self.list_challenges()}"
            )

        settings_ranges = self._challenges[name]
        settings = {}

        for key, options in settings_ranges.items():
            if randomize_settings and len(options) > 1:
                # Randomly select from available options
                settings[key] = self.rng.choice(options)
            else:
                # Use first (default) option
                settings[key] = options[0]

        # Generate a seed for this specific game instance
        settings["seed"] = self.rng.randint(0, 0xFFFFFFFF)

        # For tw-cooking, add recipe-seed
        if name == "tw-cooking":
            settings["recipe-seed"] = self.rng.randint(0, 0xFFFFFFFF)

        return name, settings


def create_textworld_registry(seed: Optional[int] = None) -> TextWorldChallengeRegistry:
    """Create a TextWorld challenge registry.

    Args:
        seed: Random seed for reproducibility

    Returns:
        TextWorldChallengeRegistry instance
    """
    return TextWorldChallengeRegistry(seed)
