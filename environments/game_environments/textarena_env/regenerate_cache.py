#!/usr/bin/env python
"""Utility script to regenerate the TextArena registry cache."""

import logging

from textarena_registry import TextArenaGameRegistry

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting registry cache regeneration...")

    # Create registry with cache reset to force regeneration
    registry = TextArenaGameRegistry(use_cache=True, reset_cache=True)

    # Discover all games - this will create the cache
    registry.discover_games()

    all_games = registry.list_available_games("all")
    logger.info(f"Cache regenerated successfully with {len(all_games)} games")

    # Show statistics
    single_player = registry.list_available_games("single")
    two_player = registry.list_available_games("two")
    multi_player = registry.list_available_games("multi")

    logger.info(f"Single-player games: {len(single_player)}")
    logger.info(f"Two-player games: {len(two_player)}")
    logger.info(f"Multi-player games: {len(multi_player)}")


if __name__ == "__main__":
    main()
