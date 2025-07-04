#!/usr/bin/env python
"""
Test script to start the Diplomacy server and create a test game.
"""

import asyncio
import time

from diplomacy import Game, Server
from diplomacy.utils.game_phase_data import Message


def run_server():
    """Start the Diplomacy server and create a test game."""
    print("Starting Diplomacy server on port 8432...")
    server = Server()

    # Create a test game
    print("Creating test game...")
    game = Game()
    server.add_game(game)

    print(f"Game created with ID: {game.game_id}")
    print(f"Game password: {game.password if game.password else 'No password'}")
    print(f"Map: {game.map.name}")
    print(f"Phase: {game.get_current_phase()}")
    print(f"Powers: {list(game.powers.keys())}")

    print("\nServer running on http://localhost:8432")
    print("You can connect via the web interface or programmatically.")
    print("Press Ctrl+C to stop the server.\n")

    try:
        server.start(port=8432)
    except KeyboardInterrupt:
        print("\nShutting down server...")


if __name__ == "__main__":
    run_server()
