#!/usr/bin/env python3
"""
Example script showing how to run a Diplomacy game with Atropos integration.

This demonstrates the clean integration between AI_Diplomacy and Atropos
using the LLM Client Proxy approach.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'AI_Diplomacy'))

import asyncio
import logging
from typing import Dict, List, Optional
import json
from pathlib import Path

# Import AI_Diplomacy components
from ai_diplomacy.lm_game import run_llm_game, DiplomacyGame
from ai_diplomacy.agent import DiplomacyAgent
from ai_diplomacy.pathfinding_context import PathfindingContext

# Import our Atropos integration
from atropos_client import AtroposClient, register_atropos_models

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def create_game_with_atropos_agents(
    game_id: str = "atropos-test-game",
    models: Optional[Dict[str, str]] = None
) -> Dict:
    """
    Create and run a Diplomacy game where all agents use Atropos models.
    
    Args:
        game_id: Unique identifier for the game
        models: Optional dict mapping power names to model names.
                If not provided, all powers use the same model.
    
    Returns:
        Game result dictionary
    """
    # Default: all powers use the same Atropos model
    if models is None:
        models = {
            power: "atropos-diplomacy-v1"
            for power in ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]
        }
    
    # Register Atropos models with AI_Diplomacy
    register_atropos_models()
    
    # Create the game
    logger.info(f"Starting game {game_id} with models: {models}")
    
    # Configure game parameters
    game_config = {
        "game_id": game_id,
        "models": models,
        "max_turns": 10,  # Limit for testing
        "deadline_seconds": 300,  # 5 minutes per phase
        "save_game": True,
        "save_path": f"./game_logs/{game_id}.json",
        "pathfinding_context": PathfindingContext(),  # For strategic analysis
        "verbose": True
    }
    
    try:
        # Run the game using AI_Diplomacy's infrastructure
        game_result = await run_llm_game(**game_config)
        
        logger.info(f"Game completed: {game_id}")
        logger.info(f"Winner: {game_result.get('winner', 'No winner')}")
        logger.info(f"Final turn: {game_result.get('current_turn', 'Unknown')}")
        
        return game_result
        
    except Exception as e:
        logger.error(f"Error running game: {e}", exc_info=True)
        raise


async def run_simple_test():
    """Run a simple test game with all Atropos agents."""
    # Ensure log directory exists
    Path("./game_logs").mkdir(exist_ok=True)
    
    # Run a test game
    result = await create_game_with_atropos_agents(
        game_id="test-atropos-integration",
        models={
            # You can mix different Atropos models here
            "AUSTRIA": "atropos-diplomacy-v1",
            "ENGLAND": "atropos-diplomacy-v1", 
            "FRANCE": "atropos-diplomacy-aggressive",  # Example variant
            "GERMANY": "atropos-diplomacy-v1",
            "ITALY": "atropos-diplomacy-v1",
            "RUSSIA": "atropos-diplomacy-v1",
            "TURKEY": "atropos-diplomacy-defensive"  # Example variant
        }
    )
    
    # Save detailed results
    with open(f"./game_logs/{result['game_id']}_summary.json", "w") as f:
        json.dump({
            "game_id": result["game_id"],
            "winner": result.get("winner"),
            "final_turn": result.get("current_turn"),
            "final_supply_centers": result.get("supply_centers"),
            "total_messages": len(result.get("all_messages", [])),
            "phases_played": len(result.get("phase_history", []))
        }, f, indent=2)
    
    logger.info("Test completed successfully!")


async def run_mixed_game_example():
    """
    Example showing how to run a game with mixed agent types.
    Some powers use Atropos, others use standard AI_Diplomacy models.
    """
    result = await create_game_with_atropos_agents(
        game_id="mixed-agents-test",
        models={
            # Atropos agents
            "AUSTRIA": "atropos-diplomacy-v1",
            "ENGLAND": "atropos-diplomacy-v1",
            "FRANCE": "atropos-diplomacy-v1",
            
            # Standard AI_Diplomacy agents (for comparison)
            "GERMANY": "gpt-4o",
            "ITALY": "claude-3-5-sonnet-20241022",
            "RUSSIA": "gpt-4o", 
            "TURKEY": "gemini-2.0-flash-exp"
        }
    )
    
    logger.info(f"Mixed game completed with winner: {result.get('winner')}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Diplomacy games with Atropos integration")
    parser.add_argument("--test", action="store_true", help="Run a simple test game")
    parser.add_argument("--mixed", action="store_true", help="Run a mixed agents game")
    parser.add_argument("--game-id", type=str, help="Custom game ID")
    parser.add_argument("--max-turns", type=int, default=10, help="Maximum turns to play")
    parser.add_argument("--atropos-server", type=str, default="http://localhost:8000",
                       help="URL of the Atropos policy server")
    
    args = parser.parse_args()
    
    # Set server URL
    os.environ["ATROPOS_SERVER_URL"] = args.atropos_server
    
    # Run the appropriate test
    if args.test:
        asyncio.run(run_simple_test())
    elif args.mixed:
        asyncio.run(run_mixed_game_example())
    else:
        # Run a custom game
        game_id = args.game_id or f"custom-game-{os.getpid()}"
        asyncio.run(create_game_with_atropos_agents(game_id=game_id))


if __name__ == "__main__":
    main()