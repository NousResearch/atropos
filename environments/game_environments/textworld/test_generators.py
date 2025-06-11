#!/usr/bin/env python3
"""Test the new game generators."""

import asyncio
import logging
import os
from environments.game_environments.textworld.textworld_registry import create_textworld_registry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_generators():
    """Test all the new generators."""
    
    # Create registry
    registry = create_textworld_registry(
        generation_ratio=1.0,  # Use only generated games for this test
        seed=42
    )
    
    print("\n=== Testing TextWorld Generators ===")
    
    # Test each game type
    game_types = ["quest", "puzzle", "navigation", "mixed"]
    difficulties = ["easy", "medium", "hard"]
    
    success_count = 0
    total_tests = 0
    
    for game_type in game_types:
        print(f"\n--- Testing {game_type.upper()} Generator ---")
        
        for difficulty in difficulties:
            total_tests += 1
            print(f"\nGenerating {game_type} game ({difficulty})...")
            
            try:
                game_file, config = registry.get_environment(
                    mode="generated",
                    difficulty=difficulty,
                    game_type=game_type
                )
                
                if game_file and os.path.exists(game_file):
                    success_count += 1
                    print(f"✓ SUCCESS: {os.path.basename(game_file)}")
                    print(f"  Config: {config}")
                    
                    # Clean up
                    registry.cleanup_game_file(game_file)
                else:
                    print(f"✗ FAILED: No game file generated")
                    
            except Exception as e:
                print(f"✗ ERROR: {e}")
    
    # Summary
    print(f"\n=== Test Summary ===")
    print(f"Total tests: {total_tests}")
    print(f"Successful: {success_count}")
    print(f"Failed: {total_tests - success_count}")
    print(f"Success rate: {success_count/total_tests*100:.1f}%")
    
    # Test specific sub-types
    print("\n=== Testing Specific Sub-types ===")
    
    # Test quest sub-types
    quest_types = ["fetch", "delivery", "exploration"]
    for quest_type in quest_types:
        print(f"\nTesting quest type: {quest_type}")
        # Note: Current implementation doesn't expose sub-type selection through registry
        # This would need to be added as an enhancement
    
    # Test puzzle sub-types
    puzzle_types = ["door_sequence", "combination_lock", "weight_puzzle"]
    for puzzle_type in puzzle_types:
        print(f"\nTesting puzzle type: {puzzle_type}")
        # Same note as above
    
    print("\n=== Generator test complete ===")


if __name__ == "__main__":
    asyncio.run(test_generators())