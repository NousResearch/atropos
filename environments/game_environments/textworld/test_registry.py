#!/usr/bin/env python3
"""Test script for TextWorld registry implementation."""

import asyncio
import logging
import os
import tempfile
from environments.game_environments.textworld.textworld_registry import create_textworld_registry
from environments.game_environments.textworld.textworld_env import TextWorldEnv, TextWorldEnvConfig
from atroposlib.envs.base import APIServerConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_registry():
    """Test the TextWorld registry functionality."""
    
    # Create a registry
    registry = create_textworld_registry(
        generation_ratio=0.5,  # 50/50 split between generated and pre-built
        seed=42
    )
    
    print("\n=== Testing TextWorld Registry ===")
    
    # Test 1: List available challenges
    print("\n1. Available pre-built challenges:")
    challenges = registry.challenge_registry.list_challenges()
    for challenge in challenges:
        print(f"   - {challenge}")
    
    # Test 2: Get a specific challenge
    print("\n2. Getting 'tw-simple' challenge:")
    name, settings = registry.challenge_registry.get_challenge("tw-simple")
    print(f"   Name: {name}")
    print(f"   Settings: {settings}")
    
    # Test 3: Random challenge selection
    print("\n3. Getting random challenge:")
    name, settings = registry.challenge_registry.get_random_challenge()
    print(f"   Name: {name}")
    print(f"   Settings: {settings}")
    
    # Test 4: Random challenge with difficulty filter
    print("\n4. Getting random 'easy' challenge:")
    name, settings = registry.challenge_registry.get_random_challenge(difficulty="easy")
    print(f"   Name: {name}")
    print(f"   Settings: {settings}")
    
    # Test 5: Generate games using registry
    print("\n5. Testing environment generation:")
    
    # Test different modes
    modes = ["random", "generated", "challenge"]
    for mode in modes:
        print(f"\n   Mode: {mode}")
        game_file, config = registry.get_environment(mode=mode)
        if game_file:
            print(f"   - Game file: {game_file}")
            print(f"   - Config: {config}")
            print(f"   - File exists: {os.path.exists(game_file)}")
            
            # Clean up
            registry.cleanup_game_file(game_file)
            print(f"   - Cleaned up: {not os.path.exists(game_file)}")
        else:
            print("   - Failed to generate game")
    
    # Test 6: Generate with specific difficulty
    print("\n6. Testing difficulty-specific generation:")
    difficulties = ["easy", "medium", "hard", "random"]
    for difficulty in difficulties:
        print(f"\n   Difficulty: {difficulty}")
        game_file, config = registry.get_environment(
            mode="random",
            difficulty=difficulty
        )
        if game_file:
            print(f"   - Game type: {config.get('game_type', config.get('challenge_name', 'N/A'))}")
            print(f"   - Source: {config.get('source')}")
            print(f"   - Actual difficulty: {config.get('difficulty', 'N/A')}")
            registry.cleanup_game_file(game_file)
    
    # Test 7: Test with TextWorldEnv integration
    print("\n7. Testing integration with TextWorldEnv:")
    
    # Create environment config with registry enabled
    config = TextWorldEnvConfig(
        use_registry=True,
        registry_mode="random",
        registry_generation_ratio=0.7,
        registry_difficulty="medium",
        max_steps=30,
        vrcli_enabled=True
    )
    
    # Create API server config
    server_config = APIServerConfig(
        model_name="NousResearch/DeepHermes-3-Mistral-24B-Preview",
        base_url="http://localhost:30000/v1",
        api_key="dummy",
    )
    
    # Create environment
    env = TextWorldEnv(
        config=config,
        server_configs=[server_config],
        slurm=False,
        testing=True
    )
    
    # Initialize registry
    await env.setup()
    
    # Get a game instance
    item = await env.get_next_item()
    if item:
        print("   - Successfully created episode via registry")
        print(f"   - Episode ID: {item['episode_id']}")
        print(f"   - Game file: {item['episode_state'].game_file}")
        
        # Clean up
        await env.cleanup()
    else:
        print("   - Failed to create episode")
    
    # Final cleanup
    print("\n8. Final cleanup:")
    registry.cleanup_all()
    print("   - All tracked files cleaned up")
    
    print("\n=== Registry test complete ===")


if __name__ == "__main__":
    asyncio.run(test_registry())