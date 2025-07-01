#!/usr/bin/env python
"""
Test script to verify AI_Diplomacy installation and explore its components.
"""

import sys
import os

# Add AI_Diplomacy to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'AI_Diplomacy'))

def test_imports():
    """Test that we can import key AI_Diplomacy components."""
    print("Testing AI_Diplomacy imports...")
    
    try:
        # Core diplomacy engine
        from diplomacy import Game, Power, Map
        print("✓ Core diplomacy engine imported successfully")
        
        # AI components
        from ai_diplomacy.agent import DiplomacyAgent
        from ai_diplomacy.clients import BaseModelClient
        print("✓ AI agent components imported successfully")
        
        # Utilities
        from ai_diplomacy.utils import assign_models_to_powers
        from ai_diplomacy.prompt_constructor import build_context_prompt
        print("✓ AI utilities imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_game_creation():
    """Test creating a basic game."""
    print("\nTesting game creation...")
    
    try:
        from diplomacy import Game
        
        # Create a standard game
        game = Game()
        print(f"✓ Created game with ID: {game.game_id}")
        print(f"  Map: {game.map.name}")
        print(f"  Current phase: {game.get_current_phase()}")
        print(f"  Powers: {', '.join(game.powers.keys())}")
        
        # Check initial units
        for power_name, power in game.powers.items():
            units = len(power.units)
            centers = len(power.centers)
            print(f"  {power_name}: {units} units, {centers} centers")
        
        return True
    except Exception as e:
        print(f"✗ Game creation error: {e}")
        return False

def test_agent_structure():
    """Test the AI agent structure."""
    print("\nTesting AI agent structure...")
    
    try:
        from ai_diplomacy.agent import DiplomacyAgent
        from ai_diplomacy.clients import BaseModelClient
        
        # Create a minimal test client that extends BaseModelClient
        class TestClient(BaseModelClient):
            """Minimal client for testing."""
            async def generate_response(self, prompt: str) -> str:
                return "Test response"
            
            async def get_orders(self, board_state, power_name, possible_orders, **kwargs):
                return possible_orders[:1] if possible_orders else []
            
            async def get_conversation_reply(self, power_name, conversation_so_far, game_phase, **kwargs):
                return "Test conversation reply"
        
        # Create test client
        client = TestClient("test-model")
        
        # Create a test agent with the required client
        agent = DiplomacyAgent("FRANCE", client)
        print(f"✓ Created agent for {agent.power_name}")
        print(f"  Initial goals: {agent.goals}")
        print(f"  Relationships: {list(agent.relationships.keys())}")
        
        # Test diary functionality
        agent.private_diary.append("Test entry")
        print(f"  Diary entries: {len(agent.private_diary)}")
        print(f"  Full diary entries: {len(agent.full_private_diary)}")
        
        # Check agent structure
        print(f"  Has private journal: {hasattr(agent, 'private_journal')}")
        print(f"  Has client: {agent.client.model_name}")
        
        return True
    except Exception as e:
        print(f"✗ Agent creation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def explore_prompts():
    """Explore available prompt templates."""
    print("\nExploring prompt templates...")
    
    prompt_dir = os.path.join(os.path.dirname(__file__), 
                              'AI_Diplomacy/ai_diplomacy/prompts')
    
    if os.path.exists(prompt_dir):
        print(f"✓ Found prompts directory: {prompt_dir}")
        
        # List prompt files
        prompt_files = [f for f in os.listdir(prompt_dir) if f.endswith('.txt')]
        print(f"  Available prompts: {len(prompt_files)} files")
        
        for prompt_file in sorted(prompt_files)[:5]:  # Show first 5
            print(f"    - {prompt_file}")
        
        if len(prompt_files) > 5:
            print(f"    ... and {len(prompt_files) - 5} more")
        
        return True
    else:
        print(f"✗ Prompts directory not found: {prompt_dir}")
        return False

def main():
    """Run all tests."""
    print("AI_Diplomacy Integration Test")
    print("="*50)
    
    tests = [
        test_imports,
        test_game_creation,
        test_agent_structure,
        explore_prompts,
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    print(f"Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ AI_Diplomacy is properly installed and ready for integration!")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()