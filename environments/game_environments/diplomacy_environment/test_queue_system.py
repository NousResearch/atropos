#!/usr/bin/env python3
"""Test the queue-based trajectory collection system."""

import asyncio
import logging
import os
import sys
import uuid
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

async def test_queue_system():
    """Test the queue-based proxy system."""
    
    # Import after path setup
    from queue_manager import get_queue_manager, PolicyRequest, PolicyResponse
    from atropos_client_minimal import AtroposClientMinimal
    
    # Get queue manager
    queue_manager = get_queue_manager()
    
    # Create a test game
    game_id = f"test-game-{uuid.uuid4()}"
    queue_pair = await queue_manager.create_game_queues(game_id)
    
    print(f"Created queues for game {game_id}")
    
    # Create a test client
    client = AtroposClientMinimal("atropos-test", game_id, queue_manager)
    
    # Start a task to handle requests (simulating the environment)
    async def handle_requests():
        """Simulate the environment handling requests."""
        while True:
            try:
                request = await queue_pair.request_queue.get()
                print(f"\nEnvironment received request:")
                print(f"  ID: {request.request_id}")
                print(f"  Power: {request.power}")
                print(f"  Prompt: {request.prompt[:50]}...")
                
                # Simulate processing and create response
                response = PolicyResponse(
                    request_id=request.request_id,
                    response=f"Test response for {request.power}: I will move my units strategically.",
                    metadata={"test": True, "power": request.power}
                )
                
                await queue_pair.response_queue.put(response)
                print(f"Environment sent response for {request.request_id}")
                
            except Exception as e:
                print(f"Handler error: {e}")
                
    # Start handler task
    handler_task = asyncio.create_task(handle_requests())
    
    # Test the client
    try:
        test_prompts = [
            "You are FRANCE. What is your strategy for Spring 1901?",
            "You are FRANCE. Please submit your orders for Spring 1901 Movement phase.",
            "You are FRANCE. How do you plan to work with ENGLAND?"
        ]
        
        for prompt in test_prompts:
            print(f"\nClient sending: {prompt[:50]}...")
            response = await client.generate_response(prompt, temperature=0.7)
            print(f"Client received: {response}")
            
        # Check interactions
        print(f"\nClient tracked {len(client.interactions)} interactions:")
        for i, interaction in enumerate(client.interactions):
            print(f"\nInteraction {i+1}:")
            print(f"  Power: {interaction['power']}")
            print(f"  Type: {interaction['task_type']}")
            print(f"  Metadata: {interaction.get('metadata', {})}")
            
    finally:
        # Clean up
        handler_task.cancel()
        await queue_manager.remove_game_queues(game_id)
        
    print("\n✅ Queue system test completed!")

if __name__ == "__main__":
    asyncio.run(test_queue_system())