#!/usr/bin/env python3
"""Test inline memory generation in TextWorld environment."""

import asyncio
import logging
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Disable some verbose logging
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

async def test_inline_memory():
    """Test the inline memory generation with TextWorld prompts."""
    
    # Load environment variables
    load_dotenv()
    
    # Configure the client to use local SGLang server
    client = AsyncOpenAI(
        api_key="EMPTY",  # SGLang doesn't need a real API key
        base_url="http://localhost:30000/v1"
    )
    
    # System prompt with inline memory instructions
    system_prompt = """You are an AI agent playing a text-based adventure game who uses extreme long chains of thought to carefully plan your actions and predict their outcomes. Your goal is to follow the objective described at the start of the game. You interact with the world by providing text commands and predicting their outcomes.

You should:
1. Enclose your thoughts and internal monologue inside <think> </think> tags. Use extremely long chains of thought to carefully consider the game state, your objectives, and the likely outcomes of your actions.
2. Generate a memory summary inside <memory> </memory> tags that captures key information from this turn. Your memory should:
   - Build upon previous memories shown in 'Relevant Memories' if present
   - Note the outcome of your last action (did it match your prediction?)
   - Update your understanding of the game state, location, and inventory
   - Track progress toward objectives and any multi-step plans
   - Be concise but comprehensive (1-3 sentences)
3. Provide your action using the execute_command function call.

<tools>
[
  {
    "type": "function",
    "function": {
      "name": "execute_command",
      "description": "Execute a text command in the adventure game and predict the outcome.",
      "parameters": {
        "type": "object",
        "properties": {
          "command": {
            "type": "string",
            "description": "The command to execute in the game."
          },
          "expected_outcome": {
            "type": "string",
            "description": "What you expect to observe after executing this command. Be specific about changes to the environment, your location, inventory, or game state."
          }
        },
        "required": ["command", "expected_outcome"]
      }
    }
  }
]
</tools>

For your function call, return a JSON object with function name and arguments within <tool_call> </tool_call> tags with the following schema:
<tool_call>
{"name": "execute_command", "arguments": {"command": "go north", "expected_outcome": "I move north to a new room"}}
</tool_call>

Remember: Your entire response must be exactly three XML blocks: <think>...</think> followed by <memory>...</memory> followed by <tool_call>...</tool_call>"""
    
    # Test cases
    test_observations = [
        # First turn - no previous memories
        """Objective: Find the cookbook and prepare a meal.

Current Location & State:
Kitchen
You are in a kitchen. There's a stove, a fridge, and a counter here. The fridge is closed.

Inventory: You are carrying nothing.
""",
        # Second turn - with relevant memories
        """Relevant Memories:
- Kitchen has stove, fridge, and counter. Main objective is finding cookbook and preparing meal.

Original Observation:
Objective: Find the cookbook and prepare a meal.

Current Location & State:
Kitchen
You are in a kitchen. There's a stove, a fridge, and a counter here. The fridge is open, revealing eggs, milk, and flour inside.

Inventory: You are carrying nothing.

Feedback from last action ('open fridge'):
You open the fridge, revealing eggs, milk, and flour inside.""",
        # Third turn - continuing the game
        """Relevant Memories:
- Kitchen has stove, fridge, and counter. Main objective is finding cookbook and preparing meal.
- Found eggs, milk, and flour in fridge. Need cookbook to know recipes.

Original Observation:
Objective: Find the cookbook and prepare a meal.

Current Location & State:
Kitchen
You are in a kitchen. There's a stove, a fridge, and a counter here. The fridge is open. There are eggs in the fridge.

Inventory: You are carrying milk and flour.

Feedback from last action ('take milk and flour'):
You take the milk and flour from the fridge."""
    ]
    
    for i, observation in enumerate(test_observations):
        print(f"\n{'='*80}")
        print(f"TEST CASE {i+1}")
        print(f"{'='*80}")
        print(f"Observation:\n{observation}\n")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": observation}
        ]
        
        try:
            # Convert messages to prompt using tokenizer (like AtroposAgent does)
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("NousResearch/DeepHermes-3-Llama-3-8B-Preview")
            prompt = tokenizer.apply_chat_template(messages, tokenize=False)
            
            # Use completions endpoint to avoid SGLang tool call parsing
            response = await client.completions.create(
                model="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                prompt=prompt,
                temperature=0.7,
                max_tokens=16384,  # Use full context length the model is trained for
                top_p=0.9
            )
            
            content = response.choices[0].text.strip()
            print(f"Agent Response:\n{content}\n")
            
            # Test memory extraction
            from environments.game_environments.textworld.utils.memory_parser import extract_memory_block, validate_memory_content
            
            memory = extract_memory_block(content)
            if memory:
                print(f"Extracted Memory: {memory}")
                print(f"Memory Valid: {validate_memory_content(memory)}")
            else:
                print("No memory block found in response!")
                
        except Exception as e:
            logger.error(f"Error in test case {i+1}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("Test completed!")

if __name__ == "__main__":
    asyncio.run(test_inline_memory())