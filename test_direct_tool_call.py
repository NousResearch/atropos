#!/usr/bin/env python3
"""Direct test of model's ability to follow tool call format"""

import asyncio
from openai import AsyncOpenAI

async def test_direct():
    client = AsyncOpenAI(
        base_url="http://localhost:30000/v1",
        api_key="dummy"
    )
    
    messages = [
        {
            "role": "system",
            "content": """You are an AI agent playing a text-based adventure game. You must respond using exactly this format:

<think>
[Your reasoning about what to do next]
</think>

<tool_call>
{"name": "execute_command", "arguments": {"command": "go north", "expected_outcome": "I move north to a new room"}}
</tool_call>

The execute_command tool lets you interact with the game world. Always predict what you expect to happen."""
        },
        {
            "role": "user", 
            "content": "You are in a room. There is a door to the north. What do you do?"
        }
    ]
    
    response = await client.chat.completions.create(
        model="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
        messages=messages,
        temperature=0.7,
        max_tokens=500
    )
    
    print("Response:")
    print(response.choices[0].message.content)
    
if __name__ == "__main__":
    asyncio.run(test_direct())