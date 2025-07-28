#!/usr/bin/env python3
"""Test if passing tools=[] fixes the TypeError."""

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("NousResearch/Hermes-4-Qwen3-14B-1-e3")

# Recreate the exact messages from the error log
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
        "required": [
          "command",
          "expected_outcome"
        ]
      }
    }
  }
]
</tools>

For your function call, return a JSON object with function name and arguments within <tool_call> </tool_call> tags with the following schema:
<tool_call>
{"name": "execute_command", "arguments": {"command": "go north", "expected_outcome": "I move north to a new room"}}
</tool_call>

EXAMPLE RESPONSE 1:
<think>
I'm in the kitchen. I see a stove and a fridge. The objective says to cook something. Let me check what's in the fridge first to see what ingredients are available.
</think>
<memory>
Kitchen has stove and fridge. Main objective is cooking. Need to find ingredients.
</memory>
<tool_call>
{"name": "execute_command", "arguments": {"command": "open fridge", "expected_outcome": "The fridge opens, revealing its contents. I expect to see various food items or ingredients inside that I can take and use for cooking."}}
</tool_call>

EXAMPLE RESPONSE 2 (with previous memories):
<think>
Looking at my previous memories, I was exploring the kitchen to find cooking ingredients. I successfully opened the fridge and found eggs, milk, and flour. My goal is still to cook something. Now I need to take these ingredients and find a recipe or mixing bowl. The previous action of opening the fridge worked as expected.
</think>
<memory>
Found eggs, milk, and flour in kitchen fridge. Still need mixing bowl or recipe to cook. Previous exploration of kitchen successful - have stove and ingredients located.
</memory>
<tool_call>
{"name": "execute_command", "arguments": {"command": "take eggs", "expected_outcome": "I take the eggs from the fridge and add them to my inventory"}}
</tool_call>

EXAMPLE RESPONSE 3:
<think>
There's a locked door here and I have a key in my inventory. I should try using the key on the door.
</think>
<memory>
Found locked door in current room. Have key in inventory that might open it.
</memory>
<tool_call>
{"name": "execute_command", "arguments": {"command": "unlock door with key", "expected_outcome": "The key turns in the lock and the door unlocks. I should now be able to open the door and go through it."}}
</tool_call>

Remember: Your entire response must be exactly three XML blocks: <think>...</think> followed by <memory>...</memory> followed by <tool_call>...</tool_call>

FINAL REMINDER: After your <think> block and <memory> block, you MUST wrap your JSON function call in <tool_call> tags. The JSON goes INSIDE the <tool_call> tags, not after them."""

user_prompt = """Objective: You are hungry! Let's cook a delicious meal. Check the cookbook in the kitchen for the recipe. Once done, enjoy your meal!

Current Location & State:
-= Kitchen =-
I am sorry to announce that you are now in the kitchen.

You make out an opened fridge. The fridge contains a diced roasted orange bell pepper and a black pepper. You see an opened oven. The oven is empty! What a waste of a day! You can make out a table. Make a note of this, you might have to put stuff on or in it later on. The table is massive. On the table you can see a cookbook. You can make out a counter. You see a knife on the counter. What's that over there? It looks like it's a stove. What a coincidence, weren't you just thinking about a stove? But the thing is empty.

There is an open screen door leading east. There is an open frosted-glass door leading west. There is an exit to the north.

Inventory: You are carrying nothing."""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]

print("Test 1: Original (expecting TypeError)")
try:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print("✗ Unexpected success! This should have failed")
except TypeError as e:
    print(f"✓ Got expected TypeError: {e}")

print("\nTest 2: With tools=[] parameter")
try:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, tools=[]
    )
    print("✓ Success with tools=[]!")
    print(f"Prompt length: {len(prompt)} chars")
    print(f"First 200 chars: {prompt[:200]}...")
except Exception as e:
    print(f"✗ Failed with: {type(e).__name__}: {e}")
