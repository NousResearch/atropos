#!/usr/bin/env python3
"""Test script to isolate Qwen tokenizer chat template issue."""

from transformers import AutoTokenizer

# Load the Qwen tokenizer
print("Loading Qwen tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Hermes-4-Qwen3-14B-1-e3")

# Create test messages similar to what AtroposAgent is using
test_messages = [
    {
        "role": "system",
        "content": "You are an AI agent playing a text-based adventure game. You should think step-by-step."
    },
    {
        "role": "user", 
        "content": "Objective: Cook a meal.\n\nCurrent Location: Kitchen\nYou see a fridge and a stove."
    }
]

print("\nTest 1: Basic messages")
print(f"Messages: {len(test_messages)} messages")
for i, msg in enumerate(test_messages):
    print(f"  Message {i}: role={msg['role']}, content_length={len(msg['content'])}")

try:
    prompt = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
    print("✓ Success! Generated prompt:")
    print(f"Prompt length: {len(prompt)} chars")
    print(f"First 200 chars: {prompt[:200]}...")
except Exception as e:
    print(f"✗ Failed with error: {type(e).__name__}: {e}")

# Test with the actual system prompt from AtroposAgent
print("\n\nTest 2: With full system prompt including tools")
full_system_prompt = """You are an AI agent playing a text-based adventure game who uses extreme long chains of thought to carefully plan your actions and predict their outcomes. Your goal is to follow the objective described at the start of the game. You interact with the world by providing text commands and predicting their outcomes.

You should:
1. Enclose your thoughts and internal monologue inside <think> </think> tags. Use extremely long chains of thought to carefully consider the game state, your objectives, and the likely outcomes of your actions.
2. Generate a memory summary inside <memory> </memory> tags that captures key information from this turn.
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
            "description": "What you expect to observe after executing this command."
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

For your function call, return a JSON object with function name and arguments within <tool_call> </tool_call> tags."""

test_messages_full = [
    {"role": "system", "content": full_system_prompt},
    {"role": "user", "content": "Objective: Cook a meal.\n\nCurrent Location: Kitchen"}
]

print(f"Messages: {len(test_messages_full)} messages")
try:
    prompt = tokenizer.apply_chat_template(test_messages_full, tokenize=False, add_generation_prompt=True)
    print("✓ Success with full prompt!")
    print(f"Prompt length: {len(prompt)} chars")
except Exception as e:
    print(f"✗ Failed with error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Test with alternating messages
print("\n\nTest 3: With alternating user/assistant messages")
alternating_messages = [
    {"role": "system", "content": "You are a helpful AI."},
    {"role": "user", "content": "First turn"},
    {"role": "assistant", "content": "First response"},
    {"role": "user", "content": "Second turn"}
]

print(f"Messages: {len(alternating_messages)} messages")
try:
    prompt = tokenizer.apply_chat_template(alternating_messages, tokenize=False, add_generation_prompt=True)
    print("✓ Success with alternating messages!")
    print(f"Prompt length: {len(prompt)} chars")
except Exception as e:
    print(f"✗ Failed with error: {type(e).__name__}: {e}")

# Check tokenizer attributes
print("\n\nTokenizer attributes:")
print(f"  tokenizer type: {type(tokenizer).__name__}")
print(f"  has chat_template: {hasattr(tokenizer, 'chat_template')}")
if hasattr(tokenizer, 'chat_template'):
    print(f"  chat_template type: {type(tokenizer.chat_template)}")
    print(f"  chat_template preview: {str(tokenizer.chat_template)[:100]}...")