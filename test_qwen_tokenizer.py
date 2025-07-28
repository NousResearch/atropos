#!/usr/bin/env python3
"""Test script to isolate Qwen tokenizer chat template issue."""

import os

from transformers import AutoTokenizer

from environments.game_environments.textworld_env.agents.atropos_agent import (
    AtroposAgent,
)
from environments.game_environments.textworld_env.utils.qwen_fixed_tokenizer import (
    QwenFixedTokenizer,
)

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("=== Testing Original Qwen Tokenizer ===")
# Load the original Qwen tokenizer
print("Loading original Qwen tokenizer...")
original_tokenizer = AutoTokenizer.from_pretrained(
    "NousResearch/Hermes-4-Qwen3-14B-1-e3"
)

print("\n=== Testing Fixed Qwen Tokenizer ===")
# Load the fixed Qwen tokenizer
print("Loading fixed Qwen tokenizer...")
fixed_tokenizer = QwenFixedTokenizer("NousResearch/Hermes-4-Qwen3-14B-1-e3")

# Create test messages similar to what AtroposAgent is using
test_messages = [
    {
        "role": "system",
        "content": "You are an AI agent playing a text-based adventure game. You should think step-by-step.",
    },
    {
        "role": "user",
        "content": "Objective: Cook a meal.\n\nCurrent Location: Kitchen\nYou see a fridge and a stove.",
    },
]

print("\nTest 1: Basic messages (both tokenizers)")
print(f"Messages: {len(test_messages)} messages")
for i, msg in enumerate(test_messages):
    print(f"  Message {i}: role={msg['role']}, content_length={len(msg['content'])}")

print("\nOriginal tokenizer:")
try:
    prompt = original_tokenizer.apply_chat_template(
        test_messages, tokenize=False, add_generation_prompt=True
    )
    print("✓ Success! Generated prompt:")
    print(f"Prompt length: {len(prompt)} chars")
    print(f"First 200 chars: {prompt[:200]}...")
except Exception as e:
    print(f"✗ Failed with error: {type(e).__name__}: {e}")

print("\nFixed tokenizer:")
try:
    prompt = fixed_tokenizer.apply_chat_template(
        test_messages, tokenize=False, add_generation_prompt=True
    )
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
    {"role": "user", "content": "Objective: Cook a meal.\n\nCurrent Location: Kitchen"},
]

print(f"Messages: {len(test_messages_full)} messages")

print("\nOriginal tokenizer:")
try:
    prompt = original_tokenizer.apply_chat_template(
        test_messages_full, tokenize=False, add_generation_prompt=True
    )
    print("✓ Success with full prompt!")
    print(f"Prompt length: {len(prompt)} chars")
except Exception as e:
    print(f"✗ Failed with error: {type(e).__name__}: {e}")

print("\nFixed tokenizer:")
try:
    prompt = fixed_tokenizer.apply_chat_template(
        test_messages_full, tokenize=False, add_generation_prompt=True
    )
    print("✓ Success with full prompt!")
    print(f"Prompt length: {len(prompt)} chars")
except Exception as e:
    print(f"✗ Failed with error: {type(e).__name__}: {e}")

# Test with alternating messages
print("\n\nTest 3: With alternating user/assistant messages")
alternating_messages = [
    {"role": "system", "content": "You are a helpful AI."},
    {"role": "user", "content": "First turn"},
    {"role": "assistant", "content": "First response"},
    {"role": "user", "content": "Second turn"},
]

print(f"Messages: {len(alternating_messages)} messages")

print("\nOriginal tokenizer:")
try:
    prompt = original_tokenizer.apply_chat_template(
        alternating_messages, tokenize=False, add_generation_prompt=True
    )
    print("✓ Success with alternating messages!")
    print(f"Prompt length: {len(prompt)} chars")
except Exception as e:
    print(f"✗ Failed with error: {type(e).__name__}: {e}")

print("\nFixed tokenizer:")
try:
    prompt = fixed_tokenizer.apply_chat_template(
        alternating_messages, tokenize=False, add_generation_prompt=True
    )
    print("✓ Success with alternating messages!")
    print(f"Prompt length: {len(prompt)} chars")
except Exception as e:
    print(f"✗ Failed with error: {type(e).__name__}: {e}")

# Test with tools parameter
print("\n\nTest 4: With tools parameter (THIS IS THE MAIN TEST)")
test_messages_with_tools = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant playing a text adventure game. Think step-by-step and then call the required tool.",
    },
    {
        "role": "user",
        "content": "You are in a kitchen. There's a stove, a fridge, and a table here.\n\nWhat would you like to do?",
    },
]

print("\n--- Testing ORIGINAL tokenizer with tools ---")
try:
    prompt = original_tokenizer.apply_chat_template(
        test_messages_with_tools,
        tools=AtroposAgent.TOOLS,
        tokenize=False,
        add_generation_prompt=True,
    )
    print("✓ ORIGINAL tokenizer: Success with tools!")
    print(f"Prompt length: {len(prompt)} chars")
    print(f"First 500 chars: {prompt[:500]}...")
except Exception as e:
    print(f"✗ ORIGINAL tokenizer: Failed with error: {type(e).__name__}: {e}")

print("\n--- Testing FIXED tokenizer with tools ---")
try:
    prompt = fixed_tokenizer.apply_chat_template(
        test_messages_with_tools,
        tools=AtroposAgent.TOOLS,
        tokenize=False,
        add_generation_prompt=True,
    )
    print("✓ FIXED tokenizer: Success with tools!")
    print(f"Prompt length: {len(prompt)} chars")
    print(f"First 500 chars: {prompt[:500]}...")

    # Verify tools are in the prompt
    if '"execute_command"' in prompt:
        print("✓ Tool definition found in prompt")
    else:
        print("✗ Tool definition NOT found in prompt")
except Exception as e:
    print(f"✗ FIXED tokenizer: Failed with error: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()

# Check tokenizer attributes
print("\n\nTokenizer attributes:")
print("Original tokenizer:")
print(f"  tokenizer type: {type(original_tokenizer).__name__}")
print(f"  has chat_template: {hasattr(original_tokenizer, 'chat_template')}")
if hasattr(original_tokenizer, "chat_template"):
    print(f"  chat_template type: {type(original_tokenizer.chat_template)}")
    print(f"  chat_template preview: {str(original_tokenizer.chat_template)[:100]}...")

print("\nFixed tokenizer:")
print(f"  tokenizer type: {type(fixed_tokenizer).__name__}")
print(f"  underlying tokenizer type: {type(fixed_tokenizer.tokenizer).__name__}")
print(f"  has chat_template: {hasattr(fixed_tokenizer.tokenizer, 'chat_template')}")
if hasattr(fixed_tokenizer.tokenizer, "chat_template"):
    print(f"  chat_template type: {type(fixed_tokenizer.tokenizer.chat_template)}")
    print(
        f"  chat_template preview: {str(fixed_tokenizer.tokenizer.chat_template)[:100]}..."
    )
