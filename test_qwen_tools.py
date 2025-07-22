#!/usr/bin/env python3
"""Test if the Qwen tokenizer needs tools parameter."""

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("NousResearch/Hermes-4-Qwen3-14B-1-e3")

messages = [
    {"role": "system", "content": "You are a helpful AI."},
    {"role": "user", "content": "Hello"}
]

print("Test 1: With tools=[]")
try:
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True,
        tools=[]  # Empty list instead of None
    )
    print("✓ Success with empty tools list!")
    print(f"Prompt: {prompt[:200]}...")
except Exception as e:
    print(f"✗ Failed: {e}")

print("\nTest 2: With actual tools")
tools = [
    {
        "type": "function",
        "function": {
            "name": "execute_command",
            "description": "Execute a text command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"}
                }
            }
        }
    }
]

try:
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True,
        tools=tools
    )
    print("✓ Success with tools!")
    print(f"Prompt: {prompt[:200]}...")
except Exception as e:
    print(f"✗ Failed: {e}")