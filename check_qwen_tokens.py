#!/usr/bin/env python3
"""Check Qwen tokenizer special tokens."""

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("NousResearch/Hermes-4-Qwen3-14B-1-e3")

print("Special tokens:")
print(f"  bos_token: {repr(tokenizer.bos_token)} (id: {tokenizer.bos_token_id})")
print(f"  eos_token: {repr(tokenizer.eos_token)} (id: {tokenizer.eos_token_id})")
print(f"  pad_token: {repr(tokenizer.pad_token)} (id: {tokenizer.pad_token_id})")
print(f"  unk_token: {repr(tokenizer.unk_token)} (id: {tokenizer.unk_token_id})")

print("\nAdditional tokens:")
if hasattr(tokenizer, 'im_start_id'):
    print(f"  im_start: {tokenizer.im_start_id}")
if hasattr(tokenizer, 'im_end_id'):
    print(f"  im_end: {tokenizer.im_end_id}")

# Check for common chat tokens
for token in ['<|im_start|>', '<|im_end|>', '<|system|>', '<|user|>', '<|assistant|>']:
    if token in tokenizer.get_vocab():
        print(f"  {token}: {tokenizer.convert_tokens_to_ids(token)}")

print("\nManually constructing a chat prompt:")
messages = [
    {"role": "system", "content": "You are a helpful AI."},
    {"role": "user", "content": "Hello"}
]

# Try to construct manually based on common Qwen format
manual_prompt = ""
for msg in messages:
    role = msg['role']
    content = msg['content']
    manual_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
manual_prompt += "<|im_start|>assistant\n"

print(f"Manual prompt:\n{manual_prompt}")

# Test if it tokenizes correctly
tokens = tokenizer.encode(manual_prompt)
print(f"\nTokenized: {len(tokens)} tokens")
print(f"Decoded back: {tokenizer.decode(tokens)[:200]}...")