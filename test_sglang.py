#!/usr/bin/env python
"""Test script to verify SGLang server is working with logprobs."""

import asyncio
import json
from openai import AsyncOpenAI

async def test_sglang():
    # Initialize client pointing to local SGLang server
    client = AsyncOpenAI(
        base_url="http://localhost:30000/v1",
        api_key="dummy"  # SGLang doesn't need a real API key
    )
    
    # Test message
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2? Answer in one word."}
    ]
    
    try:
        # Make a completion request with logprobs
        response = await client.chat.completions.create(
            model="NousResearch/DeepHermes-3-Mistral-24B-Preview",
            messages=messages,
            temperature=0.7,
            max_tokens=50,
            logprobs=True,  # Request logprobs
            top_logprobs=5   # Get top 5 logprobs for each token
        )
        
        print("Response received!")
        print(f"Content: {response.choices[0].message.content}")
        print(f"\nUsage: {response.usage}")
        
        # Check if logprobs are available
        if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
            print("\nLogprobs available: YES")
            logprobs_content = response.choices[0].logprobs.content
            if logprobs_content:
                print(f"Number of tokens with logprobs: {len(logprobs_content)}")
                # Show first token's logprobs
                if len(logprobs_content) > 0:
                    first_token = logprobs_content[0]
                    print(f"\nFirst token: '{first_token.token}'")
                    print(f"Log probability: {first_token.logprob}")
                    print(f"Top alternatives:")
                    for alt in first_token.top_logprobs[:3]:
                        print(f"  - '{alt.token}': {alt.logprob}")
        else:
            print("\nLogprobs available: NO")
            
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(test_sglang())