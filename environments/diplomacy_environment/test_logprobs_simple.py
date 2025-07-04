#!/usr/bin/env python3
"""Simple test for logprobs functionality."""

import asyncio
import os

from openai import AsyncOpenAI


async def test_logprobs():
    """Test if we can get logprobs from OpenAI API."""
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print("Testing completion API with logprobs...")

    try:
        # Test 1: Completion API (legacy)
        completion = await client.completions.create(
            model="gpt-3.5-turbo-instruct",  # Completion models only
            prompt="The capital of France is",
            max_tokens=5,
            temperature=0.7,
            logprobs=5,  # Request top 5 logprobs
        )

        print(f"Response: {completion.choices[0].text}")
        if completion.choices[0].logprobs:
            print(f"Logprobs available: Yes")
            print(
                f"Token logprobs: {completion.choices[0].logprobs.token_logprobs[:5]}"
            )
        else:
            print("No logprobs returned")

    except Exception as e:
        print(f"Completion API error: {e}")

    print("\nTesting chat completion API...")

    try:
        # Test 2: Chat Completion API (newer)
        chat_completion = await client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": "The capital of France is"}],
            max_tokens=5,
            temperature=0.7,
            logprobs=True,
            top_logprobs=5,
        )

        print(f"Response: {chat_completion.choices[0].message.content}")
        if (
            hasattr(chat_completion.choices[0], "logprobs")
            and chat_completion.choices[0].logprobs
        ):
            print(f"Logprobs available: Yes")
            # Chat API has different logprobs structure
            if chat_completion.choices[0].logprobs.content:
                print(
                    f"First token logprob: {chat_completion.choices[0].logprobs.content[0].logprob}"
                )
        else:
            print("No logprobs returned for chat API")

    except Exception as e:
        print(f"Chat completion API error: {e}")


if __name__ == "__main__":
    asyncio.run(test_logprobs())
