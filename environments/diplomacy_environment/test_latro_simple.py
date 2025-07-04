#!/usr/bin/env python3
"""Simple test for LaTRo rewards without running full game."""

import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from atroposlib.envs.base import APIServerConfig
from atroposlib.envs.server_handling.server_manager import ServerManager


async def test_latro_logprobs():
    """Test if we can get logprobs from Ollama."""
    print("\n=== Testing LaTRo Logprobs with Ollama ===\n")

    # Create server config for Ollama
    server_configs = [
        APIServerConfig(
            model_name="mistral-small3.1:latest",  # Use the exact model name
            base_url="http://localhost:11434",  # No /v1 for Ollama
            api_key="dummy",
            server_type="openai",  # Use OpenAI compatibility
        )
    ]

    # Create server manager
    server = ServerManager(server_configs)

    # Test messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    try:
        # Test chat completion with logprobs
        print("Testing chat_completion with logprobs...")
        response = await server.chat_completion(
            messages=messages,
            n=1,
            max_tokens=10,
            temperature=0.7,
            model="mistral-small3.1:latest",
            logprobs=True,
            top_logprobs=5,
        )

        print(f"Response: {response.choices[0].message.content}")

        # Check logprobs
        if hasattr(response.choices[0], "logprobs") and response.choices[0].logprobs:
            print("✓ Logprobs available!")
            if response.choices[0].logprobs.content:
                print(f"  Tokens: {len(response.choices[0].logprobs.content)}")
                # Show first few tokens
                for i, token_data in enumerate(
                    response.choices[0].logprobs.content[:3]
                ):
                    if hasattr(token_data, "token"):
                        print(
                            f"  Token {i}: '{token_data.token}' logprob={token_data.logprob:.3f}"
                        )

        else:
            print("✗ No logprobs returned from chat API")

    except Exception as e:
        print(f"Chat completion error: {e}")
        print("\nTrying completion API with tokenizer...")

        # Test with completion API
        try:
            from transformers import AutoTokenizer

            # Load a tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.1"
            )
            tokenizer.pad_token = tokenizer.eos_token

            # Format messages
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            print(f"Formatted prompt: {prompt[:100]}...")

            response = await server.completion(
                prompt=prompt,
                n=1,
                max_tokens=10,
                temperature=0.7,
                model="mistral-small3.1:latest",
                logprobs=5,
                echo=False,
            )

            print(f"Response: {response.choices[0].text}")

            if (
                hasattr(response.choices[0], "logprobs")
                and response.choices[0].logprobs
            ):
                print("✓ Logprobs available from completion API!")
                if hasattr(response.choices[0].logprobs, "token_logprobs"):
                    logprobs = [
                        lp
                        for lp in response.choices[0].logprobs.token_logprobs
                        if lp is not None
                    ]
                    print(f"  Got {len(logprobs)} logprobs")
                    print(f"  Sum of logprobs: {sum(logprobs):.3f}")
            else:
                print("✗ No logprobs from completion API either")

        except Exception as e2:
            print(f"Completion API error: {e2}")


async def main():
    try:
        await test_latro_logprobs()
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
