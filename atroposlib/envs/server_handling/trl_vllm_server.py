"""
This is a server that interfaces with trl's vLLM server.

Developed with much help from @winglian when they worked on integrating Atropos into Axolotl.
"""

import time
import uuid
import asyncio

import aiohttp
from openai.types.chat.chat_completion import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
)
from transformers import AutoTokenizer

from atroposlib.envs.server_handling.server_baseline import APIServer, APIServerConfig


class TrlVllmServer(APIServer):
    """
    A server that interfaces with trl's vLLM server.
    """

    def __init__(self, config: APIServerConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        super().__init__(config)

    async def check_server_status_task(self, chat_completion: bool = True):
        """
        Perform a health check for the TRL VLLM server by sending a minimal request to /generate.
        """
        health_check_url = f"{self.config.base_url}/generate/"
        # Minimal payload that the /generate endpoint would accept without erroring.
        # This might need adjustment based on the TRL VLLM server's specific requirements.
        minimal_payload = {
            "prompts": ["test"], # Using a non-empty prompt as some servers might require it
            "max_tokens": 1,
            "n": 1
        }
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(health_check_url, json=minimal_payload, timeout=10) as response: # Added timeout
                        if response.status == 200:
                            # Optionally, further check response content if a specific "healthy" body is expected
                            # For now, status 200 is considered healthy.
                            self.server_healthy = True
                        else:
                            # Log warning for non-200 status if a logger is available/configured
                            # logger.warning(f"TRL VLLM server health check failed: Status {response.status} for {health_check_url}")
                            self.server_healthy = False
            except aiohttp.ClientError as e:
                # Log error for connection issues if a logger is available/configured
                # logger.error(f"TRL VLLM server health check connection error: {e}")
                self.server_healthy = False
            except Exception as e: # Catch any other unexpected errors during the check
                # Log error for unexpected issues if a logger is available/configured
                # logger.error(f"Unexpected error during TRL VLLM server health check: {e}")
                self.server_healthy = False
            await asyncio.sleep(10) # Check periodically (e.g., every 10 seconds)

    async def _chat_completion_wrapper(self, **kwargs) -> ChatCompletion:
        """
        Wrapper for the chat completion using the trl's vLLM server.
        """
        url = f"{self.config.base_url}/generate/"
        prompt = kwargs.get("messages", [])
        prompt = self.tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json={
                    "prompts": [prompt],
                    "n": kwargs.get("n", 1),
                    "repetition_penalty": kwargs.get("repetition_penalty", 1.0),
                    "temperature": kwargs.get("temperature", 1.0),
                    "top_p": kwargs.get("top_p", 1.0),
                    "top_k": kwargs.get("top_k", -1),
                    "min_p": kwargs.get("min_p", 0.0),
                    "max_tokens": kwargs.get("max_tokens", 1024),
                },
            ) as response:
                completions = await response.json()
        completions = ChatCompletion(
            id=str(uuid.uuid4()),
            object="chat.completion",
            created=int(time.time()),
            model=self.config.model_name,
            choices=[
                Choice(
                    finish_reason=(
                        "stop"
                        if self.tokenizer.eos_token_id in completion
                        else "length"
                    ),
                    index=i,
                    message=ChatCompletionMessage(
                        content=self.tokenizer.decode(completion),
                        role="assistant",
                    ),
                )
                for i, completion in enumerate(completions["completion_ids"])
            ],
        )
        return completions

    async def _completion_wrapper(self, **kwargs) -> ChatCompletion:
        """
        Wrapper for the completion using the trl's vLLM server.
        """
        url = f"{self.config.base_url}/generate/"
        prompt = kwargs.get("prompt", "")
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json={
                    "prompts": [prompt],
                    "n": kwargs.get("n", 1),
                    "repetition_penalty": kwargs.get("repetition_penalty", 1.0),
                    "temperature": kwargs.get("temperature", 1.0),
                    "top_p": kwargs.get("top_p", 1.0),
                    "top_k": kwargs.get("top_k", -1),
                    "min_p": kwargs.get("min_p", 0.0),
                    "max_tokens": kwargs.get("max_tokens", 1024),
                },
            ) as response:
                completions = await response.json()
        completions = ChatCompletion(
            id=str(uuid.uuid4()),
            object="chat.completion",
            created=int(time.time()),
            model=self.config.model_name,
            choices=[
                Choice(
                    finish_reason=(
                        "stop"
                        if self.tokenizer.eos_token_id in completion
                        else "length"
                    ),
                    index=i,
                    message=ChatCompletionMessage(
                        content=self.tokenizer.decode(completion),
                        role="assistant",
                    ),
                )
                for i, completion in enumerate(completions["completion_ids"])
            ],
        )
        return completions
