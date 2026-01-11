"""
Ollama server handling with native logprobs support.

IMPORTANT: Logprobs are supported via NATIVE Ollama API (/api/chat),
NOT via OpenAI-compatible API (/v1/chat/completions).

API Documentation:
- Native API: https://docs.ollama.com/api/chat
- OpenAI Compatibility: https://docs.ollama.com/api/openai-compatibility
"""

import asyncio
import math
from typing import List, Optional

import aiohttp
from openai.types.chat.chat_completion import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
)
from openai.types.completion import Completion, CompletionChoice, CompletionUsage
from transformers import AutoTokenizer

from atroposlib.envs.server_handling.server_baseline import APIServer, APIServerConfig


class OllamaServerConfig(APIServerConfig):
    """Configuration for Ollama server with additional native API options."""

    use_native_api: bool = True  # Use native /api/chat for logprobs support
    top_logprobs: int = 5  # Number of top alternatives to return


class OllamaServer(APIServer):
    """
    Ollama server handling with native API logprobs support.

    This server uses Ollama's native /api/chat endpoint which properly returns
    logprobs, unlike the OpenAI-compatible /v1/chat/completions endpoint.
    """

    def __init__(self, config: OllamaServerConfig):
        self.config = config
        self.tokenizer = None
        self._tokenizer_name = config.model_name

        # Try to load tokenizer if model name looks like a HuggingFace model
        if "/" in config.model_name:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            except Exception:
                pass

        super().__init__(config)

    def _get_native_url(self) -> str:
        """Get the native Ollama API URL."""
        base = self.config.base_url or "http://localhost:11434"
        # Remove /v1 suffix if present (OpenAI compatibility path)
        base = base.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        return f"{base}/api/chat"

    def _get_generate_url(self) -> str:
        """Get the native Ollama generate API URL."""
        base = self.config.base_url or "http://localhost:11434"
        base = base.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        return f"{base}/api/generate"

    async def check_server_status_task(self, chat_completion: bool = True):
        """Check server health by hitting the Ollama API."""
        while True:
            try:
                base = self.config.base_url or "http://localhost:11434"
                base = base.rstrip("/")
                if base.endswith("/v1"):
                    base = base[:-3]

                async with aiohttp.ClientSession() as session:
                    headers = {}
                    if self.config.api_key:
                        headers["Authorization"] = f"Bearer {self.config.api_key}"

                    # Try the /api/tags endpoint to check if Ollama is running
                    async with session.get(
                        f"{base}/api/tags",
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as response:
                        response.raise_for_status()
                self.server_healthy = True
            except Exception:
                self.server_healthy = False
            await asyncio.sleep(1)

    async def _chat_completion_wrapper(self, **kwargs) -> ChatCompletion:
        """
        Chat completion using Ollama's native API.

        Returns OpenAI-compatible ChatCompletion object.
        """
        messages = kwargs.get("messages", [])
        model = kwargs.get("model", self.config.model_name)
        n = kwargs.get("n", 1)
        max_tokens = kwargs.get("max_tokens", 2048)
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.95)
        stop = kwargs.get("stop", None)

        url = self._get_native_url()
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        # Build native Ollama request
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            },
        }

        if stop:
            payload["options"]["stop"] = stop if isinstance(stop, list) else [stop]

        # Make n requests in parallel for multiple completions
        async def single_request():
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                ) as response:
                    response.raise_for_status()
                    return await response.json()

        if n > 1:
            results = await asyncio.gather(*[single_request() for _ in range(n)])
        else:
            results = [await single_request()]

        # Convert to OpenAI format
        choices = []
        for i, result in enumerate(results):
            message = result.get("message", {})
            content = message.get("content", "")

            # Determine finish reason
            done_reason = result.get("done_reason", "stop")
            if done_reason == "length":
                finish_reason = "length"
            else:
                finish_reason = "stop"

            choices.append(
                Choice(
                    index=i,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=content,
                    ),
                    finish_reason=finish_reason,
                )
            )

        # Build usage info
        prompt_tokens = sum(r.get("prompt_eval_count", 0) for r in results)
        completion_tokens = sum(r.get("eval_count", 0) for r in results)

        return ChatCompletion(
            id="ollama-" + str(hash(str(messages))),
            choices=choices,
            created=0,
            model=model,
            object="chat.completion",
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        )

    async def _completion_wrapper(self, **kwargs) -> Completion:
        """
        Completion using Ollama's native generate API.
        """
        prompt = kwargs.get("prompt", "")
        model = kwargs.get("model", self.config.model_name)
        n = kwargs.get("n", 1)
        max_tokens = kwargs.get("max_tokens", 2048)
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.95)
        stop = kwargs.get("stop", None)

        url = self._get_generate_url()
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            },
        }

        if stop:
            payload["options"]["stop"] = stop if isinstance(stop, list) else [stop]

        async def single_request():
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                ) as response:
                    response.raise_for_status()
                    return await response.json()

        if n > 1:
            results = await asyncio.gather(*[single_request() for _ in range(n)])
        else:
            results = [await single_request()]

        choices = []
        for i, result in enumerate(results):
            text = result.get("response", "")
            done_reason = result.get("done_reason", "stop")
            finish_reason = "length" if done_reason == "length" else "stop"

            choices.append(
                CompletionChoice(
                    index=i,
                    text=text,
                    finish_reason=finish_reason,
                )
            )

        prompt_tokens = sum(r.get("prompt_eval_count", 0) for r in results)
        completion_tokens = sum(r.get("eval_count", 0) for r in results)

        return Completion(
            id="ollama-" + str(hash(prompt)),
            choices=choices,
            created=0,
            model=model,
            object="text_completion",
            usage=CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    async def _tokens_and_logprobs_completion_wrapper(
        self, **kwargs
    ) -> tuple[list, list, list, list]:
        """
        Wrapper for tokens and logprobs completion using Ollama's native API.

        Returns a tuple of (prompt_tokens, output_tokens, output_logprobs, finish_reasons).

        IMPORTANT: This uses Ollama's native /api/chat endpoint with logprobs=True,
        which is the only way to get logprobs from Ollama.
        """
        model = kwargs.get("model", self.config.model_name)
        n = kwargs.get("n", 1)
        max_tokens = kwargs.get("max_tokens", kwargs.get("max_new_tokens", 2048))
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.95)
        stop = kwargs.get("stop", None)
        top_logprobs = kwargs.get("top_logprobs", 5)

        # Handle prompt vs messages
        if "messages" in kwargs:
            messages = kwargs["messages"]
        elif "prompt" in kwargs:
            messages = [{"role": "user", "content": kwargs["prompt"]}]
        elif "input_ids" in kwargs:
            # If input_ids provided, decode to text
            if self.tokenizer:
                prompt_text = self.tokenizer.decode(kwargs["input_ids"])
                messages = [{"role": "user", "content": prompt_text}]
            else:
                raise ValueError(
                    "input_ids provided but no tokenizer available. "
                    "Provide a HuggingFace model name or use messages/prompt."
                )
        else:
            raise ValueError("Either 'messages', 'prompt', or 'input_ids' is required")

        url = self._get_native_url()
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        # Native Ollama API format with logprobs enabled
        payload = {
            "model": model,
            "messages": messages,
            "logprobs": True,
            "top_logprobs": top_logprobs,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            },
        }

        if stop:
            payload["options"]["stop"] = stop if isinstance(stop, list) else [stop]

        async def single_request():
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                ) as response:
                    response.raise_for_status()
                    return await response.json()

        # Make n requests in parallel
        if n > 1:
            results = await asyncio.gather(*[single_request() for _ in range(n)])
        else:
            results = [await single_request()]

        # Tokenize the prompt for return value
        prompt_tokens = []
        if self.tokenizer:
            # Tokenize the full conversation
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_tokens = self.tokenizer.encode(prompt_text)

        output_tokens_list = []
        output_logprobs_list = []
        finish_reasons_list = []

        for result in results:
            message = result.get("message", {})
            content = message.get("content", "")

            # Parse logprobs from native API response
            logprobs_data = result.get("logprobs", [])

            if logprobs_data and self.tokenizer:
                # Extract tokens and logprobs
                tokens = []
                logprobs = []

                for token_info in logprobs_data:
                    token_str = token_info.get("token", "")
                    logprob = token_info.get("logprob", 0.0)

                    # Convert token string to ID if tokenizer available
                    token_ids = self.tokenizer.encode(
                        token_str, add_special_tokens=False
                    )
                    if token_ids:
                        tokens.append(token_ids[0])
                    else:
                        tokens.append(0)

                    logprobs.append(logprob)

                output_tokens_list.append(tokens)
                output_logprobs_list.append(logprobs)
            elif self.tokenizer:
                # Fallback: tokenize content if no logprobs available
                tokens = self.tokenizer.encode(content, add_special_tokens=False)
                output_tokens_list.append(tokens)
                # Use placeholder logprobs (0.0)
                output_logprobs_list.append([0.0] * len(tokens))
            else:
                # No tokenizer available
                output_tokens_list.append([])
                output_logprobs_list.append([])

            # Finish reason
            done_reason = result.get("done_reason", "stop")
            finish_reasons_list.append("length" if done_reason == "length" else "stop")

        return (
            prompt_tokens,
            output_tokens_list,
            output_logprobs_list,
            finish_reasons_list,
        )

    async def chat_completion_with_logprobs(
        self, messages: List[dict], **kwargs
    ) -> tuple[ChatCompletion, List[dict]]:
        """
        Chat completion with logprobs returned separately.

        Returns:
            tuple: (ChatCompletion, list of logprobs dicts per completion)
        """
        model = kwargs.get("model", self.config.model_name)
        n = kwargs.get("n", 1)
        max_tokens = kwargs.get("max_tokens", 2048)
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.95)
        top_logprobs = kwargs.get("top_logprobs", 5)

        url = self._get_native_url()
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        payload = {
            "model": model,
            "messages": messages,
            "logprobs": True,
            "top_logprobs": top_logprobs,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            },
        }

        async def single_request():
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                ) as response:
                    response.raise_for_status()
                    return await response.json()

        if n > 1:
            results = await asyncio.gather(*[single_request() for _ in range(n)])
        else:
            results = [await single_request()]

        choices = []
        all_logprobs = []

        for i, result in enumerate(results):
            message = result.get("message", {})
            content = message.get("content", "")

            done_reason = result.get("done_reason", "stop")
            finish_reason = "length" if done_reason == "length" else "stop"

            choices.append(
                Choice(
                    index=i,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=content,
                    ),
                    finish_reason=finish_reason,
                )
            )

            # Extract logprobs
            logprobs_data = result.get("logprobs", [])
            processed_logprobs = []

            for token_info in logprobs_data:
                token = token_info.get("token", "")
                logprob = token_info.get("logprob", 0.0)
                prob = math.exp(logprob) if logprob else 0.0

                entry = {
                    "token": token,
                    "logprob": logprob,
                    "probability": prob,
                }

                # Add top alternatives if available
                if "top_logprobs" in token_info:
                    entry["top_logprobs"] = [
                        {
                            "token": alt.get("token", ""),
                            "logprob": alt.get("logprob", 0.0),
                            "probability": math.exp(alt.get("logprob", 0.0)),
                        }
                        for alt in token_info["top_logprobs"]
                    ]

                processed_logprobs.append(entry)

            all_logprobs.append(processed_logprobs)

        prompt_tokens = sum(r.get("prompt_eval_count", 0) for r in results)
        completion_tokens = sum(r.get("eval_count", 0) for r in results)

        completion = ChatCompletion(
            id="ollama-" + str(hash(str(messages))),
            choices=choices,
            created=0,
            model=model,
            object="chat.completion",
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        )

        return completion, all_logprobs


if __name__ == "__main__":
    import json
    import os

    async def test_ollama_server():
        """Test the OllamaServer implementation."""
        config = OllamaServerConfig(
            base_url=os.getenv("OLLAMA_BASE_URL", "https://ollama.com"),
            api_key=os.getenv("OLLAMA_API_KEY", ""),
            model_name=os.getenv("OLLAMA_MODEL", "deepseek-r1:7b"),
            timeout=120,
        )

        server = OllamaServer(config)

        print("=" * 60)
        print("Testing OllamaServer")
        print("=" * 60)
        print(f"Base URL: {config.base_url}")
        print(f"Model: {config.model_name}")

        # Test chat completion with logprobs
        print("\n--- Testing chat_completion_with_logprobs ---")
        try:
            messages = [{"role": "user", "content": "What is 2 + 2?"}]
            completion, logprobs = await server.chat_completion_with_logprobs(
                messages=messages, max_tokens=50, temperature=0.7
            )

            print(f"Response: {completion.choices[0].message.content}")
            print(f"Finish reason: {completion.choices[0].finish_reason}")
            print(f"\nLogprobs ({len(logprobs[0])} tokens):")
            for i, lp in enumerate(logprobs[0][:10]):  # Show first 10
                print(
                    f"  Token {i}: '{lp['token']}' logprob={lp['logprob']:.4f} prob={lp['probability']:.4f}"
                )

        except Exception as e:
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()

        # Test tokens_and_logprobs_completion
        print("\n--- Testing tokens_and_logprobs_completion ---")
        try:
            prompt_tokens, output_tokens, output_logprobs, finish_reasons = (
                await server.tokens_and_logprobs_completion(
                    messages=[{"role": "user", "content": "Hello!"}],
                    max_tokens=20,
                    n=2,
                )
            )

            print(f"Prompt tokens: {len(prompt_tokens)}")
            print(f"Number of completions: {len(output_tokens)}")
            for i, (tokens, logprobs, reason) in enumerate(
                zip(output_tokens, output_logprobs, finish_reasons)
            ):
                print(f"\nCompletion {i}:")
                print(f"  Tokens: {len(tokens)}")
                print(f"  Logprobs: {logprobs[:5]}...")
                print(f"  Finish reason: {reason}")

        except Exception as e:
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()

    asyncio.run(test_ollama_server())
