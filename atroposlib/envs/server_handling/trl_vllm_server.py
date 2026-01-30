"""
This is a server that interfaces with trl's vLLM server.

TRL's vLLM server is started via `trl vllm-serve --model <model_name>` and provides
a `/generate/` endpoint for text generation. This server handler adapts Atropos's
API server interface to work with TRL's server.

Developed with much help from @winglian when they worked on integrating Atropos into Axolotl.

Limitations:
    - Token-level logprobs are not available through TRL's server API.
      If you need logprobs, use VLLMServer with a native vLLM server instead.
"""

import asyncio
import time
import uuid

import aiohttp
from openai.types.chat.chat_completion import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
)
from openai.types.completion import Completion, CompletionChoice
from transformers import AutoTokenizer

from atroposlib.envs.server_handling.server_baseline import APIServer, APIServerConfig


class TrlVllmServer(APIServer):
    """
    A server that interfaces with TRL's vLLM server.

    TRL (Transformer Reinforcement Learning) provides a vLLM server via the
    `trl vllm-serve` command. This class adapts that server's API to the
    Atropos APIServer interface.

    Note:
        This server does NOT support token-level logprobs. The TRL server's
        `/generate/` endpoint returns completion token IDs but not their
        associated logprobs. If you need logprobs for training (e.g., for
        PPO or GRPO), use `VLLMServer` with a native vLLM server instead.
    """

    def __init__(self, config: APIServerConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        super().__init__(config)

    async def check_server_status_task(self, chat_completion: bool = True):
        """
        Periodically check the health of the TRL vLLM server.

        This method runs in a loop, checking server availability every second.
        It attempts to make a lightweight request to the server's generate
        endpoint to verify it's responsive.

        Args:
            chat_completion: Unused parameter, kept for API compatibility.
        """
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    # Try to reach the generate endpoint with minimal request
                    # TRL's server doesn't have a dedicated /health endpoint,
                    # so we check if the generate endpoint is responsive
                    async with session.post(
                        f"{self.config.base_url}/generate/",
                        json={
                            "prompts": [""],
                            "n": 1,
                            "max_tokens": 1,
                        },
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as response:
                        # Any response (even error) means server is reachable
                        # A 200 means it's fully operational
                        if response.status < 500:
                            self.server_healthy = True
                        else:
                            self.server_healthy = False
            except (
                aiohttp.ClientError,
                asyncio.TimeoutError,
                Exception,
            ):
                self.server_healthy = False
            await asyncio.sleep(1)

    async def _chat_completion_wrapper(self, **kwargs) -> ChatCompletion:
        """
        Wrapper for chat completion using TRL's vLLM server.

        Converts chat messages to a prompt using the tokenizer's chat template,
        sends the request to TRL's /generate/ endpoint, and returns an
        OpenAI-compatible ChatCompletion object.
        """
        url = f"{self.config.base_url}/generate/"
        messages = kwargs.get("messages", [])
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
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
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            ) as response:
                response.raise_for_status()
                result = await response.json()

        completion = ChatCompletion(
            id=str(uuid.uuid4()),
            object="chat.completion",
            created=int(time.time()),
            model=self.config.model_name,
            choices=[
                Choice(
                    finish_reason=(
                        "stop"
                        if self.tokenizer.eos_token_id in completion_ids
                        else "length"
                    ),
                    index=i,
                    message=ChatCompletionMessage(
                        content=self.tokenizer.decode(
                            completion_ids, skip_special_tokens=True
                        ),
                        role="assistant",
                    ),
                )
                for i, completion_ids in enumerate(result["completion_ids"])
            ],
        )
        return completion

    async def _completion_wrapper(self, **kwargs) -> Completion:
        """
        Wrapper for text completion using TRL's vLLM server.

        Sends a prompt to TRL's /generate/ endpoint and returns an
        OpenAI-compatible Completion object.
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
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            ) as response:
                response.raise_for_status()
                result = await response.json()

        completion = Completion(
            id=str(uuid.uuid4()),
            object="text_completion",
            created=int(time.time()),
            model=self.config.model_name,
            choices=[
                CompletionChoice(
                    finish_reason=(
                        "stop"
                        if self.tokenizer.eos_token_id in completion_ids
                        else "length"
                    ),
                    index=i,
                    text=self.tokenizer.decode(
                        completion_ids, skip_special_tokens=True
                    ),
                )
                for i, completion_ids in enumerate(result["completion_ids"])
            ],
        )
        return completion

    async def _tokens_and_logprobs_completion_wrapper(
        self, **kwargs
    ) -> tuple[list, list, list, list]:
        """
        Token-level logprobs completion - NOT SUPPORTED by TRL's vLLM server.

        TRL's vLLM server (started via `trl vllm-serve`) does not expose
        token-level logprobs through its `/generate/` endpoint. The server
        returns only the generated token IDs, not their associated probabilities.

        If you need token-level logprobs for training algorithms like PPO or GRPO,
        use one of these alternatives:
            - `VLLMServer`: Direct interface to native vLLM server with full
              logprob support via the `/generate` endpoint.
            - `SGLangServer`: Interface to SGLang server with logprob support.

        Raises:
            NotImplementedError: Always raised as this functionality is not
                available through TRL's server API.
        """
        raise NotImplementedError(
            "Token-level logprobs are not supported by TRL's vLLM server. "
            "TRL's /generate/ endpoint returns only completion token IDs, "
            "not their associated logprobs. If you need logprobs for training, "
            "use VLLMServer or SGLangServer with a native vLLM/SGLang server instead."
        )
