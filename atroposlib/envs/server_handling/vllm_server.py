# This requires a customized vLLM api server
# see example_trainer/vllm_api_server.py for an example

import asyncio
import warnings

import aiohttp
import openai
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.completion import Completion
from transformers import AutoTokenizer

from atroposlib.envs.server_handling.config_utils import (
    resolve_openai_configs,
)
from atroposlib.envs.server_handling.server_baseline import APIServer, APIServerConfig

__all__ = ["VLLMServer", "resolve_openai_configs"]


class VLLMServer(APIServer):
    """
    VLLM server handling.
    """

    def __init__(self, config: APIServerConfig):
        self.openai = openai.AsyncClient(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        super().__init__(config)

    async def check_server_status_task(self, chat_completion: bool = True):
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self.config.base_url.replace('/v1', '')}/health",
                        headers=(
                            {"Authorization": f"Bearer {self.config.api_key}"}
                            if self.config.api_key
                            else {}
                        ),
                        timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                    ) as response:
                        response.raise_for_status()
                self.server_healthy = True
            except (
                aiohttp.ClientError,
                openai.OpenAIError,
                openai.APITimeoutError,
                Exception,
            ):
                self.server_healthy = False
            await asyncio.sleep(1)

    async def _chat_completion_wrapper(self, **kwargs) -> ChatCompletion:
        """
        Wrapper for the chat completion using the openai client.
        """
        assert (
            kwargs.get("model", None) is not None
        ), "Model is required for chat completion!"
        assert (
            kwargs.get("messages", None) is not None
        ), "Messages are required for chat completion!"
        if self.config.n_kwarg_is_ignored:
            n = kwargs.pop("n", 1)
            completion_list = await asyncio.gather(
                *[self.openai.chat.completions.create(**kwargs) for _ in range(n)]
            )
            completions = completion_list[0]
            if n > 1:
                for c in completion_list[1:]:
                    completions.choices.extend(c.choices)
            else:
                completions = await self.openai.chat.completions.create(**kwargs)
        else:
            if "n" in kwargs:
                n = kwargs["n"]
            else:
                n = 1
            completions = await self.openai.chat.completions.create(**kwargs)
            if len(completions.choices) != n:
                if len(completions.choices) != 1:
                    raise ValueError(
                        f"Expected 1 or {n} completions, got {len(completions.choices)}!"
                    )
                else:
                    warnings.warn("n kwarg is ignored by the API, setting to True")
                    self.config.n_kwarg_is_ignored = True
                    completion_list = await asyncio.gather(
                        *[
                            self.openai.chat.completions.create(**kwargs)
                            for _ in range(1, n)
                        ]
                    )
                    for c in completion_list:
                        completions.choices.extend(c.choices)
        return completions

    async def _completion_wrapper(self, **kwargs) -> Completion:
        """
        Wrapper for the completion using the openai client.
        """
        assert (
            kwargs.get("model", None) is not None
        ), "Model is required for completion!"
        assert (
            kwargs.get("prompt", None) is not None
        ), "Prompt is required for completion!"
        if self.config.n_kwarg_is_ignored:
            n = kwargs.pop("n", 1)
            completion_list = await asyncio.gather(
                *[self.openai.completions.create(**kwargs) for _ in range(n)]
            )
            completions = completion_list[0]
            if n > 1:
                for c in completion_list[1:]:
                    completions.choices.extend(c.choices)
        else:
            if "n" in kwargs:
                n = kwargs["n"]
            else:
                n = 1
            completions = await self.openai.completions.create(**kwargs)
            if len(completions.choices) != n:
                if len(completions.choices) != 1:
                    raise ValueError(
                        f"Expected 1 or {n} completions, got {len(completions.choices)}!"
                    )
                else:
                    warnings.warn("n kwarg is ignored by the API, setting to True")
                    self.config.n_kwarg_is_ignored = True
                    completion_list = await asyncio.gather(
                        *[self.openai.completions.create(**kwargs) for _ in range(1, n)]
                    )
                    for c in completion_list:
                        completions.choices.extend(c.choices)
        return completions

    async def _tokens_and_logprobs_completion_wrapper(
        self, **kwargs
    ) -> tuple[list, list, list, list]:
        """
        Wrapper for tokens and logprobs completion using VLLM's native API.
        Returns a tuple of (prompt_tokens, output_tokens, output_logprobs, finish_reasons).
        Each element is a list of lists (one per completion in the batch).
        """
        assert (
            kwargs.get("model", None) is not None
        ), "Model is required for completion!"
        assert (
            kwargs.get("prompt", None) is not None
            or kwargs.get("input_ids", None) is not None
        ), "Prompt or input_ids is required for completion!"

        # Use input_ids if provided (from ManagedServer), otherwise tokenize prompt
        if "input_ids" in kwargs:
            prompt_tokens = kwargs.pop("input_ids")
            kwargs.pop("prompt", None)  # Remove prompt if it exists
        else:
            prompt_tokens = self.tokenizer.encode(kwargs.pop("prompt"))

        # Check for double BOS token, can happen if you use chat templates and forget that they insert a BOS token
        if (
            len(prompt_tokens) >= 2
            and prompt_tokens[0] == self.tokenizer.bos_token_id == prompt_tokens[1]
        ):
            prompt_tokens = prompt_tokens[1:]
        if "max_new_tokens" in kwargs:
            kwargs["max_tokens"] = kwargs.pop("max_new_tokens")
        if "model" in kwargs:
            kwargs.pop("model")
        # Prepare request for VLLM native API
        request_data = {"prompt": {"prompt_token_ids": prompt_tokens}, "logprobs": 0}
        request_data.update(kwargs)

        # Make async request to VLLM /generate endpoint
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.config.base_url.replace('/v1', '')}/generate",
                json=request_data,
                headers=(
                    {"Authorization": f"Bearer {self.config.api_key}"}
                    if self.config.api_key
                    else {}
                ),
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            ) as response:
                response.raise_for_status()
                results = await response.json()
        output_tokens_list = []
        output_logprobs_list = []
        finish_reasons_list = []
        for output_token_logprobs, finish_reason in zip(
            results["logprobs"], results["finish_reasons"]
        ):
            logprobs = [
                list(item[0].values())[0] for item in output_token_logprobs
            ]  # Extract logprob from [{id: logprob}]
            output_ids = [
                int(list(item[0].keys())[0]) for item in output_token_logprobs
            ]  # Extract token ID from [{id: logprob}]

            # Get finish reason
            finish_reason = finish_reason

            output_tokens_list.append(output_ids)
            output_logprobs_list.append(logprobs)
            finish_reasons_list.append(finish_reason)

        return (
            prompt_tokens,
            output_tokens_list,
            output_logprobs_list,
            finish_reasons_list,
        )


if __name__ == "__main__":

    async def test_tokens_and_logprobs():
        # Configure the server - update these values for your setup
        config = APIServerConfig(
            api_key="",  # Add your API key if needed
            base_url="http://localhost:8000",  # Update to your VLLM server URL
            model_name="Qwen/Qwen2.5-7B",  # Update to your model name
            timeout=120,
        )

        server = VLLMServer(config)

        # Test the tokens_and_logprobs_completion method
        print("Testing tokens_and_logprobs_completion...")
        try:
            prompt_tokens, output_tokens, output_logprobs, finish_reasons = (
                await server.tokens_and_logprobs_completion(
                    prompt="The capital of France is",
                    n=4,
                    max_tokens=32,
                    temperature=1.0,
                    top_p=1.0,
                    stop=["User:", "Human:", "Assistant:", "</answer>"],
                )
            )

            print("\nResults:")
            print(f"Prompt tokens: {prompt_tokens}")
            print(f"Output tokens: {output_tokens}")
            print(f"Output logprobs (first 5): {[lp[:5] for lp in output_logprobs]}")
            print(f"Finish reasons: {finish_reasons}")
            print(f"\nNumber of completions: {len(output_tokens)}")
            print(f"Output length: {[len(tokens) for tokens in output_tokens]}")
            responses = "\n\n".join(
                [server.tokenizer.decode(tokens) for tokens in output_tokens]
            )
            print(f"Responses:\n-{responses}")
            print(f"Health: {server.server_healthy}")

        except Exception as e:
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()

    # Run the test
    asyncio.run(test_tokens_and_logprobs())
