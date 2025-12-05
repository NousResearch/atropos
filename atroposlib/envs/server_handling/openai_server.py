import asyncio
import warnings

import aiohttp
import openai
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.completion import Completion

from atroposlib.envs.server_handling.config_utils import (
    resolve_openai_configs,
)
from atroposlib.envs.server_handling.server_baseline import APIServer, APIServerConfig

__all__ = ["OpenAIServer", "resolve_openai_configs"]


class OpenAIServer(APIServer):
    """
    OpenAI server handling.
    """

    def __init__(self, config: APIServerConfig):
        self.openai = openai.AsyncClient(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
        )
        super().__init__(config)

    async def check_server_status_task(self, chat_completion: bool = True):
        while True:
            try:
                if chat_completion:
                    await self.openai.chat.completions.create(
                        model=self.config.model_name,
                        messages=[{"role": "user", "content": "hi"}],
                        max_tokens=1,
                    )
                else:
                    await self.openai.completions.create(
                        model=self.config.model_name,
                        prompt="hi",
                        max_tokens=1,
                    )
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
        Wrapper for the tokens and logprobs completion using the openai client.
        """
        raise NotImplementedError(
            "Tokens and logprobs not supported by base OpenAI API, use specific API servers."
        )
