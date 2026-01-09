from __future__ import annotations

from typing import Any


class AtroposChatCompletionsProxy:
    def __init__(self, server: Any, split: str):
        self._server = server
        self._split = split

    async def create(self, *args, **kwargs):
        if args:
            raise TypeError("Only keyword arguments are supported.")
        messages = kwargs.pop("messages", None)
        if messages is None:
            raise TypeError("Missing required kwarg: messages")
        kwargs.pop("model", None)
        if "max_completion_tokens" in kwargs and "max_tokens" not in kwargs:
            kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
        return await self._server.chat_completion(
            messages=messages,
            split=self._split,
            **kwargs,
        )


class AtroposChatProxy:
    def __init__(self, server: Any, split: str):
        self.completions = AtroposChatCompletionsProxy(server=server, split=split)


class AtroposCompletionsProxy:
    def __init__(self, server: Any, split: str):
        self._server = server
        self._split = split

    async def create(self, *args, **kwargs):
        if args:
            raise TypeError("Only keyword arguments are supported.")
        prompt = kwargs.pop("prompt", None)
        if prompt is None:
            raise TypeError("Missing required kwarg: prompt")
        kwargs.pop("model", None)
        return await self._server.completion(
            prompt=prompt,
            split=self._split,
            **kwargs,
        )


class AtroposOpenAIProxy:
    def __init__(self, server: Any, split: str):
        self.chat = AtroposChatProxy(server=server, split=split)
        self.completions = AtroposCompletionsProxy(server=server, split=split)
