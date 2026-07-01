"""Tests that chat_completion uses a single (3-attempt) retry budget."""

import asyncio

import pytest
from tenacity import RetryError

from atroposlib.envs.server_handling.server_baseline import (
    APIServer,
    APIServerConfig,
)


class _CountingServer(APIServer):
    def __init__(self, config):
        super().__init__(config)
        self.chat_wrapper_calls = 0

    async def _chat_completion_wrapper(self, **kwargs):
        self.chat_wrapper_calls += 1
        raise RuntimeError("boom")

    async def _completion_wrapper(self, **kwargs):
        raise RuntimeError("boom")

    async def _tokens_and_logprobs_completion_wrapper(self, **kwargs):
        raise RuntimeError("boom")

    async def check_server_status_task(self, *args, **kwargs):
        return


async def test_chat_completion_retries_three_times(monkeypatch):
    # Make tenacity's backoff instant so the test is fast/deterministic.
    async def _instant(*args, **kwargs):
        return

    monkeypatch.setattr(asyncio, "sleep", _instant)

    config = APIServerConfig(
        model_name="m", base_url=None, api_key="x", health_check=False
    )
    server = _CountingServer(config)
    server.server_healthy = True
    server.initialized = True

    with pytest.raises(RetryError):
        await server.chat_completion(messages=[{"role": "user", "content": "hi"}])

    # A single 3-attempt retry budget -- not the nested 3 x 3 = 9.
    assert server.chat_wrapper_calls == 3
