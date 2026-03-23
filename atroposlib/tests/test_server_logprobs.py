"""Tests for get_logprobs wrappers and server-manager routing."""

import pytest

from atroposlib.envs.server_handling.server_baseline import (
    APIServer,
    APIServerConfig,
    AsyncSemWithAdaptiveWeight,
)
from atroposlib.envs.server_handling.server_manager import ServerManager


class _FakeAPIServer(APIServer):
    def __init__(self, config: APIServerConfig):
        super().__init__(config=config, reasoning_config=None)
        self.calls = 0
        self.last_kwargs = None

    async def check_server_status_task(self, chat_completion: bool = True):
        self.server_healthy = True

    async def _chat_completion_wrapper(self, **kwargs):
        raise NotImplementedError

    async def _completion_wrapper(self, **kwargs):
        raise NotImplementedError

    async def _tokens_and_logprobs_completion_wrapper(self, **kwargs):
        raise NotImplementedError

    async def _get_logprobs_wrapper(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        prompt = kwargs.get("prompt", "")
        prompt_tokens = [ord(c) for c in prompt]
        return {
            "prompt_tokens": prompt_tokens,
            "prompt_topk_token_ids": [[t] for t in prompt_tokens],
            "prompt_topk_logprobs": [[-0.1] for _ in prompt_tokens],
        }


class _FakeRoutedServer:
    def __init__(
        self, name: str, train_slots: int, eval_slots: int, healthy: bool = True
    ):
        self.name = name
        self.server_healthy = healthy
        self.sem = AsyncSemWithAdaptiveWeight(4)
        self.eval_sem = AsyncSemWithAdaptiveWeight(4)
        self.sem._value = train_slots
        self.eval_sem._value = eval_slots
        self.calls = 0

    async def get_logprobs(self, **kwargs):
        self.calls += 1
        return {
            "server": self.name,
            "prompt_tokens": [1],
            "prompt_topk_token_ids": [[1]],
            "prompt_topk_logprobs": [[-0.1]],
        }


@pytest.mark.asyncio
async def test_apiserver_get_logprobs_train_eval_wrappers():
    cfg = APIServerConfig(
        model_name="test-model",
        base_url="",
        health_check=False,
    )
    server = _FakeAPIServer(cfg)

    train_out = await server.get_logprobs(prompt="hi", split="train")
    assert train_out["prompt_tokens"] == [ord("h"), ord("i")]
    assert server.calls == 1
    assert server.last_kwargs["model"] == "test-model"
    assert len(server.request_timings) == 1
    assert len(server.attempts_list) == 1
    assert len(server.eval_request_timings) == 0
    assert len(server.eval_attempts_list) == 0

    eval_out = await server.get_logprobs(prompt="ok", split="eval")
    assert eval_out["prompt_tokens"] == [ord("o"), ord("k")]
    assert server.calls == 2
    assert len(server.eval_request_timings) == 1
    assert len(server.eval_attempts_list) == 1


@pytest.mark.asyncio
async def test_server_manager_get_logprobs_routes_to_most_available_server():
    s1 = _FakeRoutedServer("s1", train_slots=1, eval_slots=4, healthy=True)
    s2 = _FakeRoutedServer("s2", train_slots=3, eval_slots=1, healthy=True)
    s3 = _FakeRoutedServer("s3", train_slots=4, eval_slots=4, healthy=False)

    manager = ServerManager.__new__(ServerManager)
    manager.servers = [s1, s2, s3]

    out_train = await ServerManager.get_logprobs(manager, prompt="x", split="train")
    assert out_train["server"] == "s2"
    assert s2.calls == 1

    out_eval = await ServerManager.get_logprobs(manager, prompt="x", split="eval")
    assert out_eval["server"] == "s1"
    assert s1.calls == 1
