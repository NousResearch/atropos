"""Tests for OpenAIServer._chat_completion_wrapper request fan-out."""

import types

from atroposlib.envs.server_handling.openai_server import OpenAIServer


class _FakeCompletion:
    def __init__(self):
        self.choices = [object()]


class _CountingCreate:
    def __init__(self):
        self.calls = 0

    async def __call__(self, **kwargs):
        self.calls += 1
        return _FakeCompletion()


def _fake_server(create, n_kwarg_is_ignored=True):
    server = types.SimpleNamespace()
    server.config = types.SimpleNamespace(n_kwarg_is_ignored=n_kwarg_is_ignored)
    server.openai = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
    )
    return server


async def test_n1_makes_single_request_when_n_ignored():
    create = _CountingCreate()
    server = _fake_server(create)
    result = await OpenAIServer._chat_completion_wrapper(
        server, model="m", messages=[{"role": "user", "content": "hi"}], n=1
    )
    assert create.calls == 1
    assert len(result.choices) == 1


async def test_n_gt_1_fans_out_and_merges_choices_when_n_ignored():
    create = _CountingCreate()
    server = _fake_server(create)
    result = await OpenAIServer._chat_completion_wrapper(
        server, model="m", messages=[{"role": "user", "content": "hi"}], n=3
    )
    assert create.calls == 3
    assert len(result.choices) == 3
