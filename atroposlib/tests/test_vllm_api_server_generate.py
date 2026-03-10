"""Optional integration test for example_trainer.vllm_api_server /generate."""

from importlib import import_module

import pytest
from fastapi.testclient import TestClient


@pytest.mark.asyncio
async def test_vllm_api_server_generate_endpoint_optional():
    """
    Validate /generate contract on the custom vLLM API server.

    This test only runs when vLLM is installed.
    """
    pytest.importorskip("vllm")

    module = import_module("example_trainer.vllm_api_server")

    class _FakeLogprob:
        def __init__(self, value: float):
            self.logprob = value

    class _FakeOutput:
        def __init__(self):
            self.text = " world"
            self.finish_reason = "stop"
            self.logprobs = [{11: _FakeLogprob(-0.3)}]
            self.token_ids = [11]

    class _FakeRequestOutput:
        def __init__(self):
            self.prompt = "hello"
            self.prompt_token_ids = [1, 2]
            self.outputs = [_FakeOutput()]

    class _FakeEngine:
        tokenizer = type("Tok", (), {"decode": staticmethod(lambda _: "hello")})()

        def generate(self, *_args, **_kwargs):
            async def _gen():
                yield _FakeRequestOutput()

            return _gen()

    old_engine = module.engine
    module.engine = _FakeEngine()
    try:
        client = TestClient(module.app)
        resp = client.post(
            "/generate",
            json={
                "prompt": "hello",
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": 1,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "text" in body and body["text"] == [" world"]
        assert body["prompt"] == "hello"
        assert body["finish_reasons"] == ["stop"]
        assert "logprobs" in body
        assert "token_ids" in body
        assert "prompt_token_ids" in body
    finally:
        module.engine = old_engine
