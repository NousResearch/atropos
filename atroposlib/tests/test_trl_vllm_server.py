"""Regression tests for TrlVllmServer wrappers (issue #183)."""

from unittest.mock import patch

import pytest
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.completion import Completion

from atroposlib.envs.server_handling.server_baseline import APIServerConfig
from atroposlib.envs.server_handling.trl_vllm_server import TrlVllmServer


class MockTokenizer:
    """Minimal tokenizer stub for wrapper tests."""

    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        prompt = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        if add_generation_prompt:
            prompt += "<assistant>"
        return prompt

    def decode(self, tokens, skip_special_tokens=False):
        return "decoded"


class MockResponse:
    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class MockSession:
    def __init__(self, payload):
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def post(self, *args, **kwargs):
        return MockResponse(self._payload)


@pytest.fixture
def mock_server():
    config = APIServerConfig(
        api_key="x",
        base_url="http://localhost:8000",
        model_name="test-model",
        timeout=30,
    )
    with patch(
        "atroposlib.envs.server_handling.trl_vllm_server.AutoTokenizer"
    ) as mock_auto:
        mock_auto.from_pretrained.return_value = MockTokenizer()
        yield TrlVllmServer(config)


@patch("atroposlib.envs.server_handling.trl_vllm_server.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_completion_wrapper_returns_text_completion(
    mock_session_cls, mock_server
):
    mock_session_cls.return_value = MockSession({"completion_ids": [[1, 2, 3]]})
    result = await mock_server._completion_wrapper(prompt="Hello")
    assert isinstance(result, Completion)
    assert result.choices[0].text == "decoded"


@patch("atroposlib.envs.server_handling.trl_vllm_server.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_chat_completion_wrapper_returns_message(mock_session_cls, mock_server):
    mock_session_cls.return_value = MockSession({"completion_ids": [[1, 2, 3]]})
    result = await mock_server._chat_completion_wrapper(
        messages=[{"role": "user", "content": "Hi"}]
    )
    assert isinstance(result, ChatCompletion)
    assert result.choices[0].message.content == "decoded"
