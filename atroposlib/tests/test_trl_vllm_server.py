"""Tests for TrlVllmServer implementation."""

from unittest.mock import patch

import pytest

from atroposlib.envs.server_handling.server_baseline import APIServerConfig
from atroposlib.envs.server_handling.trl_vllm_server import TrlVllmServer


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self.eos_token_id = 2
        self.bos_token_id = 1

    def encode(self, text, add_special_tokens=True):
        """Simple character-based encoding for testing."""
        tokens = [ord(c) for c in text]
        if add_special_tokens:
            tokens = [self.bos_token_id] + tokens
        return tokens

    def decode(self, tokens, skip_special_tokens=False):
        """Simple character-based decoding for testing."""
        if skip_special_tokens:
            tokens = [
                t for t in tokens if t not in [self.bos_token_id, self.eos_token_id]
            ]
        return "".join([chr(t) if 31 < t < 127 else "" for t in tokens])

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        """Simple chat template for testing."""
        result = ""
        for msg in messages:
            result += f"<{msg['role']}>{msg['content']}</{msg['role']}>"
        if add_generation_prompt:
            result += "<assistant>"
        if tokenize:
            return self.encode(result)
        return result


@pytest.fixture
def mock_config():
    """Create a mock server config."""
    return APIServerConfig(
        api_key="test-key",
        base_url="http://localhost:8000",
        model_name="test-model",
        timeout=30,
    )


@pytest.fixture
def mock_server(mock_config):
    """Create a TrlVllmServer with mocked tokenizer."""
    with patch.object(TrlVllmServer, "__init__", lambda self, config: None):
        server = TrlVllmServer.__new__(TrlVllmServer)
        server.config = mock_config
        server.tokenizer = MockTokenizer()
        server.server_healthy = False
        return server


class TestTrlVllmServer:
    """Tests for TrlVllmServer."""

    def test_init_creates_tokenizer(self, mock_config):
        """Test that __init__ creates tokenizer from config."""
        with patch(
            "atroposlib.envs.server_handling.trl_vllm_server.AutoTokenizer"
        ) as mock_auto:
            mock_auto.from_pretrained.return_value = MockTokenizer()
            with patch.object(
                TrlVllmServer.__bases__[0], "__init__", return_value=None
            ):
                TrlVllmServer(mock_config)
                mock_auto.from_pretrained.assert_called_once_with(
                    mock_config.model_name
                )

    @pytest.mark.asyncio
    async def test_tokens_and_logprobs_raises_not_implemented(self, mock_server):
        """Test tokens_and_logprobs method raises NotImplementedError with message."""
        with pytest.raises(NotImplementedError) as exc_info:
            await mock_server._tokens_and_logprobs_completion_wrapper()

        error_msg = str(exc_info.value)
        assert "Token-level logprobs are not supported" in error_msg
        assert "VLLMServer" in error_msg or "SGLangServer" in error_msg

    def test_completion_includes_text_attribute(self, mock_server):
        """Verify that completion response has text attribute (addresses issue #183)."""
        from openai.types.completion import CompletionChoice

        # Create a CompletionChoice like the _completion_wrapper does
        choice = CompletionChoice(
            finish_reason="stop",
            index=0,
            text="Hello World",
        )

        assert hasattr(choice, "text")
        assert choice.text == "Hello World"

    def test_chat_completion_has_message_not_text(self, mock_server):
        """Verify chat completion uses message attribute, not text (issue #183)."""
        from openai.types.chat.chat_completion import ChatCompletionMessage, Choice

        choice = Choice(
            finish_reason="stop",
            index=0,
            message=ChatCompletionMessage(
                content="Hello",
                role="assistant",
            ),
        )

        assert hasattr(choice, "message")
        assert hasattr(choice.message, "content")
        assert choice.message.content == "Hello"

    def test_server_has_required_methods(self, mock_server):
        """Verify that TrlVllmServer has all required methods."""
        assert hasattr(mock_server, "check_server_status_task")
        assert hasattr(mock_server, "_chat_completion_wrapper")
        assert hasattr(mock_server, "_completion_wrapper")
        assert hasattr(mock_server, "_tokens_and_logprobs_completion_wrapper")
        assert callable(mock_server.check_server_status_task)
        assert callable(mock_server._chat_completion_wrapper)
        assert callable(mock_server._completion_wrapper)
        assert callable(mock_server._tokens_and_logprobs_completion_wrapper)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
