"""Tests for AtroposManagedClient - AsyncOpenAI-compatible wrapper for ManagedServer."""

import pytest

from atroposlib.envs.server_handling.atropos_managed_client import (
    AtroposManagedClient,
    ChoiceLogprobs,
    EnhancedChatCompletion,
    LogprobContent,
)
from atroposlib.envs.server_handling.managed_server import ManagedServer
from atroposlib.envs.server_handling.server_harness import ServerHarness


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
        return "".join([chr(t) if t > 31 else "" for t in tokens])

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
def mock_server():
    """Create a mock server with a tokenizer."""
    server = ServerHarness()
    server.tokenizer = MockTokenizer()

    class Config:
        model_name = "test_model"

    server.config = Config()
    return server


@pytest.fixture
def managed_client(mock_server):
    """Create an AtroposManagedClient with mocked server."""
    managed = ManagedServer(mock_server, tokenizer=mock_server.tokenizer)
    return AtroposManagedClient(managed_server=managed, model="test_model")


class TestDataclasses:
    """Test the enhanced dataclasses."""

    def test_logprob_content(self):
        """Test LogprobContent creation."""
        lp = LogprobContent(logprob=-0.5, token="hello", token_id=100)
        assert lp.logprob == -0.5
        assert lp.token == "hello"
        assert lp.token_id == 100

    def test_choice_logprobs(self):
        """Test ChoiceLogprobs structure."""
        content = [
            LogprobContent(logprob=-0.1),
            LogprobContent(logprob=-0.2),
        ]
        logprobs = ChoiceLogprobs(content=content)
        assert len(logprobs.content) == 2
        assert logprobs.content[0].logprob == -0.1


class TestAtroposManagedClient:
    """Test AtroposManagedClient behavior."""

    def test_reset(self, managed_client):
        """Test reset clears ManagedServer state."""
        # Add some state to managed server
        managed_client.managed_server.current_nodes = ["dummy"]

        # Reset should clear it
        managed_client.reset()
        assert len(managed_client.managed_server.current_nodes) == 0

    def test_copy_returns_self(self, managed_client):
        """Test copy returns same instance for state sharing."""
        copied = managed_client.copy()
        assert copied is managed_client

    def test_namespace_structure(self, managed_client):
        """Test client has correct namespace structure like AsyncOpenAI."""
        assert hasattr(managed_client, "chat")
        assert hasattr(managed_client.chat, "completions")
        assert hasattr(managed_client.chat.completions, "create")

    @pytest.mark.asyncio
    async def test_close_is_noop(self, managed_client):
        """Test close() doesn't raise."""
        await managed_client.close()  # Should not raise


class TestChatCompletionCreate:
    """Test the chat.completions.create() method."""

    @pytest.mark.asyncio
    async def test_basic_completion(self, mock_server, managed_client):
        """Test basic chat completion returns enhanced response."""
        messages = [{"role": "user", "content": "Hello"}]
        managed = managed_client.managed_server
        prompt = managed._convert_messages_to_prompt(messages)
        prompt_tokens = mock_server.tokenizer.encode(prompt)

        output_text = "Hi there!"
        output_tokens = [ord(c) for c in output_text]
        output_logprobs = [-0.1] * len(output_tokens)

        mock_server.set_tokens_and_logprobs_response(
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            output_tokens_list=[output_tokens],
            output_logprobs_list=[output_logprobs],
            finish_reasons=["stop"],
        )

        result = await managed_client.chat.completions.create(
            messages=messages,
            max_tokens=100,
        )

        # Should return EnhancedChatCompletion
        assert isinstance(result, EnhancedChatCompletion)
        assert len(result.choices) == 1
        assert result.choices[0].message.content == output_text

        # Should have prompt_token_ids
        assert len(result.prompt_token_ids) == len(prompt_tokens)

        # Should have token_ids on choice
        assert len(result.choices[0].token_ids) == len(output_tokens)
        assert result.choices[0].token_ids == output_tokens

        # Should have logprobs
        assert len(result.choices[0].logprobs.content) == len(output_tokens)
        assert result.choices[0].logprobs.content[0].logprob == -0.1

    @pytest.mark.asyncio
    async def test_max_completion_tokens_param(self, mock_server, managed_client):
        """Test max_completion_tokens is preferred over max_tokens."""
        messages = [{"role": "user", "content": "Hi"}]
        managed = managed_client.managed_server
        prompt = managed._convert_messages_to_prompt(messages)
        prompt_tokens = mock_server.tokenizer.encode(prompt)

        output_tokens = [ord("!")]
        output_logprobs = [-0.1]

        mock_server.set_tokens_and_logprobs_response(
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            output_tokens_list=[output_tokens],
            output_logprobs_list=[output_logprobs],
            finish_reasons=["stop"],
        )

        # Should accept max_completion_tokens (new OpenAI param)
        result = await managed_client.chat.completions.create(
            messages=messages,
            max_completion_tokens=50,
        )

        assert isinstance(result, EnhancedChatCompletion)

    @pytest.mark.asyncio
    async def test_reset_between_rollouts(self, mock_server, managed_client):
        """Test that reset clears state between rollouts."""
        messages = [{"role": "user", "content": "Hello"}]
        managed = managed_client.managed_server
        prompt = managed._convert_messages_to_prompt(messages)
        prompt_tokens = mock_server.tokenizer.encode(prompt)

        output_tokens = [ord("!")]
        output_logprobs = [-0.1]

        mock_server.set_tokens_and_logprobs_response(
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            output_tokens_list=[output_tokens],
            output_logprobs_list=[output_logprobs],
            finish_reasons=["stop"],
        )

        # First rollout
        await managed_client.chat.completions.create(messages=messages, max_tokens=10)
        state = managed_client.managed_server.get_state()
        assert len(state["nodes"]) == 1

        # Reset
        managed_client.reset()
        state = managed_client.managed_server.get_state()
        assert len(state["nodes"]) == 0

        # Setup for second rollout
        mock_server.set_tokens_and_logprobs_response(
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            output_tokens_list=[output_tokens],
            output_logprobs_list=[output_logprobs],
            finish_reasons=["stop"],
        )

        # Second rollout
        await managed_client.chat.completions.create(messages=messages, max_tokens=10)
        state = managed_client.managed_server.get_state()
        assert len(state["nodes"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
