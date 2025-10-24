"""Tests for ManagedServer tracking of sequences with tokens and logprobs."""

import pytest

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
            # Filter out special tokens
            tokens = [
                t for t in tokens if t not in [self.bos_token_id, self.eos_token_id]
            ]
        return "".join([chr(t) if t > 31 else "" for t in tokens])

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True
    ):
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
    # Add config for compatibility
    class Config:
        model_name = "test_model"

    server.config = Config()
    return server


@pytest.mark.asyncio
async def test_single_completion(mock_server):
    """Test single completion tracking."""
    managed = ManagedServer(mock_server, tokenizer=mock_server.tokenizer)

    prompt = "Hello"
    prompt_tokens = mock_server.tokenizer.encode(prompt)
    output_text = " World"
    output_tokens = [ord(c) for c in output_text]  # Don't include BOS
    output_logprobs = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6]

    # Set up mock response
    mock_server.set_tokens_and_logprobs_response(
        prompt=prompt,
        prompt_tokens=prompt_tokens,
        output_tokens_list=[output_tokens],
        output_logprobs_list=[output_logprobs],
        finish_reasons=["stop"],
    )

    # Call managed server
    result = await managed.completion(prompt=prompt)

    # Check response
    assert result.choices[0].text == " World"
    assert result.choices[0].finish_reason == "stop"

    # Check tracked state (default mode uses nodes list)
    state = managed.get_state()
    assert len(state["nodes"]) == 1

    # Get the sequence node
    node = state["nodes"][0]
    full_text = prompt + output_text

    # Check structure
    assert node.full_text == full_text
    assert len(node.tokens) == len(prompt_tokens) + len(output_tokens)

    # Check masking: prompt should be -100, completion should have actual tokens
    prompt_len = len(prompt_tokens)
    assert all(t == -100 for t in node.masked_tokens[:prompt_len])
    assert node.masked_tokens[prompt_len:] == output_tokens

    # Check logprobs: prompt should be 0.0, completion should have actual logprobs
    assert all(lp == 0.0 for lp in node.logprobs[:prompt_len])
    assert node.logprobs[prompt_len:] == output_logprobs


@pytest.mark.asyncio
async def test_chat_completion(mock_server):
    """Test chat completion with message conversion."""
    managed = ManagedServer(mock_server, tokenizer=mock_server.tokenizer)

    messages = [{"role": "user", "content": "Hello"}]
    prompt = managed._convert_messages_to_prompt(messages)
    prompt_tokens = mock_server.tokenizer.encode(prompt)
    output_text = "Hi there!"
    output_tokens = [ord(c) for c in output_text]
    output_logprobs = [-0.1] * len(output_tokens)

    # Set up mock response
    mock_server.set_tokens_and_logprobs_response(
        prompt=prompt,
        prompt_tokens=prompt_tokens,
        output_tokens_list=[output_tokens],
        output_logprobs_list=[output_logprobs],
        finish_reasons=["stop"],
    )

    # Call managed server
    result = await managed.chat_completion(messages=messages)

    # Check response
    assert result.choices[0].message.content == output_text
    assert result.choices[0].message.role == "assistant"

    # Check tracked state
    state = managed.get_state()
    assert len(state["nodes"]) == 1

    # Verify tokens are properly tracked
    node = state["nodes"][0]
    prompt_len = len(prompt_tokens)
    assert all(t == -100 for t in node.masked_tokens[:prompt_len])


@pytest.mark.asyncio
async def test_multi_turn_conversation(mock_server):
    """Test multi-turn conversation with parent node merging."""
    managed = ManagedServer(mock_server, tokenizer=mock_server.tokenizer)

    # Turn 1
    prompt_1 = "Hello"
    prompt_tokens_1 = mock_server.tokenizer.encode(prompt_1)
    output_1 = " World"
    output_tokens_1 = [ord(c) for c in output_1]
    output_logprobs_1 = [-0.1] * len(output_tokens_1)

    mock_server.set_tokens_and_logprobs_response(
        prompt=prompt_1,
        prompt_tokens=prompt_tokens_1,
        output_tokens_list=[output_tokens_1],
        output_logprobs_list=[output_logprobs_1],
        finish_reasons=["stop"],
    )

    await managed.completion(prompt=prompt_1)

    # Turn 2: extends turn 1
    prompt_2 = "Hello World"  # Parent's full_text
    # In a real scenario, prompt_tokens would be from parent node
    full_tokens_from_turn_1 = prompt_tokens_1 + output_tokens_1
    output_2 = "!"
    output_tokens_2 = [ord(c) for c in output_2]
    output_logprobs_2 = [-0.2]

    mock_server.set_tokens_and_logprobs_response(
        prompt=prompt_2,
        prompt_tokens=full_tokens_from_turn_1,  # Use parent's full tokens
        output_tokens_list=[output_tokens_2],
        output_logprobs_list=[output_logprobs_2],
        finish_reasons=["stop"],
    )

    await managed.completion(prompt=prompt_2)

    # Check state - turn 2 extended turn 1, so it should replace that node
    state = managed.get_state()
    assert len(state["nodes"]) == 1

    # Check the extended node
    node = state["nodes"][0]
    assert node.full_text == "Hello World!"

    # Check the extended node has correct tokens
    # Tokens should be: turn_1_full + output_2
    assert len(node.tokens) == len(full_tokens_from_turn_1) + len(output_tokens_2)


@pytest.mark.asyncio
async def test_branching_with_n(mock_server):
    """Test branching when n > 1."""
    managed = ManagedServer(mock_server, tokenizer=mock_server.tokenizer)

    prompt = "Hello"
    prompt_tokens = mock_server.tokenizer.encode(prompt)

    # Three different completions
    output_texts = [" World", " There", " Friend"]
    output_tokens_list = [[ord(c) for c in text] for text in output_texts]
    output_logprobs_list = [[-0.1] * len(tokens) for tokens in output_tokens_list]

    mock_server.set_tokens_and_logprobs_response(
        prompt=prompt,
        prompt_tokens=prompt_tokens,
        output_tokens_list=output_tokens_list,
        output_logprobs_list=output_logprobs_list,
        finish_reasons=["stop", "stop", "stop"],
    )

    result = await managed.completion(prompt=prompt, n=3)

    # Check we got 3 completions
    assert len(result.choices) == 3

    # Check state has 3 nodes (one per branch)
    state = managed.get_state()
    assert len(state["nodes"]) == 3

    # Verify each node has different text
    full_texts = {node.full_text for node in state["nodes"]}
    assert full_texts == {"Hello World", "Hello There", "Hello Friend"}


@pytest.mark.asyncio
async def test_bos_token_handling(mock_server):
    """Test that BOS token is only at the start of sequence."""
    managed = ManagedServer(mock_server, tokenizer=mock_server.tokenizer)

    prompt = "Test"
    # Tokenizer adds BOS
    prompt_tokens = mock_server.tokenizer.encode(prompt)  # [1, 84, 101, 115, 116]
    assert prompt_tokens[0] == mock_server.tokenizer.bos_token_id

    output_text = "ing"
    output_tokens = [ord(c) for c in output_text]  # Should NOT have BOS
    output_logprobs = [-0.1] * len(output_tokens)

    mock_server.set_tokens_and_logprobs_response(
        prompt=prompt,
        prompt_tokens=prompt_tokens,
        output_tokens_list=[output_tokens],
        output_logprobs_list=[output_logprobs],
        finish_reasons=["stop"],
    )

    await managed.completion(prompt=prompt)

    # Check sequence
    state = managed.get_state()
    assert len(state["nodes"]) == 1
    node = state["nodes"][0]

    # Should have exactly one BOS at the start
    assert node.tokens[0] == mock_server.tokenizer.bos_token_id
    # And no BOS in the rest of the sequence
    assert mock_server.tokenizer.bos_token_id not in node.tokens[1:]


@pytest.mark.asyncio
async def test_reset_clears_sequences(mock_server):
    """Test that reset() clears all tracked sequences."""
    managed = ManagedServer(mock_server, tokenizer=mock_server.tokenizer)

    prompt = "Test"
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

    await managed.completion(prompt=prompt)

    # Verify nodes exist
    state = managed.get_state()
    assert len(state["nodes"]) > 0

    # Reset
    managed.reset()

    # Verify nodes are cleared
    state = managed.get_state()
    assert len(state["nodes"]) == 0


@pytest.mark.asyncio
async def test_tokenizer_initialization_from_server(mock_server):
    """Test that tokenizer is initialized from server if available."""
    managed = ManagedServer(mock_server)  # Don't pass tokenizer

    # Should have gotten tokenizer from server
    assert managed.tokenizer is not None
    assert managed.tokenizer == mock_server.tokenizer


@pytest.mark.asyncio
async def test_input_ids_extension(mock_server):
    """Test that input_ids are computed correctly when extending sequences."""
    managed = ManagedServer(mock_server, tokenizer=mock_server.tokenizer)

    # Turn 1
    prompt_1 = "Hello"
    prompt_tokens_1 = mock_server.tokenizer.encode(prompt_1)  # [1, 72, 101, 108, 108, 111]
    output_1 = " World"
    output_tokens_1 = [ord(c) for c in output_1]
    output_logprobs_1 = [-0.1] * len(output_tokens_1)

    mock_server.set_tokens_and_logprobs_response(
        prompt=prompt_1,
        prompt_tokens=prompt_tokens_1,
        output_tokens_list=[output_tokens_1],
        output_logprobs_list=[output_logprobs_1],
        finish_reasons=["stop"],
    )

    await managed.completion(prompt=prompt_1)

    # Turn 2: extends turn 1 with new text
    prompt_2 = "Hello World!"  # Extends "Hello World" with "!"
    # The input_ids should be: existing_node_tokens + tokenize("!")
    node_1 = managed.current_nodes[0]
    expected_input_ids = node_1.tokens + mock_server.tokenizer.encode("!", add_special_tokens=False)

    output_2 = " Yay"
    output_tokens_2 = [ord(c) for c in output_2]
    output_logprobs_2 = [-0.2] * len(output_tokens_2)

    # The server should receive the computed input_ids
    mock_server.set_tokens_and_logprobs_response(
        prompt=prompt_2,
        prompt_tokens=expected_input_ids,  # Should match what ManagedServer computes!
        output_tokens_list=[output_tokens_2],
        output_logprobs_list=[output_logprobs_2],
        finish_reasons=["stop"],
    )

    result = await managed.completion(prompt=prompt_2)

    # Verify the response
    assert result.choices[0].text == " Yay"

    # Verify we have 1 node (turn 2 replaced turn 1 since it extended)
    state = managed.get_state()
    assert len(state["nodes"]) == 1

    # Verify the node has the correct combined tokens
    node = state["nodes"][0]
    assert node.tokens == expected_input_ids + output_tokens_2


@pytest.mark.asyncio
async def test_multi_turn_chat_with_branching(mock_server):
    """
    Test complex multi-turn scenario:
    - Turn 1: n=8 group completion → 8 nodes (1 assistant turn each)
    - Turn 2: 8 individual calls extending each → extends those 8 nodes (2 assistant turns each)
    - Turn 3: Add system prompt, 8 calls → 8 NEW nodes (different context, 3 assistant turns each)
    Final: 16 nodes total
    """
    managed = ManagedServer(mock_server, tokenizer=mock_server.tokenizer)

    # Turn 1: Single completion with n=8
    messages_1 = [{"role": "user", "content": "Hello"}]
    prompt_1 = managed._convert_messages_to_prompt(messages_1)
    prompt_tokens_1 = mock_server.tokenizer.encode(prompt_1)

    # Create 8 different responses
    responses_1 = [f"Response{i}" for i in range(8)]
    output_tokens_1 = [[ord(c) for c in resp] for resp in responses_1]
    output_logprobs_1 = [[-0.1] * len(tokens) for tokens in output_tokens_1]

    mock_server.set_tokens_and_logprobs_response(
        prompt=prompt_1,
        prompt_tokens=prompt_tokens_1,
        output_tokens_list=output_tokens_1,
        output_logprobs_list=output_logprobs_1,
        finish_reasons=["stop"] * 8,
    )

    await managed.chat_completion(messages=messages_1, n=8)

    # After turn 1: should have 8 nodes
    state = managed.get_state()
    assert len(state["nodes"]) == 8

    # Save references to turn 1 nodes for later verification
    turn_1_nodes = [node.full_text for node in state["nodes"]]

    # Turn 2: For each of the 8 nodes, extend with another user+assistant turn
    for i in range(8):
        messages_2 = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": responses_1[i]},
            {"role": "user", "content": "Continue"},
        ]
        prompt_2 = managed._convert_messages_to_prompt(messages_2)

        # This prompt extends turn 1's output, so input_ids should use existing tokens
        extending_node = state["nodes"][i]
        # The new part is just the user turn
        new_suffix = prompt_2[len(extending_node.full_text):]
        expected_input_ids = extending_node.tokens + mock_server.tokenizer.encode(
            new_suffix, add_special_tokens=False
        )

        response_2 = f"Continued{i}"
        output_tokens_2 = [ord(c) for c in response_2]
        output_logprobs_2 = [-0.2] * len(output_tokens_2)

        mock_server.set_tokens_and_logprobs_response(
            prompt=prompt_2,
            prompt_tokens=expected_input_ids,
            output_tokens_list=[output_tokens_2],
            output_logprobs_list=[output_logprobs_2],
            finish_reasons=["stop"],
        )

        await managed.chat_completion(messages=messages_2, n=1)

    # After turn 2: still 8 nodes (they were extended/replaced, not added)
    state = managed.get_state()
    assert len(state["nodes"]) == 8

    # Verify turn 2 nodes have 2 assistant turns each
    for i in range(8):
        node = state["nodes"][i]
        assert f"Response{i}" in node.full_text
        assert f"Continued{i}" in node.full_text

    # Turn 3: Add system prompt at start - this creates a DIFFERENT context
    # These won't extend because the prefix is different (system prompt added)
    for i in range(8):
        messages_3 = [
            {"role": "system", "content": "Helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": responses_1[i]},
            {"role": "user", "content": "Continue"},
            {"role": "assistant", "content": f"Continued{i}"},
            {"role": "user", "content": "More"},
        ]
        prompt_3 = managed._convert_messages_to_prompt(messages_3)
        prompt_tokens_3 = mock_server.tokenizer.encode(prompt_3)

        response_3 = f"More{i}"
        output_tokens_3 = [ord(c) for c in response_3]
        output_logprobs_3 = [-0.3] * len(output_tokens_3)

        mock_server.set_tokens_and_logprobs_response(
            prompt=prompt_3,
            prompt_tokens=prompt_tokens_3,
            output_tokens_list=[output_tokens_3],
            output_logprobs_list=[output_logprobs_3],
            finish_reasons=["stop"],
        )

        await managed.chat_completion(messages=messages_3, n=1)

    # After turn 3: 8 (turn 2 nodes) + 8 (turn 3 new context) = 16 nodes total!
    state = managed.get_state()
    assert len(state["nodes"]) == 16

    # Verify structure:
    # First 8 nodes: 2 assistant turns (no system prompt)
    for i in range(8):
        node = state["nodes"][i]
        assert "Helpful" not in node.full_text  # No system prompt
        assert f"Response{i}" in node.full_text
        assert f"Continued{i}" in node.full_text
        assert f"More{i}" not in node.full_text  # Not the third turn

    # Last 8 nodes: 3 assistant turns (with system prompt)
    for i in range(8, 16):
        node = state["nodes"][i]
        actual_i = i - 8
        assert "Helpful" in node.full_text  # Has system prompt
        assert f"Response{actual_i}" in node.full_text
        assert f"Continued{actual_i}" in node.full_text
        assert f"More{actual_i}" in node.full_text  # Has third turn


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
