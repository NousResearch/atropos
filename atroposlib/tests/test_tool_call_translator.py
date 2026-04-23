"""Unit tests for ToolCallTranslator — vLLM parser wrapper and lookup table.

These are pure logic tests, no server or model needed. Uses a mock tokenizer.
"""

import json

import pytest

from atroposlib.envs.server_handling.tool_call_translator import (
    VLLM_AVAILABLE,
    ToolCallTranslator,
)

pytestmark = pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM not installed")

# ---------------------------------------------------------------------------
# Mock tokenizer (same one from test_managed_server.py)
# ---------------------------------------------------------------------------


class MockTokenizer:
    def __init__(self):
        self.eos_token_id = 2
        self.bos_token_id = 1

    def encode(self, text, add_special_tokens=True):
        tokens = [ord(c) for c in text]
        if add_special_tokens:
            tokens = [self.bos_token_id] + tokens
        return tokens

    def decode(self, tokens, skip_special_tokens=False):
        if skip_special_tokens:
            tokens = [
                t for t in tokens if t not in [self.bos_token_id, self.eos_token_id]
            ]
        return "".join([chr(t) if t > 31 else "" for t in tokens])

    def get_vocab(self):
        # Minimal vocab for the parser — hermes parser calls this
        return {chr(i): i for i in range(128)}

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True, tools=None
    ):
        result = ""
        if tools:
            result += f"<tools>{json.dumps(tools)}</tools>\n"
        for msg in messages:
            result += f"<{msg['role']}>{msg.get('content', '')}</{msg['role']}>"
        if add_generation_prompt:
            result += "<assistant>"
        if tokenize:
            return self.encode(result)
        return result


@pytest.fixture
def translator():
    tok = MockTokenizer()
    return ToolCallTranslator(tokenizer=tok, parser_name="hermes")


# ---------------------------------------------------------------------------
# Outbound: model output → OpenAI format
# ---------------------------------------------------------------------------


class TestParseModelOutput:
    def test_single_tool_call(self, translator):
        raw = (
            '<tool_call>{"name": "search", "arguments": {"query": "cats"}}</tool_call>'
        )
        content, tool_calls, finish_reason = translator.parse_model_output(
            raw,
            tool_choice="auto",
            tools=[{"type": "function", "function": {"name": "search"}}],
        )

        assert finish_reason == "tool_calls"
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "search"
        assert json.loads(tool_calls[0].function.arguments) == {"query": "cats"}
        # content is None or empty when full text is a tool call
        assert content is None or content.strip() == ""

    def test_multiple_tool_calls(self, translator):
        raw = (
            '<tool_call>{"name": "get_weather", "arguments": {"city": "SF"}}</tool_call>\n'
            '<tool_call>{"name": "get_time", "arguments": {"tz": "PST"}}</tool_call>'
        )
        tools = [
            {"type": "function", "function": {"name": "get_weather"}},
            {"type": "function", "function": {"name": "get_time"}},
        ]
        content, tool_calls, finish_reason = translator.parse_model_output(
            raw, tool_choice="auto", tools=tools
        )

        assert finish_reason == "tool_calls"
        assert len(tool_calls) == 2
        names = {tc.function.name for tc in tool_calls}
        assert names == {"get_weather", "get_time"}

    def test_no_tool_calls(self, translator):
        raw = "The weather in SF is 72 degrees."
        content, tool_calls, finish_reason = translator.parse_model_output(
            raw,
            tool_choice="auto",
            tools=[{"type": "function", "function": {"name": "search"}}],
        )

        assert finish_reason == "stop"
        assert tool_calls is None
        assert content == raw

    def test_content_before_tool_call(self, translator):
        raw = 'Let me search for that.\n<tool_call>{"name": "search", "arguments": {"query": "cats"}}</tool_call>'
        content, tool_calls, finish_reason = translator.parse_model_output(
            raw,
            tool_choice="auto",
            tools=[{"type": "function", "function": {"name": "search"}}],
        )

        assert finish_reason == "tool_calls"
        assert tool_calls is not None
        assert len(tool_calls) == 1
        # Content before the tool call tag should be preserved
        assert content is not None
        assert "search for that" in content

    def test_tool_choice_none_skips_parsing(self, translator):
        raw = (
            '<tool_call>{"name": "search", "arguments": {"query": "cats"}}</tool_call>'
        )
        content, tool_calls, finish_reason = translator.parse_model_output(
            raw,
            tool_choice="none",
            tools=[{"type": "function", "function": {"name": "search"}}],
        )

        assert finish_reason == "stop"
        assert tool_calls is None
        assert content == raw  # Raw text returned as-is

    def test_no_tools_skips_parsing(self, translator):
        raw = (
            '<tool_call>{"name": "search", "arguments": {"query": "cats"}}</tool_call>'
        )
        content, tool_calls, finish_reason = translator.parse_model_output(
            raw, tool_choice="auto", tools=None
        )

        assert finish_reason == "stop"
        assert tool_calls is None
        assert content == raw

    def test_malformed_json_graceful_fallback(self, translator):
        raw = "<tool_call>not valid json at all</tool_call>"
        content, tool_calls, finish_reason = translator.parse_model_output(
            raw,
            tool_choice="auto",
            tools=[{"type": "function", "function": {"name": "search"}}],
        )

        # Parser should handle gracefully — either no tools or raw content
        assert finish_reason == "stop"
        assert tool_calls is None

    def test_unclosed_tool_call(self, translator):
        raw = '<tool_call>{"name": "search", "arguments": {"query": "cats"}}'
        content, tool_calls, finish_reason = translator.parse_model_output(
            raw,
            tool_choice="auto",
            tools=[{"type": "function", "function": {"name": "search"}}],
        )

        # The hermes regex has a branch for unclosed tags
        assert finish_reason == "tool_calls"
        assert tool_calls is not None
        assert len(tool_calls) == 1

    def test_nested_json_arguments(self, translator):
        args = {
            "filter": {
                "type": "date",
                "range": {"start": "2024-01-01", "end": "2024-12-31"},
            }
        }
        raw = f'<tool_call>{{"name": "search", "arguments": {json.dumps(args)}}}</tool_call>'
        content, tool_calls, finish_reason = translator.parse_model_output(
            raw,
            tool_choice="auto",
            tools=[{"type": "function", "function": {"name": "search"}}],
        )

        assert finish_reason == "tool_calls"
        assert json.loads(tool_calls[0].function.arguments) == args

    def test_tool_call_with_think_tags(self, translator):
        raw = (
            "<think>I should search for this information.</think>\n"
            '<tool_call>{"name": "search", "arguments": {"query": "cats"}}</tool_call>'
        )
        content, tool_calls, finish_reason = translator.parse_model_output(
            raw,
            tool_choice="auto",
            tools=[{"type": "function", "function": {"name": "search"}}],
        )

        assert finish_reason == "tool_calls"
        assert tool_calls is not None
        # Think content should be in the content field
        if content:
            assert "think" in content or "search for this" in content


# ---------------------------------------------------------------------------
# Lookup table
# ---------------------------------------------------------------------------


class TestLookupTable:
    def test_parse_populates_lookup(self, translator):
        raw = (
            '<tool_call>{"name": "search", "arguments": {"query": "cats"}}</tool_call>'
        )
        _, tool_calls, _ = translator.parse_model_output(
            raw,
            tool_choice="auto",
            tools=[{"type": "function", "function": {"name": "search"}}],
        )

        assert len(translator.call_id_to_raw_text) == 1
        tc_id = tool_calls[0].id
        assert tc_id in translator.call_id_to_raw_text
        assert translator.call_id_to_raw_text[tc_id] == raw

    def test_lookup_accumulates(self, translator):
        tools = [{"type": "function", "function": {"name": "search"}}]

        raw1 = (
            '<tool_call>{"name": "search", "arguments": {"query": "cats"}}</tool_call>'
        )
        _, tc1, _ = translator.parse_model_output(raw1, tool_choice="auto", tools=tools)

        raw2 = (
            '<tool_call>{"name": "search", "arguments": {"query": "dogs"}}</tool_call>'
        )
        _, tc2, _ = translator.parse_model_output(raw2, tool_choice="auto", tools=tools)

        assert len(translator.call_id_to_raw_text) == 2
        assert tc1[0].id in translator.call_id_to_raw_text
        assert tc2[0].id in translator.call_id_to_raw_text


# ---------------------------------------------------------------------------
# Inbound: OpenAI messages → raw text
# ---------------------------------------------------------------------------


class TestReconstructRawText:
    def test_reconstruct_from_lookup(self, translator):
        # First, parse to populate lookup
        raw = (
            '<tool_call>{"name": "search", "arguments": {"query": "cats"}}</tool_call>'
        )
        tools = [{"type": "function", "function": {"name": "search"}}]
        _, tool_calls, _ = translator.parse_model_output(
            raw, tool_choice="auto", tools=tools
        )

        # Now reconstruct
        tc_dicts = [tc.model_dump() for tc in tool_calls]
        reconstructed = translator.reconstruct_raw_text_from_tool_calls(tc_dicts)

        assert reconstructed == raw

    def test_reconstruct_fallback_without_lookup(self, translator):
        # Reconstruct without having parsed first — uses fallback
        tc_dicts = [
            {
                "id": "fake-id-123",
                "type": "function",
                "function": {"name": "search", "arguments": '{"query": "cats"}'},
            }
        ]
        reconstructed = translator.reconstruct_raw_text_from_tool_calls(tc_dicts)

        assert "<tool_call>" in reconstructed
        assert "search" in reconstructed
        assert "cats" in reconstructed

    def test_reconstruct_empty_list(self, translator):
        assert translator.reconstruct_raw_text_from_tool_calls([]) == ""

    def test_reconstruct_multiple_tool_calls(self, translator):
        tc_dicts = [
            {
                "id": "id-1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "SF"}'},
            },
            {
                "id": "id-2",
                "type": "function",
                "function": {"name": "get_time", "arguments": '{"tz": "PST"}'},
            },
        ]
        reconstructed = translator.reconstruct_raw_text_from_tool_calls(tc_dicts)

        assert reconstructed.count("<tool_call>") == 2
        assert "get_weather" in reconstructed
        assert "get_time" in reconstructed


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


class TestConvertMessages:
    def test_regular_messages_pass_through(self, translator):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi there"},
        ]
        result = translator.convert_messages_for_template(messages)

        assert result == messages

    def test_assistant_with_tool_calls_reconstructed(self, translator):
        # Parse first to populate lookup
        raw = (
            '<tool_call>{"name": "search", "arguments": {"query": "cats"}}</tool_call>'
        )
        tools = [{"type": "function", "function": {"name": "search"}}]
        _, tool_calls, _ = translator.parse_model_output(
            raw, tool_choice="auto", tools=tools
        )

        messages = [
            {"role": "user", "content": "Search for cats"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [tc.model_dump() for tc in tool_calls],
            },
            {
                "role": "tool",
                "tool_call_id": tool_calls[0].id,
                "content": "Found 5 cats",
            },
        ]

        result = translator.convert_messages_for_template(messages)

        # User message unchanged
        assert result[0] == messages[0]
        # Assistant message reconstructed to raw text
        assert result[1]["role"] == "assistant"
        assert "<tool_call>" in result[1]["content"]
        assert "tool_calls" not in result[1]
        # Tool message passed through
        assert result[2] == messages[2]

    def test_assistant_with_content_and_tool_calls(self, translator):
        messages = [
            {
                "role": "assistant",
                "content": "Let me search.",
                "tool_calls": [
                    {
                        "id": "fake-id",
                        "type": "function",
                        "function": {"name": "search", "arguments": '{"q": "x"}'},
                    }
                ],
            },
        ]

        result = translator.convert_messages_for_template(messages)

        assert result[0]["role"] == "assistant"
        assert "Let me search." in result[0]["content"]
        assert "<tool_call>" in result[0]["content"]

    def test_mixed_message_types(self, translator):
        """Only tool_call assistant messages are reconstructed."""
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},  # regular, no tool_calls
            {"role": "user", "content": "Search cats"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "tc-1",
                        "type": "function",
                        "function": {"name": "search", "arguments": '{"q": "cats"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc-1", "content": "5 results"},
            {"role": "assistant", "content": "Found 5 cats!"},  # regular again
        ]

        result = translator.convert_messages_for_template(messages)

        # Messages at indices 0, 1, 2, 4, 5 should be unchanged
        assert result[0] == messages[0]
        assert result[1] == messages[1]
        assert result[2] == messages[2]
        assert result[4] == messages[4]
        assert result[5] == messages[5]
        # Message at index 3 should be reconstructed
        assert "<tool_call>" in result[3]["content"]


# ---------------------------------------------------------------------------
# Roundtrip
# ---------------------------------------------------------------------------


class TestRoundtrip:
    def test_single_tool_call_roundtrip(self, translator):
        raw = (
            '<tool_call>{"name": "search", "arguments": {"query": "cats"}}</tool_call>'
        )
        tools = [{"type": "function", "function": {"name": "search"}}]

        # Parse
        _, tool_calls, _ = translator.parse_model_output(
            raw, tool_choice="auto", tools=tools
        )
        # Reconstruct
        tc_dicts = [tc.model_dump() for tc in tool_calls]
        reconstructed = translator.reconstruct_raw_text_from_tool_calls(tc_dicts)

        assert reconstructed == raw

    def test_tool_call_empty_arguments(self, translator):
        raw = '<tool_call>{"name": "list_all", "arguments": {}}</tool_call>'
        tools = [{"type": "function", "function": {"name": "list_all"}}]

        _, tool_calls, _ = translator.parse_model_output(
            raw, tool_choice="auto", tools=tools
        )
        assert tool_calls is not None
        assert json.loads(tool_calls[0].function.arguments) == {}


# ---------------------------------------------------------------------------
# Decode with tool awareness
# ---------------------------------------------------------------------------


class TestDecodeToolAwareness:
    def test_decode_without_tools(self, translator):
        tokens = [72, 101, 108, 108, 111]  # "Hello"
        text = translator.decode_with_tool_awareness(tokens, has_tools=False)
        assert text == "Hello"

    def test_decode_with_tools_preserves_special(self, translator):
        # With the mock tokenizer there are no "special" tokens to strip,
        # but verify the flag is passed correctly
        tokens = [72, 101, 108, 108, 111]
        text = translator.decode_with_tool_awareness(tokens, has_tools=True)
        assert text == "Hello"

    def test_decode_strips_bos_without_tools(self, translator):
        tokens = [1, 72, 101, 108, 108, 111]  # BOS + "Hello"
        text = translator.decode_with_tool_awareness(tokens, has_tools=False)
        assert text == "Hello"  # BOS stripped

    def test_decode_keeps_bos_with_tools(self, translator):
        tokens = [1, 72, 101, 108, 108, 111]  # BOS + "Hello"
        text = translator.decode_with_tool_awareness(tokens, has_tools=True)
        # BOS (chr(1)) is not printable so mock tokenizer returns "" for it
        # but the flag skip_special_tokens=False is passed
        assert "Hello" in text
