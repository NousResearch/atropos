import json

from atroposlib.envs.server_handling.tool_call_translator import ToolCallTranslator


class MockTokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def decode(self, tokens, skip_special_tokens=False):
        return "".join(chr(t) for t in tokens if t > 31)


def test_hermes_fallback_parses_tool_call_without_vllm():
    translator = ToolCallTranslator(tokenizer=MockTokenizer(), parser_name="hermes")
    translator.parser = None

    raw = (
        '<tool_call>\n'
        '{"name": "navigate", "arguments": {"url": "https://browserbase.com"}}\n'
        "</tool_call><|im_end|>"
    )

    content, tool_calls, finish_reason = translator.parse_model_output(
        raw,
        tool_choice="auto",
        tools=[{"type": "function", "function": {"name": "navigate"}}],
    )

    assert finish_reason == "tool_calls"
    assert content is None
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "navigate"
    assert json.loads(tool_calls[0].function.arguments) == {
        "url": "https://browserbase.com"
    }


def test_hermes_fallback_returns_raw_text_for_non_tool_output():
    translator = ToolCallTranslator(tokenizer=MockTokenizer(), parser_name="hermes")
    translator.parser = None

    raw = "Plain text final answer"
    content, tool_calls, finish_reason = translator.parse_model_output(
        raw,
        tool_choice="auto",
        tools=[{"type": "function", "function": {"name": "navigate"}}],
    )

    assert finish_reason == "stop"
    assert tool_calls is None
    assert content == raw
