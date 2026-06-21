from atroposlib.utils.tool_call_parser import extract_tool_call, parse_tool_call

TOOLS = [{"name": "web_search"}, {"name": "send_message"}]


def test_plain_json_tool_call():
    resp = '<tool_call>{"name": "web_search", "arguments": {"query": "weather today"}}</tool_call>'
    name, args, is_error = parse_tool_call(resp, available_tools=TOOLS)
    assert not is_error
    assert name == "web_search"
    assert args == {"query": "weather today"}


def test_apostrophe_in_value_is_preserved():
    # Valid JSON whose string value contains an apostrophe must still parse.
    resp = '<tool_call>{"name": "web_search", "arguments": {"query": "what\'s the weather"}}</tool_call>'
    name, args, is_error = parse_tool_call(resp, available_tools=TOOLS)
    assert not is_error
    assert name == "web_search"
    assert args == {"query": "what's the weather"}


def test_possessive_in_value_is_preserved():
    resp = '<tool_call>{"name": "send_message", "arguments": {"text": "the user\'s file is ready"}}</tool_call>'
    name, args, is_error = parse_tool_call(resp, available_tools=TOOLS)
    assert not is_error
    assert args == {"text": "the user's file is ready"}


def test_single_quoted_dict_still_parses():
    # Python-style single-quoted dicts remain supported via the literal-eval fallback.
    resp = "<tool_call>{'name': 'web_search', 'arguments': {'query': 'weather today'}}</tool_call>"
    name, args, is_error = parse_tool_call(resp, available_tools=TOOLS)
    assert not is_error
    assert name == "web_search"
    assert args == {"query": "weather today"}


def test_unknown_tool_is_error():
    resp = '<tool_call>{"name": "no_such_tool", "arguments": {}}</tool_call>'
    name, _, is_error = parse_tool_call(resp, available_tools=TOOLS)
    assert is_error
    assert name == "-ERROR-"


def test_missing_tool_call_is_error():
    name, args, is_error = parse_tool_call("just some text", available_tools=TOOLS)
    assert is_error
    assert name == "-ERROR-"
    assert args == {}


def test_extract_tool_call_returns_inner_content():
    assert extract_tool_call("<tool_call>abc</tool_call>") == "abc"
    assert extract_tool_call("no tags here") is None
