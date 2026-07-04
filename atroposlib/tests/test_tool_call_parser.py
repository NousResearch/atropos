from atroposlib.utils.tool_call_parser import parse_tool_call

def test_parse_tool_call_with_apostrophe_in_argument():
    tools = [{"name": "web_search"}]
    resp = '<tool_call>{"name": "web_search", "arguments": {"query": "what\'s the weather"}}</tool_call>'
    name, args, is_error = parse_tool_call(resp, available_tools=tools)
    assert is_error is False
    assert name == "web_search"
    assert args["query"] == "what's the weather"

def test_parse_tool_call_single_quoted_dict_still_works():
    tools = [{"name": "web_search"}]
    resp = "<tool_call>{'name': 'web_search', 'arguments': {'query': 'weather today'}}</tool_call>"
    name, args, is_error = parse_tool_call(resp, available_tools=tools)
    assert is_error is False
    assert name == "web_search"
    assert args["query"] == "weather today"
