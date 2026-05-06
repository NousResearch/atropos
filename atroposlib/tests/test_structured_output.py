import pytest

from atroposlib.utils.structured_output import (
    count_tag_occurrences,
    extract_all_tagged_blocks,
    extract_boxed,
    extract_fenced_block,
    extract_json,
    extract_tagged,
    extract_tagged_or_raw,
    normalize_boxed_answer,
    safe_json_loads,
    split_after_think,
    strip_think_blocks,
    validate_single_think_block,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<think>reasoning</think><answer>42</answer>", True),
        ("prefix <think>reasoning</think> suffix", True),
        ("<think>one</think><think>two</think>", False),
        ("<think>unterminated", False),
    ],
)
def test_validate_single_think_block(text, expected):
    assert validate_single_think_block(text) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "<think>reasoning</think><answer>42</answer>",
            ("<think>reasoning</think>", "<answer>42</answer>"),
        ),
        (
            "intro<think>reasoning</think>tail",
            ("intro<think>reasoning</think>", "tail"),
        ),
        ("<think>missing close", None),
        ("<think>one</think><think>two</think>", None),
    ],
)
def test_split_after_think(text, expected):
    assert split_after_think(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a<think>x</think>b", "ab"),
        ("<think>x</think><think>y</think>", ""),
        ("no think here", "no think here"),
        ("<think>unterminated", "<think>unterminated"),
    ],
)
def test_strip_think_blocks(text, expected):
    assert strip_think_blocks(text) == expected


@pytest.mark.parametrize(
    ("text", "outside_think_only", "expected"),
    [
        ("<answer>1</answer><answer>2</answer>", False, 2),
        ("<think><answer>x</answer></think><answer>y</answer>", True, 1),
        ("<think><answer>x</answer></think><answer>y</answer>", False, 2),
        ("no tags", False, 0),
    ],
)
def test_count_tag_occurrences(text, outside_think_only, expected):
    assert count_tag_occurrences(text, "answer", outside_think_only) == expected


@pytest.mark.parametrize(
    ("text", "kwargs", "expected"),
    [
        ("<answer>42</answer>", {}, "42"),
        (
            "<think>r</think><answer>42</answer>",
            {"after_think_only": True},
            "42",
        ),
        (
            "<answer>1</answer><answer>2</answer>",
            {"strict_single": True},
            None,
        ),
        ("no tags", {}, None),
    ],
)
def test_extract_tagged(text, kwargs, expected):
    assert extract_tagged(text, **kwargs) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<answer>42</answer>", "42"),
        (" plain text ", "plain text"),
        ("<answer>1</answer><answer>2</answer>", "1"),
        ("", ""),
    ],
)
def test_extract_tagged_or_raw(text, expected):
    assert extract_tagged_or_raw(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<tool_call>a</tool_call><tool_call>b</tool_call>", ["a", "b"]),
        ("<tool_call> one </tool_call>", ["one"]),
        ("no tags", []),
        ("<tool_call>missing close", []),
    ],
)
def test_extract_all_tagged_blocks(text, expected):
    assert extract_all_tagged_blocks(text, "tool_call") == expected


@pytest.mark.parametrize(
    ("text", "strict_single", "expected"),
    [
        ("answer is \\boxed{42}", False, "42"),
        ("nested \\boxed{a^{2}} done", False, "a^{2}"),
        ("\\boxed{1} and \\boxed{2}", True, None),
        ("no box", False, None),
    ],
)
def test_extract_boxed(text, strict_single, expected):
    assert extract_boxed(text, strict_single=strict_single) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("\\boxed{42}", "42"),
        (" result \\boxed{a^{2}} ", "a^{2}"),
        ("plain answer", "plain answer"),
        ("", ""),
    ],
)
def test_normalize_boxed_answer(text, expected):
    assert normalize_boxed_answer(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        (" [1, 2] ", [1, 2]),
        ("not json", None),
        ('{"a": }', None),
    ],
)
def test_safe_json_loads(text, expected):
    assert safe_json_loads(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('prefix {"a": [1, 2]} suffix', {"a": [1, 2]}),
        ('noise ["x", {"y": 2}] tail', ["x", {"y": 2}]),
        ('text {"a":"}"} end', {"a": "}"}),
        ("no json here", None),
    ],
)
def test_extract_json(text, expected):
    assert extract_json(text) == expected


@pytest.mark.parametrize(
    ("text", "language", "expected"),
    [
        ('```json\n{"a":1}\n```', "json", '{"a":1}'),
        ("```python\nprint(1)\n```", None, "print(1)"),
        ("```txt\nhello\n```", "json", None),
        ("```python\nunterminated", "python", None),
    ],
)
def test_extract_fenced_block(text, language, expected):
    assert extract_fenced_block(text, language=language) == expected
