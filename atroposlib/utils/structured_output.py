"""Pure text, tag, and JSON parsing helpers for environment outputs."""

from __future__ import annotations

import json
import re
from typing import Any


THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def validate_single_think_block(text: str) -> bool:
    """Return True when text contains exactly one well-formed ``<think>`` block.

    A valid block requires exactly one opening tag, exactly one closing tag, and a
    complete ``<think>...</think>`` span. Non-string inputs are treated as invalid.
    """

    if not isinstance(text, str):
        return False

    open_count = len(re.findall(r"<think>", text, re.IGNORECASE))
    close_count = len(re.findall(r"</think>", text, re.IGNORECASE))
    if open_count != 1 or close_count != 1:
        return False

    match = THINK_BLOCK_PATTERN.search(text)
    return match is not None


def split_after_think(text: str) -> tuple[str, str] | None:
    """Split text into ``(prefix_through_think_block, suffix_after_think)``.

    Returns ``None`` when there is not exactly one valid ``<think>`` block. The
    first element preserves any text before the opening ``<think>`` tag and
    includes the closing ``</think>`` tag.
    """

    if not validate_single_think_block(text):
        return None

    match = THINK_BLOCK_PATTERN.search(text)
    if match is None:
        return None

    return text[: match.end()], text[match.end() :]


def strip_think_blocks(text: str) -> str:
    """Remove all complete ``<think>...</think>`` blocks from text.

    Returns the original value for non-string input converted to ``str``. Incomplete
    think tags are left untouched because there is no complete block to strip.
    """

    if not isinstance(text, str):
        return str(text)
    return THINK_BLOCK_PATTERN.sub("", text)


def count_tag_occurrences(text: str, tag: str, outside_think_only: bool = False) -> int:
    """Count complete occurrences of a tag pair like ``<answer>...</answer>``.

    If ``outside_think_only`` is True, complete think blocks are removed before
    counting. Blank tag names return ``0``.
    """

    if not isinstance(text, str) or not tag:
        return 0

    haystack = strip_think_blocks(text) if outside_think_only else text
    pattern = re.compile(
        rf"<{re.escape(tag)}\b[^>]*>.*?</{re.escape(tag)}>",
        re.DOTALL | re.IGNORECASE,
    )
    return len(pattern.findall(haystack))


def extract_tagged(
    text: str,
    tag: str = "answer",
    strict_single: bool = False,
    after_think_only: bool = False,
) -> str | None:
    """Extract content from the first matching tag block.

    Returns the inner text of ``<tag>...</tag>`` with surrounding whitespace
    stripped. If ``strict_single`` is True, exactly one matching block must exist.
    If ``after_think_only`` is True, extraction happens only on the text after a
    valid single think block; otherwise ``None`` is returned.
    """

    if not isinstance(text, str) or not tag:
        return None

    haystack = text
    if after_think_only:
        parts = split_after_think(text)
        if parts is None:
            return None
        haystack = parts[1]

    pattern = re.compile(
        rf"<{re.escape(tag)}\b[^>]*>(.*?)</{re.escape(tag)}>",
        re.DOTALL | re.IGNORECASE,
    )
    matches = pattern.findall(haystack)
    if not matches:
        return None
    if strict_single and len(matches) != 1:
        return None
    return matches[0].strip()


def extract_tagged_or_raw(text: str, tag: str = "answer") -> str:
    """Extract tagged content when present, otherwise return stripped raw text.

    Missing tags, malformed tags, or non-string values fall back to ``str(value)``
    with leading/trailing whitespace removed.
    """

    extracted = extract_tagged(text, tag=tag, strict_single=False, after_think_only=False)
    if extracted is not None:
        return extracted
    return text.strip() if isinstance(text, str) else str(text).strip()


def extract_all_tagged_blocks(text: str, tag: str) -> list[str]:
    """Return all inner contents for matching tag blocks.

    The returned list preserves source order and strips surrounding whitespace from
    each captured block. Invalid inputs return an empty list.
    """

    if not isinstance(text, str) or not tag:
        return []

    pattern = re.compile(
        rf"<{re.escape(tag)}\b[^>]*>(.*?)</{re.escape(tag)}>",
        re.DOTALL | re.IGNORECASE,
    )
    return [match.strip() for match in pattern.findall(text)]


def extract_boxed(text: str, strict_single: bool = False) -> str | None:
    """Extract the content of the first ``\\boxed{...}`` expression.

    Nested braces inside the boxed content are supported. If ``strict_single`` is
    True, exactly one boxed expression must be present. Returns ``None`` when no
    complete boxed span is found.
    """

    if not isinstance(text, str):
        return None

    matches = BOXED_PATTERN.findall(text)
    if not matches:
        return None
    if strict_single and len(matches) != 1:
        return None
    return matches[0].strip()


def normalize_boxed_answer(text: str) -> str:
    """Return the first boxed payload when present, otherwise stripped text.

    This helper is intentionally shallow: it removes a single outer ``\\boxed{}``
    wrapper but does not perform mathematical normalization or semantic checking.
    """

    if not isinstance(text, str):
        return str(text).strip()

    extracted = extract_boxed(text, strict_single=False)
    return extracted if extracted is not None else text.strip()


def safe_json_loads(text: str) -> Any | None:
    """Parse JSON and return ``None`` instead of raising on malformed input.

    Non-string inputs return ``None``. Leading and trailing whitespace are ignored.
    """

    if not isinstance(text, str):
        return None
    try:
        return json.loads(text.strip())
    except (TypeError, json.JSONDecodeError, ValueError):
        return None


def extract_json(text: str) -> dict | list | None:
    """Extract and parse the first top-level JSON object or array from text.

    The scan is quote-aware, so braces or brackets inside JSON strings do not break
    matching. Returns only ``dict`` or ``list`` results; scalar JSON values are
    treated as unsupported and return ``None``.
    """

    if not isinstance(text, str):
        return None

    start_positions = [
        (idx, char) for idx, char in enumerate(text) if char in "{["
    ]
    for start_idx, start_char in start_positions:
        end_char = "}" if start_char == "{" else "]"
        depth = 0
        in_string = False
        escape = False

        for idx in range(start_idx, len(text)):
            char = text[idx]

            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
                continue

            if char == start_char:
                depth += 1
            elif char == end_char:
                depth -= 1
                if depth == 0:
                    candidate = text[start_idx : idx + 1]
                    parsed = safe_json_loads(candidate)
                    if isinstance(parsed, (dict, list)):
                        return parsed
                    break

    return None


def extract_fenced_block(text: str, language: str | None = None) -> str | None:
    """Extract the first fenced code block from Markdown-like text.

    If ``language`` is provided, only fences with that language label are matched.
    The returned content excludes the backticks and preserves internal newlines.
    Returns ``None`` for malformed or absent fences.
    """

    if not isinstance(text, str):
        return None

    if language is None:
        pattern = re.compile(r"```[^\n]*\n(.*?)\n```", re.DOTALL)
    else:
        pattern = re.compile(
            rf"```{re.escape(language)}[^\n]*\n(.*?)\n```",
            re.DOTALL | re.IGNORECASE,
        )

    match = pattern.search(text)
    if match is None:
        return None
    return match.group(1)
