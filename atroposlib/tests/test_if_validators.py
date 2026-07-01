r"""Regression tests for the instruction-following reward validators.

Several validators in ``instruction_following_algorithm_environment`` were
ported from allenai/open-instruct with an extra layer of backslash escaping,
e.g. ``re.findall(r"\\[(.*?)\\]", text)`` — as the regex engine sees it that
is ``\\[`` (a literal backslash followed by a character class), not ``\[`` (a
literal bracket). Those patterns therefore never matched normal model output,
so the affected validators silently returned 0 / False and zeroed out the RL
reward for any rollout whose ground-truth constraint used them.

These tests pin the intended behavior: a response that satisfies the
constraint must score as satisfying it.
"""

import pytest

# The environment module pulls in the training stack; skip cleanly where those
# optional deps aren't installed rather than erroring at collection time.
pytest.importorskip("datasets")
pytest.importorskip("wandb")
pytest.importorskip("langdetect")

from environments.instruction_following_algorithm_environment import (  # noqa: E402
    validate_frequency_capital_words,
    validate_placeholders,
    validate_sections,
    verify_bullet_points,
    verify_keyword_frequency,
)


def test_validate_placeholders_matches_bracketed_spans():
    ok, found = validate_placeholders("Ship to [address] and [name].", 2)
    assert ok is True
    assert found == ["address", "name"]


def test_verify_bullet_points_counts_markdown_bullets():
    assert verify_bullet_points("* one\n* two\n* three", 3) is True
    assert verify_bullet_points("- a\n+ b", 2) is True


def test_validate_sections_counts_splitter_occurrences():
    assert validate_sections("SECTION 1: alpha SECTION 2: beta", 2, "SECTION") is True


def test_verify_keyword_frequency_extracts_words():
    # _extract_words must find real word tokens for the count to work.
    assert verify_keyword_frequency("cat cat dog", "cat", 2) is True
    assert verify_keyword_frequency("cat cat dog", "cat", 1) is False


def test_validate_frequency_capital_words_finds_all_caps():
    assert (
        validate_frequency_capital_words("USA and NASA are here", 2, "at least") is True
    )
    assert (
        validate_frequency_capital_words("nothing shouted here", 1, "at least") is False
    )
