"""Tests for ensure_trajectory_token_limit trajectory truncation."""

from atroposlib.utils import message_history_utils
from atroposlib.utils.message_history_utils import ensure_trajectory_token_limit


def _fake_tokenize_for_trainer(tokenizer, messages, *args, **kwargs):
    """Deterministic stand-in: token count = sum of each message's ``_ntok``."""
    tokens, masks = [], []
    for msg in messages:
        n = msg.get("_ntok", 10)
        tokens.extend([1] * n)
        masks.extend([-100] * n)
    return {"tokens": tokens, "masks": masks}


def _msg(role, ntok, content):
    return {"role": role, "content": content, "_ntok": ntok}


def _make_step(alternatives):
    return {
        "seed": 1,
        "parsed_actions": [0] * len(alternatives),
        "scores": [0.0] * len(alternatives),
        "messages": [[m.copy() for m in alt] for alt in alternatives],
        "tokens": [
            _fake_tokenize_for_trainer(None, alt)["tokens"] for alt in alternatives
        ],
        "masks": [
            _fake_tokenize_for_trainer(None, alt)["masks"] for alt in alternatives
        ],
    }


def test_truncation_preserves_minimal_alternative(monkeypatch):
    """A short alternative already at the preserve-minimum must be left intact
    when a longer alternative in the same step drives truncation.
    """
    monkeypatch.setattr(
        message_history_utils, "tokenize_for_trainer", _fake_tokenize_for_trainer
    )
    long_alt = [
        _msg("system", 5, "sys"),
        _msg("environment", 5, "env1"),
        _msg("agent", 50, "ag1"),
        _msg("environment", 5, "env2"),
        _msg("agent", 50, "ag2"),
    ]  # 115 tokens, over the limit
    minimal_alt = [
        _msg("system", 5, "sys"),
        _msg("environment", 5, "env"),
        _msg("assistant", 10, "keep-me"),
    ]  # 20 tokens, under the limit and already at the preserve-minimum

    result = ensure_trajectory_token_limit(
        [_make_step([long_alt, minimal_alt])],
        tokenizer=None,
        max_trajectory_tokens=60,
    )

    assert len(result) == 1
    kept_minimal = result[0]["messages"][1]
    assert [m["role"] for m in kept_minimal] == [
        "system",
        "environment",
        "assistant",
    ]
    assert kept_minimal[-1]["content"] == "keep-me"
    # tokens/masks stay consistent with the (untouched) messages.
    assert len(result[0]["tokens"][1]) == 20
    # the long alternative is still truncated to fit the limit.
    assert len(result[0]["tokens"][0]) <= 60


def test_truncation_keeps_all_alternatives_within_limit(monkeypatch):
    """With several alternatives of different lengths, each is truncated only
    as much as its own budget allows and all end up within the limit.
    """
    monkeypatch.setattr(
        message_history_utils, "tokenize_for_trainer", _fake_tokenize_for_trainer
    )
    alternatives = [
        [
            _msg("system", 5, "s"),
            _msg("environment", 5, "e"),
            _msg("agent", 40, "a"),
            _msg("environment", 5, "e"),
            _msg("agent", 40, "a"),
            _msg("environment", 5, "e"),
            _msg("agent", 40, "a"),
        ],
        [
            _msg("system", 5, "s"),
            _msg("environment", 5, "e"),
            _msg("agent", 30, "a"),
            _msg("environment", 5, "e"),
            _msg("agent", 30, "a"),
        ],
        [
            _msg("system", 5, "s"),
            _msg("environment", 5, "e"),
            _msg("assistant", 8, "keep"),
        ],
    ]

    result = ensure_trajectory_token_limit(
        [_make_step(alternatives)], tokenizer=None, max_trajectory_tokens=60
    )

    assert len(result) == 1
    for alt_tokens in result[0]["tokens"]:
        assert len(alt_tokens) <= 60
    # the already-minimal alternative kept its assistant turn.
    assert result[0]["messages"][2][-1]["content"] == "keep"
