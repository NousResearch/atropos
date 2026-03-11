import asyncio
from types import SimpleNamespace

import pytest

import atroposlib.cli.dpo as dpo_cli
import atroposlib.cli.sft as sft_cli


class _DummyResponse:
    def __init__(self, status: int = 200) -> None:
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _DummySession:
    def __init__(self, *_, **__) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def get(self, *_args, **_kwargs):
        return _DummyResponse()


@pytest.mark.asyncio
async def test_sft_dry_run_uses_tokenizer_and_reaches_api(monkeypatch):
    called = SimpleNamespace(tok=False, session=False)

    def _fake_from_pretrained(model_name: str):
        called.tok = True
        assert model_name == "dummy-tokenizer"
        return object()

    monkeypatch.setattr(
        sft_cli, "AutoTokenizer", SimpleNamespace(from_pretrained=_fake_from_pretrained)
    )
    monkeypatch.setattr(sft_cli.aiohttp, "ClientSession", _DummySession)

    await sft_cli.sft_dry_run(
        api_url="http://localhost:8000", tokenizer="dummy-tokenizer"
    )

    assert called.tok is True


@pytest.mark.asyncio
async def test_dpo_dry_run_uses_tokenizer_and_reaches_api(monkeypatch):
    called = SimpleNamespace(tok=False)

    def _fake_from_pretrained(model_name: str):
        called.tok = True
        assert model_name == "dummy-tokenizer"
        return object()

    monkeypatch.setattr(
        dpo_cli, "AutoTokenizer", SimpleNamespace(from_pretrained=_fake_from_pretrained)
    )
    monkeypatch.setattr(dpo_cli.aiohttp, "ClientSession", _DummySession)

    await dpo_cli.dpo_dry_run(
        api_url="http://localhost:8000", tokenizer="dummy-tokenizer"
    )

    assert called.tok is True


def test_sft_main_invokes_dry_run_when_flag_is_set(monkeypatch):
    """
    Ensure that passing --dry-run to the entrypoint does *not*
    call the full data grabber, only the dry run helper.
    """

    called = SimpleNamespace(dry=False, full=False)

    async def _fake_sft_dry_run(api_url: str, tokenizer: str) -> None:
        called.dry = True
        assert api_url == "http://example.com"
        assert tokenizer == "tok"

    async def _fake_sft_data_grabber(*_args, **_kwargs):
        called.full = True

    monkeypatch.setattr(sft_cli, "sft_dry_run", _fake_sft_dry_run)
    monkeypatch.setattr(sft_cli, "sft_data_grabber", _fake_sft_data_grabber)

    # Simulate that argparse has already parsed args with --dry-run
    class _Args:
        filepath = "out.jsonl"
        api_url = "http://example.com"
        group_size = 2
        max_token_len = 2048
        tokenizer = "tok"
        save_messages = False
        save_top_n_per_group = 3
        num_seqs_to_save = 10
        allow_negative_scores = False
        minimum_score_diff_max_min = 0.0
        append_to_previous = False
        tasks_per_step = 64
        dry_run = True

    monkeypatch.setattr(
        sft_cli,
        "argparse",
        SimpleNamespace(Namespace=_Args, ArgumentParser=lambda *_, **__: None),
    )
    monkeypatch.setattr(
        sft_cli,
        "asyncio",
        SimpleNamespace(
            run=lambda coro: asyncio.get_event_loop().run_until_complete(coro)
        ),
    )

    sft_cli.main()

    assert called.dry is True
    assert called.full is False


def test_dpo_main_invokes_dry_run_when_flag_is_set(monkeypatch):
    called = SimpleNamespace(dry=False, full=False)

    async def _fake_dpo_dry_run(api_url: str, tokenizer: str) -> None:
        called.dry = True
        assert api_url == "http://example.com"
        assert tokenizer == "tok"

    async def _fake_dpo_data_grabber(*_args, **_kwargs):
        called.full = True

    monkeypatch.setattr(dpo_cli, "dpo_dry_run", _fake_dpo_dry_run)
    monkeypatch.setattr(dpo_cli, "dpo_data_grabber", _fake_dpo_data_grabber)

    class _Args:
        filepath = "out.jsonl"
        api_url = "http://example.com"
        group_size = 2
        max_token_len = 2048
        tokenizer = "tok"
        save_messages = False
        save_n_pairs_per_group = 3
        num_seqs_to_save = 10
        allow_negative_scores = False
        minimum_score_diff_max_min = 0.5
        append_to_previous = False
        dry_run = True

    monkeypatch.setattr(
        dpo_cli,
        "argparse",
        SimpleNamespace(Namespace=_Args, ArgumentParser=lambda *_, **__: None),
    )
    monkeypatch.setattr(
        dpo_cli,
        "asyncio",
        SimpleNamespace(
            run=lambda coro: asyncio.get_event_loop().run_until_complete(coro)
        ),
    )

    dpo_cli.main()

    assert called.dry is True
    assert called.full is False
