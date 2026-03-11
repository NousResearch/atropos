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

    # Build a fake argv
    argv = [
        "atropos-sft-gen",
        "out.jsonl",
        "--api-url",
        "http://example.com",
        "--tokenizer",
        "tok",
        "--dry-run",
    ]

    monkeypatch.setattr(sft_cli, "asyncio", SimpleNamespace(run=asyncio.run))

    # Patch argparse to use our argv
    import argparse

    parser = argparse.ArgumentParser()
    # We don't care about exact options here; we just call main() with patched argv via parse_args
    # by temporarily replacing argparse.ArgumentParser with one that uses our argv internally.

    original_argparse = sft_cli.argparse

    class _FakeParser(argparse.ArgumentParser):
        def parse_args(self, _args=None, *_a, **_k):
            return original_argparse.ArgumentParser(
                description="Grab SFT data from an Atropos API instance."
            ).parse_args(argv[1:])

    monkeypatch.setattr(
        sft_cli, "argparse", SimpleNamespace(ArgumentParser=_FakeParser)
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

    argv = [
        "atropos-dpo-gen",
        "out.jsonl",
        "--api-url",
        "http://example.com",
        "--tokenizer",
        "tok",
        "--dry-run",
    ]

    monkeypatch.setattr(dpo_cli, "asyncio", SimpleNamespace(run=asyncio.run))

    import argparse

    original_argparse = dpo_cli.argparse

    class _FakeParser(argparse.ArgumentParser):
        def parse_args(self, _args=None, *_a, **_k):
            return original_argparse.ArgumentParser(
                description="Grab DPO data from an Atropos API instance."
            ).parse_args(argv[1:])

    monkeypatch.setattr(
        dpo_cli, "argparse", SimpleNamespace(ArgumentParser=_FakeParser)
    )

    dpo_cli.main()

    assert called.dry is True
    assert called.full is False
