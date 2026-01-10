from __future__ import annotations

from atroposlib.envs.server_handling.openai_server import resolve_openai_configs
from atroposlib.envs.server_handling.server_baseline import APIServerConfig


class _Logger:
    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass


def test_resolve_openai_configs_interpolates_api_key_from_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    cfg = resolve_openai_configs(
        default_server_configs=[APIServerConfig()],
        openai_config_dict={
            "api_key": "$OPENROUTER_API_KEY",
            "base_url": "https://openrouter.ai/api/v1",
            "model_name": "openai/gpt-5-nano",
        },
        yaml_config={},
        cli_passed_flags={},
        logger=_Logger(),
    )

    assert isinstance(cfg, list)
    assert isinstance(cfg[0], APIServerConfig)
    assert cfg[0].api_key == "test-key"


def test_resolve_openai_configs_interpolates_braced_api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-2")

    cfg = resolve_openai_configs(
        default_server_configs=[APIServerConfig()],
        openai_config_dict={
            "api_key": "${OPENROUTER_API_KEY}",
            "base_url": "https://openrouter.ai/api/v1",
            "model_name": "openai/gpt-5-nano",
        },
        yaml_config={},
        cli_passed_flags={},
        logger=_Logger(),
    )

    assert isinstance(cfg, list)
    assert isinstance(cfg[0], APIServerConfig)
    assert cfg[0].api_key == "test-key-2"


def test_resolve_openai_configs_leaves_plain_api_key_untouched(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    cfg = resolve_openai_configs(
        default_server_configs=[APIServerConfig()],
        openai_config_dict={
            "api_key": "literal-key",
            "base_url": "https://openrouter.ai/api/v1",
            "model_name": "openai/gpt-5-nano",
        },
        yaml_config={},
        cli_passed_flags={},
        logger=_Logger(),
    )

    assert isinstance(cfg, list)
    assert isinstance(cfg[0], APIServerConfig)
    assert cfg[0].api_key == "literal-key"
