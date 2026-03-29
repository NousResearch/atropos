"""Tests for TeacherDistillationEnv distillation enrichment."""

from types import SimpleNamespace

import pytest

from atroposlib.envs.server_handling.server_baseline import APIServerConfig
from atroposlib.envs.teacher_distillation_env import TeacherDistillationEnv


class _FakeTeacherServer:
    def __init__(self, fail_on_call: int = -1):
        self.calls = 0
        self.fail_on_call = fail_on_call

    async def get_logprobs(self, **kwargs):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("teacher backend failure")
        seq = kwargs["input_ids"]
        return {
            "prompt_tokens": seq,
            "prompt_topk_token_ids": [[tok, tok + 1] for tok in seq],
            "prompt_topk_logprobs": [[-0.1, -0.2] for _ in seq],
        }


class _ConcreteTeacherEnv(TeacherDistillationEnv):
    async def get_next_item(self):
        return None

    async def evaluate(self, *args, **kwargs):
        return None


class _DummyTokenizer:
    name_or_path = "student-model"

    def get_vocab(self):
        return {"a": 1}


class _CapturingServerManager:
    def __init__(self, configs, slurm=False, testing=False):
        self.configs = configs
        self.slurm = slurm
        self.testing = testing


@pytest.mark.asyncio
async def test_attach_teacher_distillation_success():
    env = object.__new__(_ConcreteTeacherEnv)
    env.config = SimpleNamespace(teacher_enabled=True, teacher_top_k=2)
    env.teacher_server = _FakeTeacherServer()

    group = {
        "tokens": [[1, 2, 3], [4, 5]],
        "group_overrides": None,
        "masks": [[-100, 2, 3], [-100, 5]],
        "scores": [1.0, 0.0],
    }
    out = await TeacherDistillationEnv._attach_teacher_distillation(env, group)
    assert out["distill_token_ids"] is not None
    assert out["distill_logprobs"] is not None
    assert len(out["distill_token_ids"]) == 2
    assert len(out["distill_token_ids"][0]) == 3
    assert len(out["distill_logprobs"][1]) == 2


@pytest.mark.asyncio
async def test_attach_teacher_distillation_failure_drops_payload():
    env = object.__new__(_ConcreteTeacherEnv)
    env.config = SimpleNamespace(teacher_enabled=True, teacher_top_k=2)
    env.teacher_server = _FakeTeacherServer(fail_on_call=2)

    group = {
        "tokens": [[1, 2, 3], [4, 5]],
        "group_overrides": None,
        "masks": [[-100, 2, 3], [-100, 5]],
        "scores": [1.0, 0.0],
    }
    out = await TeacherDistillationEnv._attach_teacher_distillation(env, group)
    assert out["distill_token_ids"] is None
    assert out["distill_logprobs"] is None


@pytest.mark.asyncio
async def test_attach_teacher_distillation_negative_topk_skips_fetch():
    env = object.__new__(_ConcreteTeacherEnv)
    env.config = SimpleNamespace(teacher_enabled=True, teacher_top_k=-1)
    env.teacher_server = _FakeTeacherServer()

    group = {
        "tokens": [[1, 2, 3]],
        "group_overrides": None,
        "masks": [[-100, 2, 3]],
        "scores": [1.0],
    }
    out = await TeacherDistillationEnv._attach_teacher_distillation(env, group)
    assert env.teacher_server.calls == 0
    assert out["distill_token_ids"] is None
    assert out["distill_logprobs"] is None


@pytest.mark.asyncio
async def test_attach_teacher_distillation_zero_topk_passthrough():
    env = object.__new__(_ConcreteTeacherEnv)
    env.config = SimpleNamespace(teacher_enabled=True, teacher_top_k=0)
    env.teacher_server = _FakeTeacherServer()

    group = {
        "tokens": [[1, 2, 3]],
        "group_overrides": None,
        "masks": [[-100, 2, 3]],
        "scores": [1.0],
    }
    out = await TeacherDistillationEnv._attach_teacher_distillation(env, group)
    assert env.teacher_server.calls == 1
    assert out["distill_token_ids"] is not None
    assert out["distill_logprobs"] is not None


@pytest.mark.asyncio
async def test_attach_teacher_distillation_group_override_topk_is_used():
    env = object.__new__(_ConcreteTeacherEnv)
    env.config = SimpleNamespace(teacher_enabled=True, teacher_top_k=0)

    seen_topks = []

    async def _fake_fetch(seq, top_k):
        seen_topks.append(top_k)
        return [[tok] for tok in seq], [[-0.1] for _ in seq]

    env.teacher_server = object()
    env._fetch_teacher_for_sequence = _fake_fetch

    group = {
        "tokens": [[1, 2, 3], [4, 5]],
        "group_overrides": {"teacher_top_k": 7},
        "masks": [[-100, 2, 3], [-100, 5]],
        "scores": [1.0, 0.0],
    }
    out = await TeacherDistillationEnv._attach_teacher_distillation(env, group)
    assert seen_topks == [7, 7]
    assert out["distill_token_ids"] is not None
    assert out["distill_logprobs"] is not None


@pytest.mark.asyncio
async def test_attach_teacher_distillation_group_override_can_skip_fetch():
    env = object.__new__(_ConcreteTeacherEnv)
    env.config = SimpleNamespace(teacher_enabled=True, teacher_top_k=2)
    env.teacher_server = _FakeTeacherServer()

    group = {
        "tokens": [[1, 2, 3]],
        "group_overrides": {"skip_teacher_top_k": True},
        "masks": [[-100, 2, 3]],
        "scores": [1.0],
    }
    out = await TeacherDistillationEnv._attach_teacher_distillation(env, group)
    assert env.teacher_server.calls == 0
    assert out["distill_token_ids"] is None
    assert out["distill_logprobs"] is None


def test_teacher_tokenizer_mismatch_raises(monkeypatch):
    env = object.__new__(_ConcreteTeacherEnv)

    class _StudentTokenizer:
        name_or_path = "student-model"

        def get_vocab(self):
            return {"a": 1}

    class _TeacherTokenizer:
        def get_vocab(self):
            return {"b": 1}

    env.tokenizer = _StudentTokenizer()
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: _TeacherTokenizer(),
    )

    with pytest.raises(
        ValueError, match="Cross-tokenizer distillation is not supported"
    ):
        TeacherDistillationEnv._validate_teacher_tokenizer_compatibility(
            env,
            teacher_tokenizer_name="teacher-model",
        )


def test_init_requires_teacher_server_source(monkeypatch):
    from atroposlib.envs import teacher_distillation_env as module

    def _fake_base_init(self, config, server_configs, slurm=False, testing=False):
        self.config = config
        self.tokenizer = _DummyTokenizer()

    monkeypatch.setattr(module.BaseEnv, "__init__", _fake_base_init)

    config = SimpleNamespace(
        teacher_enabled=True,
        teacher_top_k=0,
    )
    with pytest.raises(
        ValueError, match="no teacher server configuration was provided"
    ):
        _ConcreteTeacherEnv(
            config=config,
            server_configs=[],
        )


def test_init_uses_explicit_teacher_server_configs(monkeypatch):
    from atroposlib.envs import teacher_distillation_env as module

    called = {}

    def _fake_base_init(self, config, server_configs, slurm=False, testing=False):
        self.config = config
        self.tokenizer = _DummyTokenizer()

    def _fake_validate(self, teacher_tokenizer_name):
        called["teacher_tokenizer_name"] = teacher_tokenizer_name

    monkeypatch.setattr(module.BaseEnv, "__init__", _fake_base_init)
    monkeypatch.setattr(module, "ServerManager", _CapturingServerManager)
    monkeypatch.setattr(
        _ConcreteTeacherEnv,
        "_validate_teacher_tokenizer_compatibility",
        _fake_validate,
    )

    explicit_cfg = APIServerConfig(
        model_name="explicit-model",
        tokenizer_name="explicit-tokenizer",
        base_url="http://explicit/v1",
        api_key="x",
        server_type="vllm",
    )
    config = SimpleNamespace(
        teacher_enabled=True,
        teacher_top_k=0,
    )

    env = _ConcreteTeacherEnv(
        config=config,
        server_configs=[],
        teacher_server_configs=[explicit_cfg],
    )

    assert isinstance(env.teacher_server, _CapturingServerManager)
    assert env.teacher_server.configs == [explicit_cfg]
    assert called["teacher_tokenizer_name"] == "explicit-tokenizer"


def test_init_wraps_bare_teacher_api_server_config(monkeypatch):
    from atroposlib.envs import teacher_distillation_env as module

    called = {}

    def _fake_base_init(self, config, server_configs, slurm=False, testing=False):
        self.config = config
        self.tokenizer = _DummyTokenizer()

    def _fake_validate(self, teacher_tokenizer_name):
        called["teacher_tokenizer_name"] = teacher_tokenizer_name

    monkeypatch.setattr(module.BaseEnv, "__init__", _fake_base_init)
    monkeypatch.setattr(module, "ServerManager", _CapturingServerManager)
    monkeypatch.setattr(
        _ConcreteTeacherEnv,
        "_validate_teacher_tokenizer_compatibility",
        _fake_validate,
    )

    explicit_cfg = APIServerConfig(
        model_name="explicit-model",
        tokenizer_name="explicit-tokenizer",
        base_url="http://explicit/v1",
        api_key="x",
        server_type="vllm",
    )
    config = SimpleNamespace(
        teacher_enabled=True,
        teacher_top_k=0,
    )

    env = _ConcreteTeacherEnv(
        config=config,
        server_configs=[],
        teacher_server_configs=explicit_cfg,
    )

    assert isinstance(env.teacher_server, _CapturingServerManager)
    assert env.teacher_server.configs == [explicit_cfg]
    assert called["teacher_tokenizer_name"] == "explicit-tokenizer"


def test_resolve_teacher_server_configs_returns_none_when_unset():
    assert (
        _ConcreteTeacherEnv._resolve_teacher_server_configs(
            default_teacher_server_configs=None,
            yaml_config={},
            cli_passed_flags={},
        )
        is None
    )


def test_resolve_teacher_server_configs_uses_teacher_namespace(monkeypatch):
    from atroposlib.envs import teacher_distillation_env as module

    captured = {}

    def _fake_resolve(**kwargs):
        captured.update(kwargs)
        return ["resolved"]

    monkeypatch.setattr(module, "resolve_openai_configs", _fake_resolve)

    default_cfg = APIServerConfig(
        model_name="teacher-model",
        base_url="http://teacher/v1",
        api_key="x",
        server_type="vllm",
    )

    out = _ConcreteTeacherEnv._resolve_teacher_server_configs(
        default_teacher_server_configs=default_cfg,
        yaml_config={"teacher": {"tokenizer_name": "teacher-tokenizer"}},
        cli_passed_flags={"teacher.base_url": "http://override/v1"},
    )

    assert out == ["resolved"]
    assert captured["openai_config_dict"]["base_url"] == "http://override/v1"
    assert captured["openai_config_dict"]["tokenizer_name"] == "teacher-tokenizer"
    assert captured["yaml_config"] == {
        "openai": {"tokenizer_name": "teacher-tokenizer"}
    }
    assert captured["cli_passed_flags"] == {"openai.base_url": "http://override/v1"}
