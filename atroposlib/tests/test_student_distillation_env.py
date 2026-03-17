"""Tests for StudentDistillationEnv distillation enrichment."""

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from atroposlib.envs.student_distillation_env import StudentDistillationEnv


class _FakeManagedServer:
    def __init__(self, prompt_tokens):
        self.prompt_tokens = prompt_tokens
        self.calls = 0
        self.kwargs = []

    async def get_logprobs(self, **kwargs):
        self.calls += 1
        self.kwargs.append(kwargs)
        return {
            "prompt_tokens": self.prompt_tokens,
            "prompt_topk_token_ids": [[tok, tok + 1] for tok in self.prompt_tokens],
            "prompt_topk_logprobs": [[-0.1, -0.2] for _ in self.prompt_tokens],
        }


class _FakeStudentServer:
    def __init__(self, fail_on_call: int = -1, managed_prompt_tokens=None):
        self.calls = 0
        self.fail_on_call = fail_on_call
        self.kwargs = []
        self.managed_calls = 0
        self.managed = _FakeManagedServer(
            prompt_tokens=managed_prompt_tokens if managed_prompt_tokens is not None else []
        )

    async def get_logprobs(self, **kwargs):
        self.calls += 1
        self.kwargs.append(kwargs)
        if self.calls == self.fail_on_call:
            raise RuntimeError("student backend failure")
        seq = kwargs["input_ids"]
        return {
            "prompt_tokens": seq,
            "prompt_topk_token_ids": [[tok, tok + 1] for tok in seq],
            "prompt_topk_logprobs": [[-0.1, -0.2] for _ in seq],
        }

    @asynccontextmanager
    async def managed_server(self, tokenizer=None):
        self.managed_calls += 1
        yield self.managed


class _ConcreteStudentEnv(StudentDistillationEnv):
    async def get_next_item(self):
        return None

    async def evaluate(self, *args, **kwargs):
        return None


@pytest.mark.asyncio
async def test_attach_student_distillation_success():
    env = object.__new__(_ConcreteStudentEnv)
    env.config = SimpleNamespace(student_distill_enabled=True, student_top_k=2)
    env.server = _FakeStudentServer()

    group = {
        "tokens": [[1, 2, 3], [4, 5]],
        "group_overrides": None,
        "overrides": None,
        "masks": [[-100, 2, 3], [-100, 5]],
        "scores": [1.0, 0.0],
    }
    out = await StudentDistillationEnv._attach_student_distillation(env, group)
    assert out["distill_token_ids"] is not None
    assert out["distill_logprobs"] is not None
    assert len(out["distill_token_ids"]) == 2
    assert len(out["distill_token_ids"][0]) == 3
    assert len(out["distill_logprobs"][1]) == 2
    assert env.server.calls == 2


@pytest.mark.asyncio
async def test_attach_student_distillation_failure_drops_payload():
    env = object.__new__(_ConcreteStudentEnv)
    env.config = SimpleNamespace(student_distill_enabled=True, student_top_k=2)
    env.server = _FakeStudentServer(fail_on_call=2)

    group = {
        "tokens": [[1, 2, 3], [4, 5]],
        "group_overrides": None,
        "overrides": None,
        "masks": [[-100, 2, 3], [-100, 5]],
        "scores": [1.0, 0.0],
    }
    out = await StudentDistillationEnv._attach_student_distillation(env, group)
    assert out["distill_token_ids"] is None
    assert out["distill_logprobs"] is None


@pytest.mark.asyncio
async def test_attach_student_distillation_negative_topk_skips_fetch():
    env = object.__new__(_ConcreteStudentEnv)
    env.config = SimpleNamespace(student_distill_enabled=True, student_top_k=-1)
    env.server = _FakeStudentServer()

    group = {
        "tokens": [[1, 2, 3]],
        "group_overrides": None,
        "overrides": None,
        "masks": [[-100, 2, 3]],
        "scores": [1.0],
    }
    out = await StudentDistillationEnv._attach_student_distillation(env, group)
    assert env.server.calls == 0
    assert out["distill_token_ids"] is None
    assert out["distill_logprobs"] is None


@pytest.mark.asyncio
async def test_attach_student_distillation_zero_topk_passthrough():
    env = object.__new__(_ConcreteStudentEnv)
    env.config = SimpleNamespace(student_distill_enabled=True, student_top_k=0)
    env.server = _FakeStudentServer()

    group = {
        "tokens": [[1, 2, 3]],
        "group_overrides": None,
        "overrides": None,
        "masks": [[-100, 2, 3]],
        "scores": [1.0],
    }
    out = await StudentDistillationEnv._attach_student_distillation(env, group)
    assert env.server.calls == 1
    assert out["distill_token_ids"] is not None
    assert out["distill_logprobs"] is not None


@pytest.mark.asyncio
async def test_attach_student_distillation_group_override_topk_is_used():
    env = object.__new__(_ConcreteStudentEnv)
    env.config = SimpleNamespace(student_distill_enabled=True, student_top_k=0)
    env.server = _FakeStudentServer()

    group = {
        "tokens": [[1, 2, 3], [4, 5]],
        "group_overrides": {"student_top_k": 7},
        "overrides": None,
        "masks": [[-100, 2, 3], [-100, 5]],
        "scores": [1.0, 0.0],
    }
    out = await StudentDistillationEnv._attach_student_distillation(env, group)
    assert env.server.kwargs[0]["top_k"] == 7
    assert env.server.kwargs[1]["top_k"] == 7
    assert out["distill_token_ids"] is not None
    assert out["distill_logprobs"] is not None


@pytest.mark.asyncio
async def test_attach_student_distillation_group_override_can_skip_fetch():
    env = object.__new__(_ConcreteStudentEnv)
    env.config = SimpleNamespace(student_distill_enabled=True, student_top_k=2)
    env.server = _FakeStudentServer()

    group = {
        "tokens": [[1, 2, 3]],
        "group_overrides": {"skip_student_top_k": True},
        "overrides": None,
        "masks": [[-100, 2, 3]],
        "scores": [1.0],
    }
    out = await StudentDistillationEnv._attach_student_distillation(env, group)
    assert env.server.calls == 0
    assert out["distill_token_ids"] is None
    assert out["distill_logprobs"] is None


def test_get_student_logprob_overrides_merges_group_and_sequence():
    env = object.__new__(_ConcreteStudentEnv)
    group = {
        "group_overrides": {
            "student_logprob_kwargs": {"temperature": 0.0, "prompt": "group"}
        },
        "overrides": [
            {"student_logprob_kwargs": {"prompt": "seq", "top_p": 1.0}},
        ],
    }

    out = StudentDistillationEnv._get_student_logprob_overrides(env, group, 0)
    assert out == {"temperature": 0.0, "prompt": "seq", "top_p": 1.0}


@pytest.mark.asyncio
async def test_fetch_student_for_sequence_uses_managed_server_for_messages():
    env = object.__new__(_ConcreteStudentEnv)
    env.server = _FakeStudentServer(managed_prompt_tokens=[1, 2, 3])
    env.tokenizer = object()

    out = await StudentDistillationEnv._fetch_student_for_sequence(
        env,
        token_ids=[1, 2, 3],
        top_k=2,
        extra_kwargs={"messages": [{"role": "user", "content": "hi"}]},
    )

    assert env.server.calls == 0
    assert env.server.managed_calls == 1
    assert env.server.managed.calls == 1
    assert "input_ids" not in env.server.managed.kwargs[0]
    assert out[0] == [[1, 2], [2, 3], [3, 4]]


@pytest.mark.asyncio
async def test_fetch_student_for_sequence_mismatch_drops_payload():
    env = object.__new__(_ConcreteStudentEnv)
    env.config = SimpleNamespace(student_distill_enabled=True, student_top_k=2)
    env.server = _FakeStudentServer(managed_prompt_tokens=[9, 9])
    env.tokenizer = object()

    group = {
        "tokens": [[1, 2, 3]],
        "group_overrides": {
            "student_logprob_kwargs": {"messages": [{"role": "user", "content": "hi"}]}
        },
        "overrides": None,
        "masks": [[-100, 2, 3]],
        "scores": [1.0],
    }

    out = await StudentDistillationEnv._attach_student_distillation(env, group)
    assert out["distill_token_ids"] is None
    assert out["distill_logprobs"] is None
