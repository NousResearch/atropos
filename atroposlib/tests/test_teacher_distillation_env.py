"""Tests for TeacherDistillationEnv distillation enrichment."""

from types import SimpleNamespace

import pytest

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
