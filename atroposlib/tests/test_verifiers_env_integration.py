from __future__ import annotations

from typing import Any

import pytest
from datasets import Dataset
from verifiers.envs.singleturn_env import SingleTurnEnv
from verifiers.rubrics.rubric import Rubric

import atroposlib.envs.base as base_env_module
import atroposlib.envs.verifiers_env as verifiers_env_module
from atroposlib.envs.server_handling.server_baseline import APIServerConfig
from atroposlib.envs.server_handling.server_harness import create_chat_completion
from atroposlib.envs.verifiers_env import VerifiersEnv, VfEnvConfig


class _DummyTokenizer:
    eos_token_id = 0

    def apply_chat_template(
        self,
        chat: list[dict[str, Any]],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        **_kwargs,
    ):
        text = "".join(f"<{m.get('role')}> {m.get('content')}\n" for m in chat)
        if add_generation_prompt:
            text += "<assistant> "
        return text if not tokenize else [ord(ch) for ch in text]

    def decode(self, token_ids, **_kwargs) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return "".join(chr(t) for t in token_ids)


def _dummy_reward_func(
    *,
    completion: Any,
    answer: str,
    state: dict[str, Any],
    info: dict[str, Any],
    **kwargs,
) -> float:
    _ = (state, info, kwargs)
    if isinstance(completion, list) and completion:
        completion = (completion[-1] or {}).get("content") or ""
    return 1.0 if answer in str(completion) else 0.0


@pytest.mark.asyncio
async def test_verifiers_env_smoke(monkeypatch, tmp_path):
    monkeypatch.setattr(
        base_env_module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: _DummyTokenizer(),
    )

    ds = Dataset.from_list(
        [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
            {"question": "Q3", "answer": "A3"},
        ]
    )
    dummy_env = SingleTurnEnv(
        dataset=ds,
        system_prompt="SYSTEM",
        rubric=Rubric(funcs=[_dummy_reward_func], weights=[1.0]),
    )

    loaded: dict[str, str] = {}

    def _load_environment(env_id: str, **_env_args):
        loaded["env_id"] = env_id
        return dummy_env

    monkeypatch.setattr(verifiers_env_module.vf, "load_environment", _load_environment)

    env_config = VfEnvConfig(
        vf_env_name="will/wordle",
        env_args={},
        group_size=1,
        ensure_scores_are_not_same=True,  # should be auto-disabled for group_size=1
        include_messages=True,
        use_wandb=False,
        max_token_length=64,
        tokenizer_name="dummy",
        eval_num_examples=2,
        max_eval_workers=2,
        data_dir_to_save_evals=str(tmp_path),
    )
    server_configs = [
        APIServerConfig(
            model_name="test-model",
            base_url="http://localhost:9001/v1",
            api_key="x",
        )
    ]

    env = VerifiersEnv(config=env_config, server_configs=server_configs, testing=True)
    assert env.config.ensure_scores_are_not_same is False
    assert loaded["env_id"] == "wordle"

    await env.setup()
    harness = env.server.servers[0]
    eval_rows = env.eval_ds.select(range(env_config.eval_num_examples))
    for row in eval_rows:
        harness.set_desired_response(
            row["prompt"], create_chat_completion(row["answer"])
        )

    item = await env.get_next_item()
    group, backlog = await env.collect_trajectories(item)
    assert backlog == []
    assert group is not None
    assert group["scores"] == [1.0]
    assert len(group["tokens"]) == 1
    assert len(group["masks"]) == 1
    assert group.get("messages") is not None

    # Structural check: should not raise when validating for API send.
    await env.handle_send_to_api(group, item=item, do_send_to_api=False)

    metrics = await env.evaluate()
    assert metrics["eval/avg_total_score"] == 1.0
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "samples.jsonl").exists()
