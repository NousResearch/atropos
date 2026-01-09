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
    all_special_ids = [0]

    def apply_chat_template(
        self,
        chat: list[dict[str, Any]],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        **kwargs,
    ):
        text = "".join(
            f"<{msg.get('role')}> {msg.get('content')}\n" for msg in chat
        ) + ("<assistant> " if add_generation_prompt else "")
        if not tokenize:
            return text
        return [ord(ch) for ch in text]

    def decode(self, token_ids, **kwargs) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return "".join(
            "<|eos|>" if t == self.eos_token_id else chr(t) for t in token_ids
        )


def _dummy_reward_func(
    *,
    completion: Any,
    answer: str,
    state: dict[str, Any],
    info: dict[str, Any],
    **kwargs,
) -> float:
    assert isinstance(state, dict)
    assert isinstance(info, dict)
    if isinstance(completion, list):
        text = ""
        for msg in completion:
            if msg.get("role") == "assistant":
                text = str(msg.get("content") or "")
                break
        return 1.0 if answer in text else 0.0
    if isinstance(completion, str):
        return 1.0 if answer in completion else 0.0
    return 0.0


@pytest.mark.asyncio
async def test_verifiers_env_collect_trajectories_chat(monkeypatch):
    monkeypatch.setattr(
        base_env_module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: _DummyTokenizer(),
    )

    ds = Dataset.from_list([{"question": "Q1", "answer": "A1"}])
    dummy_env = SingleTurnEnv(
        dataset=ds,
        system_prompt="SYSTEM",
        rubric=Rubric(funcs=[_dummy_reward_func], weights=[1.0]),
    )

    loaded: dict[str, str] = {}

    def _load_environment(env_id: str, **env_args):
        loaded["env_id"] = env_id
        loaded["env_args"] = str(env_args)
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
    item = await env.get_next_item()

    harness = env.server.servers[0]
    harness.set_desired_response(item["prompt"], create_chat_completion("A1"))

    group, backlog = await env.collect_trajectories(item)
    assert backlog == []
    assert group is not None
    assert group["scores"] == [1.0]
    assert len(group["tokens"]) == 1
    assert len(group["masks"]) == 1
    assert group.get("messages") is not None

    # Structural check: should not raise when validating for API send.
    await env.handle_send_to_api(group, item=item, do_send_to_api=False)


@pytest.mark.asyncio
async def test_verifiers_env_evaluate_writes_metrics(monkeypatch, tmp_path):
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

    monkeypatch.setattr(
        verifiers_env_module.vf, "load_environment", lambda *args, **kwargs: dummy_env
    )

    env_config = VfEnvConfig(
        vf_env_name="will/wordle",
        env_args={},
        group_size=1,
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
    await env.setup()

    harness = env.server.servers[0]
    eval_rows = env.eval_ds.select(range(env_config.eval_num_examples))
    for row in eval_rows:
        prompt, answer, _, _ = env._extract_prompt_answer_task_info(row)
        harness.set_desired_response(prompt, create_chat_completion(answer))

    metrics = await env.evaluate()
    assert metrics["eval/avg_total_score"] == 1.0
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "samples.jsonl").exists()
