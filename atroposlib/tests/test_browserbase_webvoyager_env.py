from __future__ import annotations

import json
from pathlib import Path

import pytest

from atroposlib.envs.base import APIServerConfig
from environments.browserbase.webvoyager_env import (
    WebVoyagerBrowserbaseEnv,
    WebVoyagerBrowserbaseEnvConfig,
)


class DummyTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return [0] if tokenize else "dummy"


class FakeSession:
    def __init__(self):
        self.id = "session-1"

    async def end(self):
        return None


class FakeStagehandClient:
    def __init__(self, **kwargs):
        self.sessions = type(
            "Sessions",
            (),
            {"start": self.start_session},
        )()
        self.session = FakeSession()

    async def start_session(self, **kwargs):
        return self.session

    async def close(self):
        return None


@pytest.mark.asyncio
async def test_webvoyager_env_loads_filtered_tasks_and_scores(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(
        "atroposlib.envs.base.AutoTokenizer.from_pretrained",
        lambda _: DummyTokenizer(),
    )
    monkeypatch.setattr(
        "environments.browserbase.dom_mode.AsyncStagehand",
        FakeStagehandClient,
    )

    dataset_path = tmp_path / "webvoyager.jsonl"
    rows = [
        {
            "ques": "Find the Browserbase docs",
            "web": "https://browserbase.com",
            "id": "wv-1",
            "web_name": "Browserbase",
        },
        {
            "ques": "Ignore this other site",
            "web": "https://example.com",
            "id": "wv-2",
            "web_name": "Example",
        },
    ]
    dataset_path.write_text("\n".join(json.dumps(row) for row in rows))

    monkeypatch.setenv("BROWSERBASE_API_KEY", "bb-key")
    monkeypatch.setenv("BROWSERBASE_PROJECT_ID", "proj-1")
    monkeypatch.setenv("MODEL_API_KEY", "model-key")

    config = WebVoyagerBrowserbaseEnvConfig(
        tokenizer_name="dummy-tokenizer",
        dataset_path=str(dataset_path),
        web_filter="Browserbase",
        num_examples=1,
        include_messages=True,
    )
    env = WebVoyagerBrowserbaseEnv(
        config=config,
        server_configs=[
            APIServerConfig(
                model_name="test-model",
                base_url="http://localhost:9001/v1",
                api_key="x",
                server_type="vllm",
            )
        ],
        testing=True,
    )

    await env.setup()
    item = await env.get_next_item()

    assert item["task_id"] == "wv-1"
    assert item["website"] == "Browserbase"

    messages = await env.build_initial_messages(item)
    assert "Start URL: https://browserbase.com" in messages[1]["content"]

    async def fake_judge(prompt: str) -> str:
        return "yes"

    env._judge_completion = fake_judge  # type: ignore[method-assign]
    score, overrides = await env.score_episode(
        item=item,
        messages=[
            messages[0],
            messages[1],
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1"}]},
            {
                "role": "tool",
                "name": "navigate",
                "tool_call_id": "call-1",
                "content": "Navigated to https://browserbase.com",
            },
            {"role": "assistant", "content": "The docs are on browserbase.com/docs"},
        ],
        runtime={},
        termination_reason="assistant_final",
    )

    assert score == 1.0
    assert overrides["termination_reason"] == "assistant_final"
    assert overrides["task_id"] == "wv-1"
