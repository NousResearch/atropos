from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pytest

from atroposlib.envs.base import APIServerConfig
from atroposlib.envs.server_handling.managed_server import SequenceNode
from atroposlib.envs.stateful_tool_episode import (
    StatefulToolEpisodeEnv,
    StatefulToolEpisodeEnvConfig,
)


class DummyTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        text = "".join(
            f"<{message['role']}>{message.get('content') or ''}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            text += "<assistant>"
        if tokenize:
            return list(range(len(text)))
        return text


class ExampleEpisodeConfig(StatefulToolEpisodeEnvConfig):
    pass


class ExampleEpisodeEnv(StatefulToolEpisodeEnv):
    env_config_cls = ExampleEpisodeConfig

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.cleaned_runtime_ids: List[str] = []
        self.executed_tools: List[Tuple[str, Dict[str, Any]]] = []

    async def setup(self) -> None:
        return None

    async def get_next_item(self) -> Dict[str, Any]:
        return {"prompt": "unused"}

    async def evaluate(self, *args: Any, **kwargs: Any) -> Dict[str, float]:
        return {}

    async def build_initial_messages(self, item: Any):
        return [{"role": "user", "content": item["prompt"]}]

    async def setup_runtime(self, item: Any, managed: Any) -> Dict[str, Any]:
        return {"runtime_id": "runtime-1"}

    async def cleanup_runtime(self, runtime: Dict[str, Any]) -> None:
        self.cleaned_runtime_ids.append(runtime["runtime_id"])

    def get_tool_schemas(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Lookup test data.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ]

    async def execute_tool_call(
        self,
        runtime: Any,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_call_id: str,
        messages,
    ):
        self.executed_tools.append((tool_name, tool_args))
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": f"lookup-result:{tool_args['query']}",
        }

    async def score_episode(
        self,
        item: Any,
        messages,
        runtime: Any,
        termination_reason: str,
    ):
        return (
            1.0,
            {
                "termination_reason": termination_reason,
                "message_count": len(messages),
            },
        )


class FakeManagedServer:
    def __init__(self):
        self.responses = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": '{"query": "browserbase"}',
                                    },
                                }
                            ],
                        )
                    )
                ]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="Task complete.",
                            tool_calls=None,
                        )
                    )
                ]
            ),
        ]
        self.nodes = [
            SequenceNode(
                full_text="mock",
                tokens=[1, 2, 3, 4],
                masked_tokens=[-100, -100, 3, 4],
                logprobs=[1.0, 1.0, -0.1, -0.2],
                metadata={"finish_reason": "stop"},
            )
        ]

    async def chat_completion(self, **kwargs):
        return self.responses.pop(0)

    def get_state(self):
        return {"nodes": self.nodes}


@pytest.mark.asyncio
async def test_stateful_tool_episode_collects_multiturn_rollout(monkeypatch):
    monkeypatch.setattr(
        "atroposlib.envs.base.AutoTokenizer.from_pretrained",
        lambda _: DummyTokenizer(),
    )

    config = ExampleEpisodeConfig(
        tokenizer_name="dummy-tokenizer",
        include_messages=True,
        max_turns=3,
    )
    env = ExampleEpisodeEnv(
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

    fake_managed = FakeManagedServer()

    @asynccontextmanager
    async def fake_managed_server(**kwargs):
        yield fake_managed

    env.server.managed_server = fake_managed_server  # type: ignore[method-assign]

    scored_item, backlog = await env.collect_trajectory({"prompt": "Find Browserbase"})

    assert backlog == []
    assert scored_item is not None
    assert scored_item["scores"] == 1.0
    assert scored_item["inference_logprobs"] == [1.0, 1.0, -0.1, -0.2]
    assert env.executed_tools == [("lookup", {"query": "browserbase"})]
    assert env.cleaned_runtime_ids == ["runtime-1"]

    messages = scored_item["messages"]
    assert messages is not None
    assert messages[1]["role"] == "assistant"
    assert messages[1]["tool_calls"][0]["function"]["name"] == "lookup"
    assert messages[2]["role"] == "tool"
    assert messages[2]["content"] == "lookup-result:browserbase"
    assert messages[-1]["content"] == "Task complete."
    assert scored_item["overrides"]["termination_reason"] == "assistant_final"
