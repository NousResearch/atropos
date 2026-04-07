from __future__ import annotations

from types import SimpleNamespace

import pytest

from environments.browserbase.dom_mode import DOMModeAdapter


class FakeActionResult:
    def __init__(self, success=True, message="ok"):
        self.success = success
        self.message = message


class FakeSession:
    def __init__(self):
        self.id = "session-1"
        self.observe_options = None
        self.act_options = None
        self.extract_options = None
        self.ended = False

    async def end(self):
        self.ended = True

    async def navigate(self, url):
        self.last_url = url

    async def observe(self, instruction, options=None):
        self.observe_options = options
        return SimpleNamespace(
            data=SimpleNamespace(
                result=[
                    SimpleNamespace(
                        description="Open the article",
                        selector="text=Open",
                        method="click",
                    )
                ]
            )
        )

    async def act(self, input, options=None):
        self.act_options = options
        return SimpleNamespace(data=SimpleNamespace(result=FakeActionResult()))

    async def extract(self, instruction, schema, options=None):
        self.extract_options = options
        return SimpleNamespace(data=SimpleNamespace(result={"headline": "Browserbase"}))


class FakeStagehandClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.sessions = SimpleNamespace(start=self.start_session)
        self.closed = False
        self.session = FakeSession()

    async def start_session(self, **kwargs):
        self.start_kwargs = kwargs
        return self.session

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_dom_mode_uses_rollout_llm_config(monkeypatch):
    monkeypatch.setattr(
        "environments.browserbase.dom_mode.AsyncStagehand",
        FakeStagehandClient,
    )

    adapter = DOMModeAdapter(
        browserbase_api_key="bb-key",
        project_id="proj-1",
        model_api_key=None,
        stagehand_model="openai/gpt-4o-mini",
        proxy_model_to_stagehand=True,
        proxies=True,
        advanced_stealth=True,
    )

    managed = SimpleNamespace(
        server=SimpleNamespace(
            config=SimpleNamespace(
                model_name="Qwen/Qwen3-4B",
                base_url="http://localhost:9001/v1",
                api_key="x",
            )
        )
    )

    runtime = await adapter.setup_runtime(item={}, managed=managed)
    assert runtime["llm_config"] == {
        "modelName": "Qwen/Qwen3-4B",
        "baseURL": "http://localhost:9001/v1",
        "apiKey": "x",
    }

    observe_message = await adapter.execute_tool_call(
        runtime=runtime,
        tool_name="observe",
        tool_args={"instruction": "find the main CTA"},
        tool_call_id="call-1",
        messages=[],
    )
    assert "Open the article" in observe_message["content"]
    assert runtime["session"].observe_options == {"model": runtime["llm_config"]}

    extract_message = await adapter.execute_tool_call(
        runtime=runtime,
        tool_name="extract",
        tool_args={
            "instruction": "extract the headline",
            "schema_json": '{"type":"object","properties":{"headline":{"type":"string"}}}',
        },
        tool_call_id="call-2",
        messages=[],
    )
    assert "Browserbase" in extract_message["content"]
    assert runtime["session"].extract_options == {"model": runtime["llm_config"]}

    await adapter.cleanup_runtime(runtime)
    assert runtime["session"].ended is True

    await adapter.teardown()
    assert adapter._client is None
