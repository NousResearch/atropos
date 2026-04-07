from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

from atroposlib.type_definitions import Message

try:
    from stagehand import AsyncStagehand
except ImportError:  # pragma: no cover - exercised via constructor error path
    AsyncStagehand = None  # type: ignore[assignment]


class DOMModeAdapter:
    def __init__(
        self,
        browserbase_api_key: str,
        project_id: str,
        model_api_key: Optional[str],
        stagehand_model: str,
        proxy_model_to_stagehand: bool,
        proxies: bool,
        advanced_stealth: bool,
    ):
        if AsyncStagehand is None:
            raise ImportError(
                "Browserbase DOM mode requires stagehand. Install `atroposlib[browser]`."
            )

        self.browserbase_api_key = browserbase_api_key
        self.project_id = project_id
        self.model_api_key = model_api_key
        self.stagehand_model = stagehand_model
        self.proxy_model_to_stagehand = proxy_model_to_stagehand
        self.proxies = proxies
        self.advanced_stealth = advanced_stealth
        self._client = None
        self._client_lock = asyncio.Lock()

    def tool_schemas(self) -> List[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "navigate",
                    "description": "Navigate the current browser session to a URL.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "Absolute URL to navigate to.",
                            }
                        },
                        "required": ["url"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "observe",
                    "description": "Find candidate actions on the current page that match an instruction.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "instruction": {
                                "type": "string",
                                "description": "Natural-language description of the element or action to find.",
                            }
                        },
                        "required": ["instruction"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "act",
                    "description": "Perform a natural-language browser action on the current page.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "instruction": {
                                "type": "string",
                                "description": "Precise natural-language action to perform.",
                            }
                        },
                        "required": ["instruction"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "extract",
                    "description": "Extract structured data from the current page using a JSON schema.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "instruction": {
                                "type": "string",
                                "description": "What data to extract from the page.",
                            },
                            "schema_json": {
                                "type": "string",
                                "description": "JSON-serialized schema describing the expected structure.",
                            },
                        },
                        "required": ["instruction", "schema_json"],
                        "additionalProperties": False,
                    },
                },
            },
        ]

    async def _get_client(self, model_api_key: str):
        async with self._client_lock:
            if self._client is None:
                self._client = AsyncStagehand(
                    browserbase_api_key=self.browserbase_api_key,
                    browserbase_project_id=self.project_id,
                    model_api_key=model_api_key,
                )
        return self._client

    def _build_browserbase_params(self) -> Optional[Dict[str, Any]]:
        browserbase_params: Dict[str, Any] = {}
        if self.proxies:
            browserbase_params["proxies"] = self.proxies
        if self.advanced_stealth:
            browserbase_params["browserSettings"] = {"advancedStealth": True}
        return browserbase_params or None

    def _build_llm_config(self, managed: Any) -> Optional[Dict[str, Any]]:
        if not self.proxy_model_to_stagehand:
            return None

        server_config = getattr(getattr(managed, "server", None), "config", None)
        if server_config is None:
            return None

        llm_config: Dict[str, Any] = {"modelName": server_config.model_name}
        base_url = getattr(server_config, "base_url", None)
        api_key = getattr(server_config, "api_key", None)

        if base_url and "api.openai.com" not in str(base_url):
            llm_config["baseURL"] = str(base_url)
        if api_key:
            llm_config["apiKey"] = api_key
        return llm_config

    async def setup_runtime(self, item: Any, managed: Any) -> Dict[str, Any]:
        llm_config = self._build_llm_config(managed)
        runtime_model_key = self.model_api_key
        if runtime_model_key is None and llm_config is not None:
            runtime_model_key = llm_config.get("apiKey")

        if not runtime_model_key:
            raise ValueError(
                "No model API key available for Stagehand. Set the configured "
                "model_api_key_var or use a rollout backend with an api_key when "
                "proxy_model_to_stagehand=True."
            )

        client = await self._get_client(runtime_model_key)
        session = await client.sessions.start(
            model_name=self.stagehand_model,
            browserbase_session_create_params=self._build_browserbase_params(),
        )
        return {
            "session": session,
            "stagehand_session_id": session.id,
            "llm_config": llm_config,
        }

    async def cleanup_runtime(self, runtime: Dict[str, Any]) -> None:
        session = runtime.get("session")
        if session is None:
            return
        try:
            await session.end()
        except Exception:
            pass

    async def teardown(self) -> None:
        if self._client is None:
            return
        try:
            await self._client.close()
        except Exception:
            pass
        finally:
            self._client = None

    async def execute_tool_call(
        self,
        runtime: Dict[str, Any],
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_call_id: str,
        messages: List[Message],
    ) -> Message:
        session = runtime["session"]
        llm_config = runtime.get("llm_config")

        if tool_name == "navigate":
            content = await self.navigate(url=tool_args["url"], session=session)
        elif tool_name == "observe":
            content = await self.observe(
                instruction=tool_args["instruction"],
                session=session,
                llm_config=llm_config,
            )
        elif tool_name == "act":
            content = await self.act(
                instruction=tool_args["instruction"],
                session=session,
                llm_config=llm_config,
            )
        elif tool_name == "extract":
            content = await self.extract(
                instruction=tool_args["instruction"],
                schema_json=tool_args["schema_json"],
                session=session,
                llm_config=llm_config,
            )
        else:
            raise ValueError(f"Unknown DOM tool: {tool_name}")

        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": content,
        }

    async def navigate(self, url: str, session: Any) -> str:
        try:
            await session.navigate(url=url)
            return f"Navigated to {url}"
        except Exception as exc:
            return f"Error navigating to {url}: {exc}"

    async def observe(
        self,
        instruction: str,
        session: Any,
        llm_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        try:
            if llm_config:
                response = await session.observe(
                    instruction=instruction,
                    options={"model": llm_config},
                )
            else:
                response = await session.observe(instruction=instruction)

            actions = [
                {
                    "description": action.description,
                    "selector": action.selector,
                    "method": action.method,
                }
                for action in response.data.result
            ]
            if not actions:
                return "No matching elements found"
            return json.dumps(actions, indent=2)
        except Exception as exc:
            return f"Error observing page: {exc}"

    async def act(
        self,
        instruction: str,
        session: Any,
        llm_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        try:
            if llm_config:
                response = await session.act(
                    input=instruction,
                    options={"model": llm_config},
                )
            else:
                response = await session.act(input=instruction)
            result = response.data.result
            status = "Success" if result.success else "Failed"
            return f"{status}: {result.message}"
        except Exception as exc:
            return f"Error executing action: {exc}"

    async def extract(
        self,
        instruction: str,
        schema_json: str,
        session: Any,
        llm_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        try:
            schema = json.loads(schema_json)
            if llm_config:
                response = await session.extract(
                    instruction=instruction,
                    schema=schema,
                    options={"model": llm_config},
                )
            else:
                response = await session.extract(instruction=instruction, schema=schema)
            return json.dumps(response.data.result, indent=2)
        except json.JSONDecodeError as exc:
            return f"Error parsing schema JSON: {exc}"
        except Exception as exc:
            return f"Error extracting data: {exc}"
