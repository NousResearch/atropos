from __future__ import annotations

import os
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field

from atroposlib.envs import StatefulToolEpisodeEnv, StatefulToolEpisodeEnvConfig

from .cua_mode import CUAModeAdapter
from .dom_mode import DOMModeAdapter


class BrowserbaseEnvConfig(StatefulToolEpisodeEnvConfig):
    mode: Literal["dom", "cua"] = Field(
        default="dom",
        description="Browser control mode.",
    )
    project_id: Optional[str] = Field(
        default=None,
        description="Optional Browserbase project ID override. Falls back to BROWSERBASE_PROJECT_ID.",
    )
    browserbase_api_key_var: str = Field(
        default="BROWSERBASE_API_KEY",
        description="Environment variable name storing the Browserbase API key.",
    )
    model_api_key_var: str = Field(
        default="MODEL_API_KEY",
        description="Environment variable name storing the Stagehand model API key.",
    )
    stagehand_model: str = Field(
        default="openai/gpt-4o-mini",
        description="Stagehand model name for DOM mode sessions.",
    )
    proxy_model_to_stagehand: bool = Field(
        default=False,
        description="Route DOM tool LLM calls through the rollout backend instead of a separate MODEL_API_KEY.",
    )
    use_sandbox: bool = Field(
        default=True,
        description="Reserved CUA config for future sandbox-backed execution.",
    )
    server_url: str = Field(
        default="http://localhost:3000",
        description="Reserved CUA config for a local server endpoint.",
    )
    env: Literal["LOCAL", "BROWSERBASE"] = Field(
        default="BROWSERBASE",
        description="Reserved CUA session environment selector.",
    )
    viewport_width: int = Field(default=1024)
    viewport_height: int = Field(default=768)
    save_screenshots: bool = Field(default=True)
    keep_recent_screenshots: Optional[int] = Field(default=2)
    proxies: bool = Field(default=False)
    advanced_stealth: bool = Field(default=False)
    server_port: int = Field(default=3000)
    server_ready_timeout: int = Field(default=120)
    server_ready_poll_interval: float = Field(default=2.0)
    docker_image: str = Field(default="node:18-slim")
    cpu_cores: int = Field(default=2)
    memory_gb: int = Field(default=4)
    disk_size_gb: int = Field(default=10)
    sandbox_timeout_minutes: int = Field(default=60)
    sandbox_timeout_per_command_seconds: int = Field(default=60)
    use_binary: bool = Field(default=True)
    use_prebuilt_image: bool = Field(default=True)
    prebuilt_image: str = Field(default="deepdream19/cua-server:latest")


class BrowserbaseEnv(StatefulToolEpisodeEnv):
    env_config_cls = BrowserbaseEnvConfig

    def __init__(self, config: BrowserbaseEnvConfig, *args: Any, **kwargs: Any):
        super().__init__(config, *args, **kwargs)
        self.browserbase_api_key = os.getenv(config.browserbase_api_key_var)
        self.project_id = config.project_id or os.getenv("BROWSERBASE_PROJECT_ID")
        self.model_api_key = os.getenv(config.model_api_key_var)

        if not self.browserbase_api_key:
            raise ValueError(
                f"Missing Browserbase API key in env var {config.browserbase_api_key_var}."
            )
        if not self.project_id:
            raise ValueError(
                "Missing Browserbase project ID. Pass config.project_id or set "
                "BROWSERBASE_PROJECT_ID."
            )
        if config.mode == "dom" and (
            not config.proxy_model_to_stagehand and not self.model_api_key
        ):
            raise ValueError(
                f"Missing Stagehand model API key in env var {config.model_api_key_var}."
            )

        if config.mode == "dom":
            self.mode_adapter = DOMModeAdapter(
                browserbase_api_key=self.browserbase_api_key,
                project_id=self.project_id,
                model_api_key=self.model_api_key,
                stagehand_model=config.stagehand_model,
                proxy_model_to_stagehand=config.proxy_model_to_stagehand,
                proxies=config.proxies,
                advanced_stealth=config.advanced_stealth,
            )
        elif config.mode == "cua":
            self.mode_adapter = CUAModeAdapter()
            self.mode_adapter.ensure_supported()
        else:
            raise ValueError(f"Unknown Browserbase mode: {config.mode}")

    async def setup(self) -> None:
        """Concrete subclasses load tasks here."""

    async def evaluate(self, *args: Any, **kwargs: Any) -> Dict[str, float]:
        return {}

    def get_tool_schemas(self) -> List[dict]:
        return self.mode_adapter.tool_schemas()

    async def setup_runtime(self, item: Any, managed: Any) -> Any:
        return await self.mode_adapter.setup_runtime(item=item, managed=managed)

    async def cleanup_runtime(self, runtime: Any) -> None:
        await self.mode_adapter.cleanup_runtime(runtime)

    async def execute_tool_call(
        self,
        runtime: Any,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_call_id: str,
        messages: List[dict],
    ) -> dict | List[dict]:
        return await self.mode_adapter.execute_tool_call(
            runtime=runtime,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_call_id=tool_call_id,
            messages=messages,
        )

    async def teardown(self) -> None:
        teardown = getattr(self.mode_adapter, "teardown", None)
        if teardown is not None:
            await teardown()
