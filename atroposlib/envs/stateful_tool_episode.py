from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from pydantic import Field

from atroposlib.envs.base import BaseEnv, BaseEnvConfig, ScoredDataItem
from atroposlib.envs.server_handling.managed_server import DummyManagedServer
from atroposlib.type_definitions import Message

logger = logging.getLogger(__name__)


class StatefulToolEpisodeEnvConfig(BaseEnvConfig):
    max_turns: int = Field(
        default=10,
        description="Maximum number of assistant turns in a single episode.",
    )
    tool_parser: str = Field(
        default="hermes",
        description="ManagedServer tool parser name used for structured tool calls.",
    )
    tool_choice: str = Field(
        default="auto",
        description='Tool choice forwarded to ManagedServer, usually "auto".',
    )
    generation_temperature: float = Field(
        default=1.0,
        description="Sampling temperature for rollout generations.",
    )
    preserve_think_blocks: bool = Field(
        default=True,
        description="Preserve assistant <think> blocks across multi-turn prompt renders.",
    )
    allow_dummy_managed_server: bool = Field(
        default=False,
        description="Allow DummyManagedServer for testing or eval-only workflows.",
    )


class StatefulToolEpisodeEnv(BaseEnv, ABC):
    env_config_cls = StatefulToolEpisodeEnvConfig

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if not getattr(self.config, "tool_parser", None):
            raise ValueError(
                "StatefulToolEpisodeEnv requires config.tool_parser to be set."
            )
        self.server.tool_parser = self.config.tool_parser

    @abstractmethod
    async def build_initial_messages(self, item: Any) -> List[Message]:
        """Build the initial conversation shown to the model."""

    @abstractmethod
    async def setup_runtime(self, item: Any, managed: Any) -> Any:
        """Allocate per-episode runtime state."""

    @abstractmethod
    async def cleanup_runtime(self, runtime: Any) -> None:
        """Clean up per-episode runtime state."""

    @abstractmethod
    def get_tool_schemas(self) -> List[dict]:
        """Return model-visible OpenAI-style tool schemas."""

    @abstractmethod
    async def execute_tool_call(
        self,
        runtime: Any,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_call_id: str,
        messages: List[Message],
    ) -> Message | List[Message]:
        """Execute one tool call against the hidden runtime."""

    @abstractmethod
    async def score_episode(
        self,
        item: Any,
        messages: List[Message],
        runtime: Any,
        termination_reason: str,
    ) -> Tuple[float, Dict[str, Any]]:
        """Return (reward, metadata) for a finished episode."""

    async def get_generation_kwargs(self, item: Any) -> Dict[str, Any]:
        return {
            "n": 1,
            "max_tokens": self.config.max_token_length,
            "temperature": self.config.generation_temperature,
        }

    def _assistant_message_to_dict(self, assistant_message: Any) -> Message:
        message: Message = {
            "role": "assistant",
            "content": assistant_message.content,
        }
        tool_calls = getattr(assistant_message, "tool_calls", None)
        if tool_calls:
            message["tool_calls"] = list(tool_calls)
        return message

    def _tool_message(
        self,
        tool_call_id: str,
        content: Any,
        name: Optional[str] = None,
    ) -> Message:
        message: Message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }
        if name is not None:
            message["name"] = name
        return message

    def _normalize_tool_messages(
        self,
        tool_result: Message | List[Message],
        tool_call_id: str,
        tool_name: str,
    ) -> List[Message]:
        if isinstance(tool_result, list):
            normalized = tool_result
        else:
            normalized = [tool_result]

        for message in normalized:
            message.setdefault("role", "tool")
            message.setdefault("tool_call_id", tool_call_id)
            message.setdefault("name", tool_name)
        return normalized

    @asynccontextmanager
    async def open_managed_server(self) -> AsyncIterator[Any]:
        async with self.server.managed_server(
            tokenizer=self.tokenizer,
            preserve_think_blocks=self.config.preserve_think_blocks,
        ) as managed:
            if (
                isinstance(managed, DummyManagedServer)
                and not self.config.allow_dummy_managed_server
            ):
                raise NotImplementedError(
                    "DummyManagedServer is not suitable for Browserbase training. "
                    "Use a backend with real token/logprob support."
                )
            yield managed

    async def collect_trajectory(
        self, item: Any
    ) -> Tuple[Optional[ScoredDataItem], List[Any]]:
        runtime = None
        messages = await self.build_initial_messages(item)
        tools = self.get_tool_schemas()
        termination_reason = "unknown"

        try:
            async with self.open_managed_server() as managed:
                runtime = await self.setup_runtime(item, managed)

                for _ in range(self.config.max_turns):
                    generation_kwargs = await self.get_generation_kwargs(item)
                    completion = await managed.chat_completion(
                        messages=messages,
                        tools=tools,
                        tool_choice=self.config.tool_choice,
                        **generation_kwargs,
                    )

                    assistant_message = completion.choices[0].message
                    assistant_dict = self._assistant_message_to_dict(assistant_message)
                    messages.append(assistant_dict)

                    tool_calls = getattr(assistant_message, "tool_calls", None) or []
                    if not tool_calls:
                        termination_reason = "assistant_final"
                        break

                    for tool_call in tool_calls:
                        tool_name = tool_call["function"]["name"]
                        tool_call_id = tool_call["id"]
                        raw_arguments = tool_call["function"].get("arguments", "{}")

                        try:
                            parsed_args = json.loads(raw_arguments)
                            if not isinstance(parsed_args, dict):
                                raise ValueError(
                                    f"Expected dict tool args, got {type(parsed_args).__name__}"
                                )
                        except Exception as exc:
                            messages.append(
                                self._tool_message(
                                    tool_call_id=tool_call_id,
                                    content=f"Tool argument parse error: {exc}",
                                    name=tool_name,
                                )
                            )
                            continue

                        try:
                            tool_result = await self.execute_tool_call(
                                runtime=runtime,
                                tool_name=tool_name,
                                tool_args=parsed_args,
                                tool_call_id=tool_call_id,
                                messages=messages,
                            )
                            messages.extend(
                                self._normalize_tool_messages(
                                    tool_result=tool_result,
                                    tool_call_id=tool_call_id,
                                    tool_name=tool_name,
                                )
                            )
                        except Exception as exc:
                            logger.exception("Tool execution failed: %s", tool_name)
                            messages.append(
                                self._tool_message(
                                    tool_call_id=tool_call_id,
                                    content=f"Tool execution error: {exc}",
                                    name=tool_name,
                                )
                            )
                else:
                    termination_reason = "max_turns"

                reward, overrides = await self.score_episode(
                    item=item,
                    messages=messages,
                    runtime=runtime,
                    termination_reason=termination_reason,
                )

                nodes = managed.get_state()["nodes"]
                if not nodes:
                    raise RuntimeError("ManagedServer returned no tracked nodes.")
                node = nodes[-1]

                scored_item: ScoredDataItem = {
                    "tokens": node.tokens,
                    "masks": node.masked_tokens,
                    "scores": reward,
                    "inference_logprobs": node.logprobs,
                    "overrides": overrides,
                    "messages": messages if self.config.include_messages else None,
                    "advantages": None,
                    "ref_logprobs": None,
                    "images": None,
                    "group_overrides": None,
                    "distill_token_ids": None,
                    "distill_logprobs": None,
                }
                return scored_item, []
        finally:
            if runtime is not None:
                await self.cleanup_runtime(runtime)
