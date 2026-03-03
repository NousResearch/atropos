"""
Client that talks to the ManagedServer OpenAI proxy over HTTP.

Implements the same interface as ManagedServer so it can be used as a
drop-in replacement via ServerManager.managed_server(use_proxy=True).

The proxy handles all the token tracking, tool call parsing, and sequence
management. This client just ferries requests/responses over HTTP and
reconstructs the SequenceNode objects from the JSON.
"""

import logging
import time
import uuid as uuid_lib
from typing import Any, Dict, List, Optional

import aiohttp
from openai.types.chat.chat_completion import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
)
from openai.types.completion import Completion  # noqa: F401 — used in type hint

from atroposlib.envs.server_handling.managed_server import SequenceNode

logger = logging.getLogger(__name__)


class ProxyManagedServer:
    """Client that talks to the ManagedServer OpenAI proxy.

    Same interface as ManagedServer — chat_completion(), completion(),
    get_state(), reset(). But instead of doing token tracking in-process,
    delegates everything to the proxy over HTTP.

    Created by ServerManager.managed_server(use_proxy=True).

    Example:
        async with server_manager.managed_server(use_proxy=True) as managed:
            # Same API as regular ManagedServer
            resp = await managed.chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                n=4, max_tokens=100, temperature=1.0,
            )
            state = managed.get_state()
            nodes = state["nodes"]

            # Extra: get URL for external apps to use directly
            url = managed.get_url()
            # → "http://proxy:9100/{uuid}/v1"
    """

    def __init__(
        self,
        proxy_url: str,
        session_uuid: str,
        model_name: str = "unknown",
        base_url: Optional[str] = None,
    ):
        """
        Args:
            proxy_url: Base URL of the proxy (e.g. "http://localhost:9100")
            session_uuid: UUID of the session on the proxy.
            model_name: Model name (for response objects).
            base_url: The backend server this session is pinned to.
        """
        self.proxy_url = proxy_url.rstrip("/")
        self.session_uuid = session_uuid
        self.model_name = model_name
        self.base_url = base_url

        # Cache for nodes (populated by get_state)
        self._cached_nodes: Optional[List[SequenceNode]] = None

    def get_url(self) -> str:
        """Get the OpenAI-compatible API URL for this session.

        External apps can use this URL with any OpenAI client:
            client = openai.OpenAI(base_url=managed.get_url())
            client.chat.completions.create(messages=..., tools=...)

        Returns:
            URL like "http://proxy:9100/{uuid}/v1"
        """
        return f"{self.proxy_url}/{self.session_uuid}/v1"

    async def _post(self, path: str, json: dict, timeout: int = 300) -> dict:
        """Make a POST request to the proxy."""
        url = f"{self.proxy_url}{path}"
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=json, timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                data = await resp.json()
                if resp.status != 200:
                    error_msg = data.get("error", {}).get("message", str(data))
                    raise RuntimeError(
                        f"Proxy request failed ({resp.status}): {error_msg}"
                    )
                return data

    async def _get(self, path: str, timeout: int = 30) -> dict:
        """Make a GET request to the proxy."""
        url = f"{self.proxy_url}{path}"
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                data = await resp.json()
                if resp.status != 200:
                    error_msg = data.get("error", {}).get("message", str(data))
                    raise RuntimeError(
                        f"Proxy request failed ({resp.status}): {error_msg}"
                    )
                return data

    async def _delete(self, path: str, timeout: int = 30) -> dict:
        """Make a DELETE request to the proxy."""
        url = f"{self.proxy_url}{path}"
        async with aiohttp.ClientSession() as session:
            async with session.delete(
                url, timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                data = await resp.json()
                return data

    async def chat_completion(self, **kwargs) -> ChatCompletion:
        """Send a chat completion request through the proxy.

        Same interface as ManagedServer.chat_completion().
        The proxy handles template rendering, tool call parsing,
        and token/logprob tracking.
        """
        # Convert messages to serializable format
        messages = kwargs.get("messages", [])
        serialized_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                serialized_messages.append(msg)
            else:
                serialized_messages.append(dict(msg))

        body = {
            "messages": serialized_messages,
            "max_tokens": kwargs.get("max_tokens", 1024),
            "temperature": kwargs.get("temperature", 1.0),
            "n": kwargs.get("n", 1),
        }
        if kwargs.get("stop"):
            body["stop"] = kwargs["stop"]
        if kwargs.get("tools"):
            body["tools"] = kwargs["tools"]
        if kwargs.get("tool_choice") is not None:
            body["tool_choice"] = kwargs["tool_choice"]

        data = await self._post(f"/{self.session_uuid}/v1/chat/completions", json=body)

        # Reconstruct ChatCompletion from proxy response
        choices = []
        for choice_data in data.get("choices", []):
            msg = choice_data.get("message", {})
            choice = Choice(
                finish_reason=choice_data.get("finish_reason", "stop"),
                index=choice_data.get("index", 0),
                message=ChatCompletionMessage(
                    content=msg.get("content"),
                    role=msg.get("role", "assistant"),
                ),
            )
            choices.append(choice)

        return ChatCompletion(
            id=data.get("id", str(uuid_lib.uuid4())),
            created=data.get("created", int(time.time())),
            model=data.get("model", self.model_name),
            object="chat.completion",
            choices=choices,
        )

    async def completion(self, **kwargs) -> Completion:
        """Send a completion request through the proxy.

        Note: the proxy's chat/completions endpoint is the primary interface.
        For raw completions, the proxy renders the prompt via chat template
        internally. If you're calling this, you probably want chat_completion()
        instead.
        """
        # For completion() calls, we'd need a /completions endpoint on the proxy.
        # Currently the proxy only exposes chat/completions. For now, raise
        # a clear error.
        raise NotImplementedError(
            "ProxyManagedServer.completion() is not supported. "
            "Use chat_completion() instead — the proxy handles template "
            "rendering internally."
        )

    def get_state(self) -> Dict[str, Any]:
        """Get the current state synchronously from cache.

        Call fetch_state() first to populate from the proxy, or use
        the nodes returned by this method after a chat_completion() call.

        Returns:
            Dict with 'nodes': List[SequenceNode]
        """
        if self._cached_nodes is not None:
            return {"nodes": self._cached_nodes}
        return {"nodes": []}

    async def fetch_state(self) -> Dict[str, Any]:
        """Fetch current state from the proxy (async).

        Returns:
            Dict with 'nodes': List[SequenceNode]
        """
        data = await self._get(f"/{self.session_uuid}/nodes")
        nodes = []
        for node_data in data.get("nodes", []):
            nodes.append(SequenceNode(**node_data))
        self._cached_nodes = nodes
        return {"nodes": nodes}

    def reset(self):
        """Clear cached state. The actual cleanup happens in __aexit__."""
        self._cached_nodes = None

    async def cleanup(self):
        """Delete the session on the proxy."""
        try:
            await self._delete(f"/{self.session_uuid}")
        except Exception as e:
            logger.warning(f"Failed to cleanup proxy session {self.session_uuid}: {e}")

    # -- context manager support --

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Fetch final state before cleanup so callers can still access nodes
        try:
            await self.fetch_state()
        except Exception:
            pass
        await self.cleanup()


async def create_proxy_session(
    proxy_url: str,
    base_url: Optional[str] = None,
    tool_parser: str = "hermes",
    track_tree: bool = False,
    model_name: str = "unknown",
) -> ProxyManagedServer:
    """Create a new session on the proxy and return a ProxyManagedServer.

    Args:
        proxy_url: Base URL of the proxy (e.g. "http://localhost:9100").
        base_url: Pin to a specific backend server. In production, this
            comes from the atropos API's server allocation.
        tool_parser: vLLM tool parser name (default: "hermes").
        track_tree: Whether to use tree mode for tracking.
        model_name: Model name for response objects.

    Returns:
        ProxyManagedServer instance ready to use.
    """
    body = {
        "tool_parser": tool_parser,
        "track_tree": track_tree,
    }
    if base_url:
        body["base_url"] = base_url

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{proxy_url.rstrip('/')}/sessions/create",
            json=body,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            data = await resp.json()
            if resp.status != 200:
                error_msg = data.get("error", {}).get("message", str(data))
                raise RuntimeError(f"Failed to create proxy session: {error_msg}")

    return ProxyManagedServer(
        proxy_url=proxy_url,
        session_uuid=data["uuid"],
        model_name=data.get("model_name", model_name),
        base_url=data.get("base_url"),
    )
