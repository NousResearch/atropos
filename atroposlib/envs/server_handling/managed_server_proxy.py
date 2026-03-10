"""
OpenAI-compatible chat completions proxy over ManagedServer.

Exposes /{uuid}/v1/chat/completions and related endpoints so that external
environment microservices can interact with ManagedServer via standard
OpenAI API during multi-step rollouts.

Each UUID maps to a session containing a ManagedServer instance. Tool call
parsing uses vLLM's parsers directly. The ManagedServer always stores raw
text — tool call translation only affects the HTTP wire format.

Uses ServerManager for routing across multiple backend servers (load balancing,
health checks, etc.) — same infrastructure as the rest of atropos.

Usage:
    # Standalone with JSON config
    python -m atroposlib.envs.server_handling.managed_server_proxy \\
        --config servers.json \\
        --port 9100

    # Or mount into existing FastAPI app
    from atroposlib.envs.server_handling.managed_server_proxy import create_app
    app = create_app(server_manager, tokenizer, model_name)

servers.json example:
    {
        "model_name": "Qwen/Qwen3-4B",
        "servers": [
            {"base_url": "http://gpu1:8000/v1", "server_type": "vllm", "api_key": ""},
            {"base_url": "http://gpu2:8000/v1", "server_type": "vllm", "api_key": ""}
        ]
    }
"""

import logging
import time
import uuid as uuid_lib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from atroposlib.envs.server_handling.managed_server import (
    DummyManagedServer,
    ManagedServer,
)
from atroposlib.envs.server_handling.openai_server import OpenAIServer
from atroposlib.envs.server_handling.server_baseline import APIServerConfig
from atroposlib.envs.server_handling.server_manager import ServerManager
from atroposlib.envs.server_handling.tool_call_translator import ToolCallTranslator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ChatCompletionProxyRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = 1024
    temperature: float = 1.0
    n: int = 1
    stop: Optional[List[str]] = None
    tools: Optional[List[dict]] = None
    tool_choice: Optional[Any] = None  # "auto", "none", "required", or dict
    # TODO: top_p, frequency_penalty, presence_penalty, seed, response_format,
    #       logprobs, top_logprobs — pass through to backend when implemented


class SessionCreateRequest(BaseModel):
    tool_parser: str = "hermes"
    track_tree: bool = False
    # Pin to a specific backend server by its base_url. In production,
    # the caller gets their assigned server from the atropos API and
    # passes it here. If omitted, falls back to picking the server
    # with the most open semaphore slots (fine for dev/testing).
    base_url: Optional[str] = None


class SessionCreateResponse(BaseModel):
    uuid: str
    model_name: str
    tool_parser: str
    base_url: Optional[str] = None  # Which backend was selected
    created_at: float


class RenderResponse(BaseModel):
    prompt_text: str
    token_ids: List[int]
    num_tokens: int


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


@dataclass
class SessionState:
    uuid: str
    managed_server: ManagedServer
    translator: ToolCallTranslator
    model_name: str
    base_url: Optional[str] = None  # Which backend server this session is pinned to
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# OpenAI error format
# ---------------------------------------------------------------------------


def openai_error(
    status_code: int, message: str, error_type: str = "invalid_request_error"
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "code": status_code,
            }
        },
    )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    server_manager: ServerManager,
    tokenizer: Any,
    model_name: str = "unknown",
) -> FastAPI:
    """Create the proxy FastAPI app.

    Args:
        server_manager: ServerManager instance managing one or more backend
            servers (VLLMServer, SGLangServer, etc.). Used to pick the most
            available server when creating sessions.
        tokenizer: HuggingFace tokenizer for the model.
        model_name: Model name to report in responses.

    Returns:
        FastAPI app with all endpoints registered.
    """

    app = FastAPI(title="ManagedServer OpenAI Proxy")
    sessions: Dict[str, SessionState] = {}

    # -- helpers --

    def _get_session(session_uuid: str) -> SessionState:
        session = sessions.get(session_uuid)
        if session is None:
            raise HTTPException(
                status_code=404, detail=f"Session {session_uuid} not found"
            )
        session.last_accessed = time.time()
        return session

    def _get_server_base_url(server) -> Optional[str]:
        """Get the base_url from a server's config, if available."""
        if hasattr(server, "config") and hasattr(server.config, "base_url"):
            return server.config.base_url
        return None

    def _select_server(base_url: Optional[str] = None):
        """Pick a server from the manager.

        Args:
            base_url: If provided, pin to the server with this base_url.
                Raises 404 if no server matches.
                If None, picks the most available server (mirrors
                ServerManager.managed_server() logic).

        Returns:
            Selected APIServer instance.
        """
        if base_url is not None:
            # Pin to specific server by base_url
            for server in server_manager.servers:
                server_url = _get_server_base_url(server)
                if server_url and server_url.rstrip("/") == base_url.rstrip("/"):
                    return server
            # No match — list available URLs in error
            available = [
                _get_server_base_url(s)
                for s in server_manager.servers
                if _get_server_base_url(s)
            ]
            raise HTTPException(
                status_code=404,
                detail=f"No server with base_url '{base_url}'. Available: {available}",
            )

        # Auto-select most available
        most_available_idx = 0
        most_available_slots = -1
        for i, server in enumerate(server_manager.servers):
            if not server.server_healthy:
                continue
            if server.sem._value > most_available_slots:
                most_available_idx = i
                most_available_slots = server.sem._value
        return server_manager.servers[most_available_idx]

    def _render_prompt(
        messages: List[Dict[str, Any]],
        tools: Optional[List[dict]] = None,
        translator: Optional[ToolCallTranslator] = None,
    ) -> str:
        """Render messages to prompt text via chat template.

        If a translator is provided, converts OpenAI tool_call messages
        back to raw text first.
        """
        # Convert messages to dicts
        msg_dicts = []
        for msg in messages:
            if isinstance(msg, BaseModel):
                msg_dicts.append(msg.model_dump(exclude_none=True))
            else:
                msg_dicts.append(msg)

        # Reconstruct raw text for assistant tool_call messages
        if translator:
            msg_dicts = translator.convert_messages_for_template(msg_dicts)

        # Build kwargs for apply_chat_template
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if tools:
            template_kwargs["tools"] = tools

        return tokenizer.apply_chat_template(msg_dicts, **template_kwargs)

    def _build_openai_response(
        choices_data: List[dict],
        model: str,
    ) -> dict:
        """Build an OpenAI ChatCompletion response dict."""
        return {
            "id": f"chatcmpl-{uuid_lib.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": choices_data,
        }

    # -- endpoints --

    @app.get("/health")
    async def health():
        healthy_servers = sum(1 for s in server_manager.servers if s.server_healthy)
        return {
            "status": "ok",
            "model": model_name,
            "sessions": len(sessions),
            "servers": len(server_manager.servers),
            "healthy_servers": healthy_servers,
        }

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "atropos",
                }
            ],
        }

    @app.post("/setup")
    async def setup(request: Request):
        """Receive server configuration from ServerManager.

        Accepts the same JSON format as the standalone --config file.
        Replaces the current server_manager's servers with the new config.
        Called by ServerManager at startup to push its config to the proxy.

        Body: {"model_name": "...", "servers": [{"base_url": "...", "server_type": "vllm"}, ...]}
        """
        config = await request.json()

        new_configs = []
        new_model = config.get("model_name", model_name)
        for srv in config.get("servers", []):
            new_configs.append(
                APIServerConfig(
                    model_name=new_model,
                    base_url=srv["base_url"],
                    api_key=srv.get("api_key", ""),
                    server_type=srv.get("server_type", "vllm"),
                    num_max_requests_at_once=srv.get("num_max_requests_at_once", 512),
                    num_requests_for_eval=srv.get("num_requests_for_eval", 64),
                    timeout=srv.get("timeout", 1200),
                    tokenizer_name=config.get("tokenizer_name", "none"),
                )
            )

        if new_configs:
            new_manager = ServerManager(configs=new_configs)
            server_manager.servers = new_manager.servers
            logger.info("Setup: replaced servers with %d new configs", len(new_configs))

        return {
            "status": "ok",
            "servers": len(server_manager.servers),
            "model_name": new_model,
        }

    @app.get("/servers")
    async def list_servers():
        """List available backend servers.

        Useful for discovery/debugging. In production, server allocation
        is managed by the atropos API — the environment gets told which
        server to use and passes that base_url to POST /sessions/create.
        """
        server_list = []
        for i, server in enumerate(server_manager.servers):
            url = _get_server_base_url(server)
            healthy = getattr(server, "server_healthy", True)
            server_list.append(
                {
                    "index": i,
                    "base_url": url,
                    "healthy": healthy,
                    "model_name": (
                        getattr(server.config, "model_name", model_name)
                        if hasattr(server, "config")
                        else model_name
                    ),
                    "server_type": (
                        getattr(server.config, "server_type", "unknown")
                        if hasattr(server, "config")
                        else "unknown"
                    ),
                }
            )
        return {"servers": server_list}

    @app.post("/sessions/create", response_model=SessionCreateResponse)
    async def create_session(request: SessionCreateRequest):
        session_uuid = str(uuid_lib.uuid4())

        # Pick server — pinned to base_url if specified, otherwise most available
        selected_server = _select_server(base_url=request.base_url)
        selected_url = _get_server_base_url(selected_server)

        # Use DummyManagedServer for OpenAI endpoints (no logprobs support)
        if isinstance(selected_server, OpenAIServer):
            logger.info(
                "Session %s using DummyManagedServer (OpenAI endpoint). "
                "Token IDs and logprobs will be placeholders.",
                session_uuid,
            )
            managed = DummyManagedServer(
                server=selected_server,
                tokenizer=tokenizer,
                track_tree=request.track_tree,
            )
        else:
            managed = ManagedServer(
                server=selected_server,
                tokenizer=tokenizer,
                track_tree=request.track_tree,
                tool_parser=request.tool_parser,
            )

        # Translator kept for the render endpoint (prompt preview)
        translator = ToolCallTranslator(
            tokenizer=tokenizer,
            parser_name=request.tool_parser,
        )
        session = SessionState(
            uuid=session_uuid,
            managed_server=managed,
            translator=translator,
            model_name=model_name,
            base_url=selected_url,
        )
        sessions[session_uuid] = session

        return SessionCreateResponse(
            uuid=session_uuid,
            model_name=model_name,
            tool_parser=request.tool_parser,
            base_url=selected_url,
            created_at=session.created_at,
        )

    @app.get("/sessions")
    async def list_sessions():
        return {
            "sessions": [
                {
                    "uuid": s.uuid,
                    "model_name": s.model_name,
                    "base_url": s.base_url,
                    "created_at": s.created_at,
                    "last_accessed": s.last_accessed,
                    "num_nodes": len(
                        s.managed_server.current_nodes
                        if hasattr(s.managed_server, "current_nodes")
                        else s.managed_server.sequences
                    ),
                }
                for s in sessions.values()
            ]
        }

    @app.post("/{session_uuid}/v1/chat/completions")
    async def chat_completions(session_uuid: str, request: ChatCompletionProxyRequest):
        session = _get_session(session_uuid)
        managed = session.managed_server

        if not request.messages:
            return openai_error(400, "messages must not be empty")

        # Convert pydantic messages to dicts
        messages = [msg.model_dump(exclude_none=True) for msg in request.messages]

        # Build kwargs — ManagedServer.chat_completion() handles all tool
        # call logic internally (template rendering, inbound reconstruction,
        # outbound parsing, skip_special_tokens)
        completion_kwargs = {
            "messages": messages,
            "n": request.n,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        if request.stop:
            completion_kwargs["stop"] = request.stop
        if request.tools:
            completion_kwargs["tools"] = request.tools
        if request.tool_choice is not None:
            completion_kwargs["tool_choice"] = request.tool_choice

        try:
            result = await managed.chat_completion(**completion_kwargs)
        except Exception as e:
            logger.exception("Completion failed")
            return openai_error(
                500, f"Completion failed: {e}", error_type="server_error"
            )

        # Convert ChatCompletion to JSON-serializable response
        choices = []
        for choice in result.choices:
            choice_data = {
                "index": choice.index,
                "message": {
                    "role": choice.message.role,
                    "content": choice.message.content,
                },
                "finish_reason": choice.finish_reason,
            }
            if choice.message.tool_calls:
                choice_data["message"]["tool_calls"] = choice.message.tool_calls

            choices.append(choice_data)

        return _build_openai_response(choices, model_name)

    @app.post("/{session_uuid}/v1/chat/completions/render")
    async def render_prompt(session_uuid: str, request: ChatCompletionProxyRequest):
        session = _get_session(session_uuid)

        try:
            prompt_text = _render_prompt(
                messages=request.messages,
                tools=request.tools,
                translator=session.translator,
            )
        except Exception as e:
            return openai_error(400, f"Failed to render prompt: {e}")

        token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

        return RenderResponse(
            prompt_text=prompt_text,
            token_ids=token_ids,
            num_tokens=len(token_ids),
        )

    @app.get("/{session_uuid}/nodes")
    async def get_nodes(session_uuid: str):
        session = _get_session(session_uuid)
        state = session.managed_server.get_state()

        if session.managed_server.track_tree:
            nodes = list(state.get("sequences", {}).values())
        else:
            nodes = state.get("nodes", [])

        return {
            "nodes": [node.model_dump() for node in nodes],
        }

    @app.delete("/{session_uuid}")
    async def delete_session(session_uuid: str):
        session = sessions.pop(session_uuid, None)
        if session is None:
            raise HTTPException(
                status_code=404, detail=f"Session {session_uuid} not found"
            )
        session.managed_server.reset()
        return {"status": "deleted", "uuid": session_uuid}

    # -- exception handler for OpenAI-style errors --

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return openai_error(exc.status_code, exc.detail)

    # NOTE on concurrent access: each UUID is meant to represent a single
    # rollout session used by one caller at a time. If you send concurrent
    # requests to the same UUID, the ManagedServer's node extension logic
    # may get confused because it does prefix matching on current_nodes.
    # Don't do that. UUIDs are cheap, make a new one.

    return app


# ---------------------------------------------------------------------------
# Standalone entrypoint
# ---------------------------------------------------------------------------


def main():
    import argparse
    import json

    import uvicorn
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description="ManagedServer OpenAI Proxy")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to JSON config file with server definitions",
    )
    parser.add_argument("--port", type=int, default=9100, help="Proxy port")
    parser.add_argument("--host", default="0.0.0.0", help="Proxy host")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = json.load(f)

    model_name = config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(config.get("tokenizer_name", model_name))

    # Build APIServerConfigs from the JSON
    server_configs = []
    for srv in config["servers"]:
        server_configs.append(
            APIServerConfig(
                model_name=model_name,
                base_url=srv["base_url"],
                api_key=srv.get("api_key", ""),
                server_type=srv.get("server_type", "vllm"),
                num_max_requests_at_once=srv.get("num_max_requests_at_once", 512),
                num_requests_for_eval=srv.get("num_requests_for_eval", 64),
                timeout=srv.get("timeout", 1200),
                tokenizer_name=config.get("tokenizer_name", "none"),
            )
        )

    server_manager = ServerManager(configs=server_configs)

    app = create_app(
        server_manager=server_manager,
        tokenizer=tokenizer,
        model_name=model_name,
    )

    print(f"Starting ManagedServer OpenAI Proxy on {args.host}:{args.port}")
    print(f"  Model: {model_name}")
    print(f"  Backends: {len(server_configs)} server(s)")
    for i, cfg in enumerate(server_configs):
        print(f"    [{i}] {cfg.base_url} ({cfg.server_type})")
    print()
    print("Endpoints:")
    print("  POST /sessions/create")
    print("  POST /{uuid}/v1/chat/completions")
    print("  POST /{uuid}/v1/chat/completions/render")
    print("  GET  /{uuid}/nodes")
    print("  DELETE /{uuid}")
    print("  GET  /sessions")
    print("  GET  /v1/models")
    print("  GET  /health")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
