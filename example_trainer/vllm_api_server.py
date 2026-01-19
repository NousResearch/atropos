#!/usr/bin/env python3
"""
Custom vLLM API server with CUDA IPC shared memory support.

This server extends the standard vLLM API with:
- Single-copy mode: Exports CUDA IPC handles so trainer can share vLLM's tensors
- LoRA hot-swap without server restart
- Bridge endpoints for coordination

ARCHITECTURE (Single-Copy Mode):
    When VLLM_ENABLE_SHARED_WEIGHTS=1:
    1. vLLM's GPUModelRunner is patched BEFORE loading
    2. Patched runner exports CUDA IPC handles to vllm_bridge_config.json
    3. Trainer reads IPC handles and attaches to the SAME tensors
    4. optimizer.step() updates weights in-place - vLLM sees changes immediately!

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    SINGLE GPU (True Shared Memory)                      │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │                    Model Weights (ONE copy!)                     │   │
    │  │              (accessible via CUDA IPC handles)                   │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    │           ▲                                          ▲                 │
    │           │ Reads (inference)                        │ Writes (train)  │
    │  ┌────────┴────────┐                     ┌───────────┴───────────┐    │
    │  │  vLLM Worker    │                     │  Trainer Process      │    │
    │  │                 │                     │  (attached via IPC)   │    │
    │  └─────────────────┘                     └───────────────────────┘    │
    └─────────────────────────────────────────────────────────────────────────┘

CRITICAL: Patches must be applied BEFORE importing vLLM!
"""

# =============================================================================
# STEP 0: Standard library imports ONLY (no vLLM yet!)
# =============================================================================
import asyncio
import json
import os
import ssl
import sys
import threading
from argparse import Namespace
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# =============================================================================
# STEP 1: Apply patches BEFORE any vLLM imports!
# =============================================================================


def _apply_patches_early() -> bool:
    """
    Apply vLLM patches if shared weights are enabled.

    This MUST be called before any vLLM imports!
    Returns True if patches were applied.
    """
    enable_shared = os.environ.get("VLLM_ENABLE_SHARED_WEIGHTS", "0") == "1"
    num_inference_nodes = int(os.environ.get("NUM_INFERENCE_NODES", "-1"))

    if not enable_shared and num_inference_nodes < 0:
        print("[vLLM Server] Shared weights not enabled, skipping patches")
        return False

    print("[vLLM Server] VLLM_ENABLE_SHARED_WEIGHTS=1, applying patches...")

    try:
        # Try relative import first (when run as module)
        try:
            from .vllm_patching import apply_patches
        except ImportError:
            # Fall back to absolute import (when run as script)
            script_dir = Path(__file__).parent
            if str(script_dir) not in sys.path:
                sys.path.insert(0, str(script_dir))
            from vllm_patching import apply_patches

        success = apply_patches()
        if success:
            print("[vLLM Server] ✓ vLLM patches applied successfully!")
        else:
            print("[vLLM Server] ✗ Failed to apply patches")
        return success

    except ImportError as e:
        print(f"[vLLM Server] Could not import vllm_patching: {e}")
        print("[vLLM Server] Shared memory weight updates will not be available")
        return False
    except Exception as e:
        print(f"[vLLM Server] Error applying patches: {e}")
        import traceback

        traceback.print_exc()
        return False


# Apply patches NOW, before any vLLM imports below!
PATCHES_APPLIED = _apply_patches_early()


# =============================================================================
# STEP 2: Now safe to import vLLM (patches are already in place)
# =============================================================================

import torch  # noqa: E402
import vllm.envs as envs  # noqa: E402
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import JSONResponse, Response, StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from vllm.engine.arg_utils import AsyncEngineArgs  # noqa: E402
from vllm.entrypoints.launcher import serve_http  # noqa: E402
from vllm.entrypoints.utils import with_cancellation  # noqa: E402
from vllm.logger import init_logger  # noqa: E402
from vllm.sampling_params import RequestOutputKind, SamplingParams  # noqa: E402
from vllm.usage.usage_lib import UsageContext  # noqa: E402
from vllm.utils import random_uuid  # noqa: E402
from vllm.v1.engine.async_llm import AsyncLLM  # noqa: E402

try:
    from vllm.utils.argparse_utils import FlexibleArgumentParser  # noqa: E402
    from vllm.utils.system_utils import set_ulimit  # noqa: E402
except ImportError:
    from vllm.utils import FlexibleArgumentParser, set_ulimit  # noqa: E402

from vllm.outputs import RequestOutput  # noqa: F401, E402
from vllm.version import __version__ as VLLM_VERSION  # noqa: E402

logger = init_logger("vllm.entrypoints.api_server")


# =============================================================================
# Global State
# =============================================================================

app = FastAPI()
engine: Optional[AsyncLLM] = None


@dataclass
class BridgeState:
    """State for shared memory and LoRA."""

    update_count: int = 0
    last_update_time: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    # LoRA state
    active_lora_path: Optional[str] = None
    lora_load_count: int = 0


bridge_state = BridgeState()


# =============================================================================
# Pydantic Models for API
# =============================================================================


class BridgeInfoResponse(BaseModel):
    enabled: bool
    update_count: int
    last_update_time: float
    model_name: str
    device: str


class LoraLoadRequest(BaseModel):
    adapter_path: str
    adapter_name: Optional[str] = None


class LoraStatusResponse(BaseModel):
    active_adapter: Optional[str]
    load_count: int
    available_adapters: List[str]


# =============================================================================
# Health Endpoints
# =============================================================================


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.get("/health_generate")
async def health_generate() -> Response:
    """Health check that verifies model can generate."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    sampling_params = SamplingParams()
    request_id = random_uuid()

    try:
        results_generator = engine.generate(
            {"prompt_token_ids": [0]}, sampling_params, request_id
        )
        async for _ in results_generator:
            pass
        return Response(status_code=200)
    except asyncio.CancelledError:
        return Response(status_code=499)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Generation Endpoints
# =============================================================================


@app.post("/generate")
async def generate(request: Request) -> Response:
    """
    Generate completion for the request.

    The request should be a JSON object with:
    - prompt: the prompt to use for generation
    - stream: whether to stream results
    - other fields: sampling parameters
    """
    request_dict = await request.json()
    return await _generate(request_dict, raw_request=request)


@with_cancellation
async def _generate(request_dict: dict, raw_request: Request) -> Response:
    """Internal generate handler."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    request_dict["output_kind"] = RequestOutputKind.FINAL_ONLY
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()

    results_generator = engine.generate(prompt, sampling_params, request_id)

    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            prompt = request_output.prompt
            assert prompt is not None
            text_outputs = [prompt + output.text for output in request_output.outputs]
            ret = {"text": text_outputs}
            yield (json.dumps(ret) + "\n").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())

    final_output = None
    try:
        async for request_output in results_generator:
            final_output = request_output
    except asyncio.CancelledError:
        return Response(status_code=499)

    assert final_output is not None
    prompt = final_output.prompt or engine.tokenizer.decode(
        final_output.prompt_token_ids
    )

    text_outputs = [output.text for output in final_output.outputs]
    finish_reasons = [output.finish_reason for output in final_output.outputs]
    ret = {"text": text_outputs, "prompt": prompt, "finish_reasons": finish_reasons}

    if sampling_params.logprobs is not None:
        output_logprobs = [
            [
                [{key: value.logprob for key, value in logprob.items()}]
                for logprob in x.logprobs
            ]
            for x in final_output.outputs
        ]
        ret["logprobs"] = output_logprobs
        ret["prompt_token_ids"] = final_output.prompt_token_ids
        ret["token_ids"] = [x.token_ids for x in final_output.outputs]

    return JSONResponse(ret)


# =============================================================================
# OpenAI-Compatible Chat Completions Endpoint
# =============================================================================


@app.post("/v1/chat/completions")
async def openai_chat_completions(request: Request) -> Response:
    """
    OpenAI-compatible chat completions endpoint.

    This is a thin wrapper around our /generate endpoint that formats
    the request/response to match OpenAI's chat completions API.

    Used by atroposlib/GSM8k environment for rollout generation.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    import time as time_module

    request_dict = await request.json()

    # Extract parameters
    model = request_dict.get("model", "")
    messages = request_dict.get("messages", [])
    max_tokens = request_dict.get("max_tokens", 256)
    temperature = request_dict.get("temperature", 1.0)
    top_p = request_dict.get("top_p", 1.0)
    n = request_dict.get("n", 1)
    stop = request_dict.get("stop", None)
    presence_penalty = request_dict.get("presence_penalty", 0.0)
    frequency_penalty = request_dict.get("frequency_penalty", 0.0)

    # Convert messages to prompt using tokenizer's chat template
    try:
        prompt = engine.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback: simple concatenation if no chat template
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt += f"{role}: {content}\n"
        prompt += "assistant: "

    # Build sampling params (reusing our existing logic)
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        n=n,
        stop=stop,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
    )

    request_id = random_uuid()

    final_output = None
    async for request_output in engine.generate(prompt, sampling_params, request_id):
        final_output = request_output

    if final_output is None:
        raise HTTPException(status_code=500, detail="Generation failed")

    # Build choices in OpenAI chat format
    choices = []
    for idx, output in enumerate(final_output.outputs):
        choices.append(
            {
                "index": idx,
                "message": {
                    "role": "assistant",
                    "content": output.text,
                },
                "finish_reason": output.finish_reason or "stop",
            }
        )

    # Build response
    prompt_tokens = len(final_output.prompt_token_ids)
    completion_tokens = sum(len(o.token_ids) for o in final_output.outputs)

    response = {
        "id": f"chatcmpl-{random_uuid()}",
        "object": "chat.completion",
        "created": int(time_module.time()),
        "model": model,
        "choices": choices,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }

    return JSONResponse(response)


@app.post("/v1/completions")
async def openai_completions(request: Request) -> Response:
    """
    OpenAI-compatible text completions endpoint.

    This is the non-chat version of completions (raw text in, text out).
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    import time as time_module

    request_dict = await request.json()

    # Extract parameters
    model = request_dict.get("model", "")
    prompt = request_dict.get("prompt", "")
    max_tokens = request_dict.get("max_tokens", 256)
    temperature = request_dict.get("temperature", 1.0)
    top_p = request_dict.get("top_p", 1.0)
    n = request_dict.get("n", 1)
    stop = request_dict.get("stop", None)
    presence_penalty = request_dict.get("presence_penalty", 0.0)
    frequency_penalty = request_dict.get("frequency_penalty", 0.0)
    logprobs_requested = request_dict.get("logprobs", None)

    # Handle single prompt or list of prompts
    prompts = [prompt] if isinstance(prompt, str) else prompt

    # Build sampling params
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        n=n,
        stop=stop,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        logprobs=logprobs_requested,
    )

    # Generate for each prompt
    all_choices = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for prompt_text in prompts:
        request_id = random_uuid()

        final_output = None
        async for request_output in engine.generate(
            prompt_text, sampling_params, request_id
        ):
            final_output = request_output

        if final_output is None:
            raise HTTPException(status_code=500, detail="Generation failed")

        # Count tokens
        total_prompt_tokens += len(final_output.prompt_token_ids)

        # Build choices
        for output in final_output.outputs:
            total_completion_tokens += len(output.token_ids)

            choice = {
                "text": output.text,
                "index": len(all_choices),
                "logprobs": None,
                "finish_reason": output.finish_reason or "stop",
            }

            # Add logprobs if requested
            if logprobs_requested is not None and output.logprobs:
                choice["logprobs"] = {
                    "tokens": [
                        engine.tokenizer.decode([tid]) for tid in output.token_ids
                    ],
                    "token_logprobs": [
                        list(lp.values())[0].logprob if lp else None
                        for lp in output.logprobs
                    ],
                    "top_logprobs": None,
                    "text_offset": [],
                }

            all_choices.append(choice)

    # Build response in OpenAI format
    response = {
        "id": f"cmpl-{random_uuid()}",
        "object": "text_completion",
        "created": int(time_module.time()),
        "model": model,
        "choices": all_choices,
        "usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
        },
    }

    return JSONResponse(response)


# =============================================================================
# Bridge Endpoints (Weight Synchronization)
# =============================================================================


@app.get("/bridge/info")
async def bridge_info() -> JSONResponse:
    """Get bridge status and configuration."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    model_name = (
        str(engine.model_config.model) if hasattr(engine, "model_config") else "unknown"
    )

    return JSONResponse(
        {
            "enabled": PATCHES_APPLIED,
            "shared_weights": PATCHES_APPLIED,
            "update_count": bridge_state.update_count,
            "last_update_time": bridge_state.last_update_time,
            "model_name": model_name,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
    )


@app.get("/bridge/state_dict_info")
async def bridge_state_dict_info() -> JSONResponse:
    """Get model parameter information."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    # Basic model info
    try:
        model_config = engine.model_config
        return JSONResponse(
            {
                "model": str(model_config.model),
                "dtype": str(model_config.dtype),
                "shared_weights_enabled": PATCHES_APPLIED,
            }
        )
    except Exception as e:
        return JSONResponse({"error": str(e)})


# =============================================================================
# Pause/Resume Endpoints
# =============================================================================


@app.post("/bridge/pause")
async def bridge_pause() -> JSONResponse:
    """Pause generation to allow weight updates."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        # vLLM v1 supports pause/resume
        if hasattr(engine, "_pause_cond"):
            async with engine._pause_cond:
                engine._paused = True
            logger.info("Engine paused")
            return JSONResponse({"status": "paused"})
        else:
            return JSONResponse({"status": "not_supported"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/bridge/resume")
async def bridge_resume() -> JSONResponse:
    """Resume generation after weight updates."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        if hasattr(engine, "_pause_cond"):
            async with engine._pause_cond:
                engine._paused = False
                engine._pause_cond.notify_all()
            logger.info("Engine resumed")
            return JSONResponse({"status": "resumed"})
        else:
            return JSONResponse({"status": "not_supported"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/bridge/is_paused")
async def bridge_is_paused() -> JSONResponse:
    """Check if engine is paused."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    paused = getattr(engine, "_paused", False)
    return JSONResponse({"paused": paused})


# =============================================================================
# Sleep/Wake Endpoints (GPU memory management)
# =============================================================================


@app.post("/bridge/sleep")
async def bridge_sleep() -> JSONResponse:
    """Put engine to sleep to free GPU memory."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        await engine.sleep()
        logger.info("Engine sleeping")
        return JSONResponse({"status": "sleeping"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/bridge/wake_up")
async def bridge_wake_up() -> JSONResponse:
    """Wake engine and reload model."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        await engine.wake_up()
        logger.info("Engine woken up")
        return JSONResponse({"status": "awake"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/bridge/is_sleeping")
async def bridge_is_sleeping() -> JSONResponse:
    """Check if engine is sleeping."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    sleeping = await engine.is_sleeping()
    return JSONResponse({"sleeping": sleeping})


# =============================================================================
# Debug Endpoints
# =============================================================================


@app.get("/bridge/debug")
async def bridge_debug() -> JSONResponse:
    """Debug endpoint to inspect engine state."""
    debug_info = {
        "engine_type": type(engine).__name__ if engine else None,
        "vllm_version": VLLM_VERSION,
        "patches_applied": PATCHES_APPLIED,
        "shared_weights_env": os.environ.get("VLLM_ENABLE_SHARED_WEIGHTS", "0"),
        "num_inference_nodes": os.environ.get("NUM_INFERENCE_NODES", "unset"),
        "logdir": os.environ.get("LOGDIR", "unset"),
    }

    if engine is not None:
        try:
            debug_info["model_config"] = {
                "model": str(engine.model_config.model),
                "dtype": str(engine.model_config.dtype),
            }
        except Exception:
            pass

    return JSONResponse(debug_info)


@app.get("/bridge/list_endpoints")
async def list_endpoints() -> JSONResponse:
    """List all available endpoints."""
    endpoints = []
    for route in app.routes:
        if hasattr(route, "path") and hasattr(route, "methods"):
            endpoints.append(
                {
                    "path": route.path,
                    "methods": list(route.methods),
                }
            )
    return JSONResponse({"endpoints": endpoints})


# =============================================================================
# LoRA Endpoints
# =============================================================================


@app.get("/lora/status")
async def lora_status() -> LoraStatusResponse:
    """Get LoRA adapter status."""
    log_dir = os.environ.get("LOGDIR", ".")
    available = []

    if os.path.exists(log_dir):
        for item in os.listdir(log_dir):
            item_path = os.path.join(log_dir, item)
            if os.path.isdir(item_path) and os.path.exists(
                os.path.join(item_path, "adapter_config.json")
            ):
                available.append(item)

    return LoraStatusResponse(
        active_adapter=bridge_state.active_lora_path,
        load_count=bridge_state.lora_load_count,
        available_adapters=available,
    )


@app.post("/lora/load")
async def lora_load(request: LoraLoadRequest) -> JSONResponse:
    """Load a LoRA adapter."""
    if not os.path.exists(request.adapter_path):
        raise HTTPException(
            status_code=404, detail=f"Adapter not found: {request.adapter_path}"
        )

    with bridge_state.lock:
        bridge_state.active_lora_path = request.adapter_path
        bridge_state.lora_load_count += 1

    logger.info(f"LoRA adapter loaded: {request.adapter_path}")

    return JSONResponse(
        {
            "status": "ok",
            "adapter_path": request.adapter_path,
            "load_count": bridge_state.lora_load_count,
        }
    )


@app.post("/lora/unload")
async def lora_unload() -> JSONResponse:
    """Unload current LoRA adapter."""
    with bridge_state.lock:
        prev = bridge_state.active_lora_path
        bridge_state.active_lora_path = None

    logger.info(f"LoRA adapter unloaded: {prev}")
    return JSONResponse({"status": "ok", "previous_adapter": prev})


# =============================================================================
# Server Setup
# =============================================================================


def build_app(args: Namespace) -> FastAPI:
    """Build the FastAPI application."""
    global app
    app.root_path = args.root_path
    return app


async def init_app(args: Namespace, llm_engine: AsyncLLM | None = None) -> FastAPI:
    """Initialize the application and vLLM engine."""
    app = build_app(args)

    global engine
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = (
        llm_engine
        if llm_engine is not None
        else AsyncLLM.from_engine_args(
            engine_args, usage_context=UsageContext.API_SERVER
        )
    )
    app.state.engine_client = engine

    # Export basic state dict info for trainers (the patched runner exports detailed info)
    _export_state_dict_info(args)

    return app


def _export_state_dict_info(args: Namespace) -> None:
    """Export basic model info to JSON for trainer (backup if patches don't run)."""
    log_dir = os.environ.get("LOGDIR", ".")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    json_path = Path(log_dir) / "vllm_bridge_config.json"

    # Only write basic info if the file doesn't exist or is empty
    # The patched runner will write complete info with param_mappings
    try:
        if json_path.exists():
            with open(json_path, "r") as f:
                existing = json.load(f)
                if (
                    existing.get("param_mappings")
                    and len(existing["param_mappings"]) > 0
                ):
                    logger.info(f"Config already has param_mappings, not overwriting")
                    return

        info = {
            "model": getattr(args, "model", "unknown"),
            "dtype": getattr(args, "dtype", "auto"),
            "tp_degree": getattr(args, "tensor_parallel_size", 1),
            "dp_shard_degree": 1,
            "param_mappings": {},
            "shared_weights_enabled": PATCHES_APPLIED,
        }

        with open(json_path, "w") as f:
            json.dump(info, f, indent=2)

        logger.info(f"Exported basic state dict info to {json_path}")
    except Exception as e:
        logger.warning(f"Failed to export state dict info: {e}")


async def run_server(
    args: Namespace, llm_engine: AsyncLLM | None = None, **uvicorn_kwargs: Any
) -> None:
    """Run the vLLM API server."""
    logger.info("vLLM API server version %s", VLLM_VERSION)
    logger.info("args: %s", args)

    if PATCHES_APPLIED:
        logger.info("=" * 60)
        logger.info("SHARED MEMORY MODE ENABLED")
        logger.info("Weight updates from trainer will be reflected immediately!")
        logger.info("=" * 60)

    set_ulimit()
    app = await init_app(args, llm_engine)

    if engine is None:
        raise RuntimeError("No engine initialized")

    # Log available endpoints
    logger.info("=" * 60)
    logger.info("Available endpoints:")
    logger.info("  POST /generate       - Generate completions")
    logger.info("  GET  /bridge/info    - Bridge status")
    logger.info("  POST /bridge/pause   - Pause generation")
    logger.info("  POST /bridge/resume  - Resume generation")
    logger.info("  GET  /lora/status    - LoRA adapter status")
    logger.info("=" * 60)

    shutdown_task = await serve_http(
        app,
        sock=None,
        enable_ssl_refresh=getattr(args, "enable_ssl_refresh", False),
        host=args.host,
        port=args.port,
        log_level=getattr(args, "log_level", "info"),
        timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
        ssl_keyfile=getattr(args, "ssl_keyfile", None),
        ssl_certfile=getattr(args, "ssl_certfile", None),
        ssl_ca_certs=getattr(args, "ssl_ca_certs", None),
        ssl_cert_reqs=getattr(args, "ssl_cert_reqs", ssl.CERT_NONE),
        **uvicorn_kwargs,
    )

    await shutdown_task


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument("--ssl-ca-certs", type=str, default=None)
    parser.add_argument("--enable-ssl-refresh", action="store_true", default=False)
    parser.add_argument("--ssl-cert-reqs", type=int, default=int(ssl.CERT_NONE))
    parser.add_argument("--root-path", type=str, default=None)
    parser.add_argument("--log-level", type=str, default="info")

    # Add vLLM engine args
    parser = AsyncEngineArgs.add_cli_args(parser)

    args = parser.parse_args()
    asyncio.run(run_server(args))
