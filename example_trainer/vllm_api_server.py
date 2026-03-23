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
import multiprocessing
import os
import ssl
import threading
from argparse import Namespace
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

# Default to v0 engine to avoid CUDA fork issues with v1 engine
# Users can override with VLLM_USE_V1=1 if needed
os.environ.setdefault("VLLM_USE_V1", "0")

# Set spawn method for multiprocessing (required for CUDA)
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # Already set

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
        from .vllm_patching import apply_patches
    except ImportError:
        # Fall back to absolute import (when run as script)
        try:
            import sys
            from pathlib import Path

            # Add parent directory to path so we can import vllm_patching
            script_dir = Path(__file__).parent
            if str(script_dir) not in sys.path:
                sys.path.insert(0, str(script_dir))
            from vllm_patching import apply_patches
        except ImportError as e:
            print(f"[vLLM Server] Could not import vllm_patching: {e}")
            print("[vLLM Server] Shared memory weight updates will not be available")
            return False

    try:
        success = apply_patches()
        if success:
            print("[vLLM Server] ✓ vLLM patches applied successfully!")
        else:
            print("[vLLM Server] ✗ Failed to apply patches")
        return success
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

# Handle vLLM version differences - FlexibleArgumentParser was removed/renamed
try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    # Create a compatible ArgumentParser that handles 'deprecated' kwarg
    # (Python 3.10 doesn't support 'deprecated' in BooleanOptionalAction)
    import argparse

    class FlexibleArgumentParser(argparse.ArgumentParser):
        """ArgumentParser that strips unsupported kwargs for Python < 3.13."""

        def add_argument(self, *args, **kwargs):
            # Remove 'deprecated' kwarg if present (not supported before Python 3.13)
            kwargs.pop("deprecated", None)
            return super().add_argument(*args, **kwargs)


# set_ulimit might not exist in all vLLM versions
try:
    from vllm.utils import set_ulimit
except ImportError:

    def set_ulimit() -> None:
        """No-op fallback for set_ulimit."""
        pass


from vllm.outputs import RequestOutput  # noqa: F401, E402
from vllm.version import __version__ as VLLM_VERSION  # noqa: E402

# Try to import LoRARequest for adapter support
try:
    from vllm.lora.request import LoRARequest  # noqa: E402

    LORA_AVAILABLE = True
except ImportError:
    LORA_AVAILABLE = False
    LoRARequest = None  # type: ignore

logger = init_logger("vllm.entrypoints.api_server")

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
    active_lora_name: Optional[str] = None
    active_lora_id: int = 0  # vLLM requires unique integer ID per adapter
    lora_load_count: int = 0


bridge_state = BridgeState()


def _get_lora_request() -> Optional["LoRARequest"]:
    """Get the current LoRA request if an adapter is active."""
    if not LORA_AVAILABLE:
        return None
    if bridge_state.active_lora_path is None:
        return None

    return LoRARequest(
        lora_name=bridge_state.active_lora_name or "default_adapter",
        lora_int_id=bridge_state.active_lora_id,
        lora_path=bridge_state.active_lora_path,
    )


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
    lora_available: bool
    active_adapter_path: Optional[str]
    active_adapter_name: Optional[str]
    active_adapter_id: Optional[int]
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

    # Get active LoRA adapter if any
    lora_request = _get_lora_request()

    results_generator = engine.generate(
        prompt, sampling_params, request_id, lora_request=lora_request
    )

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
        lora_available=LORA_AVAILABLE,
        active_adapter_path=bridge_state.active_lora_path,
        active_adapter_name=bridge_state.active_lora_name,
        active_adapter_id=(
            bridge_state.active_lora_id if bridge_state.active_lora_path else None
        ),
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

    # Read adapter config to validate and log details
    adapter_config_path = os.path.join(request.adapter_path, "adapter_config.json")
    adapter_info = {}

    if os.path.exists(adapter_config_path):
        try:
            with open(adapter_config_path, "r") as f:
                adapter_config = json.load(f)
            adapter_info = {
                "r": adapter_config.get("r"),
                "lora_alpha": adapter_config.get("lora_alpha"),
                "target_modules": adapter_config.get("target_modules"),
                "base_model": adapter_config.get("base_model_name_or_path"),
            }
            logger.info(f"LoRA adapter config: {adapter_info}")
        except Exception as e:
            logger.warning(f"Could not read adapter_config.json: {e}")
    else:
        logger.warning(f"No adapter_config.json found at {adapter_config_path}")

    with bridge_state.lock:
        bridge_state.active_lora_path = request.adapter_path
        bridge_state.active_lora_name = (
            request.adapter_name or f"adapter_{bridge_state.lora_load_count}"
        )
        bridge_state.active_lora_id = (
            bridge_state.lora_load_count + 1
        )  # vLLM needs unique int ID
        bridge_state.lora_load_count += 1

    logger.info(
        f"LoRA adapter loaded: {request.adapter_path} (id={bridge_state.active_lora_id})"
    )

    return JSONResponse(
        {
            "status": "ok",
            "adapter_path": request.adapter_path,
            "adapter_name": bridge_state.active_lora_name,
            "adapter_id": bridge_state.active_lora_id,
            "load_count": bridge_state.lora_load_count,
            "adapter_config": adapter_info,
        }
    )


@app.post("/lora/unload")
async def lora_unload() -> JSONResponse:
    """Unload current LoRA adapter."""
    with bridge_state.lock:
        prev_path = bridge_state.active_lora_path
        prev_name = bridge_state.active_lora_name
        bridge_state.active_lora_path = None
        bridge_state.active_lora_name = None
        bridge_state.active_lora_id = 0

    logger.info(f"LoRA adapter unloaded: {prev_path} ({prev_name})")
    return JSONResponse(
        {
            "status": "ok",
            "previous_adapter": prev_path,
            "previous_name": prev_name,
        }
    )


# =============================================================================
# Server Setup
# =============================================================================


def build_app(args: Namespace) -> FastAPI:
    """Build the FastAPI application."""
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
    # Allow explicit config path via env var, otherwise use LOGDIR
    config_path = os.environ.get("VLLM_BRIDGE_CONFIG_PATH")
    if config_path:
        json_path = Path(config_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
    else:
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
                    logger.info("Config already has param_mappings, not overwriting")
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
    logger.info("Streamlined vLLM Server - Training-Focused API")
    logger.info("Available endpoints:")
    logger.info("  POST /generate         - Generate with logprobs (primary endpoint)")
    logger.info("  GET  /health           - Health check")
    logger.info("  GET  /bridge/info      - Bridge status")
    logger.info("  POST /bridge/pause     - Pause generation")
    logger.info("  POST /bridge/resume    - Resume generation")
    logger.info("  GET  /lora/status      - LoRA adapter status")
    logger.info("  POST /lora/load        - Load LoRA adapter")
    logger.info("  POST /lora/unload      - Unload LoRA adapter")
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
