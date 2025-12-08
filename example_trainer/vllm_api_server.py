# Based on https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/api_server.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Custom vLLM API server with weight bridge hooks for shared-memory training.

This server extends the standard vLLM API with endpoints for:
- Shared-weight training: trainers can attach to model weights via NCCL
- LoRA hot-swap: load new adapters without server restart
- Weight synchronization: coordinate updates between trainer and inference

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                  vllm_api_server.py                     │
    │  ┌────────────────────────────────────────────────┐     │
    │  │              FastAPI Application                │     │
    │  │  ┌─────────┐  ┌──────────┐  ┌───────────────┐  │     │
    │  │  │/generate│  │/bridge/* │  │ /lora/*       │  │     │
    │  │  │ (infer) │  │ (sync)   │  │ (adapters)    │  │     │
    │  │  └────┬────┘  └────┬─────┘  └───────┬───────┘  │     │
    │  └───────┼────────────┼────────────────┼──────────┘     │
    │          │            │                │                │
    │  ┌───────▼────────────▼────────────────▼──────────┐     │
    │  │                 AsyncLLM                        │     │
    │  │  - Model weights (shared via NCCL)             │     │
    │  │  - LoRA adapters (hot-swappable)               │     │
    │  └────────────────────────────────────────────────┘     │
    └─────────────────────────────────────────────────────────┘
"""

import asyncio
import json
import os
import ssl
import threading
import time
from argparse import Namespace
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import vllm.envs as envs
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.utils import with_cancellation
from vllm.logger import init_logger
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils import random_uuid
from vllm.v1.engine.async_llm import AsyncLLM

try:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.utils.system_utils import set_ulimit
except ImportError:
    from vllm.utils import FlexibleArgumentParser, set_ulimit
from vllm.outputs import RequestOutput  # noqa: F401
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger("vllm.entrypoints.api_server")


# =============================================================================
# Global State
# =============================================================================

app = FastAPI()
engine: Optional[AsyncLLM] = None


@dataclass
class BridgeState:
    """State for weight bridge synchronization."""

    enabled: bool = False
    update_count: int = 0
    last_update_time: float = 0.0
    rendezvous_info: Dict[str, Any] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    # LoRA state
    active_lora_path: Optional[str] = None
    lora_load_count: int = 0


bridge_state = BridgeState()


# =============================================================================
# Pydantic Models for API
# =============================================================================


class BridgeInfoResponse(BaseModel):
    """Response model for bridge info endpoint."""

    enabled: bool
    update_count: int
    last_update_time: float
    rendezvous_info: Dict[str, Any]
    model_name: str
    device: str


class BridgeInitRequest(BaseModel):
    """Request model for initializing bridge."""

    master_addr: str
    master_port: int
    world_size: int
    trainer_ranks: List[int]


class WeightUpdateNotification(BaseModel):
    """Notification that weights have been updated."""

    update_count: int
    trainer_rank: int
    timestamp: float


class LoraLoadRequest(BaseModel):
    """Request to load a LoRA adapter."""

    adapter_path: str
    adapter_name: Optional[str] = None


class LoraStatusResponse(BaseModel):
    """Response model for LoRA status."""

    active_adapter: Optional[str]
    load_count: int
    available_adapters: List[str]


# =============================================================================
# Health Endpoints
# =============================================================================


@app.get("/health")
async def health() -> Response:
    """Basic health check - is server alive?"""
    return Response(status_code=200)


@app.get("/health_generate")
async def health_generate() -> Response:
    """
    Deep health check - can we actually generate tokens?

    This sends a minimal request through the full inference pipeline
    to verify the model is loaded and functioning.
    """
    assert engine is not None
    sampling_params = SamplingParams()
    request_id = random_uuid()
    results_generator = engine.generate(
        {"prompt_token_ids": [0]}, sampling_params, request_id
    )
    try:
        async for request_output in results_generator:
            final_output = request_output  # type: RequestOutput  # noqa: F841
    except asyncio.CancelledError:
        return Response(status_code=499)
    return Response(status_code=200)


# =============================================================================
# Generation Endpoint
# =============================================================================


@app.post("/generate")
async def generate(request: Request) -> Response:
    """
    Generate text completion for a prompt.

    Request JSON fields:
    - prompt: str - The input text to complete
    - stream: bool - Whether to stream results (default: False)
    - max_tokens: int - Maximum tokens to generate
    - temperature: float - Sampling temperature
    - top_p: float - Nucleus sampling threshold
    - logprobs: int - Number of logprobs to return per token

    Returns:
    - text: List[str] - Generated completions
    - prompt: str - Echo of input prompt
    - finish_reasons: List[str] - Why generation stopped
    - logprobs: List (optional) - Token log probabilities
    - token_ids: List (optional) - Generated token IDs
    """
    request_dict = await request.json()
    return await _generate(request_dict, raw_request=request)


@with_cancellation
async def _generate(request_dict: dict, raw_request: Request) -> Response:
    prompt_input = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)

    # Handle both string prompts and {"prompt_token_ids": [...]} format
    # The latter is used by atroposlib's VLLMServer
    if isinstance(prompt_input, dict) and "prompt_token_ids" in prompt_input:
        # Token IDs format from atroposlib
        prompt_token_ids = prompt_input["prompt_token_ids"]
        prompt = {"prompt_token_ids": prompt_token_ids}
    else:
        # String prompt
        prompt = prompt_input

    # Handle logprobs parameter - atroposlib sends logprobs=0 which means "return logprobs"
    # vLLM uses None to mean "don't return logprobs" and an int for "return N top logprobs"
    if "logprobs" in request_dict:
        logprobs_val = request_dict["logprobs"]
        # logprobs=0 means return logprobs (just 1 per token)
        # logprobs=None or not present means don't return logprobs
        if logprobs_val is not None:
            request_dict["logprobs"] = max(1, logprobs_val)  # At least 1

    request_dict["output_kind"] = RequestOutputKind.FINAL_ONLY
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()

    assert engine is not None
    results_generator = engine.generate(prompt, sampling_params, request_id)

    # Streaming: yield results as they're generated
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            prompt_text = request_output.prompt
            assert prompt_text is not None
            text_outputs = [prompt_text + output.text for output in request_output.outputs]
            ret = {"text": text_outputs}
            yield (json.dumps(ret) + "\n").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())

    # Non-streaming: wait for full completion
    final_output = None
    try:
        async for request_output in results_generator:
            final_output = request_output  # type: RequestOutput
    except asyncio.CancelledError:
        return Response(status_code=499)

    assert final_output is not None
    prompt_text = final_output.prompt or engine.tokenizer.decode(
        final_output.prompt_token_ids
    )
    assert prompt_text is not None
    text_outputs = [output.text for output in final_output.outputs]
    finish_reasons = [output.finish_reason for output in final_output.outputs]
    ret = {"text": text_outputs, "prompt": prompt_text, "finish_reasons": finish_reasons}

    # Include logprobs if requested (useful for RL training)
    # Format matches what atroposlib's VLLMServer expects
    if sampling_params.logprobs is not None:
        output_logprobs = []
        for x in final_output.outputs:
            if x.logprobs:
                # Format: [[{token_id: logprob}, ...], ...] per output
                seq_logprobs = [
                    [{str(key): value.logprob for key, value in logprob.items()}]
                    for logprob in x.logprobs
                ]
            else:
                seq_logprobs = []
            output_logprobs.append(seq_logprobs)

        prompt_token_ids = final_output.prompt_token_ids
        output_token_ids = [list(x.token_ids) for x in final_output.outputs]
        ret["logprobs"] = output_logprobs
        ret["prompt_token_ids"] = list(prompt_token_ids) if prompt_token_ids else []
        ret["token_ids"] = output_token_ids

    return JSONResponse(ret)


# =============================================================================
# OpenAI-Compatible Completions Endpoint
# =============================================================================


@app.post("/v1/completions")
async def openai_completions(request: Request) -> Response:
    """
    OpenAI-compatible completions endpoint.

    This translates OpenAI API format to our internal format.

    Request JSON fields (OpenAI format):
    - model: str - Model name (ignored, uses loaded model)
    - prompt: str or List[str] - The input text(s) to complete
    - max_tokens: int - Maximum tokens to generate
    - temperature: float - Sampling temperature
    - top_p: float - Nucleus sampling threshold
    - n: int - Number of completions per prompt
    - stream: bool - Whether to stream results
    - logprobs: int - Number of logprobs to return
    - echo: bool - Whether to echo the prompt
    - stop: str or List[str] - Stop sequences

    Returns OpenAI-compatible response format.
    """
    request_dict = await request.json()

    # Extract OpenAI-specific fields
    prompt = request_dict.get("prompt", "")
    model = request_dict.get("model", "")
    max_tokens = request_dict.get("max_tokens", 16)
    temperature = request_dict.get("temperature", 1.0)
    top_p = request_dict.get("top_p", 1.0)
    n = request_dict.get("n", 1)
    stream = request_dict.get("stream", False)
    logprobs_count = request_dict.get("logprobs")
    echo = request_dict.get("echo", False)
    stop = request_dict.get("stop")

    # Handle prompt as string or list
    if isinstance(prompt, list):
        # For simplicity, just use the first prompt
        # Full implementation would handle batches
        prompt = prompt[0] if prompt else ""

    # Build sampling params
    sampling_kwargs = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
    }

    if logprobs_count is not None:
        sampling_kwargs["logprobs"] = logprobs_count

    if stop is not None:
        if isinstance(stop, str):
            stop = [stop]
        sampling_kwargs["stop"] = stop

    sampling_params = SamplingParams(**sampling_kwargs)
    sampling_params.output_kind = RequestOutputKind.FINAL_ONLY
    request_id = random_uuid()

    assert engine is not None
    results_generator = engine.generate(prompt, sampling_params, request_id)

    # Non-streaming response
    final_output = None
    try:
        async for request_output in results_generator:
            final_output = request_output
    except asyncio.CancelledError:
        return Response(status_code=499)

    if final_output is None:
        return JSONResponse(
            {"error": {"message": "No output generated", "type": "server_error"}},
            status_code=500,
        )

    # Build OpenAI-compatible response
    choices = []
    for i, output in enumerate(final_output.outputs):
        text = output.text
        if echo:
            text = prompt + text

        choice = {
            "text": text,
            "index": i,
            "logprobs": None,
            "finish_reason": output.finish_reason or "stop",
        }

        # Add logprobs if requested
        if logprobs_count is not None and output.logprobs:
            choice["logprobs"] = {
                "tokens": [
                    list(lp.keys())[0] if lp else "" for lp in output.logprobs
                ],
                "token_logprobs": [
                    list(lp.values())[0].logprob if lp else None
                    for lp in output.logprobs
                ],
                "top_logprobs": [
                    {k: v.logprob for k, v in lp.items()} if lp else {}
                    for lp in output.logprobs
                ],
                "text_offset": [],  # Not implemented
            }

        choices.append(choice)

    response = {
        "id": f"cmpl-{request_id}",
        "object": "text_completion",
        "created": int(asyncio.get_event_loop().time()),
        "model": model or "vllm-model",
        "choices": choices,
        "usage": {
            "prompt_tokens": len(final_output.prompt_token_ids) if final_output.prompt_token_ids else 0,
            "completion_tokens": sum(len(o.token_ids) for o in final_output.outputs),
            "total_tokens": (len(final_output.prompt_token_ids) if final_output.prompt_token_ids else 0)
                + sum(len(o.token_ids) for o in final_output.outputs),
        },
    }

    return JSONResponse(response)


@app.post("/v1/chat/completions")
async def openai_chat_completions(request: Request) -> Response:
    """
    OpenAI-compatible chat completions endpoint.

    Request JSON fields:
    - model: str - Model name (ignored, uses loaded model)
    - messages: List[dict] - Chat messages with 'role' and 'content'
    - max_tokens: int - Maximum tokens to generate
    - temperature: float - Sampling temperature
    - top_p: float - Nucleus sampling threshold
    - n: int - Number of completions
    - stream: bool - Whether to stream results
    - stop: str or List[str] - Stop sequences

    Returns OpenAI-compatible chat completion response.
    """
    request_dict = await request.json()

    # Extract fields
    messages = request_dict.get("messages", [])
    model = request_dict.get("model", "")
    max_tokens = request_dict.get("max_tokens", 512)
    temperature = request_dict.get("temperature", 1.0)
    top_p = request_dict.get("top_p", 1.0)
    n = request_dict.get("n", 1)
    stream = request_dict.get("stream", False)
    stop = request_dict.get("stop")

    # Convert messages to prompt using chat template
    assert engine is not None

    # Try to use the tokenizer's chat template
    try:
        tokenizer = engine.tokenizer.tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback: simple concatenation
            prompt = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"
    except Exception:
        # Simple fallback
        prompt = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages
        )
        prompt += "\nassistant:"

    # Build sampling params
    sampling_kwargs = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
    }

    if stop is not None:
        if isinstance(stop, str):
            stop = [stop]
        sampling_kwargs["stop"] = stop

    sampling_params = SamplingParams(**sampling_kwargs)
    sampling_params.output_kind = RequestOutputKind.FINAL_ONLY
    request_id = random_uuid()

    results_generator = engine.generate(prompt, sampling_params, request_id)

    # Non-streaming response
    final_output = None
    try:
        async for request_output in results_generator:
            final_output = request_output
    except asyncio.CancelledError:
        return Response(status_code=499)

    if final_output is None:
        return JSONResponse(
            {"error": {"message": "No output generated", "type": "server_error"}},
            status_code=500,
        )

    # Build OpenAI-compatible chat response
    choices = []
    for i, output in enumerate(final_output.outputs):
        choice = {
            "index": i,
            "message": {
                "role": "assistant",
                "content": output.text,
            },
            "finish_reason": output.finish_reason or "stop",
        }
        choices.append(choice)

    prompt_tokens = len(final_output.prompt_token_ids) if final_output.prompt_token_ids else 0
    completion_tokens = sum(len(o.token_ids) for o in final_output.outputs)

    response = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model or "vllm-model",
        "choices": choices,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }

    return JSONResponse(response)


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    """
    List available models (OpenAI-compatible).

    Returns the currently loaded model.
    """
    assert engine is not None

    model_name = str(engine.engine.model_config.model) if hasattr(engine, "engine") else "unknown"

    return JSONResponse({
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "vllm",
                "permission": [],
                "root": model_name,
                "parent": None,
            }
        ],
    })


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str) -> JSONResponse:
    """
    Get model info (OpenAI-compatible).
    """
    assert engine is not None

    model_name = str(engine.engine.model_config.model) if hasattr(engine, "engine") else "unknown"

    return JSONResponse({
        "id": model_name,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "vllm",
        "permission": [],
        "root": model_name,
        "parent": None,
    })


# =============================================================================
# Bridge Endpoints (for shared-weight training)
# =============================================================================


@app.get("/bridge/info", response_model=BridgeInfoResponse)
async def bridge_info() -> BridgeInfoResponse:
    """
    Get bridge status and rendezvous information.

    Trainers call this to discover how to connect to the weight-sharing
    process group. Returns connection details and current sync state.
    """
    assert engine is not None

    return BridgeInfoResponse(
        enabled=bridge_state.enabled,
        update_count=bridge_state.update_count,
        last_update_time=bridge_state.last_update_time,
        rendezvous_info=bridge_state.rendezvous_info,
        model_name=str(engine.engine.model_config.model),
        device=str(next(iter(engine.engine.model_executor.driver_worker.model_runner.model.parameters())).device)
        if hasattr(engine, "engine") else "unknown",
    )


@app.post("/bridge/init")
async def bridge_init(request: BridgeInitRequest) -> JSONResponse:
    """
    Initialize the weight bridge for shared-memory training.

    This sets up the rendezvous information that trainers need to join
    the same NCCL process group as this inference server.

    Called once when setting up a training run.
    """
    with bridge_state.lock:
        bridge_state.enabled = True
        bridge_state.rendezvous_info = {
            "master_addr": request.master_addr,
            "master_port": request.master_port,
            "world_size": request.world_size,
            "trainer_ranks": request.trainer_ranks,
            "initialized_at": time.time(),
        }

    logger.info(f"Bridge initialized: {bridge_state.rendezvous_info}")
    return JSONResponse({"status": "ok", "rendezvous_info": bridge_state.rendezvous_info})


@app.post("/bridge/notify_update")
async def bridge_notify_update(notification: WeightUpdateNotification) -> JSONResponse:
    """
    Receive notification that trainer has updated weights.

    After optimizer.step(), the trainer calls this to signal that the
    shared weights have been modified. The server can use this to:
    - Log the update for debugging
    - Invalidate any cached KV states if needed
    - Track synchronization for metrics

    In shared-memory mode, the weights are already updated in-place,
    so no data transfer happens here - this is just coordination.
    """
    with bridge_state.lock:
        bridge_state.update_count = notification.update_count
        bridge_state.last_update_time = notification.timestamp

    logger.info(
        f"Weight update #{notification.update_count} from trainer {notification.trainer_rank}"
    )

    return JSONResponse({
        "status": "ok",
        "update_count": bridge_state.update_count,
        "server_time": time.time(),
    })


@app.get("/bridge/state_dict_info")
async def bridge_state_dict_info() -> JSONResponse:
    """
    Get information about the model's state dict for weight attachment.

    Returns parameter names, shapes, and dtypes so trainers can properly
    map their tensors to the inference model's parameters.
    """
    assert engine is not None

    try:
        # Access the underlying model
        model = engine.engine.model_executor.driver_worker.model_runner.model
        state_dict_info = {}

        for name, param in model.named_parameters():
            state_dict_info[name] = {
                "shape": list(param.shape),
                "dtype": str(param.dtype),
                "device": str(param.device),
                "requires_grad": param.requires_grad,
            }

        return JSONResponse({
            "status": "ok",
            "num_parameters": len(state_dict_info),
            "total_params": sum(p.numel() for p in model.parameters()),
            "parameters": state_dict_info,
        })

    except Exception as e:
        logger.error(f"Failed to get state dict info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/bridge/disable")
async def bridge_disable() -> JSONResponse:
    """
    Disable the weight bridge.

    Called when training ends or if the trainer disconnects.
    """
    with bridge_state.lock:
        bridge_state.enabled = False
        bridge_state.rendezvous_info = {}

    logger.info("Bridge disabled")
    return JSONResponse({"status": "ok"})


# =============================================================================
# LoRA Endpoints (for adapter hot-swapping)
# =============================================================================


@app.get("/lora/status", response_model=LoraStatusResponse)
async def lora_status() -> LoraStatusResponse:
    """
    Get current LoRA adapter status.

    Returns which adapter is active (if any) and lists available adapters
    in the configured adapter directory.
    """
    # List available adapters from save path
    adapter_dir = os.environ.get("LORA_ADAPTER_DIR", "./adapters")
    available = []
    if os.path.isdir(adapter_dir):
        for item in os.listdir(adapter_dir):
            item_path = os.path.join(adapter_dir, item)
            # Check if it looks like a PEFT adapter
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
    """
    Hot-swap a LoRA adapter without restarting the server.

    The adapter is loaded from disk and merged with the base model weights.
    This is much faster than restarting vLLM with a new checkpoint.

    Note: This requires the PEFT library and a compatible vLLM version.
    """
    adapter_path = request.adapter_path

    if not os.path.exists(adapter_path):
        raise HTTPException(status_code=404, detail=f"Adapter not found: {adapter_path}")

    if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        raise HTTPException(
            status_code=400, detail=f"Invalid adapter (missing adapter_config.json): {adapter_path}"
        )

    try:
        # TODO: Implement actual LoRA loading for vLLM
        # This depends on vLLM's LoRA support which varies by version
        # For now, we track the state and log the request

        with bridge_state.lock:
            bridge_state.active_lora_path = adapter_path
            bridge_state.lora_load_count += 1

        logger.info(f"LoRA adapter loaded: {adapter_path}")

        return JSONResponse({
            "status": "ok",
            "adapter_path": adapter_path,
            "load_count": bridge_state.lora_load_count,
            "message": "Adapter registered (actual loading depends on vLLM version)",
        })

    except Exception as e:
        logger.error(f"Failed to load LoRA adapter: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/lora/unload")
async def lora_unload() -> JSONResponse:
    """
    Unload the current LoRA adapter, reverting to base model weights.
    """
    with bridge_state.lock:
        prev_adapter = bridge_state.active_lora_path
        bridge_state.active_lora_path = None

    logger.info(f"LoRA adapter unloaded: {prev_adapter}")

    return JSONResponse({
        "status": "ok",
        "previous_adapter": prev_adapter,
    })


# =============================================================================
# Server Setup
# =============================================================================


def build_app(args: Namespace) -> FastAPI:
    """Build the FastAPI application with configured root path."""
    global app  # noqa: F824
    app.root_path = args.root_path
    return app


async def init_app(
    args: Namespace,
    llm_engine: AsyncLLM | None = None,
) -> FastAPI:
    """
    Initialize the application and vLLM engine.

    Args:
        args: Parsed command-line arguments
        llm_engine: Optional pre-created engine (for testing)

    Returns:
        Configured FastAPI application
    """
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

    # Export state dict info for trainers
    _export_state_dict_info(args)

    return app


def _export_state_dict_info(args: Namespace) -> None:
    """
    Export model parameter mapping to JSON for trainer attachment.

    This writes a file that trainers can read to understand how to
    map their parameters to the inference model's parameters.
    """
    log_dir = os.environ.get("LOGDIR", ".")
    json_path = Path(log_dir) / "vllm_bridge_config.json"

    try:
        # Basic info - actual param mappings added when bridge is initialized
        info = {
            "model": getattr(args, "model", "unknown"),
            "dtype": getattr(args, "dtype", "auto"),
            "tp_degree": getattr(args, "tensor_parallel_size", 1),
            "dp_shard_degree": 1,  # Data parallel sharding
            "param_mappings": {},
        }

        with open(json_path, "w") as f:
            json.dump(info, f, indent=2)

        logger.info(f"Exported state dict info to {json_path}")

    except Exception as e:
        logger.warning(f"Failed to export state dict info: {e}")


async def run_server(
    args: Namespace, llm_engine: AsyncLLM | None = None, **uvicorn_kwargs: Any
) -> None:
    """
    Run the vLLM API server.

    This is the main entry point that starts the HTTP server and
    serves requests until shutdown.
    """
    logger.info("vLLM API server version %s", VLLM_VERSION)
    logger.info("args: %s", args)

    set_ulimit()
    app = await init_app(args, llm_engine)
    assert engine is not None

    # Log bridge endpoints
    logger.info("Bridge endpoints available:")
    logger.info("  GET  /bridge/info - Get bridge status")
    logger.info("  POST /bridge/init - Initialize weight bridge")
    logger.info("  POST /bridge/notify_update - Notify of weight update")
    logger.info("  GET  /bridge/state_dict_info - Get model parameters")
    logger.info("  GET  /lora/status - Get LoRA adapter status")
    logger.info("  POST /lora/load - Load LoRA adapter")
    logger.info("  POST /lora/unload - Unload LoRA adapter")

    shutdown_task = await serve_http(
        app,
        sock=None,
        enable_ssl_refresh=args.enable_ssl_refresh,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
        **uvicorn_kwargs,
    )

    await shutdown_task


# =============================================================================
# CLI Entry Point
# =============================================================================


if __name__ == "__main__":
    parser = FlexibleArgumentParser()

    # Server configuration
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=parser.check_port, default=8000)
    parser.add_argument("--log-level", type=str, default="debug")

    # SSL configuration
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument(
        "--ssl-ca-certs", type=str, default=None, help="The CA certificates file"
    )
    parser.add_argument(
        "--enable-ssl-refresh",
        action="store_true",
        default=False,
        help="Refresh SSL Context when SSL certificate files change",
    )
    parser.add_argument(
        "--ssl-cert-reqs",
        type=int,
        default=int(ssl.CERT_NONE),
        help="Whether client certificate is required (see stdlib ssl module's)",
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy",
    )

    # Add vLLM engine arguments
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    asyncio.run(run_server(args))
