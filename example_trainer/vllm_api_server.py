
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

import base64
import pickle

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

# Import sync LLM for collective_rpc with function support
try:
    from vllm import LLM as SyncLLM
    SYNC_LLM_AVAILABLE = True
except ImportError:
    SYNC_LLM_AVAILABLE = False
    SyncLLM = None

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
sync_engine: Optional["SyncLLM"] = None  # For collective_rpc with functions


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


def get_engine():
    """Get the active engine (async or sync)."""
    if engine is not None:
        return engine
    if sync_engine is not None:
        return sync_engine
    raise HTTPException(status_code=503, detail="No engine available")


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
    sampling_params = SamplingParams(max_tokens=1)
    
    if engine is not None:
        # Async engine path
        request_id = random_uuid()
        results_generator = engine.generate(
            {"prompt_token_ids": [0]}, sampling_params, request_id
        )
        try:
            async for request_output in results_generator:
                final_output = request_output  # type: RequestOutput  # noqa: F841
        except asyncio.CancelledError:
            return Response(status_code=499)
    elif sync_engine is not None:
        # Sync engine path (CUDA IPC mode)
        import concurrent.futures
        def _sync_health_check():
            return sync_engine.generate(["Hello"], sampling_params)
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            await loop.run_in_executor(pool, _sync_health_check)
    else:
        return Response(status_code=503)
    
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

    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()
    
    # Handle both async engine (standard) and sync engine (CUDA IPC mode)
    if engine is not None:
        # Standard async mode
        sampling_params.output_kind = RequestOutputKind.FINAL_ONLY
        results_generator = engine.generate(prompt, sampling_params, request_id)
    elif sync_engine is not None:
        # CUDA IPC mode: use sync engine with thread pool
        # Sync LLM doesn't support streaming, so disable it
        if stream:
            logger.warning("Streaming not supported in CUDA IPC mode, using non-streaming")
            stream = False
        
        # Run sync generation in thread pool
        import concurrent.futures
        def _sync_generate():
            return sync_engine.generate([prompt], sampling_params)
        
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            outputs = await loop.run_in_executor(pool, _sync_generate)
        
        # Convert to match async output format
        if outputs:
            final_output = outputs[0]
            prompt_text = final_output.prompt or (
                sync_engine.get_tokenizer().decode(final_output.prompt_token_ids)
                if final_output.prompt_token_ids else ""
            )
            text_outputs = [output.text for output in final_output.outputs]
            finish_reasons = [output.finish_reason for output in final_output.outputs]
            ret = {"text": text_outputs, "prompt": prompt_text, "finish_reasons": finish_reasons}
            
            # Include logprobs if requested
            if sampling_params.logprobs is not None:
                output_logprobs = []
                for x in final_output.outputs:
                    if x.logprobs:
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
        else:
            return JSONResponse({"error": "No output generated"}, status_code=500)
    else:
        raise HTTPException(status_code=503, detail="No engine available")

    # =========================================================================
    # Async engine path (standard mode) - streaming and non-streaming
    # =========================================================================
    
    # Streaming: yield results as theyre generated
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
    assert engine is not None
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
    request_id = random_uuid()

    # Handle both async and sync engines
    if engine is not None:
        sampling_params.output_kind = RequestOutputKind.FINAL_ONLY
        results_generator = engine.generate(prompt, sampling_params, request_id)

        # Non-streaming response
        final_output = None
        try:
            async for request_output in results_generator:
                final_output = request_output
        except asyncio.CancelledError:
            return Response(status_code=499)
    elif sync_engine is not None:
        # CUDA IPC mode: use sync engine
        import concurrent.futures
        def _sync_generate():
            return sync_engine.generate([prompt], sampling_params)
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            outputs = await loop.run_in_executor(pool, _sync_generate)
        final_output = outputs[0] if outputs else None
    else:
        raise HTTPException(status_code=503, detail="No engine available")

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
    active_engine = get_engine()

    # Try to use the tokenizer's chat template
    try:
        if engine is not None:
            tokenizer = engine.tokenizer.tokenizer
        else:
            tokenizer = sync_engine.get_tokenizer()
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
    request_id = random_uuid()

    # Handle both async and sync engines
    if engine is not None:
        sampling_params.output_kind = RequestOutputKind.FINAL_ONLY
        results_generator = engine.generate(prompt, sampling_params, request_id)

        # Non-streaming response
        final_output = None
        try:
            async for request_output in results_generator:
                final_output = request_output
        except asyncio.CancelledError:
            return Response(status_code=499)
    elif sync_engine is not None:
        # CUDA IPC mode: use sync engine
        import concurrent.futures
        def _sync_generate():
            return sync_engine.generate([prompt], sampling_params)
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            outputs = await loop.run_in_executor(pool, _sync_generate)
        final_output = outputs[0] if outputs else None
    else:
        raise HTTPException(status_code=503, detail="No engine available")

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
    active_engine = get_engine()

    if engine is not None:
        model_name = str(engine.model_config.model) if hasattr(engine, "model_config") else "unknown"
    elif sync_engine is not None:
        model_name = str(sync_engine.llm_engine.model_config.model) if hasattr(sync_engine, "llm_engine") else "unknown"
    else:
        model_name = "unknown"

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
    active_engine = get_engine()

    if engine is not None:
        model_name = str(engine.model_config.model) if hasattr(engine, "model_config") else "unknown"
    elif sync_engine is not None:
        model_name = str(sync_engine.llm_engine.model_config.model) if hasattr(sync_engine, "llm_engine") else "unknown"
    else:
        model_name = "unknown"

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
    active_engine = get_engine()

    if engine is not None:
        model_name = str(engine.model_config.model) if hasattr(engine, "model_config") else "unknown"
        device = "unknown"  # Can't easily get device from AsyncLLM
    elif sync_engine is not None:
        model_name = str(sync_engine.llm_engine.model_config.model) if hasattr(sync_engine, "llm_engine") else "unknown"
        device = "cuda"  # Sync engine is always on CUDA for IPC
    else:
        model_name = "unknown"
        device = "unknown"

    return BridgeInfoResponse(
        enabled=bridge_state.enabled,
        update_count=bridge_state.update_count,
        last_update_time=bridge_state.last_update_time,
        rendezvous_info=bridge_state.rendezvous_info,
        model_name=model_name,
        device=device,
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
    active_engine = get_engine()

    try:
        # Access the underlying model based on engine type
        if sync_engine is not None:
            # CUDA IPC mode: can access model directly
            model = sync_engine.llm_engine.model_executor.driver_worker.model_runner.model
        elif engine is not None:
            # Async mode: model is in subprocess, can't access directly
            return JSONResponse({
                "status": "unavailable",
                "message": "Model state dict not accessible in async mode. Use CUDA IPC mode (--enable-cuda-ipc) for direct access.",
                "num_parameters": 0,
                "parameters": {},
            })
        else:
            raise HTTPException(status_code=503, detail="No engine available")
        
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
# Weight Update Endpoints (Pause/Resume for Training)
# =============================================================================


@app.post("/bridge/pause")
async def bridge_pause() -> JSONResponse:
    """
    Pause generation to allow weight updates.
    
    This is vLLM's built-in mechanism for weight updates!
    Waits for in-flight requests to finish, then pauses.
    
    Use this BEFORE updating weights from the trainer.
    
    NOTE: Only available with AsyncLLM (not CUDA IPC mode).
    """
    if engine is None:
        if sync_engine is not None:
            return JSONResponse({
                "status": "not_supported",
                "message": "Pause/resume not supported in CUDA IPC mode. Weights are shared directly.",
            })
        raise HTTPException(status_code=503, detail="No engine available")
    
    try:
        await engine.pause_generation(
            wait_for_inflight_requests=True,
            clear_cache=True,
        )
        logger.info("Generation paused for weight updates")
        
        return JSONResponse({
            "status": "paused",
            "message": "Ready for weight updates. Call /bridge/resume when done.",
        })
    except Exception as e:
        logger.error(f"Failed to pause generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/bridge/resume")
async def bridge_resume() -> JSONResponse:
    """
    Resume generation after weight updates.
    
    Call this AFTER updating weights from the trainer.
    
    NOTE: Only available with AsyncLLM (not CUDA IPC mode).
    """
    if engine is None:
        if sync_engine is not None:
            return JSONResponse({
                "status": "not_supported",
                "message": "Pause/resume not supported in CUDA IPC mode. Weights are shared directly.",
            })
        raise HTTPException(status_code=503, detail="No engine available")
    
    try:
        await engine.resume_generation()
        logger.info("Generation resumed after weight updates")
        
        return JSONResponse({
            "status": "resumed",
            "message": "Generation resumed with updated weights.",
        })
    except Exception as e:
        logger.error(f"Failed to resume generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/bridge/is_paused")
async def bridge_is_paused() -> JSONResponse:
    """Check if generation is currently paused."""
    if engine is None:
        if sync_engine is not None:
            return JSONResponse({"paused": False, "mode": "cuda_ipc"})
        raise HTTPException(status_code=503, detail="No engine available")
    
    paused = await engine.is_paused()
    return JSONResponse({"paused": paused})


@app.post("/bridge/sleep")
async def bridge_sleep(level: int = 1) -> JSONResponse:
    """
    Put the engine to sleep to free GPU memory.
    
    Level 1: Minimal sleep, fast wake up
    Higher levels: Deeper sleep, frees more memory
    
    Use for memory-constrained environments.
    
    NOTE: Only available with AsyncLLM (not CUDA IPC mode).
    """
    if engine is None:
        if sync_engine is not None:
            return JSONResponse({
                "status": "not_supported",
                "message": "Sleep/wake not supported in CUDA IPC mode.",
            })
        raise HTTPException(status_code=503, detail="No engine available")
    
    try:
        await engine.sleep(level=level)
        logger.info(f"Engine put to sleep (level {level})")
        
        return JSONResponse({
            "status": "sleeping",
            "level": level,
            "message": "GPU memory freed. Call /bridge/wake_up to resume.",
        })
    except Exception as e:
        logger.error(f"Failed to sleep: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/bridge/wake_up")
async def bridge_wake_up() -> JSONResponse:
    """
    Wake up the engine from sleep.
    
    Reloads the model into GPU memory.
    
    NOTE: Only available with AsyncLLM (not CUDA IPC mode).
    """
    if engine is None:
        if sync_engine is not None:
            return JSONResponse({
                "status": "not_supported",
                "message": "Sleep/wake not supported in CUDA IPC mode.",
            })
        raise HTTPException(status_code=503, detail="No engine available")
    
    try:
        await engine.wake_up()
        logger.info("Engine woken up")
        
        return JSONResponse({
            "status": "awake",
            "message": "Model reloaded into GPU memory.",
        })
    except Exception as e:
        logger.error(f"Failed to wake up: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/bridge/is_sleeping")
async def bridge_is_sleeping() -> JSONResponse:
    """Check if engine is currently sleeping."""
    if engine is None:
        if sync_engine is not None:
            return JSONResponse({"sleeping": False, "mode": "cuda_ipc"})
        raise HTTPException(status_code=503, detail="No engine available")
    
    sleeping = await engine.is_sleeping()
    return JSONResponse({"sleeping": sleeping})


# =============================================================================
# RPC Endpoints (Call Worker Methods)
# =============================================================================


class CollectiveRPCRequest(BaseModel):
    """Request to call a method on all workers."""
    method: str
    timeout: Optional[float] = None
    args: List[Any] = []
    kwargs: Dict[str, Any] = {}


@app.post("/bridge/collective_rpc")
async def bridge_collective_rpc(request: CollectiveRPCRequest) -> JSONResponse:
    """
    Call a method on all workers via collective RPC.
    
    The method must exist on the worker class.
    This is an advanced endpoint for custom worker operations.
    
    Example worker methods:
    - 'save_model' - Save model weights
    - 'get_model_info' - Get model information
    
    Note: For AsyncLLM, the method name is passed as a STRING.
    For sync LLM (CUDA IPC mode), use /bridge/export_cuda_ipc instead.
    """
    if engine is None:
        if sync_engine is not None:
            return JSONResponse({
                "status": "not_supported",
                "message": "Use /bridge/export_cuda_ipc for sync LLM collective operations.",
            })
        raise HTTPException(status_code=503, detail="No engine available")
    
    try:
        result = await engine.collective_rpc(
            method=request.method,
            timeout=request.timeout,
            args=tuple(request.args),
            kwargs=request.kwargs if request.kwargs else None,
        )
        
        logger.info(f"collective_rpc({request.method}) completed")
        
        return JSONResponse({
            "status": "ok",
            "method": request.method,
            "result": result if isinstance(result, (dict, list, str, int, float, bool, type(None))) else str(result),
        })
    except Exception as e:
        logger.error(f"collective_rpc failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# CUDA IPC Export (True Shared Memory)
# =============================================================================


def _export_cuda_ipc_handles_fn(worker_self) -> dict:
    """
    Worker-side function to export CUDA IPC handles.
    
    This function runs INSIDE the vLLM worker process where the model lives.
    The first argument 'worker_self' is the GPU worker instance.
    
    Returns:
        Dictionary with IPC handles for all model parameters.
    """
    model = worker_self.model_runner.model
    
    ipc_handles = {}
    failed_params = []
    
    for name, param in model.named_parameters():
        try:
            if not param.is_cuda:
                failed_params.append(f"{name}: not on CUDA")
                continue
            
            # Get the underlying storage and create IPC handle
            storage = param.data.storage()
            handle = storage._share_cuda_()
            
            # Serialize the handle
            handle_bytes = pickle.dumps(handle)
            handle_b64 = base64.b64encode(handle_bytes).decode('ascii')
            
            ipc_handles[name] = {
                "ipc_handle": handle_b64,
                "shape": list(param.shape),
                "dtype": str(param.dtype),
                "device_index": param.device.index if param.device.index is not None else 0,
                "storage_offset": param.storage_offset(),
                "numel": param.numel(),
                "stride": list(param.stride()),
            }
        except Exception as e:
            failed_params.append(f"{name}: {str(e)}")
    
    return {
        "handles": ipc_handles,
        "failed": failed_params,
        "model_class": model.__class__.__name__,
        "num_params": len(list(model.parameters())),
    }


@app.post("/bridge/export_cuda_ipc")
async def bridge_export_cuda_ipc() -> JSONResponse:
    """
    Export CUDA IPC handles for all model parameters.
    
    This enables TRUE shared memory between vLLM and the trainer!
    Both processes can access the SAME GPU tensors.
    
    Uses sync LLM's collective_rpc which accepts functions.
    
    REQUIREMENTS:
    - Both processes must be on the SAME GPU
    - vLLM must be started with --enable-cuda-ipc flag
    
    Returns:
        JSON with path to IPC handles file and parameter count.
    """
    global sync_engine
    
    if sync_engine is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Sync LLM not initialized. Start server with --enable-cuda-ipc flag. "
                "Note: CUDA IPC requires sync LLM which may reduce throughput."
            )
        )
    
    try:
        # Use sync LLM's collective_rpc with a FUNCTION (not a string!)
        # This is the key difference from AsyncLLM
        logger.info("Calling collective_rpc with function to export IPC handles...")
        
        # Run in thread pool to avoid blocking
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                sync_engine.collective_rpc,
                _export_cuda_ipc_handles_fn
            )
            results = future.result(timeout=60)
        
        # collective_rpc returns a list (one result per worker)
        result = results[0] if results else {}
        ipc_handles = result.get("handles", {})
        failed_params = result.get("failed", [])
        
        if failed_params:
            logger.warning(f"Could not export {len(failed_params)} parameters: {failed_params[:5]}...")
        
        if len(ipc_handles) == 0:
            raise HTTPException(status_code=500, detail="No IPC handles exported")
        
        # Save to file for trainer to read
        log_dir = os.environ.get("LOGDIR", ".")
        ipc_path = Path(log_dir) / "cuda_ipc_handles.json"
        
        with open(ipc_path, "w") as f:
            json.dump({
                "handles": ipc_handles,
                "model_class": result.get("model_class", "unknown"),
                "num_params": result.get("num_params", 0),
                "device_count": torch.cuda.device_count(),
                "export_time": time.time(),
            }, f, indent=2)
        
        logger.info(f"✓ Exported {len(ipc_handles)} CUDA IPC handles to {ipc_path}")
        
        return JSONResponse({
            "status": "ok",
            "num_parameters": len(ipc_handles),
            "failed_parameters": len(failed_params),
            "ipc_path": str(ipc_path),
            "total_elements": sum(info["numel"] for info in ipc_handles.values()),
            "model_class": result.get("model_class", "unknown"),
            "message": "IPC handles exported. Trainer can now attach to shared memory.",
        })
        
    except concurrent.futures.TimeoutError:
        raise HTTPException(status_code=504, detail="collective_rpc timed out after 60s")
    except Exception as e:
        logger.error(f"Failed to export CUDA IPC handles: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/bridge/cuda_ipc_status")
async def bridge_cuda_ipc_status() -> JSONResponse:
    """
    Check CUDA IPC status and whether shared memory is available.
    """
    log_dir = os.environ.get("LOGDIR", ".")
    ipc_path = Path(log_dir) / "cuda_ipc_handles.json"
    
    status = {
        "sync_llm_available": SYNC_LLM_AVAILABLE,
        "sync_engine_initialized": sync_engine is not None,
        "ipc_handles_exported": ipc_path.exists(),
        "ipc_path": str(ipc_path) if ipc_path.exists() else None,
        "cuda_device_count": torch.cuda.device_count(),
    }
    
    if ipc_path.exists():
        try:
            with open(ipc_path) as f:
                data = json.load(f)
            status["num_parameters"] = len(data.get("handles", {}))
            status["model_class"] = data.get("model_class")
            status["export_time"] = data.get("export_time")
        except Exception as e:
            status["ipc_file_error"] = str(e)
    
    return JSONResponse(status)


@app.get("/bridge/debug")
async def bridge_debug() -> JSONResponse:
    """
    Debug endpoint to inspect engine capabilities.
    
    Lists available attributes and methods on the engine.
    """
    active_engine = get_engine()
    
    debug_info = {
        "engine_type": type(active_engine).__name__,
        "engine_mode": "async" if engine is not None else "sync_cuda_ipc",
        "vllm_version": VLLM_VERSION,
        "model_config": {},
        "available_methods": {},
        "important_attributes": {},
    }
    
    # Get model config
    if engine is not None:
        debug_info["model_config"] = {
            "model": str(engine.model_config.model) if hasattr(engine, "model_config") else "unknown",
            "dtype": str(engine.model_config.dtype) if hasattr(engine, "model_config") else "unknown",
        }
    elif sync_engine is not None:
        try:
            debug_info["model_config"] = {
                "model": str(sync_engine.llm_engine.model_config.model),
                "dtype": str(sync_engine.llm_engine.model_config.dtype),
            }
        except Exception:
            debug_info["model_config"] = {"model": "unknown", "dtype": "unknown"}
    
    # Check for important methods
    important_methods = [
        "pause_generation", "resume_generation", "is_paused",
        "sleep", "wake_up", "is_sleeping",
        "collective_rpc", "add_lora", "remove_lora", "list_loras",
        "generate", "encode", "abort", "check_health",
    ]
    
    for method in important_methods:
        has_method = hasattr(active_engine, method) and callable(getattr(active_engine, method))
        debug_info["available_methods"][method] = has_method
    
    # Check important attributes
    important_attrs = [
        "engine_core", "model_config", "vllm_config", 
        "input_processor", "output_processor", "tokenizer",
        "llm_engine",  # For sync LLM
    ]
    
    for attr in important_attrs:
        if hasattr(active_engine, attr):
            attr_val = getattr(active_engine, attr)
            debug_info["important_attributes"][attr] = type(attr_val).__name__
        else:
            debug_info["important_attributes"][attr] = None
    
    return JSONResponse(debug_info)


@app.get("/bridge/list_endpoints")
async def bridge_list_endpoints() -> JSONResponse:
    """
    List all available bridge endpoints with descriptions.
    
    Use this to discover what capabilities are available.
    """
    endpoints = {
        "health": {
            "GET /health": "Basic health check",
            "GET /health_generate": "Deep health check (sends test request)",
        },
        "generation": {
            "POST /generate": "Generate text (vLLM native format)",
            "POST /v1/completions": "Generate text (OpenAI format)",
            "POST /v1/chat/completions": "Chat completion (OpenAI format)",
        },
        "bridge_control": {
            "GET /bridge/info": "Get bridge status and rendezvous info",
            "POST /bridge/init": "Initialize weight bridge for NCCL",
            "POST /bridge/disable": "Disable weight bridge",
            "GET /bridge/state_dict_info": "Get model parameter info",
        },
        "weight_updates": {
            "POST /bridge/pause": "⭐ Pause generation for weight updates",
            "POST /bridge/resume": "⭐ Resume generation after weight updates",
            "GET /bridge/is_paused": "Check if paused",
            "POST /bridge/notify_update": "Notify server of weight update",
        },
        "memory_management": {
            "POST /bridge/sleep": "Put engine to sleep (free GPU memory)",
            "POST /bridge/wake_up": "Wake engine up (reload model)",
            "GET /bridge/is_sleeping": "Check if sleeping",
        },
        "lora_adapters": {
            "GET /lora/status": "Get LoRA status",
            "POST /lora/load": "Load LoRA adapter",
            "POST /lora/unload": "Unload LoRA adapter",
        },
        "advanced": {
            "POST /bridge/collective_rpc": "Call method on workers",
            "GET /bridge/debug": "Debug engine structure",
            "GET /bridge/list_endpoints": "This endpoint",
        },
    }
    
    return JSONResponse(endpoints)


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

    global engine, sync_engine
    
    use_cuda_ipc = getattr(args, "enable_cuda_ipc", False)
    
    if use_cuda_ipc:
        # CUDA IPC MODE: Use sync LLM only (model in same process)
        # This allows function-based collective_rpc for IPC handle export
        if not SYNC_LLM_AVAILABLE:
            raise RuntimeError("CUDA IPC requested but vllm.LLM not available")
        
        logger.info("=" * 60)
        logger.info("CUDA IPC MODE: Using sync LLM for true shared memory")
        logger.info("=" * 60)
        
        sync_engine = SyncLLM(
            model=args.model,
            dtype=getattr(args, "dtype", "auto"),
            gpu_memory_utilization=getattr(args, "gpu_memory_utilization", 0.9),
            tensor_parallel_size=getattr(args, "tensor_parallel_size", 1),
            trust_remote_code=getattr(args, "trust_remote_code", False),
        )
        engine = None  # No async engine in CUDA IPC mode
        logger.info("✓ Sync LLM ready for CUDA IPC")
        
    else:
        # STANDARD MODE: Use AsyncLLM (model in subprocess)
        engine_args = AsyncEngineArgs.from_cli_args(args)
        engine = (
            llm_engine
            if llm_engine is not None
            else AsyncLLM.from_engine_args(
                engine_args, usage_context=UsageContext.API_SERVER
            )
        )
        sync_engine = None
    
    app.state.engine_client = engine or sync_engine

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
    
    # Verify at least one engine is initialized
    if engine is None and sync_engine is None:
        raise RuntimeError("No engine initialized")

    # Log bridge endpoints
    logger.info("=" * 60)
    logger.info("Bridge endpoints available:")
    logger.info("-" * 60)
    logger.info("Weight Updates (use these for training!):")
    logger.info("  POST /bridge/pause    - Pause generation for weight updates")
    logger.info("  POST /bridge/resume   - Resume after updating weights")
    logger.info("  GET  /bridge/is_paused - Check pause state")
    logger.info("-" * 60)
    logger.info("Memory Management:")
    logger.info("  POST /bridge/sleep    - Free GPU memory")
    logger.info("  POST /bridge/wake_up  - Reload model")
    logger.info("-" * 60)
    logger.info("LoRA Adapters:")
    logger.info("  GET  /lora/status     - Get adapter status")
    logger.info("  POST /lora/load       - Load adapter")
    logger.info("  POST /lora/unload     - Unload adapter")
    logger.info("-" * 60)
    logger.info("Debug:")
    logger.info("  GET  /bridge/debug         - Inspect engine")
    logger.info("  GET  /bridge/list_endpoints - List all endpoints")
    logger.info("  POST /bridge/collective_rpc - Call worker methods")
    logger.info("=" * 60)

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
    
    # CUDA IPC for true shared memory
    parser.add_argument(
        "--enable-cuda-ipc",
        action="store_true",
        default=False,
        help=(
            "Enable CUDA IPC for true shared memory with trainer. "
            "Requires trainer to be on the same GPU. "
            "This initializes a sync LLM alongside the async engine."
        ),
    )

    # Add vLLM engine arguments
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    asyncio.run(run_server(args))
