import asyncio
import warnings

import aiohttp

from atroposlib.envs.server_handling.server_baseline import APIServerConfig
from atroposlib.envs.server_handling.sglang_server import SGLangServer


class StatefulSGLangServer(SGLangServer):
    """
    SGLangServer extension for stateful Delta-Sync protocol.
    Optimizes network payload by sending only token deltas.
    Includes auto-rebuild for cache-miss resilience.
    """

    def __init__(self, config: APIServerConfig, reasoning_config=None):
        super().__init__(config, reasoning_config=reasoning_config)
        self._session = None

    async def _get_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )
        return self._session

    async def _tokens_and_logprobs_completion_wrapper(
        self, **kwargs
    ) -> tuple[list, list, list, list]:
        """
        Interacts with SGLang /generate via raw HTTP, optimized for stateful deltas.
        """
        assert (
            kwargs.get("model", None) is not None
        ), "Model is required for completion!"
        assert (
            kwargs.get("prompt", None) is not None
            or kwargs.get("input_ids", None) is not None
        ), "Prompt or input_ids is required!"

        if "input_ids" in kwargs:
            prompt_tokens_full = kwargs.pop("input_ids")
            kwargs.pop("prompt", None)
        else:
            prompt_tokens_full = self.tokenizer.encode(kwargs.pop("prompt"))

        # Clean double BOS if needed
        if (
            len(prompt_tokens_full) >= 2
            and prompt_tokens_full[0]
            == self.tokenizer.bos_token_id
            == prompt_tokens_full[1]
        ):
            prompt_tokens_full = prompt_tokens_full[1:]

        if "max_tokens" in kwargs:
            kwargs["max_new_tokens"] = kwargs.pop("max_tokens")
        if "model" in kwargs:
            kwargs.pop("model")

        # Extract new tokens (delta) if this is a continuation.
        is_delta_request = False
        if "delta_input_ids" in kwargs:
            payload_input_ids = kwargs.pop("delta_input_ids")
            is_delta_request = True
        else:
            payload_input_ids = prompt_tokens_full

        request_data = {
            "input_ids": payload_input_ids,
            "sampling_params": kwargs,
            "return_logprob": True,
            "return_text_in_logprobs": False,
        }

        async def fetch_generate(payload):
            session = await self._get_session()
            async with session.post(
                f"{self.config.base_url.replace('/v1', '')}/generate",
                json=payload,
                headers=(
                    {"Authorization": f"Bearer {self.config.api_key}"}
                    if self.config.api_key
                    else {}
                ),
            ) as response:
                response.raise_for_status()
                return await response.json()

        try:
            results = await fetch_generate(request_data)
        except Exception as e:
            if is_delta_request:
                warnings.warn(
                    f"Stateful request backfired ({e}). Attempting stateless fallback..."
                )
                request_data["input_ids"] = prompt_tokens_full
                results = await fetch_generate(request_data)
            else:
                raise e

        if not isinstance(results, list):
            results = [results]

        output_tokens_list = []
        output_logprobs_list = []
        finish_reasons_list = []

        for result in results:
            meta_info = result.get("meta_info", {})
            output_token_logprobs = meta_info.get("output_token_logprobs", [])
            logprobs = [item[0] for item in output_token_logprobs]
            output_ids = [item[1] for item in output_token_logprobs]
            finish_reason = meta_info.get("finish_reason", None)

            output_tokens_list.append(output_ids)
            output_logprobs_list.append(logprobs)
            finish_reasons_list.append(finish_reason)

        return (
            prompt_tokens_full,
            output_tokens_list,
            output_logprobs_list,
            finish_reasons_list,
        )
