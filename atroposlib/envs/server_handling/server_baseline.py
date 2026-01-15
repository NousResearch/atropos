import asyncio
import collections
import time
from abc import ABC, abstractmethod
from asyncio import exceptions
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import numpy as np
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.completion import Completion
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Valid reasoning effort levels
VALID_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}


@dataclass
class ReasoningConfig:
    """
    Configuration for reasoning/thinking model support.

    This config is used by ServerManager to automatically inject the appropriate
    extra_body parameters into API requests based on the provider (OpenAI vs others).

    Attributes:
        enabled: Whether reasoning mode is enabled. Auto-set to True if effort or
                 max_tokens are specified.
        effort: Reasoning effort level. One of: "none", "minimal", "low", "medium",
                "high", "xhigh". Default None (not specified).
        max_tokens: Maximum tokens for reasoning. No validation enforced - provider
                   limits vary (e.g., OpenRouter currently caps Anthropic at 1024-32000,
                   but native Anthropic supports up to 128k). Default None.
    """

    enabled: bool = False
    effort: Optional[str] = None
    max_tokens: Optional[int] = None

    def __post_init__(self):
        """Validate and auto-enable if effort or max_tokens are set."""
        # Validate effort if provided
        if self.effort is not None and self.effort not in VALID_REASONING_EFFORTS:
            raise ValueError(
                f"Invalid reasoning_effort: {self.effort}. "
                f"Must be one of: {VALID_REASONING_EFFORTS}"
            )

        # Note: As of 2024, OpenRouter caps Anthropic reasoning tokens at 1024-32000
        # See: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
        # However, we don't enforce this limit here since providers may extend ranges
        # (e.g., Anthropic's latest models support up to 128k extended thinking)

        # Auto-enable if effort or max_tokens are specified
        # Because if either of these are enabled, reasoning in
        # OpenRouter must also be set to Enabled
        if self.effort is not None or self.max_tokens is not None:
            self.enabled = True

    def is_reasoning_kwargs_active(self) -> bool:
        """Check if reasoning is active (enabled with any settings)."""
        return self.enabled

    # Mapping from effort levels to approximate max_tokens values
    # Based on OpenRouter's effort-to-budget_tokens formula percentages:
    # https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
    # Calculated as percentage of 32k base: none=min, minimal=10%, low=20%,
    # medium=50%, high=80%, xhigh=95%
    EFFORT_TO_MAX_TOKENS = {
        "none": 1024,  # Minimum/disabled
        "minimal": 3200,  # ~10% of 32k
        "low": 6400,  # ~20% of 32k
        "medium": 16000,  # ~50% of 32k
        "high": 25600,  # ~80% of 32k
        "xhigh": 30400,  # ~95% of 32k
    }

    def build_extra_body(
        self, base_url: Optional[str] = None, use_max_tokens: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Build the extra_body dict for API requests based on provider.

        Args:
            base_url: The API base URL, used to detect OpenAI official endpoint.
            use_max_tokens: If True, convert effort levels to max_tokens values
                           instead of passing effort strings. Useful for providers
                           that only support token-based reasoning limits.

        Returns:
            Dict to merge into extra_body, or None if reasoning not active.

        Note:
            OpenRouter only allows ONE of effort or max_tokens, not both.
            When both are specified, effort takes priority (unless use_max_tokens=True).
        """
        if not self.is_reasoning_kwargs_active():
            return None

        # Detect if using official OpenAI endpoint
        is_openai_official = base_url and "api.openai.com" in base_url

        if is_openai_official:
            # OpenAI only accepts reasoning_effort at top level, not nested reasoning object
            # They also don't support max_tokens for reasoning
            effort = self.effort if self.effort else "medium"
            # Map our extended effort levels to OpenAI's supported values
            openai_effort_map = {
                "none": "low",  # OpenAI doesn't have "none", use low
                "minimal": "low",  # OpenAI doesn't have "minimal", use low
                "low": "low",
                "medium": "medium",
                "high": "high",
                "xhigh": "high",  # OpenAI doesn't have "xhigh", use high
            }
            return {"reasoning_effort": openai_effort_map.get(effort, "medium")}
        else:
            # Standard format for OpenRouter, Nebius, Nous Portal, etc.
            reasoning = {"enabled": True}

            # If use_max_tokens is True, convert effort to max_tokens
            if use_max_tokens and self.effort is not None:
                reasoning["max_tokens"] = self.EFFORT_TO_MAX_TOKENS.get(
                    self.effort, 8192
                )
            elif self.effort is not None:
                # Pass effort string directly (provider may or may not support it)
                reasoning["effort"] = self.effort
            elif self.max_tokens is not None:
                # Use explicit max_tokens if provided
                reasoning["max_tokens"] = self.max_tokens

            return {"reasoning": reasoning}

    @classmethod
    def from_env_config(cls, env_config) -> "ReasoningConfig":
        """
        Create a ReasoningConfig from a BaseEnvConfig.

        This is used by BaseEnv to convert environment config settings
        into the reasoning configuration used by ServerManager.

        Args:
            env_config: A BaseEnvConfig (or subclass) instance with reasoning fields.

        Returns:
            A ReasoningConfig instance configured based on the env_config.
        """
        # Get reasoning settings from env config
        thinking_mode = getattr(env_config, "thinking_mode", False)
        reasoning_effort = getattr(env_config, "reasoning_effort", None)
        max_reasoning_tokens = getattr(env_config, "max_reasoning_tokens", None)

        # Determine if enabled: explicitly True, or implied by effort/max_tokens
        enabled = (
            thinking_mode
            or reasoning_effort is not None
            or max_reasoning_tokens is not None
        )

        return cls(
            enabled=enabled,
            effort=reasoning_effort,
            max_tokens=max_reasoning_tokens,
        )


class AsyncSemWithAdaptiveWeight(asyncio.Semaphore):
    def __init__(self, value: int):
        super().__init__(value=value)
        self.max_val = value
        self.weight = 1.0

    def update_weight(self, weight: float) -> None:
        """
        Update the weight of the semaphore.
        """
        self.weight = weight

    def min_val(self):
        """
        Returns the minimum value of the semaphore.
        """
        return self.max_val * (1.0 - self.weight)

    def release(self):
        """Release a semaphore, incrementing the internal counter by one.

        When it was zero on entry and another coroutine is waiting for it to
        become larger than zero again, wake up that coroutine.

        If weight is set, it'll only wake up next if the value is greater than the max_val * weight
        """
        self._value += 1
        if self._value > self.min_val():
            self._wake_up_next()

    def locked(self):
        """Returns True if semaphore cannot be acquired immediately."""
        return self._value <= self.min_val() or (
            any(not w.cancelled() for w in (self._waiters or ()))
        )

    async def acquire(self):
        """Acquire a semaphore.

        If the internal counter is larger than zero on entry,
        decrement it by one and return True immediately.  If it is
        zero on entry, block, waiting until some other coroutine has
        called release() to make it larger than 0, and then return
        True.
        """
        if not self.locked():
            self._value -= 1
            return True

        if self._waiters is None:
            self._waiters = collections.deque()
        fut = self._get_loop().create_future()
        self._waiters.append(fut)

        # Finally block should be called before the CancelledError
        # handling as we don't want CancelledError to call
        # _wake_up_first() and attempt to wake up itself.
        try:
            try:
                await fut
            finally:
                self._waiters.remove(fut)
        except exceptions.CancelledError:
            if not fut.cancelled():
                self._value += 1
                self._wake_up_next()
            raise

        if self._value > self.min_val():
            self._wake_up_next()
        return True


class ServerBaseline(BaseModel):
    """
    Baseline configuration for server information. If local, uses ports 9004-9007 for the servers,
    assuming a 1:1 split of GPUs.
    """

    timeout: int = Field(
        default=1200, description="Timeout for the request in seconds."
    )
    num_max_requests_at_once: int = Field(
        default=512,
        description="Maximum number of concurrent requests. You should divide this by the n kwarg.",
    )
    num_requests_for_eval: int = Field(
        default=64, description="Maximum number of concurrent requests for evaluation."
    )
    model_name: str = Field(
        default="default",
        description="The model name to use. Only works with sglang, please provide the model name.",
    )
    rolling_buffer_length: int = Field(
        default=1000, description="Length of the rolling buffer to store metrics."
    )
    server_type: Literal["openai", "trl", "sglang", "vllm"] = Field(
        default="openai", description="Type of server to use"
    )


class APIServerConfig(ServerBaseline):
    """
    API server configuration.
    """

    api_key: Optional[str] = Field(default="", description="API key for the server.")
    base_url: Optional[str] = Field(default="", description="Base URL for the server.")
    n_kwarg_is_ignored: bool = Field(
        default=False, description="Whether the n kwarg is ignored by this API server."
    )
    health_check: bool = Field(
        default=True, description="Whether to perform a health check on the server."
    )


class APIServer(ABC):
    """
    Abstract class for API servers.
    """

    def __init__(
        self,
        config: APIServerConfig,
        reasoning_config: Optional[ReasoningConfig] = None,
    ):
        self.config = config
        self.reasoning_config = reasoning_config
        self.sem = AsyncSemWithAdaptiveWeight(config.num_max_requests_at_once)
        self.eval_sem = AsyncSemWithAdaptiveWeight(config.num_requests_for_eval)
        self.server_healthy = True
        self.attempts_list = []
        self.request_timings = []
        # in case eval is much different, we should keep different buffers
        self.eval_attempts_list = []
        self.eval_request_timings = []
        self.check_task = None
        self.initialized = False

    def _inject_reasoning_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inject reasoning configuration into kwargs if reasoning is enabled.

        This method can be overridden by subclasses to handle implementation-specific
        quirks for different server types (vLLM, SGLang, OpenAI, etc.).

        The caller can pass `skip_reasoning=True` in kwargs to bypass injection.

        Args:
            kwargs: The kwargs dict to potentially modify

        Returns:
            Modified kwargs dict with reasoning config injected (if applicable)
        """
        # Check if caller explicitly wants to skip reasoning injection
        skip_reasoning = kwargs.pop("skip_reasoning", False)
        if skip_reasoning:
            return kwargs

        # Check if reasoning is configured and active
        if (
            self.reasoning_config is None
            or not self.reasoning_config.is_reasoning_kwargs_active()
        ):
            return kwargs

        # Get base_url to determine provider type
        base_url = getattr(self.config, "base_url", None)
        is_openai_official = base_url and "api.openai.com" in base_url

        # Build the extra_body for reasoning
        reasoning_extra_body = self.reasoning_config.build_extra_body(base_url)

        if reasoning_extra_body:
            # Merge with any existing extra_body in kwargs
            existing_extra_body = kwargs.get("extra_body", {}) or {}
            kwargs["extra_body"] = {**existing_extra_body, **reasoning_extra_body}

        # OpenAI reasoning models have specific requirements
        if is_openai_official:
            # OpenAI reasoning models require temperature=1.0 (or unset)
            kwargs["temperature"] = 1.0

            # OpenAI reasoning models use max_completion_tokens instead of max_tokens
            if "max_tokens" in kwargs and kwargs["max_tokens"]:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")

        return kwargs

    async def update_weight(self, weight: float) -> None:
        """
        Update the weight of the semaphores
        """
        # need to update sems
        self.sem.update_weight(weight)
        self.eval_sem.update_weight(weight)

    @abstractmethod
    async def check_server_status_task(self, chat_completion: bool = True):
        """
        Check the status of the server. Should be overridden by the child class.
        Set self.server_healthy to True if the server is healthy.
        """
        self.server_healthy = False

    async def wandb_metrics(
        self, metrics_dict: Optional[dict], server_name: Optional[str]
    ):
        """
        Add metrics to the metrics dictionary.

        If you want to add more metrics, you can do so by overriding this method, but make sure to call
        super().wandb_metrics(metrics_dict, server_name) first to get the default metrics, if you still want them.
        """
        if server_name is None:
            server_name = "server"
        if len(self.request_timings) > 0:
            metrics_dict[f"server/{server_name}_request_time_avg"] = np.mean(
                self.request_timings
            )
            metrics_dict[f"server/{server_name}_request_time_std"] = np.std(
                self.request_timings
            )
            metrics_dict[f"server/{server_name}_request_time_99p"] = np.percentile(
                self.request_timings, 99
            )
        if len(self.eval_request_timings) > 0:
            metrics_dict[f"server/{server_name}_eval_request_time_avg"] = np.mean(
                self.eval_request_timings
            )
            metrics_dict[f"server/{server_name}_eval_request_time_std"] = np.std(
                self.eval_request_timings
            )
            metrics_dict[f"server/{server_name}_eval_request_time_99p"] = np.percentile(
                self.eval_request_timings, 99
            )
        if len(self.attempts_list) > 0:
            metrics_dict[f"server/{server_name}_average_num_attempts"] = np.mean(
                self.attempts_list
            )
        if len(self.eval_attempts_list) > 0:
            metrics_dict[f"server/{server_name}_eval_retry_rate"] = np.mean(
                self.eval_attempts_list
            )
        return metrics_dict

    @abstractmethod
    async def _chat_completion_wrapper(self, **kwargs) -> ChatCompletion:
        """
        Wrapper for the chat completion. Should be overridden by the child class and return a ChatCompletion object.
        """
        pass

    @abstractmethod
    async def _completion_wrapper(self, **kwargs) -> Completion:
        """
        Wrapper for the completion. Should be overridden by the child class and return a Completion object.
        """
        pass

    @abstractmethod
    async def _tokens_and_logprobs_completion_wrapper(
        self, **kwargs
    ) -> tuple[list, list, list, list]:
        """
        Wrapper for tokens and logprobs completion. Should be overridden by the child class.
        Returns a tuple of (prompt_tokens, output_tokens, output_logprobs, finish_reasons).
        """
        pass

    @retry(
        stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=10)
    )
    async def _chat_comp(self, stat_dict, **kwargs) -> ChatCompletion:
        """
        Simple retry and stat collection wrapper for the chat completion.
        """
        while not self.server_healthy:
            await asyncio.sleep(1)
        async with self.sem:
            if stat_dict.get("start", None) is None:
                stat_dict["start"] = time.time()
            stat_dict["attempts"] += 1
            completions = await self._chat_completion_wrapper(**kwargs)
            stat_dict["end"] = time.time()
            return completions

    @retry(
        stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=10)
    )
    async def _chat_eval(self, stat_dict, **kwargs) -> ChatCompletion:
        """
        Simple retry and stat collection wrapper for the chat completion.
        """
        while not self.server_healthy:
            await asyncio.sleep(1)
        async with self.eval_sem:
            if stat_dict.get("start", None) is None:
                stat_dict["start"] = time.time()
            stat_dict["attempts"] += 1
            completions = await self._chat_completion_wrapper(**kwargs)
            stat_dict["end"] = time.time()
            return completions

    @retry(
        stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=10)
    )
    async def chat_completion(self, **kwargs) -> ChatCompletion:
        """
        Chat completion handler, waits for the server to be healthy and then calls the chat completion wrapper.

        Automatically injects reasoning config if configured. Pass `skip_reasoning=True`
        to bypass reasoning injection for this specific call.
        """
        if not self.initialized:
            if self.config.health_check:
                if (
                    self.config.base_url is not None
                ):  # skip health check if using OpenAI API
                    self.check_task = asyncio.create_task(
                        self.check_server_status_task()
                    )
                else:
                    self.server_healthy = True
            else:
                self.server_healthy = True
            self.initialized = True
        kwargs["model"] = self.config.model_name
        split = kwargs.pop("split", "train")

        # Inject reasoning config if enabled (can be skipped via skip_reasoning=True)
        kwargs = self._inject_reasoning_kwargs(kwargs)

        stat_dict = {}
        stat_dict["attempts"] = 0
        if split == "train":
            ret_data = await self._chat_comp(stat_dict, **kwargs)
            self.request_timings.append(stat_dict["end"] - stat_dict["start"])
            self.attempts_list.append(stat_dict["attempts"])
        else:
            # Give separate eval workers, if desired, gotta go fast for those evals
            ret_data = await self._chat_eval(stat_dict, **kwargs)
            self.eval_request_timings.append(stat_dict["end"] - stat_dict["start"])
            self.eval_attempts_list.append(stat_dict["attempts"])
        return ret_data

    @retry(
        stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=10)
    )
    async def _comp(self, stat_dict, **kwargs) -> Completion:
        """
        Simple retry and stat collection wrapper for the completion.
        """
        while not self.server_healthy:
            await asyncio.sleep(1)
        async with self.sem:
            if stat_dict.get("start", None) is None:
                stat_dict["start"] = time.time()
            stat_dict["attempts"] += 1
            completions = await self._completion_wrapper(**kwargs)
            stat_dict["end"] = time.time()
            return completions

    @retry(
        stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=10)
    )
    async def _comp_eval(self, stat_dict, **kwargs) -> Completion:
        """
        Simple retry and stat collection wrapper for the completion.
        """
        while not self.server_healthy:
            await asyncio.sleep(1)
        async with self.eval_sem:
            if stat_dict.get("start", None) is None:
                stat_dict["start"] = time.time()
            stat_dict["attempts"] += 1
            completions = await self._completion_wrapper(**kwargs)
            stat_dict["end"] = time.time()
            return completions

    async def completion(self, **kwargs) -> Completion:
        """
        Completion handler, waits for the server to be healthy and then calls the completion wrapper.

        Automatically injects reasoning config if configured. Pass `skip_reasoning=True`
        to bypass reasoning injection for this specific call.
        """
        if not self.initialized:
            if self.config.health_check:
                if (
                    self.config.base_url is not None
                ):  # skip health check if using OpenAI API
                    self.check_task = asyncio.create_task(
                        self.check_server_status_task(chat_completion=False)
                    )
                else:
                    self.server_healthy = True
            else:
                # If health_check is False, always assume healthy
                self.server_healthy = True
            self.initialized = True
        kwargs["model"] = self.config.model_name
        split = kwargs.pop("split", "train")

        # Inject reasoning config if enabled (can be skipped via skip_reasoning=True)
        kwargs = self._inject_reasoning_kwargs(kwargs)

        stat_dict = {}
        stat_dict["attempts"] = 0
        if split == "train":
            ret_data = await self._comp(stat_dict, **kwargs)
            self.request_timings.append(stat_dict["end"] - stat_dict["start"])
            self.attempts_list.append(stat_dict["attempts"])
        else:
            # Give separate eval workers, if desired, gotta go fast for those evals
            ret_data = await self._comp_eval(stat_dict, **kwargs)
            self.eval_request_timings.append(stat_dict["end"] - stat_dict["start"])
            self.eval_attempts_list.append(stat_dict["attempts"])
        return ret_data

    @retry(
        stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=10)
    )
    async def _tokens_and_logprobs_comp(
        self, stat_dict, **kwargs
    ) -> tuple[list, list, list, list]:
        """
        Simple retry and stat collection wrapper for tokens and logprobs completion.
        """
        while not self.server_healthy:
            await asyncio.sleep(1)
        async with self.sem:
            if stat_dict.get("start", None) is None:
                stat_dict["start"] = time.time()
            stat_dict["attempts"] += 1
            completions = await self._tokens_and_logprobs_completion_wrapper(**kwargs)
            stat_dict["end"] = time.time()
            return completions

    @retry(
        stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=10)
    )
    async def _tokens_and_logprobs_comp_eval(
        self, stat_dict, **kwargs
    ) -> tuple[list, list, list, list]:
        """
        Simple retry and stat collection wrapper for tokens and logprobs completion.
        """
        while not self.server_healthy:
            await asyncio.sleep(1)
        async with self.eval_sem:
            if stat_dict.get("start", None) is None:
                stat_dict["start"] = time.time()
            stat_dict["attempts"] += 1
            completions = await self._tokens_and_logprobs_completion_wrapper(**kwargs)
            stat_dict["end"] = time.time()
            return completions

    async def tokens_and_logprobs_completion(
        self, **kwargs
    ) -> tuple[list, list, list, list]:
        """
        Tokens and logprobs completion handler, waits for the server to be healthy and then calls the wrapper.
        Returns a tuple of (prompt_tokens, output_tokens, output_logprobs, finish_reasons).
        """
        if not self.initialized:
            if self.config.health_check:
                if (
                    self.config.base_url is not None
                ):  # skip health check if using OpenAI API
                    self.check_task = asyncio.create_task(
                        self.check_server_status_task(chat_completion=False)
                    )
                else:
                    self.server_healthy = True
            else:
                # If health_check is False, always assume healthy
                self.server_healthy = True
            self.initialized = True
        kwargs["model"] = self.config.model_name
        split = kwargs.pop("split", "train")
        stat_dict = {}
        stat_dict["attempts"] = 0
        if split == "train":
            ret_data = await self._tokens_and_logprobs_comp(stat_dict, **kwargs)
            self.request_timings.append(stat_dict["end"] - stat_dict["start"])
            self.attempts_list.append(stat_dict["attempts"])
        else:
            # Give separate eval workers, if desired, gotta go fast for those evals
            ret_data = await self._tokens_and_logprobs_comp_eval(stat_dict, **kwargs)
            self.eval_request_timings.append(stat_dict["end"] - stat_dict["start"])
            self.eval_attempts_list.append(stat_dict["attempts"])
        return ret_data
