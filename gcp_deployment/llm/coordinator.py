"""
LLM Coordinator for managing multiple agent LLM requests
"""

import asyncio
import logging
import os
import random  # For jitter
import time
from collections import OrderedDict  # Added OrderedDict for LRU cache
from dataclasses import (  # Added asdict for completeness, though not strictly used in LLMCoordinator serialization yet
    asdict,
    dataclass,
    field,
)
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from asyncio_throttle import Throttler

# Gemini API
try:
    import google.api_core.exceptions  # For specific Gemini error handling
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    google = None  # Make it explicit that google.api_core won't exist
    logging.warning("google-generativeai not installed, falling back to mock responses")

logger = logging.getLogger(__name__)


@dataclass
class LLMRequest:
    """LLM request structure"""

    agent_id: str
    prompt: str
    max_tokens: int
    response_future: asyncio.Future  # Future to signal completion
    temperature: float = 0.7
    timestamp: float = field(default_factory=time.time)  # Use factory for timestamp
    model_preference: Optional[str] = None  # e.g., "primary", "secondary"


@dataclass
class LLMResponse:
    """LLM response structure"""

    agent_id: str
    response: str
    latency: float
    success: bool
    error_message: Optional[str] = None
    retries_attempted: int = 0


class LLMCoordinator:
    """
    Coordinates LLM requests for multiple agents
    Handles batching, caching, and fallback strategies
    """

    def __init__(self, config):
        self.config = config
        llm_config = config.llm if hasattr(config, "llm") else config
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.response_cache: OrderedDict[str, str] = OrderedDict()
        self.max_cache_size = getattr(llm_config, "max_cache_size", 1000)
        self.rate_limit_per_second = getattr(llm_config, "rate_limit_per_second", 10)
        self.throttler = Throttler(rate_limit=self.rate_limit_per_second, period=1.0)
        self.total_requests = 0
        self.cached_responses = 0
        self.failed_requests_after_retries = 0
        self.api_errors_before_retry = 0
        self._batch_processor_task: Optional[asyncio.Task] = None
        self._running = False

        self.gemini_model = None  # For the primary model
        self.gemini_model_secondary = None  # For the optional secondary model

        self._init_gemini()
        primary_model_name_for_log = getattr(llm_config, "model_name", "gemini-pro")
        secondary_model_name_for_log = getattr(llm_config, "secondary_model_name", None)
        logger.info(
            f"LLM Coordinator initialized. Primary Model: {primary_model_name_for_log}, Secondary Model: {secondary_model_name_for_log or 'N/A'}, Rate Limit: {self.rate_limit_per_second}/s, Cache Size: {self.max_cache_size}"
        )

    def _init_gemini(self):
        """Initialize Gemini API"""
        self.gemini_model = None
        self.gemini_model_secondary = None
        llm_config_obj = self.config.llm if hasattr(self.config, "llm") else self.config

        if not GEMINI_AVAILABLE:
            logger.warning(
                "Gemini SDK (google-generativeai) not available. LLMCoordinator will use mock responses for all models."
            )
            return

        # Get API key from environment
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.warning(
                "No Gemini API key (GEMINI_API_KEY/GOOGLE_API_KEY). LLMCoordinator will use mock responses for all models."
            )
            return

        try:
            # Configure Gemini
            genai.configure(api_key=api_key)

            # Initialize primary model
            primary_model_name_from_config = getattr(
                llm_config_obj, "model_name", "gemini-pro"
            )
            # Ensure a valid default if config string is empty or not gemini related for primary
            primary_model_name = (
                primary_model_name_from_config
                if primary_model_name_from_config
                and "gemini" in primary_model_name_from_config.lower()
                else "gemini-pro"
            )
            try:
                self.gemini_model = genai.GenerativeModel(primary_model_name)
                logger.info(f"Primary Gemini model initialized: {primary_model_name}")
            except Exception as e_primary:
                logger.error(
                    "Failed to initialize primary Gemini model "{primary_model_name}': {e_primary}. Primary model will be unavailable."
                )
                self.gemini_model = None  # Ensure it's None if primary fails

            # Initialize secondary model if configured
            secondary_model_name_from_config = getattr(
                llm_config_obj, "secondary_model_name", None
            )
            if secondary_model_name_from_config:
                # Ensure a valid default if config string is not gemini related for secondary (though usually specific name is intended)
                secondary_model_name = (
                    secondary_model_name_from_config
                    if "gemini" in secondary_model_name_from_config.lower()
                    else None
                )
                if secondary_model_name:
                    try:
                        self.gemini_model_secondary = genai.GenerativeModel(
                            secondary_model_name
                        )
                        logger.info(
                            f"Secondary Gemini model initialized: {secondary_model_name}"
                        )
                    except Exception as e_secondary:
                        logger.error(
                            "Failed to initialize secondary Gemini model "{secondary_model_name}': {e_secondary}. Secondary model will be unavailable."
                        )
                        self.gemini_model_secondary = (
                            None  # Ensure it's None if secondary fails
                        )
                else:
                    logger.warning(
                        "Secondary model name "{secondary_model_name_from_config}' configured but not recognized as a Gemini model type. Secondary model will be unavailable."
                    )
            else:
                logger.info("No secondary Gemini model configured.")

        except Exception as e_configure:
            logger.error(
                f"Failed during Gemini configuration or model loading: {e_configure}. All Gemini models will be unavailable.",
                exc_info=True,
            )
            self.gemini_model = None
            self.gemini_model_secondary = None

    async def start(self):
        """Start the coordinator background tasks"""
        if self._running:
            return

        self._running = True
        self._batch_processor_task = asyncio.create_task(self._batch_processor())
        logger.info("LLM Coordinator started with batch processor.")

    async def stop(self):
        """Stop the coordinator and cleanup"""
        self._running = False

        if self.request_queue:
            # Optionally, handle any pending requests in the queue upon stop
            # For now, tasks in flight via _batch_processor might complete if not cancelled hard
            pass
        if self._batch_processor_task:
            self._batch_processor_task.cancel()
            try:
                await self._batch_processor_task
            except asyncio.CancelledError:
                logger.debug("Batch processor task cancelled.")
            self._batch_processor_task = None

        logger.info("LLM Coordinator stopped.")

    async def get_response(
        self,
        agent_id: str,
        prompt: str,
        max_tokens: int = 150,
        temperature: float = 0.7,
        model_preference: Optional[str] = None,
    ) -> str:
        """
        Get LLM response for an agent
        """
        self.total_requests += 1

        # Check cache first
        cache_key = self._get_cache_key(prompt, temperature, model_preference)
        if cache_key in self.response_cache and getattr(
            self.config.llm, "cache_responses", True
        ):
            self.cached_responses += 1
            self.response_cache.move_to_end(cache_key)  # Mark as recently used
            logger.debug(
                f"LLM Cache hit for agent {agent_id} (prompt hash: {hash(prompt[:100])}, model: {model_preference or 'primary'})"
            )
            return self.response_cache[cache_key]

        if (
            not self._running
            or not self._batch_processor_task
            or self._batch_processor_task.done()
        ):
            # This might happen if start() wasn't called or the processor died.
            logger.error(
                f"LLMCoordinator batch processor not running for agent {agent_id}. Returning fallback."
            )
            self.failed_requests_after_retries += 1  # Consider this a failure path
            return self._get_fallback_response(agent_id, prompt)

        response_future = asyncio.get_event_loop().create_future()
        request = LLMRequest(
            agent_id=agent_id,
            prompt=prompt,
            max_tokens=max_tokens,
            response_future=response_future,  # Pass the future
            temperature=temperature,
            model_preference=model_preference,  # Pass preference to request
        )

        await self.request_queue.put(request)
        logger.debug(
            f"Agent {agent_id} request queued (pref: {model_preference or 'primary'}). Prompt hash: {hash(prompt[:100])}. Queue size: {self.request_queue.qsize()}"
        )

        try:
            # Wait for the batch processor to fulfill this future
            llm_response_obj = await asyncio.wait_for(
                response_future,
                timeout=getattr(self.config.llm, "request_timeout", 60.0),
            )

            if llm_response_obj.success:
                if getattr(self.config.llm, "cache_responses", True):
                    self.response_cache[cache_key] = llm_response_obj.response
                    if len(self.response_cache) > self.max_cache_size:
                        self.response_cache.popitem(last=False)
                        logger.debug(
                            f"LLM Cache evicted. Size: {len(self.response_cache)}"
                        )
                return llm_response_obj.response
            else:
                self.failed_requests_after_retries += 1
                logger.error(
                    f"LLM request failed for {agent_id} after {llm_response_obj.retries_attempted} retries (via future, model pref: {model_preference or 'primary'}). Error: {llm_response_obj.error_message}"
                )
                return self._get_fallback_response(agent_id, prompt)
        except asyncio.TimeoutError:
            self.failed_requests_after_retries += 1
            logger.error(
                f"LLM request timed out for agent {agent_id} (prompt hash: {hash(prompt[:100])})."
            )
            return self._get_fallback_response(agent_id, prompt)
        except Exception as e_fut:
            self.failed_requests_after_retries += 1
            logger.error(
                f"Error awaiting LLM response future for agent {agent_id}: {e_fut}",
                exc_info=True,
            )
            return self._get_fallback_response(agent_id, prompt)

    async def _process_single_request(self, request: LLMRequest) -> LLMResponse:
        """Process a single LLM request"""
        start_time = time.time()

        try:
            # Apply rate limiting
            async with self.throttler:
                # Try Gemini first, fallback to mock
                if request.model_preference == "primary" and self.gemini_model:
                    response_text, retries_attempted = (
                        await self._call_gemini_api_with_retries(
                            request, self.gemini_model, "primary"
                        )
                    )
                    success = True
                    error_message = None
                elif (
                    request.model_preference == "secondary"
                    and self.gemini_model_secondary
                ):
                    response_text, retries_attempted = (
                        await self._call_gemini_api_with_retries(
                            request, self.gemini_model_secondary, "secondary"
                        )
                    )
                    success = True
                    error_message = None
                else:
                    # Use mock responses
                    response_text = await self._mock_llm_response(request)
                    success = True
                    error_message = None

        except Exception as e:
            logger.warning(
                f"Gemini API failed for {request.agent_id}: {e}, falling back to mock"
            )
            response_text = await self._mock_llm_response(request)
            success = True  # Mock always succeeds
            error_message = str(e)

        latency = time.time() - start_time

        return LLMResponse(
            agent_id=request.agent_id,
            response=response_text,
            latency=latency,
            success=success,
            error_message=error_message,
            retries_attempted=retries_attempted,
        )

    async def _call_gemini_api_with_retries(
        self, request: LLMRequest, model_instance: Any, model_name_for_log: str
    ) -> Tuple[str, int]:
        llm_config_obj = self.config.llm if hasattr(self.config, "llm") else self.config
        max_retries = getattr(llm_config_obj, "api_max_retries", 3)
        base_backoff_seconds = getattr(llm_config_obj, "api_base_backof", 1.0)
        max_backoff_seconds = getattr(llm_config_obj, "api_max_backof", 16.0)

        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                return (
                    await self._call_gemini_api(
                        request, model_instance, model_name_for_log
                    ),
                    attempt,
                )  # Return response and attempts count
            except google.api_core.exceptions.ResourceExhausted as e:  # Rate limit
                last_exception = e
                logger.warning(
                    f"Gemini API rate limit (ResourceExhausted) for {request.agent_id}, attempt {attempt+1}/{max_retries+1}. Error: {e}"
                )
                self.api_errors_before_retry += 1
            except google.api_core.exceptions.ServiceUnavailable as e:  # Server error
                last_exception = e
                logger.warning(
                    f"Gemini API service unavailable for {request.agent_id}, attempt {attempt+1}/{max_retries+1}. Error: {e}"
                )
                self.api_errors_before_retry += 1
            except (
                google.api_core.exceptions.GoogleAPIError
            ) as e:  # Other Google API errors
                last_exception = e
                logger.warning(
                    f"Gemini API GoogleAPIError for {request.agent_id}, attempt {attempt+1}/{max_retries+1}. Error: {e}"
                )
                self.api_errors_before_retry += 1
            except Exception as e:  # Catch other general exceptions during API call
                last_exception = e
                logger.error(
                    f"Generic error in _call_gemini_api for {request.agent_id}, attempt {attempt+1}/{max_retries+1}: {e}",
                    exc_info=True,
                )
                self.api_errors_before_retry += 1
                # For some errors (e.g. auth, invalid prompt), we might not want to retry.
                # For now, retrying on most errors for simplicity of demonstration.
                # Consider adding specific non-retryable exception checks here.
                # if isinstance(e, NonRetryableError): raise

            if attempt < max_retries:
                backoff_duration = min(
                    base_backoff_seconds * (2**attempt), max_backoff_seconds
                )
                jitter = random.uniform(
                    0, backoff_duration * 0.1
                )  # Add up to 10% jitter
                sleep_duration = backoff_duration + jitter
                logger.info(
                    f"Retrying Gemini API for {request.agent_id} in {sleep_duration:.2f} seconds..."
                )
                await asyncio.sleep(sleep_duration)
            else:
                logger.error(
                    f"Gemini API call failed for {request.agent_id} after {max_retries+1} attempts."
                )
                raise last_exception  # Re-raise the last caught exception after all retries
        # Should not be reached if max_retries >= 0
        raise RuntimeError("Exited retry loop unexpectedly.")

    async def _call_gemini_api(
        self, request: LLMRequest, model_instance: Any, model_name_for_log: str
    ) -> str:
        """Call Gemini API"""
        if not model_instance:
            raise Exception(
                f"Gemini model ({model_name_for_log}) not initialized for API call attempt."
            )
        try:
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=request.temperature,
                max_output_tokens=request.max_tokens,
                candidate_count=1,
            )

            # Generate response in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            api_response = await loop.run_in_executor(
                None,
                lambda: model_instance.generate_content(
                    request.prompt, generation_config=generation_config
                ),
            )

            # Extract text from response
            if api_response.parts:
                # Check for safety ratings and blocked content
                if (
                    api_response.prompt_feedback
                    and api_response.prompt_feedback.block_reason
                ):
                    raise Exception(
                        f"Prompt blocked due to: {api_response.prompt_feedback.block_reason.name}"
                    )
                if (
                    not api_response.candidates
                    or not api_response.candidates[0].content.parts
                ):
                    # Check if content is blocked in candidate
                    if (
                        api_response.candidates
                        and api_response.candidates[0].finish_reason
                        == genai.types.Candidate.FinishReason.SAFETY
                    ):
                        # Find the safety rating that caused blocking
                        blocked_rating = next(
                            (
                                r
                                for r in api_response.candidates[0].safety_ratings
                                if r.blocked
                            ),
                            None,
                        )
                        block_reason_detail = (
                            blocked_rating.category.name
                            if blocked_rating
                            else "Unknown safety reason"
                        )
                        raise Exception(
                            f"Response content blocked due to safety: {block_reason_detail}"
                        )
                    raise Exception(
                        "No response parts or content generated, possibly due to safety filters or empty response."
                    )
                return api_response.text.strip()
            else:
                # This case might also indicate blocked prompt or other issues
                block_reason = "Unknown issue"
                if (
                    api_response.prompt_feedback
                    and api_response.prompt_feedback.block_reason
                ):
                    block_reason = api_response.prompt_feedback.block_reason.name
                raise Exception(
                    f"No response parts generated. Prompt block reason: {block_reason}"
                )

        except Exception as e:
            # Catching broad exception to include any issue from generate_content or response processing
            # Specific exceptions like google.api_core.exceptions.ResourceExhausted should be caught by the caller for retry logic.
            logger.debug(
                f"Detailed error in _call_gemini_api for {request.agent_id}: {type(e).__name__} - {e}"
            )
            raise  # Re-raise to be handled by _call_gemini_api_with_retries

    async def _mock_llm_response(self, request: LLMRequest) -> str:
        """Generate mock LLM responses for testing"""
        # Add small delay to simulate API latency
        await asyncio.sleep(0.01 + 0.005 * (hash(request.agent_id) % 10) / 10)

        prompt_lower = request.prompt.lower()

        # Simple rule-based responses based on prompt content
        if "move_to" in prompt_lower or "coordinates" in prompt_lower:
            # Movement related
            responses = [
                "move_to 25 30",
                "move_to 15 45",
                "move_to 40 20",
                "move_to 35 35",
                "rest",
            ]
        elif "talk_to" in prompt_lower or "socialize" in prompt_lower:
            # Social interaction
            responses = [
                f"talk_to agent_{(hash(request.agent_id) % 100):03d}",
                "rest",
                "gather_resources",
                "move_to 20 25",
            ]
        elif "create" in prompt_lower:
            # Creation related
            responses = [
                "create_object wooden chair",
                "create_object clay pot",
                "create_object stone tool",
                "create_object woven basket",
                "rest",
            ]
        elif "gather" in prompt_lower or "resources" in prompt_lower:
            responses = ["gather_resources", "move_to 10 50", "rest"]
        else:
            # Default responses
            responses = [
                "rest",
                "move_to 30 30",
                "gather_resources",
                f"talk_to agent_{(hash(request.agent_id) % 100):03d}",
                "create_object simple tool",
            ]

        # Select response based on agent ID for consistency
        response_idx = hash(request.agent_id + request.prompt[:20]) % len(responses)
        return responses[response_idx]

    def _get_fallback_response(self, agent_id: str, prompt: str) -> str:
        """Get a simple fallback response when LLM fails"""
        fallbacks = ["rest", "move_to 25 25", "gather_resources"]
        return fallbacks[hash(agent_id) % len(fallbacks)]

    def _get_cache_key(
        self, prompt: str, temperature: float, model_preference: Optional[str] = None
    ) -> str:
        """Generate cache key for prompt"""
        pref_key = model_preference or "primary"
        return f"{hash(prompt[:100])}_{temperature:.2f}_{pref_key}"

    async def _batch_processor(self):
        """Background task to process LLM requests from the queue."""
        logger.info("LLM batch processor started.")
        while self._running:
            try:
                # Wait for a request from the queue
                # Add a small timeout to allow the loop to check self._running periodically
                try:
                    request: LLMRequest = await asyncio.wait_for(
                        self.request_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue  # No request in queue, loop back to check self._running

                if (
                    not request or not request.response_future
                ):  # Should not happen if queueing is correct
                    if request:
                        self.request_queue.task_done()  # Mark done even if invalid
                    continue

                # Check if the future is already done (e.g., timed out by caller)
                if request.response_future.done():
                    logger.debug(
                        f"Future for agent {request.agent_id} (prompt hash: {hash(request.prompt[:100])}) already done. Skipping processing."
                    )
                    self.request_queue.task_done()
                    continue

                logger.debug(
                    f"Processing request for agent {request.agent_id} (prompt hash: {hash(request.prompt[:100])}). Queue size: {self.request_queue.qsize()}"
                )

                llm_response_obj: Optional[LLMResponse] = None
                try:
                    # Process the request (this handles caching, API calls, retries)
                    llm_response_obj = await self._process_single_request(request)
                    request.response_future.set_result(llm_response_obj)
                except Exception as e_process:
                    # This catch is for unexpected errors in _process_single_request itself,
                    # though it's designed to return an LLMResponse object even on API failure.
                    logger.error(
                        f"Unexpected error in _process_single_request for {request.agent_id}: {e_process}",
                        exc_info=True,
                    )
                    # Ensure future is resolved with an error or a failure LLMResponse
                    if not request.response_future.done():
                        # Construct a generic failure response if llm_response_obj is not set
                        failed_response = LLMResponse(
                            agent_id=request.agent_id,
                            response="",  # No response text
                            latency=0,  # Latency calculation might be skewed
                            success=False,
                            error_message=f"Internal coordinator error: {str(e_process)}",
                            retries_attempted=getattr(
                                llm_response_obj, "retries_attempted", 0
                            ),  # if llm_response_obj exists
                        )
                        request.response_future.set_result(failed_response)
                finally:
                    self.request_queue.task_done()

            except asyncio.CancelledError:
                logger.info("LLM batch processor task cancelled.")
                break  # Exit the loop if cancelled
            except Exception as e_loop:
                logger.error(
                    f"Unexpected error in LLM batch processor loop: {e_loop}",
                    exc_info=True,
                )
                # Avoid busy-looping on persistent errors
                await asyncio.sleep(1)
        logger.info("LLM batch processor stopped.")

    def get_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics"""
        return {
            "total_requests": self.total_requests,
            "cached_responses": self.cached_responses,
            "api_errors_before_retry": self.api_errors_before_retry,
            "failed_requests_after_retries": self.failed_requests_after_retries,
            "cache_hit_rate": self.cached_responses / max(1, self.total_requests),
            "effective_failure_rate": (
                (
                    self.failed_requests_after_retries
                    / max(1, self.total_requests - self.cached_responses)
                )
                if (self.total_requests - self.cached_responses) > 0
                else 0
            ),
            "cache_size": len(self.response_cache),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the LLMCoordinator's state."""
        return {
            "response_cache": dict(
                self.response_cache
            ),  # OrderedDict to dict for simple serialization
            "total_requests": self.total_requests,
            "cached_responses": self.cached_responses,
            "api_errors_before_retry": self.api_errors_before_retry,
            "failed_requests_after_retries": self.failed_requests_after_retries,
            # Note: Throttler state, running state, and active tasks are not serialized.
            # They are re-initialized on load based on config.
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], config: Any) -> "LLMCoordinator":
        """Deserializes state into a new LLMCoordinator instance."""
        # config object must be passed in, as it's needed for __init__ (especially for model name, rate limits etc)
        coordinator = cls(config)

        coordinator.response_cache = OrderedDict(data.get("response_cache", {}))
        # Trim if loaded cache exceeds max_cache_size (oldest items first)
        while len(coordinator.response_cache) > coordinator.max_cache_size:
            coordinator.response_cache.popitem(last=False)

        coordinator.total_requests = data.get("total_requests", 0)
        coordinator.cached_responses = data.get("cached_responses", 0)
        coordinator.api_errors_before_retry = data.get("api_errors_before_retry", 0)
        coordinator.failed_requests_after_retries = data.get(
            "failed_requests_after_retries", 0
        )

        logger.info(
            f"LLMCoordinator state restored. Cache items: {len(coordinator.response_cache)} (max: {coordinator.max_cache_size}), Total Requests: {coordinator.total_requests}"
        )
        return coordinator
