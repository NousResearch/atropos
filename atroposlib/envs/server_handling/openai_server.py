import asyncio
import collections
import time
from asyncio import exceptions
from typing import Optional

import aiohttp
import numpy as np
import openai
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.completion import Completion
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_random_exponential

from atroposlib.envs.server_handling.server_baseline import APIServerConfig


class AsyncSemWithAdaptiveWeight(asyncio.Semaphore):
    def __init__(self, value: int):
        super().__init__(value=value)
        self.max_val = value
        self.weight = 1.0

    def update_weight(self, weight: float) -> None:
        self.weight = weight

    def min_val(self):
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


class OpenAIServer:
    def __init__(self, config: APIServerConfig):
        self.config = config
        self.openai = openai.AsyncClient(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
        )
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

    async def update_weight(self, weight: float) -> None:
        # need to update sems
        self.sem.update_weight(weight)
        self.eval_sem.update_weight(weight)

    async def check_server_status_task(self):
        while True:
            try:
                await self.openai.completions.create(
                    model=self.config.model_name,
                    prompt="hi",
                    max_tokens=1,
                )
                self.server_healthy = True
            except (
                aiohttp.ClientError,
                openai.OpenAIError,
                openai.APITimeoutError,
                Exception,
            ):
                self.server_healthy = False
            await asyncio.sleep(1)

    async def wandb_metrics(
        self, metrics_dict: Optional[dict], server_name: Optional[str]
    ):
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

    @retry(
        stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=10)
    )
    async def _chat_comp(self, stat_dict, **kwargs) -> ChatCompletion:
        while not self.server_healthy:
            await asyncio.sleep(1)
        async with self.sem:
            if stat_dict.get("start", None) is None:
                stat_dict["start"] = time.time()
            stat_dict["attempts"] += 1
            completions = await self.openai.chat.completions.create(**kwargs)
            stat_dict["end"] = time.time()
            return completions

    @retry(
        stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=10)
    )
    async def _chat_eval(self, stat_dict, **kwargs) -> ChatCompletion:
        while not self.server_healthy:
            await asyncio.sleep(1)
        async with self.eval_sem:
            if stat_dict.get("start", None) is None:
                stat_dict["start"] = time.time()
            stat_dict["attempts"] += 1
            completions = await self.openai.chat.completions.create(**kwargs)
            stat_dict["end"] = time.time()
            return completions

    @retry(
        stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=10)
    )
    async def chat_completion(self, **kwargs) -> ChatCompletion:
        if not self.initialized:
            if (
                self.config.base_url is not None
            ):  # skip health check if using OpenAI API
                self.check_task = asyncio.create_task(self.check_server_status_task())
            else:
                self.server_healthy = True
            self.initialized = True
        kwargs["model"] = self.config.model_name
        split = kwargs.pop("split", "train")
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
        while not self.server_healthy:
            await asyncio.sleep(1)
        async with self.sem:
            if stat_dict.get("start", None) is None:
                stat_dict["start"] = time.time()
            stat_dict["attempts"] += 1
            completions = await self.openai.completions.create(**kwargs)
            stat_dict["end"] = time.time()
            return completions

    @retry(
        stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=10)
    )
    async def _comp_eval(self, stat_dict, **kwargs) -> Completion:
        while not self.server_healthy:
            await asyncio.sleep(1)
        async with self.eval_sem:
            if stat_dict.get("start", None) is None:
                stat_dict["start"] = time.time()
            stat_dict["attempts"] += 1
            completions = await self.openai.completions.create(**kwargs)
            stat_dict["end"] = time.time()
            return completions

    async def completion(self, **kwargs) -> Completion:
        if not self.initialized:
            if (
                self.config.base_url is not None
            ):  # skip health check if using OpenAI API
                self.check_task = asyncio.create_task(self.check_server_status_task())
            else:
                self.server_healthy = True
            self.initialized = True
        kwargs["model"] = self.config.model_name
        split = kwargs.pop("split", "train")
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
