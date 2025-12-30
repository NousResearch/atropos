import asyncio
import inspect
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.completion import Completion
from pydantic import BaseModel, Field

from atroposlib.envs.server_handling.managed_server import ManagedServer
from atroposlib.envs.server_handling.openai_server import OpenAIServer
from atroposlib.envs.server_handling.server_baseline import (
    APIServer,
    APIServerConfig,
    ReasoningConfig,
    ServerBaseline,
)
from atroposlib.envs.server_handling.server_harness import ServerHarness
from atroposlib.envs.server_handling.sglang_server import SGLangServer
from atroposlib.envs.server_handling.trl_vllm_server import TrlVllmServer
from atroposlib.envs.server_handling.vllm_server import VLLMServer


class ServerManagerConfig(BaseModel):
    slurm: bool = Field(
        default=False, description="Whether environment is running on slurm or not."
    )
    testing: bool = Field(
        default=False, description="If set to True, environment uses mock OpenAI data."
    )
    max_n_completions: int = Field(
        default=8,
        description=(
            "The maximum number of completions to request at once per server call. "
            "Will split any n larger than this into multiple calls. "
            "This is to help load balance servers."
        ),
    )


class ServerManager:
    def __init__(
        self,
        configs: Union[ServerBaseline, List[APIServerConfig]],
        server_class: APIServer = APIServer,
        slurm=False,
        testing=False,
        max_n_completions=8,
        reasoning_config: Optional[ReasoningConfig] = None,
    ):
        self.max_n_completions = max_n_completions
        self.reasoning_config = reasoning_config
        # First we check to see if it's the base server class, and if so, we need to select the appropriate server class
        # You can't use type() to check if it's the base server class, because it's an abstract class, it'll appear as
        # an ABCMeta, not what you're expecting.
        if inspect.isabstract(server_class):
            if not isinstance(configs, list):
                if configs.server_type == "openai":
                    server_class = OpenAIServer
                elif configs.server_type == "trl":
                    server_class = TrlVllmServer
                elif configs.server_type == "sglang":
                    server_class = SGLangServer
                elif configs.server_type == "vllm":
                    server_class = VLLMServer
                else:
                    raise ValueError(f"Invalid server type: {configs.server_type}")
            else:
                if configs[0].server_type == "openai":
                    server_class = OpenAIServer
                elif configs[0].server_type == "trl":
                    server_class = TrlVllmServer
                elif configs[0].server_type == "sglang":
                    server_class = SGLangServer
                elif configs[0].server_type == "vllm":
                    server_class = VLLMServer
                else:
                    raise ValueError(f"Invalid server type: {configs[0].server_type}")
        if testing:
            # testing :)
            self.servers = [ServerHarness()]
            return
        if not isinstance(configs, list):
            urls = []
            if os.environ.get("SLURM_JOB_NODELIST", None) is not None:
                nodelist = (
                    os.popen(
                        f'scontrol show hostnames {os.environ["SLURM_JOB_NODELIST"]}'
                    )
                    .read()
                    .split("\n")
                )
                nodelist = [node for node in nodelist if node != ""]
                if len(nodelist) < 2:
                    # localhost!
                    for i in range(4):
                        urls.append(f"http://localhost:{9000 + i + 4}/v1")
                else:
                    num_training_nodes = int(os.environ.get("NUM_TRAINING_NODES"))
                    for node in nodelist[num_training_nodes:]:
                        for i in range(8 // os.environ.get("INFER_TP", 1)):
                            urls.append(f"http://{node}:{9000 + i}/v1")
                openai_configs = []
            else:
                # localhost!
                for i in range(4):
                    urls.append(f"http://localhost:{9000 + i + 4}/v1")
                openai_configs = []
            for url in urls:
                openai_configs.append(
                    APIServerConfig(
                        base_url=url,
                        timeout=configs.timeout,
                        num_max_requests_at_once=configs.num_max_requests_at_once,
                        num_requests_for_eval=configs.num_requests_for_eval,
                        model_name=configs.model_name,
                        rolling_buffer_length=configs.rolling_buffer_length,
                        api_key="x",
                    )
                )
            self.servers = [server_class(config) for config in openai_configs]
        elif not slurm:
            self.servers = [server_class(config) for config in configs]
        else:
            nodelist = (
                os.popen(f'scontrol show hostnames {os.environ["SLURM_JOB_NODELIST"]}')
                .read()
                .split("\n")
            )
            nodelist = [node for node in nodelist if node != ""]
            if len(nodelist) < 2:
                print(
                    "Not enough nodes to distribute to, assuming single node"
                    " and you've setup your sglang appropriately."
                )
                self.servers = [server_class(config) for config in configs]
                return
            urls = []
            num_training_nodes = int(os.environ.get("NUM_TRAINING_NODES"))
            for node in nodelist[num_training_nodes:]:
                if node == "":
                    continue
                for i in range(8 // os.environ.get("INFER_TP", 1)):
                    urls.append(f"http://{node}:{9000 + i}/v1")
            # assume at least one good config is passed in
            new_configs = []
            for i in range(len(urls)):
                new_conf = configs[0].model_copy(deep=True)
                new_conf.base_url = urls[i]
                new_configs.append(new_conf)
            self.servers = [server_class(config) for config in new_configs]

    async def update_weight(self, weight: float):
        for server in self.servers:
            await server.update_weight(weight)

    def _get_server_base_url(self, server_idx: int = 0) -> Optional[str]:
        """Get the base_url from a server's config."""
        if not self.servers:
            return None
        server = self.servers[server_idx]
        if hasattr(server, "config") and hasattr(server.config, "base_url"):
            return server.config.base_url
        return None

    def _inject_reasoning_extra_body(
        self, kwargs: Dict[str, Any], server_idx: int = 0
    ) -> Dict[str, Any]:
        """
        Inject reasoning extra_body into kwargs if reasoning is configured.
        
        This method handles the differences between OpenAI and other providers:
        - OpenAI: Uses {"reasoning_effort": "..."} at top level, requires temperature=1.0,
                  and uses max_completion_tokens instead of max_tokens
        - Others: Uses {"reasoning": {"enabled": True, "effort": "...", "max_tokens": ...}}
        
        Args:
            kwargs: The kwargs dict to modify
            server_idx: Index of the server to use for base_url detection
            
        Returns:
            Modified kwargs dict with extra_body injected if reasoning is active
        """
        if self.reasoning_config is None or not self.reasoning_config.is_active():
            return kwargs
        
        # Get the base_url to determine provider type
        base_url = self._get_server_base_url(server_idx)
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
            # Override any temperature setting
            kwargs["temperature"] = 1.0
            
            # OpenAI reasoning models use max_completion_tokens instead of max_tokens
            # Convert if max_tokens is set
            if "max_tokens" in kwargs and kwargs["max_tokens"]:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
        
        return kwargs

    async def wait_for_sem(self, is_training: bool):
        """
        Wait for a server to be available. This is used to prevent the client from
        overwhelming the server with requests.
        """

        def get_available_slots():
            if is_training:
                eval_vals = [
                    (
                        max(0, server.eval_sem._value - server.eval_sem.min_val())
                        if server.eval_sem._value != server.eval_sem.max_val
                        else 0
                    )
                    for server in self.servers
                ]
                return [
                    max(0, (server.sem._value - server.sem.min_val()) - eval_val)
                    for server, eval_val in zip(self.servers, eval_vals)
                ]
            else:
                return [
                    max(0, server.eval_sem._value - server.eval_sem.min_val())
                    for server in self.servers
                ]

        sem_vals = get_available_slots()
        while all(sem_val <= 0 for sem_val in sem_vals):
            # None available... wait
            await asyncio.sleep(1)
            sem_vals = get_available_slots()

    async def chat_completion(self, **kwargs) -> ChatCompletion:
        """
        Route chat completion to the most available server.
        
        Automatically injects reasoning extra_body if reasoning_config is active.
        """
        n = kwargs.get("n", 1)
        if n > self.max_n_completions:
            # Split into multiple completions
            completions = []
            total_n = n
            while total_n > 0:
                n_to_use = min(total_n, self.max_n_completions)
                kwargs["n"] = n_to_use
                completions.append(self.chat_completion(**kwargs))
                total_n -= n_to_use
            completions = await asyncio.gather(
                *completions
            )  # type: List[ChatCompletion]
            # merge choices into one
            out = completions[0]
            for completion in completions[1:]:
                out.choices.extend(completion.choices)
            return out
        is_train = kwargs.pop("split", "train") == "train"
        most_available_server = 0
        most_available_server_num_slots = -1
        await self.wait_for_sem(is_train)
        for i, server in enumerate(self.servers):
            if not server.server_healthy:
                continue
            if (
                server.sem._value if is_train else server.eval_sem._value
            ) > most_available_server_num_slots:
                most_available_server = i
                most_available_server_num_slots = (
                    server.sem._value if is_train else server.eval_sem._value
                )
        
        # Inject reasoning extra_body if configured
        kwargs = self._inject_reasoning_extra_body(kwargs, most_available_server)
        
        return await self.servers[most_available_server].chat_completion(**kwargs)

    async def completion(self, **kwargs) -> Completion:
        """
        Route completion to the most available server.
        
        Automatically injects reasoning extra_body if reasoning_config is active.
        """
        n = kwargs.get("n", 1)
        if n > self.max_n_completions:
            # Split into multiple completions
            completions = []
            total_n = n
            while total_n > 0:
                n_to_use = min(total_n, self.max_n_completions)
                kwargs["n"] = n_to_use
                completions.append(self.completion(**kwargs))
                total_n -= n_to_use
            completions = await asyncio.gather(*completions)  # type: List[Completion]
            # merge choices into one
            out = completions[0]
            for completion in completions[1:]:
                out.choices.extend(completion.choices)
            return out
        is_train = kwargs.pop("split", "train") == "train"
        most_available_server = 0
        most_available_server_num_slots = -1
        await self.wait_for_sem(is_train)
        for i, server in enumerate(self.servers):
            if not server.server_healthy:
                continue
            if (
                server.sem._value if is_train else server.eval_sem._value
            ) > most_available_server_num_slots:
                most_available_server = i
                most_available_server_num_slots = (
                    server.sem._value if is_train else server.eval_sem._value
                )
        
        # Inject reasoning extra_body if configured
        kwargs = self._inject_reasoning_extra_body(kwargs, most_available_server)
        
        return await self.servers[most_available_server].completion(**kwargs)

    async def tokens_and_logprobs_completion(
        self, **kwargs
    ) -> tuple[list, list, list, list]:
        """
        Get tokens and logprobs from completion.
        Returns (prompt_tokens, output_tokens, output_logprobs, finish_reasons).
        
        Automatically injects reasoning extra_body if reasoning_config is active.
        """
        n = kwargs.get("n", 1)
        if n > self.max_n_completions:
            # Split into multiple completions
            results = []
            total_n = n
            while total_n > 0:
                n_to_use = min(total_n, self.max_n_completions)
                kwargs["n"] = n_to_use
                results.append(self.tokens_and_logprobs_completion(**kwargs))
                total_n -= n_to_use
            results = await asyncio.gather(*results)
            # Merge results - prompt_tokens should be same, extend output lists
            prompt_tokens = results[0][0]
            output_tokens = []
            output_logprobs = []
            finish_reasons = []
            for _, out_tokens, out_logprobs, out_finish_reasons in results:
                output_tokens.extend(out_tokens)
                output_logprobs.extend(out_logprobs)
                finish_reasons.extend(out_finish_reasons)
            return (prompt_tokens, output_tokens, output_logprobs, finish_reasons)

        is_train = kwargs.pop("split", "train") == "train"
        most_available_server = 0
        most_available_server_num_slots = -1
        await self.wait_for_sem(is_train)
        for i, server in enumerate(self.servers):
            if not server.server_healthy:
                continue
            if (
                server.sem._value if is_train else server.eval_sem._value
            ) > most_available_server_num_slots:
                most_available_server = i
                most_available_server_num_slots = (
                    server.sem._value if is_train else server.eval_sem._value
                )
        
        # Inject reasoning extra_body if configured
        kwargs = self._inject_reasoning_extra_body(kwargs, most_available_server)
        
        return await self.servers[most_available_server].tokens_and_logprobs_completion(
            **kwargs
        )

    @asynccontextmanager
    async def dedicated_server(self) -> AsyncGenerator[OpenAIServer, None]:
        most_available_server = 0
        most_available_server_num_slots = -1
        for i, server in enumerate(self.servers):
            if not server.server_healthy:
                continue
            if server.sem._value > most_available_server_num_slots:
                most_available_server = i
                most_available_server_num_slots = server.sem._value
        async with self.servers[most_available_server].sem:
            try:
                yield self.servers[most_available_server]
            finally:
                pass

    @asynccontextmanager
    async def managed_server(
        self, tokenizer=None
    ) -> AsyncGenerator[ManagedServer, None]:
        """
        Context manager that provides a ManagedServer instance.

        The ManagedServer wraps the most available server and tracks text sequences
        with aligned tokens and logprobs. State is automatically cleared on exit.

        Args:
            tokenizer: Optional tokenizer to use. If not provided, will attempt to
                      extract from server or create from model name.

        Yields:
            ManagedServer instance wrapping the selected server

        Example:
            async with server_manager.managed_server() as managed:
                response = await managed.chat_completion(
                    messages=[{"role": "user", "content": "Hello"}],
                    n=2
                )
                state = managed.get_state()
                # Process state...
                # State is automatically cleared when exiting context
        """
        most_available_server = 0
        most_available_server_num_slots = -1
        for i, server in enumerate(self.servers):
            if not server.server_healthy:
                continue
            if server.sem._value > most_available_server_num_slots:
                most_available_server = i
                most_available_server_num_slots = server.sem._value

        # Create ManagedServer wrapping the selected server
        managed = ManagedServer(
            server=self.servers[most_available_server], tokenizer=tokenizer
        )

        try:
            yield managed
        finally:
            # Clean up: reset tracked sequences
            managed.reset()
