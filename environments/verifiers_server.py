"""
Verifiers Environment wrapper for Atropos.

Enables Verifiers environments to be used for Atropos Evaluation, Data Generation and Training (SFT/RL).
Uses verifiers' native rollout for correct multi-turn support and token capture.
"""

import asyncio
import json
import logging
import random
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx

# Verifiers imports
import verifiers as vf
from openai import AsyncOpenAI
from pydantic import Field, Json
from verifiers.types import RolloutInput, State
from verifiers.utils.error_utils import ErrorChain
from verifiers.utils.message_utils import messages_to_printable, sanitize_tool_calls

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.type_definitions import Item

# Ensure logs are visible
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verifiers_bridge")


def _to_json_safe(obj: Any) -> Any:
    """Convert object to JSON-serializable form."""
    try:
        return json.loads(json.dumps(obj, default=str))
    except (TypeError, ValueError):
        return str(obj)


class VerifiersEnvConfig(BaseEnvConfig):
    """Configuration for VerifiersEnv."""

    vf_env_name: str = Field(
        ..., description="Verifiers environment name (e.g., 'gsm8k', 'wordle')"
    )
    env_args: Union[Dict[str, Any], Json[Dict[str, Any]]] = Field(
        default_factory=dict,
        description="Arguments passed to vf.load_environment()",
    )
    extra_env_kwargs: Union[Dict[str, Any], Json[Dict[str, Any]]] = Field(
        default_factory=dict,
        description="Runtime arguments passed to vf_env.set_kwargs()",
    )
    # Atropos config overrides
    max_token_length: int = Field(default=2048, description="Max generation tokens")
    group_size: int = Field(
        default=4, description="Number of samples to generate per prompt"
    )
    # Eval specific
    num_examples: int = Field(
        default=5, description="Number of examples to evaluate (eval mode only)"
    )
    rollouts_per_example: int = Field(
        default=1, description="Rollouts per example (eval mode only)"
    )
    eval_temperature: Optional[float] = Field(
        default=None,
        description="Temperature for generation (None = use server default)",
    )


class VerifiersEnv(BaseEnv):
    """
    Atropos environment that wraps any Verifiers Environment Hub environment.

    Uses verifiers' native rollout for correct multi-turn support.
    Supports both Evaluation (verifiers-driven) and Training/DataGen (atropos-driven).
    """

    name = "verifiers"
    env_config_cls = VerifiersEnvConfig

    def __init__(
        self,
        config: VerifiersEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=False,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: VerifiersEnvConfig = config

        # Load Verifiers Env
        self.vf_env = vf.load_environment(config.vf_env_name, **config.env_args)
        if config.extra_env_kwargs:
            self.vf_env.set_kwargs(**config.extra_env_kwargs)

        # Dataset iteration state
        self.dataset = None
        self.iter = 0

        # AsyncOpenAI client for verifiers rollout (lazy init in setup)
        self._vf_client: Optional[AsyncOpenAI] = None

        # Buffers for logging metrics to WandB
        self.reward_buffer = []
        self.metrics_buffer = defaultdict(list)

    @classmethod
    def config_init(cls) -> Tuple[VerifiersEnvConfig, List[APIServerConfig]]:
        """Default configuration for CLI usage."""
        env_config = VerifiersEnvConfig(
            vf_env_name="gsm8k",
            env_args={},
            tokenizer_name="Qwen/Qwen2.5-1.5B",
            max_token_length=2048,
            group_size=4,
            batch_size=64,
            wandb_name="verifiers-bridge",
            use_wandb=True,
        )
        server_configs = [
            APIServerConfig(
                model_name="Qwen/Qwen2.5-1.5B",
                base_url="http://localhost:8000/v1",
                api_key="x",
                num_requests_for_eval=128,
            ),
        ]
        return env_config, server_configs

    async def setup(self):
        """Initialize dataset iterator and AsyncOpenAI client for verifiers."""
        if self.dataset is None:
            self.dataset = self.vf_env.get_dataset().shuffle(
                seed=random.randint(0, 10000)
            )
            self.iter = 0
            logger.info(f"Loaded dataset with {len(self.dataset)} examples")

        # Create AsyncOpenAI client for verifiers rollout
        if self._vf_client is None and self.server and self.server.servers:
            server_config = self.server.servers[0].config
            limits = httpx.Limits(max_connections=1000, max_keepalive_connections=100)
            http_client = httpx.AsyncClient(
                limits=limits, timeout=server_config.timeout
            )
            self._vf_client = AsyncOpenAI(
                base_url=server_config.base_url,
                api_key=server_config.api_key or "x",
                http_client=http_client,
            )
            logger.info(
                f"Created AsyncOpenAI client for verifiers: {server_config.base_url}"
            )

    async def get_next_item(self) -> Item:
        """Get next item for Atropos data generation loop."""
        if self.dataset is None:
            await self.setup()

        item = self.dataset[self.iter % len(self.dataset)]
        self.iter += 1
        return item

    def _item_to_rollout_input(self, item: Item, example_id: int = 0) -> RolloutInput:
        """Convert Atropos Item to verifiers RolloutInput."""
        return RolloutInput(
            prompt=item.get("prompt", []),
            example_id=example_id,
            task=item.get("task", "default"),
            answer=item.get("answer", ""),
            info=item.get("info", {}),
        )

    def _states_to_scored_data(self, states: List[State]) -> Optional[ScoredDataGroup]:
        """
        Convert verifiers States to Atropos ScoredDataGroup.

        Handles both:
        - vLLM backend: native tokens from trajectory
        - OpenAI backend: post-hoc tokenization with chat template
        """
        if not states:
            return None

        scored_data: ScoredDataGroup = {
            "tokens": [],
            "masks": [],
            "scores": [],
            "messages": [],
            "inference_logprobs": [],
            "overrides": [],
            "group_overrides": {},
            "advantages": None,
            "ref_logprobs": None,
            "generation_params": None,
            "images": None,
        }

        for state in states:
            # Skip failed rollouts
            if state.get("error") is not None:
                logger.warning(f"Skipping failed rollout: {ErrorChain(state['error'])}")
                continue

            # Check if vLLM provided native tokens
            trajectory = state.get("trajectory", [])
            has_native_tokens = len(trajectory) > 0 and all(
                step.get("tokens") is not None for step in trajectory
            )

            if has_native_tokens:
                # vLLM case: stitch native tokens from trajectory
                full_ids, full_mask, full_logprobs = self._stitch_trajectory_tokens(
                    state
                )
            else:
                # OpenAI case: post-hoc tokenize with chat template
                full_ids, full_mask, full_logprobs = self._tokenize_from_messages(state)

            # Build full messages for logging
            prompt = state.get("prompt", [])
            completion = state.get("completion", [])
            full_messages = (
                prompt + completion if isinstance(completion, list) else prompt
            )

            scored_data["tokens"].append(full_ids)
            scored_data["masks"].append(full_mask)

            reward = state.get("reward", 0.0)

            # Capture metrics for WandB logging
            self.reward_buffer.append(reward)

            # Verifiers states often contain a breakdown of metrics (e.g. strict accuracy vs partial)
            state_metrics = state.get("metrics", {})
            if state_metrics:
                for k, v in state_metrics.items():
                    if isinstance(v, (int, float)):
                        self.metrics_buffer[k].append(v)

            scored_data["scores"].append(reward)
            scored_data["messages"].append(full_messages)
            scored_data["inference_logprobs"].append(full_logprobs)
            scored_data["overrides"].append({})

        # Return None if all rollouts failed
        if not scored_data["tokens"]:
            return None

        return scored_data

    def _stitch_trajectory_tokens(
        self, state: State
    ) -> Tuple[List[int], List[int], List[float]]:
        """
        Stitch tokens from all trajectory steps (vLLM case).

        For multi-turn: concatenates all steps, masking non-assistant tokens.
        """
        trajectory = state.get("trajectory", [])
        if not trajectory:
            return [], [], []

        full_ids = []
        full_mask = []
        full_logprobs = []

        # Track the length of the sequence we have built so far to find the "diff"
        # in the prompt tokens for subsequent turns.
        current_len = 0

        for i, step in enumerate(trajectory):
            tokens_data = step.get("tokens")
            if not tokens_data:
                # If any step is missing tokens, abort this vLLM-specific stitching
                # and fall back to the generic tokenizer method.
                return [], [], []

            p_ids = tokens_data.get("prompt_ids", [])
            c_ids = tokens_data.get("completion_ids", [])
            c_logprobs = tokens_data.get("completion_logprobs", [])

            # --- 1. Handle Prompt (User/Env) ---
            # For step 0, take the whole prompt.
            # For step > 0, p_ids includes the entire history.
            new_prompt_ids = p_ids[current_len:]

            full_ids.extend(new_prompt_ids)
            full_mask.extend([-100] * len(new_prompt_ids))  # Always mask prompt/env
            full_logprobs.extend([0.0] * len(new_prompt_ids))

            # --- 2. Handle Completion (Assistant) ---
            full_ids.extend(c_ids)
            full_mask.extend(c_ids)  # Unmasked: use IDs as labels
            full_logprobs.extend(c_logprobs)

            # Update current length for next iteration
            current_len = len(p_ids) + len(c_ids)

        return full_ids, full_mask, full_logprobs

    def _tokenize_from_messages(
        self, state: State
    ) -> Tuple[List[int], List[int], List[float]]:
        """
        Post-hoc tokenize from message content (OpenAI case).

        Uses chat template to ensure proper formatting.
        """
        prompt = state.get("prompt", [])
        completion = state.get("completion", [])

        if not prompt:
            return [], [], []

        # Full conversation
        full_messages = prompt + completion if isinstance(completion, list) else prompt

        # Tokenize full conversation with chat template
        try:
            full_ids = self.tokenizer.apply_chat_template(
                full_messages,
                tokenize=True,
                add_generation_prompt=False,
            )
        except Exception as e:
            logger.error(f"Failed to tokenize messages: {e}")
            return [], [], []

        # Tokenize just the prompt to find the boundary
        try:
            prompt_ids = self.tokenizer.apply_chat_template(
                prompt,
                tokenize=True,
                add_generation_prompt=True,  # Include generation prompt marker
            )
        except Exception as e:
            logger.error(f"Failed to tokenize prompt: {e}")
            prompt_ids = []

        prompt_len = len(prompt_ids)

        # Build mask: -100 for prompt, token IDs for completion
        if prompt_len < len(full_ids):
            full_mask = [-100] * prompt_len + list(full_ids[prompt_len:])
        else:
            # Edge case: completion is empty
            full_mask = [-100] * len(full_ids)

        # Dummy logprobs for OpenAI (we don't have them)
        full_logprobs = [0.0] * len(full_ids)

        return list(full_ids), full_mask, full_logprobs

    async def collect_trajectories(
        self, item: Item
    ) -> Tuple[Optional[ScoredDataGroup], List[Item]]:
        """
        Generate and score trajectories using verifiers' native rollout.

        Delegates to vf_env.run_group which handles:
        - Multi-turn loops
        - Stop conditions
        - State tracking
        - Scoring
        """
        if self._vf_client is None:
            await self.setup()

        if self._vf_client is None:
            logger.error("Failed to initialize AsyncOpenAI client")
            return None, []

        # Prepare inputs for verifiers (one per group member)
        rollout_inputs = [
            self._item_to_rollout_input(item, example_id=i)
            for i in range(self.config.group_size)
        ]

        # Sampling args for generation
        sampling_args = {
            "max_tokens": self.config.max_token_length,
            "logprobs": True,
            "extra_body": {"return_token_ids": True},
        }
        if self.config.eval_temperature is not None:
            sampling_args["temperature"] = self.config.eval_temperature
        else:
            sampling_args["temperature"] = 1.0  # Default for training

        # Get model name
        model_name = "default"
        if self.server and self.server.servers:
            model_name = self.server.servers[0].config.model_name

        try:
            # Delegate to verifiers' native run_group
            gen_sem = asyncio.Semaphore(self.config.group_size)
            score_sem = asyncio.Semaphore(100)

            states = await self.vf_env.run_group(
                group_inputs=rollout_inputs,
                client=self._vf_client,
                model=model_name,
                gen_sampling_args=sampling_args,
                gen_sem=gen_sem,
                score_sem=score_sem,
                score=True,
            )
        except Exception as e:
            logger.error(f"Error during verifiers rollout: {e}")
            return None, []

        # Convert states to ScoredDataGroup
        scored_data = self._states_to_scored_data(list(states))
        return scored_data, []

    async def evaluate(self, *args, **kwargs):
        """
        Evaluate using verifiers' built-in evaluation logic.
        """
        start_time = time.time()
        server_config = self.server.servers[0].config

        limits = httpx.Limits(max_connections=28000, max_keepalive_connections=28000)
        http_client = httpx.AsyncClient(limits=limits, timeout=server_config.timeout)

        # Create client for verifiers evaluation
        client = AsyncOpenAI(
            base_url=server_config.base_url,
            api_key=server_config.api_key or "x",
            http_client=http_client,
            max_retries=10,
        )

        sampling_args = {"max_tokens": self.config.max_token_length}
        if self.config.eval_temperature is not None:
            sampling_args["temperature"] = self.config.eval_temperature

        results = await self.vf_env.evaluate(
            client=client,
            model=server_config.model_name,
            sampling_args=sampling_args,
            num_examples=self.config.num_examples,
            rollouts_per_example=self.config.rollouts_per_example,
            max_concurrent=self.config.max_eval_workers,
        )

        end_time = time.time()
        avg_reward = results["metadata"]["avg_reward"]
        avg_metrics = results["metadata"]["avg_metrics"]

        samples = []
        for state in results["state"]:
            clean_prompt = sanitize_tool_calls(
                messages_to_printable(state.get("prompt"))
            )
            clean_completion = sanitize_tool_calls(
                messages_to_printable(state.get("completion"))
            )
            sample = {
                "prompt": _to_json_safe(clean_prompt),
                "completion": _to_json_safe(clean_completion),
                "reward": state.get("reward"),
                "answer": state.get("answer"),
                "stop_condition": state.get("stop_condition"),
            }
            if state.get("error"):
                sample["error"] = str(ErrorChain(state["error"]))

            samples.append(sample)

        await self.evaluate_log(
            metrics={
                "eval/avg_reward": avg_reward,
                **{f"eval/{k}": v for k, v in avg_metrics.items()},
            },
            samples=samples,
            start_time=start_time,
            end_time=end_time,
            generation_parameters=sampling_args,
        )

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """
        Override base logger to include Verifiers-specific metrics.
        """
        if wandb_metrics is None:
            wandb_metrics = {}

        # Log average reward
        if self.reward_buffer:
            avg_reward = sum(self.reward_buffer) / len(self.reward_buffer)
            wandb_metrics["metrics/mean_reward"] = avg_reward
            self.reward_buffer = []

        # Log custom metrics from the Verifiers environment (e.g. accuracy, format_score)
        if self.metrics_buffer:
            for metric_name, values in self.metrics_buffer.items():
                if values:
                    avg_metric = sum(values) / len(values)
                    # Prefix with train/ to keep dashboard clean
                    wandb_metrics[f"metrics/{metric_name}"] = avg_metric
            self.metrics_buffer = defaultdict(list)

        # Call parent to handle server metrics, perf stats, and the rollouts table
        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    VerifiersEnv.cli()
