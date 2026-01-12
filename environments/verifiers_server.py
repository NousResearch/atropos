"""
Verifiers Training Environment for Atropos

Supports TWO modes:
- serve: RL training with local inference server (requires ManagedServer for logprobs)
- process: SFT data generation with ANY API (OpenAI, Claude, local, etc.)

Usage:
  # RL Training (requires local vLLM/SGLang server)
  python verifiers_server.py serve \
      --env.vf_env_name "will/wordle" \
      --openai.base_url http://localhost:9001/v1 \
      --slurm false

  # SFT Data Generation with OpenAI GPT-4o
  python verifiers_server.py process \
      --env.vf_env_name "will/wordle" \
      --env.data_path_to_save_groups gpt4o_sft_data.jsonl \
      --env.total_steps 100 \
      --env.group_size 4 \
      --openai.model_name gpt-4o \
      --openai.base_url https://api.openai.com/v1

  # SFT Data Generation with local server
  python verifiers_server.py process \
      --env.vf_env_name "will/wordle" \
      --env.data_path_to_save_groups local_sft_data.jsonl \
      --openai.base_url http://localhost:9001/v1

  # Evaluation (uses ManagedServer by default, falls back to direct API in process mode)
  python verifiers_server.py evaluate \
      --env.vf_env_name "will/wordle" \
      --openai.base_url http://localhost:9001/v1

To install a Verifiers/Prime environment:
1. uv tool install prime
2. prime login
3. prime env install will/wordle (or any owner/environment)
Docs: https://docs.primeintellect.ai/tutorials-environments/install
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

import verifiers as vf
from openai import AsyncOpenAI
from pydantic import Field

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

logger = logging.getLogger(__name__)


class VfEnvConfig(BaseEnvConfig):
    vf_env_name: str = ""
    env_args: Dict[str, Any] = Field(default_factory=dict)


class VerifiersEnv(BaseEnv):
    name = "verifiers"
    env_config_cls = VfEnvConfig  # type: ignore[assignment]

    def __init__(
        self,
        config: VfEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=False,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.percent_correct_buffer: List[float] = []

        logger.info("Loading verifiers environment: %s", config.vf_env_name)
        self.vf_env = vf.load_environment(config.vf_env_name, **config.env_args)
        self.rubric = self.vf_env.rubric
        self.system_prompt = self.vf_env.system_prompt
        logger.info("Reward functions: %s", self.rubric._get_reward_func_names())

    @classmethod
    def config_init(cls) -> Tuple[VfEnvConfig, List[APIServerConfig]]:
        env_config = VfEnvConfig(
            tokenizer_name="Qwen/Qwen2.5-1.5B-Instruct",
            group_size=8,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=4,
            steps_per_eval=100,
            max_token_length=2048,
            wandb_name="verifiers",
        )
        server_configs = [
            APIServerConfig(
                model_name="gpt-4.1-nano",
                base_url="https://api.openai.com/v1",
                api_key="x",
                num_requests_for_eval=4,
            ),
        ]
        return env_config, server_configs

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}

        if self.percent_correct_buffer:
            wandb_metrics["train/percent_correct"] = sum(
                self.percent_correct_buffer
            ) / len(self.percent_correct_buffer)
            self.percent_correct_buffer = []

        await super().wandb_log(wandb_metrics)

    async def setup(self):
        train_data = self.vf_env.get_dataset()
        columns_to_keep = ["question", "answer", "info"]
        available_columns = [c for c in columns_to_keep if c in train_data.column_names]
        self.train = train_data.select_columns(available_columns).to_list()
        self.iter = 0

    def save_checkpoint(self, step, data=None):
        if data is None:
            data = {}
        data["iter"] = self.iter
        super().save_checkpoint(step, data)

    async def get_next_item(self):
        next_item = self.train[self.iter % len(self.train)]
        self.iter += 1
        return next_item

    async def evaluate(self) -> Dict[str, float]:
        """No-op. Use environments/eval_environments/verifiers_eval.py for evaluation."""
        return {}

    def _build_initial_messages(self, question: str) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": question})
        return messages

    async def collect_trajectories(self, item) -> Tuple[ScoredDataGroup, list]:
        is_process_mode = getattr(self, "process_mode", False)
        if is_process_mode:
            return await self._collect_for_sft(item)
        return await self._collect_for_rl(item)

    async def _collect_for_sft(
        self, item: Dict[str, Any]
    ) -> Tuple[ScoredDataGroup, list]:
        """SFT mode: uses vf_env.generate() for parallel generation + scoring."""
        server_config = self.server.servers[0].config
        client = AsyncOpenAI(
            api_key=server_config.api_key,
            base_url=server_config.base_url,
            timeout=server_config.timeout,
        )

        initial_messages = self._build_initial_messages(item["question"])
        inputs = [
            {
                "prompt": initial_messages,
                "answer": item["answer"],
                "example_id": i,
                "task": self.config.vf_env_name,
                "info": item.get("info", {}),
            }
            for i in range(self.config.group_size)
        ]

        results = await self.vf_env.generate(
            inputs=inputs,
            client=client,
            model=server_config.model_name,
            sampling_args={
                "temperature": 1.0,
                "max_completion_tokens": self.config.max_token_length,
            },
            max_concurrent=self.config.group_size,
            max_concurrent_scoring=self.config.group_size,
            save_results=False,
            independent_scoring=True,
        )

        scored_data = ScoredDataGroup()
        scored_data["tokens"] = []
        scored_data["masks"] = []
        scored_data["scores"] = []
        scored_data["messages"] = []

        for state in results["state"]:
            messages = list(state.get("prompt", [])) + list(state.get("completion", []))
            messages = [
                {**msg, "content": msg.get("content") or ""} for msg in messages
            ]

            trajectory = state.get("trajectory", [])
            finish_reason = (
                trajectory[-1]["response"].choices[0].finish_reason
                if trajectory
                else "stop"
            )

            tokenized = tokenize_for_trainer(
                tokenizer=self.tokenizer,
                chat=messages,
                include_messages=True,
                finish_reason=finish_reason,
                train_on_all_assistant_turns=True,
            )

            scored_data["tokens"].append(tokenized["tokens"])
            scored_data["masks"].append(tokenized["masks"])
            scored_data["messages"].append(messages)
            scored_data["scores"].append(state.get("reward", 0.0))

        for score in scored_data["scores"]:
            self.percent_correct_buffer.append(max(score, 0))

        return scored_data, []

    async def _collect_for_rl(
        self, item: Dict[str, Any]
    ) -> Tuple[ScoredDataGroup, list]:
        """RL mode: uses ManagedServer for logprobs tracking."""
        from atroposlib.envs.server_handling.atropos_managed_client import (
            AtroposManagedClient,
        )

        initial_messages = self._build_initial_messages(item["question"])
        sampling_args = {
            "temperature": 1.0,
            "max_completion_tokens": self.config.max_token_length,
        }
        score_sem = asyncio.Semaphore(self.config.group_size)
        model = self.server.servers[0].config.model_name

        async def run_rollout(
            example_id: int,
        ) -> Tuple[List[int], List[int], List[float], float]:
            async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
                client = AtroposManagedClient(managed_server=managed, model=model)

                state = await self.vf_env.rollout(
                    input={
                        "prompt": initial_messages,
                        "answer": item["answer"],
                        "example_id": example_id,
                        "task": self.config.vf_env_name,
                        "info": item.get("info", {}),
                    },
                    client=client,
                    model=model,
                    sampling_args=sampling_args,
                )
                await self.rubric.score_rollout(state, score_sem=score_sem)
                return self._extract_from_state(state)

        results = await asyncio.gather(
            *[run_rollout(i) for i in range(self.config.group_size)]
        )

        scored_data = ScoredDataGroup()
        scored_data["tokens"] = []
        scored_data["masks"] = []
        scored_data["scores"] = []
        scored_data["inference_logprobs"] = []

        for tokens, masks, logprobs, score in results:
            scored_data["tokens"].append(tokens)
            scored_data["masks"].append(masks)
            scored_data["inference_logprobs"].append(logprobs)
            scored_data["scores"].append(score)

        for score in scored_data["scores"]:
            self.percent_correct_buffer.append(max(score, 0))

        return scored_data, []

    def _extract_from_state(
        self, state: Any
    ) -> Tuple[List[int], List[int], List[float], float]:
        """Extract tokens/masks/logprobs from rollout state (RL mode only)."""
        all_tokens: List[int] = []
        all_masks: List[int] = []
        all_logprobs: List[float] = []

        for step in state.get("trajectory", []):
            tokens = step["tokens"]
            prompt_ids = tokens["prompt_ids"]
            completion_ids = tokens["completion_ids"]
            completion_logprobs = tokens["completion_logprobs"]

            all_tokens.extend(prompt_ids)
            all_tokens.extend(completion_ids)
            all_masks.extend([-100] * len(prompt_ids))
            all_masks.extend(completion_ids)
            all_logprobs.extend([1.0] * len(prompt_ids))
            all_logprobs.extend(completion_logprobs)

        return all_tokens, all_masks, all_logprobs, state["reward"]


if __name__ == "__main__":
    VerifiersEnv.cli()
