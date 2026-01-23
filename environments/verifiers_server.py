"""
Verifiers Training Environment for Atropos

Unified environment that works for both RL training (serve) and SFT data generation (process).
Uses vf_env.generate() with ManagedServer (via adapter) for automatic token and logprob tracking.

Usage:
  # RL Training (GRPO - no inference logprobs needed)
  python verifiers_server.py serve \
      --env.vf_env_name "primeintellect/alphabet-sort" \
      --openai.base_url http://localhost:9001/v1 \
      --slurm false

  # SFT Data Generation with OpenAI GPT-4o
  python verifiers_server.py process \
      --env.vf_env_name "primeintellect/alphabet-sort" \
      --env.data_path_to_save_groups gpt4o_sft_data.jsonl \
      --env.total_steps 100 \
      --env.group_size 4 \
      --openai.model_name gpt-4o \
      --openai.base_url https://api.openai.com/v1

  # SFT Data Generation with local server
  python verifiers_server.py process \
      --env.vf_env_name "primeintellect/alphabet-sort" \
      --env.data_path_to_save_groups local_sft_data.jsonl \
      --openai.base_url http://localhost:9001/v1

To install a Verifiers/Prime environment:
1. uv tool install prime
2. prime login
3. prime env install primeintellect/alphabet-sort (or any owner/environment)
Docs: https://docs.primeintellect.ai/tutorials-environments/install
"""

import json
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import verifiers as vf

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.envs.server_handling.managed_server import ManagedServerAdapter

logger = logging.getLogger(__name__)


class VfEnvConfig(BaseEnvConfig):
    vf_env_name: str = ""
    env_args: str = "{}"

    def get_env_args(self) -> Dict[str, Any]:
        """Parse env_args JSON string into dict."""
        if isinstance(self.env_args, dict):
            return self.env_args
        return json.loads(self.env_args)


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

        # Metrics buffers for wandb logging
        self.reward_buffer: List[float] = []
        self.metrics_buffer: Dict[str, List[float]] = defaultdict(list)
        self.num_turns_buffer: List[int] = []
        self.groups_with_identical_scores: int = 0
        self.groups_total: int = 0

        logger.info("Loading verifiers environment: %s", config.vf_env_name)
        env_args = config.get_env_args()
        if env_args:
            logger.info("Environment args: %s", env_args)
        self.vf_env = vf.load_environment(config.vf_env_name, **env_args)
        self.rubric = self.vf_env.rubric
        self.system_prompt = self.vf_env.system_prompt

        # Get reward function names for metrics reporting
        self.reward_func_names = self.rubric._get_reward_func_names()
        logger.info("Reward functions: %s", self.reward_func_names)

        # Log multi-turn config if available
        if hasattr(self.vf_env, "max_turns"):
            logger.info("Max turns: %d", self.vf_env.max_turns)

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
                model_name="Qwen/Qwen2.5-1.5B-Instruct",
                base_url="http://localhost:9001/v1",
                api_key="x",
                num_requests_for_eval=4,
                server_type="sglang",
            ),
        ]
        return env_config, server_configs

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Enhanced wandb logging with verifiers-specific metrics."""
        if wandb_metrics is None:
            wandb_metrics = {}

        # Log mean reward across all rollouts
        if self.reward_buffer:
            wandb_metrics["metrics/mean_reward"] = sum(self.reward_buffer) / len(
                self.reward_buffer
            )
            wandb_metrics["metrics/reward_std"] = (
                (
                    sum(
                        (r - wandb_metrics["metrics/mean_reward"]) ** 2
                        for r in self.reward_buffer
                    )
                    / len(self.reward_buffer)
                )
                ** 0.5
                if len(self.reward_buffer) > 1
                else 0.0
            )
            self.reward_buffer = []

        # Log per-reward-function metrics (e.g., strict_accuracy, format_score)
        if self.metrics_buffer:
            for metric_name, values in self.metrics_buffer.items():
                if values:
                    avg_metric = sum(values) / len(values)
                    wandb_metrics[f"metrics/{metric_name}"] = avg_metric
            self.metrics_buffer = defaultdict(list)

        # Log multi-turn statistics
        if self.num_turns_buffer:
            wandb_metrics["metrics/avg_num_turns"] = sum(self.num_turns_buffer) / len(
                self.num_turns_buffer
            )
            wandb_metrics["metrics/max_num_turns"] = max(self.num_turns_buffer)
            self.num_turns_buffer = []

        # Log group filtering statistics (helpful for debugging)
        if self.groups_total > 0:
            wandb_metrics["metrics/groups_with_identical_scores"] = (
                self.groups_with_identical_scores
            )
            wandb_metrics["metrics/groups_total"] = self.groups_total
            wandb_metrics["metrics/identical_score_rate"] = (
                self.groups_with_identical_scores / self.groups_total
            )
            # Reset counters
            self.groups_with_identical_scores = 0
            self.groups_total = 0

        await super().wandb_log(wandb_metrics)

    async def setup(self):
        # Dataset already has: prompt, answer, info, example_id, task
        train_data = self.vf_env.get_dataset()
        self.train = train_data.to_list()
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

    async def collect_trajectories(self, item) -> Tuple[ScoredDataGroup, list]:
        """Unified trajectory collection using vf_env.generate() with ManagedServer.

        Works for both RL training (serve) and SFT data generation (process).
        Uses ManagedServer adapter for automatic token and logprob tracking.
        """
        # Get server config (handle both real servers and test harness)
        if hasattr(self.server, "servers") and self.server.servers:
            server_config = self.server.servers[0].config
        else:
            # Fallback for testing
            server_config = APIServerConfig(
                model_name=self.config.tokenizer_name,
                base_url="http://localhost:8000/v1",
            )

        # Build inputs for group_size rollouts
        inputs = [
            {
                "prompt": item["prompt"],
                "answer": item.get("answer", ""),
                "example_id": item["example_id"],
                "task": item.get("task", self.config.vf_env_name),
                "info": item.get("info", {}),
            }
            for _ in range(self.config.group_size)
        ]

        # Use ManagedServer for automatic token/logprob tracking
        async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
            # Create adapter that looks like AsyncOpenAI for verifiers
            adapter = ManagedServerAdapter(
                managed_server=managed,
                base_url=server_config.base_url,
            )

            # Use vf_env.generate() - handles batching and scoring internally
            results = await self.vf_env.generate(
                inputs=inputs,
                client=adapter,
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

            # Get tracked state from ManagedServer
            managed_state = managed.get_state()
            nodes = managed_state["nodes"]

        scored_data: ScoredDataGroup = {
            "tokens": [],
            "masks": [],
            "scores": [],
            "messages": [],
            "inference_logprobs": [],
        }

        # Zip verifiers states with ManagedServer nodes for logprob tracking
        for i, vf_state in enumerate(results["state"]):
            # Extract messages from state
            messages = list(vf_state.get("prompt", [])) + list(
                vf_state.get("completion", [])
            )
            messages = [
                {**msg, "content": msg.get("content") or ""} for msg in messages
            ]

            # Get trajectory for metrics
            trajectory = vf_state.get("trajectory", [])

            # Get tokens, masks, and logprobs from ManagedServer
            # IMPORTANT: We use ManagedServer's tokens (not re-tokenize) to ensure
            # alignment with logprobs. ManagedServer tracks tokens and logprobs together.
            if i >= len(nodes):
                raise RuntimeError(
                    f"Node count mismatch: expected at least {i + 1} nodes, got {len(nodes)}. "
                    "ManagedServer should track all rollouts."
                )

            node = nodes[i]
            scored_data["tokens"].append(node.tokens)
            scored_data["masks"].append(node.masked_tokens)
            scored_data["inference_logprobs"].append(node.logprobs)
            scored_data["messages"].append(messages)

            reward = vf_state.get("reward", 0.0)
            scored_data["scores"].append(reward)

            # Metrics logging
            self.reward_buffer.append(reward)
            num_turns = len(trajectory)
            self.num_turns_buffer.append(num_turns)
            logger.debug("Rollout: %d turns, reward=%.3f", num_turns, reward)

            # Per-function metrics from verifiers state
            state_metrics = vf_state.get("metrics", {})
            for metric_name, metric_value in state_metrics.items():
                if isinstance(metric_value, (int, float)):
                    self.metrics_buffer[metric_name].append(float(metric_value))

        # Log group summary
        turns = [len(s.get("trajectory", [])) for s in results["state"]]
        logger.info(
            "Group: %d rollouts, turns=%s, rewards=%s, nodes=%d",
            len(results["state"]),
            turns,
            [f"{s:.3f}" for s in scored_data["scores"]],
            len(nodes),
        )

        # Track identical scores for debugging
        self.groups_total += 1
        if len(set(scored_data["scores"])) == 1:
            self.groups_with_identical_scores += 1
            logger.debug(
                "Group has identical scores (%.3f) - will be filtered by base env",
                scored_data["scores"][0],
            )

        return scored_data, []


if __name__ == "__main__":
    VerifiersEnv.cli()
