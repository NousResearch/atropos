"""
Verifiers Evaluation Environment for Atropos

This environment evaluates models using Prime Intellect's Verifiers library.
It supports any environment registered with the Verifiers ecosystem.

Uses verifiers' native rollout and scoring machinery - just pass an OpenAI-compatible
client and verifiers handles generation, parsing, and scoring.

To install a Verifiers/Prime environment:
1. uv tool install prime
2. prime login
3. prime env install will/wordle (or any owner/environment)
Docs: https://docs.primeintellect.ai/tutorials-environments/install

Usage:
    python verifiers_eval.py evaluate \
        --env.vf_env_name primeintellect/gsm8k \
        --openai.model_name gpt-4.1-nano \
        --openai.api_key $OPENAI_API_KEY

Works with any OpenAI-compatible API (OpenAI, vLLM, SGLang, Ollama, etc.):
    python verifiers_eval.py evaluate \
        --env.vf_env_name primeintellect/gsm8k \
        --openai.model_name Qwen/Qwen2.5-7B-Instruct \
        --openai.base_url http://localhost:8000/v1
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import verifiers as vf
import wandb
from openai import AsyncOpenAI
from pydantic import Field

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
)


# Patch math_verify timeout to work in async context
# The signal-based timeout doesn't work in non-main threads (asyncio event loop)
def _no_signal_timeout(_timeout_seconds: int):
    """Replacement timeout decorator that doesn't use signals."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Just call the function without timeout - safe in async context
            return func(*args, **kwargs)

        return wrapper

    return decorator


try:
    import math_verify.grader
    import math_verify.parser
    import math_verify.utils

    # Patch all modules that use the timeout decorator
    math_verify.utils.timeout = _no_signal_timeout
    math_verify.parser.timeout = _no_signal_timeout
    math_verify.grader.timeout = _no_signal_timeout
except ImportError:
    pass  # math_verify not installed


class VerifiersEvaluationConfig(BaseEnvConfig):
    """Configuration for Verifiers evaluation environment."""

    vf_env_name: str = Field(
        default="",
        description="Verifiers environment name (e.g., primeintellect/gsm8k)",
    )
    env_args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional arguments for verifiers environment",
    )
    temperature: float = Field(
        default=0.0,
        description="Temperature for generation (0.0 for deterministic)",
    )
    max_eval_items: int = Field(
        default=-1,
        description="Maximum number of items to evaluate (-1 for all)",
    )
    max_concurrent: int = Field(
        default=64,
        description="Maximum concurrent requests to the model",
    )

    # Override BaseEnvConfig defaults for evaluation
    group_size: int = 1
    max_num_workers: int = 1024
    max_eval_workers: int = 256
    max_num_workers_per_node: int = 128
    use_wandb: bool = True
    rollout_server_url: str = "http://localhost:8000"
    total_steps: int = 1
    steps_per_eval: int = 1
    wandb_name: str = "verifiers_eval"


class VerifiersEvaluationEnv(BaseEnv):
    """
    Verifiers Evaluation Environment.

    Evaluates models using Prime Intellect's Verifiers library.
    Uses verifiers' native rollout and scoring machinery.

    Works with any OpenAI-compatible API (OpenAI, vLLM, SGLang, Ollama, etc.)
    """

    name = "verifiers_evaluation"
    env_config_cls = VerifiersEvaluationConfig  # type: ignore[assignment]

    def __init__(
        self,
        config: VerifiersEvaluationConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = False,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: VerifiersEvaluationConfig = config

        self.vf_env = vf.load_environment(config.vf_env_name, **config.env_args)

        # Get reward function names for metrics reporting
        self.reward_func_names = self.vf_env.rubric._get_reward_func_names()

    @classmethod
    def config_init(cls) -> Tuple[VerifiersEvaluationConfig, List[APIServerConfig]]:
        """Default configuration for evaluation."""
        env_config = VerifiersEvaluationConfig(
            vf_env_name="primeintellect/gsm8k",
        )
        server_configs = [
            APIServerConfig(
                model_name="gpt-4.1-nano",
                base_url="https://api.openai.com/v1",
                api_key=os.getenv("OPENAI_API_KEY"),
            ),
        ]
        return env_config, server_configs

    def _get_openai_client(self) -> AsyncOpenAI:
        """Create AsyncOpenAI client from first server config."""
        server = self.server.servers[0]
        config = server.config
        return AsyncOpenAI(
            api_key=config.api_key or "x",
            base_url=config.base_url,
            timeout=config.timeout,
        )

    def _get_model_name(self) -> str:
        """Get model name from first server config."""
        return self.server.servers[0].config.model_name

    async def setup(self) -> None:
        """Initialize the environment."""
        num_eval = len(self.vf_env.get_eval_dataset())
        if self.config.max_eval_items > 0:
            num_eval = min(num_eval, self.config.max_eval_items)

        print("\nVerifiers Evaluation Setup:")
        print(f"  Environment: {self.config.vf_env_name}")
        print(f"  Reward functions: {self.reward_func_names}")
        print(f"  Evaluation items: {num_eval}")
        print(f"  Max concurrent: {self.config.max_concurrent}")

    async def evaluate(self) -> Dict:
        """Run evaluation using verifiers' native machinery."""
        num_examples = (
            self.config.max_eval_items if self.config.max_eval_items > 0 else -1
        )

        print(f"\n{'=' * 60}")
        print(f"Starting Verifiers Evaluation: {self.config.vf_env_name}")
        print(f"{'=' * 60}")
        print(f"  Model: {self._get_model_name()}")
        print(f"  Temperature: {self.config.temperature}")
        print(f"  Max concurrent: {self.config.max_concurrent}")
        print(f"{'=' * 60}\n")

        start_time = time.time()

        # Create OpenAI client from atropos server config
        client = self._get_openai_client()
        model = self._get_model_name()

        # Let verifiers handle everything: rollouts + scoring
        results = await self.vf_env.evaluate(
            client=client,
            model=model,
            sampling_args={
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_token_length,
            },
            num_examples=num_examples,
            max_concurrent=self.config.max_concurrent,
            save_results=False,
        )

        end_time = time.time()

        # Extract metrics from verifiers output
        rewards = results["reward"]
        per_func_metrics = results["metrics"]  # dict of func_name -> list[float]
        prompts = results["prompt"]
        completions = results["completion"]
        answers = results["answer"]

        total = len(rewards)
        correct = sum(1 for r in rewards if r > 0)
        avg_score = sum(rewards) / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        # Per-reward function breakdown
        reward_breakdown = {}
        for func_name, values in per_func_metrics.items():
            if values:
                reward_breakdown[func_name] = {
                    "avg": sum(values) / len(values),
                    "correct": sum(1 for v in values if v > 0),
                }

        metrics = {
            "avg_score": avg_score,
            "accuracy": accuracy,
            "total_evaluated": total,
            "total_correct": correct,
            "reward_breakdown": reward_breakdown,
        }

        # Print results
        print(f"\n{'=' * 60}")
        print("Verifiers Evaluation Results")
        print(f"{'=' * 60}")
        print(f"  Average Score: {avg_score:.4f}")
        print(f"  Accuracy: {accuracy:.2%} ({correct}/{total})")
        print(f"  Time: {end_time - start_time:.1f}s")
        print("\n  Per-Reward Function:")
        for name, data in reward_breakdown.items():
            print(
                f"    {name}: avg={data['avg']:.4f}, correct={data['correct']}/{total}"
            )
        print(f"{'=' * 60}\n")

        # Log to evaluate_log (atropos's logging system)
        system_prompt = self.vf_env.system_prompt or ""
        samples = []
        for i in range(min(total, 100)):  # Limit samples for logging
            prompt_msgs = prompts[i] if isinstance(prompts[i], list) else []
            completion_msgs = completions[i] if completions[i] else []

            # Build full message list
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.extend(prompt_msgs)
            if isinstance(completion_msgs, list):
                messages.extend(completion_msgs)

            samples.append(
                {
                    "messages": messages,
                    "gold_answer": answers[i] if i < len(answers) else "",
                    "score": rewards[i],
                    "correct": rewards[i] > 0,
                    "metrics": {
                        k: v[i] for k, v in per_func_metrics.items() if i < len(v)
                    },
                }
            )

        await self.evaluate_log(
            metrics={"accuracy": accuracy, "avg_score": avg_score},
            samples=samples,
            start_time=start_time,
            end_time=end_time,
            generation_parameters={
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_token_length,
            },
        )

        # Log to wandb
        await self.wandb_log(metrics)

        return metrics

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None) -> None:
        """Log metrics to Weights & Biases."""
        if not self.config.use_wandb or wandb_metrics is None:
            return

        # Lazy init if wandb not already initialized
        if wandb.run is None:
            wandb.init(
                project="atropos-environments",
                name=self.config.wandb_name,
                config=self.config.model_dump(),
            )

        log_dict = {
            "verifiers/accuracy": wandb_metrics.get("accuracy", 0),
            "verifiers/avg_score": wandb_metrics.get("avg_score", 0),
            "verifiers/total_evaluated": wandb_metrics.get("total_evaluated", 0),
            "verifiers/total_correct": wandb_metrics.get("total_correct", 0),
        }

        # Add per-reward function metrics
        reward_breakdown = wandb_metrics.get("reward_breakdown", {})
        for func_name, data in reward_breakdown.items():
            log_dict[f"verifiers/{func_name}_avg"] = data.get("avg", 0)
            log_dict[f"verifiers/{func_name}_correct"] = data.get("correct", 0)

        wandb.log(log_dict)

    # Required abstract method implementations (stubs for evaluation-only mode)
    async def get_next_item(self) -> Optional[Dict]:
        """Not used in evaluation mode."""
        return None

    async def collect_trajectories(self, item) -> Tuple[List, List]:  # noqa: ARG002
        """Not used in evaluation mode."""
        return [], []


if __name__ == "__main__":
    VerifiersEvaluationEnv.cli()
