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
    # Evaluate with local server
    python verifiers_eval.py evaluate \
        --env.vf_env_name "primeintellect/gsm8k" \
        --env.max_eval_items 100 \
        --openai.model_name "Qwen/Qwen2.5-7B-Instruct" \
        --openai.base_url "http://localhost:8000/v1"

    # Evaluate with OpenAI
    python verifiers_eval.py evaluate \
        --env.vf_env_name "primeintellect/gsm8k" \
        --env.max_eval_items 50 \
        --openai.model_name "gpt-4o" \
        --openai.api_key "$OPENAI_API_KEY" \
        --openai.base_url "https://api.openai.com/v1"

Works with any OpenAI-compatible API (OpenAI, vLLM, SGLang, Ollama, etc.)
"""

import json
import time
from typing import Any, Dict, List, Tuple

import verifiers as vf

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)

# Import ManagedServerAdapter from shared location
from atroposlib.envs.server_handling.managed_server import ManagedServerAdapter

# Patch math_verify timeout to work in async context
# The signal-based timeout doesn't work in non-main threads (asyncio event loop)


def _no_signal_timeout(
    _timeout_seconds: int | None = None, *, timeout_seconds: int | None = None
):
    """Replacement timeout decorator that doesn't use signals.

    Accepts both positional arg (timeout(5)) and keyword arg (timeout(timeout_seconds=5)).
    """
    # Silence unused parameter warnings - these match the original API signature
    del _timeout_seconds, timeout_seconds

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


class VfEvalConfig(BaseEnvConfig):
    """Configuration for Verifiers evaluation environment."""

    vf_env_name: str = "primeintellect/gsm8k"
    env_args: str = "{}"  # JSON string for environment-specific args
    eval_temperature: float = 0.0
    eval_max_tokens: int = 2048
    max_eval_items: int = -1  # -1 means evaluate all items
    max_concurrent: int = 64

    # Override BaseEnvConfig defaults for eval mode
    group_size: int = 1
    total_steps: int = 1
    steps_per_eval: int = 1
    use_wandb: bool = True

    def get_env_args(self) -> Dict[str, Any]:
        """Parse env_args JSON string into dict."""
        if isinstance(self.env_args, dict):
            return self.env_args
        return json.loads(self.env_args)


class VerifiersEvalEnv(BaseEnv):
    """
    Verifiers Evaluation Environment using BaseEnv pattern.

    Uses verifiers' native batch evaluation for efficiency,
    with BaseEnv's standardized logging via evaluate_log().

    Works with any OpenAI-compatible API (OpenAI, vLLM, SGLang, Ollama, etc.)
    """

    name = "verifiers_eval"
    env_config_cls = VfEvalConfig  # type: ignore[assignment]

    @classmethod
    def config_init(cls) -> Tuple[VfEvalConfig, List[APIServerConfig]]:
        """Return default configurations."""
        env_config = VfEvalConfig(
            tokenizer_name="Qwen/Qwen2.5-1.5B-Instruct",
            vf_env_name="primeintellect/gsm8k",
            wandb_name="verifiers_eval",
        )
        server_configs = [
            APIServerConfig(
                model_name="Qwen/Qwen2.5-1.5B-Instruct",
                base_url="http://localhost:9001/v1",
                api_key="x",
            ),
        ]
        return env_config, server_configs

    async def setup(self):
        """Load verifiers environment and dataset."""
        env_args = self.config.get_env_args()
        self.vf_env = vf.load_environment(self.config.vf_env_name, **env_args)
        self.reward_func_names = self.vf_env.rubric._get_reward_func_names()

        # Load evaluation dataset
        dataset = self.vf_env.get_eval_dataset()
        if self.config.max_eval_items > 0:
            n = min(len(dataset), self.config.max_eval_items)
            dataset = dataset.select(range(n))
        self.data = dataset.to_list()

    async def get_next_item(self):
        """Not used in eval mode - stub implementation."""
        return None

    async def collect_trajectories(self, item) -> Tuple[ScoredDataGroup, list]:
        """Not used in eval mode - stub implementation."""
        _ = item  # unused in eval mode
        return (
            ScoredDataGroup(
                tokens=[],
                masks=[],
                scores=[],
                messages=[],
                inference_logprobs=[],
                advantages=[],
                ref_logprobs=[],
                generation_params=None,
                group_overrides=None,
                overrides=[],
                images=[],
            ),
            [],
        )

    async def evaluate(self, *args, **kwargs) -> Dict[str, float]:
        """Run evaluation using verifiers with ManagedServer."""
        start_time = time.time()

        # Get server config
        if hasattr(self.server, "servers") and self.server.servers:
            server_config = self.server.servers[0].config
        else:
            server_config = self.server_configs[0]

        model_name = server_config.model_name

        print(f"\n{'=' * 60}")
        print(f"Verifiers Evaluation: {self.config.vf_env_name}")
        print(f"{'=' * 60}")
        print(f"  Model: {model_name}")
        print(f"  Items: {len(self.data)}")
        print(f"  Reward functions: {self.reward_func_names}")
        print(f"  Temperature: {self.config.eval_temperature}")
        print(f"  Max concurrent: {self.config.max_concurrent}")
        print(f"{'=' * 60}\n")

        num_examples = (
            self.config.max_eval_items if self.config.max_eval_items > 0 else -1
        )

        # Use ManagedServer for automatic token/logprob tracking
        async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
            # Create adapter that looks like AsyncOpenAI for verifiers
            adapter = ManagedServerAdapter(
                managed_server=managed,
                base_url=server_config.base_url,
            )

            # Use verifiers' batch evaluation
            results = await self.vf_env.evaluate(
                client=adapter,
                model=model_name,
                sampling_args={
                    "temperature": self.config.eval_temperature,
                    "max_tokens": self.config.eval_max_tokens,
                },
                num_examples=num_examples,
                max_concurrent=self.config.max_concurrent,
                save_results=False,
            )

        end_time = time.time()

        # Extract from verifiers output
        rewards = results["reward"]
        per_func_metrics = results["metrics"]
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
            "accuracy": accuracy,
            "avg_score": avg_score,
        }

        # Add per-function metrics
        for func_name, data in reward_breakdown.items():
            metrics[f"{func_name}_avg"] = data["avg"]
            metrics[f"{func_name}_correct_rate"] = data["correct"] / total

        # Print results summary
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

        # Build samples for logging
        system_prompt = self.vf_env.system_prompt or ""
        samples = []
        for i in range(min(total, 100)):  # Limit samples for logging
            prompt_msgs = prompts[i] if isinstance(prompts[i], list) else []
            completion_msgs = completions[i] if completions[i] else []

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

        # Use BaseEnv's evaluate_log
        task_name = f"VerifiersEval@{self.config.vf_env_name.replace('/', '_')}"
        await self.evaluate_log(
            metrics=metrics,
            task_name=task_name,
            model_name=model_name,
            start_time=start_time,
            end_time=end_time,
            generation_parameters={
                "temperature": self.config.eval_temperature,
                "max_tokens": self.config.eval_max_tokens,
                "n": 1,
            },
            samples=samples,
            verbose=True,
        )

        return metrics


if __name__ == "__main__":
    VerifiersEvalEnv.cli()
