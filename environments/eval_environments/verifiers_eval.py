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
    python verifiers_eval.py \
        --server-url http://localhost:8000/v1 \
        --model-name Qwen/Qwen2.5-7B-Instruct \
        --vf-env-name primeintellect/gsm8k \
        --max-eval-items 100

Works with any OpenAI-compatible API (OpenAI, vLLM, SGLang, Ollama, etc.)
"""

import argparse
import asyncio
import time
from typing import Tuple

import verifiers as vf
from openai import AsyncOpenAI

from atroposlib.envs.eval import EvalBase, evaluate_log
from atroposlib.envs.server_handling.server_manager import ServerManager


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


class VerifiersEval(EvalBase):
    """
    Verifiers Evaluation using EvalBase pattern.

    Uses verifiers' native batch evaluation for efficiency,
    with EvalBase's standardized logging via evaluate_log().

    Works with any OpenAI-compatible API (OpenAI, vLLM, SGLang, Ollama, etc.)
    """

    def __init__(
        self,
        vf_env_name: str = "primeintellect/gsm8k",
        env_args: dict = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        max_eval_items: int = -1,
        max_concurrent: int = 64,
        eval_dir: str = None,
        verbose: bool = True,
        **kwargs,
    ):
        self.vf_env_name = vf_env_name
        self.env_args = env_args or {}
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_eval_items = max_eval_items
        self.max_concurrent = max_concurrent

        # Load verifiers environment
        self.vf_env = vf.load_environment(vf_env_name, **self.env_args)
        self.reward_func_names = self.vf_env.rubric._get_reward_func_names()

        # Initialize EvalBase (calls setup_data)
        super().__init__(
            eval_dir=eval_dir,
            verbose=verbose,
            **kwargs,
        )

    def get_generation_params(self):
        """Generation params for logging."""
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "n": 1,
        }

    def setup_data(self) -> list:
        """Return evaluation dataset from verifiers environment."""
        dataset = self.vf_env.get_eval_dataset()
        if self.max_eval_items > 0:
            n = min(len(dataset), self.max_eval_items)
            dataset = dataset.select(range(n))
        return dataset.to_list()

    async def run_item(
        self, server: ServerManager, data_item: dict  # noqa: ARG002
    ) -> Tuple[dict, list]:
        """Not used - we override __call__ for batch evaluation."""
        raise NotImplementedError(
            "VerifiersEval uses batch evaluation via __call__, not per-item run_item"
        )

    async def __call__(self, server_manager: ServerManager):
        """Run evaluation using verifiers' native batch machinery."""
        start_time = time.time()

        # Create OpenAI client from server config
        server = server_manager.servers[0]
        client = AsyncOpenAI(
            api_key=server.config.api_key or "x",
            base_url=server.config.base_url,
            timeout=getattr(server.config, "timeout", 600),
        )
        model = server.config.model_name

        print(f"\n{'=' * 60}")
        print(f"Verifiers Evaluation: {self.vf_env_name}")
        print(f"{'=' * 60}")
        print(f"  Model: {model}")
        print(f"  Items: {len(self.data)}")
        print(f"  Reward functions: {self.reward_func_names}")
        print(f"  Temperature: {self.temperature}")
        print(f"  Max concurrent: {self.max_concurrent}")
        print(f"{'=' * 60}\n")

        num_examples = self.max_eval_items if self.max_eval_items > 0 else -1

        # Use verifiers' batch evaluation
        results = await self.vf_env.evaluate(
            client=client,
            model=model,
            sampling_args={
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            num_examples=num_examples,
            max_concurrent=self.max_concurrent,
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

        # Use EvalBase's evaluate_log
        task_name = f"VerifiersEval@{self.vf_env_name.replace('/', '_')}"
        evaluate_log(
            metrics=metrics,
            eval_dir=getattr(self, "eval_dir", None),
            task_name=task_name,
            model_name=model,
            start_time=start_time,
            end_time=end_time,
            generation_parameters=self.get_generation_params(),
            samples=samples,
            verbose=getattr(self, "verbose", False),
        )

        return metrics


async def main():
    """CLI entry point for verifiers evaluation."""
    import os

    from atroposlib.envs.server_handling.server_baseline import APIServerConfig

    parser = argparse.ArgumentParser(
        description="Evaluate models using Verifiers environments"
    )
    # Server args (same as eval_runner)
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8000/v1",
        help="URL of the inference server",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name to evaluate",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY", "x"),
        help="API key (defaults to OPENAI_API_KEY env var)",
    )
    # Verifiers-specific args
    parser.add_argument(
        "--vf-env-name",
        type=str,
        default="primeintellect/gsm8k",
        help="Verifiers environment name (e.g., primeintellect/gsm8k)",
    )
    parser.add_argument(
        "--max-eval-items",
        type=int,
        default=-1,
        help="Maximum items to evaluate (-1 for all)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens per completion",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=64,
        help="Maximum concurrent requests",
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default=None,
        help="Directory to save evaluation results",
    )
    args = parser.parse_args()

    # Create server manager
    server_manager = ServerManager(
        configs=[
            APIServerConfig(
                api_key=args.api_key,
                base_url=args.server_url,
                model_name=args.model_name,
                health_check=False,
            ),
        ]
    )

    # Create and run evaluation
    eval_instance = VerifiersEval(
        vf_env_name=args.vf_env_name,
        max_eval_items=args.max_eval_items,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_concurrent=args.max_concurrent,
        eval_dir=args.eval_dir,
        verbose=True,
    )
    return await eval_instance(server_manager)


if __name__ == "__main__":
    asyncio.run(main())
