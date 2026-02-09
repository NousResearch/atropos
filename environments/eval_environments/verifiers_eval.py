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
    # Evaluate with OpenAI
    python verifiers_eval.py \
        --server-url https://api.openai.com/v1 \
        --model-name gpt-4o \
        --vf-env-name primeintellect/gsm8k \
        --max-eval-items 50

    # Evaluate with local server
    python verifiers_eval.py \
        --server-url http://localhost:8000/v1 \
        --model-name Qwen/Qwen2.5-7B-Instruct \
        --vf-env-name primeintellect/gsm8k

Works with any OpenAI-compatible API (OpenAI, vLLM, SGLang, Ollama, etc.)
"""

import argparse
import json
import time
from typing import Any, Dict

import verifiers as vf

from atroposlib.envs.eval import EvalBase, evaluate_log
from atroposlib.envs.server_handling.managed_server import ManagedServerAdapter
from atroposlib.envs.server_handling.server_baseline import APIServerConfig
from atroposlib.envs.server_handling.server_manager import ServerManager

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


class VerifiersEval(EvalBase):
    """
    Verifiers Evaluation using EvalBase pattern.

    Uses verifiers' native batch evaluation for efficiency,
    with ManagedServerAdapter for token/logprob tracking.

    Works with any OpenAI-compatible API (OpenAI, vLLM, SGLang, Ollama, etc.)
    """

    def __init__(
        self,
        vf_env_name: str = "primeintellect/gsm8k",
        env_args: str = "{}",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        max_eval_items: int = -1,
        max_concurrent: int = 64,
        **kwargs,
    ):
        self.vf_env_name = vf_env_name
        self.env_args_str = env_args
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_eval_items = max_eval_items
        self.max_concurrent = max_concurrent
        super().__init__(**kwargs)

    def get_env_args(self) -> Dict[str, Any]:
        """Parse env_args JSON string into dict."""
        if isinstance(self.env_args_str, dict):
            return self.env_args_str
        return json.loads(self.env_args_str)

    def setup_data(self) -> list:
        """Load verifiers environment and dataset."""
        env_args = self.get_env_args()
        self.vf_env = vf.load_environment(self.vf_env_name, **env_args)
        self.reward_func_names = self.vf_env.rubric._get_reward_func_names()

        # Load evaluation dataset
        dataset = self.vf_env.get_eval_dataset()
        if self.max_eval_items > 0:
            n = min(len(dataset), self.max_eval_items)
            dataset = dataset.select(range(n))
        return dataset.to_list()

    async def run_item(self, server: ServerManager, data_item: dict):
        """Not used - verifiers uses batch evaluation in __call__."""
        # This won't be called since we override __call__
        raise NotImplementedError("VerifiersEval uses batch evaluation in __call__")

    async def __call__(self, server_manager: ServerManager):
        """Run evaluation using verifiers with ManagedServerAdapter."""
        start_time = time.time()

        # Get server config
        server = server_manager.servers[0]
        model_name = server.config.model_name

        num_examples = self.max_eval_items if self.max_eval_items > 0 else -1

        # Use ManagedServer for automatic token/logprob tracking
        async with server_manager.managed_server(tokenizer=None) as managed:
            # Create adapter that looks like AsyncOpenAI for verifiers
            adapter = ManagedServerAdapter(
                managed_server=managed,
                base_url=server.config.base_url,
            )

            # Use verifiers' batch evaluation
            results = await self.vf_env.evaluate(
                client=adapter,
                model=model_name,
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

        # Log results
        task_name = f"VerifiersEval@{self.vf_env_name.replace('/', '_')}"
        evaluate_log(
            metrics=metrics,
            eval_dir=getattr(self, "eval_dir", None),
            task_name=task_name,
            model_name=model_name,
            start_time=start_time,
            end_time=end_time,
            generation_parameters={
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "n": 1,
            },
            samples=samples,
            verbose=getattr(self, "verbose", True),
        )

        return metrics


async def main():
    """Run verifiers evaluation with argparse CLI."""
    import os

    parser = argparse.ArgumentParser(
        description="Evaluate models using Prime Intellect's Verifiers library"
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8000/v1",
        help="URL of the inference server (default: http://localhost:8000/v1)",
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
        default=None,
        help="API key (default: uses OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--vf-env-name",
        type=str,
        default="primeintellect/gsm8k",
        help="Verifiers environment name (default: primeintellect/gsm8k)",
    )
    parser.add_argument(
        "--env-args",
        type=str,
        default="{}",
        help="JSON string of environment-specific args (default: {})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature (default: 0.0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens per completion (default: 2048)",
    )
    parser.add_argument(
        "--max-eval-items",
        type=int,
        default=-1,
        help="Maximum items to evaluate, -1 for all (default: -1)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=64,
        help="Maximum concurrent requests (default: 64)",
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default=None,
        help="Directory to save evaluation results (default: None)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print verbose output (default: True)",
    )

    args = parser.parse_args()

    # Get API key from args or environment
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "dummy")

    # Create evaluation instance
    eval_env = VerifiersEval(
        vf_env_name=args.vf_env_name,
        env_args=args.env_args,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_eval_items=args.max_eval_items,
        max_concurrent=args.max_concurrent,
        eval_dir=args.eval_dir,
        verbose=args.verbose,
    )

    # Create server manager
    server_manager = ServerManager(
        configs=[
            APIServerConfig(
                api_key=api_key,
                base_url=args.server_url,
                model_name=args.model_name,
                health_check=False,
            ),
        ]
    )

    # Run evaluation
    return await eval_env(server_manager)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
