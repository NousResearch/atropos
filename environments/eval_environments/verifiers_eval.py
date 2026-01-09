"""
Verifiers Evaluation Environment for Atropos

This environment evaluates models using Prime Intellect's Verifiers library.
It supports any environment registered with the Verifiers ecosystem.

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
"""

import asyncio
import os
import time
from typing import Dict, List, Optional, Tuple

import verifiers as vf
from pydantic import Field
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
)


class VerifiersEvaluationConfig(BaseEnvConfig):
    """Configuration for Verifiers evaluation environment."""

    # Verifiers environment
    vf_env_name: str = Field(
        default="",
        description="Verifiers environment name (e.g., primeintellect/gsm8k)",
    )
    env_args: dict = Field(
        default_factory=dict,
        description="Additional arguments for verifiers environment",
    )

    # Generation parameters
    temperature: float = Field(
        default=0.0, description="Temperature for generation (0.0 for deterministic)"
    )
    max_tokens: int = Field(default=2048, description="Maximum tokens for generation")

    # Retry and debug configuration
    max_retries: int = Field(
        default=3, description="Maximum retries for failed API calls"
    )
    retry_delay: float = Field(
        default=1.0, description="Delay between retries in seconds"
    )
    min_response_length: int = Field(
        default=1, description="Minimum response length to consider valid"
    )
    full_debug: bool = Field(default=False, description="Enable full debug output")

    # Override defaults for evaluation mode
    group_size: int = 1
    max_num_workers: int = 256
    max_num_workers_per_node: int = 64
    use_wandb: bool = True
    rollout_server_url: str = "http://localhost:8000"
    total_steps: int = 1
    wandb_name: str = "verifiers_evaluation"
    steps_per_eval: int = 1


class VerifiersEvaluationEnv(BaseEnv):
    """
    Verifiers Evaluation Environment.

    Evaluates models using Prime Intellect's Verifiers library rubrics.
    Works with any OpenAI-compatible API (OpenAI, vLLM, SGLang, etc.)
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

        # Load verifiers environment
        self.vf_env = vf.load_environment(config.vf_env_name, **config.env_args)
        self.rubric = self.vf_env.rubric

        # Extract rubric components
        self.parser = self.rubric.parser
        self.reward_funcs = self.rubric.funcs
        self.reward_weights = self.rubric.weights
        self.reward_scales = [
            weight / sum(self.reward_weights) for weight in self.reward_weights
        ]
        self.system_prompt = self.vf_env.system_prompt

        # Tracking
        self.eval_items: List[Dict] = []
        self._dataset_loaded = False

    @classmethod
    def config_init(cls) -> Tuple[VerifiersEvaluationConfig, List[APIServerConfig]]:
        """Default configuration for evaluation."""
        env_config = VerifiersEvaluationConfig(
            vf_env_name="primeintellect/gsm8k",
            temperature=0.0,
            max_tokens=2048,
            use_wandb=True,
            wandb_name="verifiers_evaluation",
        )
        server_configs = [
            APIServerConfig(
                model_name="gpt-4.1-nano",
                base_url=None,
                api_key=os.getenv("OPENAI_API_KEY"),
                num_requests_for_eval=256,
            ),
        ]
        return env_config, server_configs

    async def setup(self) -> None:
        """Initialize the environment and load datasets."""
        if not self._dataset_loaded:
            # Load datasets from verifiers environment
            test_data = self.vf_env.get_eval_dataset()
            self.eval_items = test_data.select_columns(["question", "answer"]).to_list()
            self._dataset_loaded = True

        print("\nVerifiers Evaluation Setup:")
        print(f"  Environment: {self.config.vf_env_name}")
        print(f"  Reward functions: {len(self.reward_funcs)}")
        print(f"  Reward weights: {self.reward_weights}")
        print(f"  Loaded {len(self.eval_items)} evaluation items")

    async def rollout_and_score(self, item: Dict) -> Optional[Dict]:
        """
        Run evaluation on a single item and return the result.

        Args:
            item: Dict with 'question' and 'answer' keys

        Returns:
            Dict with evaluation results or None if failed
        """
        question = item["question"]
        answer = item["answer"]

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]

        # Build API call parameters
        kwargs = {
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "n": 1,
        }

        response_text = ""
        for attempt in range(self.config.max_retries):
            try:
                # Direct API call (no ManagedServer) - eval doesn't need token tracking
                response = await self.server.chat_completion(**kwargs)
                response_text = response.choices[0].message.content or ""

                if len(response_text) >= self.config.min_response_length:
                    break

            except Exception as e:
                if self.config.full_debug:
                    print(f"  API error (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay)
                continue

        if not response_text:
            return None

        # Build completion messages for scoring
        completion_messages = messages + [
            {"role": "assistant", "content": response_text}
        ]

        # Parse answer
        answer_parsed = self.parser.parse_answer(completion=response_text)

        # Score using reward funcs
        rewards = []
        for func in self.reward_funcs:
            reward = func(
                parser=self.parser,
                completion=completion_messages,
                answer=answer,
            )
            rewards.append(reward)

        weighted_rewards = [r * self.reward_scales[j] for j, r in enumerate(rewards)]
        score = sum(weighted_rewards)

        if self.config.full_debug:
            print("\n--- Item ---")
            print(f"Question: {question[:100]}...")
            print(f"Gold answer: {answer}")
            print(f"Model parsed: {answer_parsed}")
            print(f"Rewards: {rewards}")
            print(f"Score: {score}")

        return {
            "question": question,
            "gold_answer": answer,
            "response": response_text,
            "model_parsed": str(answer_parsed) if answer_parsed else None,
            "rewards": rewards,
            "weighted_rewards": weighted_rewards,
            "score": score,
            "correct": bool(score > 0),
        }

    async def evaluate(self, *args, **kwargs) -> Dict:
        """Run the full evaluation."""
        print(f"\n{'=' * 60}")
        print(f"Starting Verifiers Evaluation: {self.config.vf_env_name}")
        print(f"{'=' * 60}")
        print(f"  Total questions: {len(self.eval_items)}")
        print(f"  Temperature: {self.config.temperature}")
        print(f"{'=' * 60}\n")

        start_time = time.time()

        # Create evaluation tasks
        tasks = [self.rollout_and_score(item) for item in self.eval_items]

        # Run with progress bar
        results = await tqdm_asyncio.gather(*tasks, desc="Evaluating")

        # Filter out failed results
        valid_results = [r for r in results if r is not None]

        if not valid_results:
            print("Warning: No valid evaluation results obtained")
            return {"error": "No valid results", "accuracy": 0.0}

        end_time = time.time()

        # Calculate metrics
        total = len(valid_results)
        scores = [r["score"] for r in valid_results]
        correct = sum(1 for r in valid_results if r["correct"])

        avg_score = sum(scores) / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        # Per-reward function breakdown
        reward_breakdown = {}
        for i, weight in enumerate(self.reward_weights):
            func_rewards = [r["rewards"][i] for r in valid_results]
            reward_breakdown[f"reward_func_{i}"] = {
                "weight": weight,
                "avg": sum(func_rewards) / len(func_rewards),
                "correct": sum(1 for r in func_rewards if r > 0),
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

        # Log to evaluate_log
        samples = [
            {
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": r["question"]},
                    {"role": "assistant", "content": r["response"]},
                ],
                "question": r["question"],
                "gold_answer": r["gold_answer"],
                "model_parsed": r["model_parsed"],
                "score": r["score"],
                "correct": r["correct"],
            }
            for r in valid_results
        ]

        await self.evaluate_log(
            metrics={"accuracy": accuracy, "avg_score": avg_score},
            samples=samples,
            start_time=start_time,
            end_time=end_time,
            generation_parameters={
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            },
        )

        return metrics

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None) -> None:
        """Log metrics to Weights & Biases."""
        if wandb_metrics is None:
            wandb_metrics = {}

        # Add config info
        wandb_metrics["config/vf_env_name"] = self.config.vf_env_name
        wandb_metrics["config/temperature"] = self.config.temperature
        wandb_metrics["config/max_tokens"] = self.config.max_tokens

        await super().wandb_log(wandb_metrics)

    # Required abstract method implementations (stubs for evaluation-only mode)
    async def get_next_item(self) -> Optional[Dict]:
        """Not used in evaluation mode."""
        raise NotImplementedError("get_next_item not supported in evaluation-only mode")

    async def collect_trajectories(self, item) -> Tuple[List, List]:
        """Not used in evaluation mode."""
        raise NotImplementedError(
            "collect_trajectories not supported in evaluation-only mode"
        )


if __name__ == "__main__":
    VerifiersEvaluationEnv.cli()
