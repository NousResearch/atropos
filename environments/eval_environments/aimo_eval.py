"""
AIMO Evaluation Environment for Atropos (Generative Mode)

This environment evaluates models on AIMO (AI Mathematical Olympiad) Progress Prize 1 -
a training set from the Kaggle AIMO competition designed to test AI mathematical reasoning.

Dataset: lighteval/aimo_progress_prize_1

The evaluation follows a generative approach:
- Models receive math olympiad problems
- Expected to provide step-by-step reasoning
- Final answer in \\boxed{} format
- Uses math_verify for robust answer verification
- Falls back to string normalization if symbolic comparison fails

Supports thinking mode with <think></think> tags for extended reasoning.
"""

import asyncio
import os
import random
import re
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple

import wandb
from datasets import load_dataset
from eval_helpers import (
    THINK_CONTENT_AFTER_PATTERN,
    create_system_content,
    extract_boxed_answers,
    extract_thinking_content,
    format_math_answer_instruction,
    get_default_thinking_prompt,
    get_math_executor,
    save_eval_results,
    score_math_answer_async,
    validate_thinking_format,
)
from pydantic import Field
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
)

# Prompt template - AIMO doesn't have a specific template in lighteval
# Using our standard math prompt with boxed instruction
AIMO_PROMPT_TEMPLATE = """Solve the following mathematical olympiad problem. {answer_instruction}

{problem}"""


class AIMOEvalConfig(BaseEnvConfig):
    """Configuration for AIMO evaluation environment."""

    # Dataset configuration
    dataset_name: str = Field(
        default="lighteval/aimo_progress_prize_1",
        description="HuggingFace dataset name",
    )
    eval_split: str = Field(
        default="train", description="Split to evaluate on (AIMO uses train split)"
    )
    shuffle_seed: int = Field(default=42, description="Random seed for shuffling")

    # Generation parameters
    eval_temperature: float = Field(
        default=0.6, description="Temperature for evaluation generation"
    )
    eval_max_tokens: int = Field(
        default=0, description="Max tokens for evaluation (0 = use model default)"
    )

    # System prompt configuration
    custom_system_prompt: Optional[str] = Field(
        default=None, description="Optional custom system prompt"
    )

    # Thinking mode configuration
    thinking_mode: bool = Field(
        default=True,
        description="Whether to use thinking mode with <think></think> tags",
    )
    custom_thinking_prompt: Optional[str] = Field(
        default=None, description="Optional custom thinking prompt"
    )

    # Math verification configuration
    include_hope_suffix: bool = Field(
        default=True,
        description="Whether to include 'I hope it is correct' in answer instruction",
    )
    max_math_workers: int = Field(
        default=64,
        description="Maximum workers for math verification ProcessPoolExecutor",
    )

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

    # Override defaults
    group_size: int = 1
    max_num_workers: int = 1024
    max_eval_workers: int = 256
    max_num_workers_per_node: int = 128
    use_wandb: bool = True
    rollout_server_url: str = "http://localhost:8000"
    total_steps: int = 1
    wandb_name: str = "aimo_eval"
    steps_per_eval: int = 1


class AIMOEvalEnv(BaseEnv):
    """
    AIMO Evaluation Environment.

    Evaluates mathematical olympiad problem solving using AIMO dataset.
    Uses math_verify for robust answer verification with string fallback.
    """

    name = "aimo_eval"

    def __init__(
        self,
        config: AIMOEvalConfig,
        server_configs: List[APIServerConfig],
        slurm_job_id: Optional[str] = None,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm_job_id, testing)
        self.config: AIMOEvalConfig = config
        self.eval_items: List[Dict] = []
        self._dataset_loaded = False
        self._math_executor: Optional[ProcessPoolExecutor] = None

    @classmethod
    def config_cls(cls) -> type:
        return AIMOEvalConfig

    async def setup(self) -> None:
        """Initialize the environment and load the dataset."""
        await super().setup()

        # Initialize math executor
        self._math_executor = get_math_executor(self.config.max_math_workers)

        if not self._dataset_loaded:
            await self._load_dataset()

        print(f"\nAIMO Evaluation Setup (Generative Mode):")
        print(f"  Dataset: {self.config.dataset_name}")
        print(f"  Evaluation split: {self.config.eval_split}")
        print(f"  Thinking mode: {self.config.thinking_mode}")
        if self.config.thinking_mode:
            thinking_prompt = get_default_thinking_prompt(
                self.config.custom_thinking_prompt
            )
            print(f"  Thinking prompt: {thinking_prompt[:80]}...")
        print(f"  Loaded {len(self.eval_items)} evaluation items")

    async def _load_dataset(self) -> None:
        """Load and process the AIMO dataset."""
        print(f"Loading AIMO dataset: {self.config.dataset_name}...")

        try:
            dataset = load_dataset(self.config.dataset_name, trust_remote_code=True)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

        if self.config.eval_split not in dataset:
            available_splits = list(dataset.keys())
            print(
                f"Split '{self.config.eval_split}' not found. Available: {available_splits}"
            )
            # Use first available split
            if available_splits:
                split_key = available_splits[0]
                print(f"Using '{split_key}' instead")
            else:
                raise ValueError("No splits available in dataset")
        else:
            split_key = self.config.eval_split

        split_data = dataset[split_key]

        # Process items
        self.eval_items = []
        for idx, item in enumerate(split_data):
            problem = item.get("problem", "")
            answer = str(item.get("answer", "")).strip()

            self.eval_items.append(
                {
                    "id": str(idx),
                    "problem": problem,
                    "answer": answer,
                }
            )

        # Shuffle with seed
        random.seed(self.config.shuffle_seed)
        random.shuffle(self.eval_items)

        self._dataset_loaded = True
        print(f"Loaded {len(self.eval_items)} evaluation items")

    def _format_prompt(self, item: Dict) -> str:
        """Format the problem into a prompt."""
        answer_instruction = format_math_answer_instruction(
            include_hope=self.config.include_hope_suffix
        )
        return AIMO_PROMPT_TEMPLATE.format(
            answer_instruction=answer_instruction, problem=item["problem"]
        )

    def _create_system_content(self) -> str:
        """Create system message content based on thinking mode."""
        return (
            create_system_content(
                self.config.thinking_mode,
                self.config.custom_thinking_prompt,
                self.config.custom_system_prompt,
            )
            or ""
        )

    async def rollout_and_score_eval(
        self,
        item: Dict,
        server: APIServerConfig,
    ) -> Optional[Dict]:
        """Run evaluation on a single item and return the result."""
        prompt = self._format_prompt(item)
        system_content = self._create_system_content()

        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": prompt})

        # Build API call parameters
        kwargs = {
            "model": server.model_name,
            "messages": messages,
            "temperature": self.config.eval_temperature,
        }
        if self.config.eval_max_tokens > 0:
            kwargs["max_tokens"] = self.config.eval_max_tokens

        response_text = ""
        for attempt in range(self.config.max_retries):
            try:
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

        # Validate thinking format
        is_valid_format, content_for_extraction = validate_thinking_format(
            response_text, self.config.thinking_mode
        )

        # Extract thinking content if present
        thinking_content = (
            extract_thinking_content(response_text)
            if self.config.thinking_mode
            else None
        )

        # Score using math_verify with string fallback
        gold_answer = item["answer"]
        is_correct, method, has_multiple_boxed = await score_math_answer_async(
            gold=gold_answer,
            response=response_text,
            after_think=self.config.thinking_mode,
            wrap_gold_boxed=True,
            executor=self._math_executor,
            debug=self.config.full_debug,
        )

        # Extract the boxed answer for logging
        if self.config.thinking_mode:
            match = THINK_CONTENT_AFTER_PATTERN.search(response_text)
            score_content = match.group(1) if match else response_text
        else:
            score_content = response_text

        boxed_answers = extract_boxed_answers(score_content)
        extracted_answer = boxed_answers[0] if boxed_answers else None

        if self.config.full_debug:
            print(f"\n--- Item: {item['id']} ---")
            print(f"Problem: {item['problem'][:100]}...")
            print(f"Gold answer: {gold_answer}")
            print(f"Extracted: {extracted_answer}")
            print(f"Correct: {is_correct} (method: {method})")

        return {
            "item_id": item["id"],
            "problem": item["problem"][:200],
            "gold_answer": gold_answer,
            "extracted_answer": extracted_answer,
            "verification_method": method,
            "is_correct": is_correct if is_correct is not None else False,
            "has_multiple_boxed": has_multiple_boxed,
            "format_valid": is_valid_format,
            "response": response_text,
            "thinking_content": thinking_content,
            "has_thinking": thinking_content is not None,
        }

    async def evaluate(self, *args, **kwargs) -> Dict:
        """Run the full AIMO evaluation."""
        print(f"\n{'='*60}")
        print("Starting AIMO Evaluation (Generative Mode)")
        print(f"{'='*60}")
        print(f"  Total problems: {len(self.eval_items)}")
        print(f"  Thinking mode: {self.config.thinking_mode}")
        print(f"{'='*60}\n")

        # Create evaluation tasks
        async def eval_task(item):
            return await self.rollout_and_score_eval(item, self.server_configs[0])

        tasks = [eval_task(item) for item in self.eval_items]

        # Run with progress bar
        results = await tqdm_asyncio.gather(*tasks, desc="Evaluating AIMO")

        # Filter out failed results
        valid_results = [r for r in results if r is not None]

        if not valid_results:
            print("Warning: No valid evaluation results obtained")
            return {"error": "No valid results", "accuracy": 0.0}

        # Calculate overall metrics
        total = len(valid_results)
        correct = sum(1 for r in valid_results if r["is_correct"])
        accuracy = correct / total if total > 0 else 0.0

        # Count verification methods and other stats
        method_counts = {}
        for r in valid_results:
            method = r.get("verification_method", "unknown")
            method_counts[method] = method_counts.get(method, 0) + 1

        multiple_boxed = sum(
            1 for r in valid_results if r.get("has_multiple_boxed", False)
        )
        format_valid = sum(1 for r in valid_results if r.get("format_valid", True))
        has_thinking = sum(1 for r in valid_results if r.get("has_thinking", False))
        has_boxed = sum(
            1 for r in valid_results if r.get("extracted_answer") is not None
        )

        metrics = {
            "accuracy": accuracy,
            "total_evaluated": total,
            "total_correct": correct,
            "has_boxed_rate": has_boxed / total if total > 0 else 0.0,
            "multiple_boxed_rate": multiple_boxed / total if total > 0 else 0.0,
            "format_compliance_rate": format_valid / total if total > 0 else 0.0,
            "thinking_utilization_rate": has_thinking / total if total > 0 else 0.0,
            "verification_methods": method_counts,
        }

        print(f"\n{'='*60}")
        print("AIMO Evaluation Results")
        print(f"{'='*60}")
        print(f"  Overall Accuracy: {accuracy:.2%} ({correct}/{total})")
        print(f"  Has \\boxed{{}} Rate: {has_boxed / total:.2%}")
        if multiple_boxed > 0:
            print(f"  Multiple \\boxed{{}} (failed): {multiple_boxed / total:.2%}")
        print(f"  Format Compliance: {format_valid / total:.2%}")
        if self.config.thinking_mode:
            print(f"  Thinking Utilization: {has_thinking / total:.2%}")
        print(f"\n  Verification Methods:")
        for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
            print(f"    {method}: {count} ({count/total:.1%})")
        print(f"{'='*60}\n")

        # Save results
        if self.config.data_dir_to_save_evals:
            self._save_results(metrics, valid_results)

        return metrics

    def _save_results(self, metrics: Dict, results: List[Dict]) -> None:
        """Save evaluation results to disk."""
        save_eval_results(self.config.data_dir_to_save_evals, metrics, results)

    async def wandb_log(self, metrics: Dict, step: int = 0) -> None:
        """Log metrics to Weights & Biases."""
        if not self.config.use_wandb:
            return

        log_dict = {
            "aimo/accuracy": metrics.get("accuracy", 0),
            "aimo/total_evaluated": metrics.get("total_evaluated", 0),
            "aimo/has_boxed_rate": metrics.get("has_boxed_rate", 0),
            "aimo/multiple_boxed_rate": metrics.get("multiple_boxed_rate", 0),
            "aimo/format_compliance_rate": metrics.get("format_compliance_rate", 0),
            "aimo/thinking_utilization_rate": metrics.get(
                "thinking_utilization_rate", 0
            ),
        }

        wandb.log(log_dict, step=step)

    # Required abstract method implementations
    async def get_next_item(self) -> Optional[Dict]:
        """Not used in evaluation mode."""
        return None

    async def collect_trajectories(self, item) -> List:
        """Not used in evaluation mode."""
        return []

    async def score(self, rollout_group_data) -> Optional[List]:
        """Not used in evaluation mode."""
        return None


if __name__ == "__main__":
    AIMOEvalEnv.cli()
