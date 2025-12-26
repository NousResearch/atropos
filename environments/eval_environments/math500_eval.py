"""
MATH-500 Evaluation Environment for Atropos (Generative Mode)

This environment evaluates models on MATH-500 - a subset of 500 problems from
the MATH benchmark that OpenAI created for their "Let's Verify Step by Step" paper.

Dataset: HuggingFaceH4/MATH-500
Paper: https://arxiv.org/abs/2305.20050

The evaluation follows a generative approach:
- Models receive challenging math problems
- Expected to provide step-by-step reasoning
- Final answer in \\boxed{} format
- Uses math_verify for robust answer verification
- Falls back to string normalization if symbolic comparison fails

Supports thinking mode with <think></think> tags for extended reasoning.
"""

import asyncio
import random
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional

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
)

# Prompt template following lighteval's MATH-500 structure
MATH500_PROMPT_TEMPLATE = """Solve the following problem. The final line of your response MUST be of the following format:
"ANSWER: $ANSWER" (without quotes) where $ANSWER is the final answer. Think step by step before answering.

However, for reliable parsing, also put your final answer in \\boxed{{}} format.
  # noqa: E501
{problem}"""


# Alternative prompt that just uses boxed (matches our standard)
MATH500_BOXED_PROMPT_TEMPLATE = """Solve the following math problem. {answer_instruction}

{problem}"""


class MATH500EvalConfig(BaseEnvConfig):
    """Configuration for MATH-500 evaluation environment."""

    # Dataset configuration
    dataset_name: str = Field(
        default="HuggingFaceH4/MATH-500", description="HuggingFace dataset name"
    )
    subset: str = Field(default="default", description="Dataset subset")
    eval_split: str = Field(default="test", description="Split to evaluate on")
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
    use_original_prompt: bool = Field(
        default=False,
        description="Use lighteval's original prompt format (ANSWER: format)",
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
    wandb_name: str = "math500_eval"
    steps_per_eval: int = 1


class MATH500EvalEnv(BaseEnv):
    """
    MATH-500 Evaluation Environment.

    Evaluates challenging math problem solving using the MATH-500 dataset.
    Uses math_verify for robust answer verification with string fallback.
    """

    name = "math500_eval"

    def __init__(
        self,
        config: MATH500EvalConfig,
        server_configs: List[APIServerConfig],
        slurm_job_id: Optional[str] = None,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm_job_id, testing)
        self.config: MATH500EvalConfig = config
        self.eval_items: List[Dict] = []
        self._dataset_loaded = False
        self._math_executor: Optional[ProcessPoolExecutor] = None

    @classmethod
    def config_cls(cls) -> type:
        return MATH500EvalConfig

    async def setup(self) -> None:
        """Initialize the environment and load the dataset."""
        await super().setup()

        # Initialize math executor
        self._math_executor = get_math_executor(self.config.max_math_workers)

        if not self._dataset_loaded:
            await self._load_dataset()

        print("\nMATH-500 Evaluation Setup (Generative Mode):")
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
        """Load and process the MATH-500 dataset."""
        print(f"Loading MATH-500 dataset: {self.config.dataset_name}...")

        try:
            dataset = load_dataset(
                self.config.dataset_name,
                self.config.subset if self.config.subset != "default" else None,
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

        if self.config.eval_split not in dataset:
            available_splits = list(dataset.keys())
            raise ValueError(
                f"Split '{self.config.eval_split}' not found. Available: {available_splits}"
            )

        split_data = dataset[self.config.eval_split]

        # Process items
        self.eval_items = []
        for idx, item in enumerate(split_data):
            problem = item.get("problem", "")
            answer = item.get("answer", "")  # MATH-500 has 'answer' field
            solution = item.get("solution", "")  # May also have solution

            # Extract final answer
            if answer:
                final_answer = answer.strip()
            elif solution:
                # Try to extract from solution
                boxed = extract_boxed_answers(solution)
                final_answer = boxed[-1] if boxed else solution.strip()
            else:
                final_answer = ""

            subject = item.get("subject", "unknown")
            level = item.get("level", "")
            unique_id = item.get("unique_id", str(idx))

            self.eval_items.append(
                {
                    "id": unique_id,
                    "problem": problem,
                    "answer": final_answer,
                    "solution": solution,
                    "subject": subject,
                    "level": level,
                }
            )

        # Shuffle with seed
        random.seed(self.config.shuffle_seed)
        random.shuffle(self.eval_items)

        self._dataset_loaded = True
        print(f"Loaded {len(self.eval_items)} evaluation items")

    def _format_prompt(self, item: Dict) -> str:
        """Format the problem into a prompt."""
        if self.config.use_original_prompt:
            return MATH500_PROMPT_TEMPLATE.format(problem=item["problem"])
        else:
            answer_instruction = format_math_answer_instruction(
                include_hope=self.config.include_hope_suffix
            )
            return MATH500_BOXED_PROMPT_TEMPLATE.format(
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
            print(
                f"Subject: {item.get('subject', 'N/A')}, Level: {item.get('level', 'N/A')}"
            )
            print(f"Problem: {item['problem'][:100]}...")
            print(f"Gold answer: {gold_answer}")
            print(f"Extracted: {extracted_answer}")
            print(f"Correct: {is_correct} (method: {method})")

        return {
            "item_id": item["id"],
            "subject": item.get("subject", "unknown"),
            "level": item.get("level", ""),
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
        """Run the full MATH-500 evaluation."""
        print(f"\n{'='*60}")
        print("Starting MATH-500 Evaluation (Generative Mode)")
        print(f"{'='*60}")
        print(f"  Total questions: {len(self.eval_items)}")
        print(f"  Thinking mode: {self.config.thinking_mode}")
        print(f"{'='*60}\n")

        # Create evaluation tasks
        async def eval_task(item):
            return await self.rollout_and_score_eval(item, self.server_configs[0])

        tasks = [eval_task(item) for item in self.eval_items]

        # Run with progress bar
        results = await tqdm_asyncio.gather(*tasks, desc="Evaluating MATH-500")

        # Filter out failed results
        valid_results = [r for r in results if r is not None]

        if not valid_results:
            print("Warning: No valid evaluation results obtained")
            return {"error": "No valid results", "accuracy": 0.0}

        # Calculate overall metrics
        total = len(valid_results)
        correct = sum(1 for r in valid_results if r["is_correct"])
        accuracy = correct / total if total > 0 else 0.0

        # Calculate per-subject metrics
        subject_metrics = {}
        for r in valid_results:
            subject = r.get("subject", "unknown")
            if subject not in subject_metrics:
                subject_metrics[subject] = {"total": 0, "correct": 0}
            subject_metrics[subject]["total"] += 1
            if r["is_correct"]:
                subject_metrics[subject]["correct"] += 1

        for subject in subject_metrics:
            s_total = subject_metrics[subject]["total"]
            s_correct = subject_metrics[subject]["correct"]
            subject_metrics[subject]["accuracy"] = (
                s_correct / s_total if s_total > 0 else 0.0
            )

        # Calculate per-level metrics
        level_metrics = {}
        for r in valid_results:
            level = r.get("level", "unknown") or "unknown"
            if level not in level_metrics:
                level_metrics[level] = {"total": 0, "correct": 0}
            level_metrics[level]["total"] += 1
            if r["is_correct"]:
                level_metrics[level]["correct"] += 1

        for level in level_metrics:
            l_total = level_metrics[level]["total"]
            l_correct = level_metrics[level]["correct"]
            level_metrics[level]["accuracy"] = (
                l_correct / l_total if l_total > 0 else 0.0
            )

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
            "num_subjects": len(subject_metrics),
            "has_boxed_rate": has_boxed / total if total > 0 else 0.0,
            "multiple_boxed_rate": multiple_boxed / total if total > 0 else 0.0,
            "format_compliance_rate": format_valid / total if total > 0 else 0.0,
            "thinking_utilization_rate": has_thinking / total if total > 0 else 0.0,
            "subject_metrics": subject_metrics,
            "level_metrics": level_metrics,
            "verification_methods": method_counts,
        }

        print(f"\n{'='*60}")
        print("MATH-500 Evaluation Results")
        print(f"{'='*60}")
        print(f"  Overall Accuracy: {accuracy:.2%} ({correct}/{total})")
        print(f"  Has \\boxed{{}} Rate: {has_boxed / total:.2%}")
        print(f"  Format Compliance: {format_valid / total:.2%}")
        if self.config.thinking_mode:
            print(f"  Thinking Utilization: {has_thinking / total:.2%}")
        if subject_metrics and len(subject_metrics) > 1:
            print("\n  Per-Subject Breakdown:")
            for subject, data in sorted(
                subject_metrics.items(), key=lambda x: -x[1]["accuracy"]
            ):
                print(
                    f"    {subject}: {data['accuracy']:.2%} ({data['correct']}/{data['total']})"
                )
        if level_metrics and len(level_metrics) > 1:
            print("\n  Per-Level Breakdown:")
            for level, data in sorted(level_metrics.items()):
                print(
                    f"    Level {level}: {data['accuracy']:.2%} ({data['correct']}/{data['total']})"
                )
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
            "math500/accuracy": metrics.get("accuracy", 0),
            "math500/total_evaluated": metrics.get("total_evaluated", 0),
            "math500/has_boxed_rate": metrics.get("has_boxed_rate", 0),
            "math500/format_compliance_rate": metrics.get("format_compliance_rate", 0),
            "math500/thinking_utilization_rate": metrics.get(
                "thinking_utilization_rate", 0
            ),
        }

        # Log per-subject accuracies
        for subject, data in metrics.get("subject_metrics", {}).items():
            safe_name = subject.replace(" ", "_")[:30]
            log_dict[f"math500/accuracy_{safe_name}"] = data.get("accuracy", 0)

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
    MATH500EvalEnv.cli()
