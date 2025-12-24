"""
BigBench Hard (BBH) Evaluation Environment for Atropos (Generative Mode)

This environment evaluates models on BigBench Hard - a collection of 23 challenging
tasks from the BIG-Bench benchmark that are particularly difficult for language models.

Dataset: lighteval/bbh
Paper: https://arxiv.org/abs/2210.09261

The evaluation follows a generative approach:
- Models receive challenging reasoning problems
- All tasks are multiple choice (MCQA) with varying number of choices
- Answers are extracted from <answer></answer> tags
- Supports thinking mode with <think></think> tags for extended reasoning

Available subsets include: causal_judgment, date_understanding, disambiguation_qa,
geometric_shapes, logical_deduction (3/5/7 objects), movie_recommendation,
navigate, reasoning_about_colored_objects, ruin_names, salient_translation_error_detection,
snarks, sports_understanding, temporal_sequences, tracking_shuffled_objects (3/5/7 objects).
"""

import asyncio
import random
from string import ascii_uppercase
from typing import Dict, List, Optional, Tuple

import wandb
from datasets import load_dataset
from eval_helpers import (
    build_mcqa_fallback_patterns,
    create_system_content,
    extract_letter_from_answer_tag,
    get_default_thinking_prompt,
    save_eval_results,
    validate_thinking_format,
)
from pydantic import Field
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
)

# All available BBH subsets
BBH_SUBSETS = [
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "geometric_shapes",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "navigate",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
]


def format_bbh_prompt(item: Dict) -> Tuple[str, List[str], int]:
    """
    Format a BBH item into a prompt.

    Args:
        item: The dataset item

    Returns:
        Tuple of (prompt_text, choices_list, gold_index)
    """
    # Build the query
    task_prefix = item.get("task_prefix", "")
    input_prefix = item.get("example_input_prefix", "\nQuestion: ")
    input_text = item.get("input", "")
    choice_prefix = item.get("choice_prefix", "\n  Choices: ")
    # Note: output_prefix from item.get("example_output_prefix") is not used in generative mode
    choices = item.get("choices", [])
    target_idx = item.get("target_idx", 0)

    # Build choice text
    num_choices = len(choices)
    valid_letters = list(ascii_uppercase[:num_choices])

    choice_text = ""
    for i, (letter, choice) in enumerate(zip(valid_letters, choices)):
        choice_text += f"\n{letter}. {choice}"

    # Add answer tag instruction
    valid_letters_str = (
        ", ".join(valid_letters[:-1]) + f", or {valid_letters[-1]}"
        if len(valid_letters) > 1
        else valid_letters[0]
    )

    query = f"""Answer the following question. Think step by step before answering.

Provide your final answer within <answer></answer> tags, containing only the letter ({valid_letters_str}).

Example format:
<answer>A</answer>

"""

    # Add task-specific content
    if task_prefix:
        query += task_prefix
    query += input_prefix
    query += input_text
    query += choice_prefix
    query += choice_text

    return query, choices, target_idx


class BBHEvalConfig(BaseEnvConfig):
    """Configuration for BigBench Hard evaluation environment."""

    # Dataset configuration
    dataset_name: str = Field(
        default="lighteval/bbh", description="HuggingFace dataset name"
    )
    subset: str = Field(
        default="all",
        description="Subset to evaluate ('all' for all subsets, or specific subset name)",
    )
    eval_split: str = Field(
        default="train",
        description="Split to evaluate on (train is typically the only available split)",
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
    wandb_name: str = "bbh_eval"
    steps_per_eval: int = 1


class BBHEvalEnv(BaseEnv):
    """
    BigBench Hard (BBH) Evaluation Environment.

    Evaluates models on challenging reasoning tasks from the BIG-Bench benchmark.
    All tasks are multiple choice with answer extraction from <answer></answer> tags.
    """

    name = "bbh_eval"

    def __init__(
        self,
        config: BBHEvalConfig,
        server_configs: List[APIServerConfig],
        slurm_job_id: Optional[str] = None,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm_job_id, testing)
        self.config: BBHEvalConfig = config
        self.eval_items: List[Dict] = []
        self._dataset_loaded = False

    @classmethod
    def config_cls(cls) -> type:
        return BBHEvalConfig

    async def setup(self) -> None:
        """Initialize the environment and load the dataset."""
        await super().setup()

        if not self._dataset_loaded:
            await self._load_dataset()

        print("\nBBH Evaluation Setup (Generative Mode):")
        print(f"  Dataset: {self.config.dataset_name}")
        print(f"  Subset: {self.config.subset}")
        print(f"  Evaluation split: {self.config.eval_split}")
        print(f"  Thinking mode: {self.config.thinking_mode}")
        if self.config.thinking_mode:
            thinking_prompt = get_default_thinking_prompt(
                self.config.custom_thinking_prompt
            )
            print(f"  Thinking prompt: {thinking_prompt[:80]}...")
        print(f"  Loaded {len(self.eval_items)} evaluation items")

    async def _load_dataset(self) -> None:
        """Load and process the BBH dataset."""

        # Determine which subsets to load
        if self.config.subset.lower() == "all":
            subsets_to_load = BBH_SUBSETS
        else:
            if self.config.subset not in BBH_SUBSETS:
                print(
                    f"Warning: Subset '{self.config.subset}' may not exist. Available: {BBH_SUBSETS}"
                )
            subsets_to_load = [self.config.subset]

        self.eval_items = []

        for subset in subsets_to_load:
            print(f"Loading BBH subset: {subset}...")

            try:
                dataset = load_dataset(
                    self.config.dataset_name, subset, trust_remote_code=True
                )
            except Exception as e:
                print(f"  Error loading subset '{subset}': {e}")
                continue

            if self.config.eval_split not in dataset:
                available_splits = list(dataset.keys())
                print(
                    f"  Split '{self.config.eval_split}' not found for {subset}. Available: {available_splits}"
                )
                continue

            split_data = dataset[self.config.eval_split]

            # Process items
            for idx, item in enumerate(split_data):
                # Skip items without choices
                choices = item.get("choices", [])
                if not choices:
                    continue

                self.eval_items.append(
                    {
                        "id": f"{subset}_{idx}",
                        "subset": subset,
                        "raw_item": item,
                        "choices": choices,
                        "target_idx": item.get("target_idx", 0),
                        "input": item.get("input", ""),
                    }
                )

            print(
                f"  Loaded {len([i for i in self.eval_items if i['subset'] == subset])} items from {subset}"
            )

        # Shuffle with seed
        random.seed(self.config.shuffle_seed)
        random.shuffle(self.eval_items)

        self._dataset_loaded = True
        print(
            f"Total: Loaded {len(self.eval_items)} evaluation items from {len(subsets_to_load)} subsets"
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

    def _extract_answer(
        self, response: str, num_choices: int, choices: List[str], debug: bool = False
    ) -> Tuple[Optional[str], str]:
        """
        Extract the letter answer from the response.

        Args:
            response: The model's response (content after </think> in thinking mode)
            num_choices: Number of valid choices
            choices: List of choice texts
            debug: Whether to print debug information

        Returns:
            Tuple of (extracted_letter or None, extraction_method)
        """
        if not response:
            return None, "empty_response"

        valid_letters = set(ascii_uppercase[:num_choices])

        # PRIMARY: Try <answer></answer> tags
        letter, method = extract_letter_from_answer_tag(
            response, valid_letters, debug=debug, choices=choices
        )
        if letter:
            return letter, method

        # FALLBACK: Use regex patterns
        fallback_patterns = build_mcqa_fallback_patterns(num_choices)

        for priority, pattern, method_name in fallback_patterns:
            matches = pattern.findall(response)
            if matches:
                # Get the last match for answer patterns
                match = (
                    matches[-1]
                    if method_name
                    in [
                        "final_answer_is",
                        "the_answer_is",
                        "answer_colon",
                        "answer_space",
                    ]
                    else matches[0]
                )

                if isinstance(match, tuple):
                    match = match[0]
                extracted = match.strip("()").upper()

                if extracted in valid_letters:
                    if debug:
                        print(
                            f"    Extracted '{extracted}' using fallback '{method_name}'"
                        )
                    return extracted, f"fallback_{method_name}"

        # Last resort: find any valid letter (take the last one)
        for letter in reversed(list(valid_letters)):
            if letter in response.upper():
                if debug:
                    print(
                        f"    Extracted '{letter}' using fallback 'last_valid_letter'"
                    )
                return letter, "fallback_last_valid_letter"

        return None, "no_match"

    async def rollout_and_score_eval(
        self,
        item: Dict,
        server: APIServerConfig,
    ) -> Optional[Dict]:
        """Run evaluation on a single item and return the result."""

        # Format the prompt
        prompt, choices, target_idx = format_bbh_prompt(item["raw_item"])
        num_choices = len(choices)
        gold_letter = (
            ascii_uppercase[target_idx] if 0 <= target_idx < num_choices else None
        )

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

        # Validate thinking format and extract content after </think>
        is_valid_format, content_for_extraction = validate_thinking_format(
            response_text, self.config.thinking_mode
        )

        # Extract thinking content if present
        thinking_content = (
            extract_thinking_content(response_text)
            if self.config.thinking_mode
            else None
        )

        # Extract answer
        extracted_answer, extraction_method = self._extract_answer(
            content_for_extraction, num_choices, choices, debug=self.config.full_debug
        )

        # Score
        is_correct = (
            extracted_answer == gold_letter
            if extracted_answer and gold_letter
            else False
        )

        if self.config.full_debug:
            print(f"\n--- Item: {item['id']} ---")
            print(f"Subset: {item['subset']}")
            print(f"Input: {item['input'][:100]}...")
            print(
                f"Gold: {gold_letter}, Extracted: {extracted_answer} (method: {extraction_method})"
            )
            print(f"Correct: {is_correct}")

        return {
            "item_id": item["id"],
            "subset": item["subset"],
            "input": item["input"][:200],
            "num_choices": num_choices,
            "gold_letter": gold_letter,
            "extracted_answer": extracted_answer,
            "extraction_method": extraction_method,
            "is_correct": is_correct,
            "format_valid": is_valid_format,
            "response": response_text,
            "thinking_content": thinking_content,
            "has_thinking": thinking_content is not None,
        }

    async def evaluate(self, *args, **kwargs) -> Dict:
        """Run the full BBH evaluation."""
        print(f"\n{'='*60}")
        print("Starting BBH Evaluation (Generative Mode)")
        print(f"{'='*60}")
        print(f"  Subset: {self.config.subset}")
        print(f"  Total questions: {len(self.eval_items)}")
        print(f"  Thinking mode: {self.config.thinking_mode}")
        print(f"{'='*60}\n")

        # Create evaluation tasks
        async def eval_task(item):
            return await self.rollout_and_score_eval(item, self.server_configs[0])

        tasks = [eval_task(item) for item in self.eval_items]

        # Run with progress bar
        results = await tqdm_asyncio.gather(*tasks, desc="Evaluating BBH")

        # Filter out failed results
        valid_results = [r for r in results if r is not None]

        if not valid_results:
            print("Warning: No valid evaluation results obtained")
            return {"error": "No valid results", "accuracy": 0.0}

        # Calculate overall metrics
        total = len(valid_results)
        correct = sum(1 for r in valid_results if r["is_correct"])
        accuracy = correct / total if total > 0 else 0.0

        # Calculate per-subset metrics
        subset_metrics = {}
        for r in valid_results:
            subset = r.get("subset", "unknown")
            if subset not in subset_metrics:
                subset_metrics[subset] = {"total": 0, "correct": 0}
            subset_metrics[subset]["total"] += 1
            if r["is_correct"]:
                subset_metrics[subset]["correct"] += 1

        for subset in subset_metrics:
            s_total = subset_metrics[subset]["total"]
            s_correct = subset_metrics[subset]["correct"]
            subset_metrics[subset]["accuracy"] = (
                s_correct / s_total if s_total > 0 else 0.0
            )

        # Format compliance and thinking utilization
        format_valid = sum(1 for r in valid_results if r.get("format_valid", True))
        has_thinking = sum(1 for r in valid_results if r.get("has_thinking", False))

        # Extraction method breakdown
        method_counts = {}
        for r in valid_results:
            method = r.get("extraction_method", "unknown")
            method_counts[method] = method_counts.get(method, 0) + 1

        metrics = {
            "accuracy": accuracy,
            "total_evaluated": total,
            "total_correct": correct,
            "num_subsets": len(subset_metrics),
            "format_compliance_rate": format_valid / total if total > 0 else 0.0,
            "thinking_utilization_rate": has_thinking / total if total > 0 else 0.0,
            "subset_metrics": subset_metrics,
            "extraction_methods": method_counts,
        }

        print(f"\n{'='*60}")
        print("BBH Evaluation Results")
        print(f"{'='*60}")
        print(f"  Overall Accuracy: {accuracy:.2%} ({correct}/{total})")
        print(f"  Number of Subsets: {len(subset_metrics)}")
        print(f"  Format Compliance: {format_valid / total:.2%}")
        if self.config.thinking_mode:
            print(f"  Thinking Utilization: {has_thinking / total:.2%}")
        print("\n  Per-Subset Breakdown:")
        for subset, data in sorted(
            subset_metrics.items(), key=lambda x: -x[1]["accuracy"]
        ):
            print(
                f"    {subset}: {data['accuracy']:.2%} ({data['correct']}/{data['total']})"
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
            "bbh/accuracy": metrics.get("accuracy", 0),
            "bbh/total_evaluated": metrics.get("total_evaluated", 0),
            "bbh/num_subsets": metrics.get("num_subsets", 0),
            "bbh/format_compliance_rate": metrics.get("format_compliance_rate", 0),
            "bbh/thinking_utilization_rate": metrics.get(
                "thinking_utilization_rate", 0
            ),
        }

        # Log per-subset accuracies
        for subset, data in metrics.get("subset_metrics", {}).items():
            safe_name = subset.replace(" ", "_")[:40]
            log_dict[f"bbh/accuracy_{safe_name}"] = data.get("accuracy", 0)

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
    BBHEvalEnv.cli()
