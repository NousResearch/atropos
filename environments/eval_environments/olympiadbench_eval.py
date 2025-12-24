"""
OlympiadBench Evaluation Environment for Atropos (Generative Mode)

This environment evaluates models on OlympiadBench - a benchmark for evaluating
language models on olympiad-level math and physics problems.

Dataset: Hothan/OlympiadBench
Paper: https://arxiv.org/abs/2402.14008

The evaluation follows a generative approach:
- Models receive challenging olympiad problems (Math or Physics)
- Expected answers can be numerical, expressions, equations, or intervals
- Answers should be provided in LaTeX format with \\boxed{answer}
- Supports thinking mode with <think></think> tags for extended reasoning

Note: This implementation supports text-only (TO) problems in both English and Chinese.
Theorem proving (TP) problems are not included as they require different evaluation.
"""

import asyncio
import os
import random
import re
import time
from typing import Dict, List, Optional, Tuple

import wandb
from datasets import load_dataset
from eval_helpers import (
    ANSWER_TAG_PATTERN,
    create_system_content,
    extract_thinking_content,
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
    EvalHandlingEnum,
)

# Available text-only subsets in OlympiadBench
AVAILABLE_SUBSETS = [
    "OE_TO_maths_en_COMP",  # Open-ended, text-only, maths, English, Competition
    "OE_TO_maths_zh_CEE",  # Open-ended, text-only, maths, Chinese, College Entrance Exam
    "OE_TO_maths_zh_COMP",  # Open-ended, text-only, maths, Chinese, Competition
    "OE_TO_physics_en_COMP",  # Open-ended, text-only, physics, English, Competition
    "OE_TO_physics_zh_CEE",  # Open-ended, text-only, physics, Chinese, College Entrance Exam
]

# Answer type descriptions (English)
ANSWER_TYPE_TEXT_EN = {
    "Numerical": "a numerical value",
    "Expression": "an expression",
    "Equation": "an equation",
    "Interval": "an interval",
}

# Answer type descriptions (Chinese)
ANSWER_TYPE_TEXT_ZH = {
    "Numerical": "数值",
    "Expression": "表达式",
    "Equation": "方程",
    "Interval": "区间",
}


def get_answer_type_text(answer_type: str, is_chinese: bool, is_multiple: bool) -> str:
    """Generate answer type instruction text."""
    if "Need_human_evaluate" in answer_type or "Tuple" in answer_type:
        return ""

    type_dict = ANSWER_TYPE_TEXT_ZH if is_chinese else ANSWER_TYPE_TEXT_EN

    # Parse answer type
    single_type = None
    for t in ["Numerical", "Expression", "Equation", "Interval"]:
        if t in answer_type:
            single_type = type_dict[t]
            break

    if not single_type:
        return ""

    if is_chinese:
        if is_multiple:
            return f"，题目有多个答案，答案类型均为{single_type}"
        else:
            return f"，答案类型为{single_type}"
    else:
        if is_multiple:
            return f"The problem has multiple answers, each of them should be {single_type}. "
        else:
            return f"The answer should be {single_type}. "


class OlympiadBenchEvalConfig(BaseEnvConfig):
    """Configuration for OlympiadBench evaluation environment."""

    # Dataset configuration
    dataset_name: str = Field(
        default="Hothan/OlympiadBench", description="HuggingFace dataset name"
    )
    subset: str = Field(
        default="OE_TO_maths_en_COMP",
        description="Dataset subset (see AVAILABLE_SUBSETS)",
    )
    eval_split: str = Field(
        default="train",
        description="Split to evaluate on (train is the only split available)",
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
    wandb_name: str = "olympiadbench_eval"
    steps_per_eval: int = 1


class OlympiadBenchEvalEnv(BaseEnv):
    """
    OlympiadBench Evaluation Environment.

    Evaluates models on olympiad-level math and physics problems.
    Uses generative evaluation with LaTeX boxed answers.
    """

    name = "olympiadbench_eval"

    def __init__(
        self,
        config: OlympiadBenchEvalConfig,
        server_configs: List[APIServerConfig],
        slurm_job_id: Optional[str] = None,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm_job_id, testing)
        self.config: OlympiadBenchEvalConfig = config
        self.eval_items: List[Dict] = []
        self._dataset_loaded = False

        # Pre-compile regex patterns for answer extraction
        self._boxed_pattern = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
        self._answer_tag_pattern = ANSWER_TAG_PATTERN

    @classmethod
    def config_cls(cls) -> type:
        return OlympiadBenchEvalConfig

    async def setup(self) -> None:
        """Initialize the environment and load the dataset."""
        await super().setup()

        if not self._dataset_loaded:
            await self._load_dataset()

        print(f"\nOlympiadBench Evaluation Setup (Generative Mode):")
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
        """Load and process the OlympiadBench dataset."""
        print(
            f"Loading OlympiadBench dataset: {self.config.dataset_name}/{self.config.subset}..."
        )

        if self.config.subset not in AVAILABLE_SUBSETS:
            print(
                f"Warning: Subset '{self.config.subset}' may not be text-only. Available text-only subsets: {AVAILABLE_SUBSETS}"
            )

        try:
            dataset = load_dataset(
                self.config.dataset_name, self.config.subset, trust_remote_code=True
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

        # Parse subset info
        is_chinese = "_zh_" in self.config.subset
        is_math = "maths" in self.config.subset or "Math" in self.config.subset

        # Process items
        self.eval_items = []
        for idx, item in enumerate(split_data):
            question = item.get("question", "")
            final_answer = item.get("final_answer", "")

            if not question or not final_answer:
                continue

            # Get metadata
            subject = item.get("subject", "Math" if is_math else "Physics")
            language = item.get("language", "Chinese" if is_chinese else "English")
            answer_type = item.get("answer_type", "")
            is_multiple = item.get("is_multiple_answer", False)
            unit = item.get("unit", "")

            self.eval_items.append(
                {
                    "id": str(idx),
                    "question": question,
                    "answer": final_answer,
                    "subject": subject,
                    "language": language,
                    "answer_type": answer_type,
                    "is_multiple_answer": is_multiple,
                    "unit": unit,
                    "is_chinese": "Chinese" in language,
                }
            )

        # Shuffle with seed
        random.seed(self.config.shuffle_seed)
        random.shuffle(self.eval_items)

        self._dataset_loaded = True
        print(f"Loaded {len(self.eval_items)} evaluation items")

    def _format_prompt(self, item: Dict) -> str:
        """Format the problem into a prompt with appropriate instructions."""
        is_chinese = item.get("is_chinese", False)
        is_math = "Math" in item.get("subject", "Math")
        is_multiple = item.get("is_multiple_answer", False)
        answer_type = item.get("answer_type", "")
        unit = item.get("unit", "")

        if is_chinese:
            subject_content = "数学" if is_math else "物理"
            answer_type_text = get_answer_type_text(
                answer_type, is_chinese=True, is_multiple=is_multiple
            )

            if is_multiple:
                multiple_answer_text = "\\boxed{用英文逗号连接的多个答案}"
            else:
                multiple_answer_text = "\\boxed{答案}"

            unit_text = ""
            if unit:
                multiple_answer_text += "(单位)"
                unit_text = "，注意答案的单位不要放在\\boxed{}中"

            instruction = f"以下是{subject_content}竞赛中的解答题{answer_type_text}。请根据题目的要求和所提供的信息计算得出答案。解答过程和结果中使用的变量和公式请使用LaTeX格式表示。"
            instruction += f"\n\n请将你的最终答案放在<answer></answer>标签中，格式为{multiple_answer_text}{unit_text}。"
            instruction += "\n\n示例格式:\n<answer>\\boxed{42}</answer>"
        else:
            subject = "Math" if is_math else "Physics"
            answer_type_text = get_answer_type_text(
                answer_type, is_chinese=False, is_multiple=is_multiple
            )

            if is_multiple:
                multiple_answer_text = "\\boxed{multiple answers connected with commas}"
            else:
                multiple_answer_text = "\\boxed{answer}"

            unit_text = ""
            if unit:
                multiple_answer_text += "(unit)"
                unit_text = ", note that the unit of the answer should not be included in \\boxed{}"

            instruction = f"The following is an open-ended problem from an International {subject} competition. {answer_type_text}Please calculate the answer according to the given requirements and the information provided. Please use LaTeX format to represent the variables and formulas used in the solution process and results."
            instruction += f"\n\nProvide your final answer within <answer></answer> tags in the format {multiple_answer_text}{unit_text}."
            instruction += "\n\nExample format:\n<answer>\\boxed{42}</answer>"

        return f"{instruction}\n\n{item['question']}"

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

    def _normalize_answer(self, answer: str) -> str:
        """Normalize an answer for comparison."""
        if not answer:
            return ""

        # Remove outer whitespace
        normalized = answer.strip()

        # Remove \boxed{} wrapper if present
        boxed_match = self._boxed_pattern.search(normalized)
        if boxed_match:
            normalized = boxed_match.group(1)

        # Normalize whitespace
        normalized = " ".join(normalized.split())

        # Remove common LaTeX spacing commands
        normalized = re.sub(r"\\[,;:!]", "", normalized)
        normalized = re.sub(r"\\quad|\\qquad", " ", normalized)

        # Normalize common math equivalents
        normalized = normalized.replace("\\times", "*")
        normalized = normalized.replace("\\cdot", "*")
        normalized = normalized.replace("\\div", "/")

        return normalized.strip()

    def _check_match(self, predicted: str, gold: str) -> Tuple[bool, str]:
        """
        Check if the predicted answer matches the gold answer.

        Handles multiple answers (comma-separated) and various formats.
        """
        pred_norm = self._normalize_answer(predicted)
        gold_norm = self._normalize_answer(gold)

        if not pred_norm:
            return False, "empty_prediction"

        # Exact match
        if pred_norm == gold_norm:
            return True, "exact"

        # Case-insensitive match
        if pred_norm.lower() == gold_norm.lower():
            return True, "case_insensitive"

        # Try numeric comparison
        try:
            pred_num = float(pred_norm.replace(",", ""))
            gold_num = float(gold_norm.replace(",", ""))
            if abs(pred_num - gold_num) < 1e-9:
                return True, "numeric_exact"
            # Allow small relative error for floating point
            if gold_num != 0 and abs(pred_num - gold_num) / abs(gold_num) < 1e-6:
                return True, "numeric_approx"
        except (ValueError, TypeError):
            pass

        # Handle multiple answers (comma-separated)
        if "," in gold_norm:
            gold_parts = set(p.strip() for p in gold_norm.split(","))
            pred_parts = set(p.strip() for p in pred_norm.split(","))
            if gold_parts == pred_parts:
                return True, "multi_answer_set"

        # Containment check (gold in prediction)
        if gold_norm and gold_norm in pred_norm:
            return True, "gold_contained"

        return False, "no_match"

    def _extract_answer(
        self, response: str, debug: bool = False
    ) -> Tuple[Optional[str], str]:
        """
        Extract the answer from the response.

        Looks for <answer></answer> tags first, then \\boxed{}.
        """
        if not response:
            return None, "empty_response"

        # Try <answer></answer> tags first
        answer_tag_match = self._answer_tag_pattern.search(response)
        if answer_tag_match:
            answer_content = answer_tag_match.group(1).strip()
            if answer_content:
                # Check if there's a boxed inside
                boxed_match = self._boxed_pattern.search(answer_content)
                if boxed_match:
                    extracted = boxed_match.group(1)
                    if debug:
                        print(
                            f"    Extracted '{extracted}' from boxed inside answer tag"
                        )
                    return extracted, "answer_tag_boxed"
                else:
                    if debug:
                        print(f"    Extracted '{answer_content}' from answer tag")
                    return answer_content, "answer_tag"

        # Fallback: Look for \boxed{} anywhere in response
        boxed_matches = self._boxed_pattern.findall(response)
        if boxed_matches:
            # Take the last boxed answer (most likely to be the final answer)
            extracted = boxed_matches[-1]
            if debug:
                print(f"    Extracted '{extracted}' from boxed fallback")
            return extracted, "boxed_fallback"

        # Last resort: Look for "answer is X" patterns
        patterns = [
            r"(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*(.+?)(?:\n|$)",
            r"(?:so\s+)?the\s+answer\s+is\s*:?\s*(.+?)(?:\n|$)",
            r"=\s*([^\n=]+?)(?:\n|$)",  # Last equation result
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                if answer and len(answer) < 100:  # Sanity check length
                    if debug:
                        print(f"    Extracted '{answer}' from pattern fallback")
                    return answer, "pattern_fallback"

        return None, "no_match"

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
            content_for_extraction, debug=self.config.full_debug
        )

        # Check match
        gold_answer = item["answer"]
        is_correct, match_type = (
            self._check_match(extracted_answer, gold_answer)
            if extracted_answer
            else (False, "no_extraction")
        )

        if self.config.full_debug:
            print(f"\n--- Item: {item['id']} ---")
            print(f"Question: {item['question'][:100]}...")
            print(f"Gold answer: {gold_answer}")
            print(f"Extracted: {extracted_answer}")
            print(f"Match: {is_correct} ({match_type})")

        return {
            "item_id": item["id"],
            "question": item["question"],
            "subject": item.get("subject", ""),
            "language": item.get("language", ""),
            "answer_type": item.get("answer_type", ""),
            "gold_answer": gold_answer,
            "extracted_answer": extracted_answer,
            "extraction_method": extraction_method,
            "is_correct": is_correct,
            "match_type": match_type,
            "format_valid": is_valid_format,
            "response": response_text,
            "thinking_content": thinking_content,
            "has_thinking": thinking_content is not None,
        }

    async def evaluate(self, *args, **kwargs) -> Dict:
        """Run the full OlympiadBench evaluation."""
        print(f"\n{'='*60}")
        print("Starting OlympiadBench Evaluation (Generative Mode)")
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
        results = await tqdm_asyncio.gather(*tasks, desc="Evaluating OlympiadBench")

        # Filter out failed results
        valid_results = [r for r in results if r is not None]

        if not valid_results:
            print("Warning: No valid evaluation results obtained")
            return {"error": "No valid results", "accuracy": 0.0}

        # Calculate metrics
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

        # Calculate per-answer-type metrics
        type_metrics = {}
        for r in valid_results:
            atype = r.get("answer_type", "unknown") or "unknown"
            if atype not in type_metrics:
                type_metrics[atype] = {"total": 0, "correct": 0}
            type_metrics[atype]["total"] += 1
            if r["is_correct"]:
                type_metrics[atype]["correct"] += 1

        for atype in type_metrics:
            t_total = type_metrics[atype]["total"]
            t_correct = type_metrics[atype]["correct"]
            type_metrics[atype]["accuracy"] = (
                t_correct / t_total if t_total > 0 else 0.0
            )

        # Format compliance and thinking utilization
        format_valid = sum(1 for r in valid_results if r.get("format_valid", True))
        has_thinking = sum(1 for r in valid_results if r.get("has_thinking", False))

        metrics = {
            "accuracy": accuracy,
            "total_evaluated": total,
            "total_correct": correct,
            "subset": self.config.subset,
            "format_compliance_rate": format_valid / total if total > 0 else 0.0,
            "thinking_utilization_rate": has_thinking / total if total > 0 else 0.0,
            "subject_metrics": subject_metrics,
            "answer_type_metrics": type_metrics,
        }

        print(f"\n{'='*60}")
        print("OlympiadBench Evaluation Results")
        print(f"{'='*60}")
        print(f"  Subset: {self.config.subset}")
        print(f"  Overall Accuracy: {accuracy:.2%} ({correct}/{total})")
        print(f"  Format Compliance: {format_valid / total:.2%}")
        if self.config.thinking_mode:
            print(f"  Thinking Utilization: {has_thinking / total:.2%}")
        if subject_metrics:
            print(f"\n  Per-Subject Breakdown:")
            for subject, data in subject_metrics.items():
                print(
                    f"    {subject}: {data['accuracy']:.2%} ({data['correct']}/{data['total']})"
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
            "olympiadbench/accuracy": metrics.get("accuracy", 0),
            "olympiadbench/total_evaluated": metrics.get("total_evaluated", 0),
            "olympiadbench/format_compliance_rate": metrics.get(
                "format_compliance_rate", 0
            ),
            "olympiadbench/thinking_utilization_rate": metrics.get(
                "thinking_utilization_rate", 0
            ),
        }

        # Log per-subject accuracies
        for subject, data in metrics.get("subject_metrics", {}).items():
            safe_name = subject.replace(" ", "_")[:30]
            log_dict[f"olympiadbench/accuracy_{safe_name}"] = data.get("accuracy", 0)

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
    OlympiadBenchEvalEnv.cli()
