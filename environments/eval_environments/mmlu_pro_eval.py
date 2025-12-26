"""
MMLU-Pro Evaluation Environment for Atropos (Generative/Reasoning Mode)

This environment evaluates models on the MMLU-Pro benchmark using a generative
approach where models can reason before answering.

Dataset: TIGER-Lab/MMLU-Pro
Paper: https://arxiv.org/abs/2406.01574

MMLU-Pro is a more robust and challenging massive multi-task understanding
dataset tailored to more rigorously benchmark large language models' capabilities.
This dataset contains 12K complex questions across various disciplines with
10 answer choices instead of 4.

The evaluation follows the lighteval generative approach:
- Models are prompted to "think step by step before answering"
- Models output their reasoning followed by "Answer: X"
- Answer is extracted using regex patterns from the response
- Simple string matching validates the extracted answer

Supports optional thinking mode with <think></think> tags for extended reasoning.
"""

import asyncio
import os
import re
import time
from string import ascii_uppercase
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset
from eval_helpers import (
    create_system_content,
    extract_letter_from_answer_tag,
    get_default_thinking_prompt,
)
from pydantic import Field
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
)

# MMLU-Pro prompt template with <answer> tag instruction
# Note: MMLU-Pro has up to 10 choices (A-J), not just 4
MMLU_PRO_PROMPT_TEMPLATE = """Answer the following multiple choice question. Think step by step before answering.

Provide your final answer within <answer></answer> tags, containing only the letter ({valid_letters}).

Example format:
<answer>A</answer>

{question}

{choices}"""


# MMLU-Pro categories for aggregate metrics
MMLU_PRO_CATEGORIES = [
    "biology",
    "business",
    "chemistry",
    "computer science",
    "economics",
    "engineering",
    "health",
    "history",
    "law",
    "math",
    "philosophy",
    "physics",
    "psychology",
    "other",
]


class MMLUProEvalConfig(BaseEnvConfig):
    """Configuration for MMLU-Pro evaluation environment (generative mode)."""

    # Thinking mode configuration
    thinking_mode: bool = Field(
        default=True,
        description="Whether to enable thinking mode with <think></think> tags.",
    )

    custom_thinking_prompt: Optional[str] = Field(
        default=None,
        description="Custom thinking prompt. If None, uses the default thinking prompt.",
    )

    # Dataset configuration
    dataset_name: str = Field(
        default="TIGER-Lab/MMLU-Pro",
        description="HuggingFace dataset name for MMLU-Pro.",
    )

    eval_split: str = Field(
        default="test",
        description="Dataset split to use for evaluation.",
    )

    few_shot_split: str = Field(
        default="validation",
        description="Dataset split to use for few-shot examples.",
    )

    # Few-shot configuration
    num_few_shot: int = Field(
        default=0,
        ge=0,
        le=5,
        description="Number of few-shot examples to include (0-5 recommended).",
    )

    # Category filtering
    categories: Optional[List[str]] = Field(
        default=None,
        description="List of categories to evaluate. If None, evaluates all categories.",
    )

    # Model generation configuration
    eval_temperature: float = Field(
        default=0.6,
        description="Temperature for evaluation.",
    )

    eval_max_tokens: int = Field(
        default=0,
        description="Maximum tokens for evaluation responses. Set high to allow reasoning.",
    )

    # Prompt configuration
    custom_system_prompt: Optional[str] = Field(
        default=None,
        description="Custom system prompt to append after thinking prompt (if thinking_mode) or use directly.",
    )

    # Retry configuration
    max_retries: int = Field(
        default=3,
        ge=1,
        description="Maximum retries for failed API calls.",
    )

    retry_delay: float = Field(
        default=1.0,
        ge=0.0,
        description="Delay between retry attempts in seconds.",
    )

    min_response_length: int = Field(
        default=1,
        ge=1,
        description="Minimum response length to consider valid.",
    )

    # Debug configuration
    full_debug: bool = Field(
        default=False,
        description="Enable verbose debug logging.",
    )


class MMLUProEvalEnv(BaseEnv):
    """
    MMLU-Pro Evaluation Environment for Atropos (Generative/Reasoning Mode).

    Evaluates models on the MMLU-Pro benchmark using a generative approach where
    models reason before answering complex multi-choice questions.

    Key features:
    - Loads MMLU-Pro dataset from HuggingFace (TIGER-Lab/MMLU-Pro)
    - Uses lighteval's exact prompt format
    - Handles 10-choice questions (A-J)
    - Optional thinking mode with <think></think> tags
    - Tracks per-category accuracy
    - Supports few-shot examples
    """

    name = "mmlu_pro_eval"
    env_config_cls = MMLUProEvalConfig

    def __init__(
        self,
        config: MMLUProEvalConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: MMLUProEvalConfig = config

        # Initialize metrics tracking
        self.eval_metrics = []

        # Pre-compile regex patterns for thinking mode
        self._think_pattern = re.compile(r"<think>")
        self._think_close_pattern = re.compile(r"</think>")
        self._think_content_pattern = re.compile(r"</think>\s*(.*)", re.DOTALL)
        self._thinking_extract_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)

        # Pre-compile regex for <answer></answer> tag extraction (primary method)
        self._answer_tag_pattern = re.compile(
            r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE
        )

        # Build fallback answer extraction patterns (supports A-J for 10 choices)
        self._build_extraction_patterns()

    def _get_thinking_prompt(self) -> str:
        """Get thinking system prompt."""
        return get_default_thinking_prompt(self.config.custom_thinking_prompt)

    def _create_system_content(self) -> Optional[str]:
        """Create system message content based on thinking mode."""
        return create_system_content(
            self.config.thinking_mode,
            self.config.custom_thinking_prompt,
            self.config.custom_system_prompt,
        )

    def _build_extraction_patterns(self):
        """Build regex patterns for extracting answer letters (A-J for 10 choices)."""
        # MMLU-Pro has up to 10 choices (A-J)
        letters = "ABCDEFGHIJ"
        letter_pattern = rf"([{letters}]|\([{letters}]\))"

        self._pattern_final_answer_hope = re.compile(
            rf"(?i:final\s+answer\s+is)\s*:?\s*{letter_pattern}\.?\s*I\s*hope",
            re.IGNORECASE,
        )
        self._pattern_final_answer_is = re.compile(
            rf"(?i:final\s+answer).{{0,100}}?\s+is\s*:?\s*{letter_pattern}",
            re.IGNORECASE | re.DOTALL,
        )
        self._pattern_the_answer_is = re.compile(
            rf"(?i:the\s+answer\s+is)\s*:?\s*{letter_pattern}", re.IGNORECASE
        )
        self._pattern_answer_colon = re.compile(
            rf"(?i:answer)\s*:\s*.{{0,50}}?{letter_pattern}", re.IGNORECASE | re.DOTALL
        )
        self._pattern_answer_space = re.compile(
            rf"(?i:answer)\s+{letter_pattern}", re.IGNORECASE
        )
        self._pattern_start = re.compile(
            rf"^\s*\**{letter_pattern}\**[\s\.\)\:]", re.IGNORECASE
        )
        self._pattern_line_start = re.compile(
            rf"\n\s*\**{letter_pattern}\**[\s\.\)\:]", re.IGNORECASE
        )
        self._pattern_standalone = re.compile(rf"\b{letter_pattern}\b", re.IGNORECASE)

        self._extraction_patterns = [
            (0, self._pattern_final_answer_hope, "final_answer_hope"),
            (50, self._pattern_final_answer_is, "final_answer_is"),
            (75, self._pattern_the_answer_is, "the_answer_is"),
            (100, self._pattern_answer_colon, "answer_colon"),
            (150, self._pattern_answer_space, "answer_space"),
            (200, self._pattern_start, "start"),
            (210, self._pattern_line_start, "line_start"),
            (250, self._pattern_standalone, "standalone"),
        ]

    @classmethod
    def config_init(cls) -> Tuple[MMLUProEvalConfig, List[APIServerConfig]]:
        """Initialize default configuration for the environment."""
        env_config = MMLUProEvalConfig(
            tokenizer_name="NousResearch/Hermes-3-Llama-3.1-8B",
            group_size=1,
            use_wandb=True,
            max_num_workers_per_node=8,
            rollout_server_url="http://localhost:8000",
            total_steps=1,
            batch_size=1,
            steps_per_eval=1,
            inference_weight=1.0,
            wandb_name="mmlu_pro_eval",
            eval_handling=EvalHandlingEnum.STOP_TRAIN,
            # MMLU-Pro specific defaults
            dataset_name="TIGER-Lab/MMLU-Pro",
            num_few_shot=0,
            eval_temperature=0.6,
            eval_max_tokens=0,  # Use model default
            thinking_mode=True,
        )

        server_configs = [
            APIServerConfig(
                model_name="Hermes-3-Llama-3.1-8B",
                base_url="http://localhost:9000/v1",
                api_key=os.getenv("OPENAI_API_KEY", "none"),
                num_max_requests_at_once=32,
                num_requests_for_eval=256,
            ),
        ]

        return env_config, server_configs

    async def setup(self) -> None:
        """Load the MMLU-Pro dataset and prepare for evaluation."""
        print("\nMMLU-Pro Evaluation Setup (Generative Mode):")
        print(f"  Dataset: {self.config.dataset_name}")
        print(f"  Few-shot examples: {self.config.num_few_shot}")
        print(f"  Max tokens for reasoning: {self.config.eval_max_tokens}")
        print(f"  Evaluation split: {self.config.eval_split}")
        print(f"  Thinking mode: {self.config.thinking_mode}")
        if self.config.thinking_mode:
            print(f"  Thinking prompt: {self._get_thinking_prompt()[:100]}...")

        # Load MMLU-Pro dataset
        try:
            dataset = load_dataset(
                self.config.dataset_name,
                split=self.config.eval_split,
            )
            self.eval_data = list(dataset)
            print(f"  Loaded {len(self.eval_data)} evaluation items")

            # Load few-shot data if needed
            if self.config.num_few_shot > 0:
                few_shot_dataset = load_dataset(
                    self.config.dataset_name,
                    split=self.config.few_shot_split,
                )
                self.few_shot_data = list(few_shot_dataset)
                print(f"  Loaded {len(self.few_shot_data)} few-shot examples")
            else:
                self.few_shot_data = []

        except Exception as e:
            print(f"Error loading MMLU-Pro dataset: {e}")
            raise

        # Filter by categories if specified
        if self.config.categories:
            self.eval_data = [
                item
                for item in self.eval_data
                if item.get("category", "").lower()
                in [c.lower() for c in self.config.categories]
            ]
            print(
                f"  Filtered to {len(self.eval_data)} items in categories: {self.config.categories}"
            )

        # Analyze category distribution
        category_counts = {}
        for item in self.eval_data:
            cat = item.get("category", "unknown")
            category_counts[cat] = category_counts.get(cat, 0) + 1

        print("\n  Category distribution:")
        for cat, count in sorted(category_counts.items()):
            print(f"    {cat}: {count} questions")

        self.all_eval_items = self.eval_data
        self.iter = 0

    def _format_choices(self, options: List[str]) -> str:
        """Format choices as A: choice1, B: choice2, etc. (MMLU-Pro format)."""
        lines = []
        for idx, option in enumerate(options):
            letter = ascii_uppercase[idx]
            lines.append(f"{letter}: {option}")
        return "\n".join(lines)

    def _format_mmlu_pro_prompt(
        self,
        question: str,
        options: List[str],
        few_shot_examples: Optional[List[Dict]] = None,
    ) -> str:
        """
        Format a question using the lighteval MMLU-Pro template.

        Uses the exact prompt format from lighteval's mmlu_pro_prompt_function.
        """
        num_choices = len(options)
        valid_letters = "".join(ascii_uppercase[:num_choices])

        # Format choices
        formatted_choices = self._format_choices(options)

        # Use lighteval's exact template
        prompt = MMLU_PRO_PROMPT_TEMPLATE.format(
            question=question,
            choices=formatted_choices,
            valid_letters=valid_letters,
        )

        # Add few-shot examples if provided
        if few_shot_examples:
            few_shot_text = self._format_few_shot_examples(few_shot_examples)
            prompt = few_shot_text + "\n\n---\n\n" + prompt

        return prompt

    def _format_few_shot_examples(self, examples: List[Dict]) -> str:
        """Format few-shot examples with answers for context."""
        formatted = []
        for example in examples:
            question = example.get("question", "")
            options = example.get("options", [])
            answer_index = example.get("answer_index", 0)

            answer_letter = ascii_uppercase[answer_index]
            formatted_choices = self._format_choices(options)

            example_text = (
                f"Question: {question}\n{formatted_choices}\n\nAnswer: {answer_letter}"
            )
            formatted.append(example_text)

        return "\n\n---\n\n".join(formatted)

    def _validate_thinking_format(self, response: str) -> Tuple[bool, str]:
        """Validate thinking format and extract content after </think> tags."""
        if not self.config.thinking_mode:
            return True, response

        think_open_count = len(self._think_pattern.findall(response))
        think_close_count = len(self._think_close_pattern.findall(response))

        if think_open_count != 1 or think_close_count != 1:
            return False, response

        match = self._think_content_pattern.search(response)
        if match:
            return True, match.group(1).strip()
        else:
            return False, response

    def _extract_thinking_content(self, response: str) -> Optional[str]:
        """Extract the content inside <think></think> tags."""
        match = self._thinking_extract_pattern.search(response)
        if match:
            return match.group(1).strip()
        return None

    def _extract_answer(
        self, response: str, num_choices: int = 10, choices: Optional[List[str]] = None
    ) -> Tuple[Optional[str], str]:
        """
        Extract the answer letter from the model's response.

        Primary method: Look for <answer></answer> tags, or match against choice texts.
        Fallback: Use priority-ordered regex patterns.
        """
        if not response:
            return None, "empty_response"

        valid_letters = set(ascii_uppercase[:num_choices])

        # PRIMARY: Try <answer></answer> tags first
        # Also matches against choice texts if provided
        letter, method = extract_letter_from_answer_tag(
            response, valid_letters, debug=self.config.full_debug, choices=choices
        )
        if letter:
            return letter, method

        # FALLBACK: Try each pattern in priority order
        for priority, pattern, method_name in self._extraction_patterns:
            matches = pattern.findall(response)
            if matches:
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
                letter = match.strip("()").upper()

                if letter in valid_letters:
                    if self.config.full_debug:
                        print(
                            f"    Extracted '{letter}' using fallback method '{method_name}'"
                        )
                    return letter, f"fallback_{method_name}"

        for letter in reversed(list(valid_letters)):
            if letter in response.upper():
                if self.config.full_debug:
                    print(
                        f"    Extracted '{letter}' using fallback 'last_valid_letter'"
                    )
                return letter, "fallback_last_valid_letter"

        return None, "no_match"

    async def get_next_item(self):
        """Get next item for training (not used in eval-only environment)."""
        self.iter += 1
        if self.all_eval_items:
            item = self.all_eval_items[self.iter % len(self.all_eval_items)]
            return item
        return None

    async def collect_trajectories(self, item):
        """Collect trajectories (not used in eval-only environment)."""
        return None, []

    async def score(self, rollout_group_data):
        """Score rollouts (not used in eval-only environment)."""
        return None

    async def rollout_and_score_eval(self, eval_item: Dict) -> Dict:
        """Evaluate a single MMLU-Pro question using generative mode."""
        try:
            question = eval_item.get("question", "")
            options = eval_item.get("options", [])
            answer_index = eval_item.get("answer_index", 0)
            category = eval_item.get("category", "unknown")

            num_choices = len(options)
            gold_letter = ascii_uppercase[answer_index]

            if not question or num_choices < 2:
                return {"is_correct": None, "sample": None}

            # Get few-shot examples
            few_shot_examples = None
            if self.config.num_few_shot > 0 and self.few_shot_data:
                # Get examples from the same category if possible
                same_cat_examples = [
                    ex for ex in self.few_shot_data if ex.get("category") == category
                ]
                if len(same_cat_examples) >= self.config.num_few_shot:
                    few_shot_examples = same_cat_examples[: self.config.num_few_shot]
                else:
                    few_shot_examples = self.few_shot_data[: self.config.num_few_shot]

            # Format the prompt (lighteval style)
            formatted_prompt = self._format_mmlu_pro_prompt(
                question=question,
                options=options,
                few_shot_examples=few_shot_examples,
            )

            # Build messages
            messages = []
            system_content = self._create_system_content()
            if system_content:
                messages.append({"role": "system", "content": system_content})
            messages.append({"role": "user", "content": formatted_prompt})

            # Get model completion with retry logic
            model_response = None
            finish_reason = None

            # Build completion kwargs - only include max_tokens if > 0
            # (0 means "use model default", so we don't pass the parameter)
            completion_kwargs = {
                "messages": messages,
                "n": 1,
                "temperature": self.config.eval_temperature,
                "split": "eval",
            }
            if self.config.eval_max_tokens > 0:
                completion_kwargs["max_tokens"] = self.config.eval_max_tokens

            for attempt in range(self.config.max_retries):
                try:
                    completion = await self.server.chat_completion(**completion_kwargs)

                    if completion.choices and completion.choices[0].message.content:
                        model_response = completion.choices[0].message.content
                        finish_reason = getattr(
                            completion.choices[0], "finish_reason", None
                        )

                        if (
                            len(model_response.strip())
                            >= self.config.min_response_length
                        ):
                            break
                        elif attempt < self.config.max_retries - 1:
                            if self.config.full_debug:
                                print("  Response too short, retrying...")
                            await asyncio.sleep(self.config.retry_delay)

                except Exception as e:
                    # Always log API errors to help diagnose issues
                    print(
                        f"  API Error (attempt {attempt + 1}/{self.config.max_retries}): {type(e).__name__}: {e}"
                    )
                    if hasattr(e, "response"):
                        try:
                            print(
                                f"    Response: {e.response.text[:500] if hasattr(e.response, 'text') else e.response}"
                            )
                        except Exception:
                            pass
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(self.config.retry_delay)
                    else:
                        print(f"  Failed after {self.config.max_retries} attempts")
                        return {"is_correct": None, "sample": None}

            if not model_response:
                return {"is_correct": None, "sample": None}

            # Validate thinking format if enabled
            format_valid, content_for_extraction = self._validate_thinking_format(
                model_response
            )

            # Extract thinking content for logging
            thinking_content = None
            if self.config.thinking_mode:
                thinking_content = self._extract_thinking_content(model_response)

            # Extract the answer (pass choices for exact text matching)
            extracted_answer, extraction_method = self._extract_answer(
                content_for_extraction, num_choices, choices=options
            )

            # Check if correct
            is_correct = extracted_answer == gold_letter if extracted_answer else False

            # Build sample record
            sample = {
                "question": question,
                "options": options,
                "gold_answer": gold_letter,
                "model_response": model_response,
                "extracted_answer": extracted_answer,
                "extraction_method": extraction_method,
                "is_correct": is_correct,
                "category": category,
                "num_choices": num_choices,
                "num_few_shot": self.config.num_few_shot,
                "finish_reason": finish_reason,
                "response_length": len(model_response),
                "thinking_mode": self.config.thinking_mode,
                "format_valid": format_valid,
            }

            if self.config.thinking_mode:
                sample["thinking_content"] = thinking_content
                sample["response_after_think"] = (
                    content_for_extraction if format_valid else None
                )

            if self.config.full_debug:
                status = "✓" if is_correct else "✗"
                print(
                    f"  [{status}] {category}: gold={gold_letter}, extracted={extracted_answer}"
                )

            return {"is_correct": is_correct, "sample": sample}

        except Exception as e:
            if self.config.full_debug:
                print(f"Error in rollout_and_score_eval: {e}")
                import traceback

                traceback.print_exc()
            return {"is_correct": None, "sample": None}

    async def evaluate(self, *args, **kwargs) -> None:
        """Run MMLU-Pro evaluation."""
        start_time = time.time()

        print(f"\n{'='*60}")
        print("Starting MMLU-Pro Evaluation (Generative/Reasoning Mode)")
        print(f"{'='*60}")
        print(f"  Total questions: {len(self.all_eval_items)}")
        print(f"  Few-shot examples: {self.config.num_few_shot}")
        print(f"  Max tokens (for reasoning): {self.config.eval_max_tokens}")
        print(f"  Thinking mode: {self.config.thinking_mode}")
        print(f"{'='*60}\n")

        try:
            eval_tasks = [
                self.rollout_and_score_eval(item) for item in self.all_eval_items
            ]
            results = await tqdm_asyncio.gather(*eval_tasks, desc="Evaluating MMLU-Pro")

            valid_results = [
                r
                for r in results
                if r and r.get("sample") is not None and r.get("is_correct") is not None
            ]

            if not valid_results:
                print("Warning: No valid evaluation results obtained")
                return

        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback

            traceback.print_exc()
            return

        end_time = time.time()

        # Compute metrics
        samples = [r["sample"] for r in valid_results]

        # Overall accuracy
        total_correct = sum(1 for r in valid_results if r["is_correct"])
        total_count = len(valid_results)
        overall_accuracy = total_correct / total_count if total_count > 0 else 0.0

        # Per-category accuracy
        category_results = {}
        for sample in samples:
            category = sample.get("category", "unknown")
            if category not in category_results:
                category_results[category] = {"correct": 0, "total": 0}
            category_results[category]["total"] += 1
            if sample["is_correct"]:
                category_results[category]["correct"] += 1

        # Extraction method statistics
        extraction_methods = {}
        for sample in samples:
            method = sample.get("extraction_method", "unknown")
            if method not in extraction_methods:
                extraction_methods[method] = {"count": 0, "correct": 0}
            extraction_methods[method]["count"] += 1
            if sample["is_correct"]:
                extraction_methods[method]["correct"] += 1

        # Average response length
        response_lengths = [s.get("response_length", 0) for s in samples]
        avg_response_length = (
            sum(response_lengths) / len(response_lengths) if response_lengths else 0
        )

        # Format compliance
        format_compliant = sum(1 for s in samples if s.get("format_valid", True))
        format_compliance_rate = format_compliant / len(samples) if samples else 0.0

        # Thinking utilization
        thinking_utilization = 0
        if self.config.thinking_mode:
            thinking_utilization = sum(1 for s in samples if s.get("thinking_content"))

        # Build metrics dictionary
        eval_metrics = {
            "eval/overall_accuracy": overall_accuracy,
            "eval/total_questions": total_count,
            "eval/total_correct": total_correct,
            "eval/num_categories": len(category_results),
            "eval/num_few_shot": self.config.num_few_shot,
            "eval/evaluation_time_seconds": end_time - start_time,
            "eval/avg_response_length": avg_response_length,
            "eval/format_compliance_rate": format_compliance_rate,
            "eval/thinking_mode_enabled": 1.0 if self.config.thinking_mode else 0.0,
        }

        if self.config.thinking_mode:
            thinking_utilization_rate = (
                thinking_utilization / len(samples) if samples else 0.0
            )
            eval_metrics["eval/thinking_utilization_rate"] = thinking_utilization_rate

        # Add category metrics
        for category, stats in category_results.items():
            if stats["total"] > 0:
                cat_accuracy = stats["correct"] / stats["total"]
                cat_key = category.replace(" ", "_").replace("-", "_").lower()
                eval_metrics[f"eval/category_{cat_key}_accuracy"] = cat_accuracy
                eval_metrics[f"eval/category_{cat_key}_total"] = stats["total"]

        # Add extraction method metrics
        for method, stats in extraction_methods.items():
            if stats["count"] > 0:
                method_accuracy = stats["correct"] / stats["count"]
                eval_metrics[f"eval/extraction_{method}_count"] = stats["count"]
                eval_metrics[f"eval/extraction_{method}_accuracy"] = method_accuracy

        # Store metrics for wandb logging
        self.eval_metrics = [(k, v) for k, v in eval_metrics.items()]

        # Print summary
        print(f"\n{'='*60}")
        print("MMLU-Pro Evaluation Results")
        print(f"{'='*60}")
        print(
            f"Overall Accuracy: {overall_accuracy:.4f} ({total_correct}/{total_count})"
        )
        print(f"Evaluation Time: {end_time - start_time:.1f} seconds")
        print(f"Avg Response Length: {avg_response_length:.0f} chars")
        if self.config.thinking_mode:
            print(f"Format Compliance: {format_compliance_rate:.4f}")
            print(f"Thinking Utilization: {thinking_utilization}/{total_count}")

        print("\nCategory Breakdown:")
        for category, stats in sorted(category_results.items()):
            if stats["total"] > 0:
                cat_acc = stats["correct"] / stats["total"]
                print(
                    f"  {category}: {cat_acc:.4f} ({stats['correct']}/{stats['total']})"
                )

        print("\nExtraction Method Statistics:")
        for method, stats in sorted(
            extraction_methods.items(), key=lambda x: -x[1]["count"]
        ):
            if stats["count"] > 0:
                method_acc = stats["correct"] / stats["count"]
                print(f"  {method}: {stats['count']} uses, {method_acc:.4f} accuracy")

        print(f"{'='*60}\n")

        # Log evaluation results
        try:
            await self.evaluate_log(
                metrics=eval_metrics,
                samples=samples,
                start_time=start_time,
                end_time=end_time,
                generation_parameters={
                    "temperature": self.config.eval_temperature,
                    "max_tokens": self.config.eval_max_tokens,
                    "num_few_shot": self.config.num_few_shot,
                    "thinking_mode": self.config.thinking_mode,
                    "mode": "generative",
                },
            )
        except Exception as e:
            print(f"Error logging evaluation results: {e}")

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Log metrics to wandb."""
        if wandb_metrics is None:
            wandb_metrics = {}

        for metric_name, metric_value in self.eval_metrics:
            wandb_metrics[metric_name] = metric_value
        self.eval_metrics = []

        wandb_metrics["config/thinking_mode"] = (
            1.0 if self.config.thinking_mode else 0.0
        )
        wandb_metrics["config/num_few_shot"] = self.config.num_few_shot
        wandb_metrics["config/eval_max_tokens"] = self.config.eval_max_tokens

        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    MMLUProEvalEnv.cli()
