"""
SuperGPQA Evaluation Environment for Atropos

This environment evaluates models on the SuperGPQA benchmark.

Dataset: m-a-p/SuperGPQA
Paper: https://www.arxiv.org/pdf/2502.14739

SuperGPQA is a comprehensive benchmark designed to evaluate the knowledge and reasoning abilities of Large Language Models (LLMs) across 285 graduate-level disciplines. 
It features at least 50 questions per discipline, covering a broad spectrum of graduate-level topics.
"""

import asyncio
import os
import random
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


SUPER_GPQA_ZERO_SHOT_PROMPT = """Answer the following multiple-choice question. There is only one correct answer.
Provide your final answer within <answer></answer> tags, containing only the letter (A, B, C, D, E, F, G, H, I, or J.).

Example format:
<answer>A</answer>

Question: {Question}"""


class SuperGPQAEvalConfig(BaseEnvConfig):
    """Configuration for SuperGPQA eval environment"""
    custom_system_prompt: Optional[str] = Field(
        default=None,
        description="Custom system prompt to append after thinking prompt (if thinking_mode) or use directly.",
    )

    custom_thinking_prompt: Optional[str] = Field(
        default=None,
        description="Custom thinking prompt. If None, uses the default thinking prompt.",
    )

    thinking_mode: bool = Field(
        default=False, # set to False if eval_model_type is 'base'
        description="Whether to enable thinking mode with <think></think> tags.",
    )

    dataset_name: str = Field(
        default="m-a-p/SuperGPQA",
        description="HuggingFace dataset for SuperGPQA.",
    )

    eval_split: str = Field(
        default="train",
        description="Dataset split to use for evaluation (SuperGPQA has a single split 'train' used for evaluation).",
    )

    eval_max_tokens: int = Field(
        default=32000,
        description="Maximum tokens for reasoning models.",
    )

    min_response_length: int = Field(
        default=1,
        ge=1,
        description="Min length for a valid response.",
    )

    # all main results in supergpqa use temperature=0
    eval_temperature: float = Field(
        default=0,
        description="Temperature for evaluation (0.0 for deterministic).",
    )

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

    full_debug: bool = Field(
        default=False,
        description="Enable verbose debug logging.",
    )


class SuperGPQAEvalEnv(BaseEnv):
    """
    SuperGPQA eval environment for Atropos, supporting reasoning/instruct models via a 0-shot prompt.

    Pipeline:
    - Load SuperGPQA from HuggingFace
    - Use 0-shot prompts from the SuperGPQA paper
    - Extract answer choice via regex
    """

    name = "supergpqa_eval"
    env_config_cls = SuperGPQAEvalConfig

    def __init__(
        self,
        config: SuperGPQAEvalConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: SuperGPQAEvalConfig = config

        self.eval_metrics = []
        if self.config.shuffle_seed is not None:
            self.shuffle_rng = random.Random(self.config.shuffle_seed)
        else:
            self.shuffle_rng = random.Random()

        self._think_pattern = re.compile(r"<think>")
        self._think_close_pattern = re.compile(r"</think>")
        self._think_content_pattern = re.compile(r"</think>\s*(.*)", re.DOTALL)
        self._thinking_extract_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        self._answer_pattern = re.compile(
            r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE
        )

        # Build fallback answer extraction patterns
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
        """Build regex patterns for extracting answer letters from model responses."""
        letters = "ABCDEFGHIJ"
        letter_pattern = rf"([{letters}]|\([{letters}]\))"

        # Patterns ordered by priority (most specific first)
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
    def config_init(cls) -> Tuple[SuperGPQAEvalConfig, List[APIServerConfig]]:
        """Initialize default configuration for the environment."""
        env_config = SuperGPQAEvalConfig(
            tokenizer_name="NousResearch/Hermes-3-Llama-3.1-8B",
            group_size=1,
            use_wandb=True,
            max_num_workers_per_node=128,
            rollout_server_url="http://localhost:8000",
            total_steps=1,
            batch_size=1,
            steps_per_eval=1,
            inference_weight=1.0,
            wandb_name="super_gpqa_eval",
            eval_handling=EvalHandlingEnum.STOP_TRAIN,
            max_eval_workers=256,
            max_num_workers=1024,
            dataset_name="m-a-p/SuperGPQA",
            eval_temperature=0,
            eval_max_tokens=0,  # this uses the default model max
            thinking_mode=False,
        )

        server_configs = [
            APIServerConfig(
                model_name="Hermes-3-Llama-3.1-8B",
                base_url="http://localhost:9000/v1",
                api_key=os.getenv("OPENAI_API_KEY", "none"),
                num_max_requests_at_once=32,
                num_requests_for_eval=1024,
            ),
        ]

        return env_config, server_configs

    async def setup(self) -> None:
        """Load SuperGPQA dataset and process all points to create prompts for rollout generation."""
        print("\nSuperGPQA Evaluation Setup:")
        print("="*20 + "DATASET DETAILS" + "="*20)
        print(f"  Dataset: {self.config.dataset_name}")
        print(f"  Evaluation split: {self.config.eval_split}")
        print("="*20 + "GENERATION DETAILS" + "="*20)
        print(f"  Thinking mode: {self.config.thinking_mode}")
        print(f"  Max tokens: {self.config.eval_max_tokens}")

        try:
            dataset = load_dataset(
                self.config.dataset_name,
                split=self.config.eval_split,
            )
            self.eval_data = list(dataset)
            print(f"  Loaded {len(self.eval_data)} evaluation items")
        except Exception as e:
            print(f"Error loading dataset from HuggingFace: {e}")
            raise

        self.all_eval_items = []
        for item in self.eval_data:
            processed = self._process_super_gpqa_item(item)
            self.all_eval_items.append(processed)

        self.iter = 0

    def _process_super_gpqa_item(self, item: Dict) -> Dict:
        # check if shuffling is done per rollout or once per eval
        """
        Process a SuperGPQA item.
        Shuffle answer positions to avoid reward hacking where the model might learn positions rather than the correct option.
        """
        # actual ans not the options
        correct_answer = item["answer"]
        incorrect_answers = item["options"]

        # Randomly place correct answer
        gold_index = self.shuffle_rng.randint(0, len(incorrect_answers)-1)
        choices = incorrect_answers.copy()
        choices.insert(gold_index, correct_answer)

        return {
            "question": item["question"],
            "choices": choices,
            "gold_index": gold_index,
            "gold_letter": ascii_uppercase[gold_index],
            "subfield": item.get("subfield", "unknown"),
            "original_item": item,
        }

    def _format_super_gpqa_prompt(self, question: str, choices: List[str]) -> str:
        """
        Format a SuperGPQA question, add answer choices to the 0-shot prompt.
        """
        return SUPER_GPQA_ZERO_SHOT_PROMPT.format(
            Question=question.strip(),
        ) + "\n\n" + "\n".join(
            [f"{ascii_uppercase[i]}) {choice.strip()}" for i, choice in enumerate(choices)]
        )

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
        self, response: str, num_choices: int, choices: Optional[List[str]] = None
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
        for _, pattern, method_name in self._extraction_patterns:
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
        """Evaluate a single SuperGPQA question."""
        try:
            question = eval_item.get("question", "")
            choices = eval_item.get("choices", [])
            gold_letter = eval_item.get("gold_letter", "A")
            subfield = eval_item.get("subfield", "unknown")

            if not question or len(choices) < 2:
                return {"is_correct": None, "sample": None}

            formatted_prompt = self._format_super_gpqa_prompt(question, choices)

            messages = []
            system_content = self._create_system_content()
            if system_content:
                messages.append({"role": "system", "content": system_content})
            messages.append({"role": "user", "content": formatted_prompt})

            model_response = None
            finish_reason = None

            completion_kwargs = {
                "messages": messages,
                "n": 1,
                "temperature": self.config.eval_temperature,
                "split": "eval",
            }
            if self.config.eval_max_tokens > 0: # 0 means "use model default", so we don't pass the parameter
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

            extracted_answer, extraction_method = self._extract_answer(
                content_for_extraction, num_choices=len(choices), choices=choices
            )
            is_correct = extracted_answer == gold_letter if extracted_answer else False

            # Build sample record
            sample = {
                "question": question,
                "choices": choices,
                "gold_answer": gold_letter,
                "model_response": model_response,
                "extracted_answer": extracted_answer,
                "extraction_method": extraction_method,
                "is_correct": is_correct,
                "subfield": subfield,
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
                    f"  [{status}] {subfield}: gold={gold_letter}, extracted={extracted_answer}"
                )

            return {"is_correct": is_correct, "sample": sample}

        except Exception as e:
            if self.config.full_debug:
                print(f"Error in rollout_and_score_eval: {e}")
                import traceback

                traceback.print_exc()
            return {"is_correct": None, "sample": None}

    async def evaluate(self, *args, **kwargs) -> None:
        """Run SuperGPQA evaluation."""
        start_time = time.time()

        print(f"\n{'='*60}")
        print("Starting SuperGPQA Evaluation:")
        print(f"{'='*60}")
        print(f"  Subset: {self.config.subset}")
        print(f"  Total questions: {len(self.all_eval_items)}")
        print(f"  Max tokens (for reasoning): {self.config.eval_max_tokens}")
        print(f"  Thinking mode: {self.config.thinking_mode}")
        print(f"{'='*60}\n")

        try:
            eval_tasks = [
                self.rollout_and_score_eval(item) for item in self.all_eval_items
            ]
            results = await tqdm_asyncio.gather(*eval_tasks, desc="Evaluating SuperGPQA")

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

        # Per-subfield accuracy
        subfield_results = {}
        for sample in samples:
            subfield = sample.get("subfield", "unknown")
            if subfield not in subfield_results:
                subfield_results[subfield] = {"correct": 0, "total": 0}
            subfield_results[subfield]["total"] += 1
            if sample["is_correct"]:
                subfield_results[subfield]["correct"] += 1

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

        # Add subfield metrics
        for subfield, stats in subfield_results.items():
            if stats["total"] > 0:
                subfield_accuracy = stats["correct"] / stats["total"]
                subfield_key = subfield.replace(" ", "_").replace("-", "_").lower()
                eval_metrics[f"eval/subfield_{subfield_key}_accuracy"] = subfield_accuracy

        # Store metrics for wandb logging
        self.eval_metrics = [(k, v) for k, v in eval_metrics.items()]

        # Print summary
        print(f"\n{'='*60}")
        print(f"SuperGPQA Evaluation Results")
        print(f"{'='*60}")
        print(
            f"Overall Accuracy: {overall_accuracy:.4f} ({total_correct}/{total_count})"
        )
        print(f"Evaluation Time: {end_time - start_time:.1f} seconds")
        print(f"Avg Response Length: {avg_response_length:.0f} chars")
        if self.config.thinking_mode:
            print(f"Format Compliance: {format_compliance_rate:.4f}")
            print(f"Thinking Utilization: {thinking_utilization}/{total_count}")

        print("\nSubfield Breakdown:")
        for subfield, stats in sorted(subfield_results.items()):
            if stats["total"] > 0:
                subfield_acc = stats["correct"] / stats["total"]
                print(
                    f"  {subfield}: {subfield_acc:.4f} ({stats['correct']}/{stats['total']})"
                )

        print("\nExtraction Method Statistics:")
        for method, stats in sorted(
            extraction_methods.items(), key=lambda x: -x[1]["count"]
        ):
            if stats["count"] > 0:
                method_acc = stats["correct"] / stats["count"]
                print(f"  {method}: {stats['count']} uses, {method_acc:.4f} accuracy")

        print(f"{'='*60}\n")

        try:
            await self.evaluate_log(
                metrics=eval_metrics,
                start_time=start_time,
                end_time=end_time,
                samples=samples,
                generation_parameters={
                    "temperature": self.config.eval_temperature,
                    "max_tokens": self.config.eval_max_tokens,
                    "thinking_mode": self.config.thinking_mode,
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
        wandb_metrics["config/eval_max_tokens"] = self.config.eval_max_tokens
        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    SuperGPQAEvalEnv.cli()
