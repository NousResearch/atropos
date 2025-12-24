"""
MuSR (Multi-Step Reasoning) Evaluation Environment for Atropos

This environment evaluates models on the MuSR benchmark - testing multi-step
reasoning in natural language narratives with long-form contexts.

Dataset: TAUR-Lab/MuSR
Paper: https://arxiv.org/abs/2310.16049

MuSR includes three challenging reasoning tasks:
- murder_mysteries: Solve murder mystery narratives
- object_placements: Track object locations through narratives
- team_allocation: Reason about team assignments

All tasks involve long narratives requiring multi-hop reasoning.

Metrics:
- Accuracy (exact match on multiple choice)

Supports optional thinking mode with <think></think> tags.
"""

import ast
import asyncio
import os
import re
import time
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset
from eval_helpers import (
    create_system_content,
    extract_number_from_answer_tag,
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

# Available MuSR subsets
MUSR_SUBSETS = ["murder_mysteries", "object_placements", "team_allocation"]


class MuSREvalConfig(BaseEnvConfig):
    """Configuration for MuSR evaluation environment."""

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
        default="TAUR-Lab/MuSR",
        description="HuggingFace dataset name for MuSR.",
    )

    subset: str = Field(
        default="murder_mysteries",
        description="MuSR subset to evaluate: murder_mysteries, object_placements, or team_allocation.",
    )

    # Model generation configuration
    eval_temperature: float = Field(
        default=0.6,
        description="Temperature for model generation.",
    )

    eval_max_tokens: int = Field(
        default=0,
        description="Maximum tokens for evaluation responses. Set to 0 for provider default.",
    )

    # Prompt configuration
    custom_system_prompt: Optional[str] = Field(
        default=None,
        description="Custom system prompt to append after thinking prompt.",
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


class MuSREvalEnv(BaseEnv):
    """
    MuSR Evaluation Environment for Atropos.

    Evaluates models on multi-step reasoning with long narratives.
    """

    name = "musr_eval"
    env_config_cls = MuSREvalConfig

    def __init__(
        self,
        config: MuSREvalConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: MuSREvalConfig = config
        self.eval_metrics = []

        # Regex patterns for thinking mode
        self._think_pattern = re.compile(r"<think>")
        self._think_close_pattern = re.compile(r"</think>")
        self._think_content_pattern = re.compile(r"</think>\s*(.*)", re.DOTALL)
        self._thinking_extract_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)

        # Pre-compile regex for <answer></answer> tag extraction (primary method)
        self._answer_tag_pattern = re.compile(
            r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE
        )

        # Build fallback answer extraction patterns for numbered choices
        self._build_extraction_patterns()

    def _build_extraction_patterns(self):
        """
        Build regex patterns for extracting answer numbers from model responses.

        Patterns are ordered by priority (lower number = higher priority).
        Takes the LAST match for answer patterns since models often repeat the final answer.
        """
        # Number pattern for choices (1-10 to be safe)
        num_pattern = r"(\d+)"

        # Priority 0: "final answer is: X" with "I hope" (very specific, highest confidence)
        self._pattern_final_answer_hope = re.compile(
            rf"(?i:final\s+answer\s+is)\s*:?\s*{num_pattern}\.?\s*I\s*hope",
            re.IGNORECASE,
        )

        # Priority 50: "final answer ... is X" (allows text between)
        self._pattern_final_answer_is = re.compile(
            rf"(?i:final\s+answer).{{0,100}}?\s+is\s*:?\s*{num_pattern}",
            re.IGNORECASE | re.DOTALL,
        )

        # Priority 75: "the answer is X"
        self._pattern_the_answer_is = re.compile(
            rf"(?i:the\s+answer\s+is)\s*:?\s*{num_pattern}", re.IGNORECASE
        )

        # Priority 100: "answer: X" or "Answer: X" (with colon)
        self._pattern_answer_colon = re.compile(
            rf"(?i:answer)\s*:\s*.{{0,50}}?{num_pattern}", re.IGNORECASE | re.DOTALL
        )

        # Priority 125: "option X" or "choice X"
        self._pattern_option = re.compile(
            rf"(?i:option|choice)\s+{num_pattern}", re.IGNORECASE
        )

        # Priority 150: "answer X" or "Answer X" (without colon)
        self._pattern_answer_space = re.compile(
            rf"(?i:answer)\s+{num_pattern}", re.IGNORECASE
        )

        # Priority 200: Response starts with number (with optional punctuation)
        self._pattern_start = re.compile(rf"^\s*{num_pattern}[\s\.\)\:]", re.IGNORECASE)

        # Priority 210: Number at start of any line
        self._pattern_line_start = re.compile(
            rf"\n\s*{num_pattern}[\s\.\)\:]", re.IGNORECASE
        )

        # Priority 300: Number at end of response
        self._pattern_end = re.compile(rf"{num_pattern}\s*$", re.IGNORECASE)

        # Store patterns in priority order
        self._extraction_patterns = [
            (0, self._pattern_final_answer_hope, "final_answer_hope"),
            (50, self._pattern_final_answer_is, "final_answer_is"),
            (75, self._pattern_the_answer_is, "the_answer_is"),
            (100, self._pattern_answer_colon, "answer_colon"),
            (125, self._pattern_option, "option"),
            (150, self._pattern_answer_space, "answer_space"),
            (200, self._pattern_start, "start"),
            (210, self._pattern_line_start, "line_start"),
            (300, self._pattern_end, "end"),
        ]

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

    @classmethod
    def config_init(cls) -> Tuple[MuSREvalConfig, List[APIServerConfig]]:
        """Initialize default configuration for the environment."""
        env_config = MuSREvalConfig(
            tokenizer_name="NousResearch/Hermes-3-Llama-3.1-8B",
            group_size=1,
            use_wandb=True,
            max_num_workers_per_node=128,
            rollout_server_url="http://localhost:8000",
            total_steps=1,
            batch_size=1,
            steps_per_eval=1,
            inference_weight=1.0,
            wandb_name="musr_eval",
            eval_handling=EvalHandlingEnum.STOP_TRAIN,
            max_eval_workers=256,
            max_num_workers=1024,
            dataset_name="TAUR-Lab/MuSR",
            subset="murder_mysteries",
            eval_temperature=0.6,
            eval_max_tokens=0,
            thinking_mode=True,
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

    def _format_musr_prompt(self, item: Dict) -> str:
        """Format a MuSR item into a prompt with <answer> tag instruction.

        Based on lighteval's musr_prompt but with explicit <answer> tag instruction.
        Uses numbered choices (1, 2, 3...) as in the original format.
        """
        narrative = item.get("narrative", "")
        question = item.get("question", "")

        # Parse choices - they're stored as a string representation of a list
        choices_raw = item.get("choices", "[]")
        if isinstance(choices_raw, str):
            try:
                choices = ast.literal_eval(choices_raw)
            except Exception:
                choices = []
        else:
            choices = choices_raw

        num_choices = len(choices)
        valid_numbers = ", ".join(str(i + 1) for i in range(num_choices))

        query = "Read the narrative and answer the question. Think step by step before answering.\n\n"
        query += f"Provide your final answer within <answer></answer> tags, containing only the number ({valid_numbers}).\n\n"  # noqa: E501
        query += "Example format:\n<answer>1</answer>\n\n"
        query += f"{narrative}\n\n{question}\n\n"
        for i, choice in enumerate(choices):
            query += f"{i + 1} - {choice}\n"

        return query, choices

    async def setup(self) -> None:
        """Load the MuSR dataset and prepare for evaluation."""
        print("\nMuSR Evaluation Setup:")
        print(f"  Dataset: {self.config.dataset_name}")
        print(f"  Subset: {self.config.subset}")
        print(f"  Max tokens: {self.config.eval_max_tokens}")
        print(f"  Thinking mode: {self.config.thinking_mode}")
        if self.config.thinking_mode:
            print(f"  Thinking prompt: {self._get_thinking_prompt()[:100]}...")

        if self.config.subset not in MUSR_SUBSETS:
            print(
                f"Warning: Unknown subset '{self.config.subset}'. Available: {MUSR_SUBSETS}"
            )

        try:
            # MuSR has splits named after the subsets
            dataset = load_dataset(
                self.config.dataset_name,
                split=self.config.subset,
            )
            self.eval_data = list(dataset)
            print(f"  Loaded {len(self.eval_data)} evaluation items")

        except Exception as e:
            print(f"Error loading MuSR dataset: {e}")
            raise

        self.all_eval_items = self.eval_data
        self.iter = 0

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

    def _extract_answer_number(
        self, response: str, num_choices: int
    ) -> Tuple[Optional[int], str]:
        """
        Extract the answer number (1-indexed) from the model's response.

        Primary method: Look for <answer></answer> tags (only take the first match).
        Fallback: Use priority-ordered regex patterns.

        Args:
            response: The model's response string (content after </think> in thinking mode)
            num_choices: Number of valid choices

        Returns:
            Tuple of (extracted_number or None, extraction_method used)
        """
        if not response:
            return None, "empty_response"

        # PRIMARY: Try <answer></answer> tags first
        # Uses word boundary matching - only accepts if EXACTLY ONE valid number found
        number, method = extract_number_from_answer_tag(
            response, num_choices, debug=self.config.full_debug
        )
        if number:
            return number, method

        # FALLBACK: Try each pattern in priority order
        for priority, pattern, method_name in self._extraction_patterns:
            matches = pattern.findall(response)
            if matches:
                # Get the LAST match for answer patterns since final answer is most reliable
                match = (
                    matches[-1]
                    if method_name
                    in [
                        "final_answer_is",
                        "the_answer_is",
                        "answer_colon",
                        "answer_space",
                        "option",
                    ]
                    else matches[0]
                )

                try:
                    num = int(match)
                    if 1 <= num <= num_choices:
                        if self.config.full_debug:
                            print(
                                f"    Extracted '{num}' using fallback method '{method_name}' (priority {priority})"
                            )
                        return num, f"fallback_{method_name}"
                except ValueError:
                    continue

        # Last resort: find any number in valid range (take the last one)
        numbers = re.findall(r"\b(\d+)\b", response)
        for num_str in reversed(numbers):
            try:
                num = int(num_str)
                if 1 <= num <= num_choices:
                    if self.config.full_debug:
                        print(
                            f"    Extracted '{num}' using fallback 'last_valid_number'"
                        )
                    return num, "fallback_last_valid_number"
            except ValueError:
                continue

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
        """Evaluate a single MuSR question."""
        try:
            prompt, choices = self._format_musr_prompt(eval_item)
            gold_index = eval_item.get("answer_index", 0)  # 0-indexed

            if not prompt or not choices:
                return {"result": None, "sample": None}

            # Build messages
            messages = []
            system_content = self._create_system_content()
            if system_content:
                messages.append({"role": "system", "content": system_content})
            messages.append({"role": "user", "content": prompt})

            # Get model response
            model_response = None
            finish_reason = None
            for attempt in range(self.config.max_retries):
                try:
                    completion_kwargs = {
                        "messages": messages,
                        "n": 1,
                        "temperature": self.config.eval_temperature,
                        "split": "eval",
                    }
                    if self.config.eval_max_tokens > 0:
                        completion_kwargs["max_tokens"] = self.config.eval_max_tokens

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

                except Exception as e:
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
                        return {"result": None, "sample": None}

            if not model_response:
                return {"result": None, "sample": None}

            # Handle thinking mode
            thinking_format_valid, response_for_eval = self._validate_thinking_format(
                model_response
            )
            thinking_content = None
            if self.config.thinking_mode:
                thinking_content = self._extract_thinking_content(model_response)

            # Extract answer (1-indexed)
            extracted_answer, extraction_method = self._extract_answer_number(
                response_for_eval, len(choices)
            )

            # Check correctness (gold_index is 0-indexed, extracted is 1-indexed)
            is_correct = (
                extracted_answer is not None and (extracted_answer - 1) == gold_index
            )

            sample = {
                "narrative": (
                    eval_item.get("narrative", "")[:300] + "..."
                    if len(eval_item.get("narrative", "")) > 300
                    else eval_item.get("narrative", "")
                ),
                "question": eval_item.get("question", ""),
                "choices": choices,
                "gold_index": gold_index,
                "gold_answer": (
                    choices[gold_index] if 0 <= gold_index < len(choices) else "N/A"
                ),
                "model_response": (
                    model_response[:500]
                    if len(model_response) > 500
                    else model_response
                ),
                "extracted_answer": extracted_answer,
                "extraction_method": extraction_method,
                "extracted_choice": (
                    choices[extracted_answer - 1]
                    if extracted_answer and 1 <= extracted_answer <= len(choices)
                    else "N/A"
                ),
                "is_correct": is_correct,
                "finish_reason": finish_reason,
                "thinking_format_valid": thinking_format_valid,
            }

            if self.config.thinking_mode:
                sample["thinking_content"] = (
                    thinking_content[:300] + "..."
                    if thinking_content and len(thinking_content) > 300
                    else thinking_content
                )

            if self.config.full_debug:
                status = "✓" if is_correct else "✗"
                print(
                    f"  [{status}] Extracted: {extracted_answer}, Gold: {gold_index + 1}"
                )

            return {"result": {"correct": is_correct}, "sample": sample}

        except Exception as e:
            if self.config.full_debug:
                print(f"Error in rollout_and_score_eval: {e}")
                import traceback

                traceback.print_exc()
            return {"result": None, "sample": None}

    async def evaluate(self, *args, **kwargs) -> None:
        """Run MuSR evaluation."""
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"Starting MuSR Evaluation ({self.config.subset})")
        print(f"{'='*60}")
        print(f"  Total questions: {len(self.all_eval_items)}")
        print(f"  Thinking mode: {self.config.thinking_mode}")
        print(f"{'='*60}\n")

        try:
            eval_tasks = [
                self.rollout_and_score_eval(item) for item in self.all_eval_items
            ]
            results = await tqdm_asyncio.gather(
                *eval_tasks, desc=f"Evaluating MuSR ({self.config.subset})"
            )

            valid_results = [
                r
                for r in results
                if r and r.get("sample") is not None and r.get("result") is not None
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
        total_count = len(valid_results)

        correct_count = sum(1 for s in samples if s.get("is_correct", False))
        accuracy = correct_count / total_count if total_count > 0 else 0.0

        # Answer extraction rate
        extracted_count = sum(
            1 for s in samples if s.get("extracted_answer") is not None
        )
        extraction_rate = extracted_count / total_count if total_count > 0 else 0.0

        # Thinking metrics
        thinking_format_compliant = sum(
            1 for s in samples if s.get("thinking_format_valid", True)
        )
        thinking_format_compliance_rate = (
            thinking_format_compliant / total_count if total_count > 0 else 0.0
        )

        thinking_utilization = (
            sum(1 for s in samples if s.get("thinking_content"))
            if self.config.thinking_mode
            else 0
        )

        eval_metrics = {
            "eval/accuracy": accuracy,
            "eval/correct_count": correct_count,
            "eval/total_count": total_count,
            "eval/answer_extraction_rate": extraction_rate,
            "eval/evaluation_time_seconds": end_time - start_time,
            "eval/thinking_mode_enabled": 1.0 if self.config.thinking_mode else 0.0,
        }

        if self.config.thinking_mode:
            eval_metrics["eval/thinking_format_compliance_rate"] = (
                thinking_format_compliance_rate
            )
            eval_metrics["eval/thinking_utilization_rate"] = (
                thinking_utilization / total_count if total_count > 0 else 0.0
            )

        self.eval_metrics = [(k, v) for k, v in eval_metrics.items()]

        # Print summary
        print(f"\n{'='*60}")
        print(f"MuSR Evaluation Results ({self.config.subset})")
        print(f"{'='*60}")
        print(f"Accuracy: {accuracy:.4f} ({correct_count}/{total_count})")
        print(f"Answer Extraction Rate: {extraction_rate:.4f}")
        print(f"Evaluation Time: {end_time - start_time:.1f} seconds")
        if self.config.thinking_mode:
            print(f"Thinking Format Compliance: {thinking_format_compliance_rate:.4f}")
        print(f"{'='*60}\n")

        try:
            await self.evaluate_log(
                metrics=eval_metrics,
                samples=samples,
                start_time=start_time,
                end_time=end_time,
                generation_parameters={
                    "temperature": self.config.eval_temperature,
                    "max_tokens": self.config.eval_max_tokens,
                    "thinking_mode": self.config.thinking_mode,
                    "subset": self.config.subset,
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
        wandb_metrics["config/subset"] = self.config.subset

        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    MuSREvalEnv.cli()
