"""
AGIEval Evaluation Environment for Atropos (Generative/Reasoning Mode)

This environment evaluates models on the AGIEval benchmark using a generative
approach where models can reason before answering.

Dataset: Multiple dmayhem93/agieval-* datasets
Paper: https://arxiv.org/abs/2304.06364

AGIEval is a human-centric benchmark specifically designed to evaluate the
general abilities of foundation models in tasks pertinent to human cognition and
problem-solving. This benchmark is derived from 20 official, public, and
high-standard admission and qualification exams intended for general human
test-takers, such as general college admission tests (e.g., Chinese College
Entrance Exam (Gaokao) and American SAT), law school admission tests, math
competitions, lawyer qualification tests, and national civil service exams.

The evaluation follows a generative approach:
- Models are prompted to "think step by step before answering"
- Models output their reasoning followed by "Answer: X"
- Answer is extracted using regex patterns from the response
- Simple string matching validates the extracted answer

Supports optional thinking mode with <think></think> tags for extended reasoning.

Available subsets:
- English: aqua-rat, logiqa-en, lsat-ar, lsat-lr, lsat-rc, sat-en, sat-en-without-passage, sat-math
- Chinese: gaokao-biology, gaokao-chemistry, gaokao-chinese, gaokao-english,
           gaokao-geography, gaokao-history, gaokao-mathqa, gaokao-physics, logiqa-zh
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

# AGIEval generative prompt template with <answer> tag instruction
AGIEVAL_PROMPT_TEMPLATE = """Answer the following multiple choice question. Think step by step before answering.

Provide your final answer within <answer></answer> tags, containing only the letter ({valid_letters}).

Example format:
<answer>A</answer>

{question}

{choices}"""


# All available AGIEval subsets with their HuggingFace repo names
AGIEVAL_SUBSETS = {
    # English subsets
    "aqua-rat": "dmayhem93/agieval-aqua-rat",
    "logiqa-en": "dmayhem93/agieval-logiqa-en",
    "lsat-ar": "dmayhem93/agieval-lsat-ar",
    "lsat-lr": "dmayhem93/agieval-lsat-lr",
    "lsat-rc": "dmayhem93/agieval-lsat-rc",
    "sat-en": "dmayhem93/agieval-sat-en",
    "sat-en-without-passage": "dmayhem93/agieval-sat-en-without-passage",
    "sat-math": "dmayhem93/agieval-sat-math",
    # Chinese subsets
    "gaokao-biology": "dmayhem93/agieval-gaokao-biology",
    "gaokao-chemistry": "dmayhem93/agieval-gaokao-chemistry",
    "gaokao-chinese": "dmayhem93/agieval-gaokao-chinese",
    "gaokao-english": "dmayhem93/agieval-gaokao-english",
    "gaokao-geography": "dmayhem93/agieval-gaokao-geography",
    "gaokao-history": "dmayhem93/agieval-gaokao-history",
    "gaokao-mathqa": "dmayhem93/agieval-gaokao-mathqa",
    "gaokao-physics": "dmayhem93/agieval-gaokao-physics",
    "logiqa-zh": "dmayhem93/agieval-logiqa-zh",
}

# English-only subsets for convenience
AGIEVAL_ENGLISH_SUBSETS = [
    "aqua-rat",
    "logiqa-en",
    "lsat-ar",
    "lsat-lr",
    "lsat-rc",
    "sat-en",
    "sat-en-without-passage",
    "sat-math",
]

# Chinese-only subsets for convenience
AGIEVAL_CHINESE_SUBSETS = [
    "gaokao-biology",
    "gaokao-chemistry",
    "gaokao-chinese",
    "gaokao-english",
    "gaokao-geography",
    "gaokao-history",
    "gaokao-mathqa",
    "gaokao-physics",
    "logiqa-zh",
]


class AGIEvalConfig(BaseEnvConfig):
    """Configuration for AGIEval evaluation environment (generative mode)."""

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
    subsets: Optional[List[str]] = Field(
        default=None,
        description="List of AGIEval subsets to evaluate. If None, evaluates all English subsets. "
        "Available: aqua-rat, logiqa-en, lsat-ar, lsat-lr, lsat-rc, sat-en, "
        "sat-en-without-passage, sat-math, gaokao-biology, gaokao-chemistry, etc.",
    )

    english_only: bool = Field(
        default=True,
        description="If True and subsets is None, only evaluate English subsets. "
        "If False and subsets is None, evaluate all subsets including Chinese.",
    )

    eval_split: str = Field(
        default="test",
        description="Dataset split to use for evaluation.",
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


class AGIEvalEnv(BaseEnv):
    """
    AGIEval Evaluation Environment for Atropos (Generative/Reasoning Mode).

    Evaluates models on the AGIEval benchmark using a generative approach where
    models reason before answering multiple-choice questions from standardized exams.

    Key features:
    - Loads multiple AGIEval subsets from HuggingFace (dmayhem93/agieval-*)
    - Uses generative prompt format with "think step by step"
    - Optional thinking mode with <think></think> tags
    - Tracks per-subset accuracy
    - Supports English and Chinese subsets
    """

    name = "agieval_eval"
    env_config_cls = AGIEvalConfig

    def __init__(
        self,
        config: AGIEvalConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: AGIEvalConfig = config

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

        # Build fallback answer extraction patterns (supports A-E for up to 5 choices)
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
        """Build regex patterns for extracting answer letters (A-E for up to 5 choices)."""
        # AGIEval typically has 4-5 choices
        letters = "ABCDE"
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
    def config_init(cls) -> Tuple[AGIEvalConfig, List[APIServerConfig]]:
        """Initialize default configuration for the environment."""
        env_config = AGIEvalConfig(
            tokenizer_name="NousResearch/Hermes-3-Llama-3.1-8B",
            group_size=1,
            use_wandb=True,
            max_num_workers_per_node=128,
            rollout_server_url="http://localhost:8000",
            total_steps=1,
            batch_size=1,
            steps_per_eval=1,
            inference_weight=1.0,
            wandb_name="agieval_eval",
            eval_handling=EvalHandlingEnum.STOP_TRAIN,
            max_eval_workers=256,
            max_num_workers=1024,
            # AGIEval specific defaults
            subsets=None,  # Defaults to English subsets
            english_only=True,
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
                num_requests_for_eval=1024,
            ),
        ]

        return env_config, server_configs

    async def setup(self) -> None:
        """Load the AGIEval dataset and prepare for evaluation."""
        print("\nAGIEval Evaluation Setup (Generative Mode):")
        print(f"  Max tokens for reasoning: {self.config.eval_max_tokens}")
        print(f"  Evaluation split: {self.config.eval_split}")
        print(f"  Thinking mode: {self.config.thinking_mode}")
        if self.config.thinking_mode:
            prompt_preview = self._get_thinking_prompt()[:100]
            print(f"  Thinking prompt: {prompt_preview}...")

        # Determine which subsets to use
        if self.config.subsets:
            subsets_to_load = self.config.subsets
        elif self.config.english_only:
            subsets_to_load = AGIEVAL_ENGLISH_SUBSETS
        else:
            subsets_to_load = list(AGIEVAL_SUBSETS.keys())

        print(f"  Subsets to evaluate: {subsets_to_load}")

        # Load all subsets
        self.eval_data = []
        subset_counts = {}

        for subset_name in subsets_to_load:
            if subset_name not in AGIEVAL_SUBSETS:
                print(f"  Warning: Unknown subset '{subset_name}', skipping.")
                continue

            repo_name = AGIEVAL_SUBSETS[subset_name]

            try:
                dataset = load_dataset(repo_name, split=self.config.eval_split)
                items = list(dataset)

                # Add subset info to each item
                for item in items:
                    item["_subset"] = subset_name

                self.eval_data.extend(items)
                subset_counts[subset_name] = len(items)
                print(f"    Loaded {len(items)} items from {subset_name}")

            except Exception as e:
                print(f"  Error loading {subset_name}: {e}")

        print(f"\n  Total evaluation items: {len(self.eval_data)}")

        # Print subset distribution
        print("\n  Subset distribution:")
        for subset, count in sorted(subset_counts.items()):
            print(f"    {subset}: {count} questions")

        self.all_eval_items = self.eval_data
        self.iter = 0

    def _clean_choice(self, choice: str) -> str:
        """Clean choice text by removing letter prefixes like (A), (B), etc."""
        # Remove patterns like "(A)", "(B)", "(C)", "(D)" at the start
        cleaned = re.sub(r"^\s*\([A-E]\)\s*", "", choice)
        return cleaned.strip()

    def _format_choices(self, choices: List[str]) -> str:
        """Format choices as A) choice1, B) choice2, etc."""
        lines = []
        for idx, choice in enumerate(choices):
            letter = ascii_uppercase[idx]
            cleaned = self._clean_choice(choice)
            lines.append(f"{letter}) {cleaned}")
        return "\n".join(lines)

    def _format_agieval_prompt(
        self,
        query: str,
        choices: List[str],
    ) -> str:
        """
        Format a question using the generative AGIEval template.
        """
        num_choices = len(choices)
        valid_letters = "".join(ascii_uppercase[:num_choices])

        # Format choices
        formatted_choices = self._format_choices(choices)

        # Use generative template (like GPQA)
        prompt = AGIEVAL_PROMPT_TEMPLATE.format(
            question=query,
            choices=formatted_choices,
            valid_letters=valid_letters,
        )

        return prompt

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
        self, response: str, num_choices: int = 4, choices: Optional[List[str]] = None
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
        """Evaluate a single AGIEval question using generative mode."""
        try:
            query = eval_item.get("query", "")
            choices = eval_item.get("choices", [])
            gold_indices = eval_item.get("gold", [])  # Note: gold is a list
            subset = eval_item.get("_subset", "unknown")

            num_choices = len(choices)

            # Handle gold index (can be a list)
            if isinstance(gold_indices, list) and len(gold_indices) > 0:
                gold_index = gold_indices[0]
            else:
                gold_index = gold_indices

            gold_letter = (
                ascii_uppercase[gold_index] if isinstance(gold_index, int) else None
            )

            if not query or num_choices < 2 or gold_letter is None:
                return {"is_correct": None, "sample": None}

            # Format the prompt (generative style like GPQA)
            formatted_prompt = self._format_agieval_prompt(
                query=query,
                choices=choices,
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
                    if self.config.full_debug:
                        print(
                            f"  Making API request (attempt {attempt + 1}/{self.config.max_retries})..."
                        )
                        try:
                            model_name = (
                                self.server.servers[0].config.model_name
                                if hasattr(self.server, "servers")
                                else "unknown"
                            )
                        except Exception:
                            model_name = "unknown"
                        print(f"    Model: {model_name}")
                        print(f"    Temperature: {self.config.eval_temperature}")
                        print(
                            f"    Max tokens: {self.config.eval_max_tokens if self.config.eval_max_tokens > 0 else 'model default'}"
                        )

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
                    # Extract the underlying error from RetryError if present
                    actual_error = e
                    error_chain = []
                    while actual_error is not None:
                        error_chain.append(
                            f"{type(actual_error).__name__}: {actual_error}"
                        )
                        # Try to get the underlying cause
                        if (
                            hasattr(actual_error, "__cause__")
                            and actual_error.__cause__ is not None
                        ):
                            actual_error = actual_error.__cause__
                        elif hasattr(actual_error, "last_attempt"):
                            # tenacity RetryError stores the last attempt's exception
                            try:
                                actual_error = actual_error.last_attempt.exception()
                            except Exception:
                                break
                        else:
                            break

                    # Always log API errors to help diagnose issues
                    print(
                        f"  API Error (attempt {attempt + 1}/{self.config.max_retries}): {type(e).__name__}: {e}"
                    )

                    # Print the full error chain for debugging
                    if len(error_chain) > 1:
                        print("    Error chain:")
                        for i, err in enumerate(error_chain):
                            print(f"      {'  ' * i}-> {err}")

                    if hasattr(e, "response"):
                        try:
                            resp_text = (
                                e.response.text[:500]
                                if hasattr(e.response, "text")
                                else str(e.response)
                            )
                            print(f"    Response: {resp_text}")
                        except Exception:
                            pass
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(self.config.retry_delay)
                    else:
                        retries = self.config.max_retries
                        print(f"  Failed after {retries} attempts")
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
                content_for_extraction, num_choices, choices=choices
            )

            # Check if correct
            is_correct = extracted_answer == gold_letter if extracted_answer else False

            # Build sample record
            sample = {
                "query": query,
                "choices": choices,
                "gold_answer": gold_letter,
                "model_response": model_response,
                "extracted_answer": extracted_answer,
                "extraction_method": extraction_method,
                "is_correct": is_correct,
                "subset": subset,
                "num_choices": num_choices,
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
                    f"  [{status}] {subset}: gold={gold_letter}, extracted={extracted_answer}"
                )

            return {"is_correct": is_correct, "sample": sample}

        except Exception as e:
            if self.config.full_debug:
                print(f"Error in rollout_and_score_eval: {e}")
                import traceback

                traceback.print_exc()
            return {"is_correct": None, "sample": None}

    async def evaluate(self, *args, **kwargs) -> None:
        """Run AGIEval evaluation."""
        start_time = time.time()

        print("\n" + "=" * 60)
        print("Starting AGIEval Evaluation (Generative/Reasoning Mode)")
        print("=" * 60)
        print(f"  Total questions: {len(self.all_eval_items)}")
        print(f"  Max tokens (for reasoning): {self.config.eval_max_tokens}")
        print(f"  Thinking mode: {self.config.thinking_mode}")
        print(f"{'='*60}\n")

        try:
            eval_tasks = [
                self.rollout_and_score_eval(item) for item in self.all_eval_items
            ]
            results = await tqdm_asyncio.gather(*eval_tasks, desc="Evaluating AGIEval")

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

        # Per-subset accuracy
        subset_results = {}
        for sample in samples:
            subset = sample.get("subset", "unknown")
            if subset not in subset_results:
                subset_results[subset] = {"correct": 0, "total": 0}
            subset_results[subset]["total"] += 1
            if sample["is_correct"]:
                subset_results[subset]["correct"] += 1

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
            "eval/num_subsets": len(subset_results),
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

        # Add subset metrics
        for subset, stats in subset_results.items():
            if stats["total"] > 0:
                subset_accuracy = stats["correct"] / stats["total"]
                subset_key = subset.replace("-", "_").replace(" ", "_").lower()
                eval_metrics[f"eval/subset_{subset_key}_accuracy"] = subset_accuracy
                eval_metrics[f"eval/subset_{subset_key}_total"] = stats["total"]

        # Add extraction method metrics
        for method, stats in extraction_methods.items():
            if stats["count"] > 0:
                method_accuracy = stats["correct"] / stats["count"]
                eval_metrics[f"eval/extraction_{method}_count"] = stats["count"]
                eval_metrics[f"eval/extraction_{method}_accuracy"] = method_accuracy

        # Store metrics for wandb logging
        self.eval_metrics = [(k, v) for k, v in eval_metrics.items()]

        # Print summary
        print("\n" + "=" * 60)
        print("AGIEval Evaluation Results")
        print("=" * 60)
        print(
            f"Overall Accuracy: {overall_accuracy:.4f} ({total_correct}/{total_count})"
        )
        print(f"Evaluation Time: {end_time - start_time:.1f} seconds")
        print(f"Avg Response Length: {avg_response_length:.0f} chars")
        if self.config.thinking_mode:
            print(f"Format Compliance: {format_compliance_rate:.4f}")
            print(f"Thinking Utilization: {thinking_utilization}/{total_count}")

        print("\nSubset Breakdown:")
        for subset, stats in sorted(subset_results.items()):
            if stats["total"] > 0:
                subset_acc = stats["correct"] / stats["total"]
                print(
                    f"  {subset}: {subset_acc:.4f} ({stats['correct']}/{stats['total']})"
                )

        print("\nExtraction Method Statistics:")
        for method, stats in sorted(
            extraction_methods.items(), key=lambda x: -x[1]["count"]
        ):
            if stats["count"] > 0:
                method_acc = stats["correct"] / stats["count"]
                print(f"  {method}: {stats['count']} uses, {method_acc:.4f} accuracy")

        print("=" * 60 + "\n")

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
        wandb_metrics["config/eval_max_tokens"] = self.config.eval_max_tokens

        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    AGIEvalEnv.cli()
