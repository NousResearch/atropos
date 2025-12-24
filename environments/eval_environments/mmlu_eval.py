"""
MMLU Evaluation Environment for Atropos (Generative/Reasoning Mode)

This environment evaluates models on the Massive Multitask Language Understanding (MMLU)
benchmark using a generative approach where models can reason before answering.

Dataset: lighteval/mmlu (or configurable)
Paper: https://arxiv.org/abs/2009.03300

The evaluation follows the lighteval generative approach (like GPQA/MMLU-Pro):
- Models are prompted to "think step by step before answering"
- Models output their reasoning followed by "Answer: X"
- Answer is extracted using regex patterns from the response
- Simple string matching validates the extracted answer

Supports optional thinking mode with <think></think> tags for extended reasoning.
"""

import asyncio
import os
import random
import re
import time
from string import ascii_uppercase
from typing import Dict, List, Optional, Tuple

import wandb
from datasets import load_dataset
from eval_helpers import (
    build_mcqa_fallback_patterns,
    create_system_content,
    extract_letter_from_answer_tag,
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

# All 57 MMLU subjects - used for dataset loading and category tracking
MMLU_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

# High-level category groupings for aggregate metrics
SUBJECT_CATEGORIES = {
    "STEM": [
        "abstract_algebra",
        "astronomy",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "electrical_engineering",
        "elementary_mathematics",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_statistics",
        "machine_learning",
        "college_medicine",
        "clinical_knowledge",
        "medical_genetics",
        "professional_medicine",
        "anatomy",
        "nutrition",
        "virology",
        "human_aging",
    ],
    "Humanities": [
        "formal_logic",
        "high_school_european_history",
        "high_school_us_history",
        "high_school_world_history",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "moral_disputes",
        "moral_scenarios",
        "philosophy",
        "prehistory",
        "professional_law",
        "world_religions",
    ],
    "Social_Sciences": [
        "econometrics",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_microeconomics",
        "high_school_psychology",
        "human_sexuality",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
    ],
    "Other": [
        "business_ethics",
        "global_facts",
        "management",
        "marketing",
        "miscellaneous",
        "professional_accounting",
    ],
}


# Generative prompt template with <answer> tag instruction
# This is the USER message content - system prompt is handled separately
LIGHTEVAL_PROMPT_TEMPLATE = """Answer the following multiple choice question. Think step by step before answering.

Provide your final answer within <answer></answer> tags, containing only the letter ({valid_letters}).

Example format:
<answer>A</answer>

{question}

{choices}"""


class MMLUEvalConfig(BaseEnvConfig):
    """Configuration for MMLU evaluation environment (generative mode)."""

    # Thinking mode configuration (like pairwise_judgement_environment)
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
        default="lighteval/mmlu",
        description="HuggingFace dataset name for MMLU.",
    )

    subjects: Optional[List[str]] = Field(
        default=None,
        description="List of MMLU subjects to evaluate. If None, evaluates all 57 subjects.",
    )

    eval_split: str = Field(
        default="test",
        description="Dataset split to use for evaluation.",
    )

    few_shot_split: str = Field(
        default="dev",
        description="Dataset split to use for few-shot examples.",
    )

    # Few-shot configuration
    num_few_shot: int = Field(
        default=0,
        ge=0,
        le=5,
        description="Number of few-shot examples to include (0-5 recommended).",
    )

    # Model generation configuration
    eval_temperature: float = Field(
        default=0.0,
        description="Temperature for evaluation (0.0 for deterministic).",
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

    include_subject_in_prompt: bool = Field(
        default=False,
        description="Whether to include the subject name in the prompt for context.",
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


class MMLUEvalEnv(BaseEnv):
    """
    MMLU Evaluation Environment for Atropos (Generative/Reasoning Mode).

    Evaluates models on the Massive Multitask Language Understanding benchmark
    using a generative approach where models reason before answering.

    Key features:
    - Loads MMLU dataset from HuggingFace (lighteval/mmlu format)
    - Uses lighteval's exact prompt format for GPQA/MMLU-Pro style evaluation
    - Optional thinking mode with <think></think> tags for extended reasoning
    - Extracts answer letters from patterns like "Answer: A", "The final answer is B", etc.
    - Tracks per-subject and per-category accuracy
    - Supports few-shot examples

    Answer extraction follows lighteval's approach with priority-ordered patterns:
    1. "final answer is: X" (highest priority)
    2. "answer: X" or "answer X"
    3. Response starts with letter
    4. Letter at start of any line
    5. Any letter A/B/C/D in response (lowest priority, fallback)
    """

    name = "mmlu_eval"
    env_config_cls = MMLUEvalConfig

    def __init__(
        self,
        config: MMLUEvalConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: MMLUEvalConfig = config

        # Initialize metrics tracking
        self.eval_metrics = []

        # Pre-compile regex patterns for thinking mode (like pairwise_judgement_environment)
        self._think_pattern = re.compile(r"<think>")
        self._think_close_pattern = re.compile(r"</think>")
        self._think_content_pattern = re.compile(r"</think>\s*(.*)", re.DOTALL)
        self._thinking_extract_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)

        # Pre-compile regex for <answer></answer> tag extraction (primary method)
        self._answer_tag_pattern = re.compile(
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
        """
        Build regex patterns for extracting answer letters from model responses.

        Following lighteval's IndicesExtractionConfig approach, patterns are
        ordered by priority (lower number = higher priority).
        """
        # Valid answer letters (default to A-D for standard MMLU)
        letters = "ABCD"

        # Build the letter matching pattern - matches A, B, C, D or (A), (B), etc.
        letter_pattern = rf"([{letters}]|\([{letters}]\))"

        # Patterns ordered by priority (most specific first)
        # Priority 0: "final answer is: X" with "I hope" (very specific, highest confidence)
        self._pattern_final_answer_hope = re.compile(
            rf"(?i:final\s+answer\s+is)\s*:?\s*{letter_pattern}\.?\s*I\s*hope",
            re.IGNORECASE,
        )

        # Priority 50: "final answer ... is X" (allows text between)
        self._pattern_final_answer_is = re.compile(
            rf"(?i:final\s+answer).{{0,100}}?\s+is\s*:?\s*{letter_pattern}",
            re.IGNORECASE | re.DOTALL,
        )

        # Priority 75: "the answer is X"
        self._pattern_the_answer_is = re.compile(
            rf"(?i:the\s+answer\s+is)\s*:?\s*{letter_pattern}", re.IGNORECASE
        )

        # Priority 100: "answer: X" or "Answer: X" (with colon)
        self._pattern_answer_colon = re.compile(
            rf"(?i:answer)\s*:\s*.{{0,50}}?{letter_pattern}", re.IGNORECASE | re.DOTALL
        )

        # Priority 150: "answer X" or "Answer X" (without colon)
        self._pattern_answer_space = re.compile(
            rf"(?i:answer)\s+{letter_pattern}", re.IGNORECASE
        )

        # Priority 200: Response starts with answer letter (with optional punctuation)
        self._pattern_start = re.compile(
            rf"^\s*\**{letter_pattern}\**[\s\.\)\:]", re.IGNORECASE
        )

        # Priority 210: Letter at start of any line (for multi-line responses)
        self._pattern_line_start = re.compile(
            rf"\n\s*\**{letter_pattern}\**[\s\.\)\:]", re.IGNORECASE
        )

        # Priority 250: Standalone letter with word boundaries
        self._pattern_standalone = re.compile(rf"\b{letter_pattern}\b", re.IGNORECASE)

        # Store patterns in priority order
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
    def config_init(cls) -> Tuple[MMLUEvalConfig, List[APIServerConfig]]:
        """Initialize default configuration for the environment."""
        env_config = MMLUEvalConfig(
            tokenizer_name="NousResearch/Hermes-3-Llama-3.1-8B",
            group_size=1,  # Eval only, no training groups needed
            use_wandb=True,
            max_num_workers_per_node=8,
            rollout_server_url="http://localhost:8000",
            total_steps=1,  # Eval-only environment
            batch_size=1,
            steps_per_eval=1,
            inference_weight=1.0,
            wandb_name="mmlu_eval",
            eval_handling=EvalHandlingEnum.STOP_TRAIN,
            # MMLU-specific defaults
            dataset_name="lighteval/mmlu",
            subjects=None,  # All subjects by default
            num_few_shot=0,  # 0-shot by default for generative
            eval_temperature=0.6,
            eval_max_tokens=0,  # Use model default
            include_subject_in_prompt=False,  # Match lighteval default
            # Thinking mode defaults
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
        """Load the MMLU dataset and prepare for evaluation."""
        # Determine which subjects to evaluate
        self.subjects = self.config.subjects or MMLU_SUBJECTS

        # Validate subjects
        invalid_subjects = [s for s in self.subjects if s not in MMLU_SUBJECTS]
        if invalid_subjects:
            print(f"Warning: Invalid subjects will be skipped: {invalid_subjects}")
            self.subjects = [s for s in self.subjects if s in MMLU_SUBJECTS]

        if not self.subjects:
            raise ValueError("No valid MMLU subjects specified for evaluation.")

        print("\nMMLU Evaluation Setup (Generative Mode):")
        print(f"  Dataset: {self.config.dataset_name}")
        print(f"  Subjects: {len(self.subjects)} subjects")
        print(f"  Few-shot examples: {self.config.num_few_shot}")
        print(f"  Max tokens for reasoning: {self.config.eval_max_tokens}")
        print(f"  Evaluation split: {self.config.eval_split}")
        print(f"  Thinking mode: {self.config.thinking_mode}")
        if self.config.thinking_mode:
            print(f"  Thinking prompt: {self._get_thinking_prompt()[:100]}...")

        # Load datasets for each subject
        self.eval_data = {}  # subject -> list of eval items
        self.few_shot_data = {}  # subject -> list of few-shot items

        total_eval_items = 0
        for subject in self.subjects:
            try:
                # Load evaluation data
                dataset = load_dataset(
                    self.config.dataset_name,
                    subject,
                    split=self.config.eval_split,
                    trust_remote_code=True,
                )
                self.eval_data[subject] = list(dataset)
                total_eval_items += len(self.eval_data[subject])

                # Load few-shot data if needed
                if self.config.num_few_shot > 0:
                    few_shot_dataset = load_dataset(
                        self.config.dataset_name,
                        subject,
                        split=self.config.few_shot_split,
                        trust_remote_code=True,
                    )
                    self.few_shot_data[subject] = list(few_shot_dataset)

                if self.config.full_debug:
                    print(
                        f"  Loaded {subject}: {len(self.eval_data[subject])} eval items"
                    )

            except Exception as e:
                print(f"  Warning: Failed to load subject '{subject}': {e}")
                continue

        print(f"  Total evaluation items: {total_eval_items}")

        # Flatten all eval items with subject metadata for iteration
        self.all_eval_items = []
        for subject, items in self.eval_data.items():
            for item in items:
                item["subject"] = subject  # Ensure subject is in each item
                self.all_eval_items.append(item)

        self.iter = 0

    def _format_choices(self, choices: List[str]) -> str:
        """Format choices as A) choice1, B) choice2, etc."""
        lines = []
        for idx, choice in enumerate(choices):
            letter = ascii_uppercase[idx]
            lines.append(f"{letter}) {choice}")
        return "\n".join(lines)

    def _format_mmlu_prompt(
        self,
        question: str,
        choices: List[str],
        subject: str,
        few_shot_examples: Optional[List[Dict]] = None,
    ) -> str:
        """
        Format a question using the lighteval MMLU template.

        Uses the exact GPQA/MMLU-Pro style prompt from lighteval that instructs
        the model to think step by step and provide the answer in a specific format.

        Args:
            question: The question text
            choices: List of answer choices
            subject: The subject name (for context in prompt)
            few_shot_examples: Optional list of few-shot example dicts

        Returns:
            Formatted prompt string (user message content)
        """
        num_choices = len(choices)
        valid_letters = "".join(ascii_uppercase[:num_choices])

        # Format choices
        formatted_choices = self._format_choices(choices)

        # Build the question - optionally include subject
        if self.config.include_subject_in_prompt:
            subject_display = subject.replace("_", " ")
            question_with_context = f"[{subject_display}]\n\n{question}"
        else:
            question_with_context = question

        # Use lighteval's exact prompt template
        prompt = LIGHTEVAL_PROMPT_TEMPLATE.format(
            question=question_with_context,
            choices=formatted_choices,
            valid_letters=valid_letters,
        )

        # Add few-shot examples if provided (prepended)
        if few_shot_examples:
            few_shot_text = self._format_few_shot_examples(few_shot_examples)
            prompt = few_shot_text + "\n\n---\n\n" + prompt

        return prompt

    def _format_few_shot_examples(self, examples: List[Dict]) -> str:
        """Format few-shot examples with answers for context."""
        formatted = []
        for example in examples:
            question = example.get("question", "")
            choices = example.get("choices", [])
            answer = example.get("answer", 0)

            # Get the answer letter
            if isinstance(answer, int):
                answer_letter = ascii_uppercase[answer]
            else:
                answer_letter = answer.upper()

            formatted_choices = self._format_choices(choices)

            example_text = (
                f"Question: {question}\n{formatted_choices}\n\nAnswer: {answer_letter}"
            )
            formatted.append(example_text)

        return "\n\n---\n\n".join(formatted)

    def _validate_thinking_format(self, response: str) -> Tuple[bool, str]:
        """
        Validate thinking format and extract content after </think> tags.

        In thinking mode, we require exactly one pair of <think></think> tags.
        Returns the content after </think> for answer extraction.

        Args:
            response: The model's full response

        Returns:
            Tuple of (is_valid, content_for_extraction)
        """
        if not self.config.thinking_mode:
            return True, response

        # Check for exactly one pair of think tags
        think_open_count = len(self._think_pattern.findall(response))
        think_close_count = len(self._think_close_pattern.findall(response))

        if think_open_count != 1 or think_close_count != 1:
            return False, response

        # Extract content after </think> tags for answer extraction
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

        Uses shared helpers from eval_helpers.py.

        Primary method: Look for <answer></answer> tags with exactly ONE valid letter,
        or match against the exact choice texts.
        Fallback: Use priority-ordered regex patterns.

        Args:
            response: The model's response string (content after </think> in thinking mode)
            num_choices: Number of valid choices (determines valid letters)
            choices: Optional list of choice texts for exact matching

        Returns:
            Tuple of (extracted_letter or None, extraction_method used)
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

        # FALLBACK: Use regex patterns
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
                            f"    Extracted '{letter}' using fallback method '{method_name}' (priority {priority})"
                        )
                    return letter, f"fallback_{method_name}"

        # Last resort: find any valid letter (take the last one)
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
        """
        Evaluate a single MMLU question using generative mode.

        The model generates a response with reasoning, then we extract
        the final answer from patterns like "Answer: A".

        In thinking mode, validates <think></think> tags and extracts
        the answer from content after the closing tag.

        Args:
            eval_item: Dictionary with question, choices, answer, and subject

        Returns:
            Dictionary with is_correct, extracted_answer, and sample details
        """
        try:
            subject = eval_item.get("subject", "unknown")
            question = eval_item.get("question", "")
            choices = eval_item.get("choices", [])
            num_choices = len(choices)

            # Get the correct answer (handle both int index and string letter)
            gold_answer = eval_item.get("answer", 0)
            if isinstance(gold_answer, int):
                gold_letter = ascii_uppercase[gold_answer]
            else:
                gold_letter = gold_answer.upper()

            if not question or num_choices < 2:
                return {"is_correct": None, "sample": None}

            # Get few-shot examples for this subject
            few_shot_examples = None
            if self.config.num_few_shot > 0 and subject in self.few_shot_data:
                available_examples = self.few_shot_data[subject]
                num_examples = min(self.config.num_few_shot, len(available_examples))
                few_shot_examples = available_examples[:num_examples]

            # Format the prompt (lighteval style - user message content)
            formatted_prompt = self._format_mmlu_prompt(
                question=question,
                choices=choices,
                subject=subject,
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
            for attempt in range(self.config.max_retries):
                try:
                    completion = await self.server.chat_completion(
                        messages=messages,
                        n=1,
                        temperature=self.config.eval_temperature,
                        max_tokens=self.config.eval_max_tokens,
                        split="eval",
                    )

                    if completion.choices and completion.choices[0].message.content:
                        model_response = completion.choices[0].message.content
                        finish_reason = getattr(
                            completion.choices[0], "finish_reason", None
                        )

                        # Check minimum response length
                        if (
                            len(model_response.strip())
                            >= self.config.min_response_length
                        ):
                            break
                        elif attempt < self.config.max_retries - 1:
                            if self.config.full_debug:
                                print(
                                    f"  Response too short ({len(model_response)} chars), retrying..."
                                )
                            await asyncio.sleep(self.config.retry_delay)
                            continue

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

            # Validate thinking format if in thinking mode
            format_valid, content_for_extraction = self._validate_thinking_format(
                model_response
            )

            # Extract thinking content for logging (if in thinking mode)
            thinking_content = None
            if self.config.thinking_mode:
                thinking_content = self._extract_thinking_content(model_response)

            # Extract the answer from the response (or content after </think>)
            # Pass choices for exact text matching support
            extracted_answer, extraction_method = self._extract_answer(
                content_for_extraction, num_choices, choices=choices
            )

            # Check if correct
            is_correct = extracted_answer == gold_letter if extracted_answer else False

            # Build sample record for logging
            sample = {
                "subject": subject,
                "question": question,
                "choices": choices,
                "gold_answer": gold_letter,
                "model_response": model_response,
                "extracted_answer": extracted_answer,
                "extraction_method": extraction_method,
                "is_correct": is_correct,
                "num_few_shot": self.config.num_few_shot,
                "finish_reason": finish_reason,
                "response_length": len(model_response),
                "thinking_mode": self.config.thinking_mode,
                "format_valid": format_valid,
            }

            # Add thinking-specific info
            if self.config.thinking_mode:
                sample["thinking_content"] = thinking_content
                sample["response_after_think"] = (
                    content_for_extraction if format_valid else None
                )

            if self.config.full_debug:
                status = "✓" if is_correct else "✗"
                format_status = "✓" if format_valid else "✗"
                print(
                    f"  [{status}] {subject}: gold={gold_letter}, extracted={extracted_answer} ({extraction_method}), format={format_status}"
                )

            return {"is_correct": is_correct, "sample": sample}

        except Exception as e:
            if self.config.full_debug:
                print(f"Error in rollout_and_score_eval: {e}")
                import traceback

                traceback.print_exc()
            return {"is_correct": None, "sample": None}

    async def evaluate(self, *args, **kwargs) -> None:
        """
        Run MMLU evaluation across all configured subjects.

        Calculates:
        - Overall accuracy
        - Per-subject accuracy
        - Per-category accuracy (STEM, Humanities, Social Sciences, Other)
        - Extraction method statistics
        - Format compliance (for thinking mode)
        - Thinking utilization metrics
        """
        start_time = time.time()

        print(f"\n{'='*60}")
        print("Starting MMLU Evaluation (Generative/Reasoning Mode)")
        print(f"{'='*60}")
        print(f"  Subjects: {len(self.subjects)}")
        print(f"  Total questions: {len(self.all_eval_items)}")
        print(f"  Few-shot examples: {self.config.num_few_shot}")
        print(f"  Max tokens (for reasoning): {self.config.eval_max_tokens}")
        print(f"  Thinking mode: {self.config.thinking_mode}")
        print(f"{'='*60}\n")

        try:
            # Run evaluation for all items
            eval_tasks = [
                self.rollout_and_score_eval(item) for item in self.all_eval_items
            ]
            results = await tqdm_asyncio.gather(*eval_tasks, desc="Evaluating MMLU")

            # Filter valid results
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

        # Per-subject accuracy
        subject_results = {}
        for sample in samples:
            subject = sample["subject"]
            if subject not in subject_results:
                subject_results[subject] = {"correct": 0, "total": 0}
            subject_results[subject]["total"] += 1
            if sample["is_correct"]:
                subject_results[subject]["correct"] += 1

        # Per-category accuracy
        category_results = {
            cat: {"correct": 0, "total": 0} for cat in SUBJECT_CATEGORIES
        }
        for subject, stats in subject_results.items():
            for category, subjects_in_cat in SUBJECT_CATEGORIES.items():
                if subject in subjects_in_cat:
                    category_results[category]["correct"] += stats["correct"]
                    category_results[category]["total"] += stats["total"]
                    break

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

        # Format compliance (for thinking mode)
        format_compliant = sum(1 for s in samples if s.get("format_valid", True))
        format_compliance_rate = format_compliant / len(samples) if samples else 0.0

        # Thinking utilization (how many responses had thinking content)
        thinking_utilization = 0
        if self.config.thinking_mode:
            thinking_utilization = sum(1 for s in samples if s.get("thinking_content"))

        # Build metrics dictionary
        eval_metrics = {
            "eval/overall_accuracy": overall_accuracy,
            "eval/total_questions": total_count,
            "eval/total_correct": total_correct,
            "eval/num_subjects": len(subject_results),
            "eval/num_few_shot": self.config.num_few_shot,
            "eval/evaluation_time_seconds": end_time - start_time,
            "eval/avg_response_length": avg_response_length,
            "eval/format_compliance_rate": format_compliance_rate,
            "eval/thinking_mode_enabled": 1.0 if self.config.thinking_mode else 0.0,
        }

        # Add thinking utilization if in thinking mode
        if self.config.thinking_mode:
            thinking_utilization_rate = (
                thinking_utilization / len(samples) if samples else 0.0
            )
            eval_metrics["eval/thinking_utilization_rate"] = thinking_utilization_rate

        # Add category metrics
        for category, stats in category_results.items():
            if stats["total"] > 0:
                cat_accuracy = stats["correct"] / stats["total"]
                eval_metrics[f"eval/category_{category.lower()}_accuracy"] = (
                    cat_accuracy
                )
                eval_metrics[f"eval/category_{category.lower()}_total"] = stats["total"]

        # Add extraction method metrics
        for method, stats in extraction_methods.items():
            if stats["count"] > 0:
                method_accuracy = stats["correct"] / stats["count"]
                eval_metrics[f"eval/extraction_{method}_count"] = stats["count"]
                eval_metrics[f"eval/extraction_{method}_accuracy"] = method_accuracy

        # Add per-subject metrics
        for subject, stats in sorted(subject_results.items()):
            if stats["total"] > 0:
                subj_accuracy = stats["correct"] / stats["total"]
                # Sanitize subject name for metric key
                subj_key = subject.replace(" ", "_").replace("-", "_")
                eval_metrics[f"eval/subject_{subj_key}_accuracy"] = subj_accuracy

        # Store metrics for wandb logging
        self.eval_metrics = [(k, v) for k, v in eval_metrics.items()]

        # Print summary
        print(f"\n{'='*60}")
        print("MMLU Evaluation Results")
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
        for category, stats in category_results.items():
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

        # Add evaluation metrics
        for metric_name, metric_value in self.eval_metrics:
            wandb_metrics[metric_name] = metric_value
        self.eval_metrics = []

        # Add config metrics
        wandb_metrics["config/thinking_mode"] = (
            1.0 if self.config.thinking_mode else 0.0
        )
        wandb_metrics["config/num_few_shot"] = self.config.num_few_shot
        wandb_metrics["config/eval_max_tokens"] = self.config.eval_max_tokens

        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    MMLUEvalEnv.cli()
