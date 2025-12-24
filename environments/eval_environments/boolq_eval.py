"""
BoolQ (Boolean Questions) Evaluation Environment for Atropos

This environment evaluates models on the BoolQ benchmark - testing
reading comprehension with yes/no questions.

Dataset: lighteval/boolq_helm
Paper: https://arxiv.org/abs/1905.11946

BoolQ tests:
- Reading comprehension
- Yes/No question answering
- Natural, unprompted questions
- Binary choice (Yes or No)

Metrics:
- Accuracy (exact match on Yes/No)

Supports optional thinking mode with <think></think> tags.
"""

import asyncio
from typing import Dict, List, Optional, Tuple

import wandb
from datasets import load_dataset
from eval_helpers import (
    ANSWER_TAG_PATTERN,
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


class BoolQEvalConfig(BaseEnvConfig):
    """Configuration for BoolQ evaluation environment."""

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
        default="lighteval/boolq_helm",
        description="HuggingFace dataset name for BoolQ.",
    )

    eval_split: str = Field(
        default="validation",
        description="Dataset split to use for evaluation.",
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


class BoolQEvalEnv(BaseEnv):
    """
    BoolQ Evaluation Environment for Atropos.

    Evaluates models on reading comprehension with yes/no questions.
    """

    name = "boolq_eval"
    env_config_cls = BoolQEvalConfig

    def __init__(
        self,
        config: BoolQEvalConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: BoolQEvalConfig = config
        self.eval_metrics = []

        # For BoolQ we use Yes/No directly, not letter choices
        self._valid_answers = {"yes", "no"}
        # But also support A/B format
        self._fallback_patterns = build_mcqa_fallback_patterns(2)
        self._valid_letters = {"A", "B"}

    @classmethod
    def config_init(cls) -> Tuple[BoolQEvalConfig, List[APIServerConfig]]:
        """Initialize default configuration for CLI usage."""
        config = BoolQEvalConfig(
            tokenizer_name="NousResearch/Hermes-3-Llama-3.1-8B",
            group_size=1,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=1,
            batch_size=1,
            steps_per_eval=1,
            max_token_length=2048,
            wandb_name="boolq_eval",
            data_path_to_save_groups=None,
            eval_max_tokens=0,
        )
        server_config = APIServerConfig(
            model_name="Hermes-3-Llama-3.1-8B",
            base_url="http://localhost:8000/v1",
            api_key="x",
            num_requests_for_eval=1024,
        )
        return config, [server_config]

    async def setup(self):
        """Load the BoolQ dataset."""
        print("\nBoolQ Evaluation Setup (Generative Mode):")
        print(f"  Dataset: {self.config.dataset_name}")
        print(f"  Evaluation split: {self.config.eval_split}")
        print(f"  Thinking mode: {self.config.thinking_mode}")
        if self.config.thinking_mode:
            print(
                f"  Thinking prompt: {get_default_thinking_prompt(self.config.custom_thinking_prompt)[:80]}..."
            )

        # Load dataset
        self.dataset = load_dataset(
            self.config.dataset_name,
            split=self.config.eval_split,
            trust_remote_code=True,
        )

        self.eval_items = list(self.dataset)
        print(f"  Loaded {len(self.eval_items)} evaluation items")

    def _format_prompt(self, item: Dict) -> str:
        """
        Format a BoolQ item into a prompt.

        BoolQ has a passage and a question that should be answered Yes or No.
        """
        passage = item["passage"]
        question = item["question"]

        # Clean up double question marks
        if question.endswith("??"):
            question = question[:-1]

        # Build the question
        query = f"Passage: {passage}\n\n"
        query += f"Question: {question}\n\n"
        query += "Based on the passage, is the answer to the question Yes or No?\n"
        query += "A. Yes\n"
        query += "B. No\n"
        query += "\nProvide your answer in <answer></answer> tags with only the letter (A or B), or 'Yes'/'No'."

        return query

    def _create_system_content(self) -> Optional[str]:
        """Create system message content based on thinking mode configuration."""
        return create_system_content(
            self.config.thinking_mode,
            self.config.custom_thinking_prompt,
            self.config.custom_system_prompt,
        )

    def _extract_answer(self, response: str) -> Tuple[Optional[str], str]:
        """
        Extract the answer from the model's response.

        Accepts:
        - A/B letters (converted to Yes/No)
        - Yes/No directly
        """
        # Get content after </think> if in thinking mode
        if self.config.thinking_mode:
            is_valid, content_after_think = validate_thinking_format(response, True)
            if is_valid:
                response_to_parse = content_after_think
            else:
                response_to_parse = response
        else:
            response_to_parse = response

        # Try <answer></answer> tags first
        answer_match = ANSWER_TAG_PATTERN.search(response_to_parse)
        if answer_match:
            answer_content = answer_match.group(1).strip().lower()

            # Direct Yes/No
            if "yes" in answer_content and "no" not in answer_content:
                return "Yes", "answer_tag_yes"
            if "no" in answer_content and "yes" not in answer_content:
                return "No", "answer_tag_no"

            # A/B letters
            if answer_content in ["a", "a.", "(a)"]:
                return "Yes", "answer_tag_letter_a"
            if answer_content in ["b", "b.", "(b)"]:
                return "No", "answer_tag_letter_b"

            # Check for letter anywhere in short content
            if len(answer_content) <= 10:
                if "a" in answer_content and "b" not in answer_content:
                    return "Yes", "answer_tag_letter_a"
                if "b" in answer_content and "a" not in answer_content:
                    return "No", "answer_tag_letter_b"

        # Fallback: Try letter patterns
        letter, method = extract_letter_from_answer_tag(
            response_to_parse,
            self._valid_letters,
            debug=self.config.full_debug,
            choices=["Yes", "No"],
        )
        if letter:
            return "Yes" if letter == "A" else "No", method

        # Fallback: Look for Yes/No in response
        response_lower = response_to_parse.lower()

        # Check for explicit patterns
        yes_patterns = [r"\byes\b", r"\banswer is yes\b", r"\bthe answer is yes\b"]
        no_patterns = [r"\bno\b", r"\banswer is no\b", r"\bthe answer is no\b"]

        yes_matches = sum(1 for p in yes_patterns if re.search(p, response_lower))
        no_matches = sum(1 for p in no_patterns if re.search(p, response_lower))

        # Only accept if one is clearly dominant
        if yes_matches > 0 and no_matches == 0:
            return "Yes", "fallback_yes_keyword"
        if no_matches > 0 and yes_matches == 0:
            return "No", "fallback_no_keyword"

        # Try MCQA fallback patterns for A/B
        for priority, pattern, method_name in self._fallback_patterns:
            matches = pattern.findall(response_to_parse)
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
                if letter in self._valid_letters:
                    return "Yes" if letter == "A" else "No", f"fallback_{method_name}"

        return None, "no_match"

    async def _generate_with_retry(
        self, messages: List[Dict], item_id: str
    ) -> Optional[str]:
        """Generate response with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                api_params = {
                    "model": self.server_configs[0].model_name,
                    "messages": messages,
                    "temperature": self.config.eval_temperature,
                }
                if self.config.eval_max_tokens > 0:
                    api_params["max_tokens"] = self.config.eval_max_tokens

                response = await self.client.chat.completions.create(**api_params)

                if response.choices and response.choices[0].message.content:
                    content = response.choices[0].message.content.strip()
                    if len(content) >= self.config.min_response_length:
                        return content

            except Exception as e:
                if self.config.full_debug:
                    print(f"  Error on item {item_id} attempt {attempt + 1}: {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))

        return None

    async def _evaluate_single_item(self, item: Dict, idx: int) -> Dict:
        """Evaluate a single BoolQ item."""
        # Format prompt
        prompt = self._format_prompt(item)

        # Build messages
        messages = []
        system_content = self._create_system_content()
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": prompt})

        # Generate response
        response = await self._generate_with_retry(messages, str(idx))

        if response is None:
            return {
                "index": idx,
                "is_correct": False,
                "extracted_answer": None,
                "gold_answer": item["answer"],
                "extraction_method": "generation_failed",
                "error": "Failed to generate response",
            }

        # Extract answer
        extracted_answer, extraction_method = self._extract_answer(response)

        # Gold answer (already Yes/No string)
        gold_answer = item["answer"]

        # Score
        is_correct = extracted_answer == gold_answer if extracted_answer else False

        result = {
            "index": idx,
            "is_correct": is_correct,
            "extracted_answer": extracted_answer,
            "gold_answer": gold_answer,
            "extraction_method": extraction_method,
        }

        if self.config.full_debug:
            result["response"] = response
            result["prompt"] = prompt

        return result

    async def evaluate(self, *args, **kwargs):
        """Run the full BoolQ evaluation."""
        print("\n" + "=" * 60)
        print("Starting BoolQ Evaluation (Generative/Reasoning Mode)")
        print("=" * 60)
        print(f"  Total questions: {len(self.eval_items)}")
        print(f"  Thinking mode: {self.config.thinking_mode}")
        print("=" * 60)

        # Evaluate all items
        tasks = [
            self._evaluate_single_item(item, idx)
            for idx, item in enumerate(self.eval_items)
        ]

        results = await tqdm_asyncio.gather(*tasks, desc="Evaluating BoolQ")

        # Calculate metrics
        valid_results = [r for r in results if r.get("gold_answer") is not None]

        if not valid_results:
            print("Warning: No valid evaluation results obtained")
            return

        correct = sum(1 for r in valid_results if r["is_correct"])
        total = len(valid_results)
        accuracy = correct / total if total > 0 else 0.0

        # Extraction method breakdown
        method_counts = {}
        for r in valid_results:
            method = r.get("extraction_method", "unknown")
            method_counts[method] = method_counts.get(method, 0) + 1

        # Yes/No breakdown
        yes_count = sum(1 for r in valid_results if r["gold_answer"] == "Yes")
        no_count = sum(1 for r in valid_results if r["gold_answer"] == "No")
        yes_correct = sum(
            1 for r in valid_results if r["gold_answer"] == "Yes" and r["is_correct"]
        )
        no_correct = sum(
            1 for r in valid_results if r["gold_answer"] == "No" and r["is_correct"]
        )

        # Print summary
        print("\n" + "=" * 60)
        print("BoolQ Evaluation Results")
        print("=" * 60)
        print(f"  Total evaluated: {total}")
        print(f"  Correct: {correct}")
        print(f"  Accuracy: {accuracy:.2%}")
        print("-" * 60)
        print(
            f"  Yes questions: {yes_count} (correct: {yes_correct}, acc: {yes_correct/yes_count:.2%})"
        )
        print(
            f"  No questions: {no_count} (correct: {no_correct}, acc: {no_correct/no_count:.2%})"
        )
        print("-" * 60)
        print("  Extraction Methods:")
        for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
            print(f"    {method}: {count} ({count/total:.1%})")
        print("=" * 60)

        # Save results
        metrics = {
            "accuracy": accuracy,
            "total_evaluated": total,
            "correct": correct,
            "yes_accuracy": yes_correct / yes_count if yes_count > 0 else 0.0,
            "no_accuracy": no_correct / no_count if no_count > 0 else 0.0,
            "extraction_methods": method_counts,
        }

        save_eval_results(self.config.data_dir_to_save_evals, metrics, results)

        self.eval_metrics = [
            {
                "accuracy": accuracy,
                "total": total,
            }
        ]

    async def wandb_log(self, step: int):
        """Log metrics to wandb."""
        if self.eval_metrics and wandb.run is not None:
            for metric in self.eval_metrics:
                wandb.log(metric, step=step)

    # Required BaseEnv interface methods
    async def get_next_item(self):
        return None

    async def collect_trajectories(self, *args, **kwargs):
        return []

    async def score(self, *args, **kwargs):
        return []


if __name__ == "__main__":
    BoolQEvalEnv.cli()
