"""
WinoGrande Evaluation Environment for Atropos

This environment evaluates models on the WinoGrande benchmark - testing
commonsense reasoning via fill-in-the-blank pronoun resolution.

Dataset: allenai/winogrande
Paper: https://arxiv.org/abs/1907.10641

WinoGrande tests:
- Commonsense reasoning
- Pronoun resolution (Winograd Schema Challenge inspired)
- Binary choice (A or B) for filling in the blank
- Adversarially filtered to reduce dataset biases

Metrics:
- Accuracy (exact match on A/B)

Supports optional thinking mode with <think></think> tags.
"""

import asyncio
import os
import re
import time
from string import ascii_uppercase
from typing import Any, Dict, List, Optional, Tuple

import wandb
from datasets import load_dataset
from pydantic import Field
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
)
from eval_helpers import (
    extract_letter_from_answer_tag,
    validate_thinking_format,
    extract_thinking_content,
    get_default_thinking_prompt,
    create_system_content,
    save_eval_results,
    build_mcqa_fallback_patterns,
)


class WinoGrandeEvalConfig(BaseEnvConfig):
    """Configuration for WinoGrande evaluation environment."""

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
        default="allenai/winogrande",
        description="HuggingFace dataset name for WinoGrande.",
    )

    subset: str = Field(
        default="winogrande_xl",
        description="Dataset subset (winogrande_xl, winogrande_l, winogrande_m, winogrande_s, winogrande_xs).",
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


class WinoGrandeEvalEnv(BaseEnv):
    """
    WinoGrande Evaluation Environment for Atropos.
    
    Evaluates models on commonsense reasoning with binary choice pronoun resolution.
    """
    
    name = "winogrande_eval"
    env_config_cls = WinoGrandeEvalConfig

    def __init__(
        self,
        config: WinoGrandeEvalConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: WinoGrandeEvalConfig = config
        self.eval_metrics = []
        
        # Pre-build fallback patterns for 2-choice (A/B)
        self._fallback_patterns = build_mcqa_fallback_patterns(2)
        self._valid_letters = {'A', 'B'}

    @classmethod
    def config_init(cls) -> Tuple[WinoGrandeEvalConfig, List[APIServerConfig]]:
        """Initialize default configuration for CLI usage."""
        config = WinoGrandeEvalConfig(
            tokenizer_name="NousResearch/Hermes-3-Llama-3.1-8B",
            group_size=1,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=1,
            batch_size=1,
            steps_per_eval=1,
            max_token_length=2048,
            wandb_name="winogrande_eval",
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
        """Load the WinoGrande dataset."""
        print(f"\nWinoGrande Evaluation Setup (Generative Mode):")
        print(f"  Dataset: {self.config.dataset_name}")
        print(f"  Subset: {self.config.subset}")
        print(f"  Evaluation split: {self.config.eval_split}")
        print(f"  Thinking mode: {self.config.thinking_mode}")
        if self.config.thinking_mode:
            print(f"  Thinking prompt: {get_default_thinking_prompt(self.config.custom_thinking_prompt)[:80]}...")
        
        # Load dataset
        self.dataset = load_dataset(
            self.config.dataset_name,
            self.config.subset,
            split=self.config.eval_split,
            trust_remote_code=True,
        )
        
        self.eval_items = list(self.dataset)
        print(f"  Loaded {len(self.eval_items)} evaluation items")

    def _format_prompt(self, item: Dict) -> Tuple[str, List[str]]:
        """
        Format a WinoGrande item into a prompt.
        
        The sentence contains an underscore "_" that should be filled with one of two options.
        
        Returns the formatted prompt and list of choice texts.
        """
        sentence = item['sentence']
        option1 = item['option1']
        option2 = item['option2']
        
        # Build the question
        query = "The following is a fill-in-the-blank question about commonsense reasoning.\n\n"
        query += f"Sentence: {sentence}\n\n"
        query += "Choose the word or phrase that best fills the blank:\n"
        query += f"A. {option1}\n"
        query += f"B. {option2}\n"
        query += "\nProvide your answer in <answer></answer> tags with only the letter (A or B)."
        
        return query, [option1, option2]

    def _create_system_content(self) -> Optional[str]:
        """Create system message content based on thinking mode configuration."""
        return create_system_content(
            self.config.thinking_mode,
            self.config.custom_thinking_prompt,
            self.config.custom_system_prompt
        )

    def _extract_answer(self, response: str, choices: List[str] = None) -> Tuple[Optional[str], str]:
        """
        Extract the answer letter from the model's response.
        
        Uses <answer> tags as primary method, with fallback patterns.
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
        
        # Primary: Try <answer></answer> tags
        letter, method = extract_letter_from_answer_tag(
            response_to_parse,
            self._valid_letters,
            debug=self.config.full_debug,
            choices=choices
        )
        if letter:
            return letter, method
        
        # Fallback: Use regex patterns
        for priority, pattern, method_name in self._fallback_patterns:
            matches = pattern.findall(response_to_parse)
            if matches:
                match = matches[-1] if method_name in ["final_answer_is", "the_answer_is", "answer_colon", "answer_space"] else matches[0]
                if isinstance(match, tuple):
                    match = match[0]
                letter = match.strip("()").upper()
                if letter in self._valid_letters:
                    return letter, f"fallback_{method_name}"
        
        return None, "no_match"

    async def _generate_with_retry(self, messages: List[Dict], item_id: str) -> Optional[str]:
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
        """Evaluate a single WinoGrande item."""
        # Format prompt
        prompt, choices = self._format_prompt(item)
        
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
                "gold_answer": ascii_uppercase[int(item['answer']) - 1] if item['answer'] != '' else None,
                "extraction_method": "generation_failed",
                "error": "Failed to generate response",
            }
        
        # Extract answer
        extracted_answer, extraction_method = self._extract_answer(response, choices)
        
        # Determine gold answer (WinoGrande uses 1/2, we convert to A/B)
        gold_answer = None
        if item['answer'] != '':
            gold_idx = int(item['answer']) - 1  # Convert 1-indexed to 0-indexed
            gold_answer = ascii_uppercase[gold_idx]
        
        # Score
        is_correct = extracted_answer == gold_answer if extracted_answer and gold_answer else False
        
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
        """Run the full WinoGrande evaluation."""
        print("\n" + "=" * 60)
        print("Starting WinoGrande Evaluation (Generative/Reasoning Mode)")
        print("=" * 60)
        print(f"  Subset: {self.config.subset}")
        print(f"  Total questions: {len(self.eval_items)}")
        print(f"  Thinking mode: {self.config.thinking_mode}")
        print("=" * 60)
        
        # Evaluate all items
        tasks = [
            self._evaluate_single_item(item, idx)
            for idx, item in enumerate(self.eval_items)
        ]
        
        results = await tqdm_asyncio.gather(*tasks, desc="Evaluating WinoGrande")
        
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
        
        # Print summary
        print("\n" + "=" * 60)
        print("WinoGrande Evaluation Results")
        print("=" * 60)
        print(f"  Subset: {self.config.subset}")
        print(f"  Total evaluated: {total}")
        print(f"  Correct: {correct}")
        print(f"  Accuracy: {accuracy:.2%}")
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
            "subset": self.config.subset,
            "extraction_methods": method_counts,
        }
        
        save_eval_results(
            self.config.data_dir_to_save_evals,
            metrics,
            results
        )
        
        self.eval_metrics = [{
            "accuracy": accuracy,
            "total": total,
        }]

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
    WinoGrandeEvalEnv.cli()

