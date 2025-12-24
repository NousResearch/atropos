"""
OpenBookQA (OBQA) Evaluation Environment for Atropos

This environment evaluates models on the OpenBookQA benchmark - testing
common sense reasoning with multiple-choice questions.

Dataset: allenai/openbookqa
Paper: https://arxiv.org/abs/1809.02789

OpenBookQA tests:
- Combining facts from an "open book" with common knowledge
- Common sense reasoning about everyday situations
- Multiple choice questions (A, B, C, D)

Metrics:
- Accuracy (exact match on A/B/C/D)

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


class OBQAEvalConfig(BaseEnvConfig):
    """Configuration for OpenBookQA evaluation environment."""

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
        default="allenai/openbookqa",
        description="HuggingFace dataset name for OpenBookQA.",
    )

    subset: str = Field(
        default="main",
        description="Dataset subset to use.",
    )

    eval_split: str = Field(
        default="test",
        description="Dataset split to use for evaluation (test or validation).",
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


class OBQAEvalEnv(BaseEnv):
    """
    OpenBookQA Evaluation Environment for Atropos.
    
    Evaluates models on common sense reasoning with multiple choice questions.
    """
    
    name = "obqa_eval"
    env_config_cls = OBQAEvalConfig

    def __init__(
        self,
        config: OBQAEvalConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: OBQAEvalConfig = config
        self.eval_metrics = []
        
        # Regex patterns for thinking mode
        self._think_pattern = re.compile(r"<think>")
        self._think_close_pattern = re.compile(r"</think>")
        self._think_content_pattern = re.compile(r"</think>\s*(.*)", re.DOTALL)
        self._thinking_extract_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        
        # Pre-compile regex for <answer></answer> tag extraction (primary method)
        self._answer_tag_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
        
        # Build fallback answer extraction patterns
        self._build_extraction_patterns()

    def _build_extraction_patterns(self):
        """
        Build regex patterns for extracting answer letters from model responses.
        
        Patterns are ordered by priority (lower number = higher priority).
        Takes the LAST match for answer patterns since models often repeat the final answer.
        """
        # Valid answer letters for OBQA (A-D)
        letters = "ABCD"
        letter_pattern = rf"([{letters}]|\([{letters}]\))"
        
        # Priority 0: "final answer is: X" with "I hope" (very specific, highest confidence)
        self._pattern_final_answer_hope = re.compile(
            rf"(?i:final\s+answer\s+is)\s*:?\s*{letter_pattern}\.?\s*I\s*hope",
            re.IGNORECASE
        )
        
        # Priority 50: "final answer ... is X" (allows text between)
        self._pattern_final_answer_is = re.compile(
            rf"(?i:final\s+answer).{{0,100}}?\s+is\s*:?\s*{letter_pattern}",
            re.IGNORECASE | re.DOTALL
        )
        
        # Priority 75: "the answer is X"
        self._pattern_the_answer_is = re.compile(
            rf"(?i:the\s+answer\s+is)\s*:?\s*{letter_pattern}",
            re.IGNORECASE
        )
        
        # Priority 100: "answer: X" or "Answer: X" (with colon)
        self._pattern_answer_colon = re.compile(
            rf"(?i:answer)\s*:\s*.{{0,50}}?{letter_pattern}",
            re.IGNORECASE | re.DOTALL
        )
        
        # Priority 150: "answer X" or "Answer X" (without colon)
        self._pattern_answer_space = re.compile(
            rf"(?i:answer)\s+{letter_pattern}",
            re.IGNORECASE
        )
        
        # Priority 200: Response starts with answer letter (with optional punctuation)
        self._pattern_start = re.compile(
            rf"^\s*\**{letter_pattern}\**[\s\.\)\:]",
            re.IGNORECASE
        )
        
        # Priority 210: Letter at start of any line (for multi-line responses)
        self._pattern_line_start = re.compile(
            rf"\n\s*\**{letter_pattern}\**[\s\.\)\:]",
            re.IGNORECASE
        )
        
        # Priority 250: Standalone letter with word boundaries
        self._pattern_standalone = re.compile(
            rf"\b{letter_pattern}\b",
            re.IGNORECASE
        )
        
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

    def _get_thinking_prompt(self) -> str:
        """Get thinking system prompt."""
        return get_default_thinking_prompt(self.config.custom_thinking_prompt)

    def _create_system_content(self) -> Optional[str]:
        """Create system message content based on thinking mode."""
        return create_system_content(
            self.config.thinking_mode,
            self.config.custom_thinking_prompt,
            self.config.custom_system_prompt
        )

    @classmethod
    def config_init(cls) -> Tuple[OBQAEvalConfig, List[APIServerConfig]]:
        """Initialize default configuration for the environment."""
        env_config = OBQAEvalConfig(
            tokenizer_name="NousResearch/Hermes-3-Llama-3.1-8B",
            group_size=1,
            use_wandb=True,
            max_num_workers_per_node=128,
            rollout_server_url="http://localhost:8000",
            total_steps=1,
            batch_size=1,
            steps_per_eval=1,
            inference_weight=1.0,
            wandb_name="obqa_eval",
            eval_handling=EvalHandlingEnum.STOP_TRAIN,
            max_eval_workers=256,
            max_num_workers=1024,
            dataset_name="allenai/openbookqa",
            subset="main",
            eval_split="test",
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

    def _format_obqa_prompt(self, item: Dict) -> str:
        """Format an OpenBookQA item into a prompt with <answer> tag instruction.
        
        Based on lighteval's openbookqa_prompt but with explicit <answer> tag instruction.
        """
        question = item.get("question_stem", "")
        choices = item.get("choices", {})
        choice_texts = choices.get("text", [])
        num_choices = len(choice_texts)
        valid_letters = ", ".join(ascii_uppercase[:num_choices])
        
        query = "Answer the following multiple choice question about common sense. Think step by step before answering.\n\n"
        query += f"Provide your final answer within <answer></answer> tags, containing only the letter ({valid_letters}).\n\n"
        query += "Example format:\n<answer>A</answer>\n\n"
        query += f"Question: {question}\n"
        
        for i, choice_text in enumerate(choice_texts):
            letter = ascii_uppercase[i]
            query += f"{letter}. {choice_text}\n"
        
        return query

    async def setup(self) -> None:
        """Load the OpenBookQA dataset and prepare for evaluation."""
        print(f"\nOpenBookQA Evaluation Setup:")
        print(f"  Dataset: {self.config.dataset_name}")
        print(f"  Subset: {self.config.subset}")
        print(f"  Eval split: {self.config.eval_split}")
        print(f"  Max tokens: {self.config.eval_max_tokens}")
        print(f"  Thinking mode: {self.config.thinking_mode}")
        if self.config.thinking_mode:
            print(f"  Thinking prompt: {self._get_thinking_prompt()[:100]}...")
        
        try:
            dataset = load_dataset(
                self.config.dataset_name,
                self.config.subset,
                split=self.config.eval_split,
            )
            self.eval_data = list(dataset)
            print(f"  Loaded {len(self.eval_data)} evaluation items")
                
        except Exception as e:
            print(f"Error loading OpenBookQA dataset: {e}")
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

    def _extract_answer_letter(
        self, 
        response: str, 
        num_choices: int,
        choices: Optional[List[str]] = None
    ) -> Tuple[Optional[str], str]:
        """
        Extract the answer letter from the model's response.
        
        Primary method: Look for <answer></answer> tags, or match against choice texts.
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
        
        # FALLBACK: Try each pattern in priority order
        for priority, pattern, method_name in self._extraction_patterns:
            matches = pattern.findall(response)
            if matches:
                # Get the LAST match for patterns that typically appear at the end
                match = matches[-1] if method_name in ["final_answer_is", "the_answer_is", "answer_colon", "answer_space"] else matches[0]
                
                # Clean up the match (remove parentheses if present)
                if isinstance(match, tuple):
                    match = match[0]
                letter = match.strip("()").upper()
                
                # Validate it's a valid choice
                if letter in valid_letters:
                    if self.config.full_debug:
                        print(f"    Extracted '{letter}' using fallback method '{method_name}' (priority {priority})")
                    return letter, f"fallback_{method_name}"
        
        # Last resort: find any valid letter (take the last one as it's likely the answer)
        for letter in reversed(list(valid_letters)):
            if letter in response.upper():
                if self.config.full_debug:
                    print(f"    Extracted '{letter}' using fallback 'last_valid_letter'")
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
        """Evaluate a single OpenBookQA question."""
        try:
            prompt = self._format_obqa_prompt(eval_item)
            gold_answer = eval_item.get("answerKey", "").strip().upper()
            choices = eval_item.get("choices", {}).get("text", [])
            
            if not prompt or not gold_answer:
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
                        finish_reason = getattr(completion.choices[0], 'finish_reason', None)
                        
                        if len(model_response.strip()) >= self.config.min_response_length:
                            break
                        
                except Exception as e:
                    print(f"  API Error (attempt {attempt + 1}/{self.config.max_retries}): {type(e).__name__}: {e}")
                    if hasattr(e, 'response'):
                        try:
                            print(f"    Response: {e.response.text[:500] if hasattr(e.response, 'text') else e.response}")
                        except:
                            pass
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(self.config.retry_delay)
                    else:
                        return {"result": None, "sample": None}
            
            if not model_response:
                return {"result": None, "sample": None}
            
            # Handle thinking mode
            thinking_format_valid, response_for_eval = self._validate_thinking_format(model_response)
            thinking_content = None
            if self.config.thinking_mode:
                thinking_content = self._extract_thinking_content(model_response)
            
            # Extract answer (pass choices for exact text matching)
            extracted_answer, extraction_method = self._extract_answer_letter(
                response_for_eval, len(choices), choices=choices
            )
            
            # Check correctness
            is_correct = extracted_answer is not None and extracted_answer == gold_answer
            
            # Get gold choice text
            gold_index = ord(gold_answer) - ord('A') if gold_answer in 'ABCD' else -1
            gold_choice_text = choices[gold_index] if 0 <= gold_index < len(choices) else "N/A"
            
            sample = {
                "question": eval_item.get("question_stem", ""),
                "choices": {ascii_uppercase[i]: c for i, c in enumerate(choices)},
                "gold_answer": gold_answer,
                "gold_choice_text": gold_choice_text,
                "model_response": model_response[:500] if len(model_response) > 500 else model_response,
                "extracted_answer": extracted_answer,
                "extraction_method": extraction_method,
                "is_correct": is_correct,
                "finish_reason": finish_reason,
                "thinking_format_valid": thinking_format_valid,
            }
            
            if self.config.thinking_mode:
                sample["thinking_content"] = thinking_content[:300] + "..." if thinking_content and len(thinking_content) > 300 else thinking_content
            
            if self.config.full_debug:
                status = "✓" if is_correct else "✗"
                print(f"  [{status}] Q: {eval_item.get('question_stem', '')[:50]}... | Pred: {extracted_answer}, Gold: {gold_answer}")
            
            return {
                "result": {"correct": is_correct},
                "sample": sample
            }
            
        except Exception as e:
            if self.config.full_debug:
                print(f"Error in rollout_and_score_eval: {e}")
                import traceback
                traceback.print_exc()
            return {"result": None, "sample": None}

    async def evaluate(self, *args, **kwargs) -> None:
        """Run OpenBookQA evaluation."""
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"Starting OpenBookQA Evaluation")
        print(f"{'='*60}")
        print(f"  Total questions: {len(self.all_eval_items)}")
        print(f"  Thinking mode: {self.config.thinking_mode}")
        print(f"{'='*60}\n")
        
        try:
            eval_tasks = [
                self.rollout_and_score_eval(item) for item in self.all_eval_items
            ]
            results = await tqdm_asyncio.gather(*eval_tasks, desc="Evaluating OpenBookQA")
            
            valid_results = [
                r for r in results 
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
        extracted_count = sum(1 for s in samples if s.get("extracted_answer") is not None)
        extraction_rate = extracted_count / total_count if total_count > 0 else 0.0
        
        # Thinking metrics
        thinking_format_compliant = sum(1 for s in samples if s.get("thinking_format_valid", True))
        thinking_format_compliance_rate = thinking_format_compliant / total_count if total_count > 0 else 0.0
        
        thinking_utilization = sum(1 for s in samples if s.get("thinking_content")) if self.config.thinking_mode else 0
        
        eval_metrics = {
            "eval/accuracy": accuracy,
            "eval/correct_count": correct_count,
            "eval/total_count": total_count,
            "eval/answer_extraction_rate": extraction_rate,
            "eval/evaluation_time_seconds": end_time - start_time,
            "eval/thinking_mode_enabled": 1.0 if self.config.thinking_mode else 0.0,
        }
        
        if self.config.thinking_mode:
            eval_metrics["eval/thinking_format_compliance_rate"] = thinking_format_compliance_rate
            eval_metrics["eval/thinking_utilization_rate"] = thinking_utilization / total_count if total_count > 0 else 0.0
        
        self.eval_metrics = [(k, v) for k, v in eval_metrics.items()]
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"OpenBookQA Evaluation Results")
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
        
        wandb_metrics["config/thinking_mode"] = 1.0 if self.config.thinking_mode else 0.0
        wandb_metrics["config/eval_max_tokens"] = self.config.eval_max_tokens
        
        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    OBQAEvalEnv.cli()

