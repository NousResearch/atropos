"""
DROP (Discrete Reasoning Over Paragraphs) Evaluation Environment for Atropos

This environment evaluates models on the DROP benchmark - testing reading
comprehension with complex reasoning (numerical, date, span extraction).

Dataset: lighteval/drop_harness
Paper: https://arxiv.org/abs/1810.00505

DROP requires models to:
- Read a passage and answer questions
- Perform discrete reasoning (counting, sorting, arithmetic)
- Extract numbers, dates, or text spans as answers

Metrics:
- Exact match accuracy
- F1 score (token-level)

Supports optional thinking mode with <think></think> tags.
"""

import asyncio
import os
import re
import time
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
    extract_freeform_from_answer_tag,
    validate_thinking_format,
    extract_thinking_content,
    get_default_thinking_prompt,
    create_system_content,
    save_eval_results,
)


class DROPEvalConfig(BaseEnvConfig):
    """Configuration for DROP evaluation environment."""

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
        default="lighteval/drop_harness",
        description="HuggingFace dataset name for DROP.",
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


class DROPEvalEnv(BaseEnv):
    """
    DROP Evaluation Environment for Atropos.
    
    Evaluates models on reading comprehension with discrete reasoning.
    """
    
    name = "drop_eval"
    env_config_cls = DROPEvalConfig

    def __init__(
        self,
        config: DROPEvalConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: DROPEvalConfig = config
        self.eval_metrics = []
        
        # Regex patterns for thinking mode
        self._think_pattern = re.compile(r"<think>")
        self._think_close_pattern = re.compile(r"</think>")
        self._think_content_pattern = re.compile(r"</think>\s*(.*)", re.DOTALL)
        self._thinking_extract_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        
        # Pre-compile regex for <answer></answer> tag extraction (primary method)
        self._answer_tag_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
        
        # Build fallback answer extraction patterns for freeform answers
        self._build_extraction_patterns()

    def _build_extraction_patterns(self):
        """
        Build regex patterns for extracting freeform answers from model responses.
        
        DROP answers can be numbers, dates, or text spans.
        Patterns are ordered by priority (lower = higher priority).
        """
        # Priority 0: "final answer is: X" 
        self._pattern_final_answer = re.compile(
            r"(?i:final\s+answer\s+is)\s*:?\s*(.+?)(?:\.|$|\n)",
            re.IGNORECASE
        )
        
        # Priority 50: "the answer is X"
        self._pattern_the_answer_is = re.compile(
            r"(?i:the\s+answer\s+is)\s*:?\s*(.+?)(?:\.|$|\n)",
            re.IGNORECASE
        )
        
        # Priority 100: "answer: X" (with colon)
        self._pattern_answer_colon = re.compile(
            r"(?i:answer)\s*:\s*(.+?)(?:\.|$|\n)",
            re.IGNORECASE
        )
        
        # Priority 150: "answer X" (without colon, careful with this one)
        self._pattern_answer_space = re.compile(
            r"(?i:answer)\s+(.+?)(?:\.|$|\n)",
            re.IGNORECASE
        )
        
        # Priority 200: Number at end (common for DROP)
        self._pattern_number_end = re.compile(
            r"(\d+(?:\.\d+)?)\s*$"
        )
        
        # Priority 250: Date pattern
        self._pattern_date = re.compile(
            r"(\d{1,2}\s+\w+\s+\d{4}|\w+\s+\d{1,2},?\s+\d{4})\s*$"
        )
        
        # Store patterns in priority order
        self._extraction_patterns = [
            (0, self._pattern_final_answer, "final_answer"),
            (50, self._pattern_the_answer_is, "the_answer_is"),
            (100, self._pattern_answer_colon, "answer_colon"),
            (150, self._pattern_answer_space, "answer_space"),
            (200, self._pattern_number_end, "number_end"),
            (250, self._pattern_date, "date"),
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
    def config_init(cls) -> Tuple[DROPEvalConfig, List[APIServerConfig]]:
        """Initialize default configuration for the environment."""
        env_config = DROPEvalConfig(
            tokenizer_name="NousResearch/Hermes-3-Llama-3.1-8B",
            group_size=1,
            use_wandb=True,
            max_num_workers_per_node=128,
            rollout_server_url="http://localhost:8000",
            total_steps=1,
            batch_size=1,
            steps_per_eval=1,
            inference_weight=1.0,
            wandb_name="drop_eval",
            eval_handling=EvalHandlingEnum.STOP_TRAIN,
            max_eval_workers=256,
            max_num_workers=1024,
            dataset_name="lighteval/drop_harness",
            eval_split="validation",
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

    def _parse_drop_answer(self, answer_dict: Dict) -> Tuple[str, ...]:
        """Parse a DROP answer dict into a tuple of strings."""
        if answer_dict.get("number", "") != "":
            return (str(answer_dict["number"]),)
        if answer_dict.get("spans", []):
            return tuple(answer_dict["spans"])
        # Date handling
        date = answer_dict.get("date", {})
        if date:
            date_str = " ".join([
                str(date.get("day", "")),
                str(date.get("month", "")),
                str(date.get("year", ""))
            ]).strip()
            if date_str:
                return (date_str,)
        return ("",)

    def _get_all_valid_answers(self, item: Dict) -> List[Tuple[str, ...]]:
        """Get all valid answer tuples from a DROP item."""
        def _flatten_validated_answers(validated_answers):
            valid_answers = []
            for i in range(len(validated_answers.get("number", []))):
                valid_answers.append({
                    "number": validated_answers["number"][i],
                    "date": validated_answers["date"][i],
                    "spans": validated_answers["spans"][i],
                })
            return valid_answers
        
        answers = []
        answers_set = set()
        
        # Primary answer
        candidates = [item.get("answer", {})]
        
        # Validated answers
        validated = item.get("validated_answers", {})
        if validated:
            candidates.extend(_flatten_validated_answers(validated))
        
        for candidate in candidates:
            answer = self._parse_drop_answer(candidate)
            if answer and answer not in answers_set:
                answers.append(answer)
                answers_set.add(answer)
        
        return answers

    def _format_drop_prompt(self, item: Dict) -> str:
        """Format a DROP item into a prompt with <answer> tag instruction.
        
        Based on lighteval format but with explicit <answer> tag instruction.
        """
        passage = item.get("passage", "")
        question = item.get("question", "")
        
        return f"""Read the passage and answer the question. Think step by step before answering.

Provide your final answer within <answer></answer> tags.

Example format:
<answer>42</answer>

Passage: {passage}

Question: {question}"""

    async def setup(self) -> None:
        """Load the DROP dataset and prepare for evaluation."""
        print(f"\nDROP Evaluation Setup:")
        print(f"  Dataset: {self.config.dataset_name}")
        print(f"  Max tokens: {self.config.eval_max_tokens}")
        print(f"  Evaluation split: {self.config.eval_split}")
        print(f"  Thinking mode: {self.config.thinking_mode}")
        if self.config.thinking_mode:
            print(f"  Thinking prompt: {self._get_thinking_prompt()[:100]}...")
        
        try:
            dataset = load_dataset(
                self.config.dataset_name,
                split=self.config.eval_split,
            )
            self.eval_data = list(dataset)
            print(f"  Loaded {len(self.eval_data)} evaluation items")
                
        except Exception as e:
            print(f"Error loading DROP dataset: {e}")
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

    def _normalize_answer(self, answer: str) -> str:
        """Normalize an answer string for comparison."""
        # Remove articles, extra whitespace, lowercase
        answer = answer.lower().strip()
        answer = re.sub(r'\b(a|an|the)\b', '', answer)
        answer = re.sub(r'\s+', ' ', answer).strip()
        # Remove punctuation at the end
        answer = answer.rstrip('.,!?;:')
        return answer

    def _compute_f1(self, prediction: str, gold: str) -> float:
        """Compute token-level F1 score between prediction and gold."""
        pred_tokens = set(self._normalize_answer(prediction).split())
        gold_tokens = set(self._normalize_answer(gold).split())
        
        if not pred_tokens or not gold_tokens:
            return 0.0
        
        common = pred_tokens & gold_tokens
        if not common:
            return 0.0
        
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gold_tokens)
        
        return 2 * precision * recall / (precision + recall)

    def _check_exact_match(self, prediction: str, valid_answers: List[Tuple[str, ...]]) -> Tuple[bool, float]:
        """Check if prediction exactly matches any valid answer. Returns (match, best_f1)."""
        pred_norm = self._normalize_answer(prediction)
        best_f1 = 0.0
        
        for answer_tuple in valid_answers:
            for answer in answer_tuple:
                gold_norm = self._normalize_answer(answer)
                
                # Exact match check
                if pred_norm == gold_norm:
                    return True, 1.0
                
                # Check if prediction contains the answer or vice versa
                if gold_norm in pred_norm or pred_norm in gold_norm:
                    return True, 1.0
                
                # Compute F1
                f1 = self._compute_f1(prediction, answer)
                best_f1 = max(best_f1, f1)
        
        return False, best_f1

    def _extract_answer_from_response(self, response: str) -> Tuple[str, str]:
        """
        Extract the answer from model response using priority-ordered patterns.
        
        Primary method: Look for <answer></answer> tags (only take the first match).
        Fallback: Use priority-ordered regex patterns.
        
        Args:
            response: The model's response string (content after </think> in thinking mode)
            
        Returns:
            Tuple of (extracted_answer, extraction_method used)
        """
        if not response:
            return "", "empty_response"
        
        # PRIMARY: Try <answer></answer> tags first
        answer, method = extract_freeform_from_answer_tag(
            response, debug=self.config.full_debug
        )
        if answer:
            return answer, method
        
        # FALLBACK: Try each pattern in priority order
        for priority, pattern, method_name in self._extraction_patterns:
            matches = pattern.findall(response)
            if matches:
                # Get the LAST match for answer patterns since final answer is most reliable
                match = matches[-1] if method_name in ["final_answer", "the_answer_is", "answer_colon", "answer_space"] else matches[0]
                
                answer = match.strip() if isinstance(match, str) else str(match).strip()
                if answer:
                    if self.config.full_debug:
                        print(f"    Extracted '{answer[:50]}' using fallback method '{method_name}' (priority {priority})")
                    return answer, f"fallback_{method_name}"
        
        # Fallback: first non-empty line
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line:
                if self.config.full_debug:
                    print(f"    Extracted '{line[:50]}' using fallback 'first_line'")
                return line, "fallback_first_line"
        
        return response.strip(), "fallback_full_response"

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
        """Evaluate a single DROP question."""
        try:
            prompt = self._format_drop_prompt(eval_item)
            valid_answers = self._get_all_valid_answers(eval_item)
            
            if not prompt or not valid_answers:
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
            
            # Extract answer from response
            extracted_answer, extraction_method = self._extract_answer_from_response(response_for_eval)
            
            # Check correctness
            is_correct, f1_score = self._check_exact_match(extracted_answer, valid_answers)
            
            # Format gold answers for display
            gold_display = ", ".join([" | ".join(a) for a in valid_answers[:3]])
            
            sample = {
                "passage": eval_item.get("passage", "")[:200] + "...",
                "question": eval_item.get("question", ""),
                "gold_answers": gold_display,
                "model_response": model_response[:500] if len(model_response) > 500 else model_response,
                "extracted_answer": extracted_answer,
                "extraction_method": extraction_method,
                "is_correct": is_correct,
                "f1_score": f1_score,
                "finish_reason": finish_reason,
                "thinking_format_valid": thinking_format_valid,
            }
            
            if self.config.thinking_mode:
                sample["thinking_content"] = thinking_content[:300] + "..." if thinking_content and len(thinking_content) > 300 else thinking_content
            
            if self.config.full_debug:
                status = "✓" if is_correct else "✗"
                print(f"  [{status}] Q: {eval_item.get('question', '')[:50]}... | A: {extracted_answer[:30]}...")
            
            return {
                "result": {"correct": is_correct, "f1": f1_score},
                "sample": sample
            }
            
        except Exception as e:
            if self.config.full_debug:
                print(f"Error in rollout_and_score_eval: {e}")
                import traceback
                traceback.print_exc()
            return {"result": None, "sample": None}

    async def evaluate(self, *args, **kwargs) -> None:
        """Run DROP evaluation."""
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"Starting DROP Evaluation")
        print(f"{'='*60}")
        print(f"  Total questions: {len(self.all_eval_items)}")
        print(f"  Thinking mode: {self.config.thinking_mode}")
        print(f"{'='*60}\n")
        
        try:
            eval_tasks = [
                self.rollout_and_score_eval(item) for item in self.all_eval_items
            ]
            results = await tqdm_asyncio.gather(*eval_tasks, desc="Evaluating DROP")
            
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
        
        avg_f1 = sum(s.get("f1_score", 0.0) for s in samples) / total_count if total_count > 0 else 0.0
        
        # Thinking metrics
        thinking_format_compliant = sum(1 for s in samples if s.get("thinking_format_valid", True))
        thinking_format_compliance_rate = thinking_format_compliant / total_count if total_count > 0 else 0.0
        
        thinking_utilization = sum(1 for s in samples if s.get("thinking_content")) if self.config.thinking_mode else 0
        
        eval_metrics = {
            "eval/accuracy": accuracy,
            "eval/avg_f1": avg_f1,
            "eval/correct_count": correct_count,
            "eval/total_count": total_count,
            "eval/evaluation_time_seconds": end_time - start_time,
            "eval/thinking_mode_enabled": 1.0 if self.config.thinking_mode else 0.0,
        }
        
        if self.config.thinking_mode:
            eval_metrics["eval/thinking_format_compliance_rate"] = thinking_format_compliance_rate
            eval_metrics["eval/thinking_utilization_rate"] = thinking_utilization / total_count if total_count > 0 else 0.0
        
        self.eval_metrics = [(k, v) for k, v in eval_metrics.items()]
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"DROP Evaluation Results")
        print(f"{'='*60}")
        print(f"Exact Match Accuracy: {accuracy:.4f} ({correct_count}/{total_count})")
        print(f"Average F1 Score: {avg_f1:.4f}")
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
    DROPEvalEnv.cli()

