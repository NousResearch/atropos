"""
HLE (Humanity's Last Exam) Evaluation Environment for Atropos (Generative Mode)

This environment evaluates models on Humanity's Last Exam - a collaborative
benchmark with questions from ~1000 subject experts across 500+ institutions.

Dataset: cais/hle
Paper: https://arxiv.org/abs/2501.14249

The evaluation follows a generative approach:
- Models receive challenging questions from expert contributors
- Expected output format: reasoning, answer, and optional confidence
- Supports thinking mode with <think></think> tags for extended reasoning
- Uses string matching for evaluation (not LLM judge like original)

Note: This implementation uses the text-only questions (filters out image questions).
"""

import asyncio
import os
import random
import re
import time
from typing import Dict, List, Optional, Tuple

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
    ANSWER_TAG_PATTERN,
    validate_thinking_format,
    extract_thinking_content,
    get_default_thinking_prompt,
    create_system_content,
    save_eval_results,
)


# Prompt template for HLE with answer tag instruction
HLE_PROMPT_TEMPLATE = """Answer the following challenging question. Think step by step and reason carefully before providing your answer.

Provide your final answer within <answer></answer> tags.

Example format:
<answer>42</answer>

Question: {question}"""


class HLEEvalConfig(BaseEnvConfig):
    """Configuration for HLE evaluation environment."""
    
    # Dataset configuration
    dataset_name: str = Field(
        default="cais/hle",
        description="HuggingFace dataset name"
    )
    subset: str = Field(
        default="default",
        description="Dataset subset"
    )
    eval_split: str = Field(
        default="test",
        description="Split to evaluate on"
    )
    shuffle_seed: int = Field(
        default=42,
        description="Random seed for shuffling"
    )
    
    # Generation parameters
    eval_temperature: float = Field(
        default=0.6,
        description="Temperature for evaluation generation"
    )
    eval_max_tokens: int = Field(
        default=0,
        description="Max tokens for evaluation (0 = use model default)"
    )
    
    # System prompt configuration
    custom_system_prompt: Optional[str] = Field(
        default=None,
        description="Optional custom system prompt"
    )
    
    # Thinking mode configuration
    thinking_mode: bool = Field(
        default=True,
        description="Whether to use thinking mode with <think></think> tags"
    )
    custom_thinking_prompt: Optional[str] = Field(
        default=None,
        description="Optional custom thinking prompt"
    )
    
    # Matching configuration
    fuzzy_match: bool = Field(
        default=True,
        description="Allow fuzzy matching (substring containment)"
    )
    case_sensitive: bool = Field(
        default=False,
        description="Whether matching should be case-sensitive"
    )
    
    # Retry and debug configuration
    max_retries: int = Field(
        default=3,
        description="Maximum retries for failed API calls"
    )
    retry_delay: float = Field(
        default=1.0,
        description="Delay between retries in seconds"
    )
    min_response_length: int = Field(
        default=1,
        description="Minimum response length to consider valid"
    )
    full_debug: bool = Field(
        default=False,
        description="Enable full debug output"
    )
    
    # Override defaults
    group_size: int = 1
    max_num_workers: int = 1024
    max_eval_workers: int = 256
    max_num_workers_per_node: int = 128
    use_wandb: bool = True
    rollout_server_url: str = "http://localhost:8000"
    total_steps: int = 1
    wandb_name: str = "hle_eval"
    steps_per_eval: int = 1


class HLEEvalEnv(BaseEnv):
    """
    HLE (Humanity's Last Exam) Evaluation Environment.
    
    Evaluates models on extremely challenging questions from expert contributors.
    Uses generative evaluation with flexible string matching.
    """

    name = "hle_eval"

    def __init__(
        self,
        config: HLEEvalConfig,
        server_configs: List[APIServerConfig],
        slurm_job_id: Optional[str] = None,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm_job_id, testing)
        self.config: HLEEvalConfig = config
        self.eval_items: List[Dict] = []
        self._dataset_loaded = False

    @classmethod
    def config_cls(cls) -> type:
        return HLEEvalConfig

    async def setup(self) -> None:
        """Initialize the environment and load the dataset."""
        await super().setup()
        
        if not self._dataset_loaded:
            await self._load_dataset()
            
        print(f"\nHLE Evaluation Setup (Generative Mode):")
        print(f"  Dataset: {self.config.dataset_name}")
        print(f"  Evaluation split: {self.config.eval_split}")
        print(f"  Thinking mode: {self.config.thinking_mode}")
        print(f"  Fuzzy matching: {self.config.fuzzy_match}")
        if self.config.thinking_mode:
            thinking_prompt = get_default_thinking_prompt(self.config.custom_thinking_prompt)
            print(f"  Thinking prompt: {thinking_prompt[:80]}...")
        print(f"  Loaded {len(self.eval_items)} text-only evaluation items")

    async def _load_dataset(self) -> None:
        """Load and process the HLE dataset."""
        print(f"Loading HLE dataset: {self.config.dataset_name}...")
        
        try:
            dataset = load_dataset(
                self.config.dataset_name,
                self.config.subset,
                trust_remote_code=True
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
        
        # Process items - filter to text-only questions
        self.eval_items = []
        skipped_image = 0
        
        for idx, item in enumerate(split_data):
            # Filter out image questions
            image = item.get("image")
            if image is not None and image != "":
                skipped_image += 1
                continue
            
            question = item.get("question", "")
            answer = item.get("answer", "")
            
            if not question or not answer:
                continue
            
            self.eval_items.append({
                "id": str(idx),
                "question": question,
                "answer": answer,
                "category": item.get("category", ""),
                "source": item.get("source", ""),
            })
        
        print(f"Filtered out {skipped_image} image questions")
        
        # Shuffle with seed
        random.seed(self.config.shuffle_seed)
        random.shuffle(self.eval_items)
        
        self._dataset_loaded = True
        print(f"Loaded {len(self.eval_items)} text-only evaluation items")

    def _format_prompt(self, item: Dict) -> str:
        """Format the question into a prompt."""
        return HLE_PROMPT_TEMPLATE.format(question=item["question"])

    def _create_system_content(self) -> str:
        """Create system message content based on thinking mode."""
        return create_system_content(
            self.config.thinking_mode,
            self.config.custom_thinking_prompt,
            self.config.custom_system_prompt
        ) or ""

    def _normalize_answer(self, answer: str) -> str:
        """Normalize an answer for comparison."""
        if not answer:
            return ""
        
        normalized = answer.strip()
        
        if not self.config.case_sensitive:
            normalized = normalized.lower()
        
        # Remove common punctuation at ends
        normalized = normalized.strip(".,;:!?\"'")
        
        # Normalize whitespace
        normalized = " ".join(normalized.split())
        
        return normalized

    def _check_match(self, predicted: str, gold: str) -> Tuple[bool, str]:
        """
        Check if the predicted answer matches the gold answer.
        
        Returns:
            Tuple of (is_match, match_type)
        """
        pred_norm = self._normalize_answer(predicted)
        gold_norm = self._normalize_answer(gold)
        
        if not pred_norm:
            return False, "empty_prediction"
        
        # Exact match
        if pred_norm == gold_norm:
            return True, "exact"
        
        # Fuzzy matching if enabled
        if self.config.fuzzy_match:
            # Check if gold is contained in prediction
            if gold_norm in pred_norm:
                return True, "gold_in_pred"
            
            # Check if prediction is contained in gold
            if pred_norm in gold_norm:
                return True, "pred_in_gold"
            
            # Check for numeric equivalence (e.g., "42" vs "42.0")
            try:
                pred_num = float(pred_norm.replace(",", ""))
                gold_num = float(gold_norm.replace(",", ""))
                if abs(pred_num - gold_num) < 1e-6:
                    return True, "numeric_equiv"
            except (ValueError, TypeError):
                pass
        
        return False, "no_match"

    def _extract_answer(self, response: str, debug: bool = False) -> Tuple[Optional[str], str]:
        """
        Extract the answer from the response.
        
        Args:
            response: The model's response (content after </think> in thinking mode)
            debug: Whether to print debug information
            
        Returns:
            Tuple of (extracted_answer or None, extraction_method)
        """
        if not response:
            return None, "empty_response"
        
        # Try <answer></answer> tags first
        answer_tag_match = ANSWER_TAG_PATTERN.search(response)
        if answer_tag_match:
            answer_content = answer_tag_match.group(1).strip()
            if answer_content:
                if debug:
                    preview = answer_content[:50] + "..." if len(answer_content) > 50 else answer_content
                    print(f"    Extracted '{preview}' using method 'answer_tag'")
                return answer_content, "answer_tag"
        
        # Fallback: Look for "Answer: X" pattern
        patterns = [
            r"(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*(.+?)(?:\n|$)",
            r"(?:my\s+)?answer\s*:\s*(.+?)(?:\n|$)",
            r"(?:so\s+)?the\s+answer\s+is\s*:?\s*(.+?)(?:\n|$)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                if answer:
                    if debug:
                        preview = answer[:50] + "..." if len(answer) > 50 else answer
                        print(f"    Extracted '{preview}' using fallback pattern")
                    return answer, "fallback_pattern"
        
        # Last resort: take the last line/sentence
        lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
        if lines:
            last_line = lines[-1]
            # Clean up common prefixes
            for prefix in ["Therefore,", "Thus,", "So,", "Hence,"]:
                if last_line.startswith(prefix):
                    last_line = last_line[len(prefix):].strip()
            
            if debug:
                preview = last_line[:50] + "..." if len(last_line) > 50 else last_line
                print(f"    Extracted '{preview}' using fallback last_line")
            return last_line, "fallback_last_line"
        
        return None, "no_match"

    async def rollout_and_score_eval(
        self,
        item: Dict,
        server: APIServerConfig,
    ) -> Optional[Dict]:
        """
        Run evaluation on a single item and return the result.
        """
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
            response_text,
            self.config.thinking_mode
        )
        
        # Extract thinking content if present
        thinking_content = extract_thinking_content(response_text) if self.config.thinking_mode else None
        
        # Extract answer from appropriate content
        extracted_answer, extraction_method = self._extract_answer(
            content_for_extraction,
            debug=self.config.full_debug
        )
        
        # Check match
        gold_answer = item["answer"]
        is_correct, match_type = self._check_match(extracted_answer, gold_answer) if extracted_answer else (False, "no_extraction")
        
        if self.config.full_debug:
            print(f"\n--- Item: {item['id']} ---")
            print(f"Question: {item['question'][:100]}...")
            print(f"Gold answer: {gold_answer[:100]}...")
            print(f"Extracted: {extracted_answer[:100] if extracted_answer else 'None'}...")
            print(f"Match: {is_correct} ({match_type})")
        
        return {
            "item_id": item["id"],
            "question": item["question"],
            "category": item.get("category", ""),
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
        """
        Run the full HLE evaluation.
        """
        print(f"\n{'='*60}")
        print("Starting HLE Evaluation (Generative Mode)")
        print(f"{'='*60}")
        print(f"  Total questions: {len(self.eval_items)}")
        print(f"  Thinking mode: {self.config.thinking_mode}")
        print(f"  Fuzzy matching: {self.config.fuzzy_match}")
        print(f"{'='*60}\n")
        
        # Create evaluation tasks
        async def eval_task(item):
            return await self.rollout_and_score_eval(item, self.server_configs[0])
        
        tasks = [eval_task(item) for item in self.eval_items]
        
        # Run with progress bar
        results = await tqdm_asyncio.gather(
            *tasks,
            desc="Evaluating HLE"
        )
        
        # Filter out failed results
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            print("Warning: No valid evaluation results obtained")
            return {"error": "No valid results", "accuracy": 0.0}
        
        # Calculate metrics
        total = len(valid_results)
        correct = sum(1 for r in valid_results if r["is_correct"])
        accuracy = correct / total if total > 0 else 0.0
        
        # Calculate per-category metrics
        category_metrics = {}
        for r in valid_results:
            cat = r.get("category", "unknown") or "unknown"
            if cat not in category_metrics:
                category_metrics[cat] = {"total": 0, "correct": 0}
            category_metrics[cat]["total"] += 1
            if r["is_correct"]:
                category_metrics[cat]["correct"] += 1
        
        for cat in category_metrics:
            cat_total = category_metrics[cat]["total"]
            cat_correct = category_metrics[cat]["correct"]
            category_metrics[cat]["accuracy"] = cat_correct / cat_total if cat_total > 0 else 0.0
        
        # Format compliance and thinking utilization
        format_valid = sum(1 for r in valid_results if r.get("format_valid", True))
        has_thinking = sum(1 for r in valid_results if r.get("has_thinking", False))
        
        # Match type breakdown
        match_counts = {}
        for r in valid_results:
            match_type = r.get("match_type", "unknown")
            match_counts[match_type] = match_counts.get(match_type, 0) + 1
        
        metrics = {
            "accuracy": accuracy,
            "total_evaluated": total,
            "total_correct": correct,
            "format_compliance_rate": format_valid / total if total > 0 else 0.0,
            "thinking_utilization_rate": has_thinking / total if total > 0 else 0.0,
            "category_metrics": category_metrics,
            "match_types": match_counts,
        }
        
        print(f"\n{'='*60}")
        print("HLE Evaluation Results")
        print(f"{'='*60}")
        print(f"  Overall Accuracy: {accuracy:.2%} ({correct}/{total})")
        print(f"  Format Compliance: {format_valid / total:.2%}")
        if self.config.thinking_mode:
            print(f"  Thinking Utilization: {has_thinking / total:.2%}")
        if category_metrics:
            print(f"\n  Per-Category Breakdown:")
            for cat, data in sorted(category_metrics.items(), key=lambda x: -x[1]["accuracy"]):
                print(f"    {cat}: {data['accuracy']:.2%} ({data['correct']}/{data['total']})")
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
            "hle/accuracy": metrics.get("accuracy", 0),
            "hle/total_evaluated": metrics.get("total_evaluated", 0),
            "hle/format_compliance_rate": metrics.get("format_compliance_rate", 0),
            "hle/thinking_utilization_rate": metrics.get("thinking_utilization_rate", 0),
        }
        
        # Log per-category accuracies (top categories)
        for cat, data in metrics.get("category_metrics", {}).items():
            safe_cat = cat.replace("/", "_").replace(" ", "_")[:30]
            log_dict[f"hle/accuracy_{safe_cat}"] = data.get("accuracy", 0)
        
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
    HLEEvalEnv.cli()

