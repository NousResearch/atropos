"""
PubMedQA Evaluation Environment for Atropos (Generative Mode)

This environment evaluates models on the PubMedQA benchmark - a biomedical
research question answering dataset.

Dataset: pubmed_qa (pqa_labeled subset)
Paper: https://pubmedqa.github.io/

The evaluation follows a generative approach:
- Models receive a question and context from biomedical literature
- Expected answer is yes/no/maybe
- Supports thinking mode with <think></think> tags for reasoning

Answers are extracted from <answer></answer> tags and validated against
the gold standard (yes/no/maybe).
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


# Valid answers for PubMedQA
VALID_ANSWERS = {"yes", "no", "maybe"}


# Prompt template for PubMedQA with answer tag instruction
PUBMEDQA_PROMPT_TEMPLATE = """Answer the following biomedical research question based on the provided context. Think step by step before answering.

Provide your final answer within <answer></answer> tags, containing only one of: yes, no, or maybe.

Example format:
<answer>yes</answer>

Question: {question}

Context:
{context}"""


class PubMedQAEvalConfig(BaseEnvConfig):
    """Configuration for PubMedQA evaluation environment."""
    
    # Dataset configuration
    dataset_name: str = Field(
        default="pubmed_qa",
        description="HuggingFace dataset name"
    )
    subset: str = Field(
        default="pqa_labeled",
        description="Dataset subset to use"
    )
    eval_split: str = Field(
        default="train",
        description="Split to evaluate on (train is the only split with labels)"
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
    wandb_name: str = "pubmedqa_eval"
    steps_per_eval: int = 1


class PubMedQAEvalEnv(BaseEnv):
    """
    PubMedQA Evaluation Environment.
    
    Evaluates biomedical QA capability using the PubMedQA dataset.
    Uses generative evaluation with yes/no/maybe answers.
    """

    name = "pubmedqa_eval"

    def __init__(
        self,
        config: PubMedQAEvalConfig,
        server_configs: List[APIServerConfig],
        slurm_job_id: Optional[str] = None,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm_job_id, testing)
        self.config: PubMedQAEvalConfig = config
        self.eval_items: List[Dict] = []
        self._dataset_loaded = False

    @classmethod
    def config_cls(cls) -> type:
        return PubMedQAEvalConfig

    async def setup(self) -> None:
        """Initialize the environment and load the dataset."""
        await super().setup()
        
        if not self._dataset_loaded:
            await self._load_dataset()
            
        print(f"\nPubMedQA Evaluation Setup (Generative Mode):")
        print(f"  Dataset: {self.config.dataset_name}")
        print(f"  Subset: {self.config.subset}")
        print(f"  Evaluation split: {self.config.eval_split}")
        print(f"  Thinking mode: {self.config.thinking_mode}")
        if self.config.thinking_mode:
            thinking_prompt = get_default_thinking_prompt(self.config.custom_thinking_prompt)
            print(f"  Thinking prompt: {thinking_prompt[:80]}...")
        print(f"  Loaded {len(self.eval_items)} evaluation items")

    async def _load_dataset(self) -> None:
        """Load and process the PubMedQA dataset."""
        print(f"Loading PubMedQA dataset: {self.config.dataset_name}/{self.config.subset}...")
        
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
        
        # Process items
        self.eval_items = []
        for idx, item in enumerate(split_data):
            # Handle different field names in the dataset
            question = item.get("QUESTION") or item.get("question", "")
            contexts = item.get("CONTEXTS") or item.get("context", [])
            
            # Contexts can be a list or string
            if isinstance(contexts, list):
                context_text = "\n\n".join(contexts)
            else:
                context_text = str(contexts)
            
            # Get the answer
            answer = item.get("final_decision") or item.get("answer", "")
            answer = answer.lower().strip()
            
            if answer not in VALID_ANSWERS:
                if self.config.full_debug:
                    print(f"Skipping item {idx} with invalid answer: {answer}")
                continue
            
            self.eval_items.append({
                "id": str(idx),
                "question": question,
                "context": context_text,
                "answer": answer,
                "pubid": item.get("PUBID") or item.get("pubid", ""),
            })
        
        # Shuffle with seed
        random.seed(self.config.shuffle_seed)
        random.shuffle(self.eval_items)
        
        self._dataset_loaded = True
        print(f"Loaded {len(self.eval_items)} evaluation items")

    def _format_prompt(self, item: Dict) -> str:
        """Format the question and context into a prompt."""
        return PUBMEDQA_PROMPT_TEMPLATE.format(
            question=item["question"],
            context=item["context"]
        )

    def _create_system_content(self) -> str:
        """Create system message content based on thinking mode."""
        return create_system_content(
            self.config.thinking_mode,
            self.config.custom_thinking_prompt,
            self.config.custom_system_prompt
        ) or ""

    def _extract_answer(self, response: str, debug: bool = False) -> Tuple[Optional[str], str]:
        """
        Extract the answer (yes/no/maybe) from the response.
        
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
            answer_content = answer_tag_match.group(1).strip().lower()
            
            # Check for exact match
            if answer_content in VALID_ANSWERS:
                if debug:
                    print(f"    Extracted '{answer_content}' using method 'answer_tag'")
                return answer_content, "answer_tag"
            
            # Check if answer contains a valid option
            for valid in VALID_ANSWERS:
                if valid in answer_content:
                    if debug:
                        print(f"    Extracted '{valid}' using method 'answer_tag_contains'")
                    return valid, "answer_tag_contains"
        
        # Fallback: Look for yes/no/maybe in the response
        response_lower = response.lower()
        
        # Try common patterns
        patterns = [
            r"(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*(yes|no|maybe)",
            r"(?:my\s+)?answer\s*:?\s*(yes|no|maybe)",
            r"\b(yes|no|maybe)\b(?=\s*[\.!\?\s]*$)",  # At end of response
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_lower)
            if match:
                answer = match.group(1)
                if debug:
                    print(f"    Extracted '{answer}' using fallback pattern")
                return answer, "fallback_pattern"
        
        # Last resort: count occurrences and pick most common
        counts = {ans: response_lower.count(ans) for ans in VALID_ANSWERS}
        if any(counts.values()):
            # Find the answer that appears most (preferring last occurrence for ties)
            best_answer = max(counts.keys(), key=lambda x: (counts[x], response_lower.rfind(x)))
            if counts[best_answer] > 0:
                if debug:
                    print(f"    Extracted '{best_answer}' using fallback count")
                return best_answer, "fallback_count"
        
        return None, "no_match"

    async def rollout_and_score_eval(
        self,
        item: Dict,
        server: APIServerConfig,
    ) -> Optional[Dict]:
        """
        Run evaluation on a single item and return the result.
        
        Args:
            item: The evaluation item containing question, context, and answer
            server: API server configuration
            
        Returns:
            Dictionary with evaluation results or None if failed
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
        extracted_answer, method = self._extract_answer(
            content_for_extraction,
            debug=self.config.full_debug
        )
        
        # Score
        gold_answer = item["answer"]
        is_correct = extracted_answer == gold_answer if extracted_answer else False
        
        if self.config.full_debug:
            print(f"\n--- Item: {item['id']} ---")
            print(f"Question: {item['question'][:100]}...")
            print(f"Gold answer: {gold_answer}")
            print(f"Extracted: {extracted_answer} (method: {method})")
            print(f"Correct: {is_correct}")
            if thinking_content:
                print(f"Thinking: {thinking_content[:100]}...")
        
        return {
            "item_id": item["id"],
            "pubid": item.get("pubid", ""),
            "question": item["question"],
            "gold_answer": gold_answer,
            "extracted_answer": extracted_answer,
            "extraction_method": method,
            "is_correct": is_correct,
            "format_valid": is_valid_format,
            "response": response_text,
            "thinking_content": thinking_content,
            "has_thinking": thinking_content is not None,
        }

    async def evaluate(self, *args, **kwargs) -> Dict:
        """
        Run the full PubMedQA evaluation.
        
        Returns:
            Dictionary containing evaluation metrics and results
        """
        print(f"\n{'='*60}")
        print("Starting PubMedQA Evaluation (Generative Mode)")
        print(f"{'='*60}")
        print(f"  Total questions: {len(self.eval_items)}")
        print(f"  Thinking mode: {self.config.thinking_mode}")
        print(f"{'='*60}\n")
        
        # Create evaluation tasks
        async def eval_task(item):
            return await self.rollout_and_score_eval(item, self.server_configs[0])
        
        tasks = [eval_task(item) for item in self.eval_items]
        
        # Run with progress bar
        results = await tqdm_asyncio.gather(
            *tasks,
            desc="Evaluating PubMedQA"
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
        
        # Calculate per-answer metrics
        answer_metrics = {}
        for answer in VALID_ANSWERS:
            answer_items = [r for r in valid_results if r["gold_answer"] == answer]
            if answer_items:
                answer_correct = sum(1 for r in answer_items if r["is_correct"])
                answer_metrics[answer] = {
                    "total": len(answer_items),
                    "correct": answer_correct,
                    "accuracy": answer_correct / len(answer_items)
                }
        
        # Format compliance and thinking utilization
        format_valid = sum(1 for r in valid_results if r.get("format_valid", True))
        has_thinking = sum(1 for r in valid_results if r.get("has_thinking", False))
        
        # Extraction method breakdown
        method_counts = {}
        for r in valid_results:
            method = r.get("extraction_method", "unknown")
            method_counts[method] = method_counts.get(method, 0) + 1
        
        metrics = {
            "accuracy": accuracy,
            "total_evaluated": total,
            "total_correct": correct,
            "format_compliance_rate": format_valid / total if total > 0 else 0.0,
            "thinking_utilization_rate": has_thinking / total if total > 0 else 0.0,
            "answer_metrics": answer_metrics,
            "extraction_methods": method_counts,
        }
        
        print(f"\n{'='*60}")
        print("PubMedQA Evaluation Results")
        print(f"{'='*60}")
        print(f"  Overall Accuracy: {accuracy:.2%} ({correct}/{total})")
        print(f"  Format Compliance: {format_valid / total:.2%}")
        if self.config.thinking_mode:
            print(f"  Thinking Utilization: {has_thinking / total:.2%}")
        print(f"\n  Per-Answer Breakdown:")
        for answer, data in answer_metrics.items():
            print(f"    {answer}: {data['accuracy']:.2%} ({data['correct']}/{data['total']})")
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
            "pubmedqa/accuracy": metrics.get("accuracy", 0),
            "pubmedqa/total_evaluated": metrics.get("total_evaluated", 0),
            "pubmedqa/format_compliance_rate": metrics.get("format_compliance_rate", 0),
            "pubmedqa/thinking_utilization_rate": metrics.get("thinking_utilization_rate", 0),
        }
        
        # Log per-answer accuracies
        for answer, data in metrics.get("answer_metrics", {}).items():
            log_dict[f"pubmedqa/accuracy_{answer}"] = data.get("accuracy", 0)
        
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
    PubMedQAEvalEnv.cli()

