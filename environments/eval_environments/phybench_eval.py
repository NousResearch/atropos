"""
PHYBench Evaluation Environment for Atropos.

This environment evaluates models on PHYBench - a benchmark for evaluating
physical perception and reasoning capabilities in Large Language Models.

Dataset: Eureka-Lab/PHYBench
Paper: https://arxiv.org/abs/2504.16074
Website: https://www.phybench.cn/

PHYBench is a human-curated benchmark with 500 original physics problems spanning:
- Mechanics (MECHANICS)
- Electromagnetism (ELECTRICITY)
- Thermodynamics (THERMODYNAMICS)
- Optics (OPTICS)
- Modern Physics (MODERN)
- Advanced Physics (ADVANCED)

Key features:
- Original problems to prevent data contamination
- Symbolic expression answers in LaTeX format
- Two evaluation metrics:
  1. Binary Accuracy: Exact match using SymPy equivalence
  2. EED Score: Expression Edit Distance for partial credit (0-100)

The EED Score provides:
- 204% improved sample efficiency over binary scoring
- Continuous scoring that captures partial correctness
- Differentiation between minor coefficient errors and structural errors

Supports thinking mode with <think></think> tags for extended reasoning.
"""

import asyncio
import os
import random
import re
from typing import Dict, List, Optional, Tuple

import wandb
from datasets import load_dataset
from eed_score import EED_AVAILABLE, compute_eed_score, extract_all_boxed
from eval_helpers import (
    THINK_CONTENT_AFTER_PATTERN,
    create_system_content,
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

# Physics domain tags in PHYBench
PHYBENCH_TAGS = [
    "MECHANICS",
    "ELECTRICITY",
    "THERMODYNAMICS",
    "OPTICS",
    "MODERN",
    "ADVANCED",
]

# Prompt template for PHYBench evaluation
PHYBENCH_PROMPT_TEMPLATE = """You are a physics expert.
Please read the following question and provide a step-by-step solution.

Put your final answer, which must be a readable LaTeX formula, in a \\boxed{{}} environment.

Question: {problem}

Answer:"""

# Alternative prompt with more detailed instructions
PHYBENCH_DETAILED_PROMPT_TEMPLATE = """Solve the following physics problem. Show your reasoning step by step.

Your final answer should be a single symbolic expression (e.g., $\\sqrt{{\\frac{{2g}}{{3R}}}}$).
- Equivalent forms are accepted
- No numerical approximations
- No equation chains

Put your final answer in \\boxed{{}} format.
For example: \\boxed{{2mg + \\frac{{4mv_0^2}}{{l}}}}

Problem:
{problem}

Solution:"""


class PHYBenchEvalConfig(BaseEnvConfig):
    """Configuration for PHYBench evaluation environment."""

    # Dataset configuration
    dataset_name: str = Field(
        default="Eureka-Lab/PHYBench",
        description="HuggingFace dataset name",
    )
    eval_split: str = Field(
        default="train",
        description="Split to evaluate on (PHYBench only has train split)",
    )
    shuffle_seed: int = Field(
        default=42,
        description="Random seed for shuffling",
    )
    max_samples: Optional[int] = Field(
        default=None,
        description="Maximum number of samples to evaluate (None = all)",
    )
    tags_filter: Optional[List[str]] = Field(
        default=None,
        description="Filter to specific physics domains (e.g., ['MECHANICS', 'OPTICS'])",
    )

    # Generation parameters
    eval_temperature: float = Field(
        default=0.6,
        description="Temperature for evaluation generation",
    )
    eval_max_tokens: int = Field(
        default=0,
        description="Max tokens for evaluation (0 = use model default)",
    )

    # System prompt configuration
    custom_system_prompt: Optional[str] = Field(
        default=None,
        description="Optional custom system prompt",
    )

    # Thinking mode configuration
    thinking_mode: bool = Field(
        default=False,
        description="Whether to use thinking mode with <think></think> tags",
    )
    custom_thinking_prompt: Optional[str] = Field(
        default=None,
        description="Optional custom thinking prompt",
    )

    # Prompt configuration
    use_detailed_prompt: bool = Field(
        default=False,
        description="Use detailed prompt with more instructions",
    )

    # Scoring configuration
    compute_eed_score: bool = Field(
        default=True,
        description="Whether to compute EED Score (requires latex2sympy2_extended)",
    )

    # Retry and debug configuration
    max_retries: int = Field(
        default=3,
        description="Maximum retries for failed API calls",
    )
    retry_delay: float = Field(
        default=1.0,
        description="Delay between retries in seconds",
    )
    min_response_length: int = Field(
        default=1,
        description="Minimum response length to consider valid",
    )
    full_debug: bool = Field(
        default=False,
        description="Enable full debug output",
    )


class PHYBenchEvalEnv(BaseEnv):
    """
    PHYBench Evaluation Environment.

    Evaluates models on physics problems requiring symbolic expression answers.
    Uses both binary accuracy and EED Score for comprehensive evaluation.
    """

    name = "phybench_eval"
    env_config_cls = PHYBenchEvalConfig

    def __init__(
        self,
        config: PHYBenchEvalConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = False,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: PHYBenchEvalConfig = config
        self.eval_items: List[Dict] = []
        self._dataset_loaded = False

        # Pre-compile regex patterns for answer extraction
        self._boxed_pattern = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")

        # Check EED availability
        if self.config.compute_eed_score and not EED_AVAILABLE:
            print(
                "Warning: EED Score requested but latex2sympy2_extended not available. "
                "Install with: pip install latex2sympy2_extended sympy"
            )

    @classmethod
    def config_init(cls) -> Tuple[PHYBenchEvalConfig, List[APIServerConfig]]:
        """Initialize default configuration for the environment."""
        env_config = PHYBenchEvalConfig(
            tokenizer_name="NousResearch/Hermes-3-Llama-3.1-8B",
            group_size=1,
            use_wandb=True,
            max_num_workers_per_node=128,
            rollout_server_url="http://localhost:8000",
            total_steps=1,
            batch_size=1,
            steps_per_eval=1,
            inference_weight=1.0,
            wandb_name="phybench_eval",
            eval_handling=EvalHandlingEnum.STOP_TRAIN,
            max_eval_workers=256,
            max_num_workers=1024,
            # PHYBench specific defaults
            dataset_name="Eureka-Lab/PHYBench",
            eval_split="train",
            eval_temperature=0.6,
            eval_max_tokens=0,  # Use model default
            thinking_mode=False,
            compute_eed_score=True,
        )

        server_configs = [
            APIServerConfig(
                model_name="gpt-4.1",
                base_url="https://api.openai.com/v1",
                api_key=os.getenv("OPENAI_API_KEY", "none"),
                num_max_requests_at_once=32,
                num_requests_for_eval=1024,
            ),
        ]

        return env_config, server_configs

    async def setup(self) -> None:
        """Initialize the environment and load the dataset."""
        if not self._dataset_loaded:
            await self._load_dataset()

        print("\nPHYBench Evaluation Setup:")
        print(f"  Dataset: {self.config.dataset_name}")
        print(f"  Evaluation split: {self.config.eval_split}")
        print(f"  Thinking mode: {self.config.thinking_mode}")
        print(f"  EED Score enabled: {self.config.compute_eed_score and EED_AVAILABLE}")
        if self.config.thinking_mode:
            thinking_prompt = get_default_thinking_prompt(
                self.config.custom_thinking_prompt
            )
            print(f"  Thinking prompt: {thinking_prompt[:80]}...")
        if self.config.tags_filter:
            print(f"  Tags filter: {self.config.tags_filter}")
        print(f"  Loaded {len(self.eval_items)} evaluation items")

    async def _load_dataset(self) -> None:
        """Load and process the PHYBench dataset."""
        print(f"Loading PHYBench dataset: {self.config.dataset_name}...")

        try:
            dataset = load_dataset(
                self.config.dataset_name,
                trust_remote_code=True,
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

        # Process items (deduplicate by content - dataset has duplicates)
        self.eval_items = []
        tag_counts: Dict[str, int] = {}
        seen_content: set = set()

        for item in split_data:
            problem_id = item.get("id", "")
            tag = item.get("tag", "UNKNOWN")
            content = item.get("content", "")
            solution = item.get("solution", "")
            answer = item.get("answer", "")

            # Skip if no content or answer
            if not content or not answer:
                continue

            # Skip duplicates (dataset contains each question twice)
            if content in seen_content:
                continue
            seen_content.add(content)

            # Apply tag filter if specified
            if self.config.tags_filter and tag not in self.config.tags_filter:
                continue

            # Track tag distribution
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

            self.eval_items.append(
                {
                    "id": problem_id,
                    "tag": tag,
                    "content": content,
                    "solution": solution,
                    "answer": answer,
                }
            )

        # Shuffle with seed for reproducibility
        random.seed(self.config.shuffle_seed)
        random.shuffle(self.eval_items)

        # Apply max_samples limit if specified
        if self.config.max_samples and len(self.eval_items) > self.config.max_samples:
            self.eval_items = self.eval_items[: self.config.max_samples]

        self._dataset_loaded = True

        # Print tag distribution
        print(f"Loaded {len(self.eval_items)} items")
        print("Tag distribution:")
        for tag, count in sorted(tag_counts.items()):
            print(f"  {tag}: {count}")

    def _format_prompt(self, item: Dict) -> str:
        """Format the problem into a prompt."""
        if self.config.use_detailed_prompt:
            return PHYBENCH_DETAILED_PROMPT_TEMPLATE.format(problem=item["content"])
        return PHYBENCH_PROMPT_TEMPLATE.format(problem=item["content"])

    def _create_system_content(self) -> str:
        """Create system message content based on thinking mode."""
        return (
            create_system_content(
                self.config.thinking_mode,
                self.config.custom_thinking_prompt,
                self.config.custom_system_prompt,
            )
            or ""
        )

    def _extract_answer(
        self, response: str, debug: bool = False
    ) -> Tuple[Optional[str], str]:
        """
        Extract the answer from the model's response.

        Looks for \\boxed{} content. If multiple found, uses the last one.

        Args:
            response: Model's response text
            debug: Whether to print debug info

        Returns:
            Tuple of (extracted_answer, extraction_method)
        """
        if not response:
            return None, "empty_response"

        # Find all boxed answers
        boxed_answers = extract_all_boxed(response)

        if not boxed_answers:
            if debug:
                print("    No \\boxed{} found in response")
            return None, "no_boxed"

        if len(boxed_answers) > 1:
            if debug:
                print(
                    f"    Multiple \\boxed{{}} found ({len(boxed_answers)}), using last one"
                )
            return boxed_answers[-1], "boxed_last"

        return boxed_answers[0], "boxed"

    def _check_equivalence(
        self,
        predicted: str,
        gold: str,
        debug: bool = False,
    ) -> Tuple[bool, str]:
        """
        Check if predicted answer is equivalent to gold answer.

        Uses SymPy for symbolic equivalence checking.

        Args:
            predicted: Predicted answer in LaTeX
            gold: Gold answer in LaTeX
            debug: Whether to print debug info

        Returns:
            Tuple of (is_correct, method)
        """
        if not predicted:
            return False, "empty_prediction"

        # Clean up the answers
        pred_clean = predicted.strip()
        gold_clean = gold.strip()

        # Exact string match
        if pred_clean == gold_clean:
            return True, "exact_match"

        # Try EED Score - if score is 100, they're equivalent
        if self.config.compute_eed_score and EED_AVAILABLE:
            try:
                score, _, _, _ = compute_eed_score(
                    gold_clean, pred_clean, debug_mode=False
                )
                if score == 100:
                    return True, "sympy_equivalent"
            except Exception:
                pass

        return False, "not_equivalent"

    def _compute_scores(
        self,
        predicted: str,
        gold: str,
        debug: bool = False,
    ) -> Dict:
        """
        Compute both accuracy and EED Score.

        Args:
            predicted: Predicted answer
            gold: Gold answer
            debug: Whether to print debug info

        Returns:
            Dictionary with scoring results
        """
        result = {
            "is_correct": False,
            "match_method": "none",
            "eed_score": 0.0,
            "eed_rel_distance": -1,
            "eed_tree_size": -1,
            "eed_distance": -1,
        }

        if not predicted:
            return result

        # Check equivalence (for binary accuracy)
        is_correct, match_method = self._check_equivalence(predicted, gold, debug)
        result["is_correct"] = is_correct
        result["match_method"] = match_method

        # Compute EED Score if enabled
        if self.config.compute_eed_score and EED_AVAILABLE:
            try:
                eed_score, rel_dist, tree_size, distance = compute_eed_score(
                    gold, predicted, debug_mode=debug
                )
                result["eed_score"] = eed_score
                result["eed_rel_distance"] = rel_dist
                result["eed_tree_size"] = tree_size
                result["eed_distance"] = distance

                # If EED score is 100, mark as correct
                if eed_score == 100 and not is_correct:
                    result["is_correct"] = True
                    result["match_method"] = "eed_equivalent"

            except Exception as e:
                if debug:
                    print(f"    EED Score error: {e}")

        return result

    async def rollout_and_score_eval(self, item: Dict) -> Optional[Dict]:
        """Run evaluation on a single item and return the result."""
        if self.config.full_debug:
            print(
                f"[DEBUG] Starting eval for item: {item.get('id', 'unknown')}",
                flush=True,
            )
        prompt = self._format_prompt(item)
        system_content = self._create_system_content()

        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": prompt})

        # Build API call parameters
        kwargs = {
            "messages": messages,
            "n": 1,
            "temperature": self.config.eval_temperature,
            "split": "eval",
        }
        if self.config.eval_max_tokens > 0:
            kwargs["max_tokens"] = self.config.eval_max_tokens

        response_text = ""
        for attempt in range(self.config.max_retries):
            try:
                if self.config.full_debug:
                    print(
                        f"  Making API request (attempt {attempt + 1}/{self.config.max_retries})...",
                        flush=True,
                    )
                    print(
                        f"    Temperature: {self.config.eval_temperature}", flush=True
                    )
                    print(
                        f"    Max tokens: {self.config.eval_max_tokens if self.config.eval_max_tokens > 0 else 'model default'}",
                        flush=True,
                    )

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
            response_text, self.config.thinking_mode
        )

        # Extract thinking content if present
        thinking_content = (
            extract_thinking_content(response_text)
            if self.config.thinking_mode
            else None
        )

        # Get content for answer extraction
        if self.config.thinking_mode:
            match = THINK_CONTENT_AFTER_PATTERN.search(response_text)
            if match:
                answer_content = match.group(1)
            else:
                answer_content = response_text
        else:
            answer_content = response_text

        # Extract answer
        extracted_answer, extraction_method = self._extract_answer(
            answer_content, debug=self.config.full_debug
        )

        # Compute scores
        gold_answer = item["answer"]
        scores = self._compute_scores(
            extracted_answer, gold_answer, debug=self.config.full_debug
        )

        if self.config.full_debug:
            status = "✓" if scores["is_correct"] else "✗"
            eed = scores["eed_score"]
            print(
                f"  [{status}] {item['tag']}: EED={eed:.1f}, gold={gold_answer[:50]}..."
            )

        return {
            "item_id": item["id"],
            "tag": item["tag"],
            "content": item["content"][:200],
            "gold_answer": gold_answer,
            "extracted_answer": extracted_answer,
            "extraction_method": extraction_method,
            "is_correct": scores["is_correct"],
            "match_method": scores["match_method"],
            "eed_score": scores["eed_score"],
            "eed_rel_distance": scores["eed_rel_distance"],
            "eed_tree_size": scores["eed_tree_size"],
            "eed_distance": scores["eed_distance"],
            "format_valid": is_valid_format,
            "response": response_text,
            "response_length": len(response_text),
            "thinking_content": thinking_content,
            "has_thinking": thinking_content is not None,
        }

    async def evaluate(self, *args, **kwargs) -> Dict:
        """Run the full PHYBench evaluation."""
        print(f"\n{'='*60}")
        print("Starting PHYBench Evaluation")
        print(f"{'='*60}")
        print(f"  Total questions: {len(self.eval_items)}")
        print(f"  Thinking mode: {self.config.thinking_mode}")
        print(f"  EED Score: {self.config.compute_eed_score and EED_AVAILABLE}")
        print(f"{'='*60}\n")

        # Create evaluation tasks
        eval_tasks = [self.rollout_and_score_eval(item) for item in self.eval_items]

        # Run with progress bar
        results = await tqdm_asyncio.gather(*eval_tasks, desc="Evaluating PHYBench")

        # Filter out failed results
        valid_results = [r for r in results if r is not None]

        if not valid_results:
            print("Warning: No valid evaluation results obtained")
            return {"error": "No valid results", "accuracy": 0.0}

        # Calculate metrics
        total = len(valid_results)
        correct = sum(1 for r in valid_results if r["is_correct"])
        accuracy = correct / total if total > 0 else 0.0

        # Calculate average EED Score
        eed_scores = [r["eed_score"] for r in valid_results if r["eed_score"] >= 0]
        avg_eed_score = sum(eed_scores) / len(eed_scores) if eed_scores else 0.0

        # Calculate per-tag metrics
        tag_metrics: Dict[str, Dict] = {}
        for r in valid_results:
            tag = r.get("tag", "UNKNOWN")
            if tag not in tag_metrics:
                tag_metrics[tag] = {"total": 0, "correct": 0, "eed_scores": []}
            tag_metrics[tag]["total"] += 1
            if r["is_correct"]:
                tag_metrics[tag]["correct"] += 1
            if r["eed_score"] >= 0:
                tag_metrics[tag]["eed_scores"].append(r["eed_score"])

        for tag in tag_metrics:
            t_total = tag_metrics[tag]["total"]
            t_correct = tag_metrics[tag]["correct"]
            t_eed_scores = tag_metrics[tag]["eed_scores"]
            tag_metrics[tag]["accuracy"] = t_correct / t_total if t_total > 0 else 0.0
            tag_metrics[tag]["avg_eed_score"] = (
                sum(t_eed_scores) / len(t_eed_scores) if t_eed_scores else 0.0
            )

        # Calculate extraction method statistics
        extraction_methods: Dict[str, int] = {}
        for r in valid_results:
            method = r.get("extraction_method", "unknown")
            extraction_methods[method] = extraction_methods.get(method, 0) + 1

        # Format compliance and thinking utilization
        format_valid = sum(1 for r in valid_results if r.get("format_valid", True))
        has_thinking = sum(1 for r in valid_results if r.get("has_thinking", False))
        has_boxed = sum(
            1 for r in valid_results if r.get("extracted_answer") is not None
        )

        # Average response length
        response_lengths = [r.get("response_length", 0) for r in valid_results]
        avg_response_length = (
            sum(response_lengths) / len(response_lengths) if response_lengths else 0
        )

        metrics = {
            "accuracy": accuracy,
            "avg_eed_score": avg_eed_score,
            "total_evaluated": total,
            "total_correct": correct,
            "has_boxed_rate": has_boxed / total if total > 0 else 0.0,
            "format_compliance_rate": format_valid / total if total > 0 else 0.0,
            "thinking_utilization_rate": has_thinking / total if total > 0 else 0.0,
            "avg_response_length": avg_response_length,
            "tag_metrics": tag_metrics,
            "extraction_methods": extraction_methods,
        }

        # Print summary
        print(f"\n{'='*60}")
        print("PHYBench Evaluation Results")
        print(f"{'='*60}")
        print(f"  Overall Accuracy: {accuracy:.2%} ({correct}/{total})")
        print(f"  Average EED Score: {avg_eed_score:.1f}/100")
        print(f"  Has \\boxed{{}} Rate: {has_boxed / total:.2%}")
        print(f"  Avg Response Length: {avg_response_length:.0f} chars")
        if self.config.thinking_mode:
            print(f"  Format Compliance: {format_valid / total:.2%}")
            print(f"  Thinking Utilization: {has_thinking / total:.2%}")

        print("\n  Per-Tag Breakdown:")
        for tag in sorted(tag_metrics.keys()):
            data = tag_metrics[tag]
            acc = data["accuracy"]
            eed = data["avg_eed_score"]
            cnt = data["total"]
            print(f"    {tag}: Acc={acc:.2%}, EED={eed:.1f} ({cnt} items)")

        print("\n  Extraction Methods:")
        for method, count in sorted(extraction_methods.items(), key=lambda x: -x[1]):
            print(f"    {method}: {count}")

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
            "phybench/accuracy": metrics.get("accuracy", 0),
            "phybench/avg_eed_score": metrics.get("avg_eed_score", 0),
            "phybench/total_evaluated": metrics.get("total_evaluated", 0),
            "phybench/has_boxed_rate": metrics.get("has_boxed_rate", 0),
            "phybench/format_compliance_rate": metrics.get("format_compliance_rate", 0),
            "phybench/thinking_utilization_rate": metrics.get(
                "thinking_utilization_rate", 0
            ),
            "phybench/avg_response_length": metrics.get("avg_response_length", 0),
        }

        # Log per-tag metrics
        for tag, data in metrics.get("tag_metrics", {}).items():
            safe_tag = tag.lower()
            log_dict[f"phybench/accuracy_{safe_tag}"] = data.get("accuracy", 0)
            log_dict[f"phybench/eed_score_{safe_tag}"] = data.get("avg_eed_score", 0)

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
    PHYBenchEvalEnv.cli()
