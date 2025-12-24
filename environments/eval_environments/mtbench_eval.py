"""
MT-Bench Evaluation Environment for Atropos (Generative Mode with LLM Judge)

This environment evaluates models on MT-Bench - a multi-turn conversational
benchmark for evaluating language models.

Dataset: lighteval/mt-bench

MT-Bench consists of 80 high-quality multi-turn questions across 8 categories:
- Writing, Roleplay, Reasoning, Math, Coding, Extraction, STEM, Humanities

Model responses are evaluated by an LLM judge on a 1-5 scale.

The evaluation follows the refusalbench pattern for LLM judge configuration:
- Separate judge model with configurable endpoint/API key
- Rate limiting and retry logic
- Concurrent call limits with semaphore
- Fallback scoring when judge fails

Supports thinking mode with <think></think> tags for extended reasoning.
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
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
)
from eval_helpers import (
    validate_thinking_format,
    extract_thinking_content,
    get_default_thinking_prompt,
    create_system_content,
    save_eval_results,
    THINK_CONTENT_AFTER_PATTERN,
)


# MT-Bench categories
MT_BENCH_CATEGORIES = [
    "writing",
    "roleplay", 
    "reasoning",
    "math",
    "coding",
    "extraction",
    "stem",
    "humanities",
]


# Judge prompt templates (from lighteval)
def judge_prompt_with_reference(question: str, answer: str, reference: Optional[str] = None) -> List[Dict]:
    """Create judge prompt with optional reference answer."""
    reference_text = f"""the reference answer is:
{reference}""" if reference else ""
    
    return [
        {
            "role": "user",
            "content": f"""# GOAL
Your job is to evaluate a task carried out by an AI system powered by a large language model.

You will be provided with the inputs and output of the task, as well as the evaluation criteria and scoring rubric. Your task is to evaluate the output of the AI system based on the evaluation criteria and scoring rubric provided.

# INPUT
Below are the inputs required for performing the task:
<inputs>
{question}
</inputs>

# OUTPUT
Below is the output of the task:
<output>
{answer}
</output>

# EVALUATION CRITERIA AND SCORING RUBRIC
Here are the evaluation criteria and the rubric that you need to use for evaluating the task:
<evaluation_criteria>
How well the response answers the question?{' ' + reference_text if reference_text else ''}
</evaluation_criteria>

<scoring_rubric>
- Score 1: The response completely fails to answer the question.
- Score 2: The response barely answers the question.
- Score 3: The response partially answers the question.
- Score 4: The response mostly answers the question.
- Score 5: The response completely answers the question.
</scoring_rubric>

# INSTRUCTIONS FOR THE EVALUATION
1. Understand the task and criteria: Familiarize yourself with the task to be evaluated. Review the evaluation criteria and scoring rubric to understand the different levels of performance and the descriptions for each score.
2. Review the inputs and output: Look at the inputs provided for the task. Examine the output generated from completing the task.
3. Compare output to score descriptions: Compare the output against the criteria and score descriptions in the scoring rubric. For each criterion, decide which description best matches the output.
4. After comparing the output to the score descriptions, pay attention to the small details that might impact the final score that you assign. Sometimes a small difference can dictate the final score.
5. Write verbal feedback justifying your evaluation that includes a detailed rationale, referring to specific aspects of the output and comparing them to the rubric.
6. Assign a final score based on the scoring rubric.

## FORMAT FOR THE EVALUATION
- Write the verbal feedback inside <feedback> tags without any additional surrounding text.
- Write the numeric score inside <score> tags, without any additional surrounding text and always after the feedback.

Please accurately evaluate the task. Strictly adhere to the evaluation criteria and rubric.""",
        }
    ]


class MTBenchEvalConfig(BaseEnvConfig):
    """Configuration for MT-Bench evaluation environment with LLM judge."""
    
    # Dataset configuration
    dataset_name: str = Field(
        default="lighteval/mt-bench",
        description="HuggingFace dataset name"
    )
    subset: str = Field(
        default="default",
        description="Dataset subset"
    )
    eval_split: str = Field(
        default="train",
        description="Split to evaluate on"
    )
    categories: Optional[List[str]] = Field(
        default=None,
        description="List of categories to evaluate (None = all categories)"
    )
    shuffle_seed: int = Field(
        default=42,
        description="Random seed for shuffling"
    )
    
    # Model generation parameters
    eval_temperature: float = Field(
        default=0.6,
        description="Temperature for model evaluation completions"
    )
    eval_max_tokens: int = Field(
        default=0,
        description="Max tokens for model evaluation (0 = use model default)"
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
    
    # Judge model configuration (following refusalbench pattern)
    judge_model_name: str = Field(
        default="gpt-4o",
        description="Model name for the judge"
    )
    judge_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for the judge model API"
    )
    judge_api_key_env: str = Field(
        default="OPENAI_API_KEY",
        description="Environment variable name containing the API key for the judge model"
    )
    judge_temperature: float = Field(
        default=0.2,
        description="Temperature for judge completions"
    )
    judge_max_tokens: int = Field(
        default=0,
        description="Maximum tokens for judge completions (0 = use model default)"
    )
    
    # Judge retry configuration
    judge_max_retries: int = Field(
        default=3,
        ge=1,
        description="Maximum number of retries for failed judge API calls"
    )
    judge_retry_multiplier: float = Field(
        default=1.0,
        ge=0.0,
        description="Exponential backoff multiplier for judge retries"
    )
    judge_retry_max_wait: int = Field(
        default=10,
        ge=1,
        description="Maximum wait time in seconds for judge retries"
    )
    
    # Rate limiting configuration
    judge_max_concurrent_calls: int = Field(
        default=5,
        ge=1,
        description="Maximum number of concurrent judge API calls"
    )
    judge_rate_limit_delay: float = Field(
        default=0.5,
        ge=0.0,
        description="Minimum delay in seconds between judge API calls"
    )
    
    # Fallback configuration
    use_fallback_scoring: bool = Field(
        default=True,
        description="Use fallback scoring (score=0) when judge API fails"
    )
    
    # Retry and debug configuration
    max_retries: int = Field(
        default=3,
        description="Maximum retries for failed model API calls"
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
    max_num_workers: int = 256
    max_eval_workers: int = 64
    max_num_workers_per_node: int = 32
    use_wandb: bool = True
    rollout_server_url: str = "http://localhost:8000"
    total_steps: int = 1
    wandb_name: str = "mtbench_eval"
    steps_per_eval: int = 1


class MTBenchEvalEnv(BaseEnv):
    """
    MT-Bench Evaluation Environment with LLM Judge.
    
    Evaluates multi-turn conversational ability using MT-Bench dataset.
    Uses an LLM judge to score responses on a 1-5 scale.
    """

    name = "mtbench_eval"

    def __init__(
        self,
        config: MTBenchEvalConfig,
        server_configs: List[APIServerConfig],
        slurm_job_id: Optional[str] = None,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm_job_id, testing)
        self.config: MTBenchEvalConfig = config
        self.eval_items: List[Dict] = []
        self._dataset_loaded = False
        
        # Setup judge client
        self.judge_client = None
        self._setup_judge_client()
        
        # Rate limiting for judge calls
        self.judge_semaphore = asyncio.Semaphore(self.config.judge_max_concurrent_calls)
        
        # Pre-compile regex for score extraction
        self._score_pattern = re.compile(r"<score>\s*(\d)\s*</score>", re.IGNORECASE)
        
        # Thread-safe metrics tracking
        self._metrics_lock = asyncio.Lock()
        self.judge_error_count = 0
        self.fallback_count = 0

    @classmethod
    def config_cls(cls) -> type:
        return MTBenchEvalConfig

    def _setup_judge_client(self):
        """Setup the judge API client (following refusalbench pattern)."""
        try:
            import openai
            
            api_key = os.getenv(self.config.judge_api_key_env)
            if not api_key:
                raise ValueError(
                    f"API key not found in environment variable: {self.config.judge_api_key_env}"
                )
            
            self.judge_client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url=self.config.judge_base_url,
            )
            
        except ImportError:
            raise ImportError(
                "OpenAI package is required for judge functionality. Install with: pip install openai"
            )

    async def setup(self) -> None:
        """Initialize the environment and load the dataset."""
        await super().setup()
        
        if not self._dataset_loaded:
            await self._load_dataset()
            
        print(f"\nMT-Bench Evaluation Setup (Multi-Turn with LLM Judge):")
        print(f"  Dataset: {self.config.dataset_name}")
        print(f"  Categories: {self.config.categories or 'all'}")
        print(f"  Evaluation split: {self.config.eval_split}")
        print(f"  Thinking mode: {self.config.thinking_mode}")
        print(f"  Judge model: {self.config.judge_model_name}")
        print(f"  Judge endpoint: {self.config.judge_base_url}")
        if self.config.thinking_mode:
            thinking_prompt = get_default_thinking_prompt(self.config.custom_thinking_prompt)
            print(f"  Thinking prompt: {thinking_prompt[:80]}...")
        print(f"  Loaded {len(self.eval_items)} evaluation items")

    async def _load_dataset(self) -> None:
        """Load and process the MT-Bench dataset."""
        print(f"Loading MT-Bench dataset: {self.config.dataset_name}...")
        
        try:
            dataset = load_dataset(
                self.config.dataset_name,
                self.config.subset if self.config.subset != "default" else None,
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
            category = item.get("category", "unknown")
            
            # Filter by categories if specified
            if self.config.categories and category.lower() not in [c.lower() for c in self.config.categories]:
                continue
            
            turns = item.get("turns", [])
            if len(turns) < 2:
                print(f"  Warning: Item {idx} has fewer than 2 turns, skipping")
                continue
            
            reference = item.get("reference", [])
            question_id = item.get("question_id", idx)
            
            self.eval_items.append({
                "id": question_id,
                "category": category,
                "turns": turns,  # List of 2 turn prompts
                "reference": reference,  # Optional reference answers
            })
        
        # Shuffle with seed
        random.seed(self.config.shuffle_seed)
        random.shuffle(self.eval_items)
        
        self._dataset_loaded = True
        print(f"Loaded {len(self.eval_items)} evaluation items")
        
        # Show category distribution
        category_counts = {}
        for item in self.eval_items:
            cat = item["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1
        print("  Category distribution:")
        for cat, count in sorted(category_counts.items()):
            print(f"    {cat}: {count}")

    def _create_system_content(self) -> str:
        """Create system message content based on thinking mode."""
        return create_system_content(
            self.config.thinking_mode,
            self.config.custom_thinking_prompt,
            self.config.custom_system_prompt
        ) or "You are a helpful assistant."

    async def _rate_limited_judge_call(self, messages: List[Dict]) -> Optional[str]:
        """Make a rate-limited API call to the judge model with retry logic."""
        
        retry_decorator = retry(
            stop=stop_after_attempt(self.config.judge_max_retries),
            wait=wait_random_exponential(
                multiplier=self.config.judge_retry_multiplier,
                max=self.config.judge_retry_max_wait,
            ),
            retry=retry_if_exception_type((Exception,)),
        )
        
        async def _inner_judge_call():
            async with self.judge_semaphore:
                await asyncio.sleep(self.config.judge_rate_limit_delay)
                return await self._judge_api_call_raw(messages)
        
        retrying_call = retry_decorator(_inner_judge_call)
        return await retrying_call()

    async def _judge_api_call_raw(self, messages: List[Dict]) -> Optional[str]:
        """Make a raw API call to the judge model."""
        try:
            kwargs = {
                "model": self.config.judge_model_name,
                "messages": messages,
                "temperature": self.config.judge_temperature,
            }
            if self.config.judge_max_tokens > 0:
                kwargs["max_tokens"] = self.config.judge_max_tokens
            
            result = await self.judge_client.chat.completions.create(**kwargs)
            if result.choices and result.choices[0].message.content:
                return result.choices[0].message.content
            return None
        except Exception as e:
            if self.config.full_debug:
                print(f"   Judge API error: {e}")
            raise

    def _parse_judge_score(self, judgment: str) -> int:
        """Parse the judge's score from the response."""
        match = self._score_pattern.search(judgment)
        if match:
            return int(match.group(1))
        return 0  # Fallback score

    async def _judge_response(
        self,
        question: str,
        answer: str,
        reference: Optional[str] = None
    ) -> Tuple[int, str]:
        """
        Judge a model response using the LLM judge.
        
        Returns:
            Tuple of (score: 1-5, judgment: str)
        """
        messages = judge_prompt_with_reference(question, answer, reference)
        
        try:
            judgment = await self._rate_limited_judge_call(messages)
            
            if not judgment:
                async with self._metrics_lock:
                    self.judge_error_count += 1
                if self.config.use_fallback_scoring:
                    async with self._metrics_lock:
                        self.fallback_count += 1
                    return 0, "JUDGE_ERROR_EMPTY_RESPONSE"
                return 0, "JUDGE_ERROR_EMPTY_RESPONSE"
            
            score = self._parse_judge_score(judgment)
            return score, judgment
            
        except Exception as e:
            async with self._metrics_lock:
                self.judge_error_count += 1
            if self.config.use_fallback_scoring:
                async with self._metrics_lock:
                    self.fallback_count += 1
                return 0, f"JUDGE_ERROR: {str(e)}"
            return 0, f"JUDGE_ERROR: {str(e)}"

    async def rollout_and_score_eval(
        self,
        item: Dict,
        server: APIServerConfig,
    ) -> Optional[Dict]:
        """Run multi-turn evaluation on a single item and return the result."""
        turns = item["turns"]
        references = item.get("reference", [])
        
        system_content = self._create_system_content()
        
        # Initialize conversation
        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        
        # Store responses and scores for each turn
        turn_responses = []
        turn_scores = []
        turn_judgments = []
        turn_valid_formats = []
        
        # Build API call parameters
        kwargs = {
            "model": server.model_name,
            "temperature": self.config.eval_temperature,
        }
        if self.config.eval_max_tokens > 0:
            kwargs["max_tokens"] = self.config.eval_max_tokens
        
        # Process each turn
        for turn_idx, turn_prompt in enumerate(turns):
            # Add user message
            messages.append({"role": "user", "content": turn_prompt})
            
            # Get model response
            response_text = ""
            for attempt in range(self.config.max_retries):
                try:
                    response = await self.server.chat_completion(
                        messages=messages,
                        **kwargs
                    )
                    response_text = response.choices[0].message.content or ""
                    
                    if len(response_text) >= self.config.min_response_length:
                        break
                        
                except Exception as e:
                    if self.config.full_debug:
                        print(f"  API error turn {turn_idx + 1} (attempt {attempt + 1}): {e}")
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(self.config.retry_delay)
                    continue
            
            if not response_text:
                return None
            
            # Validate thinking format and extract actual response
            is_valid_format, extracted_response = validate_thinking_format(
                response_text,
                self.config.thinking_mode
            )
            turn_valid_formats.append(is_valid_format)
            
            # Add assistant response to conversation
            messages.append({"role": "assistant", "content": response_text})
            turn_responses.append(response_text)
            
            # Build context for judging (include conversation history for turn 2)
            if turn_idx == 0:
                judge_question = turn_prompt
            else:
                # For turn 2, include context from turn 1
                judge_question = f"Context from previous turn:\nUser: {turns[0]}\nAssistant: {turn_responses[0]}\n\nCurrent turn:\nUser: {turn_prompt}"
            
            # Get reference for this turn if available
            turn_reference = references[turn_idx] if turn_idx < len(references) else None
            
            # Judge the response (use extracted response without think tags)
            score, judgment = await self._judge_response(
                judge_question,
                extracted_response,
                turn_reference
            )
            
            turn_scores.append(score)
            turn_judgments.append(judgment)
            
            if self.config.full_debug:
                print(f"  Turn {turn_idx + 1} score: {score}")
        
        # Extract thinking content if applicable
        thinking_contents = []
        for resp in turn_responses:
            thinking = extract_thinking_content(resp) if self.config.thinking_mode else None
            thinking_contents.append(thinking)
        
        return {
            "item_id": item["id"],
            "category": item["category"],
            "turns": turns,
            "responses": turn_responses,
            "extracted_responses": [
                validate_thinking_format(r, self.config.thinking_mode)[1] 
                for r in turn_responses
            ],
            "scores": turn_scores,
            "judgments": turn_judgments,
            "references": references,
            "format_valid": turn_valid_formats,
            "thinking_contents": thinking_contents,
            "avg_score": sum(turn_scores) / len(turn_scores) if turn_scores else 0,
            "score_turn_1": turn_scores[0] if len(turn_scores) > 0 else 0,
            "score_turn_2": turn_scores[1] if len(turn_scores) > 1 else 0,
        }

    async def evaluate(self, *args, **kwargs) -> Dict:
        """Run the full MT-Bench evaluation."""
        print(f"\n{'='*60}")
        print("Starting MT-Bench Evaluation (Multi-Turn with LLM Judge)")
        print(f"{'='*60}")
        print(f"  Total questions: {len(self.eval_items)}")
        print(f"  Judge model: {self.config.judge_model_name}")
        print(f"  Thinking mode: {self.config.thinking_mode}")
        print(f"{'='*60}\n")
        
        # Reset metrics
        self.judge_error_count = 0
        self.fallback_count = 0
        
        # Create evaluation tasks
        async def eval_task(item):
            return await self.rollout_and_score_eval(item, self.server_configs[0])
        
        tasks = [eval_task(item) for item in self.eval_items]
        
        # Run with progress bar
        results = await tqdm_asyncio.gather(
            *tasks,
            desc="Evaluating MT-Bench"
        )
        
        # Filter out failed results
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            print("Warning: No valid evaluation results obtained")
            return {"error": "No valid results", "avg_score": 0.0}
        
        # Calculate overall metrics
        total = len(valid_results)
        avg_score = sum(r["avg_score"] for r in valid_results) / total
        avg_turn_1 = sum(r["score_turn_1"] for r in valid_results) / total
        avg_turn_2 = sum(r["score_turn_2"] for r in valid_results) / total
        
        # Calculate per-category metrics
        category_metrics = {}
        for r in valid_results:
            cat = r["category"]
            if cat not in category_metrics:
                category_metrics[cat] = {"scores": [], "turn_1": [], "turn_2": []}
            category_metrics[cat]["scores"].append(r["avg_score"])
            category_metrics[cat]["turn_1"].append(r["score_turn_1"])
            category_metrics[cat]["turn_2"].append(r["score_turn_2"])
        
        for cat in category_metrics:
            scores = category_metrics[cat]["scores"]
            t1 = category_metrics[cat]["turn_1"]
            t2 = category_metrics[cat]["turn_2"]
            category_metrics[cat]["avg_score"] = sum(scores) / len(scores) if scores else 0
            category_metrics[cat]["avg_turn_1"] = sum(t1) / len(t1) if t1 else 0
            category_metrics[cat]["avg_turn_2"] = sum(t2) / len(t2) if t2 else 0
            category_metrics[cat]["count"] = len(scores)
        
        # Format compliance
        format_valid_t1 = sum(1 for r in valid_results if r["format_valid"][0])
        format_valid_t2 = sum(1 for r in valid_results if len(r["format_valid"]) > 1 and r["format_valid"][1])
        
        metrics = {
            "avg_score": avg_score,
            "avg_score_turn_1": avg_turn_1,
            "avg_score_turn_2": avg_turn_2,
            "total_evaluated": total,
            "format_compliance_turn_1": format_valid_t1 / total if total > 0 else 0.0,
            "format_compliance_turn_2": format_valid_t2 / total if total > 0 else 0.0,
            "judge_error_rate": self.judge_error_count / (total * 2) if total > 0 else 0.0,
            "fallback_rate": self.fallback_count / (total * 2) if total > 0 else 0.0,
            "category_metrics": category_metrics,
        }
        
        print(f"\n{'='*60}")
        print("MT-Bench Evaluation Results")
        print(f"{'='*60}")
        print(f"  Overall Average Score: {avg_score:.2f}/5.0")
        print(f"  Turn 1 Average: {avg_turn_1:.2f}/5.0")
        print(f"  Turn 2 Average: {avg_turn_2:.2f}/5.0")
        print(f"  Total Evaluated: {total}")
        if self.config.thinking_mode:
            print(f"  Format Compliance (T1): {format_valid_t1 / total:.2%}")
            print(f"  Format Compliance (T2): {format_valid_t2 / total:.2%}")
        print(f"\n  Per-Category Breakdown:")
        for cat, data in sorted(category_metrics.items(), key=lambda x: -x[1]["avg_score"]):
            print(f"    {cat}: {data['avg_score']:.2f} (T1: {data['avg_turn_1']:.2f}, T2: {data['avg_turn_2']:.2f}) [{data['count']} items]")
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
            "mtbench/avg_score": metrics.get("avg_score", 0),
            "mtbench/avg_score_turn_1": metrics.get("avg_score_turn_1", 0),
            "mtbench/avg_score_turn_2": metrics.get("avg_score_turn_2", 0),
            "mtbench/total_evaluated": metrics.get("total_evaluated", 0),
            "mtbench/format_compliance_turn_1": metrics.get("format_compliance_turn_1", 0),
            "mtbench/format_compliance_turn_2": metrics.get("format_compliance_turn_2", 0),
            "mtbench/judge_error_rate": metrics.get("judge_error_rate", 0),
        }
        
        # Log per-category scores
        for cat, data in metrics.get("category_metrics", {}).items():
            safe_name = cat.replace(" ", "_")[:20]
            log_dict[f"mtbench/score_{safe_name}"] = data.get("avg_score", 0)
        
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
    MTBenchEvalEnv.cli()

