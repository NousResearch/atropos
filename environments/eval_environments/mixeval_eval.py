"""
MixEval Evaluation Environment for Atropos (Generative Mode with LLM Judge)

This environment evaluates models on MixEval - a ground-truth-based dynamic benchmark
derived from off-the-shelf benchmark mixtures, which evaluates LLMs with a highly
capable model ranking (0.96 correlation with Chatbot Arena).

Dataset: MixEval/MixEval
Paper: https://mixeval.github.io/

MixEval has two difficulty levels:
- MixEval (easy)
- MixEval_Hard

And two question types:
- Freeform: Open-ended questions scored on 0-5 scale
- Multiple Choice: Scored as 0 or 1 (correct/incorrect)

Model responses are evaluated by an LLM judge.

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
from string import ascii_uppercase
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


# Prompt construction helpers
MULTI_CHOICE_PROMPT = "Answer with the option letter from the given choices directly."
FREE_FORM_PROMPT = "Answer the question shortly."
FREE_FORM_PROMPT_BBH = "Answer the question. \nLet's think step by step."
FREE_FORM_PROMPT_GSM8K = "Answer the question. \nLet's think step by step."
FREE_FORM_PROMPT_MATH = "Answer the question. \nLet's think step by step."


def parse_options(options: List[str]) -> str:
    """Format options as A. option, B. option, etc."""
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    return "\n".join([f"{letter}. {opt}" for letter, opt in zip(option_letters, options)])


def construct_prompt_multichoice(entry: Dict) -> str:
    """Construct prompt for multiple choice questions."""
    prompt = entry.get("prompt", "")
    options = entry.get("options", [])
    context = entry.get("context", "")
    
    parsed_options = parse_options(options)
    
    if context and str(context).lower() not in ["none", "null", ""]:
        return f"{context}\n{prompt}\n{parsed_options}\n{MULTI_CHOICE_PROMPT}"
    else:
        return f"{prompt}\n{parsed_options}\n{MULTI_CHOICE_PROMPT}"


def construct_prompt_freeform(entry: Dict) -> str:
    """Construct prompt for freeform questions."""
    prompt = entry.get("prompt", "")
    context = entry.get("context", "")
    benchmark_name = entry.get("benchmark_name", "")
    
    # Select prompt suffix based on benchmark
    if benchmark_name == "BBH":
        suffix = FREE_FORM_PROMPT_BBH
    elif benchmark_name == "GSM8k":
        suffix = FREE_FORM_PROMPT_GSM8K
    elif benchmark_name == "MATH":
        suffix = FREE_FORM_PROMPT_MATH
    else:
        suffix = FREE_FORM_PROMPT
    
    prompt_with_suffix = f"{prompt}\n{suffix}"
    
    if context and str(context).lower() not in ["none", "null", ""]:
        return f"Question: {context}\n{prompt_with_suffix}"
    else:
        return f"Question: {prompt_with_suffix}"


# Judge prompt templates
def judge_freeform_prompt(question: str, answer: str, gold: str) -> List[Dict]:
    """Create judge prompt for freeform questions (scores 0.0 to 1.0)."""
    return [
        {"role": "system", "content": "In this task, I want you to act as a judge."},
        {
            "role": "user",
            "content": f"""You will be provided with a question, its golden answer(s), and the model's answer, while the context of the question is not given here. Your task is to judge how correct the model's answer is based on the golden answer(s), without seeing the context of the question, and then give a correctness score. The correctness score should be one of the below numbers: 0.0 (totally wrong), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, or 1.0 (totally right). Your should first briefly give your reasoning process regarding how the model's answer conforms to or contradicts the golden answer(s), and then give the correctness score. The correctness score must strictly follow this format: "[[score]]", e.g., "The correctness score: [[0.5]]". 

Note that each one of the golden answers is considered correct. Thus if the model's answer matches any one of the golden answers, it should be considered correct. Judge the below case, give the brief reasoning process and the correctness score.

Question: {question}
Golden Answer(s): {gold}
Model's Answer: {answer}
Your Judgment:
""",
        },
    ]


def judge_multichoice_prompt(question: str, options: List[str], answer: str, gold: str) -> List[Dict]:
    """Create judge prompt for multiple choice questions (scores 0 or 1)."""
    parsed_options = parse_options(options)
    return [
        {"role": "system", "content": "In this task, I want you to act as an option extractor."},
        {
            "role": "user",
            "content": f"""You will be provided with a multiple-choice question, its options, the gold answer, and the model's answer, while the context of the question is not given here. Your task is to extract or judge which option is chosen by the model based on its response, and to determine whether or not the model answered correctly. The model scores can either be 0 (incorrect) or 1 (correct). The correctness score must strictly follow this format: "[[score]]", e.g., "The correctness score: [[1]]".

Question: {question}
Options:
{parsed_options}
Golden Answer: {gold}
Model's Answer: {answer}
Your Judgment:
""",
        },
    ]


class MixEvalConfig(BaseEnvConfig):
    """Configuration for MixEval evaluation environment with LLM judge."""
    
    # Dataset configuration
    dataset_name: str = Field(
        default="MixEval/MixEval",
        description="HuggingFace dataset name"
    )
    difficulty: str = Field(
        default="easy",
        description="Difficulty level: 'easy' (MixEval) or 'hard' (MixEval_Hard)"
    )
    question_types: List[str] = Field(
        default=["freeform", "multichoice"],
        description="Question types to evaluate: 'freeform', 'multichoice', or both"
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
        default=10,
        ge=1,
        description="Maximum number of concurrent judge API calls"
    )
    judge_rate_limit_delay: float = Field(
        default=0.2,
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
    max_num_workers: int = 512
    max_eval_workers: int = 128
    max_num_workers_per_node: int = 64
    use_wandb: bool = True
    rollout_server_url: str = "http://localhost:8000"
    total_steps: int = 1
    wandb_name: str = "mixeval_eval"
    steps_per_eval: int = 1


class MixEvalEnv(BaseEnv):
    """
    MixEval Evaluation Environment with LLM Judge.
    
    Evaluates general knowledge and reasoning using MixEval benchmark.
    Uses an LLM judge to score responses.
    """

    name = "mixeval_eval"

    def __init__(
        self,
        config: MixEvalConfig,
        server_configs: List[APIServerConfig],
        slurm_job_id: Optional[str] = None,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm_job_id, testing)
        self.config: MixEvalConfig = config
        self.eval_items: List[Dict] = []
        self._dataset_loaded = False
        
        # Setup judge client
        self.judge_client = None
        self._setup_judge_client()
        
        # Rate limiting for judge calls
        self.judge_semaphore = asyncio.Semaphore(self.config.judge_max_concurrent_calls)
        
        # Pre-compile regex for score extraction
        self._score_pattern_float = re.compile(r"\[\[(\d\.?\d?)\]\]")
        self._score_pattern_int = re.compile(r"\[\[([01])\]\]")
        
        # Thread-safe metrics tracking
        self._metrics_lock = asyncio.Lock()
        self.judge_error_count = 0
        self.fallback_count = 0

    @classmethod
    def config_cls(cls) -> type:
        return MixEvalConfig

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
            
        print(f"\nMixEval Evaluation Setup (with LLM Judge):")
        print(f"  Dataset: {self.config.dataset_name}")
        print(f"  Difficulty: {self.config.difficulty}")
        print(f"  Question types: {self.config.question_types}")
        print(f"  Thinking mode: {self.config.thinking_mode}")
        print(f"  Judge model: {self.config.judge_model_name}")
        print(f"  Judge endpoint: {self.config.judge_base_url}")
        if self.config.thinking_mode:
            thinking_prompt = get_default_thinking_prompt(self.config.custom_thinking_prompt)
            print(f"  Thinking prompt: {thinking_prompt[:80]}...")
        print(f"  Loaded {len(self.eval_items)} evaluation items")

    async def _load_dataset(self) -> None:
        """Load and process the MixEval dataset."""
        # Determine subset based on difficulty
        subset = "MixEval" if self.config.difficulty == "easy" else "MixEval_Hard"
        
        print(f"Loading MixEval dataset: {self.config.dataset_name} ({subset})...")
        
        self.eval_items = []
        
        try:
            dataset = load_dataset(
                self.config.dataset_name,
                subset,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
        
        # Load freeform questions
        if "freeform" in self.config.question_types:
            if "free_form" in dataset:
                for idx, item in enumerate(dataset["free_form"]):
                    prompt = construct_prompt_freeform(item)
                    targets = item.get("target", [])
                    gold = "; ".join([f"<answer {i+1}> {t}" for i, t in enumerate(targets)])
                    
                    self.eval_items.append({
                        "id": f"freeform_{idx}",
                        "type": "freeform",
                        "prompt": prompt,
                        "raw_prompt": item.get("prompt", ""),
                        "target": targets,
                        "gold_formatted": gold,
                        "options": None,
                        "benchmark_name": item.get("benchmark_name", ""),
                        "problem_type": item.get("problem_type", ""),
                    })
                print(f"  Loaded {len([i for i in self.eval_items if i['type'] == 'freeform'])} freeform items")
            else:
                print("  Warning: free_form split not found")
        
        # Load multiple choice questions
        if "multichoice" in self.config.question_types:
            if "multiple_choice" in dataset:
                for idx, item in enumerate(dataset["multiple_choice"]):
                    prompt = construct_prompt_multichoice(item)
                    options = item.get("options", [])
                    targets = item.get("target", [])
                    
                    # Convert target indices to letters
                    gold_letters = [ascii_uppercase[int(t)] for t in targets if int(t) < len(options)]
                    gold = f"{gold_letters[0]}. {options[int(targets[0])]}" if targets and int(targets[0]) < len(options) else ""
                    
                    self.eval_items.append({
                        "id": f"multichoice_{idx}",
                        "type": "multichoice",
                        "prompt": prompt,
                        "raw_prompt": item.get("prompt", ""),
                        "target": targets,
                        "gold_formatted": gold,
                        "options": options,
                        "benchmark_name": item.get("benchmark_name", ""),
                        "problem_type": item.get("problem_type", ""),
                    })
                print(f"  Loaded {len([i for i in self.eval_items if i['type'] == 'multichoice'])} multichoice items")
            else:
                print("  Warning: multiple_choice split not found")
        
        # Shuffle with seed
        random.seed(self.config.shuffle_seed)
        random.shuffle(self.eval_items)
        
        self._dataset_loaded = True
        print(f"Total: Loaded {len(self.eval_items)} evaluation items")

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

    def _parse_freeform_score(self, judgment: str) -> float:
        """Parse the judge's score for freeform questions (0.0-1.0)."""
        match = self._score_pattern_float.search(judgment)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return 0.0
        return 0.0

    def _parse_multichoice_score(self, judgment: str) -> int:
        """Parse the judge's score for multiple choice questions (0 or 1)."""
        match = self._score_pattern_int.search(judgment)
        if match:
            return int(match.group(1))
        return 0

    async def _judge_response(
        self,
        item: Dict,
        answer: str,
    ) -> Tuple[float, str]:
        """
        Judge a model response using the LLM judge.
        
        Returns:
            Tuple of (score: float, judgment: str)
        """
        if item["type"] == "freeform":
            messages = judge_freeform_prompt(
                item["raw_prompt"],
                answer,
                item["gold_formatted"]
            )
        else:  # multichoice
            messages = judge_multichoice_prompt(
                item["raw_prompt"],
                item["options"],
                answer,
                item["gold_formatted"]
            )
        
        try:
            judgment = await self._rate_limited_judge_call(messages)
            
            if not judgment:
                async with self._metrics_lock:
                    self.judge_error_count += 1
                if self.config.use_fallback_scoring:
                    async with self._metrics_lock:
                        self.fallback_count += 1
                    return 0.0, "JUDGE_ERROR_EMPTY_RESPONSE"
                return 0.0, "JUDGE_ERROR_EMPTY_RESPONSE"
            
            if item["type"] == "freeform":
                score = self._parse_freeform_score(judgment)
            else:
                score = float(self._parse_multichoice_score(judgment))
            
            return score, judgment
            
        except Exception as e:
            async with self._metrics_lock:
                self.judge_error_count += 1
            if self.config.use_fallback_scoring:
                async with self._metrics_lock:
                    self.fallback_count += 1
                return 0.0, f"JUDGE_ERROR: {str(e)}"
            return 0.0, f"JUDGE_ERROR: {str(e)}"

    async def rollout_and_score_eval(
        self,
        item: Dict,
        server: APIServerConfig,
    ) -> Optional[Dict]:
        """Run evaluation on a single item and return the result."""
        system_content = self._create_system_content()
        
        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": item["prompt"]})
        
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
        
        # Validate thinking format and extract actual response
        is_valid_format, extracted_response = validate_thinking_format(
            response_text,
            self.config.thinking_mode
        )
        
        # Extract thinking content if present
        thinking_content = extract_thinking_content(response_text) if self.config.thinking_mode else None
        
        # Judge the response
        score, judgment = await self._judge_response(item, extracted_response)
        
        if self.config.full_debug:
            print(f"\n--- Item: {item['id']} ({item['type']}) ---")
            print(f"Score: {score}")
        
        return {
            "item_id": item["id"],
            "type": item["type"],
            "benchmark_name": item.get("benchmark_name", ""),
            "problem_type": item.get("problem_type", ""),
            "prompt": item["prompt"][:200],
            "gold": item["gold_formatted"],
            "response": response_text,
            "extracted_response": extracted_response,
            "score": score,
            "judgment": judgment,
            "format_valid": is_valid_format,
            "thinking_content": thinking_content,
            "has_thinking": thinking_content is not None,
        }

    async def evaluate(self, *args, **kwargs) -> Dict:
        """Run the full MixEval evaluation."""
        print(f"\n{'='*60}")
        print("Starting MixEval Evaluation (with LLM Judge)")
        print(f"{'='*60}")
        print(f"  Difficulty: {self.config.difficulty}")
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
            desc="Evaluating MixEval"
        )
        
        # Filter out failed results
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            print("Warning: No valid evaluation results obtained")
            return {"error": "No valid results", "avg_score": 0.0}
        
        # Calculate overall metrics
        total = len(valid_results)
        avg_score = sum(r["score"] for r in valid_results) / total
        
        # Calculate per-type metrics
        freeform_results = [r for r in valid_results if r["type"] == "freeform"]
        multichoice_results = [r for r in valid_results if r["type"] == "multichoice"]
        
        freeform_score = sum(r["score"] for r in freeform_results) / len(freeform_results) if freeform_results else 0
        multichoice_score = sum(r["score"] for r in multichoice_results) / len(multichoice_results) if multichoice_results else 0
        
        # Calculate per-benchmark metrics
        benchmark_metrics = {}
        for r in valid_results:
            bench = r.get("benchmark_name", "unknown") or "unknown"
            if bench not in benchmark_metrics:
                benchmark_metrics[bench] = {"scores": [], "count": 0}
            benchmark_metrics[bench]["scores"].append(r["score"])
            benchmark_metrics[bench]["count"] += 1
        
        for bench in benchmark_metrics:
            scores = benchmark_metrics[bench]["scores"]
            benchmark_metrics[bench]["avg_score"] = sum(scores) / len(scores) if scores else 0
        
        # Format compliance
        format_valid = sum(1 for r in valid_results if r.get("format_valid", True))
        has_thinking = sum(1 for r in valid_results if r.get("has_thinking", False))
        
        metrics = {
            "avg_score": avg_score,
            "freeform_score": freeform_score,
            "multichoice_accuracy": multichoice_score,
            "total_evaluated": total,
            "freeform_count": len(freeform_results),
            "multichoice_count": len(multichoice_results),
            "format_compliance_rate": format_valid / total if total > 0 else 0.0,
            "thinking_utilization_rate": has_thinking / total if total > 0 else 0.0,
            "judge_error_rate": self.judge_error_count / total if total > 0 else 0.0,
            "fallback_rate": self.fallback_count / total if total > 0 else 0.0,
            "benchmark_metrics": benchmark_metrics,
        }
        
        print(f"\n{'='*60}")
        print("MixEval Evaluation Results")
        print(f"{'='*60}")
        print(f"  Overall Average Score: {avg_score:.2%}")
        print(f"  Freeform Score: {freeform_score:.2%} ({len(freeform_results)} items)")
        print(f"  Multiple Choice Accuracy: {multichoice_score:.2%} ({len(multichoice_results)} items)")
        print(f"  Total Evaluated: {total}")
        if self.config.thinking_mode:
            print(f"  Format Compliance: {format_valid / total:.2%}")
            print(f"  Thinking Utilization: {has_thinking / total:.2%}")
        print(f"  Judge Error Rate: {self.judge_error_count / total:.2%}")
        print(f"\n  Per-Benchmark Breakdown:")
        for bench, data in sorted(benchmark_metrics.items(), key=lambda x: -x[1]["avg_score"]):
            print(f"    {bench}: {data['avg_score']:.2%} [{data['count']} items]")
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
            "mixeval/avg_score": metrics.get("avg_score", 0),
            "mixeval/freeform_score": metrics.get("freeform_score", 0),
            "mixeval/multichoice_accuracy": metrics.get("multichoice_accuracy", 0),
            "mixeval/total_evaluated": metrics.get("total_evaluated", 0),
            "mixeval/format_compliance_rate": metrics.get("format_compliance_rate", 0),
            "mixeval/thinking_utilization_rate": metrics.get("thinking_utilization_rate", 0),
            "mixeval/judge_error_rate": metrics.get("judge_error_rate", 0),
        }
        
        # Log per-benchmark scores
        for bench, data in metrics.get("benchmark_metrics", {}).items():
            safe_name = bench.replace(" ", "_")[:20]
            log_dict[f"mixeval/score_{safe_name}"] = data.get("avg_score", 0)
        
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
    MixEvalEnv.cli()

