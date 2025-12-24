import asyncio
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import openai
from datasets import load_dataset
from pydantic import Field
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
    Item,
    ScoredDataGroup,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer
from eval_helpers import (
    get_default_thinking_prompt,
    create_system_content,
    save_eval_results,
)


class ArenaHardConfig(BaseEnvConfig):
    """Configuration for Arena-Hard environment with Claude Sonnet judge."""

    # Thinking mode configuration
    thinking_mode: bool = Field(
        default=False,
        description="Whether to enable thinking mode with <think></think> tags for model responses.",
    )

    custom_thinking_prompt: Optional[str] = Field(
        default=None,
        description="Custom thinking prompt for model responses. If None, uses the default thinking prompt.",
    )

    custom_system_prompt: Optional[str] = Field(
        default=None,
        description="Custom system prompt for model responses. In non-thinking mode, used directly. In thinking mode, appended to thinking prompt.",  # noqa
    )

    # Judge model configuration
    judge_model_name: str = Field(
        default="claude-sonnet-4-20250514",
        description="Model name for the judge (e.g., 'claude-sonnet-4-20250514', 'gpt-4o', etc.)",
    )

    judge_base_url: str = Field(
        default="https://api.anthropic.com/v1",
        description="Base URL for the judge model API (e.g., 'https://api.anthropic.com/v1', 'https://api.openai.com/v1')",  # noqa
    )

    judge_api_key_env: str = Field(
        default="ANTHROPIC_API_KEY",
        description="Environment variable name containing the API key for the judge model (e.g., 'ANTHROPIC_API_KEY', 'OPENAI_API_KEY')",  # noqa
    )

    # Judge configuration
    judge_temperature: float = Field(
        default=0.2,
        description="Temperature for judge completions.",
    )

    judge_max_tokens: int = Field(
        default=0,
        description="Maximum tokens for judge completions (0 = use model default).",
    )

    # Retry configuration for judge calls
    judge_max_retries: int = Field(
        default=3,
        ge=1,
        description="Maximum number of retries for failed judge API calls.",
    )

    judge_retry_multiplier: float = Field(
        default=1.0,
        ge=0.0,
        description="Exponential backoff multiplier for judge retries.",
    )

    judge_retry_max_wait: int = Field(
        default=10,
        ge=1,
        description="Maximum wait time in seconds for judge retries.",
    )

    # Model generation configuration
    eval_temperature: float = Field(
        default=0.6,
        description="Temperature for model evaluation completions.",
    )

    rollout_temperature: float = Field(
        default=1.0,
        description="Temperature for training rollout completions.",
    )

    eval_max_tokens: int = Field(
        default=0,
        description="Maximum tokens for evaluation completions (0 = use model default).",
    )

    train_max_tokens: int = Field(
        default=16384,
        description="Maximum tokens for training completions.",
    )

    # Retry configuration
    max_retries: int = Field(
        default=3,
        ge=1,
        description="Maximum number of retries for failed API calls.",
    )

    retry_delay: float = Field(
        default=1.0,
        ge=0.0,
        description="Delay in seconds between retry attempts.",
    )

    min_response_length: int = Field(
        default=10,
        ge=1,
        description="Minimum response length to consider valid.",
    )

    # Dataset configuration
    train_prompt_dataset: str = Field(
        default="NousResearch/arena-hard-v1-prompts",
        description="Training prompt dataset name (HuggingFace) or path to local JSONL file.",
    )

    train_baseline_dataset: str = Field(
        default="NousResearch/gpt-4-0314-baseline-arenahard",
        description="Training baseline responses dataset name (HuggingFace) or path to local JSONL file.",
    )

    eval_prompt_dataset: str = Field(
        default="NousResearch/arena-hard-v1-prompts",
        description="Evaluation prompt dataset name (HuggingFace) or path to local JSONL file.",
    )

    eval_baseline_dataset: str = Field(
        default="NousResearch/gpt-4-0314-baseline-arenahard",
        description="Evaluation baseline responses dataset name (HuggingFace) or path to local JSONL file.",
    )

    train_split: str = Field(
        default="train",
        description="Split to use for training datasets (only for HuggingFace datasets).",
    )

    eval_split: str = Field(
        default="train",
        description="Split to use for evaluation datasets (only for HuggingFace datasets).",
    )

    # Debug configuration
    full_debug: bool = Field(
        default=False,
        description="Enable full debug mode - logs every API request and response.",
    )


class ArenaHardEnv(BaseEnv):
    name = "arena_hard"
    env_config_cls = ArenaHardConfig

    def __init__(
        self,
        config: ArenaHardConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: ArenaHardConfig = config
        self.percent_correct_buffer = []
        self.eval_metrics = []

        # Metrics tracking
        self.win_count = 0
        self.tie_count = 0
        self.loss_count = 0
        self.total_judgments = 0
        self.error_count = 0
        self.rollouts_for_wandb = []

        # Initialize judge client with configurable settings
        judge_api_key = os.environ.get(self.config.judge_api_key_env)
        if not judge_api_key:
            raise ValueError(
                f"Judge API key not found in environment variable: {self.config.judge_api_key_env}"
            )

        self.judge_client = openai.AsyncOpenAI(
            api_key=judge_api_key,
            base_url=self.config.judge_base_url,
        )

        # Pre-compile regex patterns for judgment parsing (Arena-Hard format)
        self._score_patterns = [
            re.compile(r"\[\[([AB<>=]+)\]\]"),
            re.compile(r"\[([AB<>=]+)\]"),
        ]

        # Pre-compile regex patterns for thinking mode parsing
        self._think_pattern = re.compile(r"<think>")
        self._think_close_pattern = re.compile(r"</think>")
        self._think_content_pattern = re.compile(r"</think>\s*(.*)", re.DOTALL)
        self._thinking_extract_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)

        # Thinking system prompt (use custom one if provided, otherwise default)
        self.thinking_system_prompt = self._get_thinking_prompt()

        # Judge prompt templates based on Arena-Hard-Auto
        self.judge_system_prompt = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by providing a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "[[A>B]]" if assistant A is better, "[[B>A]]" if assistant B is better, and "[[A=B]]" for a tie."""  # noqa

        self.judge_prompt_template = """<|User Prompt|>
{question}

<|The Start of Assistant A's Answer|>
{answer_a}
<|The End of Assistant A's Answer|>

<|The Start of Assistant B's Answer|>
{answer_b}
<|The End of Assistant B's Answer|>"""

        self.iter = 0

    def _get_thinking_prompt(self) -> str:
        """Get thinking system prompt for model responses."""
        # Arena Hard uses a slightly different default thinking prompt
        return get_default_thinking_prompt(self.config.custom_thinking_prompt)

    def _get_system_prompt(self) -> Optional[str]:
        """Get system prompt for non-thinking mode."""
        return self.config.custom_system_prompt
    
    def _create_system_content(self) -> Optional[str]:
        """Create system message content based on thinking mode."""
        return create_system_content(
            self.config.thinking_mode,
            self.config.custom_thinking_prompt,
            self.config.custom_system_prompt
        )

    def _load_dataset(self, dataset_path: str, split: str = None) -> List[Dict]:
        """Load dataset using HuggingFace load_dataset."""
        import os

        try:
            # Check if it's a local file
            if os.path.exists(dataset_path):
                if dataset_path.endswith(".jsonl") or dataset_path.endswith(".json"):
                    dataset = load_dataset(
                        "json",
                        data_files=dataset_path,
                        split=split or "train",
                        trust_remote_code=True,
                    )
                elif dataset_path.endswith(".csv"):
                    dataset = load_dataset(
                        "csv",
                        data_files=dataset_path,
                        split=split or "train",
                        trust_remote_code=True,
                    )
                elif dataset_path.endswith(".parquet"):
                    dataset = load_dataset(
                        "parquet",
                        data_files=dataset_path,
                        split=split or "train",
                        trust_remote_code=True,
                    )
                else:
                    dataset = load_dataset(
                        "json",
                        data_files=dataset_path,
                        split=split or "train",
                        trust_remote_code=True,
                    )

                print(
                    f"Loaded local dataset from {dataset_path} with {len(dataset)} examples"
                )

            else:
                # HuggingFace dataset
                if split:
                    dataset = load_dataset(
                        dataset_path, split=split, trust_remote_code=True
                    )
                else:
                    dataset_dict = load_dataset(dataset_path, trust_remote_code=True)
                    if hasattr(dataset_dict, "keys"):
                        available_splits = list(dataset_dict.keys())
                        if available_splits:
                            dataset = dataset_dict[available_splits[0]]
                            print(
                                f"No split specified, using '{available_splits[0]}' split"
                            )
                        else:
                            dataset = dataset_dict
                    else:
                        dataset = dataset_dict

                print(
                    f"Loaded HuggingFace dataset {dataset_path} with {len(dataset)} examples"
                )

            return dataset

        except Exception as e:
            print(f"Error loading dataset {dataset_path}: {e}")
            raise

    @classmethod
    def config_init(cls) -> Tuple[ArenaHardConfig, List[APIServerConfig]]:
        env_config = ArenaHardConfig(
            tokenizer_name="NousResearch/Hermes-3-Llama-3.1-8B",
            group_size=8,
            use_wandb=True,
            max_num_workers_per_node=16,
            rollout_server_url="http://localhost:8000",
            total_steps=500,
            batch_size=512,
            steps_per_eval=25,
            max_token_length=1024 * 16,
            inference_weight=1.0,
            wandb_name="arena_hard",
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.1,
            min_batch_allocation=0.1,
            judge_model_name="claude-sonnet-4-20250514",
            judge_base_url="https://api.anthropic.com/v1",
            judge_api_key_env="ANTHROPIC_API_KEY",
            judge_temperature=0.0,
            judge_max_tokens=0,  # Use model default
            judge_max_retries=3,
            judge_retry_multiplier=1.0,
            judge_retry_max_wait=10,
            eval_temperature=0.0,
            rollout_temperature=1.0,
            eval_max_tokens=0,  # Use model default
            train_max_tokens=1024 * 16,
            full_debug=True,
            max_retries=3,
            retry_delay=1.0,
            min_response_length=10,
            thinking_mode=False,  # Enable thinking mode if desired
            custom_thinking_prompt=None,  # Use default thinking prompt
            custom_system_prompt=None,  # Custom system prompt (optional)
        )
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/Hermes-3-Llama-3.1-8B",
                base_url="http://localhost:9004/v1",
                api_key="x",
                num_max_requests_at_once=32,
                num_requests_for_eval=256,
            ),
        ]
        return env_config, server_configs

    async def setup(self) -> None:
        """Set up the environment by loading datasets."""
        # Load training datasets
        try:
            self.train_prompts = self._load_dataset(
                self.config.train_prompt_dataset, self.config.train_split
            )
            self.train_baselines = self._load_dataset(
                self.config.train_baseline_dataset, self.config.train_split
            )
            print(
                f"Loaded training datasets: {len(self.train_prompts)} prompts, {len(self.train_baselines)} baselines"
            )
        except Exception as e:
            print(f"Error loading training datasets: {e}")
            # Don't create fallback data as requested
            raise

        # Load evaluation datasets
        try:
            self.eval_prompts = self._load_dataset(
                self.config.eval_prompt_dataset, self.config.eval_split
            )
            self.eval_baselines = self._load_dataset(
                self.config.eval_baseline_dataset, self.config.eval_split
            )
            print(
                f"Loaded evaluation datasets: {len(self.eval_prompts)} prompts, {len(self.eval_baselines)} baselines"
            )
        except Exception as e:
            print(f"Error loading evaluation datasets: {e}")
            raise

        # Create UID to item mappings for quick lookup
        self.train_baseline_by_uid = {
            item.get("uid"): item for item in self.train_baselines
        }
        self.eval_baseline_by_uid = {
            item.get("uid"): item for item in self.eval_baselines
        }

        # Show configuration info
        print("\nArena-Hard Configuration:")
        print(f"  - Judge model: {self.config.judge_model_name}")
        print(f"  - Judge base URL: {self.config.judge_base_url}")
        print(f"  - Judge API key env: {self.config.judge_api_key_env}")
        print(f"  - Judge temperature: {self.config.judge_temperature}")
        print(f"  - Judge max retries: {self.config.judge_max_retries}")
        print(f"  - Judge retry multiplier: {self.config.judge_retry_multiplier}")
        print(f"  - Judge retry max wait: {self.config.judge_retry_max_wait}s")
        print(f"  - Eval temperature: {self.config.eval_temperature}")
        print(f"  - Rollout temperature: {self.config.rollout_temperature}")
        print(f"  - Group size: {self.config.group_size}")
        print(f"  - Thinking mode: {self.config.thinking_mode}")
        if self.config.thinking_mode:
            print(
                f"  - Custom thinking prompt: {'Yes' if self.config.custom_thinking_prompt else 'No (using default)'}"
            )
        print(
            f"  - Custom system prompt: {'Yes' if self.config.custom_system_prompt else 'No'}"
        )

        self.iter = 0

    def _format_debug_text(self, text: str, label: str) -> str:
        """Format text for debug output."""
        if not text:
            return f"{label}: <empty>"

        text_clean = text.strip()
        if len(text_clean) <= 200:
            return f"{label}: '{text_clean}'"

        first_100 = text_clean[:100]
        last_100 = text_clean[-100:]
        return f"{label}: '{first_100}...{last_100}' (total {len(text_clean)} chars)"

    def _log_full_debug_request(
        self, messages: List[Dict], params: Dict, context: str = ""
    ):
        """Log full debug information for API requests."""
        if not self.config.full_debug:
            return

        print(f"\n🔍 FULL DEBUG - API REQUEST [{context}]")
        print(f"   Parameters: {params}")

        for i, message in enumerate(messages):
            role = message.get("role", "unknown")
            content = message.get("content", "")
            print(
                f"   Message {i+1} ({role}): {self._format_debug_text(content, 'Content')}"
            )

    def _log_full_debug_response(self, completion, context: str = ""):
        """Log full debug information for API responses."""
        if not self.config.full_debug:
            return

        print(f"\n🔍 FULL DEBUG - API RESPONSE [{context}]")

        if hasattr(completion, "usage"):
            print(f"   Usage: {completion.usage}")

        if hasattr(completion, "choices") and completion.choices:
            for i, choice in enumerate(completion.choices):
                content = choice.message.content if hasattr(choice, "message") else ""
                finish_reason = (
                    choice.finish_reason
                    if hasattr(choice, "finish_reason")
                    else "unknown"
                )
                print(
                    f"   Choice {i+1}: {self._format_debug_text(content, 'Response')}"
                )
                print(f"   Finish reason: {finish_reason}")
        else:
            print("   No choices in response")

    def _reset_metrics(self) -> None:
        """Reset training metrics."""
        self.percent_correct_buffer = []
        self.win_count = 0
        self.tie_count = 0
        self.loss_count = 0
        self.total_judgments = 0
        self.error_count = 0

    async def get_next_item(self) -> Item:
        """Generate next training item."""
        self.iter += 1

        # Get next prompt sequentially
        prompt_item = self.train_prompts[self.iter % len(self.train_prompts)]

        # Find corresponding baseline response
        baseline_item = self.train_baseline_by_uid.get(prompt_item.get("uid"))
        if baseline_item is None:
            print(f"Warning: No baseline found for prompt {prompt_item.get('uid')}")
            # Skip to next item
            return await self.get_next_item()

        # Create prompt for model to answer
        prompt_text = prompt_item.get("prompt", "")
        if not prompt_text:
            print(f"Warning: Empty prompt for {prompt_item.get('uid')}")
            return await self.get_next_item()

        # Create prompt tuple for model completion
        if self.config.thinking_mode:
            prompt = tuple(
                [
                    frozenset(
                        {
                            "role": "system",
                            "content": self.thinking_system_prompt,
                        }.items()
                    ),
                    frozenset({"role": "user", "content": prompt_text}.items()),
                ]
            )
        else:
            system_prompt = self._get_system_prompt()
            if system_prompt:
                prompt = tuple(
                    [
                        frozenset({"role": "system", "content": system_prompt}.items()),
                        frozenset({"role": "user", "content": prompt_text}.items()),
                    ]
                )
            else:
                prompt = tuple(
                    [
                        frozenset({"role": "user", "content": prompt_text}.items()),
                    ]
                )

        # Return item with prompt and baseline for later judgment
        return (prompt, {"prompt_item": prompt_item, "baseline_item": baseline_item})

    @retry(
        stop=stop_after_attempt(3),  # Will be overridden by instance config
        wait=wait_random_exponential(
            multiplier=1, max=10
        ),  # Will be overridden by instance config
    )
    async def _judge_api_call(self, messages: List[Dict]):
        """Make a single judge API call with retry decorator."""
        debug_params = {"temperature": self.config.judge_temperature}
        if self.config.judge_max_tokens > 0:
            debug_params["max_tokens"] = self.config.judge_max_tokens
        
        self._log_full_debug_request(
            messages,
            debug_params,
            "JUDGE API CALL",
        )

        kwargs = {
            "model": self.config.judge_model_name,
            "messages": messages,
            "temperature": self.config.judge_temperature,
        }
        if self.config.judge_max_tokens > 0:
            kwargs["max_tokens"] = self.config.judge_max_tokens
        
        completion = await self.judge_client.chat.completions.create(**kwargs)

        self._log_full_debug_response(completion, "JUDGE API CALL")

        if not completion.choices:
            raise ValueError("No response from judge")

        if not completion.choices[0].message.content:
            raise ValueError("Empty judgment from judge")

        return completion

    async def judge_responses(
        self, question: str, answer_a: str, answer_b: str
    ) -> Tuple[str, str]:
        """Use configured judge model to judge two responses."""
        # Format the judge prompt
        user_content = self.judge_prompt_template.format(
            question=question, answer_a=answer_a, answer_b=answer_b
        )

        messages = [
            {"role": "system", "content": self.judge_system_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            # Create a new retry decorator with instance-specific configuration
            retry_decorator = retry(
                stop=stop_after_attempt(self.config.judge_max_retries),
                wait=wait_random_exponential(
                    multiplier=self.config.judge_retry_multiplier,
                    max=self.config.judge_retry_max_wait,
                ),
            )

            # Apply the retry decorator to the API call method
            retrying_judge_call = retry_decorator(self._judge_api_call)

            # Make the API call with retries
            completion = await retrying_judge_call(messages)

            judgment = completion.choices[0].message.content
            score = self._parse_judgment(judgment)
            return score, judgment

        except Exception as e:
            print(
                f"Judge API call failed after {self.config.judge_max_retries} attempts: {e}"
            )
            return "ERROR", f"Judge error: {e}"

    def _parse_judgment(self, judgment: str) -> str:
        """Parse judgment from Claude response using Arena-Hard format."""
        if not judgment:
            return "ERROR"

        # Look for [[A>B]], [[B>A]], [[A=B]] patterns
        for pattern in self._score_patterns:
            matches = pattern.findall(judgment.upper())
            if matches:
                match = matches[-1]  # Take the last match
                # Normalize the match to handle different formats
                if ">" in match and "A" in match and "B" in match:
                    if match.startswith("A"):
                        return "A>B"
                    elif match.startswith("B"):
                        return "B>A"
                elif "=" in match and "A" in match and "B" in match:
                    return "A=B"
                # Handle legacy formats just in case
                elif match == "A":
                    return "A>B"
                elif match == "B":
                    return "B>A"
                elif match == "C":
                    return "A=B"

        # Also check for explicit patterns in the text
        judgment_upper = judgment.upper()
        if "A>B" in judgment_upper or "[[A>B]]" in judgment_upper:
            return "A>B"
        elif "B>A" in judgment_upper or "[[B>A]]" in judgment_upper:
            return "B>A"
        elif "A=B" in judgment_upper or "[[A=B]]" in judgment_upper:
            return "A=B"
        elif "TIE" in judgment_upper:
            return "A=B"

        return "ERROR"

    def _validate_and_extract_thinking(self, response: str) -> Tuple[bool, str]:
        """
        Validate thinking format and extract content after </think> tags.

        Returns:
            (is_valid, extracted_content)
            - is_valid: True if thinking format is valid (exactly one pair of think tags)
            - extracted_content: Content after </think> tags, or original response if not thinking mode
        """
        if not self.config.thinking_mode:
            return True, response

        if not response or not isinstance(response, str):
            return False, ""

        # Check for exactly one pair of think tags
        think_open_count = len(self._think_pattern.findall(response))
        think_close_count = len(self._think_close_pattern.findall(response))

        if think_open_count != 1 or think_close_count != 1:
            return False, ""

        # Extract content after </think> tags
        match = self._think_content_pattern.search(response)
        if match:
            extracted_content = match.group(1).strip()
            return True, extracted_content
        else:
            return False, ""

    async def collect_trajectories(self, item: Item) -> Tuple[ScoredDataGroup, List]:
        """Collect and score model trajectories."""
        prompt_tuple, metadata = item
        prompt_item = metadata["prompt_item"]
        baseline_item = metadata["baseline_item"]

        # Convert prompt tuple to messages
        messages = []
        for role_dict in prompt_tuple:
            messages.append(dict(role_dict))

        # Generate multiple responses for training
        completion_params = {
            "n": self.config.group_size,
            "max_tokens": self.config.train_max_tokens,
            "temperature": self.config.rollout_temperature,
        }

        # Retry logic for model completion
        for attempt in range(self.config.max_retries):
            try:
                self._log_full_debug_request(
                    messages,
                    completion_params,
                    f"MODEL_TRAIN attempt {attempt + 1}/{self.config.max_retries}",
                )

                completions = await self.server.chat_completion(
                    messages=messages, **completion_params
                )

                self._log_full_debug_response(
                    completions,
                    f"MODEL_TRAIN attempt {attempt + 1}/{self.config.max_retries}",
                )

                if not completions.choices:
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(self.config.retry_delay)
                        continue
                    else:
                        return None, []

                # Validate completions
                valid_completions = []
                for completion_choice in completions.choices:
                    if (
                        completion_choice.message.content is not None
                        and isinstance(completion_choice.message.content, str)
                        and len(completion_choice.message.content.strip())
                        >= self.config.min_response_length
                    ):
                        valid_completions.append(completion_choice)

                if len(valid_completions) < len(completions.choices) // 2:
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(self.config.retry_delay)
                        continue

                break

            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    print(
                        f"Model completion failed (attempt {attempt + 1}/{self.config.max_retries}): {e}"
                    )
                    await asyncio.sleep(self.config.retry_delay)
                    continue
                else:
                    print(
                        f"Model completion failed after {self.config.max_retries} attempts: {e}"
                    )
                    return None, []

        # Judge each response against baseline
        to_score = []
        question = prompt_item.get("prompt", "")

        # Extract baseline answer from simplified structure
        baseline_answer = baseline_item.get("answer", "")

        if not baseline_answer:
            print(
                f"Warning: Could not extract baseline answer for {prompt_item.get('uid')}"
            )
            return None, []

        for completion_choice in valid_completions:
            model_answer = completion_choice.message.content

            # Validate thinking format if thinking mode is enabled
            is_valid_thinking, extracted_content = self._validate_and_extract_thinking(
                model_answer
            )

            if not is_valid_thinking:
                # Score 0 for invalid thinking format without judging
                trajectory_messages = messages + [
                    {"role": "assistant", "content": model_answer}
                ]
                to_score.append((tuple(trajectory_messages), 0.0))
                continue

            # Use extracted content for judging (original content if not thinking mode)
            content_for_judging = extracted_content

            # Judge model answer vs baseline (round 1: model=A, baseline=B)
            score_1, judgment_1 = await self.judge_responses(
                question, content_for_judging, baseline_answer
            )

            # Judge baseline vs model answer (round 2: baseline=A, model=B)
            score_2, judgment_2 = await self.judge_responses(
                question, baseline_answer, content_for_judging
            )

            # Combine scores using Arena-Hard logic
            final_score = self._combine_scores(score_1, score_2)

            # Create trajectory
            trajectory_messages = messages + [
                {"role": "assistant", "content": model_answer}
            ]
            to_score.append((tuple(trajectory_messages), final_score))

        # Score the trajectories
        scored_data = await self.score(to_score)
        return scored_data, []

    def _combine_scores(self, score_1: str, score_2: str) -> float:
        """Combine two judgment scores using Arena-Hard logic."""

        # Convert scores to numeric values
        def score_to_value(score: str, is_first_round: bool) -> float:
            if score == "ERROR":
                return 0.0
            elif score == "A>B":
                return (
                    1.0 if is_first_round else -1.0
                )  # A>B in round 1 = model wins, A>B in round 2 = baseline wins
            elif score == "B>A":
                return (
                    -1.0 if is_first_round else 1.0
                )  # B>A in round 1 = baseline wins, B>A in round 2 = model wins
            elif score == "A=B":
                return 0.0  # Tie
            else:
                return 0.0

        value_1 = score_to_value(score_1, True)
        value_2 = score_to_value(score_2, False)

        # Average the two scores
        return (value_1 + value_2) / 2.0

    async def score(self, rollout_group_data: List[Tuple]) -> Optional[ScoredDataGroup]:
        """Score a group of rollout data."""
        if not rollout_group_data:
            return None

        try:
            scores = ScoredDataGroup()
            scores["tokens"] = []
            scores["masks"] = []
            scores["scores"] = []

            for item in rollout_group_data:
                if not item or len(item) < 2 or not item[0]:
                    continue

                trajectory_messages = item[0]
                reward = item[1]

                # Update metrics
                self.total_judgments += 1
                if reward > 0.5:
                    self.win_count += 1
                elif reward < -0.5:
                    self.loss_count += 1
                else:
                    self.tie_count += 1

                # Tokenize for trainer
                out_dict = tokenize_for_trainer(self.tokenizer, trajectory_messages)
                tokens = out_dict["tokens"]
                masks = out_dict["masks"]

                # Skip bad examples
                if len([1 for mask in masks if mask != -100]) < 10:
                    continue

                scores["tokens"].append(tokens)
                scores["masks"].append(masks)
                scores["scores"].append(reward)

                if len(scores["tokens"]) >= self.config.group_size:
                    break

            if not scores["tokens"]:
                return None

            # Update percent correct buffer (convert -1,0,1 to 0,0.5,1 for compatibility)
            for score in scores["scores"]:
                normalized_score = max((score + 1) / 2, 0)  # Convert [-1,1] to [0,1]
                self.percent_correct_buffer.append(normalized_score)

            return scores

        except Exception as e:
            print(f"Error in score method: {e}")
            return None

    async def rollout_and_score_eval(self, eval_item: dict) -> dict:
        """Evaluate a single item."""
        try:
            prompt_item = eval_item
            baseline_item = self.eval_baseline_by_uid.get(prompt_item.get("uid"))

            if baseline_item is None:
                return {"score": 0.0, "sample": None}

            question = prompt_item.get("prompt", "")
            if not question:
                return {"score": 0.0, "sample": None}

            # Generate model response
            if self.config.thinking_mode:
                messages = [
                    {"role": "system", "content": self.thinking_system_prompt},
                    {"role": "user", "content": question},
                ]
            else:
                system_prompt = self._get_system_prompt()
                if system_prompt:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                    ]
                else:
                    messages = [{"role": "user", "content": question}]

            completion_params = {
                "n": 1,
                "temperature": self.config.eval_temperature,
                "split": "eval",
            }
            if self.config.eval_max_tokens > 0:
                completion_params["max_tokens"] = self.config.eval_max_tokens

            # Retry logic for evaluation
            for attempt in range(self.config.max_retries):
                try:
                    self._log_full_debug_request(
                        messages,
                        completion_params,
                        f"MODEL_EVAL attempt {attempt + 1}/{self.config.max_retries}",
                    )

                    completion = await self.server.chat_completion(
                        messages=messages, **completion_params
                    )

                    self._log_full_debug_response(
                        completion,
                        f"MODEL_EVAL attempt {attempt + 1}/{self.config.max_retries}",
                    )

                    if (
                        not completion.choices
                        or not completion.choices[0].message.content
                    ):
                        if attempt < self.config.max_retries - 1:
                            await asyncio.sleep(self.config.retry_delay)
                            continue
                        else:
                            return {"score": 0.0, "sample": None}

                    model_answer = completion.choices[0].message.content

                    if len(model_answer.strip()) < self.config.min_response_length:
                        if attempt < self.config.max_retries - 1:
                            await asyncio.sleep(self.config.retry_delay)
                            continue
                        else:
                            return {"score": 0.0, "sample": None}

                    break

                except Exception:
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(self.config.retry_delay)
                        continue
                    else:
                        return {"score": 0.0, "sample": None}

            # Validate thinking format and extract content for judging
            is_valid_thinking, content_for_judging = (
                self._validate_and_extract_thinking(model_answer)
            )

            if not is_valid_thinking:
                # Return 0 score for invalid thinking format
                return {"score": 0.0, "sample": None}

            # Extract baseline answer from simplified structure
            baseline_answer = baseline_item.get("answer", "")

            if not baseline_answer:
                return {"score": 0.0, "sample": None}

            # Judge both rounds using extracted content
            score_1, judgment_1 = await self.judge_responses(
                question, content_for_judging, baseline_answer
            )
            score_2, judgment_2 = await self.judge_responses(
                question, baseline_answer, content_for_judging
            )

            # Calculate final score for Arena-Hard compatibility
            final_score = self._combine_scores(score_1, score_2)

            # Convert to Arena-Hard winrate format (0.0 to 1.0)
            arena_score = max((final_score + 1) / 2, 0.0)

            sample = {
                "question": question,
                "model_answer": model_answer,  # Full response including thinking tags
                "model_answer_for_judging": content_for_judging,  # Content used for judging
                "baseline_answer": baseline_answer,
                "score_round_1": score_1,
                "judgment_round_1": judgment_1,
                "score_round_2": score_2,
                "judgment_round_2": judgment_2,
                "final_score": final_score,
                "arena_score": arena_score,
                "uid": prompt_item.get("uid"),
                "category": prompt_item.get("category", "unknown"),
                "cluster": prompt_item.get("cluster", "unknown"),
                "thinking_mode": self.config.thinking_mode,
                "thinking_valid": is_valid_thinking,
            }

            return {"score": arena_score, "sample": sample}

        except Exception as e:
            print(f"Error in evaluation: {e}")
            return {"score": 0.0, "sample": None}

    async def evaluate(self, *args, **kwargs) -> None:
        """Evaluate the model on the test dataset."""
        start_time = time.time()

        try:
            eval_tasks = [
                self.rollout_and_score_eval(prompt_item)
                for prompt_item in self.eval_prompts
            ]
            results = await tqdm_asyncio.gather(*eval_tasks)

            # Filter valid results
            valid_results = [
                result
                for result in results
                if result and result.get("sample") is not None
            ]

            if not valid_results:
                print("Warning: No valid evaluation results obtained")
                return

        except Exception as e:
            print(f"Error during evaluation: {e}")
            return

        # Extract scores and samples
        scores = [result["score"] for result in valid_results]
        samples = [result["sample"] for result in valid_results]

        # Calculate overall winrate (Arena-Hard style)
        overall_winrate = sum(scores) / len(scores) if scores else 0.0

        # Calculate category-specific winrates
        category_scores = {}
        for sample in samples:
            category = sample.get("category", "unknown")
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(sample["arena_score"])

        # Log metrics
        self.eval_metrics.append(("eval/overall_winrate", overall_winrate))
        self.eval_metrics.append(("eval/total_samples", len(samples)))

        for category, cat_scores in category_scores.items():
            cat_winrate = sum(cat_scores) / len(cat_scores)
            self.eval_metrics.append((f"eval/winrate_{category}", cat_winrate))

        # Judge-specific metrics
        win_count = sum(1 for s in samples if s["final_score"] > 0.5)
        tie_count = sum(1 for s in samples if abs(s["final_score"]) <= 0.5)
        loss_count = sum(1 for s in samples if s["final_score"] < -0.5)

        self.eval_metrics.append(("eval/win_count", win_count))
        self.eval_metrics.append(("eval/tie_count", tie_count))
        self.eval_metrics.append(("eval/loss_count", loss_count))
        self.eval_metrics.append(("eval/win_rate", win_count / len(samples)))
        self.eval_metrics.append(("eval/tie_rate", tie_count / len(samples)))
        self.eval_metrics.append(("eval/loss_rate", loss_count / len(samples)))

        end_time = time.time()

        # Build evaluation metrics dict
        eval_metrics = {
            "eval/overall_winrate": overall_winrate,
            "eval/total_samples": len(samples),
            "eval/win_count": win_count,
            "eval/tie_count": tie_count,
            "eval/loss_count": loss_count,
            "eval/win_rate": win_count / len(samples),
            "eval/tie_rate": tie_count / len(samples),
            "eval/loss_rate": loss_count / len(samples),
        }

        # Add category metrics
        for category, cat_scores in category_scores.items():
            cat_winrate = sum(cat_scores) / len(cat_scores)
            eval_metrics[f"eval/winrate_{category}"] = cat_winrate

        try:
            await self.evaluate_log(
                metrics=eval_metrics,
                samples=samples,
                start_time=start_time,
                end_time=end_time,
                generation_parameters={
                    "temperature": self.config.eval_temperature,
                    "max_tokens": self.config.eval_max_tokens,
                    "judge_temperature": self.config.judge_temperature,
                },
            )
        except Exception as e:
            print(f"Error logging evaluation results: {e}")

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Log metrics to wandb."""
        if wandb_metrics is None:
            wandb_metrics = {}

        # Training accuracy metrics
        if self.percent_correct_buffer:
            wandb_metrics["train/winrate"] = sum(self.percent_correct_buffer) / len(
                self.percent_correct_buffer
            )

        # Judge outcome distribution
        if self.total_judgments > 0:
            wandb_metrics.update(
                {
                    "train/win_rate": self.win_count / self.total_judgments,
                    "train/tie_rate": self.tie_count / self.total_judgments,
                    "train/loss_rate": self.loss_count / self.total_judgments,
                    "train/total_judgments": self.total_judgments,
                }
            )

        # Configuration metrics
        wandb_metrics.update(
            {
                "config/group_size": self.config.group_size,
                "config/max_token_length": self.config.max_token_length,
                "config/judge_temperature": self.config.judge_temperature,
                "config/eval_temperature": self.config.eval_temperature,
                "config/rollout_temperature": self.config.rollout_temperature,
                "config/thinking_mode": 1.0 if self.config.thinking_mode else 0.0,
            }
        )

        # Reset training metrics
        self._reset_metrics()

        # Add evaluation metrics
        for metric_name, metric_value in self.eval_metrics:
            wandb_metrics[metric_name] = metric_value
        self.eval_metrics = []

        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    ArenaHardEnv.cli()
