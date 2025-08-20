import asyncio
import math
import random
import re
import time
from typing import Dict, List, Optional, Tuple, Union

import wandb
from datasets import load_dataset
from pydantic import Field
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


class TextReversalConfig(BaseEnvConfig):
    """Configuration for TextReversalEnv with thinking mode and configurable options."""

    thinking_mode: bool = Field(
        default=False,
        description="Whether to enable thinking mode with <think></think> tags.",
    )

    custom_base_system_prompt: Optional[str] = Field(
        default=None,
        description="Custom thinking prompt. If None, uses the default thinking prompt.",
    )

    custom_thinking_prompt: Optional[str] = Field(
        default=None,
        description="Custom thinking prompt. If None, uses the default thinking prompt.",
    )

    eval_temperature: float = Field(
        default=0.6,
        description="Temperature for evaluation completions.",
    )

    rollout_temperature: float = Field(
        default=1.0,
        description="Temperature for training rollout completions.",
    )

    eval_max_tokens: int = Field(
        default=1024 * 16,
        description="Maximum tokens for evaluation completions.",
    )

    train_max_tokens: int = Field(
        default=1024 * 7,
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
        default=5,
        ge=1,
        description="Minimum response length to consider valid (filters out EOS-only responses).",
    )

    # Dataset configuration
    train_dataset: str = Field(
        default="NousResearch/TextReversalDataset",
        description="Training dataset name (HuggingFace) or path to local JSONL file.",
    )

    eval_dataset: str = Field(
        default="NousResearch/TextReversalDataset",
        description="Evaluation dataset name (HuggingFace) or path to local JSONL file.",
    )

    train_split: str = Field(
        default="train",
        description="Split to use for training dataset (only for HuggingFace datasets).",
    )

    eval_split: str = Field(
        default="test",
        description="Split to use for evaluation dataset (only for HuggingFace datasets).",
    )

    # Debug configuration
    full_debug: bool = Field(
        default=False,
        description="Enable full debug mode - logs every API request and response with truncated content.",
    )


class TextReversalEnv(BaseEnv):
    name = "text_reversal"
    env_config_cls = TextReversalConfig

    def __init__(
        self,
        config: TextReversalConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: TextReversalConfig = config
        self.percent_correct_buffer = []
        self.eval_metrics = []

        # Tracking for training metrics
        self.successful_reversals = 0
        self.failed_reversals = 0
        self.format_errors = 0  # Failed to follow format
        self.total_attempts = 0
        self.rollouts_for_wandb = []

        # Pre-compile regex patterns for performance
        self._think_pattern = re.compile(r"<think>")
        self._think_close_pattern = re.compile(r"</think>")
        self._think_content_pattern = re.compile(r"</think>\s*(.*)", re.DOTALL)
        self._thinking_extract_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        self._reversed_text_pattern = re.compile(
            r"<reversed_text>\s*(.*?)\s*</reversed_text>", re.DOTALL
        )

        # System prompts (use custom thinking prompt if provided, otherwise default)
        self.thinking_system_prompt = self._get_thinking_prompt()
        self.base_system_prompt = self.config.custom_base_system_prompt or ""

    def _get_thinking_prompt(self) -> str:
        """Get thinking system prompt."""
        return (
            self.config.custom_thinking_prompt
            if self.config.custom_thinking_prompt
            else "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the "
            "problem and deliberate with yourself via systematic reasoning processes to help come to a correct "
            "solution prior to answering. You should enclose your thoughts and internal monologue inside <think> "
            "</think> tags, and then provide your solution or response to the problem."
        )

    def _format_debug_text(self, text: str, label: str) -> str:
        """Format text for debug output (first 100 + last 100 chars)."""
        if not text:
            return f"{label}: <empty>"

        text_clean = text.strip()
        if len(text_clean) <= 200:
            return f"{label}: '{text_clean}'"

        first_100 = text_clean[:100]
        last_100 = text_clean[-100:]
        return f"{label}: '{first_100}...{last_100}' (total {len(text_clean)} chars)"

    def _log_full_debug_request(
        self,
        messages: List[Dict],
        params: Dict,
        context: str = "",
        item_id: str = "unknown",
    ):
        """Log full debug information for API requests."""
        if not self.config.full_debug:
            return

        print(f"\n🔍 FULL DEBUG - API REQUEST [{context}]")
        print(f"   Item ID: {item_id}")
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
            print(f"   Completion object: {completion}")

    def _reset_metrics(self) -> None:
        """Reset training metrics."""
        self.percent_correct_buffer = []
        self.successful_reversals = 0
        self.failed_reversals = 0
        self.format_errors = 0
        self.total_attempts = 0

    def _create_system_content(self) -> str:
        """Create system message content based on thinking mode."""
        if self.config.thinking_mode:
            return f"{self.thinking_system_prompt}\n\n{self.base_system_prompt}"
        return self.base_system_prompt

    @classmethod
    def config_init(cls) -> Tuple[TextReversalConfig, List[APIServerConfig]]:
        env_config = TextReversalConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=8,
            use_wandb=True,
            max_num_workers_per_node=8,
            rollout_server_url="http://localhost:8000",
            total_steps=2000,
            batch_size=1024,
            steps_per_eval=25,
            train_max_tokens=1024 * 7,
            eval_max_tokens=1024 * 8,
            inference_weight=1.0,
            wandb_name="text_reversal",
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.1,
            min_batch_allocation=0.1,
            thinking_mode=True,
            custom_base_system_prompt=None,
            # Debug and retry configuration
            full_debug=False,  # Set to True to enable detailed API request/response logging
            max_retries=3,
            retry_delay=1.0,
            min_response_length=5,
        )
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9004/v1",
                api_key="x",
            ),
        ]
        return env_config, server_configs

    async def setup(self) -> None:
        """Set up the environment by loading datasets."""
        # Load training dataset
        try:
            self.train = self._load_dataset(
                self.config.train_dataset, self.config.train_split
            )
            # Shuffle training dataset for reproducibility
            if hasattr(self.train, "shuffle"):
                self.train = self.train.shuffle(seed=42)
            else:
                # For list-like objects, convert to list and shuffle
                train_list = list(self.train)
                random.seed(42)
                random.shuffle(train_list)
                self.train = train_list
        except Exception as e:
            print(f"Error loading training dataset '{self.config.train_dataset}': {e}")
            # Create minimal fallback data in expected format
            self.train = [
                {"text": "Hello world"},
                {"text": "Python programming"},
                {"text": "Machine learning"},
                {"text": "Artificial intelligence"},
                {"text": "Deep learning models"},
            ] * 20  # 100 examples
            print(f"Using fallback training data with {len(self.train)} examples")

        # Load evaluation dataset
        try:
            self.test = self._load_dataset(
                self.config.eval_dataset, self.config.eval_split
            )
        except Exception as e:
            print(f"Error loading evaluation dataset '{self.config.eval_dataset}': {e}")
            raise  # Evaluation dataset must work

        # Analyze training dataset composition
        if hasattr(self.train, "__iter__"):
            total_train_items = len(self.train)
            print(f"\nTraining dataset analysis ({total_train_items} total items):")

            # Show some sample text lengths
            text_lengths = []
            for item in list(self.train)[:100]:  # Sample first 100 items
                text = item.get("text", "")
                text_lengths.append(len(text))

            if text_lengths:
                avg_length = sum(text_lengths) / len(text_lengths)
                min_length = min(text_lengths)
                max_length = max(text_lengths)
                print(
                    f"  - Sample text lengths: avg={avg_length:.1f}, min={min_length}, max={max_length}"
                )

        # Analyze evaluation dataset composition
        if hasattr(self.test, "__iter__"):
            total_eval_items = len(self.test)
            print(f"\nEvaluation dataset analysis ({total_eval_items} total items):")

            # Show some sample text lengths
            text_lengths = []
            for item in list(self.test)[:100]:  # Sample first 100 items
                text = item.get("text", "")
                text_lengths.append(len(text))

            if text_lengths:
                avg_length = sum(text_lengths) / len(text_lengths)
                min_length = min(text_lengths)
                max_length = max(text_lengths)
                print(
                    f"  - Sample text lengths: avg={avg_length:.1f}, min={min_length}, max={max_length}"
                )

        # Show configuration info
        print("\nText Reversal Configuration:")
        print(
            f"  - Training dataset: {self.config.train_dataset} (split: {self.config.train_split})"
        )
        print(
            f"  - Evaluation dataset: {self.config.eval_dataset} (split: {self.config.eval_split})"
        )
        print(f"  - Thinking mode: {self.config.thinking_mode}")
        print(f"  - Eval temperature: {self.config.eval_temperature}")

        # Show sample training item structure
        if len(self.train) > 0:
            try:
                sample_train_item = self.train[0]
                print("\nSample training item structure:")
                print(f"- Available keys: {list(sample_train_item.keys())}")
                if "text" in sample_train_item:
                    print(f"- Text: {sample_train_item['text'][:100]}...")
            except Exception as e:
                print(f"Warning: Could not display sample training item structure: {e}")

        # Show debug mode status
        if self.config.full_debug:
            print(
                "\n🔍 FULL DEBUG MODE ENABLED - Will log all API requests and responses"
            )
            print("   📊 Will show: first/last 100 chars of prompts and responses")
            print(
                f"   ⚙️  Retry settings: max_retries={self.config.max_retries}, retry_delay={self.config.retry_delay}s"
            )
            print(f"   📏 Min response length: {self.config.min_response_length} chars")
        else:
            print(
                "\n🔍 Full debug mode disabled - Use full_debug=True to enable detailed logging"
            )

        self.iter = 0

    def _load_dataset(self, dataset_path: str, split: str = None) -> List[Dict]:
        """
        Load dataset using HuggingFace load_dataset (supports both HF datasets and local files).

        Args:
            dataset_path: Either HuggingFace dataset name or path to local file
            split: Split to use

        Returns:
            List of dataset items
        """
        import os

        try:
            # Check if it's a local file
            if os.path.exists(dataset_path):
                # Local file - use appropriate loader based on extension
                if dataset_path.endswith(".jsonl") or dataset_path.endswith(".json"):
                    dataset = load_dataset(
                        "json", data_files=dataset_path, split=split or "train"
                    )
                elif dataset_path.endswith(".csv"):
                    dataset = load_dataset(
                        "csv", data_files=dataset_path, split=split or "train"
                    )
                elif dataset_path.endswith(".parquet"):
                    dataset = load_dataset(
                        "parquet", data_files=dataset_path, split=split or "train"
                    )
                else:
                    # Try JSON as default
                    dataset = load_dataset(
                        "json", data_files=dataset_path, split=split or "train"
                    )

                print(
                    f"Loaded local dataset from {dataset_path} with {len(dataset)} examples"
                )

            else:
                # HuggingFace dataset
                if split:
                    dataset = load_dataset(dataset_path, split=split)
                else:
                    dataset_dict = load_dataset(dataset_path)
                    # If no split specified, try to get the first available split
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

    def save_checkpoint(self, step: int, data: Optional[Dict] = None) -> None:
        """Save checkpoint including iteration state."""
        if data is None:
            data = {}
        data["iter"] = self.iter
        super().save_checkpoint(step, data)

    def _extract_reversed_text(self, response: str) -> Optional[str]:
        """
        Extract text from within <reversed_text> tags.

        Args:
            response: Model response text

        Returns:
            Extracted text or None if not found or multiple blocks found
        """
        if self.config.thinking_mode:
            # Check for exactly one pair of think tags
            think_open_count = len(self._think_pattern.findall(response))
            think_close_count = len(self._think_close_pattern.findall(response))

            if think_open_count != 1 or think_close_count != 1:
                return None

            # Parse only content after </think> tags
            match = self._think_content_pattern.search(response)
            if match:
                response = match.group(1)
            else:
                return None

        # Find all content between <reversed_text> tags
        matches = self._reversed_text_pattern.findall(response)

        # Must have exactly one reversed_text block
        if len(matches) != 1:
            return None

        return matches[0].strip()

    def _create_reversal_prompt(self, text: str) -> str:
        """Create the user prompt for text reversal task."""
        return (
            "Please reverse the characters of the following text and wrap your reversed text in "
            "<reversed_text> </reversed_text> tags.\n\n"
            f"The text to reverse:\n{text}"
        )

    async def get_next_item(self) -> Item:
        """Generate next training item with text reversal task."""
        self.iter += 1

        # Get next training example sequentially
        example = self.train[self.iter % len(self.train)]

        # Extract text from training data
        original_text = example.get("text", "")
        if not original_text:
            # Fallback if text field is missing
            original_text = "Hello"

        # Create the expected reversed text (ground truth)
        expected_reversed = original_text[::-1]

        # Create system message
        system_content = self._create_system_content()

        # Create user prompt
        user_content = self._create_reversal_prompt(original_text)

        prompt = tuple(
            [
                frozenset({"role": "system", "content": system_content}.items()),
                frozenset({"role": "user", "content": user_content}.items()),
            ]
        )

        return (prompt, expected_reversed)

    def prepare_eval_item(self, item: dict) -> Tuple[Optional[Tuple], Optional[str]]:
        """
        Prepare an evaluation item from the text reversal dataset.
        """
        try:
            original_text = item.get("text", "")

            # Validate required fields
            if not original_text:
                return None, None

            # Create the expected reversed text (ground truth)
            expected_reversed = original_text[::-1]

            # Create system message
            system_content = self._create_system_content()

            # Create user prompt
            user_content = self._create_reversal_prompt(original_text)

            prompt = tuple(
                [
                    frozenset({"role": "system", "content": system_content}.items()),
                    frozenset({"role": "user", "content": user_content}.items()),
                ]
            )

            return prompt, expected_reversed

        except Exception as e:
            print(f"Error preparing evaluation item: {e}")
            print(f"DEBUG: Exception type: {type(e)}")
            print(f"DEBUG: item keys: {list(item.keys()) if item else 'item is None'}")
            return None, None

    async def collect_trajectories(self, item: Item) -> Tuple[ScoredDataGroup, List]:
        """Collect and score model trajectories."""
        messages = self._convert_messages_to_list(item[0])
        completion_params = self._get_train_completion_params()

        # Retry logic for training trajectories
        max_retries = self.config.max_retries
        retry_delay = self.config.retry_delay

        # Get item info for debug logging
        item_id = f"train_{self.iter if hasattr(self, 'iter') else 'unknown'}"

        for attempt in range(max_retries):
            try:
                # Log full debug request
                self._log_full_debug_request(
                    messages,
                    completion_params,
                    f"TRAINING attempt {attempt + 1}/{max_retries}",
                    item_id,
                )

                completions = await self.server.chat_completion(
                    messages=messages, **completion_params
                )

                # Log full debug response
                self._log_full_debug_response(
                    completions, f"TRAINING attempt {attempt + 1}/{max_retries}"
                )

                # Check if we got valid completions
                if not completions.choices:
                    if attempt < max_retries - 1:
                        print(
                            f"DEBUG: No choices in collect_trajectories (attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        print(
                            f"DEBUG: No choices in collect_trajectories after {max_retries} attempts"
                        )
                        return None, []

                # Check if any completion has None content
                valid_completions = []
                for completion_choice in completions.choices:
                    if (
                        completion_choice.message.content is not None
                        and isinstance(completion_choice.message.content, str)
                        and len(completion_choice.message.content.strip())
                        >= self.config.min_response_length
                    ):
                        valid_completions.append(completion_choice)

                # If we don't have enough valid completions, retry
                if (
                    len(valid_completions) < len(completions.choices) // 2
                ):  # If less than half are valid
                    if attempt < max_retries - 1:
                        print(
                            f"DEBUG: Only {len(valid_completions)}/{len(completions.choices)} valid completions (attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        print(
                            f"DEBUG: Only {len(valid_completions)}/{len(completions.choices)} valid completions after {max_retries} attempts"
                        )
                        # Continue with what we have

                # Build trajectories using valid completions
                to_score = []
                for completion_choice in valid_completions:
                    # Add assistant response to existing messages
                    trajectory_messages = messages + [
                        {
                            "role": "assistant",
                            "content": completion_choice.message.content,
                        }
                    ]
                    to_score.append((tuple(trajectory_messages), item[1]))

                # Success - we got at least some valid trajectories
                break

            except Exception as e:
                if attempt < max_retries - 1:
                    print(
                        f"DEBUG: collect_trajectories API call failed (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    print(
                        f"DEBUG: collect_trajectories API call failed after {max_retries} attempts: {e}"
                    )
                    return None, []

        scored_data = await self.score(to_score)

        # Add rollouts for wandb visualization
        if scored_data is not None:
            await self.add_rollouts_for_wandb(scored_data, item)

        return scored_data, []

    def _convert_messages_to_list(self, prompt_tuple: Tuple) -> List[Dict]:
        """Convert frozenset message format to list format."""
        messages = []
        for role_dict in prompt_tuple:
            messages.append(dict(role_dict))
        return messages

    def _get_train_completion_params(self) -> Dict:
        """Get completion parameters for training rollouts."""
        return {
            "n": self.config.group_size,
            "max_tokens": self.config.train_max_tokens,
            "temperature": self.config.rollout_temperature,
        }

    def _get_eval_completion_params(self) -> Dict:
        """Get completion parameters for evaluation."""
        return {
            "n": 1,
            "max_tokens": self.config.eval_max_tokens,
            "temperature": self.config.eval_temperature,
            "split": "eval",
        }

    async def score(self, rollout_group_data: List[Tuple]) -> Optional[ScoredDataGroup]:
        """Score a group of rollout data."""
        if not rollout_group_data:
            return None

        try:
            scores = ScoredDataGroup()
            scores["tokens"] = []
            scores["masks"] = []
            scores["scores"] = []

            random.shuffle(rollout_group_data)

            for item in rollout_group_data:
                # Simplified validation
                if not item or len(item) < 2 or not item[0]:
                    continue

                model_response = item[0][-1]["content"]
                expected_reversed = item[1]

                # Extract reversed text from model response
                extracted_reversed = self._extract_reversed_text(model_response)

                # Score 1.0 if exact match, 0.0 otherwise
                reward = 1.0 if extracted_reversed == expected_reversed else 0.0

                # Track metrics
                self.total_attempts += 1
                if reward == 1.0:
                    self.successful_reversals += 1
                else:
                    self.failed_reversals += 1
                    if extracted_reversed is None:
                        self.format_errors += 1

                # Tokenize the conversation for learning
                out_dict = tokenize_for_trainer(self.tokenizer, item[0])
                tokens = out_dict["tokens"]
                masks = out_dict["masks"]

                # Skip obviously bad examples
                if len([1 for mask in masks if mask != -100]) < 10:
                    continue

                scores["tokens"].append(tokens)
                scores["masks"].append(masks)
                scores["scores"].append(reward)

                if len(scores["tokens"]) >= self.config.group_size:
                    break

            if not scores["tokens"]:
                return None

            # Group-level logging after scoring is complete
            group_successes = sum(1 for score in scores["scores"] if score == 1.0)
            group_size = len(scores["scores"])
            any_success = group_successes > 0
            success_indicator = "✅" if any_success else "❌"

            # Calculate running totals
            total_success_rate = (
                (self.successful_reversals / self.total_attempts * 100)
                if self.total_attempts > 0
                else 0.0
            )

            print(
                f"{success_indicator} Group scored: {group_successes}/{group_size} successful reversals | "
                f"Total success rate: {self.successful_reversals}/{self.total_attempts} ({total_success_rate:.1f}%)"
            )

            # Update percent correct buffer
            for score in scores["scores"]:
                self.percent_correct_buffer.append(max(score, 0))

            # Return None if all scores are the same (no learning signal)
            if len(set(scores["scores"])) == 1:
                return None

            return scores

        except Exception as e:
            print(f"Error in score method: {e}")
            print(f"DEBUG: Exception type: {type(e)}")
            print(
                f"DEBUG: rollout_group_data length: {len(rollout_group_data) if rollout_group_data else 'None'}"
            )
            if rollout_group_data:
                print(f"DEBUG: first item type: {type(rollout_group_data[0])}")
                print(
                    f"DEBUG: first item length: {len(rollout_group_data[0]) if rollout_group_data[0] else 'None'}"
                )
            return None

    async def rollout_and_score_eval(self, test_item: dict) -> dict:
        """Rollout and score evaluation."""
        try:
            prompt, expected_reversed = self.prepare_eval_item(test_item)
            if prompt is None:
                return {"score": 0.0, "sample": None}

            messages = self._convert_messages_to_list(prompt)
            completion_params = self._get_eval_completion_params()

            # Retry logic for failed API calls
            max_retries = self.config.max_retries
            retry_delay = self.config.retry_delay

            # Get item info for debug logging
            item_id = test_item.get("id", f"eval_{hash(test_item.get('text', ''))}")

            for attempt in range(max_retries):
                try:
                    # Log full debug request
                    self._log_full_debug_request(
                        messages,
                        completion_params,
                        f"EVAL attempt {attempt + 1}/{max_retries}",
                        item_id,
                    )

                    completion = await self.server.chat_completion(
                        messages=messages, **completion_params
                    )

                    # Log full debug response
                    self._log_full_debug_response(
                        completion, f"EVAL attempt {attempt + 1}/{max_retries}"
                    )

                    if not completion.choices:
                        if attempt < max_retries - 1:
                            print(
                                f"DEBUG: No choices in completion (attempt {attempt + 1}/{max_retries})"
                            )
                            await asyncio.sleep(retry_delay)
                            continue
                        else:
                            print(f"DEBUG: No choices after {max_retries} attempts")
                            return {"score": 0.0, "sample": None}

                    model_response = completion.choices[0].message.content

                    # Check for None content or very short responses
                    if model_response is None:
                        if attempt < max_retries - 1:
                            print(
                                f"DEBUG: model_response is None (attempt {attempt + 1}/{max_retries})"
                            )
                            await asyncio.sleep(retry_delay)
                            continue
                        else:
                            print(
                                f"DEBUG: model_response is None after {max_retries} attempts"
                            )
                            return {"score": 0.0, "sample": None}

                    if not isinstance(model_response, str):
                        if attempt < max_retries - 1:
                            print(
                                f"DEBUG: model_response is not a string. Type: {type(model_response)} (attempt {attempt + 1}/{max_retries})"
                            )
                            await asyncio.sleep(retry_delay)
                            continue
                        else:
                            print(
                                f"DEBUG: model_response is not a string after {max_retries} attempts. Type: {type(model_response)}"
                            )
                            return {"score": 0.0, "sample": None}

                    # Check for very short responses (likely just EOS token)
                    if len(model_response.strip()) < self.config.min_response_length:
                        if attempt < max_retries - 1:
                            print(
                                f"DEBUG: Very short response (likely EOS token only): '{model_response}' (attempt {attempt + 1}/{max_retries})"
                            )
                            await asyncio.sleep(retry_delay)
                            continue
                        else:
                            print(
                                f"DEBUG: Very short response after {max_retries} attempts: '{model_response}'"
                            )
                            return {"score": 0.0, "sample": None}

                    # Success - we got a valid response
                    break

                except Exception as e:
                    if attempt < max_retries - 1:
                        print(
                            f"DEBUG: API call failed (attempt {attempt + 1}/{max_retries}): {e}"
                        )
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        print(
                            f"DEBUG: API call failed after {max_retries} attempts: {e}"
                        )
                        raise

            # Extract reversed text from model response
            extracted_reversed = self._extract_reversed_text(model_response)

            # Score 1.0 if exact match, 0.0 otherwise
            score = 1.0 if extracted_reversed == expected_reversed else 0.0

            # Get original text from the user message
            original_text = test_item.get("text", "")

            # Add full conversation including model response
            full_messages = messages + [
                {"role": "assistant", "content": model_response}
            ]

            sample = {
                "messages": full_messages,
                "original_text": original_text,
                "expected_reversed": expected_reversed,
                "extracted_reversed": extracted_reversed,
                "score": int(score),
                "correct": bool(score),
                "finish_reason": completion.choices[0].finish_reason,
                "thinking_mode": self.config.thinking_mode,
                "format_compliant": extracted_reversed is not None,
                "dataset_item_id": item_id,
            }

            # Add thinking-specific parsing info
            if self.config.thinking_mode:
                if "</think>" in model_response:
                    sample["response_after_think"] = model_response.split("</think>")[
                        -1
                    ].strip()
                    sample["thinking_content"] = self._thinking_extract_pattern.search(
                        model_response
                    )
                    if sample["thinking_content"]:
                        sample["thinking_content"] = (
                            sample["thinking_content"].group(1).strip()
                        )
                else:
                    sample["response_after_think"] = model_response
                    sample["thinking_content"] = None

            return {"score": score, "sample": sample}

        except Exception as e:
            print(f"Error in evaluation: {e}")
            print(f"DEBUG: Exception type: {type(e)}")
            print(
                f"DEBUG: test_item keys: {list(test_item.keys()) if test_item else 'test_item is None'}"
            )
            return {"score": 0.0, "sample": None}

    async def evaluate(self, *args, **kwargs) -> None:
        """Evaluate the model on the test dataset."""
        start_time = time.time()

        try:
            eval_tasks = [
                self.rollout_and_score_eval(test_item) for test_item in self.test
            ]
            results = await tqdm_asyncio.gather(*eval_tasks)

            # Filter valid results
            valid_results = [
                result
                for result in results
                if not isinstance(result, Exception)
                and result
                and result.get("sample") is not None
            ]

            if not valid_results:
                print("Warning: No valid evaluation results obtained")
                return

        except Exception as e:
            print(f"Error during evaluation: {e}")
            return

        # Extract scores and samples from valid results
        scores = [result["score"] for result in valid_results]
        samples = [result["sample"] for result in valid_results]
        valid_scores = [s for s in scores if s is not None]

        if not valid_scores:
            print("Warning: No valid scores found during evaluation")
            return

        percent_correct = sum(valid_scores) / len(valid_scores)
        self.eval_metrics.append(("eval/percent_correct", percent_correct))

        # Calculate additional metrics
        format_compliant = sum(
            1 for sample in samples if sample.get("format_compliant", False)
        )

        thinking_mode_used = self.config.thinking_mode

        # Get response metrics
        response_lengths = []
        thinking_utilization = 0

        for sample in samples:
            if not sample:
                continue

            # Track response length
            messages = sample.get("messages", [])
            if messages:
                assistant_msg = messages[-1].get("content", "")
                response_lengths.append(len(assistant_msg))

            # Track thinking utilization in thinking mode
            if thinking_mode_used:
                thinking_content = sample.get("thinking_content")
                if thinking_content:
                    thinking_utilization += 1

        # Response length metrics
        if response_lengths:
            avg_response_length = sum(response_lengths) / len(response_lengths)
            response_length_std = (
                sum((x - avg_response_length) ** 2 for x in response_lengths)
                / len(response_lengths)
            ) ** 0.5
            self.eval_metrics.append(("eval/avg_response_length", avg_response_length))
            self.eval_metrics.append(("eval/response_length_std", response_length_std))

        # Thinking utilization rate
        if thinking_mode_used and samples:
            thinking_utilization_rate = thinking_utilization / len(samples)
            self.eval_metrics.append(
                ("eval/thinking_utilization_rate", thinking_utilization_rate)
            )

        # Add overall dataset statistics
        total_dataset_items = len(self.test) if hasattr(self, "test") else 0
        evaluated_items = len(samples)
        self.eval_metrics.append(("eval/total_dataset_items", total_dataset_items))
        self.eval_metrics.append(("eval/evaluated_items", evaluated_items))
        self.eval_metrics.append(("eval/valid_scores", len(valid_scores)))
        self.eval_metrics.append(
            (
                "eval/format_compliance_rate",
                format_compliant / len(samples) if samples else 0.0,
            )
        )

        end_time = time.time()

        # Build evaluation metrics dict
        eval_metrics = {
            "eval/percent_correct": percent_correct,
            "eval/total_samples": len(samples),
            "eval/correct_samples": sum(valid_scores),
            "eval/format_compliance_rate": (
                format_compliant / len(samples) if samples else 0.0
            ),
        }

        # Add response length metrics
        if response_lengths:
            eval_metrics["eval/avg_response_length"] = avg_response_length
            eval_metrics["eval/response_length_std"] = response_length_std

        # Add thinking utilization
        if thinking_mode_used and samples:
            eval_metrics["eval/thinking_utilization_rate"] = thinking_utilization_rate

        try:
            await self.evaluate_log(
                metrics=eval_metrics,
                samples=samples,
                start_time=start_time,
                end_time=end_time,
                generation_parameters={
                    "temperature": self.config.eval_temperature,
                    "max_tokens": self.config.eval_max_tokens,
                    "thinking_mode": thinking_mode_used,
                },
            )
        except Exception as e:
            print(f"Error logging evaluation results: {e}")

    async def add_rollouts_for_wandb(
        self,
        scored_data: Union[ScoredDataGroup, List[ScoredDataGroup]],
        item: Item = None,
    ) -> None:
        """Add rollouts to wandb for visualization."""
        if item is None or scored_data is None or not scored_data.get("tokens"):
            return

        # Extract ground truth
        expected_reversed = item[1]

        # Extract original text from the item prompt
        original_text = "unknown_text"
        try:
            # The item[0] contains the prompt tuple with system and user messages
            for role_dict in item[0]:
                role_dict_converted = dict(role_dict)
                if role_dict_converted.get("role") == "user":
                    user_content = role_dict_converted.get("content", "")
                    # Extract original text from the user message
                    lines = user_content.split("\n")
                    for line in lines:
                        if (
                            line.strip()
                            and not line.startswith("Please reverse")
                            and not line.startswith("The text to reverse:")
                        ):
                            original_text = line.strip()
                            break
                    break
        except Exception as e:
            print(f"DEBUG: Exception in add_rollouts_for_wandb text extraction: {e}")
            original_text = "extraction_failed"

        # Keep a reasonable number of rollouts
        num_keep = self.config.num_rollouts_per_group_for_logging
        if num_keep == -1:
            num_keep = self.config.group_size

        num_keep = min(num_keep, len(scored_data["tokens"]))

        current_rollouts = []
        mode = "thinking" if self.config.thinking_mode else "direct"

        for i in range(num_keep):
            # Decode the full trajectory
            full_text = self.tokenizer.decode(
                scored_data["tokens"][i], skip_special_tokens=True
            )
            score_val = scored_data["scores"][i]

            # Extract the model's reversed text
            extracted_reversed = "unknown"
            try:
                # Try to get model response from messages or decode from tokens
                messages = scored_data.get("messages", [])
                if i < len(messages) and isinstance(messages[i], list) and messages[i]:
                    model_response = messages[i][-1].get("content", "")
                else:
                    # Fallback to decoding tokens
                    model_response = full_text

                extracted_reversed = (
                    self._extract_reversed_text(model_response) or "format_error"
                )
            except Exception as e:
                print(
                    f"DEBUG: Exception in add_rollouts_for_wandb reversal parsing: {e}"
                )
                extracted_reversed = "parse_error"

            current_rollouts.append(
                (
                    full_text,
                    score_val,
                    expected_reversed,
                    extracted_reversed,
                    original_text,
                    mode,
                )
            )

        self.rollouts_for_wandb.append(current_rollouts)

        # Keep only recent rollouts
        if len(self.rollouts_for_wandb) > self.config.num_rollouts_to_keep:
            self.rollouts_for_wandb.pop(0)

    async def create_rollout_table(self, wandb_metrics: Dict) -> Dict:
        """Create wandb table for rollout visualization."""
        if not self.rollouts_for_wandb:
            return wandb_metrics

        table = wandb.Table(
            columns=[
                "full_text",
                "score",
                "expected_reversed",
                "extracted_reversed",
                "original_text",
                "mode",
            ]
        )

        for group_rollouts in self.rollouts_for_wandb:
            for rollout_tuple in group_rollouts:
                if len(rollout_tuple) == 6:
                    table.add_data(*rollout_tuple)

        wandb_metrics["train/rollouts"] = table
        self.rollouts_for_wandb = []
        return wandb_metrics

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Log metrics to wandb."""
        if wandb_metrics is None:
            wandb_metrics = {}

        # Basic accuracy metrics
        if self.percent_correct_buffer:
            wandb_metrics["train/percent_correct"] = sum(
                self.percent_correct_buffer
            ) / len(self.percent_correct_buffer)

        # Reversal-specific metrics
        if self.total_attempts > 0:
            wandb_metrics["train/success_rate"] = (
                self.successful_reversals / self.total_attempts
            )
            wandb_metrics["train/failure_rate"] = (
                self.failed_reversals / self.total_attempts
            )
            wandb_metrics["train/format_error_rate"] = (
                self.format_errors / self.total_attempts
            )
            wandb_metrics["train/format_compliance_rate"] = 1.0 - (
                self.format_errors / self.total_attempts
            )

        # Configuration metrics
        wandb_metrics.update(
            {
                "train/thinking_mode_enabled": (
                    1.0 if self.config.thinking_mode else 0.0
                ),
                "train/total_attempts": self.total_attempts,
                "train/successful_reversals": self.successful_reversals,
                "train/failed_reversals": self.failed_reversals,
                "train/format_errors": self.format_errors,
                "config/group_size": self.config.group_size,
                "config/train_max_tokens": self.config.train_max_tokens,
                "config/eval_max_tokens": self.config.eval_max_tokens,
            }
        )

        # Reset training metrics
        self._reset_metrics()

        # Add evaluation metrics
        for metric_name, metric_value in self.eval_metrics:
            wandb_metrics[metric_name] = metric_value
        self.eval_metrics = []

        # Add rollout table
        wandb_metrics = await self.create_rollout_table(wandb_metrics)

        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    TextReversalEnv.cli()
