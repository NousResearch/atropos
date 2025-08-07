import json
import os
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


class ToolCallingConfig(BaseEnvConfig):
    """Configuration for SingleToolCallingEnv with thinking mode and configurable options."""

    thinking_mode: bool = Field(
        default=True,
        description="Whether to enable thinking mode with <think></think> tags.",
    )

    custom_thinking_prompt: Optional[str] = Field(
        default=None,
        description="Custom thinking prompt. If None, uses the default thinking prompt.",
    )

    custom_system_prompt: Optional[str] = Field(
        default=None,
        description="Custom system prompt. In non-thinking mode, used directly. In thinking mode, appended to thinking prompt.",
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
        default=1024 * 32,
        description="Maximum tokens for evaluation completions.",
    )

    train_max_tokens: int = Field(
        default=1024 * 16,
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
        default="NousResearch/XLAM-Atropos",
        description="Training dataset name (HuggingFace) or path to local JSONL file.",
    )

    eval_dataset: str = Field(
        default="NousResearch/XLAM-Atropos",
        description="Evaluation dataset name (HuggingFace) or path to local JSONL file.",
    )

    train_split: str = Field(
        default="train",
        description="Split to use for training dataset (only for HuggingFace datasets).",
    )

    eval_split: str = Field(
        default="train",
        description="Split to use for evaluation dataset (only for HuggingFace datasets).",
    )

    eval_test_size: int = Field(
        default=100,
        ge=1,
        description="Number of samples to use for evaluation test set.",
    )

    # Debug configuration
    full_debug: bool = Field(
        default=False,
        description="Enable full debug logging including request/response details.",
    )


class SingleToolCallingEnv(BaseEnv):
    name = "tool_calling"
    env_config_cls = ToolCallingConfig

    def __init__(
        self,
        config: ToolCallingConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.percent_correct_buffer = list()
        self.eval_metrics = list()
        # Add tracking for wandb visualizations
        self.rollouts_for_wandb = []
        self.completion_lengths = []

        # Thinking mode pattern compilation for efficiency
        if self.config.thinking_mode:
            self._think_pattern = re.compile(r"<think>")
            self._think_close_pattern = re.compile(r"</think>")
            self._think_content_pattern = re.compile(
                r"</think>\s*(.*)", re.DOTALL
            )

        # Metrics tracking
        self._reset_metrics()

    def _reset_metrics(self) -> None:
        """Reset metrics tracking variables."""
        self.total_evaluations = 0
        self.successful_evaluations = 0
        self.format_errors = 0
        self.thinking_errors = 0

    def _get_thinking_prompt(self) -> str:
        """Get the thinking prompt based on configuration."""
        if self.config.custom_thinking_prompt:
            return self.config.custom_thinking_prompt
        return (
            "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the "
            "problem and deliberate with yourself via systematic reasoning processes to help come to a correct "
            "solution prior to answering. You should enclose your thoughts and internal monologue inside <think> "
            "</think> tags, and then provide your solution or response to the problem."
        )

    def _get_system_prompt(self) -> str:
        """Get the system prompt based on configuration and thinking mode."""
        if not self.config.thinking_mode:
            # Non-thinking mode: use custom system prompt if provided, otherwise empty
            return self.config.custom_system_prompt or ""
        else:
            # Thinking mode: start with thinking prompt, append custom system prompt if provided
            base_prompt = self._get_thinking_prompt()
            if self.config.custom_system_prompt:
                return f"{base_prompt}\n\n{self.config.custom_system_prompt}"
            return base_prompt

    def _format_debug_text(self, text: str, label: str) -> str:
        """Format text for debug output with proper truncation."""
        if not text:
            return f"{label}: [EMPTY]"
        
        truncated = text[:500] + "..." if len(text) > 500 else text
        return f"{label}:\n{truncated}"

    def _log_full_debug_request(
        self, messages: List[Dict], params: Dict, context: str = ""
    ):
        """Log full request details if full debug is enabled."""
        if not self.config.full_debug:
            return
        
        print(f"\n{'='*50} DEBUG REQUEST {context} {'='*50}")
        print(f"Messages: {json.dumps(messages, indent=2)}")
        print(f"Parameters: {json.dumps(params, indent=2)}")
        print("="*120)

    def _log_full_debug_response(self, completion, context: str = ""):
        """Log full response details if full debug is enabled."""
        if not self.config.full_debug:
            return
        
        print(f"\n{'='*50} DEBUG RESPONSE {context} {'='*50}")
        if hasattr(completion, 'choices') and completion.choices:
            for i, choice in enumerate(completion.choices):
                response_text = choice.text if hasattr(choice, 'text') else str(choice)
                print(f"Choice {i}: {self._format_debug_text(response_text, 'Response')}")
        else:
            print(f"Completion: {completion}")
        print("="*120)

    @classmethod
    def config_init(cls) -> Tuple[ToolCallingConfig, List[APIServerConfig]]:
        env_config = ToolCallingConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=16,
            use_wandb=True,
            max_num_workers_per_node=16,
            rollout_server_url="http://localhost:8000",
            total_steps=2000,
            batch_size=1024,
            steps_per_eval=25,
            max_token_length=1024 * 16,
            inference_weight=1.0,
            wandb_name="toolcall_think",
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.1,
            min_batch_allocation=0.1,
            thinking_mode=False,
        )
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9004/v1",
                api_key="x",
                num_max_requests_at_once=32,
                num_requests_for_eval=256,
            ),
        ]

        return env_config, server_configs

    async def create_rollout_table(self, wandb_metrics):
        if len(self.rollouts_for_wandb) > 0:
            table = wandb.Table(columns=["text", "score", "expected_tool_call"])
            for group in self.rollouts_for_wandb:
                for item in group:
                    table.add_data(item[0], item[1], item[2])
            wandb_metrics["train/rollouts"] = table

        self.rollouts_for_wandb = []
        return wandb_metrics

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """
        Log to wandb with comprehensive metrics.
        """
        if wandb_metrics is None:
            wandb_metrics = dict()

        # Try to calculate percent_correct, skip if there's a division by zero
        try:
            wandb_metrics["train/percent_correct"] = sum(
                self.percent_correct_buffer
            ) / len(self.percent_correct_buffer)
        except ZeroDivisionError:
            # Skip if buffer is empty
            pass

        # Add evaluation metrics
        self.percent_correct_buffer = list()
        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = list()

        # Add error tracking metrics
        if self.total_evaluations > 0:
            wandb_metrics["eval/format_error_rate"] = self.format_errors / self.total_evaluations
            wandb_metrics["eval/thinking_error_rate"] = self.thinking_errors / self.total_evaluations
            wandb_metrics["eval/success_rate"] = self.successful_evaluations / self.total_evaluations

        await super().wandb_log(wandb_metrics)

    def _load_dataset(self, dataset_path: str, split: str = None) -> List[Dict]:
        """Load dataset from HuggingFace or local JSONL file."""
        try:
            if dataset_path.endswith('.jsonl'):
                # Load from local JSONL file
                data = []
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data.append(json.loads(line.strip()))
                return data
            else:
                # Load from HuggingFace
                dataset = load_dataset(dataset_path, split=split or "train")
                return list(dataset)
        except Exception as e:
            print(f"Error loading dataset {dataset_path}: {e}")
            return []

    async def setup(self):
        # Load the full dataset
        full_dataset = load_dataset(
            self.config.train_dataset,
            "default",
            split=self.config.train_split,
        )

        full_dataset = full_dataset.shuffle(seed=42)

        # Create train/test split on the fly
        split_dataset = full_dataset.train_test_split(test_size=self.config.eval_test_size, seed=42)

        # Keep the splits as is - no need to reformat
        self.train = split_dataset["train"]
        self.test = split_dataset["test"]

        self.iter = 0

    def save_checkpoint(self, step: int, data: Optional[Dict] = None) -> None:
        """Save checkpoint including iteration state."""
        if data is None:
            data = {}
        data["iter"] = self.iter
        super().save_checkpoint(step, data)

    def _validate_and_extract_thinking(self, response: str) -> Tuple[bool, str]:
        """
        Validate thinking tags and extract content after </think>.
        
        Returns:
            Tuple of (is_valid, processed_content)
        """
        if not self.config.thinking_mode:
            return True, response
            
        # Check for exactly one pair of think tags
        think_open_count = len(self._think_pattern.findall(response))
        think_close_count = len(self._think_close_pattern.findall(response))
        
        if think_open_count != 1 or think_close_count != 1:
            return False, response
            
        # Extract content after </think>
        match = self._think_content_pattern.search(response)
        if match:
            return True, match.group(1).strip()
        else:
            return False, response

    async def rollout_and_score_eval(self, test_item):
        # Extract conversations from test item
        conversations = test_item["conversations"]

        # Find system message and human message
        system_message = next(
            (msg for msg in conversations if msg["from"] == "system"), None
        )
        human_message = next(
            (msg for msg in conversations if msg["from"] == "human"), None
        )
        expected_gpt_message = next(
            (msg for msg in conversations if msg["from"] == "gpt"), None
        )

        if not human_message or not expected_gpt_message:
            return {"score": 0, "sample": None}  # Skip invalid conversations

        # Create messages for model
        messages = []
        system_prompt = self._get_system_prompt()
        
        if system_message:
            combined_system_content = system_prompt + "\n\n" + system_message["value"]
            messages.append(
                {
                    "role": "system",
                    "content": combined_system_content,
                }
            )
        elif system_prompt:
            messages.append(
                {
                    "role": "system", 
                    "content": system_prompt,
                }
            )
        
        messages.append({"role": "user", "content": human_message["value"]})

        # Apply chat template to convert messages to a single string
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        # Log debug information
        if self.config.full_debug:
            self._log_full_debug_request(
                messages, 
                {"temperature": self.config.eval_temperature, "max_tokens": self.config.eval_max_tokens},
                "EVAL"
            )

        # Get model completion using completion() instead of chat_completion()
        completion = await self.server.completion(
            prompt=prompt,
            n=1,
            max_tokens=self.config.eval_max_tokens,
            temperature=self.config.eval_temperature,
            split="eval",
        )

        if self.config.full_debug:
            self._log_full_debug_response(completion, "EVAL")

        # Extract the model's response from the completion
        model_response = completion.choices[0].text
        expected_response = expected_gpt_message["value"]

        # Track evaluation metrics
        self.total_evaluations += 1

        # Validate thinking tags if in thinking mode
        thinking_valid, processed_response = self._validate_and_extract_thinking(model_response)
        if not thinking_valid:
            self.thinking_errors += 1
            return {
                "score": 0,
                "sample": {
                    "prompt": human_message["value"],
                    "model_response": model_response,
                    "expected_response": expected_response,
                    "score": 0,
                    "error_type": "thinking_format_error",
                    "thinking_mode": self.config.thinking_mode,
                }
            }

        # Extract and compare tool calls
        score = self._compare_tool_calls(processed_response, expected_response)
        
        if score > 0:
            self.successful_evaluations += 1
        
        # Create sample for logging
        sample = {
            "prompt": human_message["value"],
            "model_response": model_response,
            "processed_response": processed_response if self.config.thinking_mode else model_response,
            "expected_response": expected_response,
            "score": score,
            "thinking_mode": self.config.thinking_mode,
            "thinking_valid": thinking_valid,
        }

        return {"score": score, "sample": sample}

    def _extract_tool_call_jsons(self, text):
        """
        Extract multiple JSONs from within <tool_call> tags

        Args:
            text: Text containing tool calls

        Returns:
            List of parsed JSON objects or empty list if extraction/parsing fails
        """
        # Find all content between <tool_call> tags
        matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
        tool_calls = []

        for match in matches:
            try:
                # Parse the JSON content
                json_str = match
                tool_call = json.loads(json_str)
                tool_calls.append(tool_call)
            except json.JSONDecodeError:
                # Skip invalid JSON but continue processing other matches
                continue

        return tool_calls

    def _compare_tool_calls(self, model_response, expected_response):
        """
        Compare multiple tool calls by extracting JSONs from <tool_call> tags and comparing content

        Returns:
            1 if all tool calls match (all required calls are present with correct values), 0 otherwise
        """
        # Extract JSONs from tool calls
        model_jsons = self._extract_tool_call_jsons(model_response)
        expected_jsons = self._extract_tool_call_jsons(expected_response)

        # If we couldn't extract any JSONs or the count doesn't match, return 0
        if not model_jsons or not expected_jsons:
            return 0

        # Copy the expected_jsons to avoid modifying the original
        remaining_expected_jsons = expected_jsons.copy()

        # For each model JSON, try to find a matching expected JSON
        for model_json in model_jsons:
            found_match = False

            for i, expected_json in enumerate(remaining_expected_jsons):
                if self._json_objects_match(model_json, expected_json):
                    # Remove the matched expected JSON
                    remaining_expected_jsons.pop(i)
                    found_match = True
                    break

            # If no match was found for this model JSON, return 0
            if not found_match:
                return 0

        # If we've matched all expected JSONs (none remaining), return 1
        return 1 if not remaining_expected_jsons else 0

    def _json_objects_match(self, json1, json2):
        """
        Check if two JSON objects match, with all fields in json2 existing in json1
        with the same values.

        Args:
            json1: First JSON object
            json2: Second JSON object (expected values)

        Returns:
            True if objects match, False otherwise
        """
        try:
            # Check if all expected fields are in model response
            for key in json2:
                if key not in json1:
                    return False

                # For nested dictionaries (like 'arguments'), check all values
                if isinstance(json2[key], dict) and isinstance(json1[key], dict):
                    for arg_key in json2[key]:
                        if arg_key not in json1[key]:
                            return False
                        if json2[key][arg_key] != json1[key][arg_key]:
                            return False
                # For non-dictionary fields, check direct equality
                elif json2[key] != json1[key]:
                    return False

            # All checks passed
            return True
        except Exception:
            # Any error in comparison counts as failure
            return False

    async def evaluate(self, *args, **kwargs):
        """Evaluate the model on the test dataset."""
        start_time = time.time()
        
        try:
            eval_tasks = []
            for test_item in self.test:
                eval_tasks.append(self.rollout_and_score_eval(test_item))
            results = await tqdm_asyncio.gather(*eval_tasks)
            
            # Filter valid results
            valid_results = [
                result for result in results 
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

        # Calculate metrics
        overall_accuracy = sum(scores) / len(scores) if scores else 0.0
        
        # Count error types
        thinking_errors = sum(1 for s in samples if s.get("error_type") == "thinking_format_error")
        successful_calls = sum(1 for s in samples if s["score"] > 0)
        
        # Log metrics
        self.eval_metrics.append(("eval/percent_correct", overall_accuracy))
        self.eval_metrics.append(("eval/total_samples", len(samples)))
        self.eval_metrics.append(("eval/successful_tool_calls", successful_calls))
        self.eval_metrics.append(("eval/thinking_format_errors", thinking_errors))
        
        if self.config.thinking_mode:
            thinking_valid_rate = sum(1 for s in samples if s.get("thinking_valid", True)) / len(samples)
            self.eval_metrics.append(("eval/thinking_valid_rate", thinking_valid_rate))

        end_time = time.time()

        # Build evaluation metrics dict
        eval_metrics = {
            "eval/percent_correct": overall_accuracy,
            "eval/total_samples": len(samples),
            "eval/successful_tool_calls": successful_calls,
            "eval/thinking_format_errors": thinking_errors,
        }
        
        if self.config.thinking_mode:
            eval_metrics["eval/thinking_valid_rate"] = thinking_valid_rate

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

    async def collect_trajectories(self, item) -> Tuple[ScoredDataGroup, List]:
        # Extract messages from the item
        messages = []
        for role_dict in item[0]:
            messages.append(dict(role_dict))

        # Apply chat template to convert messages to a single string
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        # Get completions from the model using completion() instead of chat_completion()
        completions = await self.server.completion(
            prompt=prompt,
            n=self.config.group_size,
            max_tokens=self.config.train_max_tokens,
            temperature=self.config.rollout_temperature,
        )
        to_score = list()

        for i, completion_choice in enumerate(completions.choices):
            # Create a copy of the prompt messages
            trajectory_messages = []
            for role_dict in item[0]:
                trajectory_messages.append(dict(role_dict))

            # Add the model's response
            trajectory_messages.append(
                {"role": "assistant", "content": completion_choice.text}
            )

            # Add to scoring queue with expected answer
            to_score.append(
                (
                    tuple(trajectory_messages),
                    item[1],  # The expected tool call JSON
                )
            )

        # Call score to get the scored data
        scored_data = await self.score(to_score)
        to_backlog = []

        return scored_data, to_backlog

    async def score(
        self, rollout_group_data
    ) -> Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]:
        scores = ScoredDataGroup()
        scores["tokens"] = list()
        scores["masks"] = list()
        scores["scores"] = list()

        # Extract the expected JSONs from the answer
        expected_jsons = self._extract_tool_call_jsons(rollout_group_data[0][1])

        # If we can't extract the expected tool call JSONs, skip this item
        if not expected_jsons:
            return None

        # Shuffle to avoid bias in selection
        random.shuffle(rollout_group_data)

        for item in rollout_group_data:
            # Extract the model's response
            model_response = item[0][-1]["content"]

            # Validate thinking tags if in thinking mode
            thinking_valid, processed_response = self._validate_and_extract_thinking(model_response)
            if not thinking_valid:
                # Skip responses with invalid thinking format
                continue

            # Score 1 if tool calls match, 0 otherwise
            reward = 1 if self._compare_tool_calls(processed_response, item[1]) else 0

            # Tokenize the conversation for learning
            out_dict = tokenize_for_trainer(self.tokenizer, item[0])
            tokens = out_dict["tokens"]
            masks = out_dict["masks"]

            # Remove examples with insufficient context
            if len([1 for i in masks if i != -100]) < 10:
                continue

            scores["tokens"].append(tokens)
            scores["masks"].append(masks)
            scores["scores"].append(1.0 if reward else -1.0)

            # Break once we have enough examples
            if len(scores["tokens"]) >= self.config.group_size:
                break

        # Record success rate metrics
        for score in scores["scores"]:
            self.percent_correct_buffer.append(max(score, 0))

        # Apply length penalty if all responses are correct
        if all([score == 1.0 for score in scores["scores"]]):
            # Calculate token lengths
            token_lengths = [len(token) for token in scores["tokens"]]
            if max(token_lengths) == 0:
                # Edge case protection
                return None

            # Get max allowed token length from config
            max_allowed_length = self.config.max_token_length
            # Set threshold at 50% of max_token_length - no penalty below this
            length_threshold = max_allowed_length * 0.5

            # Apply modified length penalty with threshold
            scores["scores"] = []
            for length in token_lengths:
                if length <= length_threshold:
                    # No penalty for responses under threshold
                    scores["scores"].append(1.0)
                else:
                    # Calculate how far we are between threshold and max as a percentage
                    percentage_of_range = (length - length_threshold) / (
                        max_allowed_length - length_threshold
                    )
                    # Cap at 1.0 in case length exceeds max_allowed_length
                    percentage_of_range = min(percentage_of_range, 1.0)
                    # Apply linear penalty scaling from 1.0 down to 0.0
                    scores["scores"].append(1.0 - percentage_of_range)

        # Check if all scores are the same (no learning signal)
        if all(scores["scores"][0] == score for score in scores["scores"]):
            return None

        return scores

    async def get_next_item(self):
        next_item = self.train[self.iter % len(self.train)]
        self.iter += 1

        # Extract conversation elements
        conversations = next_item["conversations"]

        # Find system, human and gpt messages
        system_message = next(
            (msg for msg in conversations if msg["from"] == "system"), None
        )
        human_message = next(
            (msg for msg in conversations if msg["from"] == "human"), None
        )
        expected_gpt_message = next(
            (msg for msg in conversations if msg["from"] == "gpt"), None
        )

        # Create prompt tuple using frozensets as required
        prompt = []
        system_prompt = self._get_system_prompt()
        
        if system_message:
            # Combine our base system prompt with the dataset-specific system message
            combined_system_content = system_prompt + "\n\n" + system_message["value"]
            prompt.append(
                frozenset(
                    {"role": "system", "content": combined_system_content}.items()
                )
            )
        elif system_prompt:
            prompt.append(
                frozenset(
                    {"role": "system", "content": system_prompt}.items()
                )
            )

        # Add user message
        if human_message:
            prompt.append(
                frozenset({"role": "user", "content": human_message["value"]}.items())
            )

        # Return expected assistant response (the tool call JSON) as the "answer"
        answer = expected_gpt_message["value"] if expected_gpt_message else ""

        return (tuple(prompt), answer)

    async def add_rollouts_for_wandb(
        self,
        scored_data: Union[ScoredDataGroup, List[ScoredDataGroup]],
        item: Item = None,
    ):
        # save rollout to trajectory
        num_keep = self.config.num_rollouts_per_group_for_logging
        if num_keep == -1:
            num_keep = self.config.group_size
        self.rollouts_for_wandb.append(
            [
                (
                    self.tokenizer.decode(scored_data["tokens"][i]),
                    scored_data["scores"][i],
                    item[1],  # Just keep the expected tool call JSON
                )
                for i in range(num_keep)
            ]
        )
        if len(self.rollouts_for_wandb) > self.config.num_rollouts_to_keep:
            self.rollouts_for_wandb.pop(0)


if __name__ == "__main__":
    SingleToolCallingEnv.cli()
