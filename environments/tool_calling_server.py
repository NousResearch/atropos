import asyncio
import json
import logging
import os
import random
import re
import uuid
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

system_prompt = (
    "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the "
    "problem and deliberate with yourself via systematic reasoning processes to help come to a correct "
    "solution prior to answering. You should enclose your thoughts and internal monologue inside <think> "
    "</think> tags, and then provide your solution or response to the problem."
)


class SingleToolCallingEnvConfig(BaseEnvConfig):
    dump_rollouts: bool = Field(
        default=False,
        description="Whether to dump rollouts to JSONL files.",
    )
    # Add other specific configs for SingleToolCallingEnv if any in the future


class SingleToolCallingEnv(BaseEnv):
    env_config_cls = SingleToolCallingEnvConfig

    def __init__(
        self,
        config: SingleToolCallingEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)

        # Explicitly initialize the logger for this class, similar to SWERLEnv
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            _handler = logging.StreamHandler()
            _formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            _handler.setFormatter(_formatter)
            self.logger.addHandler(_handler)
            self.logger.setLevel(logging.INFO)
        self.logger.disabled = False  # Ensure logger is enabled

        self.percent_correct_buffer = list()
        self.eval_metrics = list()
        # Add tracking for wandb visualizations
        self.rollouts_for_wandb = []
        self.completion_lengths = []

        # For saving rollouts to JSONL
        self.run_uuid = str(uuid.uuid4())
        # Buffer will store a list of item groups, where each group has an item_id and a list of its rollouts.
        # Each rollout detail will contain conversation, score, and expected_tool_call.
        RolloutDetail = Dict[
            str, Union[List[Dict[str, str]], float, str]
        ]  # Type alias for clarity
        ItemGroup = Dict[str, Union[str, List[RolloutDetail]]]
        self.rollouts_to_save_buffer: List[ItemGroup] = []
        self.processed_item_count = 0
        # Creates .../atropos/environments/datadumps/ relative to this file
        self.datadumps_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "datadumps"
        )
        self.save_file_batch_num = 0

    @classmethod
    def config_init(self) -> Tuple[SingleToolCallingEnvConfig, List[APIServerConfig]]:
        env_config = SingleToolCallingEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=16,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=2000,
            batch_size=1024,
            steps_per_eval=1000,
            max_token_length=1024 * 16,
            inference_weight=1.0,
            wandb_name="toolcall_think",
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.1,
            dump_rollouts=True,
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

        self.percent_correct_buffer = list()
        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = list()
        await super().wandb_log(wandb_metrics)

    async def setup(self):
        # Load the full dataset
        full_dataset = load_dataset(
            "NousResearch/XLAM-Atropos",
            "default",
            split="train",
        )

        full_dataset = full_dataset.shuffle(seed=42)

        # Create train/test split on the fly (e.g., 95% train, 5% test)
        split_dataset = full_dataset.train_test_split(test_size=0.02, seed=42)

        # Keep the splits as is - no need to reformat
        # self.train = split_dataset["train"]
        # self.test = split_dataset["test"]

        # Helper function to add static IDs
        def add_static_id_func(example, idx, prefix="train"):
            # Ensures we are working with a mutable copy if necessary,
            # though .map typically handles this by creating new examples.
            example_copy = dict(example)
            example_copy["dataset_static_id"] = f"{prefix}_ds_entry_{idx}"
            return example_copy

        # Add static IDs to train and test datasets
        # The with_indices=True argument provides the index of each example.
        self.train = split_dataset["train"].map(
            add_static_id_func, with_indices=True, fn_kwargs={"prefix": "train"}
        )
        self.test = split_dataset["test"].map(
            add_static_id_func, with_indices=True, fn_kwargs={"prefix": "test"}
        )

        self.iter = 0

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
            return 0  # Skip invalid conversations

        # Create messages for model
        messages = []
        if system_message:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt + "\n\n" + system_message["value"],
                }
            )
        messages.append({"role": "user", "content": human_message["value"]})

        # Apply chat template to convert messages to a single string
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        # Get model completion using completion() instead of chat_completion()
        completion = await self.server.completion(
            prompt=prompt,
            n=1,
            max_tokens=1024 * 15,
            temperature=1.0,
            split="eval",
        )

        # Extract the model's response from the completion
        model_response = completion.choices[0].text
        expected_response = expected_gpt_message["value"]

        # Extract and compare tool calls
        score = self._compare_tool_calls(model_response, expected_response)
        return score

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
        eval_tasks = []
        for test_item in self.test:
            eval_tasks.append(self.rollout_and_score_eval(test_item))
        scores = await tqdm_asyncio.gather(*eval_tasks)
        self.eval_metrics.append(("eval/percent_correct", sum(scores) / len(scores)))

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
            max_tokens=1024 * 15,
            temperature=0.8,  # Using temperature to get diverse responses
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

        # Log after scoring, before checking dump_rollouts condition
        current_batch_progress = self.processed_item_count % 100
        log_message_group_processed = (
            f"GROUP_PROC - Item Iter: {self.iter-1}, Scored Data Present: {bool(scored_data)}, "
            f"Dump Rollouts Cfg: {self.config.dump_rollouts}, "
            f"Total Items Processed (for save): {self.processed_item_count}, Batch Counter: {current_batch_progress}/99"
        )
        self.logger.info(log_message_group_processed)

        # If rollouts were generated and scored, and data dumping is enabled,
        # prepare them for saving.
        self.logger.info(
            f"COLLECT_TRAJ - dump_rollouts: {self.config.dump_rollouts}, processed_item_count: {self.processed_item_count}, current_buffer_size: {len(self.rollouts_to_save_buffer)}"  # noqa: E501
        )
        if scored_data and self.config.dump_rollouts:
            rollouts_for_current_item = (
                []
            )  # Store dicts of {conversation, score, expected_tool_call}

            num_scored_rollouts = len(scored_data.get("scores", []))
            conversation_messages_batch = scored_data.get("messages", [])

            for i in range(num_scored_rollouts):
                conversation_messages = (
                    conversation_messages_batch[i]
                    if i < len(conversation_messages_batch)
                    else []
                )
                score_for_rollout = scored_data["scores"][i]

                rollouts_for_current_item.append(
                    {
                        # item_id will be part of the parent group
                        "conversation": conversation_messages,  # Full conversation history
                        "score": score_for_rollout,
                        "expected_tool_call": item[
                            1
                        ],  # Include expected tool call from the item
                    }
                )

            if rollouts_for_current_item:
                # item is (prompt_tuple, answer_string)
                # Create a unique item_id for the source prompt/item.
                # Using self.iter which was incremented in get_next_item should give
                # a reasonable ID for the item itself.
                # source_item_id = f"item_{self.iter-1}"
                # Use the static_id from the item tuple (item[2]) as the source_item_id
                source_item_id = item[
                    2
                ]  # item is (prompt_tuple, answer_string, static_id_string)

                item_data_to_save = {
                    "item_id": source_item_id,
                    "rollouts": rollouts_for_current_item,
                }
                self.rollouts_to_save_buffer.append(item_data_to_save)
                self.processed_item_count += 1  # Increment per item (group)

                # Check if it's time to save a batch of items (groups)
                if (
                    self.config.dump_rollouts
                    and self.processed_item_count > 0
                    and self.processed_item_count % 100
                    == 0  # Save every 100 processed items (groups)
                ):
                    log_msg = (
                        f"Reached {self.processed_item_count} processed items. "
                        f"Triggering save for {len(self.rollouts_to_save_buffer)} item groups."
                    )
                    self.logger.info(log_msg)
                    await self._save_rollouts_to_jsonl()

        return scored_data, to_backlog

    async def score(
        self, rollout_group_data
    ) -> Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]:
        scores = ScoredDataGroup()
        scores["tokens"] = list()
        scores["masks"] = list()
        scores["scores"] = list()
        scores["messages"] = list()

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

            # Score 1 if tool calls match, 0 otherwise
            reward = 1 if self._compare_tool_calls(model_response, item[1]) else 0

            # Tokenize the conversation for learning
            out_dict = tokenize_for_trainer(
                self.tokenizer, item[0], include_messages=True
            )
            tokens = out_dict["tokens"]
            masks = out_dict["masks"]

            # Remove examples with insufficient context
            if len([1 for i in masks if i != -100]) < 10:
                continue

            scores["tokens"].append(tokens)
            scores["masks"].append(masks)
            scores["scores"].append(1.0 if reward else -1.0)
            scores["messages"].append(out_dict.get("messages", item[0]))

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

        # Calculate and log average score for the current group (mimicking swe_rl_env)
        current_scores_tc = scores.get("scores", [])
        if current_scores_tc:
            average_score_tc = sum(current_scores_tc) / len(current_scores_tc)
            log_message_main_tc = (
                f"ToolCalling Group average score: {average_score_tc:.4f}"
            )
            if all(s == 1.0 for s in current_scores_tc):
                self.logger.info(
                    f"{log_message_main_tc} (All successes in this group!)"
                )
            elif all(
                s == 0.0 or s == -1.0 for s in current_scores_tc
            ):  # Assuming -1.0 is also a failure state
                self.logger.info(f"{log_message_main_tc} (All failures in this group!)")
            else:
                self.logger.info(log_message_main_tc)

        return scores

    async def _save_rollouts_to_jsonl(self):
        """Saves the buffered rollouts to a JSONL file in the datadumps directory."""
        if not self.rollouts_to_save_buffer:
            self.logger.info("No rollouts in buffer to save.")
            return

        try:
            if not os.path.exists(self.datadumps_dir):
                os.makedirs(self.datadumps_dir)
                self.logger.info(f"Created directory: {self.datadumps_dir}")
        except OSError as e:
            self.logger.error(f"Error creating directory {self.datadumps_dir}: {e}")
            return  # Don't proceed if directory creation fails

        file_path = os.path.join(
            self.datadumps_dir,
            f"tool_calling_rollouts_{self.run_uuid}_{self.save_file_batch_num:04d}.jsonl",
        )

        try:
            with open(file_path, "w") as f:
                for rollout_dict in self.rollouts_to_save_buffer:
                    json.dump(rollout_dict, f)
                    f.write("\n")
            self.logger.info(
                f"Successfully saved {len(self.rollouts_to_save_buffer)} rollouts to {file_path}"
            )
            self.rollouts_to_save_buffer.clear()  # Clear buffer after successful save
            self.save_file_batch_num += 1
        except IOError as e:
            self.logger.error(f"Error writing rollouts to {file_path}: {e}")
        except Exception as e:
            self.logger.error(
                f"An unexpected error occurred while saving rollouts to {file_path}: {e}"
            )

    async def get_next_item(self):
        next_item = self.train[self.iter % len(self.train)]
        self.iter += 1

        # Extract conversation elements
        conversations = next_item["conversations"]
        # Extract the static ID
        static_id = next_item["dataset_static_id"]

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
        if system_message:
            # Combine our base system prompt with the dataset-specific system message
            combined_system_content = system_prompt + "\n\n" + system_message["value"]
            prompt.append(
                frozenset(
                    {"role": "system", "content": combined_system_content}.items()
                )
            )

        # Add user message
        if human_message:
            prompt.append(
                frozenset({"role": "user", "content": human_message["value"]}.items())
            )

        # Return expected assistant response (the tool call JSON) as the "answer"
        answer = expected_gpt_message["value"] if expected_gpt_message else ""

        return (tuple(prompt), answer, static_id)

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
                    item[
                        1
                    ],  # item is (prompt_tuple, answer_string, static_id_string), so item[1] is the expected tool call
                )
                for i in range(num_keep)
            ]
        )
        if len(self.rollouts_for_wandb) > self.config.num_rollouts_to_keep:
            self.rollouts_for_wandb.pop(0)

    async def close(self):
        """Clean up and save any remaining rollouts before exiting."""
        self.logger.info(
            "Closing SingleToolCallingEnv. Attempting to save any remaining rollouts..."
        )
        if (
            self.config.dump_rollouts and self.rollouts_to_save_buffer
        ):  # Check if there's anything to save
            self.logger.info(
                f"Found {len(self.rollouts_to_save_buffer)} rollouts in buffer. Saving now."
            )
            await self._save_rollouts_to_jsonl()
        else:
            self.logger.info("No rollouts in buffer to save upon closing.")

        # Call the superclass's close method if it exists and is async
        if hasattr(super(), "close") and asyncio.iscoroutinefunction(super().close):
            await super().close()
        elif hasattr(super(), "close"):
            super().close()  # Assuming it's a synchronous method
        self.logger.info("SingleToolCallingEnv closed.")


if __name__ == "__main__":
    SingleToolCallingEnv.cli()
