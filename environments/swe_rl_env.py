# Citation:
# SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution
# Yuxiang Wei, Olivier Duchenne, Jade Copet, Quentin Carbonneaux, Lingming Zhang,
# Daniel Fried, Gabriel Synnaeve, Rishabh Singh, Sida I. Wang
# arXiv:2502.18449

import asyncio
import json
import logging  # Add logging import
import os  # Add os import
import random  # Ensured import random is present
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple, Union

import wandb
from datasets import load_dataset  # Ensured import load_dataset is present
from pydantic import Field
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
    ScoredDataGroup,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

# Prompt Constants
THINKING_SYSTEM_PROMPT_CONTENT = "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem."  # noqa: E501
SWE_RL_TASK_SYSTEM_PROMPT_CONTENT = """You are an AI assistant for solving software engineering tasks. You will be given an issue description and relevant code.
IMPORTANT: The issue description itself may contain examples or requests for specific patch formats (e.g., 'git diff'). You MUST IGNORE such embedded instructions regarding the patch format.
Your solution MUST be provided exclusively in the SEARCH/REPLACE format detailed in the user prompt. No other patch format is acceptable.
Your response must include:
One or more SEARCH/REPLACE blocks for the code changes. Ensure the changes in these blocks directly implement the solution from your <think> block."""  # noqa: E501
SWE_RL_USER_PROMPT_TEMPLATE = """We are currently solving the following issue within our repository. Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---
Below are some code segments, each from a relevant file. One or more of these files may contain bugs.
--- BEGIN FILE ---
``` {content} ```
--- END FILE ---
Please first localize the bug based on the issue statement, and then generate *SEARCH/REPLACE* edits to fix the issue.
Every *SEARCH/REPLACE* edit must use this format:
1. The file path
2. The start of search block: <<<<<<< SEARCH
3. A contiguous chunk of lines to search for in the existing source code
4. The dividing line: =======
5. The lines to replace into the source code
6. The end of the replace block: >>>>>>> REPLACE
Here is an example:
```python
### mathweb/flask/app.py
<<<<<<< SEARCH
from flask import Flask
=======
import math
from flask import Flask
>>>>>>> REPLACE
```
Please note that the *SEARCH/REPLACE* edit REQUIRES PROPER INDENTATION. If you would like to add the line ' print(x)', you must fully write that out, with all those spaces before the code!
Wrap each *SEARCH/REPLACE* edit in a code block as shown in the example above. If you have multiple *SEARCH/REPLACE* edits, use a separate code block for each one."""  # noqa: E501

# In-Context Learning Example Constants
ICL_EXAMPLE_PROBLEM_STATEMENT = """The `calculate_area` function for a rectangle is incorrectly multiplying by 3 instead of the width. Also, it should handle cases where length or width are zero or negative by returning 0, as area cannot be negative or based on non-positive dimensions."""  # noqa: E501
ICL_EXAMPLE_CODE_CONTEXT = """### geometry/shapes.py
def calculate_area(length, width):
    # Intended to calculate area of a rectangle
    if length <= 0 or width <= 0:
        # Should return 0 for invalid dimensions
        pass
    return length * 3 # Incorrect calculation
"""  # noqa: E501
ICL_EXAMPLE_ASSISTANT_THINKING = """"""  # Empty think block for ICL # noqa: E501
ICL_EXAMPLE_ASSISTANT_PATCH_STR = """```python
### geometry/shapes.py
<<<<<<< SEARCH
def calculate_area(length, width):
    # Intended to calculate area of a rectangle
    if length <= 0 or width <= 0:
        # Should return 0 for invalid dimensions
        pass
    return length * 3 # Incorrect calculation
=======
def calculate_area(length, width):
    # Intended to calculate area of a rectangle
    if length <= 0 or width <= 0:
        return 0 # Handle non-positive dimensions
    return length * width # Correct calculation
>>>>>>> REPLACE
```"""  # noqa: E501


class SWERLEnvConfig(BaseEnvConfig):
    eval_n_samples: int = Field(
        default=1, description="Number of samples to generate per eval item."
    )
    # HF Dataset Configs
    dataset_name: str = Field(
        default="princeton-nlp/SWE-bench_Lite_oracle",
        description="Name of the Hugging Face dataset to load.",
    )
    dataset_config_name: Optional[str] = Field(
        default=None,
        description="Configuration name for the Hugging Face dataset (e.g., a subset).",
    )
    dataset_split_train: str = Field(
        default="train", description="Dataset split to use for training."
    )
    dataset_split_eval: str = Field(
        default="test", description="Dataset split to use for evaluation."
    )
    dataset_issue_column: str = Field(
        default="problem_statement",
        description="Column name for the issue/problem statement.",
    )
    dataset_code_context_column: str = Field(
        default="text", description="Column name for the code context."
    )
    dataset_oracle_patch_column: str = Field(
        default="patch", description="Column name for the oracle patch."
    )
    max_train_samples: Optional[int] = Field(
        default=None,
        description="Maximum number of training samples to load. None for all.",
    )
    max_test_samples: Optional[int] = Field(
        default=None,
        description="Maximum number of test samples to load. None for all.",
    )
    # Curriculum Learning Configs
    use_curriculum_learning: bool = Field(
        default=True,
        description="Whether to use curriculum learning with an ICL prompt initially.",
    )
    icl_prompt_threshold: float = Field(
        default=0.20,
        description="The train/avg_patch_format_accuracy threshold at which to switch from ICL to standard prompt.",
    )
    dump_rollouts: bool = Field(
        default=False,
        description="Whether to dump rollouts to JSONL files.",
    )


class SWERLEnv(BaseEnv):
    name = "swe_rl"
    env_config_cls = SWERLEnvConfig

    def __init__(
        self,
        config: SWERLEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)

        # Initialize the logger. This is typically done in the base class,
        # but added here to resolve the AttributeError if the base class doesn't.
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            # Add a basic stream handler if no handlers are configured.
            # This prevents "No handlers could be found for logger" messages
            # and ensures logs are output to the console.
            _handler = logging.StreamHandler()
            _formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            _handler.setFormatter(_formatter)
            self.logger.addHandler(_handler)
            self.logger.setLevel(logging.INFO)  # Set a default logging level.
        # Ensure the logger itself is enabled (e.g. if BaseEnv might have disabled it by name)
        self.logger.disabled = False

        self.percent_format_correct_buffer = []
        self.similarity_score_buffer = []
        self.eval_metrics = []
        self.train_dataset: List[Dict[str, str]] = []
        self.test_dataset: List[Dict[str, str]] = []
        self.iter = 0
        self.think_tags_present_buffer = []
        self.think_tags_well_formed_buffer = []

        # For saving rollouts to JSONL
        self.rollouts_to_save_buffer: List[
            Dict[str, Union[str, List[Dict[str, Union[List[Dict[str, str]], float]]]]]
        ] = []
        self.processed_item_count = 0
        # Creates .../atropos/environments/datadumps/ relative to the project structure
        self.datadumps_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "datadumps"
        )
        self.save_file_batch_num = 0

        # Curriculum Learning State
        self.using_icl_prompt: bool = self.config.use_curriculum_learning

    @classmethod
    def config_init(cls) -> Tuple[SWERLEnvConfig, List[APIServerConfig]]:
        env_config = SWERLEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=16,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=10000,
            batch_size=512,
            steps_per_eval=100,
            max_token_length=1024 * 15,
            inference_weight=1.0,
            wandb_name="swe_rl_env_deep_hermes_hf_dataset",  # Updated wandb_name
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.1,
            ensure_scores_are_not_same=True,
            eval_n_samples=1,
            # HF Dataset Configs
            dataset_name="princeton-nlp/SWE-bench_Lite_oracle",
            dataset_config_name=None,
            dataset_split_train="test",
            dataset_split_eval="test",
            dataset_issue_column="problem_statement",
            dataset_code_context_column="text",
            dataset_oracle_patch_column="patch",
            max_train_samples=1000,
            max_test_samples=200,
        )
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9004/v1",
                api_key="x",
                num_max_requests_at_once=32,
                num_requests_for_eval=64,
            ),
        ]
        return env_config, server_configs

    async def setup(self):
        self.logger.info(
            f"Setting up SWE-RL environment with dataset: {self.config.dataset_name}"
        )
        self.train_dataset = []
        self.test_dataset = []

        try:
            # Load training data
            self.logger.info(
                f"Loading training data from {self.config.dataset_name}, split: {self.config.dataset_split_train}"
            )
            train_data_raw = load_dataset(
                path=self.config.dataset_name,
                name=self.config.dataset_config_name,
                split=self.config.dataset_split_train,
                streaming=False,
            )
            if self.config.max_train_samples is not None:
                self.logger.info(
                    f"Taking max {self.config.max_train_samples} training samples."
                )
                # For datasets.Dataset, select directly. For IterableDataset, .take()
                if hasattr(train_data_raw, "select"):  # It's a Dataset
                    indices = range(
                        min(len(train_data_raw), self.config.max_train_samples)
                    )
                    train_data_raw = train_data_raw.select(indices)
                else:  # It's an IterableDataset
                    train_data_raw = train_data_raw.take(self.config.max_train_samples)

            # Load evaluation data
            self.logger.info(
                f"Loading evaluation data from {self.config.dataset_name}, split: {self.config.dataset_split_eval}"
            )
            eval_data_raw = load_dataset(
                path=self.config.dataset_name,
                name=self.config.dataset_config_name,
                split=self.config.dataset_split_eval,
                streaming=False,
            )
            if self.config.max_test_samples is not None:
                self.logger.info(
                    f"Taking max {self.config.max_test_samples} test samples."
                )
                if hasattr(eval_data_raw, "select"):  # It's a Dataset
                    indices = range(
                        min(len(eval_data_raw), self.config.max_test_samples)
                    )
                    eval_data_raw = eval_data_raw.select(indices)
                else:  # It's an IterableDataset
                    eval_data_raw = eval_data_raw.take(self.config.max_test_samples)

            self.logger.info("Mapping dataset columns...")
            for item_idx, raw_item in enumerate(train_data_raw):
                try:
                    self.train_dataset.append(
                        {
                            "item_id": f"train_{self.config.dataset_name}_{item_idx}",
                            "issue": raw_item[self.config.dataset_issue_column],
                            "code_context": raw_item[
                                self.config.dataset_code_context_column
                            ],
                            "oracle_patch": raw_item[
                                self.config.dataset_oracle_patch_column
                            ],
                        }
                    )
                except KeyError as e:
                    self.logger.error(
                        f"Column mapping error for training item {item_idx}: {e}. Skipping item. Raw: {str(raw_item)[:500]}"  # noqa: E501
                    )
                    continue

            for item_idx, raw_item in enumerate(eval_data_raw):
                try:
                    self.test_dataset.append(
                        {
                            "item_id": f"test_{self.config.dataset_name}_{item_idx}",
                            "issue": raw_item[self.config.dataset_issue_column],
                            "code_context": raw_item[
                                self.config.dataset_code_context_column
                            ],
                            "oracle_patch": raw_item[
                                self.config.dataset_oracle_patch_column
                            ],
                        }
                    )
                except KeyError as e:
                    self.logger.error(
                        f"Column mapping error for eval item {item_idx}: {e}. Skipping item. Raw: {str(raw_item)[:500]}"
                    )
                    continue

            random.shuffle(self.train_dataset)
            random.shuffle(self.test_dataset)

            self.logger.info(
                f"Loaded dataset: {len(self.train_dataset)} training examples, {len(self.test_dataset)} test examples."
            )

        except Exception as e:
            self.logger.error(
                f"Error loading or processing dataset '{self.config.dataset_name}': {e}"
            )
            self.train_dataset = []
            self.test_dataset = []
            # Consider re-raising if dataset is critical: raise e

        self.iter = 0
        if not self.train_dataset:
            self.logger.warning(
                "Training dataset is empty after setup. Check dataset configuration and availability."
            )
        if not self.test_dataset:
            self.logger.warning(
                "Test dataset is empty after setup. Check dataset configuration and availability."
            )

    async def get_next_item(self) -> Optional[Dict[str, str]]:
        if not self.train_dataset:
            self.logger.warning("Train dataset is empty. Cannot get next item.")
            return None
        item_index = self.iter % len(self.train_dataset)
        next_raw_item = self.train_dataset[item_index]  # Already mapped
        self.iter += 1
        # The item from self.train_dataset already has "issue", "code_context", "oracle_patch", "item_id"
        # We need to rename them to "problem_statement", "code_context", "oracle_patch" for collect_trajectories
        return {
            "problem_statement": next_raw_item["issue"],
            "code_context": next_raw_item["code_context"],
            "oracle_patch": next_raw_item["oracle_patch"],
            "item_id": next_raw_item["item_id"],
        }

    async def collect_trajectories(
        self, item: Dict[str, str]
    ) -> Tuple[Optional[ScoredDataGroup], List[Dict[str, str]]]:
        problem_statement = item["problem_statement"]
        code_context = item["code_context"]
        oracle_patch = item["oracle_patch"]
        item_id = item.get("item_id", "unknown_item")

        # Combine system prompts
        combined_system_content = (
            THINKING_SYSTEM_PROMPT_CONTENT + "\n\n" + SWE_RL_TASK_SYSTEM_PROMPT_CONTENT
        )

        # Prepare messages for the LLM
        messages_for_llm_prompt: List[Dict[str, str]] = []
        messages_for_llm_prompt.append(
            {"role": "system", "content": combined_system_content}
        )

        if self.config.use_curriculum_learning and self.using_icl_prompt:
            # Add ICL example to the prompt
            icl_user_content = SWE_RL_USER_PROMPT_TEMPLATE.format(
                problem_statement=ICL_EXAMPLE_PROBLEM_STATEMENT,
                content=ICL_EXAMPLE_CODE_CONTEXT,
            )
            messages_for_llm_prompt.append(
                {"role": "user", "content": icl_user_content}
            )
            messages_for_llm_prompt.append(
                {
                    "role": "assistant",
                    "content": ICL_EXAMPLE_ASSISTANT_THINKING
                    + "\n\n"
                    + ICL_EXAMPLE_ASSISTANT_PATCH_STR,
                }
            )

        # Add the actual current item
        formatted_user_content_current_item = SWE_RL_USER_PROMPT_TEMPLATE.format(
            problem_statement=problem_statement, content=code_context
        )
        messages_for_llm_prompt.append(
            {"role": "user", "content": formatted_user_content_current_item}
        )

        try:
            if not self.tokenizer:
                self.logger.error(f"Tokenizer not available for item {item_id}.")
                return None, []
            prompt_for_llm = self.tokenizer.apply_chat_template(
                messages_for_llm_prompt, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            # Log the messages that caused the error for easier debugging
            self.logger.error(
                f"Error applying chat template for item {item_id}: {e}. Messages: {messages_for_llm_prompt}"
            )
            return None, []

        stop_tokens = ["<|eot_id|>", "<|end_of_text|>"]
        if (
            self.tokenizer
            and self.tokenizer.eos_token
            and self.tokenizer.eos_token not in stop_tokens
        ):
            stop_tokens.insert(0, self.tokenizer.eos_token)

        completions = await self.server.completion(
            prompt=prompt_for_llm,
            n=self.config.group_size,
            max_tokens=self.config.max_token_length,
            temperature=0.8,
            stop=stop_tokens,
        )
        if not completions or not completions.choices:
            self.logger.warning(f"No completions received for item_id: {item_id}")
            return None, []

        # Prepare to collect all conversations and their potential scores for this item
        # This list will hold tuples of (conversation_messages, oracle_patch, finish_reason)
        # which is the input format expected by the self.score method.
        raw_rollouts_for_scoring = []

        for choice in completions.choices:
            current_trajectory_messages = messages_for_llm_prompt + [
                {"role": "assistant", "content": choice.text.strip()}
            ]
            raw_rollouts_for_scoring.append(
                (current_trajectory_messages, oracle_patch, choice.finish_reason)
            )

        if not raw_rollouts_for_scoring:
            return None, []

        # Score all generated rollouts for the current item
        scored_data = await self.score(raw_rollouts_for_scoring)

        # If rollouts were generated and scored, and data dumping is enabled,
        # prepare them for saving.
        if scored_data and self.config.dump_rollouts:
            rollouts_with_scores_to_save = []

            num_scored_rollouts = len(scored_data.get("scores", []))
            for i in range(num_scored_rollouts):
                # raw_rollouts_for_scoring[i][0] is the list of message dicts for the i-th rollout
                conversation_messages = raw_rollouts_for_scoring[i][0]
                score_for_rollout = scored_data["scores"][i]
                rollouts_with_scores_to_save.append(
                    {
                        "conversation": conversation_messages,  # Full conversation history
                        "score": score_for_rollout,
                    }
                )

            if rollouts_with_scores_to_save:
                item_data_to_save = {
                    "item_id": item_id,
                    "rollouts": rollouts_with_scores_to_save,  # Changed from "conversations"
                }
                self.rollouts_to_save_buffer.append(item_data_to_save)
                self.processed_item_count += 1

                # Check if it's time to save a batch of rollouts
                if (
                    self.config.dump_rollouts
                    and self.processed_item_count % 100 == 0
                    and self.processed_item_count > 0
                ):
                    log_msg = (
                        f"Reached {self.processed_item_count} processed items. "
                        f"Triggering save for {len(self.rollouts_to_save_buffer)} items "
                        f"(each with multiple scored rollouts)."
                    )
                    self.logger.info(log_msg)
                    await self._save_rollouts_to_jsonl()

        if scored_data and item_id != "unknown_item":
            scored_data["item_ids"] = [item_id] * len(scored_data.get("scores", []))

        return scored_data, []

    def _extract_content_after_think_tags(
        self, response_text: str
    ) -> Tuple[Optional[str], bool, bool]:
        think_start_match = re.search(r"<think>", response_text, re.IGNORECASE)
        think_end_match = re.search(r"</think>", response_text, re.IGNORECASE)
        think_tags_present = (
            think_start_match is not None and think_end_match is not None
        )
        think_tags_well_formed = False
        content_after_think_tags = None
        if think_tags_present:
            think_start_pos = think_start_match.start()
            think_end_pos = think_end_match.start()
            if think_start_pos < think_end_pos:
                think_tags_well_formed = True
                content_after_think_tags = response_text[
                    think_end_match.end() :
                ].strip()
            else:
                self.logger.debug("Think tags malformed: </think> not after <think>.")
        elif think_start_match and not think_end_match:
            self.logger.debug("Think tags malformed: <think> present but no </think>.")
        elif not think_start_match and think_end_match:
            self.logger.debug("Think tags malformed: </think> present but no <think>.")
        return content_after_think_tags, think_tags_present, think_tags_well_formed

    def _parse_search_replace_patch(
        self, patch_text: str
    ) -> Optional[List[Dict[str, str]]]:
        hunks = []
        # Use splitlines to handle \r\n and \n robustly, then strip each line
        lines = [line.strip() for line in patch_text.strip().splitlines(keepends=False)]
        idx = 0
        if not any(
            line for line in lines
        ):  # Check if effectively empty after stripping
            self.logger.debug(
                "Patch parsing error: Patch text is empty or contains only whitespace after initial processing."
            )
            return None

        while idx < len(lines):
            line = lines[idx]  # Already stripped
            if (
                not line
            ):  # Skip empty lines (e.g. if there were multiple newlines between hunks)
                idx += 1
                continue
            if not line.startswith("### "):
                self.logger.debug(
                    f"Patch parsing error: Expected file path (### path/to/file), got: '{line}'"
                )
                return None
            file_path = line[4:].strip()
            if not file_path:  # Ensure file path is not empty after "### "
                self.logger.debug(
                    f"Patch parsing error: File path is empty after '### '. Line: '{line}'"
                )
                return None
            idx += 1

            if (
                idx >= len(lines) or lines[idx] != "<<<<<<< SEARCH"
            ):  # lines[idx] is already stripped
                self.logger.debug(
                    f"Patch parsing error: Expected '<<<<<<< SEARCH' for file {file_path}, got '{lines[idx] if idx < len(lines) else 'EOF'}'"  # noqa: E501
                )
                return None
            idx += 1
            search_lines_list = []
            while idx < len(lines) and lines[idx] != "=======":
                search_lines_list.append(lines[idx])  # Append the stripped line
                idx += 1

            if idx >= len(lines) or lines[idx] != "=======":
                self.logger.debug(
                    f"Patch parsing error: Expected '=======' for file {file_path}, got '{lines[idx] if idx < len(lines) else 'EOF'}'"  # noqa: E501
                )
                return None
            idx += 1
            replace_lines_list = []
            while idx < len(lines) and lines[idx] != ">>>>>>> REPLACE":
                replace_lines_list.append(lines[idx])  # Append the stripped line
                idx += 1
            if idx >= len(lines) or lines[idx] != ">>>>>>> REPLACE":
                self.logger.debug(
                    f"Patch parsing error: Expected '>>>>>>> REPLACE' for file {file_path}, got '{lines[idx] if idx < len(lines) else 'EOF'}'"  # noqa: E501
                )
                return None
            idx += 1
            hunks.append(
                {
                    "file_path": file_path,
                    "search_lines": "\n".join(search_lines_list),
                    "replace_lines": "\n".join(replace_lines_list),
                }
            )

        if not hunks:
            self.logger.debug(
                "Patch parsing error: No valid hunks found in patch_text despite non-empty input."
            )
            return None
        return hunks

    def _reconstruct_patch_from_parsed(self, parsed_hunks: List[Dict[str, str]]) -> str:
        full_patch_parts = []
        for hunk in parsed_hunks:
            full_patch_parts.extend(
                [
                    f"### {hunk['file_path']}",
                    "<<<<<<< SEARCH",
                    hunk["search_lines"],
                    "=======",
                    hunk["replace_lines"],
                    ">>>>>>> REPLACE",
                ]
            )
        return "\n".join(full_patch_parts)

    async def score(
        self, rollout_group_data: List[Tuple[List[Dict[str, str]], str, str]]
    ) -> Optional[ScoredDataGroup]:
        scored_data = ScoredDataGroup()
        scored_data["tokens"] = []
        scored_data["masks"] = []
        scored_data["scores"] = []
        scored_data["messages"] = []
        scored_data["overrides"] = []

        patch_format_correct_count_batch = 0
        similarity_scores_batch_temp = []
        think_tags_present_count_batch = 0
        think_tags_well_formed_count_batch = 0

        for trajectory_messages, oracle_patch_str, finish_reason in rollout_group_data:
            assistant_response = ""
            if (
                trajectory_messages
                and isinstance(trajectory_messages, list)
                and len(trajectory_messages) > 0
                and trajectory_messages[-1].get("role") == "assistant"
            ):
                assistant_response = trajectory_messages[-1].get("content", "")

            override_dict = {}
            reward = -1.0

            content_to_parse_for_patch, think_present, think_well_formed = (
                self._extract_content_after_think_tags(assistant_response)
            )

            if think_present:
                think_tags_present_count_batch += 1
            if think_well_formed:
                think_tags_well_formed_count_batch += 1

            if finish_reason == "length":
                override_dict["set_advantage_to_zero"] = True
            elif think_present and not think_well_formed:
                pass  # reward remains -1.0
            else:
                patch_input_text = (
                    content_to_parse_for_patch
                    if think_well_formed
                    else assistant_response
                )
                if patch_input_text is None and think_well_formed:
                    pass
                elif patch_input_text is not None:
                    parsed_predicted_patch = self._parse_search_replace_patch(
                        patch_input_text
                    )
                    if parsed_predicted_patch is None:
                        pass
                    else:
                        patch_format_correct_count_batch += 1
                        reconstructed_predicted_patch = (
                            self._reconstruct_patch_from_parsed(parsed_predicted_patch)
                        )
                        reward = SequenceMatcher(
                            None, reconstructed_predicted_patch, oracle_patch_str
                        ).ratio()
                        similarity_scores_batch_temp.append(reward)
                else:
                    pass

            try:
                tokenized_output = tokenize_for_trainer(
                    tokenizer=self.tokenizer,
                    chat=trajectory_messages,
                    include_messages=True,
                )
            except Exception as e:
                self.logger.error(f"Tokenization failed: {e}")
                continue
            if (
                not tokenized_output
                or not tokenized_output.get("tokens")
                or not tokenized_output["tokens"][0]
            ):
                continue

            scored_data["tokens"].append(tokenized_output["tokens"])
            scored_data["masks"].append(tokenized_output["masks"])
            scored_data["scores"].append(reward)
            scored_data["messages"].append(
                tokenized_output.get("messages", trajectory_messages)
            )
            scored_data["overrides"].append(override_dict)
            if len(scored_data["scores"]) >= self.config.group_size:
                break

        if not scored_data["scores"]:
            return None
        if rollout_group_data:
            self.percent_format_correct_buffer.append(
                patch_format_correct_count_batch / len(rollout_group_data)
            )
            self.think_tags_present_buffer.append(
                think_tags_present_count_batch / len(rollout_group_data)
            )
            self.think_tags_well_formed_buffer.append(
                think_tags_well_formed_count_batch / len(rollout_group_data)
            )
        if similarity_scores_batch_temp:
            self.similarity_score_buffer.extend(similarity_scores_batch_temp)

        # Calculate and log average score for the current group
        current_scores = scored_data.get("scores", [])
        if current_scores:
            average_score = sum(current_scores) / len(current_scores)
            log_message_main = f"Group average score: {average_score:.4f}"
            if all(s == 1.0 for s in current_scores):
                self.logger.info(f"{log_message_main} (All successes in this group!)")
            elif all(
                s == 0.0 or s == -1.0 for s in current_scores
            ):  # Assuming -1.0 is also a failure state
                self.logger.info(
                    f"{log_message_main} (All failures in this group!)"
                )  # noqa: E501
            else:
                self.logger.info(log_message_main)

        if (
            self.config.ensure_scores_are_not_same
            and len(scored_data["scores"]) > 1
            and all(s == scored_data["scores"][0] for s in scored_data["scores"])
        ):
            return None
        return scored_data

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
            self.datadumps_dir, f"rollouts_batch_{self.save_file_batch_num:04d}.jsonl"
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

    async def _rollout_and_score_eval_item(self, test_item: Dict[str, str]) -> Dict:
        # Renamed internal item to avoid conflict with 'item' parameter in collect_trajectories
        current_test_item = test_item
        problem_statement, code_context, oracle_patch_str = (
            current_test_item["issue"],
            current_test_item["code_context"],
            current_test_item["oracle_patch"],
        )
        item_id = current_test_item.get("item_id", "unknown_eval_item")

        final_similarity_score, final_patch_format_correct = -1.0, 0
        llm_raw_response, think_present_eval, think_well_formed_eval = (
            "INIT_ERROR",
            0,
            0,
        )

        formatted_user_content = SWE_RL_USER_PROMPT_TEMPLATE.format(
            problem_statement=problem_statement, content=code_context
        )

        # Combine system prompts
        combined_system_content_eval = (
            THINKING_SYSTEM_PROMPT_CONTENT + "\n\n" + SWE_RL_TASK_SYSTEM_PROMPT_CONTENT
        )
        messages_for_prompt = [
            {"role": "system", "content": combined_system_content_eval},
            {"role": "user", "content": formatted_user_content},
        ]

        prompt_for_llm = "ERROR_APPLYING_CHAT_TEMPLATE"
        try:
            if not self.tokenizer:
                raise ValueError("Tokenizer not available for eval")
            prompt_for_llm = self.tokenizer.apply_chat_template(
                messages_for_prompt, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            self.logger.error(
                f"Eval prompt chat template application error for item {item_id}: {e}"
            )
            return {
                "item_id": item_id,
                "similarity_score": -1.0,
                "format_correct": 0,
                "predicted_patch": "CHAT_TEMPLATE_ERROR",
                "oracle_patch": oracle_patch_str,
                "prompt": prompt_for_llm,
                "think_tags_present": 0,
                "think_tags_well_formed": 0,
            }

        stop_tokens = ["<|eot_id|>", "<|end_of_text|>"]
        if (
            self.tokenizer
            and self.tokenizer.eos_token
            and self.tokenizer.eos_token not in stop_tokens
        ):
            stop_tokens.insert(0, self.tokenizer.eos_token)

        completions = await self.server.completion(
            prompt=prompt_for_llm,
            n=self.config.eval_n_samples,
            max_tokens=self.config.max_token_length,
            temperature=0.1,
            stop=stop_tokens,
            split="eval",
        )

        if completions and completions.choices:
            choice = completions.choices[0]
            llm_raw_response = choice.text.strip()
            content_after_think, think_present, think_well_formed = (
                self._extract_content_after_think_tags(llm_raw_response)
            )
            think_present_eval, think_well_formed_eval = int(think_present), int(
                think_well_formed
            )
            if choice.finish_reason == "length" or (
                think_present and not think_well_formed
            ):
                pass
            else:
                patch_input_text = (
                    content_after_think if think_well_formed else llm_raw_response
                )
                if patch_input_text is not None:
                    parsed_patch = self._parse_search_replace_patch(patch_input_text)
                    if parsed_patch:
                        final_patch_format_correct = 1
                        final_similarity_score = SequenceMatcher(
                            None,
                            self._reconstruct_patch_from_parsed(parsed_patch),
                            oracle_patch_str,
                        ).ratio()
        else:
            llm_raw_response = "NO_COMPLETION_RECEIVED"

        return {
            "item_id": item_id,
            "similarity_score": final_similarity_score,
            "format_correct": final_patch_format_correct,
            "predicted_patch": llm_raw_response,
            "oracle_patch": oracle_patch_str,
            "prompt": prompt_for_llm,
            "think_tags_present": think_present_eval,
            "think_tags_well_formed": think_well_formed_eval,
        }

    async def evaluate(self, *args, **kwargs):
        self.logger.info("Starting evaluation...")
        if not self.test_dataset:
            self.logger.warning("Test dataset is empty.")
            self.eval_metrics = []
            return

        # Use internal keys "issue", "code_context", "oracle_patch" for _rollout_and_score_eval_item
        tasks = [
            self._rollout_and_score_eval_item(item) for item in self.test_dataset
        ]  # test_dataset items are already mapped

        results = await tqdm_asyncio.gather(*tasks)
        if not results:
            self.logger.warning("No results from eval tasks.")
            self.eval_metrics = []
            return

        total_items = len(results)
        num_patch_format_correct = sum(r["format_correct"] for r in results)
        correct_format_sim_scores = [
            r["similarity_score"]
            for r in results
            if r["format_correct"] == 1 and r["similarity_score"] != -1.0
        ]
        num_pass_at_1 = sum(
            1
            for r in results
            if r["format_correct"] == 1 and r["similarity_score"] == 1.0
        )
        num_think_tags_present = sum(r["think_tags_present"] for r in results)
        num_think_tags_well_formed = sum(r["think_tags_well_formed"] for r in results)

        self.eval_metrics = [
            (
                "eval/avg_similarity_score_correct_patch_format",
                (
                    sum(correct_format_sim_scores) / len(correct_format_sim_scores)
                    if correct_format_sim_scores
                    else 0.0
                ),
            ),
            (
                "eval/patch_format_accuracy",
                num_patch_format_correct / total_items if total_items > 0 else 0.0,
            ),
            ("eval/pass_at_1", num_pass_at_1 / total_items if total_items > 0 else 0.0),
            ("eval/total_eval_items", float(total_items)),
            ("eval/total_patch_format_correct", float(num_patch_format_correct)),
            (
                "eval/avg_think_tags_present",
                num_think_tags_present / total_items if total_items else 0.0,
            ),
            (
                "eval/avg_think_tags_well_formed",
                num_think_tags_well_formed / total_items if total_items else 0.0,
            ),
        ]
        self.logger.info("Evaluation finished.")
        self.logger.info(f"Metrics: {self.eval_metrics}")  # noqa: E501

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}

        def _log_buffer_avg(buffer, metric_name):
            if buffer:
                wandb_metrics[metric_name] = sum(buffer) / len(buffer)
            else:
                wandb_metrics[metric_name] = 0.0
            buffer.clear()

        _log_buffer_avg(
            self.percent_format_correct_buffer, "train/avg_patch_format_accuracy"
        )
        _log_buffer_avg(
            self.similarity_score_buffer,
            "train/avg_similarity_score_for_correct_patches",
        )
        _log_buffer_avg(
            self.think_tags_present_buffer, "train/avg_think_tags_present_accuracy"
        )
        _log_buffer_avg(
            self.think_tags_well_formed_buffer,
            "train/avg_think_tags_well_formed_accuracy",
        )

        # Curriculum learning: Check threshold and potentially switch off ICL prompt
        if self.config.use_curriculum_learning and self.using_icl_prompt:
            current_patch_format_accuracy = wandb_metrics.get(
                "train/avg_patch_format_accuracy", 0.0
            )
            if current_patch_format_accuracy >= self.config.icl_prompt_threshold:
                self.using_icl_prompt = False
                self.logger.info(
                    f"ICL threshold met ({current_patch_format_accuracy:.2f} >= "
                    f"{self.config.icl_prompt_threshold:.2f}). Switching off ICL prompt "
                    f"for subsequent training items."
                )

        # Log ICL status
        if self.config.use_curriculum_learning:
            wandb_metrics["env/using_icl_prompt"] = (
                1.0 if self.using_icl_prompt else 0.0
            )

        if hasattr(self, "eval_metrics") and self.eval_metrics:
            for key, value in self.eval_metrics:
                wandb_metrics[key] = value
        self.eval_metrics = []
        await super().wandb_log(wandb_metrics)

    async def add_rollouts_for_wandb(
        self, scored_data: ScoredDataGroup, item: Optional[Dict[str, str]] = None
    ):
        # item here is the output of get_next_item, so it has "problem_statement"
        oracle_patch_str = item.get("oracle_patch", "Missing") if item else "Unknown"
        problem_statement_str = (
            item.get("problem_statement", "Missing") if item else "Unknown"
        )

        num_keep = self.config.num_rollouts_per_group_for_logging
        tokens_batch, scores_batch = scored_data.get("tokens"), scored_data.get(
            "scores"
        )
        num_scores_in_batch = len(scores_batch) if scores_batch else 0
        if num_keep == -1:
            num_keep = num_scores_in_batch
        item_ids_list_from_data = scored_data.get("item_ids")
        default_item_id_base = (
            item.get("item_id", "unknown_item") if item else "unknown_item"
        )
        item_ids_list = [
            f"{default_item_id_base}_{j}" for j in range(num_scores_in_batch)
        ]
        if (
            item_ids_list_from_data
            and len(item_ids_list_from_data) == num_scores_in_batch
        ):
            item_ids_list = item_ids_list_from_data
        if not tokens_batch or not scores_batch:
            return
        for i in range(min(num_keep, num_scores_in_batch)):
            try:
                full_interaction_text = self.tokenizer.decode(tokens_batch[i])
                self.rollouts_for_wandb.append(
                    {
                        "item_id": item_ids_list[i],
                        "problem_statement": problem_statement_str,
                        "full_interaction_text": full_interaction_text,
                        "oracle_patch": oracle_patch_str,
                        "score": scores_batch[i],
                    }
                )
            except Exception as e:
                self.logger.error(f"Error preparing rollout for WandB: {e}")
        while len(self.rollouts_for_wandb) > self.config.num_rollouts_to_keep:
            self.rollouts_for_wandb.pop(0)

    async def create_rollout_table(self, wandb_metrics: Dict) -> Dict:
        if hasattr(self, "rollouts_for_wandb") and self.rollouts_for_wandb:
            try:
                columns = [
                    "Item ID",
                    "Problem Statement",
                    "Full Interaction Text",
                    "Oracle Patch",
                    "Score",
                ]
                table_data = [
                    [
                        r["item_id"],
                        r["problem_statement"],
                        r["full_interaction_text"],
                        r["oracle_patch"],
                        r["score"],
                    ]
                    for r in self.rollouts_for_wandb
                ]
                wandb_metrics["train/rollouts"] = wandb.Table(
                    columns=columns, data=table_data
                )
            except Exception as e:
                self.logger.error(f"Error creating WandB rollout table: {e}")
        self.rollouts_for_wandb = []
        return wandb_metrics

    async def close(self):
        """Clean up and save any remaining rollouts before exiting."""
        self.logger.info(
            "Closing SWERLEnv. Attempting to save any remaining rollouts..."
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

        # Call the superclass's close method if it exists and is async, or handle appropriately
        # This is a placeholder; actual implementation depends on BaseEnv's close method.
        if hasattr(super(), "close") and asyncio.iscoroutinefunction(super().close):
            await super().close()
        elif hasattr(super(), "close"):
            super().close()  # Assuming it's a synchronous method
        self.logger.info("SWERLEnv closed.")


if __name__ == "__main__":
    SWERLEnv.cli()
