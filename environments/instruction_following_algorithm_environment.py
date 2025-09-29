import json
import os
import random
import re
import uuid
from typing import Dict, List, Optional, Tuple

import wandb
from datasets import Dataset, load_dataset
from langdetect import LangDetectException, detect
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

# System prompt can be reused or adapted for instruction following tasks
system_prompt = (
    "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the "
    "problem and deliberate with yourself via systematic reasoning processes to help come to a correct "
    "solution prior to answering. You should enclose your thoughts and internal monologue inside <think> "
    "</think> tags, and then provide your solution or response to the problem."
)


class IFConfig(BaseEnvConfig):
    dataset_name: str = Field("allenai/RLVR-IFeval", description="Default dataset name")
    dataset_config_name: Optional[str] = Field(
        None, description="Dataset config name, if any"
    )
    test_set_ratio: float = Field(
        0.05, description="The ratio of the selected dataset for testing"
    )
    dump_rollouts: bool = Field(
        False, description="Whether to dump successful rollouts to JSONL files"
    )
    dump_failed_rollouts: bool = Field(
        False,
        description="Whether to dump failed rollouts (all 0 scores) to JSONL files for debugging",
    )
    rollout_save_score_threshold: float = Field(
        0.7, description="Minimum score threshold for saving rollouts to data dumps"
    )
    max_group_average_for_training: float = Field(
        0.75,
        description="Maximum group average score to use for training (skip groups that are too easy)",
    )
    dataset_shuffle_seed: int = Field(
        42, description="Seed for shuffling the dataset during setup"
    )
    resume_from_unsolved_dataset: Optional[str] = Field(
        None,
        description="Path to a remaining_unsolved.jsonl file to resume training from specific unsolved items",
    )
    suppress_base_env_logs: bool = Field(
        default=True,
        description="Suppress verbose base environment logs (like status dict updates).",
    )
    solve_on_single_correct: bool = Field(
        default=False,
        description="Mark item as solved if even one rollout in the group gets it correct (removes from circulation)",
    )


class InstructionFollowingEnv(BaseEnv):
    env_config_cls = IFConfig

    def __init__(
        self,
        config: IFConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.percent_correct_buffer = list()
        self.eval_metrics = list()
        self.rollouts_for_wandb = []

        # Data dumping infrastructure
        self.rollouts_to_save_buffer = []
        self.failed_rollouts_to_save_buffer = []
        self.run_uuid = str(uuid.uuid4())[:8]
        self.save_file_batch_num = 0
        self.failed_save_file_batch_num = 0

        # Adaptive curriculum: cycling queue for unsolved items
        self.active_train_queue = []  # Items currently in circulation
        self.solved_items = []  # Items that have been solved (removed from circulation)
        self.item_attempt_counts = (
            {}
        )  # Track how many times each item has been attempted

        # Create data dumps directory
        self.datadumps_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data_dumps"
        )

        # Create datasets directory for curriculum state dumps
        self.datasets_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "datasets"
        )

        # Validate configuration for potential conflicts
        self._validate_config()

        # Configure logging suppression
        if self.config.suppress_base_env_logs:
            import logging

            # Suppress specific loggers that are too verbose
            logging.getLogger("atroposlib.envs.base").setLevel(logging.WARNING)
            logging.getLogger("httpx").setLevel(logging.WARNING)
            logging.getLogger("httpcore").setLevel(logging.WARNING)

    @classmethod
    def config_init(
        self,
    ) -> Tuple[IFConfig, List[APIServerConfig]]:
        # Configuration for the Instruction Following Environment
        env_config = IFConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=32,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=500,
            batch_size=1024,
            steps_per_eval=20,
            max_token_length=1024 * 15,
            inference_weight=1.0,
            wandb_name="instruction_following_rlvr_ifeval",  # Specific WandB project name
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.1,
            dataset_name="allenai/RLVR-IFeval",  # Default dataset
            dataset_config_name=None,  # RLVR-IFeval doesn't have a specific config name, uses 'default'
            test_set_ratio=0.05,  # The ratio of the selelcted dataset in %
            dump_rollouts=False,  # Enable data dumping if needed
            dump_failed_rollouts=False,  # Enable failed rollout dumping for debugging
            rollout_save_score_threshold=0.7,  # Save rollouts with score >= 0.7
            max_group_average_for_training=0.75,  # Skip groups that are too easy for training
            dataset_shuffle_seed=42,  # Seed for dataset shuffling
            resume_from_unsolved_dataset=None,  # Path to resume from unsolved items
            suppress_base_env_logs=True,  # Suppress verbose base environment logs
            solve_on_single_correct=False,  # Mark item as solved if any rollout gets it correct
        )
        # Server configurations can be similar to SingleToolCallingEnv or adjusted
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9004/v1",
                api_key="x",
                num_max_requests_at_once=32,
                num_requests_for_eval=256,
            )
        ]
        return env_config, server_configs

    async def create_rollout_table(self, wandb_metrics):
        # Logs rollouts to a WandB table for visualization
        if len(self.rollouts_for_wandb) > 0:
            table = wandb.Table(columns=["text", "score", "constraint_details"])
            for group in self.rollouts_for_wandb:
                for item in group:
                    # item[0] is model output, item[1] is score, item[2] is constraint info
                    table.add_data(item[0], item[1], json.dumps(item[2]))
            wandb_metrics["train/rollouts"] = table
        self.rollouts_for_wandb = []
        return wandb_metrics

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        # Logs metrics to WandB
        if wandb_metrics is None:
            wandb_metrics = dict()

        try:
            wandb_metrics["train/percent_correct"] = sum(
                self.percent_correct_buffer
            ) / len(self.percent_correct_buffer)
        except ZeroDivisionError:
            pass  # Buffer might be empty

        # Add adaptive curriculum metrics
        total_items = len(self.active_train_queue) + len(self.solved_items)
        if total_items > 0:
            wandb_metrics["curriculum/active_items"] = len(self.active_train_queue)
            wandb_metrics["curriculum/solved_items"] = len(self.solved_items)
            wandb_metrics["curriculum/percent_solved"] = (
                len(self.solved_items) / total_items
            )
            wandb_metrics["curriculum/total_items"] = total_items

            # Average attempt count for items still in circulation
            if self.item_attempt_counts:
                active_attempts = [
                    count
                    for item_id, count in self.item_attempt_counts.items()
                    if any(
                        f"{item['func_name']}_{hash(str(item)) % 100000}" == item_id
                        for item in self.active_train_queue
                    )
                ]
                if active_attempts:
                    wandb_metrics["curriculum/avg_attempts_active"] = sum(
                        active_attempts
                    ) / len(active_attempts)

        self.percent_correct_buffer = list()
        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = list()
        await super().wandb_log(wandb_metrics)

    async def setup(self):
        """
        Load and preprocess the dataset for instruction following.
        This method is specifically tailored to process 'allenai/RLVR-IFeval' dataset structure.
        Each item from RLVR-IFeval is expected to have:
        - 'messages': A list of dictionaries, e.g., [{'role': 'user', 'content': 'instruction...'}]
        - 'ground_truth': A JSON string containing 'func_name' and arguments for the verifier.

        The method will parse these to produce items for the environment with:
        - 'prompt': The user's instruction string.
        - 'func_name': The string name of the verifier function.
        - 'args': A dictionary of arguments for that verifier function.
        """  # noqa: E501
        dataset_name = getattr(self.config, "dataset_name", "allenai/RLVR-IFeval")
        dataset_config_name = getattr(
            self.config, "dataset_config_name", None
        )  # Default is None, RLVR-IFeval has no sub-config

        processed_items = []
        try:
            print(
                f"Attempting to load dataset: {dataset_name}, "
                f"config: {dataset_config_name if dataset_config_name else 'default'}"
            )
            if dataset_config_name:
                full_dataset_raw = load_dataset(
                    dataset_name,
                    dataset_config_name,
                    split="train",
                    trust_remote_code=True,
                )
            else:
                full_dataset_raw = load_dataset(
                    dataset_name, split="train", trust_remote_code=True
                )
            print(
                f"Successfully loaded raw dataset. Number of items: {len(full_dataset_raw)}"
            )

            for i, item in enumerate(full_dataset_raw):
                # Extract prompt from 'messages' field
                item_messages = item.get("messages")
                if (
                    not item_messages
                    or not isinstance(item_messages, list)
                    or len(item_messages) == 0
                ):
                    print(
                        f"Warning: Item {i} has invalid or empty 'messages' field. Skipping. Item: {item}"
                    )
                    continue
                # Assuming the relevant prompt is the content of the first message in the list
                # (or last, if multiple user messages were possible, but IFEval is typically single user instruction)
                prompt_text = item_messages[0].get("content")
                if not prompt_text:
                    print(
                        f"Warning: Item {i} '{item_messages[0]}' has no content. Skipping."
                    )
                    continue

                # Get the ground_truth JSON string
                ground_truth_json_str = item.get("ground_truth")
                if not ground_truth_json_str or not isinstance(
                    ground_truth_json_str, str
                ):
                    print(
                        f"Warning: Item {i} missing or has invalid 'ground_truth' string. Skipping. "
                        f"Prompt: {prompt_text[:50]}..."
                    )
                    continue

                try:
                    parsed_gt = json.loads(ground_truth_json_str)
                    if not isinstance(parsed_gt, dict):
                        raise ValueError("Parsed ground_truth is not a dictionary.")
                except (json.JSONDecodeError, ValueError) as e:
                    print(
                        f"Warning: Could not parse 'ground_truth' JSON for item {i}. Error: {e}. "
                        f"GT String: '{ground_truth_json_str}'. Prompt: {prompt_text[:50]}... Skipping."
                    )
                    continue

                func_name_from_gt = parsed_gt.get("func_name")
                if not func_name_from_gt:
                    print(
                        f"Warning: Item {i} parsed 'ground_truth' has no 'func_name'. GT: {parsed_gt}. "
                        f"Prompt: {prompt_text[:50]}... Skipping."
                    )
                    continue

                if func_name_from_gt not in IF_FUNCTIONS_MAP:
                    print(
                        f"Warning: func_name '{func_name_from_gt}' in item {i} not in IF_FUNCTIONS_MAP. "
                        f"Prompt: {prompt_text[:50]}... Skipping."
                    )
                    continue

                # Prepare args for the verifier function: remove func_name and keep others.
                # Verifier functions will only use args they expect.
                args_dict = {
                    k: v
                    for k, v in parsed_gt.items()
                    if k != "func_name" and v is not None
                }

                processed_items.append(
                    {
                        "prompt": prompt_text,
                        "func_name": func_name_from_gt,
                        "args": args_dict,
                        "original_constraints_for_logging": str(
                            item.get("constraint", "")
                        ),  # For logging, from RLVR-IFeval structure
                        "expected_response_for_logging": "",
                    }
                )

            if not processed_items:
                print(
                    "Warning: No items successfully processed from the dataset. "
                    "Check dataset format/content or parsing logic."
                )
                raise ValueError(
                    "Dataset processing resulted in no valid items for RLVR-IFeval. Cannot proceed without data."
                )

            full_dataset = Dataset.from_list(processed_items)
            print(
                f"Successfully processed {len(full_dataset)} items from dataset '{dataset_name}'."
            )

        except Exception as e:
            # This block is a fallback if the primary dataset loading/processing fails catastrophically.
            # For RLVR-IFeval, a failure here suggests issues with Hugging Face access,
            # dataset integrity, or fundamental code errors.
            print(
                f"CRITICAL: Failed to load or process primary dataset '{dataset_name}': {e}. "
                f"Using a DUMMY dataset as fallback."
            )
            dummy_data_for_fallback = [
                {
                    "prompt": "Dummy Instruction 1: Ensure your response contains the word 'example'.",
                    "func_name": "verify_keywords",
                    "args": {"keyword_list": ["example"]},
                    "original_constraints_for_logging": "Contains 'example'",
                    "expected_response_for_logging": "This is an example response.",
                },
                {
                    "prompt": "Dummy Instruction 2: Output a valid JSON with key 'data' and value 'test'.",
                    "func_name": "validate_json_format",
                    "args": {},
                    "original_constraints_for_logging": "Output valid JSON.",
                    "expected_response_for_logging": '{\\"data\\": \\"test\\"}',
                },
            ]
            full_dataset = Dataset.from_list(dummy_data_for_fallback)
            print(
                f"Initialized with DUMMY dataset of {len(full_dataset)} items "
                f"due to previous errors."
            )

        full_dataset = full_dataset.shuffle(seed=self.config.dataset_shuffle_seed)

        actual_test_size = self.config.test_set_ratio  # Read from config
        num_items = len(full_dataset)

        if num_items == 0:
            print("ERROR: Dataset is empty. Cannot create train/test split.")
            self.train = Dataset.from_list([])
            self.test = Dataset.from_list([])
        elif num_items == 1:
            print("Warning: Dataset has only 1 item. Using it for both train and test.")
            self.train = full_dataset
            self.test = full_dataset
        else:  # num_items > 1
            # Ensure test_size results in at least 1 item for test set if possible, but not more than train set
            if num_items < 5:  # For 2,3,4 items, make test size 1
                min_test_items = 1
            else:  # For 5+ items, 20% is fine
                min_test_items = max(1, int(num_items * actual_test_size))

            # Ensure test split is not too large, e.g. not more than 50% unless dataset is very small
            # And ensure train always has at least one item if num_items > 1
            calculated_test_size = min_test_items / num_items
            if (
                calculated_test_size >= 0.5 and num_items > 2
            ):  # If test is 50% or more and we have 3+ items
                calculated_test_size = (
                    num_items - 1
                ) / num_items  # Make train have at least 1

            split_dataset = full_dataset.train_test_split(
                test_size=calculated_test_size, seed=42
            )
            self.train = split_dataset["train"]
            self.test = split_dataset["test"]
            # Final check for empty train/test after split, should not happen with logic above if num_items > 0
            if len(self.train) == 0 and len(self.test) > 0:
                print(
                    "Warning: Train set empty after split, test set has data. "
                    "This is unusual. Swapping."
                )
                self.train = self.test  # Fallback, though indicates issue
            elif len(self.test) == 0 and len(self.train) > 0:
                print(
                    "Warning: Test set empty after split, train set has data. "
                    "Using full train set for test as well."
                )
                self.test = self.train

        self.iter = 0

        # Initialize the adaptive curriculum queue
        if self.config.resume_from_unsolved_dataset:
            print(
                f"🔄 Resume mode: Loading unsolved items from {self.config.resume_from_unsolved_dataset}"
            )
            print(
                f"   Note: This will override the dataset_name '{self.config.dataset_name}' for training items"
            )
            print(f"   Test set will still use items from '{self.config.dataset_name}'")
            await self._load_from_unsolved_dataset()
        else:
            # Initialize with all training items
            self.active_train_queue = list(self.train)
            self.solved_items = []
            self.item_attempt_counts = {}

        print(
            f"Dataset setup complete. Train size: {len(self.train)}, Test size: {len(self.test)}"
        )
        print(
            f"Adaptive curriculum initialized with {len(self.active_train_queue)} items in active queue"
        )

    def _validate_config(self):
        """Validate configuration for potential conflicts and warn user."""
        if self.config.resume_from_unsolved_dataset and self.config.dataset_name:
            print("⚠️  Configuration Notice:")
            print(
                f"   Both 'dataset_name' ({self.config.dataset_name}) and 'resume_from_unsolved_dataset' are set"
            )
            print("   Behavior:")
            print(
                "   - Training items: Will come from the resume file (overrides dataset_name)"
            )
            print("   - Test/eval items: Will come from dataset_name")
            print(
                "   - This is useful for resuming training while keeping consistent evaluation"
            )
            print()

    async def _get_score_from_verifier(
        self, model_response_text: str, func_name: str, args: Dict
    ) -> float:
        """Helper to call verifier function and get a numerical score.
        Also enforces strict <think>...</think> formatting.
        """

        # 1. Count <think> and </think> tags
        num_think_open = len(re.findall(r"<think>", model_response_text, re.IGNORECASE))
        num_think_close = len(
            re.findall(r"</think>", model_response_text, re.IGNORECASE)
        )

        if not (num_think_open == 1 and num_think_close == 1):
            return 0.0

        # 3. Find the first occurrence of <think> and </think>
        try:
            think_open_match = re.search(r"<think>", model_response_text, re.IGNORECASE)
            think_close_match = re.search(
                r"</think>", model_response_text, re.IGNORECASE
            )

            # These should exist due to the count check, but access .start() and .end() safely
            idx_think_open = think_open_match.start()
            idx_think_close_start = think_close_match.start()
            idx_think_close_end = think_close_match.end()

        except AttributeError:
            return 0.0

        # 4. If <think> appears after </think>, malformed.
        if idx_think_open >= idx_think_close_start:
            # print(f"DEBUG: <think> tag appears at or after </think> tag. Response: '{model_response_text[:200]}...'")
            return 0.0

        # 5. Extract text_to_verify (content after the first </think>)
        text_to_verify = model_response_text[idx_think_close_end:].strip()

        # 6. Check if text_to_verify itself contains any further <think> or </think> tags.
        if re.search(r"<think>", text_to_verify, re.IGNORECASE) or re.search(
            r"</think>", text_to_verify, re.IGNORECASE
        ):
            return 0.0

        # If all checks pass, proceed with verification using text_to_verify
        if func_name not in IF_FUNCTIONS_MAP:
            print(
                f"Warning: Verifier function '{func_name}' not found in IF_FUNCTIONS_MAP."
            )
            return 0.0

        verifier_func = IF_FUNCTIONS_MAP[func_name]

        raw_score = None
        try:
            if func_name == "validate_placeholders":
                raw_score = verifier_func(text_to_verify, N=args.get("N"))
            elif func_name == "verify_bullet_points":
                raw_score = verifier_func(text_to_verify, N=args.get("N"))
            elif func_name == "validate_repeat_prompt":
                raw_score = verifier_func(
                    text_to_verify, args.get("original_prompt", "")
                )
            else:
                from inspect import signature

                sig = signature(verifier_func)
                valid_params = [p for p in sig.parameters if p != "text"]
                filtered_args = {
                    k: args[k]
                    for k in valid_params
                    if k in args and args[k] is not None
                }
                raw_score = verifier_func(text_to_verify, **filtered_args)

        except LangDetectException:
            print(
                f"Warning: langdetect failed for func_name '{func_name}'. Scoring as incorrect."
            )
            return 0.0
        except ImportError as e:
            print(
                f"Warning: ImportError during verifier function '{func_name}': {e}. Check dependencies."
            )
            return 0.0
        except TypeError as e:
            print(
                f"TypeError calling {func_name} with args {args}: {e}. Text: '{text_to_verify[:100]}...'"
            )
            return 0.0
        except Exception as e:
            print(
                f"Unexpected error in verifier function '{func_name}' with args {args}: {e}"
            )
            return 0.0

        if isinstance(raw_score, tuple):
            score_value = float(raw_score[0])
        elif isinstance(raw_score, bool):
            score_value = float(raw_score)
        else:
            print(
                f"Warning: Verifier '{func_name}' returned unexpected type: {type(raw_score)}. Expected bool or tuple."
            )
            score_value = 0.0

        return score_value

    async def rollout_and_score_eval(self, test_item: Dict):
        # test_item is a dictionary from the test set, processed by setup()
        # It should contain 'prompt', 'func_name', 'args'
        instruction_prompt_text = test_item["prompt"]
        func_name = test_item["func_name"]
        args_for_verifier = test_item["args"]

        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": instruction_prompt_text})

        prompt_str = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        completion = await self.server.completion(
            prompt=prompt_str,
            n=1,
            max_tokens=self.config.max_token_length,  # Use config for max_tokens
            temperature=0.2,  # Temperature for eval, can be 0 for deterministic
            split="eval",
        )

        model_response_text = completion.choices[0].text
        score_value = await self._get_score_from_verifier(
            model_response_text, func_name, args_for_verifier
        )

        return (
            score_value  # Returns 1.0 for correct, 0.0 for incorrect based on verifier
        )

    async def evaluate(self, *args, **kwargs):
        # Evaluates the model on the test set
        if not self.test or len(self.test) == 0:
            print("Warning: Test set is empty. Skipping evaluation.")
            self.eval_metrics.append(("eval/percent_correct", 0.0))
            return

        print(f"Starting evaluation on {len(self.test)} items...")
        eval_tasks = []
        for test_item_dict in self.test:  # self.test contains dicts after setup
            eval_tasks.append(self.rollout_and_score_eval(test_item_dict))

        scores = await tqdm_asyncio.gather(*eval_tasks, desc="Evaluating")

        if not scores:  # If gather returns empty list
            percent_correct = 0.0
        else:
            percent_correct = sum(scores) / len(scores)

        self.eval_metrics.append(("eval/percent_correct", percent_correct))
        print(f"Evaluation finished. Percent correct: {percent_correct:.4f}")

    async def collect_trajectories(
        self, item: Item
    ) -> Tuple[Optional[ScoredDataGroup], List]:
        # item = (prompt_messages_tuple, answer_info_dict)
        # answer_info_dict = {"func_name": ..., "args": ...}
        prompt_messages_list = [dict(msg_fset) for msg_fset in item[0]]
        answer_info = item[1]

        prompt_str = self.tokenizer.apply_chat_template(
            prompt_messages_list, add_generation_prompt=True, tokenize=False
        )

        try:
            completions = await self.server.completion(
                prompt=prompt_str,
                n=self.config.group_size,
                max_tokens=self.config.max_token_length,
                temperature=0.8,  # Temperature for diverse responses during training rollouts
            )
        except Exception as e:
            print(f"ERROR: Exception during completion generation: {e}")
            return None, []

        to_score_list = []
        for choice in completions.choices:
            trajectory_messages = [dict(msg_fset) for msg_fset in item[0]]  # Fresh copy
            trajectory_messages.append({"role": "assistant", "content": choice.text})
            to_score_list.append(
                (tuple(trajectory_messages), answer_info)
            )  # Pass answer_info

        if not to_score_list:
            return None, []

        scored_data = await self.score(to_score_list)

        # Handle adaptive curriculum: decide whether to keep item in circulation
        if scored_data and scored_data.get("scores"):
            group_average_score = sum(scored_data["scores"]) / len(
                scored_data["scores"]
            )
            self._handle_item_result(item, group_average_score, scored_data["scores"])
        elif scored_data is None:
            # If scored_data is None, it might be because the group was skipped for being too easy
            # We need to calculate the scores ourselves to handle the item properly
            temp_scores = []
            for trajectory_messages, answer_info in to_score_list:
                model_response_text = trajectory_messages[-1]["content"]
                func_name = answer_info["func_name"]
                args_for_verifier = answer_info["args"]

                # Get score (1.0 for correct, 0.0 for incorrect from verifier)
                score_value = await self._get_score_from_verifier(
                    model_response_text, func_name, args_for_verifier
                )
                reward = 1.0 if score_value == 1.0 else 0
                temp_scores.append(reward)

            if temp_scores:
                group_average_score = sum(temp_scores) / len(temp_scores)
                self._handle_item_result(item, group_average_score, temp_scores)

        to_backlog = []  # Backlog not currently used but part of signature

        return scored_data, to_backlog

    def save_checkpoint(self, step, data=None):
        if data is None:
            data = {}
        data["iter"] = self.iter
        data["save_file_batch_num"] = self.save_file_batch_num
        data["failed_save_file_batch_num"] = self.failed_save_file_batch_num
        # Save adaptive curriculum state
        data["active_train_queue"] = self.active_train_queue
        data["solved_items"] = self.solved_items
        data["item_attempt_counts"] = self.item_attempt_counts
        super().save_checkpoint(step, data)

    async def close(self):
        """Save any remaining rollouts and curriculum state before closing."""
        if self.config.dump_rollouts and self.rollouts_to_save_buffer:
            print(
                f"Saving {len(self.rollouts_to_save_buffer)} remaining rollouts before closing..."
            )
            await self._save_rollouts_to_jsonl()

        if self.config.dump_failed_rollouts and self.failed_rollouts_to_save_buffer:
            print(
                f"Saving {len(self.failed_rollouts_to_save_buffer)} remaining failed rollouts before closing..."
            )
            await self._save_failed_rollouts_to_jsonl()

        # Save final curriculum state
        if self.active_train_queue:
            print(
                f"Saving final curriculum state with {len(self.active_train_queue)} unsolved items..."
            )
            await self._dump_active_queue_dataset()

        await super().close()

    async def score(
        self, rollout_group_data: List[Tuple[tuple, Dict]]
    ) -> Optional[ScoredDataGroup]:
        # rollout_group_data is a list of (trajectory_messages_tuple, answer_info_dict)
        # answer_info_dict = {"func_name": ..., "args": ...}

        scores_container = ScoredDataGroup()
        scores_container["tokens"] = list()
        scores_container["masks"] = list()
        scores_container["scores"] = list()

        if not rollout_group_data:
            return None

        # The 'answer_info' (func_name, args) is consistent for all items in this group,
        # as it comes from the same initial prompt.
        # We can extract it once if needed, but it's passed per item.

        random.shuffle(rollout_group_data)  # Shuffle to avoid bias

        # Data dumping: collect rollouts for saving (group format)
        rollouts_for_this_group = []
        failed_rollouts_for_this_group = []

        for trajectory_item in rollout_group_data:
            full_trajectory_messages = trajectory_item[0]
            answer_info = trajectory_item[1]  # {"func_name": ..., "args": ...}

            model_response_text = full_trajectory_messages[-1]["content"]
            func_name = answer_info["func_name"]
            args_for_verifier = answer_info["args"]

            # Get score (1.0 for correct, 0.0 for incorrect from verifier)
            score_value = await self._get_score_from_verifier(
                model_response_text, func_name, args_for_verifier
            )

            # Map to reward: 1.0 for correct, 0 for incorrect
            reward = 1.0 if score_value == 1.0 else 0

            # Prepare structured conversation for data dumping
            conversation = [dict(msg) for msg in full_trajectory_messages]

            # Create rollout dict for this specific rollout in the group
            rollout_dict = {
                "conversation": conversation,
                "score": reward,
            }

            # Collect rollouts for this group based on score and config
            if (
                self.config.dump_rollouts
                and reward >= self.config.rollout_save_score_threshold
            ):
                rollouts_for_this_group.append(rollout_dict)
            elif self.config.dump_failed_rollouts and reward == 0:
                failed_rollouts_for_this_group.append(rollout_dict)

            # Tokenize the conversation for PPO training
            # Ensure full_trajectory_messages is a list of dicts
            list_of_dicts_trajectory = [dict(msg) for msg in full_trajectory_messages]
            out_dict = tokenize_for_trainer(self.tokenizer, list_of_dicts_trajectory)
            tokens = out_dict["tokens"]
            masks = out_dict["masks"]

            # Filter out examples with insufficient context (too short)
            if (
                sum(1 for m_val in masks if m_val != -100) < 10
            ):  # At least 10 non-masked tokens
                continue

            scores_container["tokens"].append(tokens)
            scores_container["masks"].append(masks)
            scores_container["scores"].append(reward)

            # Stop if we have enough examples for the group
            if len(scores_container["tokens"]) >= self.config.group_size:
                break

        if not scores_container["tokens"]:  # No valid items collected
            return None

        # Calculate group average score for difficulty filtering and logging
        current_scores = scores_container.get("scores", [])
        if current_scores:
            average_score = sum(current_scores) / len(current_scores)
            # Get task info from the first rollout's answer_info
            answer_info = rollout_group_data[0][1] if rollout_group_data else {}
            func_name = answer_info.get("func_name", "unknown_task")

            # Check if group is too easy for training (but still allow data dumping)
            if average_score > self.config.max_group_average_for_training:
                print(
                    f"Task: {func_name} | Group average score: {average_score:.4f} (SKIPPED - too easy for training, threshold: {self.config.max_group_average_for_training})"  # noqa
                )

                # Still handle data dumping for groups that are too easy for training
                # but might be useful for analysis
                if (
                    rollouts_for_this_group
                    and average_score
                    <= self.config.max_group_average_for_training + 0.1
                ):  # Small buffer for data collection
                    # Extract item info for the group - get from first rollout's answer_info
                    answer_info = rollout_group_data[0][1]
                    item_id = f"allenai_RLVR-IFeval_train_item_{answer_info.get('func_name', 'unknown')}_{hash(str(answer_info)) % 100000}"  # noqa

                    group_data_to_save = {
                        "item_id": item_id,
                        "rollouts": rollouts_for_this_group,
                        "constraint_details": answer_info,  # Store group-level metadata
                        "group_average_score": average_score,  # Add group average for analysis
                        "skipped_for_training": True,  # Mark as skipped for training
                    }
                    self.rollouts_to_save_buffer.append(group_data_to_save)

                if failed_rollouts_for_this_group:
                    # Extract item info for the failed group
                    answer_info = rollout_group_data[0][1]
                    item_id = f"allenai_RLVR-IFeval_train_item_{answer_info.get('func_name', 'unknown')}_{hash(str(answer_info)) % 100000}"  # noqa

                    failed_group_data_to_save = {
                        "item_id": item_id,
                        "rollouts": failed_rollouts_for_this_group,
                        "constraint_details": answer_info,  # Store group-level metadata
                        "group_average_score": average_score,  # Add group average for analysis
                        "skipped_for_training": True,  # Mark as skipped for training
                    }
                    self.failed_rollouts_to_save_buffer.append(
                        failed_group_data_to_save
                    )

                # Save rollouts if buffer is getting large (batch processing)
                if (
                    self.config.dump_rollouts
                    and len(self.rollouts_to_save_buffer) >= 100
                ):
                    await self._save_rollouts_to_jsonl()
                if (
                    self.config.dump_failed_rollouts
                    and len(self.failed_rollouts_to_save_buffer) >= 50
                ):
                    await self._save_failed_rollouts_to_jsonl()

                return None  # Skip this group for training

            log_message = (
                f"Task: {func_name} | Group average score: {average_score:.4f}"
            )
            if all(s >= 0.5 for s in current_scores):
                print(f"{log_message} (All correct in this group!)")
            elif all(s == 0.0 for s in current_scores):
                print(f"{log_message} (All failed - format/constraint violations!)")
            elif all(s < 0.5 for s in current_scores):
                print(f"{log_message} (All incorrect but some partial credit!)")
            else:
                print(log_message)

        # Create group data structure and add to buffers for data dumping (for training groups)
        if rollouts_for_this_group:
            # Extract item info for the group - get from first rollout's answer_info
            answer_info = rollout_group_data[0][1]
            item_id = f"allenai_RLVR-IFeval_train_item_{answer_info.get('func_name', 'unknown')}_{hash(str(answer_info)) % 100000}"  # noqa

            group_data_to_save = {
                "item_id": item_id,
                "rollouts": rollouts_for_this_group,
                "constraint_details": answer_info,  # Store group-level metadata
                "group_average_score": (
                    current_scores[0]
                    if len(current_scores) == 1
                    else sum(current_scores) / len(current_scores)
                ),  # Add group average for analysis
                "skipped_for_training": False,  # Mark as used for training
            }
            self.rollouts_to_save_buffer.append(group_data_to_save)

        if failed_rollouts_for_this_group:
            # Extract item info for the failed group
            answer_info = rollout_group_data[0][1]
            item_id = f"allenai_RLVR-IFeval_train_item_{answer_info.get('func_name', 'unknown')}_{hash(str(answer_info)) % 100000}"  # noqa

            failed_group_data_to_save = {
                "item_id": item_id,
                "rollouts": failed_rollouts_for_this_group,
                "constraint_details": answer_info,  # Store group-level metadata
                "group_average_score": (
                    current_scores[0]
                    if len(current_scores) == 1
                    else sum(current_scores) / len(current_scores)
                ),  # Add group average for analysis
                "skipped_for_training": False,  # Mark as used for training
            }
            self.failed_rollouts_to_save_buffer.append(failed_group_data_to_save)

        # Save rollouts if buffer is getting large (batch processing)
        if self.config.dump_rollouts and len(self.rollouts_to_save_buffer) >= 100:
            await self._save_rollouts_to_jsonl()
        if (
            self.config.dump_failed_rollouts
            and len(self.failed_rollouts_to_save_buffer) >= 50
        ):
            await self._save_failed_rollouts_to_jsonl()

        # Record success rate for logging (based on positive rewards)
        for rwd in scores_container["scores"]:
            self.percent_correct_buffer.append(
                max(0, rwd)
            )  # If reward is 1.0, it's a success

        # Optional: Apply length penalty if all responses are correct (reward 1.0)
        # This logic is from SingleToolCallingEnv, may need adjustment for IF
        if all(s == 1.0 for s in scores_container["scores"]):
            token_lengths = [len(t) for t in scores_container["tokens"]]
            if not token_lengths or max(token_lengths) == 0:
                return scores_container  # Avoid division by zero, or if all empty

            max_allowed_length = self.config.max_token_length
            # Threshold can be adjusted, e.g., 75% of max_token_length
            length_threshold = max_allowed_length * 0.75

            penalized_scores = []
            for i, length in enumerate(token_lengths):
                original_score = scores_container["scores"][i]  # Should be 1.0 here
                if length <= length_threshold:
                    penalized_scores.append(original_score)
                else:
                    # Linear penalty for exceeding threshold
                    penalty_factor = (length - length_threshold) / (
                        max_allowed_length - length_threshold
                    )
                    penalty_factor = min(penalty_factor, 1.0)  # Cap penalty factor at 1
                    # Penalized score scales from original_score down to original_score * (1-1) = 0
                    penalized_scores.append(original_score * (1.0 - penalty_factor))
            scores_container["scores"] = penalized_scores

        # If all scores are identical after potential penalties, no learning signal
        if (
            len(set(scores_container["scores"])) <= 1
            and len(scores_container["scores"]) > 1
        ):
            return None  # Avoid sending data with no variance

        return scores_container

    async def get_next_item(self) -> Item:
        # Fetches the next item from the adaptive curriculum queue
        if not self.active_train_queue:
            # If active queue is empty, check if we have any items left
            if not self.solved_items:
                print("Error: No training data available in get_next_item.")
                # Return a dummy item to prevent crashes
                dummy_prompt_messages = (
                    frozenset({"role": "system", "content": system_prompt}.items()),
                    frozenset(
                        {
                            "role": "user",
                            "content": "Dummy instruction: say hello.",
                        }.items()
                    ),
                )
                dummy_answer_info = {
                    "func_name": "verify_keywords",
                    "args": {"keyword_list": ["hello"]},
                }
                return (dummy_prompt_messages, dummy_answer_info)
            else:
                # All items have been solved! Reset the queue with solved items for continued training
                print(
                    f"🎉 All {len(self.solved_items)} items have been solved! Resetting queue for continued training..."
                )
                self.active_train_queue = list(self.solved_items)
                self.solved_items = []
                # Reset attempt counts for the new cycle
                self.item_attempt_counts = {}

        # Get the next item from the front of the active queue
        raw_item = self.active_train_queue.pop(0)
        self.iter += 1

        # Create a unique identifier for this item for tracking
        item_id = f"{raw_item['func_name']}_{hash(str(raw_item)) % 100000}"

        # Track attempt count
        if item_id not in self.item_attempt_counts:
            self.item_attempt_counts[item_id] = 0
        self.item_attempt_counts[item_id] += 1

        instruction_prompt_text = raw_item["prompt"]

        # Construct messages for the LLM (prompt tuple part of Item)
        # Using frozenset as required by BaseEnv's Item type hint
        prompt_messages_tuple = (
            frozenset({"role": "system", "content": system_prompt}.items()),
            frozenset({"role": "user", "content": instruction_prompt_text}.items()),
        )

        # The "answer" part for scoring purposes (answer_info dict part of Item)
        answer_info = {
            "func_name": raw_item["func_name"],
            "args": raw_item["args"],
            # Add item tracking info
            "item_id": item_id,
            "raw_item": raw_item,  # Store the full item for queue management
            "attempt_count": self.item_attempt_counts[item_id],
            # Optionally include other info for logging/debugging if needed from raw_item
            "original_constraints_for_logging": raw_item.get(
                "original_constraints", ""
            ),
            "expected_response_for_logging": raw_item.get(
                "expected_response_for_logging", ""
            ),
        }

        # Dump active queue every 100 iterations for resumability
        if self.iter % 100 == 0 and self.iter > 0:
            await self._dump_active_queue_dataset()

        return (prompt_messages_tuple, answer_info)

    async def _dump_active_queue_dataset(self):
        """
        Dumps the current active queue to a JSONL file so training can be resumed
        from the unsolved items if the environment is shut down.
        """
        if not self.active_train_queue:
            print("No active items to dump - all items have been solved!")
            return

        try:
            if not os.path.exists(self.datasets_dir):
                os.makedirs(self.datasets_dir)
                print(f"Created datasets directory: {self.datasets_dir}")
        except Exception as e:
            print(f"Error creating datasets directory {self.datasets_dir}: {e}")
            return

        filename = os.path.join(self.datasets_dir, "remaining_unsolved.jsonl")

        try:
            with open(filename, "w", encoding="utf-8") as f:
                for item in self.active_train_queue:
                    # Add metadata about the current state
                    item_with_metadata = dict(item)
                    item_id = f"{item['func_name']}_{hash(str(item)) % 100000}"
                    item_with_metadata["_curriculum_metadata"] = {
                        "item_id": item_id,
                        "attempt_count": self.item_attempt_counts.get(item_id, 0),
                        "queue_position": self.active_train_queue.index(item),
                        "total_active": len(self.active_train_queue),
                        "total_solved": len(self.solved_items),
                        "iteration_dumped": self.iter,
                    }
                    json.dump(item_with_metadata, f, ensure_ascii=False)
                    f.write("\n")

            print(
                f"📁 Dumped {len(self.active_train_queue)} unsolved items to {filename} (iteration {self.iter})"
            )
            print(
                f"   Queue status: {len(self.active_train_queue)} active, {len(self.solved_items)} solved"
            )

        except Exception as e:
            print(f"Error dumping active queue to {filename}: {e}")

    async def _load_from_unsolved_dataset(self):
        """
        Load the active queue from a previously saved remaining_unsolved.jsonl file.
        This allows resuming training from where it left off.

        Note: When resuming, the training items come from the saved file, but the test set
        still comes from the current dataset_name configuration.
        """
        try:
            with open(
                self.config.resume_from_unsolved_dataset, "r", encoding="utf-8"
            ) as f:
                loaded_items = []
                loaded_attempt_counts = {}
                original_dataset_info = None

                for line in f:
                    item_data = json.loads(line.strip())

                    # Extract curriculum metadata if present
                    metadata = item_data.pop("_curriculum_metadata", {})
                    item_id = metadata.get("item_id")
                    attempt_count = metadata.get("attempt_count", 0)

                    # Store info about the original dataset for validation
                    if original_dataset_info is None and "iteration_dumped" in metadata:
                        original_dataset_info = {
                            "total_active_at_dump": metadata.get("total_active"),
                            "total_solved_at_dump": metadata.get("total_solved"),
                            "iteration_dumped": metadata.get("iteration_dumped"),
                        }

                    if item_id and attempt_count > 0:
                        loaded_attempt_counts[item_id] = attempt_count

                    # Validate that the item has the expected structure
                    required_fields = ["prompt", "func_name", "args"]
                    if not all(field in item_data for field in required_fields):
                        print(
                            f"Warning: Skipping malformed item missing required fields: {list(item_data.keys())}"
                        )
                        continue

                    loaded_items.append(item_data)

                if not loaded_items:
                    raise ValueError("No valid items found in resume file")

                self.active_train_queue = loaded_items
                self.solved_items = []  # Start with no solved items when resuming
                self.item_attempt_counts = loaded_attempt_counts

                print(
                    f"📂 Loaded {len(loaded_items)} unsolved items from {self.config.resume_from_unsolved_dataset}"
                )
                if loaded_attempt_counts:
                    avg_attempts = sum(loaded_attempt_counts.values()) / len(
                        loaded_attempt_counts
                    )
                    print(
                        f"   Restored attempt counts for {len(loaded_attempt_counts)} items (avg: {avg_attempts:.1f} attempts)"  # noqa
                    )

                if original_dataset_info:
                    print(
                        f"   Original dump info: {original_dataset_info['total_solved_at_dump']} solved, "
                        f"{original_dataset_info['total_active_at_dump']} active at iteration {original_dataset_info['iteration_dumped']}"  # noqa
                    )

                # Validate compatibility with current dataset
                if hasattr(self, "train") and len(self.train) > 0:
                    original_total = original_dataset_info.get(
                        "total_active_at_dump", 0
                    ) + original_dataset_info.get("total_solved_at_dump", 0)
                    current_total = len(self.train)

                    if original_total != current_total:
                        print("⚠️  Warning: Dataset size mismatch!")
                        print(
                            f"   Original dataset had {original_total} items, current dataset has {current_total} items"  # noqa
                        )
                        print(
                            "   This might indicate different dataset versions or configurations"
                        )

        except FileNotFoundError:
            print(
                f"❌ Resume file not found: {self.config.resume_from_unsolved_dataset}"
            )
            print("Falling back to full dataset initialization...")
            self.active_train_queue = list(self.train)
            self.solved_items = []
            self.item_attempt_counts = {}
        except Exception as e:
            print(f"❌ Error loading from unsolved dataset: {e}")
            print("Falling back to full dataset initialization...")
            self.active_train_queue = list(self.train)
            self.solved_items = []
            self.item_attempt_counts = {}

    def _handle_item_result(
        self, item: Item, group_average_score: float, group_scores: List[float] = None
    ):
        """
        Handle the result of an item based on its group average score and individual scores.
        If solved (high score), remove from circulation.
        If not solved (low score), add back to the end of the queue.
        """
        _, answer_info = item
        raw_item = answer_info.get("raw_item")
        item_id = answer_info.get("item_id")
        attempt_count = answer_info.get("attempt_count", 1)

        if not raw_item or not item_id:
            return  # Skip if we don't have the necessary info

        # Define "solved" based on configuration options
        is_solved = False
        solve_reason = ""

        # Check if solved based on single correct rollout
        if self.config.solve_on_single_correct and group_scores:
            if any(score >= 1.0 for score in group_scores):
                is_solved = True
                solve_reason = " (single correct)"

        # Check if solved based on group average thresholds (original logic)
        if not is_solved:
            if group_average_score > self.config.max_group_average_for_training:
                is_solved = True
                solve_reason = " (too easy)"
            elif group_average_score >= 0.9:  # Very high performance threshold
                is_solved = True
                solve_reason = " (mastered)"

        if is_solved:
            # Item is solved - move to solved items (remove from circulation)
            self.solved_items.append(raw_item)
            status = f"SOLVED - removed from circulation{solve_reason}"
        else:
            # Item not solved - add back to the end of the active queue
            self.active_train_queue.append(raw_item)
            status = "NOT SOLVED - added back to queue"

        # Log the decision periodically or for items with many attempts
        if attempt_count % 5 == 1 or attempt_count <= 3 or is_solved:
            func_name = answer_info.get("func_name", "unknown")
            print(
                f"📚 Item {item_id} (attempt #{attempt_count}): {func_name} | Score: {group_average_score:.4f} | {status}"  # noqa
            )
            print(
                f"   Queue status: {len(self.active_train_queue)} active, {len(self.solved_items)} solved"
            )

    async def add_rollouts_for_wandb(
        self,
        scored_data: ScoredDataGroup,  # Assuming single ScoredDataGroup here
        item: Item = None,  # item = (prompt_messages_tuple, answer_info_dict)
    ):
        # Saves rollouts for WandB logging
        num_keep = self.config.num_rollouts_per_group_for_logging
        if num_keep == -1:  # Log all rollouts in the group
            num_keep = len(scored_data["tokens"])

        # item[1] is the answer_info_dict containing func_name and args
        constraint_details_for_log = item[1] if item else {}

        rollout_batch = []
        for i in range(min(num_keep, len(scored_data["tokens"]))):
            decoded_text = self.tokenizer.decode(
                scored_data["tokens"][i], skip_special_tokens=False
            )
            score = scored_data["scores"][i]
            rollout_batch.append((decoded_text, score, constraint_details_for_log))

        self.rollouts_for_wandb.append(rollout_batch)

        # Limit the number of rollout groups stored
        if len(self.rollouts_for_wandb) > self.config.num_rollouts_to_keep:
            self.rollouts_for_wandb.pop(0)

    async def _save_rollouts_to_jsonl(self):
        """Saves the buffered rollouts to a JSONL file in the datadumps directory."""
        if not self.rollouts_to_save_buffer:
            print("Warning: _save_rollouts_to_jsonl called but buffer is empty!")
            return

        buffer_size = len(self.rollouts_to_save_buffer)
        print(f"Starting save of {buffer_size} rollout groups to JSONL file...")

        try:
            if not os.path.exists(self.datadumps_dir):
                os.makedirs(self.datadumps_dir)
                print(f"Created directory: {self.datadumps_dir}")
        except Exception as e:
            print(f"Error creating directory {self.datadumps_dir}: {e}")
            return

        filename = os.path.join(
            self.datadumps_dir,
            f"instruction_following_rollouts_{self.run_uuid}_{self.save_file_batch_num:04d}.jsonl",
        )

        try:
            with open(filename, "w", encoding="utf-8") as f:
                for rollout_dict in self.rollouts_to_save_buffer:
                    json.dump(rollout_dict, f, ensure_ascii=False)
                    f.write("\n")

            print(f"Successfully saved {buffer_size} rollout groups to {filename}")
            self.save_file_batch_num += 1
            self.rollouts_to_save_buffer.clear()

        except Exception as e:
            print(f"Error saving rollouts to {filename}: {e}")

    async def _save_failed_rollouts_to_jsonl(self):
        """Saves the buffered failed rollouts to a JSONL file for debugging."""
        if not self.failed_rollouts_to_save_buffer:
            print("Warning: _save_failed_rollouts_to_jsonl called but buffer is empty!")
            return

        buffer_size = len(self.failed_rollouts_to_save_buffer)
        print(f"Starting save of {buffer_size} failed rollout groups to JSONL file...")

        try:
            if not os.path.exists(self.datadumps_dir):
                os.makedirs(self.datadumps_dir)
                print(f"Created directory: {self.datadumps_dir}")
        except Exception as e:
            print(f"Error creating directory {self.datadumps_dir}: {e}")
            return

        filename = os.path.join(
            self.datadumps_dir,
            f"instruction_following_failed_rollouts_{self.run_uuid}_{self.failed_save_file_batch_num:04d}.jsonl",
        )

        try:
            with open(filename, "w", encoding="utf-8") as f:
                for rollout_dict in self.failed_rollouts_to_save_buffer:
                    json.dump(rollout_dict, f, ensure_ascii=False)
                    f.write("\n")

            print(
                f"Successfully saved {buffer_size} failed rollout groups to {filename}"
            )
            self.failed_save_file_batch_num += 1
            self.failed_rollouts_to_save_buffer.clear()

        except Exception as e:
            print(f"Error saving failed rollouts to {filename}: {e}")


# ----- IFEval Verifier Functions and Map -----
# adapted from https://github.com/allenai/open-instruct/blob/main/scripts/eval_constraints/if_functions.py


# Helper function for verify_keyword_frequency, moved import re to top level
def _extract_words(text: str) -> List[str]:
    return re.findall(r"\\b\\w+\\b", text.lower())


# include keywords: Include keywords {keyword1}, {keyword2} in your response
def verify_keywords(text: str, keyword_list: List[str]) -> bool:
    response_lower = text.lower()
    return all(keyword.lower() in response_lower for keyword in keyword_list)


# Keyword Frequency: In your response, the word {word} should appear {N} times.
def verify_keyword_frequency(text: str, word: str, N: int) -> bool:
    text_lower = text.lower()
    keyword_lower = word.lower()
    words = _extract_words(text_lower)
    actual_count = sum(1 for w in words if w == keyword_lower)
    return actual_count == N


# Forbidden Words: Do not include keywords {forbidden words} in the response.
def validate_forbidden_words(text: str, forbidden_words: List[str]) -> bool:
    text_lower = text.lower()
    return not any(word.lower() in text_lower for word in forbidden_words)


# Letter Frequency : In your response, the letter {letter} should appear {N} times.
def verify_letter_frequency(text: str, letter: str, N: int) -> bool:
    if len(letter) != 1:
        # This should ideally raise ValueError, but for RL reward, return False
        return False
    actual_count = text.count(letter)
    return actual_count == N


# Response Language: Your ENTIRE response should be in {language}, no other language is allowed.
def validate_response_language(text: str, language: str) -> bool:
    try:
        detected_language = detect(text)
        return detected_language == language
    except LangDetectException:  # Catching specific exception from detect()
        print(
            f"Warning: langdetect failed to detect language for text: '{text[:50]}...'"
        )
        return False


# Number Paragraphs: Your response should contain {N} paragraphs. You separate paragraphs using the markdown divider:
# * * *
def verify_paragraph_count(text: str, N: int) -> bool:
    def clean_text(txt: str) -> str:
        return "\\n".join(line.strip() for line in txt.splitlines()).strip()

    cleaned_text = clean_text(text)
    # Paragraphs are separated by '* * *'. N dividers mean N+1 paragraphs.
    # If the text IS paragraphs, then N paragraphs will have N-1 dividers.
    # The prompt implies N paragraphs are expected.
    # If N=1, 0 dividers. If N=2, 1 divider. So, count of parts = N.
    paragraphs = cleaned_text.split("* * *")
    actual_count = len(paragraphs)
    # Verify each split resulted in non-empty content, if text itself is not empty
    if not cleaned_text and N == 0:
        return True  # 0 paragraphs, empty text
    if not cleaned_text and N > 0:
        return False

    # This check might be too strict if empty paragraphs are allowed by the constraint definition
    # If "paragraph" implies non-empty content:
    # return len(valid_paragraphs) == N and actual_count == N
    # If constraint just means N segments separated by dividers:
    return actual_count == N


# Number Words: Answer with at least / around / at most {N} words
def validate_word_constraint(text: str, N: int, quantifier: str) -> bool:
    words = text.strip().split()
    actual_count = len(words)
    tolerance = max(round(N * 0.1), 1)  # For 'around'

    if quantifier == "at least":
        return actual_count >= N
    elif quantifier == "at most":
        return actual_count <= N
    elif quantifier == "around":
        return abs(actual_count - N) <= tolerance
    return False


# Number Sentences: Answer with at least / around / at most {N} sentences.
def verify_sentence_constraint(text: str, N: int, quantifier: str) -> bool:
    # Basic sentence splitting, might need more robust NLP for complex cases
    sentences = re.split(
        r"(?<![a-zA-Z0-9_]\.[a-zA-Z0-9_]\.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s",
        text.strip(),
    )
    # Filter out empty strings that might result from splitting
    sentences = [s for s in sentences if s.strip()]
    actual_count = len(sentences)

    if quantifier == "at least":
        return actual_count >= N
    elif quantifier == "around":
        # "around" for sentences usually means exact or +/-1
        return abs(actual_count - N) <= 1
    elif quantifier == "at most":
        return actual_count <= N
    return False


# Number Paragraphs + First Word in i-th Paragraph
def validate_paragraphs(text: str, N: int, first_word: str, i: int) -> bool:
    # Paragraphs separated by double line breaks
    paragraphs = text.split("\\n\\n")
    if len(paragraphs) != N:
        return False
    # i is 1-indexed for paragraph number
    if not (1 <= i <= len(paragraphs)):
        return False
    # Check first word of the i-th paragraph
    # .strip() to handle leading/trailing whitespace in paragraph
    # .split()[0] to get the first word
    try:
        actual_first_word = paragraphs[i - 1].strip().split()[0]
        # Case-insensitive comparison for first_word might be more robust
        return actual_first_word.lower() == first_word.lower()
    except IndexError:  # Handles empty paragraph or paragraph without words
        return False


# Postscript: At the end of your response, please explicitly add a postscript starting with {postscript marker}
def verify_postscript(text: str, postscript_marker: str) -> bool:
    marker_index = text.rfind(postscript_marker)  # Find last occurrence
    if marker_index == -1:
        return False
    # Check if it's truly a postscript (i.e., near the end, and has content after marker)
    # This interpretation: marker exists, and something follows it OR it's at the very end.
    # The original IFEval might have a stricter definition (e.g. specific distance from end)
    # A simple check: marker is present and the text from marker to end is mostly the postscript.
    # For RL, simpler: marker is present and is not just prefix of a word.
    # Test if the marker is at a word boundary if it's not the start of the string
    if (
        marker_index > 0
        and text[marker_index - 1].isalnum()
        and postscript_marker[0].isalnum()
    ):
        # Avoid matching mid-word, e.g. "script" in "postscript" if marker is "script"
        # This check is heuristic. A regex with word boundaries might be better.
        pass  # Heuristic, might need refinement

    # Check if content exists after marker, or if marker itself is the end
    remaining_text = text[marker_index:].strip()
    return len(remaining_text) >= len(postscript_marker.strip())


# Number Placeholder: The response must contain at least {N} placeholders ... [address].
def validate_placeholders(text: str, N: int) -> Tuple[bool, List[str]]:
    placeholders_found = re.findall(r"\\[(.*?)\\]", text)  # Matches [content]
    return len(placeholders_found) >= N, placeholders_found


# Number Bullets: Your answer must contain exactly {N} bullet points. * This is a point.
def verify_bullet_points(
    text: str, N: int
) -> bool:  # Original had tuple[bool,str] in doc, bool in code
    lines = text.splitlines()
    # Markdown bullets usually start with '*', '-', or '+' followed by a space.
    bullet_points = [
        line.strip()
        for line in lines
        if re.match(r"^(\\s*)[\\*\\-\\+]\\s+", line.strip())
    ]
    return len(bullet_points) == N


# Title: Your answer must contain a title, wrapped in double angular brackets, such as <<poem of joy>>.
def validate_title(text: str) -> bool:
    return bool(re.search(r"<<(.*?)>>", text))


# Choose: From Answer with one of the following options: {options}
def validate_choice(text: str, options: List[str]) -> bool:
    # Assuming 'text' should be one of the 'options' exactly, or contain one of them.
    # The original prompt "Answer with one of..." implies the response *is* one of the options.
    # Case-insensitive comparison for robustness.
    text_cleaned = text.strip().lower()
    return any(text_cleaned == opt.strip().lower() for opt in options)


# Minimum Number Highlighted Section: Highlight at least {N} sections ... *highlighted section*
def validate_highlighted_sections(text: str, N: int) -> bool:
    # Markdown italics/bold *highlight* or **highlight**
    # This regex looks for single asterisks: *content*
    matches = re.findall(
        r"\*(.*?)(?<!\\)\*", text  # Ensure the closing * is not escaped
    )
    # Filter out empty matches or those that are just whitespace if needed.
    # matches = [m for m in matches if m.strip()]
    return len(matches) >= N


# Multiple Sections: Your response must have {N} sections. Mark ... with {section splitter} X.
def validate_sections(text: str, N: int, section_splitter: str) -> bool:
    # Example: section_splitter = "Section" -> "Section 1", "Section 2"
    # This implies the splitter itself might include a number or be just the prefix.
    # If splitter is "---", then text.split("---").
    # If splitter is "Topic X:", this is more complex.
    # Assuming a simple string split is intended by the original IFEval function.
    # The prompt phrasing "Mark the beginning of each section with {section splitter} X"
    # suggests counting occurrences of the splitter pattern.

    # If section_splitter is like "SECTION", we'd look for "SECTION 1", "SECTION 2", ...
    # This is hard to generalize perfectly without knowing how IFEval defines 'X'.
    # Simplest: count occurrences of the base splitter string.
    # sections = text.split(section_splitter)
    # num_sections = len(sections) -1 if sections[0].strip() == "" else len(sections)
    # A slightly more robust way for "Splitter X":
    # Count how many times "splitter" followed by something (like a number) appears.
    # Example: if splitter is "Chapter", we look for "Chapter 1", "Chapter ...".
    # This regex is a placeholder for more specific logic IFEval might use.

    # Let's use a simple count of the splitter string for now.
    # This might need to be adjusted based on IFEval's exact expectation for "X".
    # For "SECTION 1.", "SECTION 2.", if splitter is "SECTION ":
    actual_sections = len(
        re.findall(
            re.escape(section_splitter) + r"\\s*\\d*[:\\.\\s]", text, re.IGNORECASE
        )
    )

    # If N=0 and no splitters, it's true. If N>0 and no splitters, false.
    if N == 0:
        return actual_sections == 0
    return actual_sections == N


# JSON Format : Entire output should be wrapped in JSON format.
def validate_json_format(text: str) -> bool:
    try:
        json.loads(text.strip())  # .strip() to handle leading/trailing whitespace
        return True
    except json.JSONDecodeError:
        return False


# Repeat Prompt: First, repeat the request without change, then give your answer
def validate_repeat_prompt(text: str, original_prompt: str) -> bool:
    # Normalize whitespace for comparison robustness
    text_norm = " ".join(text.strip().split())
    original_prompt_norm = " ".join(original_prompt.strip().split())
    return text_norm.startswith(original_prompt_norm)


# Two Responses: Give two different responses. Separated by 6 asterisk symbols: ******.
def validate_two_responses(text: str) -> bool:
    if text.count("******") == 1:
        parts = text.split("******")
        if len(parts) == 2:
            # Check if parts are non-empty and different
            resp1 = parts[0].strip()
            resp2 = parts[1].strip()
            return bool(resp1 and resp2 and resp1 != resp2)
    return False


# All Uppercase: Your entire response should be in English, capital letters only.
def validate_uppercase(text: str) -> bool:
    # Check if it has letters and all letters are uppercase
    if not any(
        c.isalpha() for c in text
    ):  # No letters, technically not violating "all capital"
        return True  # Or False, depending on interpretation of "response"
    return text == text.upper()


# All Lowercase: Your entire response should be in English, and in all lowercase letters.
def validate_lowercase(text: str) -> bool:
    if not any(c.isalpha() for c in text):
        return True
    return text == text.lower()


# Frequency of All-capital Words
def validate_frequency_capital_words(text: str, N: int, quantifier: str) -> bool:
    # Words with all capital letters, e.g., "NASA", "AI". Min 2 chars to be a "word".
    capital_words = re.findall(r"\\b[A-Z]{2,}\\b", text)
    actual_count = len(capital_words)
    tolerance = max(round(N * 0.1), 1)  # For 'around'

    if quantifier == "at least":
        return actual_count >= N
    elif quantifier == "at most":
        return actual_count <= N
    elif (
        quantifier == "around"
    ):  # Using exact for 'around' with capital words unless specified
        return abs(actual_count - N) <= tolerance  # Or just actual_count == N
    return False


# End Checker: Finish your response with this exact phrase {end phrase}.
def validate_end(text: str, end_phrase: str) -> bool:
    # Normalize whitespace at the end of text for robustness
    return text.strip().endswith(end_phrase.strip())


# Quotation: Wrap your entire response with double quotation marks.
def validate_quotation(text: str) -> bool:
    stripped_text = text.strip()
    return stripped_text.startswith('"') and stripped_text.endswith('"')


# No Commas: In your entire response, refrain from the use of any commas.
def validate_no_commas(text: str) -> bool:
    return "," not in text


IF_FUNCTIONS_MAP = {
    "verify_keywords": verify_keywords,
    "verify_keyword_frequency": verify_keyword_frequency,
    "validate_forbidden_words": validate_forbidden_words,
    "verify_letter_frequency": verify_letter_frequency,
    "validate_response_language": validate_response_language,
    "verify_paragraph_count": verify_paragraph_count,
    "validate_word_constraint": validate_word_constraint,
    "verify_sentence_constraint": verify_sentence_constraint,
    "validate_paragraphs": validate_paragraphs,
    "verify_postscript": verify_postscript,
    "validate_placeholders": validate_placeholders,
    "verify_bullet_points": verify_bullet_points,
    "validate_title": validate_title,
    "validate_choice": validate_choice,
    "validate_highlighted_sections": validate_highlighted_sections,
    "validate_sections": validate_sections,
    "validate_json_format": validate_json_format,
    "validate_repeat_prompt": validate_repeat_prompt,
    "validate_two_responses": validate_two_responses,
    "validate_uppercase": validate_uppercase,
    "validate_lowercase": validate_lowercase,
    "validate_frequency_capital_words": validate_frequency_capital_words,
    "validate_end": validate_end,
    "validate_quotation": validate_quotation,
    "validate_no_commas": validate_no_commas,
}

if __name__ == "__main__":
    InstructionFollowingEnv.cli()
