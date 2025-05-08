import asyncio
import json
import logging
import os
import random
import string
import time
import uuid
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import aiohttp
import jsonlines
import numpy as np
import wandb
import yaml
from pydantic import BaseModel, Field
from pydantic_cli import Cmd, FailedExecutionException, run_and_exit
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import AutoTokenizer

from atroposlib.envs.constants import (
    ENV_NAMESPACE,
    NAMESPACE_SEP,
    OPENAI_NAMESPACE,
    SERVER_MANAGER_NAMESPACE,
)
from atroposlib.frontend.jsonl2html import generate_html
from atroposlib.type_definitions import UUID
from atroposlib.utils.cli import (
    adjust_model_defaults,
    extract_namespace,
    get_double_dash_flags,
    get_prefixed_pydantic_model,
    merge_dicts,
)
from atroposlib.utils.metrics import get_std_min_max_avg

from ..type_definitions import Item, Message
from .server_handling.server_manager import (
    OpenaiConfig,
    ServerBaseline,
    ServerManager,
    ServerManagerConfig,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ScoredDataGroup(TypedDict):
    tokens: List[List[int]]
    masks: List[List[int]]
    scores: List[float]
    advantages: Optional[List[List[float]]]
    ref_logprobs: Optional[List[List[float]]]
    messages: Optional[List[List[Message]]]
    group_overrides: Optional[Dict]
    overrides: Optional[List[Dict]]


class EvalHandlingEnum(Enum):
    """
    Enum for handling evals.
    """

    STOP_TRAIN = "STOP_TRAIN"
    LIMIT_TRAIN = "LIMIT_TRAIN"
    NONE = "NONE"


class BaseEnvConfig(BaseModel):
    """
    Basic env configuration.
    """

    group_size: int = Field(
        default=4, description="How many responses are grouped together for scoring"
    )
    max_num_workers: int = Field(
        default=-1,
        description="Maximum number of workers to use, -1 calculates from max_num_workers_per_node",
    )
    max_eval_workers: int = Field(
        default=16, description="Maximum number of workers to use for evaluation"
    )
    max_num_workers_per_node: int = Field(
        default=8, description="Maximum number of workers to use per node"
    )
    steps_per_eval: int = Field(
        default=100, description="Number of steps to take before evaluating"
    )
    max_token_length: int = Field(
        default=2048, description="Maximum token length used in generations"
    )
    eval_handling: EvalHandlingEnum = Field(
        default=EvalHandlingEnum.STOP_TRAIN, description="How to handle evaluations"
    )
    eval_limit_ratio: float = Field(
        default=0.5, description="Ratio of training workers to limit during evals"
    )
    inference_weight: float = Field(
        default=1.0,
        description="Inference weight, set to -1 to ignore it if you're doing something special here.",
    )
    batch_size: int = Field(
        default=-1,
        description="Batch size for training. Default is -1, typically set by environment-specific YAML configs or overridden by the trainer/server.",  # noqa: E501
    )
    max_batches_offpolicy: int = Field(
        default=3, description="Maximum number of batches to have in queue."
    )
    tokenizer_name: str = Field(
        default="NousResearch/DeepHermes-3-Llama-3-3B-Preview",
        description="Hugging Face tokenzer to use.",
    )
    use_wandb: bool = Field(default=True, description="Whether to use wandb")
    rollout_server_url: str = Field(
        default="http://localhost:8000", description="URL of the rollout server"
    )
    total_steps: int = Field(default=1000, description="Total number of steps to run")
    wandb_name: str | None = Field(
        default=None,
        description="Name to be grouped by in wandb",
    )
    num_rollouts_to_keep: int = Field(
        default=32, description="Number of rollouts to display on wandb"
    )
    num_rollouts_per_group_for_logging: int = Field(
        default=1,
        description="Number of rollouts per group to keep for logging. If -1, keep all rollouts",
    )
    ensure_scores_are_not_same: bool = Field(
        default=True,
        description="Ensure that the scores are not the same, should usually be True",
    )
    data_path_to_save_groups: Optional[str] = Field(
        default=None,
        description="Path to save the groups, if set, will write groups to this jsonl",
    )
    min_items_sent_before_logging: int = Field(
        default=2,
        description="Minimum number of items sent before logging, if 0 or less, logs every time",
    )
    include_messages: bool = Field(
        default=False,
        description="Whether to include messages in the output transmitted to the trainer",
    )


class BaseEnv(ABC):

    name = None
    env_config_cls = BaseEnvConfig

    def __init__(
        self,
        config: BaseEnvConfig,
        server_configs: Union[ServerBaseline, List[OpenaiConfig]],
        slurm=False,
        testing=False,
    ):
        self.items_sent_this_step = 0
        self.eval_runner = None  # type: Optional[asyncio.Task]
        self.workers_added_list = list()
        self.succeeded_task_duration = list()
        self.failed_task_duration = list()
        self.task_duration = list()
        self.mainloop_timings = list()
        self.task_successful = list()
        self.last_loop_time = None
        self.last_completed_item = None
        self.config = config
        self.server = ServerManager(server_configs, slurm=slurm, testing=testing)
        self.workers = set()
        self.eval_workers = set()
        self.backlog = []
        self.rollouts_for_wandb = []
        self.running_items: dict[UUID, Item] = dict()
        self.wandb_project = None
        self.wandb_group = None
        self.curr_step = 0
        self.max_token_len = -1
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        self.completion_lengths = []
        self.max_num_workers = config.max_num_workers
        if self.max_num_workers == -1:
            self.max_num_workers = config.max_num_workers_per_node * len(
                self.server.servers
            )
        self.wandb_prepend = None
        self.checkpoint_dir = ""
        self.checkpoint_interval = -1
        if self.config.data_path_to_save_groups is not None:

            Path(self.config.data_path_to_save_groups).parent.mkdir(
                parents=True, exist_ok=True
            )
            # Find a suitable filename by appending _1, _2, etc. if the file already exists
            original_path = self.config.data_path_to_save_groups
            counter = 1
            path_changed = False
            while os.path.exists(self.config.data_path_to_save_groups):
                path_obj = Path(original_path)
                self.config.data_path_to_save_groups = str(
                    path_obj.with_stem(f"{path_obj.stem}_{counter}")
                )
                counter += 1
                path_changed = True
            if path_changed:
                print(
                    f"Changed data path to {self.config.data_path_to_save_groups} because {original_path} already exists."  # noqa: E501
                )

            self.jsonl_writer = jsonlines.open(
                self.config.data_path_to_save_groups, "w"
            )  # type: jsonlines.Writer
        else:
            self.jsonl_writer = None

    @classmethod
    def config_init(
        cls, config_name: Optional[str] = None
    ) -> Tuple[BaseEnvConfig, Union[ServerBaseline, List[OpenaiConfig]]]:
        """
        Initialize the config. Can optionally specify a config file name.

        Args:
            config_name: Optional name of config file to load (without extension)

        Returns:
            Tuple of (env_config, server_configs)
        """
        return cls.env_config_cls(), ServerBaseline()

    async def collect_trajectory(self, item: Item) -> Tuple[Any | None, List[Item]]:
        raise NotImplementedError(
            "Handle env single method must be implemented in subclass "
        )

    async def collect_trajectories(self, item: Item) -> Tuple[
        Union[
            Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]], List[Any | None]
        ],
        List[Item],
    ]:
        """

        :param item:
        :return:
        """
        tasks = []
        for _ in range(self.config.group_size):
            tasks.append(self.collect_trajectory(item))
        results = await asyncio.gather(*tasks)
        backlog = []
        to_postprocess = []
        for result in results:
            if result[0] is not None:
                to_postprocess.append(result[0])
            backlog.extend(result[1])
        random.shuffle(backlog)
        return to_postprocess, backlog

    async def postprocess_histories(
        self,
        trajectories: Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]],
    ) -> Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]:
        """
        Postprocess the histories, this is called after the collect_trajectories method

        If you don't need to do anything to the trajectories, you may safely ignore this.

        :param trajectories:
        :return:
        """
        return trajectories

    @abstractmethod
    async def get_next_item(self) -> Item:
        """
        Get the next items to be rolled out
        """
        raise NotImplementedError(
            "Get_next_items method must be implemented in subclass "
        )

    @abstractmethod
    async def evaluate(self, *args, **kwargs):
        """
        Evaluate the environment, this is called every steps_per_eval steps

        Included here is an example on how to use eval workers to run a task.

        You may however do whatever you want in this method.

        :param args:
        :param kwargs:
        :return: None.
        """
        for data in ["my", "eval", "data"]:
            while len(self.eval_workers) >= self.config.max_eval_workers:
                await asyncio.sleep(0.1)
            worker = asyncio.create_task(asyncio.sleep(0.1))
            self.eval_workers.add(worker)
            worker.add_done_callback(self.eval_workers.discard)
        raise NotImplementedError("Evaluate method must be implemented in subclass ")

    def load_checkpoint(self):
        # check if file exists...
        ckpt_path = os.path.join(
            self.checkpoint_dir,
            "env_checkpoints",
            self.wandb_prepend,
            f"step-{self.curr_step}.json",
        )
        if os.path.exists(ckpt_path):
            with open(ckpt_path, "r") as f:
                data = json.load(f)
            # now load the data
            for key in data:
                setattr(self, key, data[key])

    def save_checkpoint(self, step, data=None):
        print(f"Saving checkpoint at step {step} with data {data}")
        if data is None:
            # Don't have anything to save, abort
            return
        # check if file exists...
        ckpt_dir = os.path.join(
            self.checkpoint_dir, "env_checkpoints", self.wandb_prepend
        )
        # create directory if necessary
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(
            self.checkpoint_dir,
            "env_checkpoints",
            self.wandb_prepend,
            f"step-{step}.json",
        )
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        with open(ckpt_path, "w") as f:
            json.dump(data, f)

    async def setup(self):
        """Setup the environment"""
        raise NotImplementedError("Setup method must be implemented in subclass")

    async def setup_wandb(self):
        if self.config.use_wandb:
            # Setup wandb getting the group and project via the server
            while self.wandb_project is None:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self.config.rollout_server_url}/wandb_info"
                    ) as resp:
                        data = await resp.json()
                        self.wandb_group = data["group"]
                        self.wandb_project = data["project"]
                if self.wandb_project is None:
                    await asyncio.sleep(1)
                else:
                    wandb.init(
                        project=self.wandb_project,
                        group=self.wandb_group,
                        config=self.config.model_dump(),
                    )
                    break

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=1, max=10),
    )
    async def _register_env(self):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config.rollout_server_url}/register-env",
                    json={
                        "max_token_length": self.config.max_token_length,
                        "desired_name": self.config.wandb_name,
                        "weight": self.config.inference_weight,
                    },
                ) as resp:
                    data = await resp.json()
                    return data
        except Exception as e:
            logger.error(f"Error registering env: {e}")
            raise e

    async def register_env(self):
        # Now register the env...
        while True:
            data = await self._register_env()
            if data['status'] != "success":
                logging.warning(f"Waiting to register the env due to status {data['status']}")
                await asyncio.sleep(1)
                continue
            self.env_id = data["env_id"]
            self.wandb_prepend = data["wandb_name"]
            self.curr_step = data["starting_step"]
            self.checkpoint_dir = data["checkpoint_dir"]
            self.checkpoint_interval = data["checkpoint_interval"]
            if self.config.total_steps == -1:
                self.config.total_steps = data["num_steps"]
                if self.config.total_steps == -1:
                    raise ValueError("Total steps not set in config or server!")
            print(
                f"Initialized env with id {self.env_id}: "
                f"curr_step: {self.curr_step}, "
                f"checkpoint_dir: {self.checkpoint_dir}, "
                f"checkpoint_interval: {self.checkpoint_interval}"
            )
            if self.curr_step > 0:
                self.load_checkpoint()
            break

    async def get_server_info(self):
        """
        Get the server info
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.config.rollout_server_url}/info") as resp:
                data = await resp.json()
                if data["batch_size"] != -1:
                    # update the batch size
                    self.config.batch_size = data["batch_size"]
                if data["max_token_len"] != -1:
                    self.max_token_len = data["max_token_len"]
        if self.config.batch_size == -1:
            logging.warning("Batch size not set by config or server!")
        if self.config.group_size > self.config.batch_size:
            raise ValueError(
                f"group_size ({self.config.group_size}) "
                f"must be less than batch_size ({self.config.batch_size})"
            )

    def perf_stats(self, metrics_dict):
        """
        returns wandb metrics for performance
        """
        if len(self.task_duration) > 1:
            get_std_min_max_avg(
                "train_perf/task_duration", self.task_duration, metrics_dict
            )
            self.task_duration = list()
        if len(self.succeeded_task_duration) > 1:
            get_std_min_max_avg(
                "train_perf/succeeded_task_duration",
                self.succeeded_task_duration,
                metrics_dict,
            )
            metrics_dict["train/items_sent_to_api"] = len(self.succeeded_task_duration)
            self.succeeded_task_duration = list()
        if len(self.failed_task_duration) > 1:
            get_std_min_max_avg(
                "train_perf/failed_task_duration",
                self.failed_task_duration,
                metrics_dict,
            )
            metrics_dict["train/items_rejected"] = len(self.failed_task_duration)
            self.failed_task_duration = list()
        if len(self.mainloop_timings) > 1:
            get_std_min_max_avg(
                "train_perf/mainloop_timings",
                self.mainloop_timings,
                metrics_dict,
            )
            self.mainloop_timings = list()
        if len(self.workers_added_list) > 1:
            get_std_min_max_avg(
                "train_perf/workers_added_per_attempt",
                self.workers_added_list,
                metrics_dict,
            )
            self.workers_added_list = list()
        return metrics_dict

    async def create_rollout_table(self, wandb_metrics):
        if len(self.rollouts_for_wandb) > 0:
            table = wandb.Table(columns=["text", "score"])
            for group in self.rollouts_for_wandb:
                for item in group:
                    table.add_data(item[0], item[1])
            wandb_metrics["train/rollouts"] = table
        return wandb_metrics

    async def add_rollouts_for_wandb(
        self,
        scored_data: Union[ScoredDataGroup, List[ScoredDataGroup]],
        item: Item = None,
    ):
        # Save rollout to trajectory
        num_keep = self.config.num_rollouts_per_group_for_logging
        if num_keep == -1:
            num_keep = self.config.group_size
        self.rollouts_for_wandb.append(
            [
                (
                    self.tokenizer.decode(scored_data["tokens"][i]),
                    scored_data["scores"][i],
                )
                for i in range(num_keep)
            ]
        )
        if len(self.rollouts_for_wandb) > self.config.num_rollouts_to_keep:
            self.rollouts_for_wandb.pop(0)

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """
        Log to wandb.

        To use this in your subclass, please ensure this is called after you do your metrics
        e.g.
        def wandb_log(self, wandb_metrics: Optional[Dict] = None):
            wandb_metrics = {}
            wandb_metrics['my_metric'] = 0.5
            super().wandb_log(wandb_metrics)
        """
        if wandb_metrics is None:
            wandb_metrics = dict()
        for i, server in enumerate(self.server.servers):
            server_wandb_metrics = await server.wandb_metrics({}, f"server_{i}")
        if len(self.completion_lengths) > 0:
            wandb_metrics["train/completion_lengths"] = sum(
                self.completion_lengths
            ) / len(self.completion_lengths)
            wandb_metrics["train/completion_lengths_std"] = np.std(
                self.completion_lengths
            )
            wandb_metrics["train/completion_lengths_max"] = np.max(
                self.completion_lengths
            )
            wandb_metrics["train/completion_lengths_min"] = np.min(
                self.completion_lengths
            )
            wandb_metrics["train/completion_lengths_p95"] = (
                np.array(self.completion_lengths) > (0.95 * self.max_token_len)
            ).mean()
        wandb_metrics = await self.create_rollout_table(wandb_metrics)
        wandb_metrics = self.perf_stats(wandb_metrics)
        self.rollouts_for_wandb = []
        self.completion_lengths = []
        if self.config.use_wandb:
            if self.wandb_prepend is not None:
                wandb_metrics = {
                    f"{self.wandb_prepend}_{k}": v for k, v in wandb_metrics.items()
                }
            # add server metrics to wandb without prepend to collate them all
            wandb_metrics.update(server_wandb_metrics)
            wandb.log(wandb_metrics, step=self.curr_step)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=1, max=10),
    )
    async def _send_scored_data_to_api(self, scored_data):
        """
        Send scored data to the API with retry logic for timeouts and server errors.
        """
        url = (
            f"{self.config.rollout_server_url}/scored_data_list"
            if isinstance(scored_data, list)
            else f"{self.config.rollout_server_url}/scored_data"
        )
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=scored_data,
            ) as resp:
                if resp.status >= 500:
                    # Server errors (5xx) should trigger a retry
                    logging.debug(f"Server error: {resp.status}, retrying...")
                    raise Exception(f"Server error: {resp.status}")
                elif resp.status >= 400:
                    # Client errors (4xx) are logged but not retried
                    logging.error(f"Client error: {resp.status}, not retrying")
                    return
                # Success case: print response text
                print(await resp.text())

    async def handle_send_to_api(
        self,
        scored_data: Union[ScoredDataGroup, List[ScoredDataGroup]],
        item: Item = None,
        do_send_to_api: bool = True,
        abort_on_any_max_length_exceeded: bool = True,
    ):
        """
        Send the chats to the API with robust error handling and support for multiple ScoredDataGroups.

        Args:
            scored_data: List or single scored item(s) to send.
            item: Optional item for context.
            do_send_to_api: Whether to actually send the data to the API.
            abort_on_any_max_length_exceeded: If True, skips sending if any token sequence in any batch exceeds max_token_len, aborting the entire operation.
        """
        if scored_data is None:
            logger.debug("handle_send_to_api: scored_data is None, returning.")
            return

        batches_to_process = scored_data if isinstance(scored_data, list) else [scored_data]
        valid_batches: List[ScoredDataGroup] = []

        for i, batch in enumerate(batches_to_process):
            if batch is None:
                logger.debug(f"handle_send_to_api: Batch {i} is None, skipping.")
                continue

            # Determine group_size for this specific batch, falling back to config default
            batch_group_size = batch.get("group_overrides", {}).get(
                "group_size", self.config.group_size
            )

            # Ensure correct group size
            actual_tokens_len = len(batch.get("tokens", []))
            if actual_tokens_len != batch_group_size:
                logger.warning(
                    f"Skipping batch {i} due to group size mismatch. "
                    f"Expected {batch_group_size}, got {actual_tokens_len}. "
                    f"Batch keys: {list(batch.keys())}"
                )
                continue

            # Skip identical scores if configured
            batch_scores = batch.get("scores", [])
            if (
                self.config.ensure_scores_are_not_same
                and len(set(batch_scores)) == 1
                and len(batch_scores) > 0  # Avoid triggering for empty scores list
            ):
                logger.warning(
                    f"Skipping batch {i} due to identical scores. First few scores: {str(batch_scores[:3])}"
                )
                continue

            # Ensure optional keys exist
            batch.setdefault("ref_logprobs", None)
            batch.setdefault("overrides", None)
            batch.setdefault("group_overrides", None)
            batch.setdefault("messages", None) # Ensure messages key exists for later population

            # Check if all mask sequences in the batch are empty
            all_masks_empty = True
            masks_list = batch.get("masks")
            if isinstance(masks_list, list):
                if not masks_list:  # Empty list of masks
                    pass  # all_masks_empty remains true, batch will be skipped
                else:
                    for mask_idx, mask_sequence in enumerate(masks_list):
                        if hasattr(mask_sequence, '__iter__') and len(mask_sequence) > 0:
                            all_masks_empty = False
                            break
                        elif not hasattr(mask_sequence, '__iter__'):
                            logger.warning(f"Mask sequence {mask_idx} in batch {i} is not iterable. Skipping batch.")
                            all_masks_empty = True # Mark to ensure skip if loop finishes
                            break # Cannot process this batch further
            else:
                logger.warning(
                    f"Skipping batch {i} due to missing or invalid 'masks' key (not a list or missing). "
                    f"Batch keys: {list(batch.keys())}"
                )
                continue # Skip this batch (all_masks_empty remains true by default or skip)

            if all_masks_empty:
                logger.warning(
                    f"Skipping batch {i} because all mask sequences are empty or masks field invalid. "
                    f"Number of mask sequences: {len(masks_list) if isinstance(masks_list, list) else 'N/A'}. "
                    f"First few scores: {str(batch_scores[:3])}"
                )
                continue # Skip this batch
            
            # Check for max token length exceeded
            batch_tokens = batch.get("tokens", [])
            if abort_on_any_max_length_exceeded and any(
                len(x) >= self.max_token_len for x in batch_tokens if x is not None
            ):
                token_lengths = [len(x) for x in batch_tokens if x is not None]
                logger.warning(
                    f"Token length limit exceeded in batch {i}, aborting send operation due to abort_on_any_max_length_exceeded=True. "
                    f"Max token len configured: {self.max_token_len}. "
                    f"Batch token lengths: {token_lengths}. "
                    f"First few scores: {str(batch_scores[:3])}"
                )
                return # Abort entire operation

            valid_batches.append(batch)

        if not valid_batches:
            logger.debug("No valid batches to send to API after filtering.")
            return

        for batch_idx, batch_data in enumerate(valid_batches):
            await self.add_rollouts_for_wandb(batch_data, item)
            
            current_masks = batch_data.get("masks", [])
            if isinstance(current_masks, list):
                for mask in current_masks:
                    if hasattr(mask, '__iter__'): # Make sure mask is iterable
                        self.completion_lengths.append(len(mask))
            
            if self.config.include_messages and batch_data.get("messages") is None:
                tokens_for_messages = batch_data.get("tokens")
                if isinstance(tokens_for_messages, list):
                    decoded_messages = []
                    for token_seq_idx, token_sequence in enumerate(tokens_for_messages):
                        if token_sequence is not None:
                            try:
                                decoded_messages.append(self.tokenizer.decode(token_sequence))
                            except Exception as e:
                                logger.error(f"Error decoding token sequence {token_seq_idx} for messages in valid batch {batch_idx}: {e}. Sequence (first 10 tokens): {str(token_sequence[:10])}")
                                decoded_messages.append(f"DECODING_ERROR: {e}") # Placeholder for error
                        else:
                            logger.warning(f"Token sequence {token_seq_idx} in valid batch {batch_idx} is None, cannot decode for messages.")
                            decoded_messages.append(None) # Handle None token_sequence
                    batch_data["messages"] = decoded_messages
                else:
                    logger.warning(f"Cannot generate messages for valid batch {batch_idx}: 'tokens' is missing or not a list.")

            if self.jsonl_writer is not None:
                try:
                    self.jsonl_writer.write(batch_data)
                    logger.debug(f"Wrote valid batch {batch_idx} to {self.config.data_path_to_save_groups}")
                except Exception as e:
                    logger.error(f"Failed to write valid batch {batch_idx} to JSONL: {e}")

        if do_send_to_api:
            try:
                self.items_sent_this_step += len(valid_batches)
                payload_to_send = valid_batches[0] if len(valid_batches) == 1 else valid_batches
                logger.debug(f"Sending {len(valid_batches)} valid batch(es) to API. Payload type: {type(payload_to_send)}. First item keys (if list): {list(payload_to_send[0].keys()) if isinstance(payload_to_send, list) and payload_to_send else list(payload_to_send.keys()) if not isinstance(payload_to_send, list) else 'N/A'}")
                await self._send_scored_data_to_api(payload_to_send)
            except Exception as e: # Catching general Exception as _send_scored_data_to_api might raise various errors
                logger.error(f"Failed to send {len(valid_batches)} scored batch(es) after retries: {e}")
        else:
            logger.debug(f"Skipping API send for {len(valid_batches)} valid batch(es) because do_send_to_api is False.")
        return

    async def handle_env(
        self, item_uuid: str
    ) -> Optional[Union[ScoredDataGroup, List[ScoredDataGroup]]]:
        """
        Handle the rollout of an item
        """
        item = self.running_items.get(item_uuid)
        if item is None:
            logger.error(f"handle_env: Item {item_uuid} not found in running_items! Returning None.")
            return None
        start_time = time.time()
        logger.debug(f"handle_env: Starting with item: {item}")
        # do a rollout with item
        try:
            to_postprocess, to_backlog = await self.collect_trajectories(item)
        except Exception as e:
            logger.exception(f"handle_env: Exception during collect_trajectories for {item_uuid}: {e}")
            to_postprocess = None
            to_backlog = []
        # add the items to the queue
        if len(to_backlog) > 0:
            self.backlog.extend(to_backlog)
        try:
            if (to_postprocess is None) or (len(to_postprocess) == 0):
                pass
            else:
                to_postprocess = await self.postprocess_histories(to_postprocess)
        except Exception as e:
            logger.error(f"Error in scoring: {item}")
            print(e)
            to_postprocess = None
        self.running_items.pop(item_uuid, None)
        duration = max(0.0, time.time() - start_time)
        self.task_duration.append(duration)
        if to_postprocess is not None:
            self.task_successful.append(1)
            self.succeeded_task_duration.append(duration)
            logger.debug(f"handle_env: Collected {len(to_postprocess)} trajectories")
            await self.handle_send_to_api(to_postprocess, item)
        else:
            self.task_successful.append(0)
            self.failed_task_duration.append(duration)
            logger.debug("handle_env: No trajectories collected")
        # Finally pop it
        await self.cleanup()
        return to_postprocess

    async def cleanup(self):
        """
        Optional: Cleanup the environment
        """
        pass

    @retry(
        stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=10)
    )
    async def get_status(self):
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.config.rollout_server_url}/status-env",
                json={"env_id": self.env_id},
            ) as resp:
                self.status_dict = await resp.json()
                new_weight = self.status_dict["env_weight"]
                max_num_workers = self.config.max_num_workers
                if max_num_workers == -1:
                    max_num_workers = self.config.max_num_workers_per_node * len(
                        self.server.servers
                    )
                self.max_num_workers = max_num_workers
                await self.server.update_weight(new_weight)

    async def env_step_checks(self):
        # Check if we need to run an eval or log...
        if self.curr_step != self.status_dict["current_step"]:
            if self.config.steps_per_eval > 0:
                if (self.curr_step % self.config.steps_per_eval) > (
                    self.status_dict["current_step"] % self.config.steps_per_eval
                ):
                    if (self.eval_runner is None) or (self.eval_runner.done()):
                        eval_task = asyncio.create_task(self.evaluate())
                        self.eval_runner = eval_task
                        if self.config.eval_handling == EvalHandlingEnum.STOP_TRAIN:
                            # Stop training if eval is running
                            self.backlog.extend(self.running_items.values())
                            for worker in self.workers:
                                worker.cancel()
                            self.workers = set()
                            self.running_items: dict[UUID, Item] = dict()
                    else:
                        warnings.warn(
                            "Eval is not finished in this iteration of the loop, skipping this eval step..."
                        )
            if self.checkpoint_interval > 0:
                if (self.curr_step % self.checkpoint_interval) > (
                    self.status_dict["current_step"] % self.checkpoint_interval
                ):
                    checkpoint_step = (
                        self.status_dict["current_step"] // self.checkpoint_interval
                    ) * self.checkpoint_interval
                    self.save_checkpoint(checkpoint_step)
            self.curr_step = self.status_dict["current_step"]
            if self.items_sent_this_step >= self.config.min_items_sent_before_logging:
                self.items_sent_this_step = 0
                await self.wandb_log({})

    async def add_train_workers(self):
        if (self.eval_runner is not None) and (not self.eval_runner.done()):
            if self.config.eval_handling == EvalHandlingEnum.STOP_TRAIN:
                return
            elif self.config.eval_handling == EvalHandlingEnum.LIMIT_TRAIN:
                max_num_workers = int(
                    self.max_num_workers * self.config.eval_limit_ratio
                )
            else:
                max_num_workers = self.max_num_workers
        else:
            max_num_workers = self.max_num_workers

        # set max_num_workers to whatever is max off policy and num workers
        queue_capacity = (
            self.config.max_batches_offpolicy
            * self.config.batch_size
            // self.config.group_size
        )
        available_queue_slots = max(0, queue_capacity - self.status_dict["queue_size"])
        original_max_workers = max_num_workers
        max_num_workers = min(max_num_workers, available_queue_slots)

        if (self.curr_step == 0) and (len(self.workers) == 0):
            # We are starting up, so we should just skip the append to the list
            pass
        else:
            # Log how many workers we *intend* to add
            workers_to_add = max(0, max_num_workers - len(self.workers))
            self.workers_added_list.append(workers_to_add)

        while len(self.workers) < max_num_workers:
            # Generate a UUID for tracking this item
            item_uuid = str(uuid.uuid4())
            item = None # Initialize item to None
            if len(self.backlog) > 0:
                item = self.backlog.pop()
            else:
                item = await self.get_next_item()

            if item is None:
                break # Exit loop if no item is available

            self.running_items[item_uuid] = item
            worker = asyncio.create_task(self.handle_env(item_uuid))
            self.workers.add(worker)
            worker.add_done_callback(
                lambda fut, i=item: (
                    (
                        self.workers.discard(fut),
                        (
                            setattr(self, "last_completed_item", i)
                            if fut.result()
                            else None
                        ),
                    )[1]
                    if fut.done() and not fut.cancelled()
                    else None
                )
            )

    async def env_manager(self):
        """
        Rollout manager
        """
        await self.setup()
        await self.setup_wandb()
        await self.register_env()
        await self.get_server_info()
        # Wait for other instances to get setup :)
        await asyncio.sleep(5)
        while True:
            if self.last_loop_time is not None:
                self.mainloop_timings.append(
                    max(0.0, time.time() - self.last_loop_time)
                )
            # get status from server
            self.last_loop_time = time.time()
            await self.get_status()
            await self.env_step_checks()
            logger.info(f"env_manager: Status dict: {self.status_dict}")
            # Log queue check values before the condition
            queue_items = self.status_dict["queue_size"] * self.config.group_size
            queue_capacity = self.config.max_batches_offpolicy * self.config.batch_size
            if (
                self.status_dict["current_step"]
                + (
                    self.status_dict["queue_size"]
                    * self.config.group_size
                    // self.config.batch_size
                )
            ) > self.config.total_steps:
                for worker in self.workers:
                    worker.cancel()
                break
            if (
                (
                    queue_items
                    >= queue_capacity
                )
                and (self.config.max_batches_offpolicy > 0)
            ) or (self.config.batch_size == -1):
                # We have too many, lets cleanup the tasks and wait a bit
                self.backlog.extend(self.running_items.values())
                for worker in self.workers:
                    worker.cancel()
                self.running_items = dict()
                self.workers = set()
            elif len(self.workers) >= self.max_num_workers:
                pass
            else:
                await self.add_train_workers()
            await asyncio.sleep(0.1)

    async def process_manager(self):
        """
        Process manager for running a specific number of groups
        """
        await self.setup()

        if self.config.use_wandb:
            random_id = "".join(random.choices(string.ascii_lowercase, k=6))
            current_date = datetime.now().strftime("%Y-%m-%d")
            wandb_run_name = f"{self.name}-{current_date}-{random_id}"
            wandb.init(
                project=self.wandb_project,
                name=wandb_run_name,
                group=self.wandb_group,
                config=self.config.model_dump(),
            )

        # Initialize the processing
        self.curr_step = 0

        print(f"Starting to process {self.n_groups_to_process} groups...")

        # Process the required number of groups
        while self.curr_step < self.n_groups_to_process:
            # Get an item to process
            item = await self.get_next_item()
            if item is None:
                print("No more items to process")
                break

            # Process the group
            print(f"Processing group {self.curr_step + 1}/{self.n_groups_to_process}")

            # Collect trajectories with the specified group size
            # Override the group_size temporarily
            self.config.group_size = self.group_size_to_process

            # Collect and process the trajectories
            to_postprocess, _ = await self.collect_trajectories(item)

            if to_postprocess:
                # Post-process the trajectories
                processed_data = await self.postprocess_histories(to_postprocess)

                # Save to output file (don't send to API)
                await self.handle_send_to_api(
                    processed_data,
                    item,
                    do_send_to_api=False,
                    abort_on_any_max_length_exceeded=False,
                )
                await self.wandb_log()

                self.curr_step += 1
                print(
                    f"Successfully processed group {self.curr_step}/{self.n_groups_to_process}"
                )
            else:
                print("Failed to process group, retrying...")

        print(f"Completed processing {self.curr_step} groups")

        # Close the output file if it's open
        if self.jsonl_writer is not None:
            self.jsonl_writer.close()

        generate_html(self.config.data_path_to_save_groups)

    @classmethod
    def cli(cls):
        """
        Command-line interface entry point for the environment.
        This method handles the CLI commands for serve and process.
        """

        # Create subcommands dictionary
        subcommands = {
            "serve": cls.get_cli_serve_config_cls(),
            "process": cls.get_cli_process_config_cls(),
        }

        # Custom exception handler for cleaner error output
        def custom_error_handler(ex: Exception) -> int:
            """Handles exceptions with clean output for known error types."""
            if isinstance(ex, FailedExecutionException):
                # Handle argparse errors (already printed by argparse)
                print()
                print(ex.message.split("error: ")[-1])
                return 2

            raise ex

        run_and_exit(
            subcommands,
            description=f"CLI for {cls.__name__}",
            exception_handler=custom_error_handler,
        )

    @classmethod
    def get_cli_serve_config_cls(cls) -> type:
        """
        Returns the CLI configuration class for serving commands.

        Returns:
            type: The CliServeConfig class for serving commands.
        """

        env_config, server_configs = cls.config_init()
        env_full_prefix = f"{ENV_NAMESPACE}{NAMESPACE_SEP}"
        openai_full_prefix = f"{OPENAI_NAMESPACE}{NAMESPACE_SEP}"

        class CliServeConfig(
            get_prefixed_pydantic_model(type(env_config), env_full_prefix),
            get_prefixed_pydantic_model(OpenaiConfig, openai_full_prefix),
            ServerManagerConfig,
            Cmd,
        ):
            """
            Configuration for the serve command.
            This combines BaseEnvConfig and OpenaiConfig into a single command.
            """

            def run(self) -> None:
                """The logic to execute for the 'serve' command."""
                # Convert this config into the formats needed by BaseEnv
                wandb_name_attr = f"{ENV_NAMESPACE}{NAMESPACE_SEP}wandb_name"
                if getattr(self, wandb_name_attr) is None and cls.name is not None:
                    setattr(self, wandb_name_attr, cls.name)
                model_dumped = self.model_dump(exclude_unset=True)
                server_manager_config = ServerManagerConfig(**model_dumped)
                # Create the environment instance
                env = cls(
                    config=env_config,
                    server_configs=server_configs,
                    slurm=server_manager_config.slurm,
                    testing=server_manager_config.testing,
                )

                # Run the environment
                asyncio.run(env.env_manager())

        return CliServeConfig

    @classmethod
    def get_cli_process_config_cls(cls) -> type:
        """
        Returns the CLI configuration class for processing commands.

        Returns:
            type: The CliProcessConfig class for processing commands.
        """

        PROCESS_MODE_ENV_DEFAULT_CONFIG = BaseEnvConfig(
            group_size=8,
            total_steps=2,
            ensure_scores_are_not_same=False,
            include_messages=True,
        )
        PROCESS_MODE_OPENAI_DEFAULT_CONFIG = OpenaiConfig(
            model_name="gpt-4.1-nano",
            base_url=None,
            api_key=None,
        )
        PROCESS_MODE_SERVER_MANAGER_DEFAULT_CONFIG = ServerManagerConfig(
            slurm=False,
            testing=False,
        )

        default_env_config, default_openai_config = cls.config_init()

        if isinstance(default_openai_config, list):
            default_openai_config = default_openai_config[0]

        env_full_prefix = f"{ENV_NAMESPACE}{NAMESPACE_SEP}"
        openai_full_prefix = f"{OPENAI_NAMESPACE}{NAMESPACE_SEP}"

        env_config_cls_new_defaults = adjust_model_defaults(
            type(default_env_config), PROCESS_MODE_ENV_DEFAULT_CONFIG
        )
        openai_config_cls_new_defaults = adjust_model_defaults(
            OpenaiConfig, PROCESS_MODE_OPENAI_DEFAULT_CONFIG
        )
        server_manager_config_cls_new_defaults = adjust_model_defaults(
            ServerManagerConfig,
            PROCESS_MODE_SERVER_MANAGER_DEFAULT_CONFIG,
        )

        class CliProcessConfig(
            get_prefixed_pydantic_model(env_config_cls_new_defaults, env_full_prefix),
            get_prefixed_pydantic_model(
                openai_config_cls_new_defaults, openai_full_prefix
            ),
            server_manager_config_cls_new_defaults,
            Cmd,
        ):
            """
            Configuration for the process command.
            """

            config: str | None = Field(
                default=None,
                description="Path to .yaml config file. CLI args override this.",
            )

            def run(self) -> None:
                """The logic to execute for the 'process' command."""
                # Setup environment configuration
                wandb_name_attr = f"{ENV_NAMESPACE}{NAMESPACE_SEP}wandb_name"
                if getattr(self, wandb_name_attr) is None and cls.name is not None:
                    setattr(self, wandb_name_attr, cls.name)

                if self.config is not None:
                    with open(self.config, "r") as f:
                        config = yaml.safe_load(f)
                    print(f"Loaded config from {self.config}")
                else:
                    config = {}

                cli_passed_flags = get_double_dash_flags()

                # cli args overrides config file which overrides class defaults which overrides process mode defaults
                env_config = env_config_cls_new_defaults(
                    **merge_dicts(
                        default_env_config.model_dump(),
                        PROCESS_MODE_ENV_DEFAULT_CONFIG.model_dump(),
                        config.get(ENV_NAMESPACE, {}),
                        extract_namespace(
                            cli_passed_flags, env_full_prefix
                        ),  # only extract namespace for cli-passed args
                    )
                )
                openai_config = openai_config_cls_new_defaults(
                    **merge_dicts(
                        default_openai_config.model_dump(),
                        PROCESS_MODE_OPENAI_DEFAULT_CONFIG.model_dump(),
                        config.get(OPENAI_NAMESPACE, {}),
                        extract_namespace(
                            cli_passed_flags, openai_full_prefix
                        ),  # only extract namespace for cli-passed args
                    )
                )

                server_manager_cli_passed_flags = {}
                if "slurm" in cli_passed_flags:
                    server_manager_cli_passed_flags["slurm"] = cli_passed_flags["slurm"]
                if "testing" in cli_passed_flags:
                    server_manager_cli_passed_flags["testing"] = cli_passed_flags[
                        "testing"
                    ]

                server_manager_config = server_manager_config_cls_new_defaults(
                    **merge_dicts(
                        ServerManagerConfig().model_dump(),
                        PROCESS_MODE_SERVER_MANAGER_DEFAULT_CONFIG.model_dump(),
                        config.get(SERVER_MANAGER_NAMESPACE, {}),
                        server_manager_cli_passed_flags,
                    )
                )

                # Create the environment instance
                env = cls(
                    config=env_config,
                    server_configs=[openai_config],
                    slurm=server_manager_config.slurm,
                    testing=server_manager_config.testing,
                )

                # Set the process mode parameters
                env.process_mode = True
                env.n_groups_to_process = env_config.total_steps
                env.group_size_to_process = env_config.group_size

                print(
                    f"Processing {env_config.total_steps} groups of "
                    f"{env_config.group_size} responses and "
                    f"writing to {env_config.data_path_to_save_groups}"
                )

                asyncio.run(env.process_manager())

                # Actual implementation would go here

        return CliProcessConfig
