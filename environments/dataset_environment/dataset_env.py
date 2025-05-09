import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
from datasets import load_dataset
from pydantic import Field

from atroposlib.envs.base import BaseEnv, BaseEnvConfig, OpenaiConfig, ScoredDataGroup
from atroposlib.envs.reward_fns import registry
from atroposlib.envs.reward_fns.combined_reward import CombinedReward
from atroposlib.type_definitions import Item
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DatasetEnvConfig(BaseEnvConfig):
    dataset_name: Optional[str] = Field(None, description="HuggingFace dataset name")
    dataset_config: Optional[str] = Field(
        None, description="Dataset configuration name"
    )
    split: str = Field("train", description="Dataset split to use")
    dataset_path: Optional[str] = Field(
        None, description="Local path to dataset (alternative to dataset_name)"
    )
    prompt_field: Optional[str] = Field(None, description="Field in dataset to use as prompt")
    answer_field: Optional[str] = Field(
        None, description="Field in dataset to use as answer"
    )
    ground_truth_field: Optional[str] = Field(
        None, description="Field in dataset containing canonical correct answer"
    )
    system_prompt: Optional[str] = Field(None, description="System prompt to use")
    prefill: Optional[str] = Field(
        None, description="Text to prefill the completion with (e.g. '<think>')"
    )
    shuffle_dataset: bool = Field(True, description="Whether to shuffle the dataset")
    max_generations_per_prompt: int = Field(
        1, description="Number of generations per prompt for collection"
    )
    include_messages_in_scoring: bool = Field(
        False, description="Whether to include messages in scoring"
    )
    reward_functions: Optional[List[Union[str, Dict[str, Any]]]] = Field(
        None, description="List of reward functions to apply (string names or full configs)"
    )

    temperature: float = Field(0.7, description="Temperature for generation")
    top_p: float = Field(0.9, description="Top-p for generation")
    max_tokens: int = Field(4096, description="Maximum tokens for generation")
    length_warmup_steps: int = Field(0, description="Steps for length warmup")
    min_tokens: int = Field(0, description="Minimum tokens for generation")

    eval_dataset_name: Optional[str] = Field(
        None, description="Evaluation dataset name"
    )
    eval_dataset_config: Optional[str] = Field(
        None, description="Evaluation dataset config"
    )
    eval_split: Optional[str] = Field(None, description="Evaluation dataset split")
    debug_mode: bool = Field(False, description="Enable debug logging")


class DatasetEnv(BaseEnv):
    name = "dataset"

    def __init__(
        self, config: DatasetEnvConfig, server_configs, slurm=True, testing=False
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config = config
        self.dataset = None
        self.iter = 0
        self.metric_buffer = {}

        # Store debug mode and set logger level
        self.debug_mode = config.debug_mode
        if self.debug_mode:
            logger.setLevel(logging.DEBUG)
        else:
            if logger.level == logging.NOTSET or logger.level > logging.WARNING:
                logger.setLevel(logging.WARNING) # Default to WARNING

        self.reward_function = self._initialize_reward_function()

    def _initialize_reward_function(self):
        if self.config.reward_functions:
            if len(self.config.reward_functions) == 1:
                return registry.create(self.config.reward_functions[0])
            else:
                return CombinedReward(
                    rewards=self.config.reward_functions, normalization="sum"
                )
        logger.warning("No reward functions configured (field is None or list is empty).")
        return None

    async def setup(self):
        if self.config.dataset_path:
            self.dataset = load_dataset(
                self.config.dataset_path, split=self.config.split
            )
        else:
            self.dataset = load_dataset(
                self.config.dataset_name,
                self.config.dataset_config,
                split=self.config.split,
            )
        logger.info(f"Dataset features: {self.dataset.features}")
        logger.info(f"Sample item keys: {list(self.dataset[0].keys())}")
        logger.info(f"Sample item: {self.dataset[0]}")

        if self.config.shuffle_dataset:
            self.dataset = self.dataset.shuffle()

        self.metric_buffer = {}

    async def get_next_item(self) -> Item:
        if not self.dataset:
            await self.setup()

        item_data = self.dataset[self.iter % len(self.dataset)]
        self.iter += 1

        logger.warning(f"get_next_item: item_data.keys(): {list(item_data.keys())}")
        logger.warning(f"get_next_item: config: prompt_field='{self.config.prompt_field}', answer_field='{self.config.answer_field}', ground_truth_field='{self.config.ground_truth_field}'")
        logger.warning(f"get_next_item: raw_prompt_data: {item_data.get(self.config.prompt_field)}")
        logger.warning(f"get_next_item: raw_answer_data: {item_data.get(self.config.answer_field)}")
        logger.warning(f"get_next_item: raw_ground_truth_data: {item_data.get(self.config.ground_truth_field)}")

        user_msg = {"role": "user", "content": item_data[self.config.prompt_field]}
        prompt = tuple([frozenset(user_msg.items())])

        answer = None
        if self.config.answer_field and self.config.answer_field in item_data:
            answer = item_data[self.config.answer_field]

        ground_truth = None
        if self.config.ground_truth_field and self.config.ground_truth_field in item_data:
            ground_truth = item_data[self.config.ground_truth_field]

        logger.warning(f"get_next_item: returning: prompt_len={len(prompt)}, answer='{answer}', ground_truth='{ground_truth}'")
        return (prompt, answer, ground_truth)

    async def collect_trajectory(self, item: Item) -> Tuple[List, List]:
        user_content = dict(item[0][0])["content"]
        answer = item[1] if len(item) > 1 else None

        messages = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})

        messages.append({"role": "user", "content": user_content})

        if self.config.prefill:
            messages.append({"role": "assistant", "content": self.config.prefill})

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)

        max_tokens = self.config.max_tokens
        if self.config.length_warmup_steps > 0:
            warmup_progress = min(1.0, self.curr_step / self.config.length_warmup_steps)
            max_tokens = int(
                self.config.min_tokens
                + warmup_progress * (self.config.max_tokens - self.config.min_tokens)
            )

        completions = await self.server.completion(
            prompt=prompt,
            n=self.config.max_generations_per_prompt,
            max_tokens=max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        to_score = []
        to_backlog = []

        for completion in completions.choices:
            completion_text = (
                completion.text
                if hasattr(completion, "text")
                else completion.message.content
            )

            full_messages = []
            if self.config.system_prompt:
                full_messages.append(
                    {"role": "system", "content": self.config.system_prompt}
                )

            full_messages.append({"role": "user", "content": user_content})

            response_content = completion_text
            if self.config.prefill:
                response_content = self.config.prefill + completion_text

            full_messages.append({"role": "assistant", "content": response_content})

            to_score.append((full_messages, answer, item[2] if len(item) > 2 else None))

        return to_score, to_backlog

    async def postprocess_histories(
        self, trajectories: List[List[Dict[str, Any]]]
    ) -> Optional[ScoredDataGroup]:
        """
        Postprocess the histories by scoring them.
        The input 'trajectories' is expected to be a list of message lists,
        which is suitable for the `score` method's `rollout_group_data` argument.
        """
        if not trajectories:
            logger.warning(
                "postprocess_histories: received empty or invalid trajectories, returning None."
            )
            return None
        # The 'score' method expects List of trajectories, where each trajectory is List[Message]
        # This matches the input 'trajectories' (which is rollout_group_data from collect_trajectories)
        scored_data = await self.score(trajectories)
        return scored_data

    async def collect_trajectories(self, item: Item) -> Tuple[List, List]:
        self.current_item = item

        user_content = dict(item[0][0])["content"]

        messages = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})

        messages.append({"role": "user", "content": user_content})

        if self.config.prefill:
            messages.append({"role": "assistant", "content": self.config.prefill})

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)

        max_tokens = self.config.max_tokens

        completions = await self.server.completion(
            prompt=prompt,
            n=self.config.group_size,
            max_tokens=max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        print(f"Completions: {completions}")
        trajectories = []
        for completion in completions.choices:
            completion_text = (
                completion.text
                if hasattr(completion, "text")
                else completion.message.content
            )

            full_messages = []
            if self.config.system_prompt:
                full_messages.append(
                    {"role": "system", "content": self.config.system_prompt}
                )

            full_messages.append({"role": "user", "content": user_content})

            response_content = completion_text
            if self.config.prefill:
                response_content = self.config.prefill + completion_text

            full_messages.append({"role": "assistant", "content": response_content})

            trajectories.append(full_messages)

        return trajectories, []

    async def score(self, rollout_group_data: List) -> Optional[ScoredDataGroup]:
        logger.warning(f"Scoring {len(rollout_group_data)} rollout items")

        logger.warning(f"score: self.current_item (type: {type(self.current_item)}): {self.current_item}")
        logger.warning(f"score: config: answer_field='{self.config.answer_field}', ground_truth_field='{self.config.ground_truth_field}'")

        scores = ScoredDataGroup()
        scores["tokens"] = []
        scores["masks"] = []
        scores["scores"] = []
        scores["advantages"] = None
        scores["ref_logprobs"] = None
        scores["messages"] = None if not self.config.include_messages_in_scoring else []

        answer = (
            self.current_item[1]
            if self.current_item and len(self.current_item) > 1
            else None
        )
        logger.warning(f"score: Extracted answer from current_item: {answer}")
        logger.warning(f"Answer for current item: {answer}")

        ground_truth = (
            self.current_item[2]
            if self.current_item and len(self.current_item) > 2
            else None
        )
        logger.warning(f"score: Extracted ground_truth from current_item: {ground_truth}")
        logger.warning(f"Ground truth for current item: {ground_truth}")

        formatted_completions = []
        for trajectory in rollout_group_data:
            if trajectory and isinstance(trajectory, list):
                assistant_messages = [
                    msg
                    for msg in trajectory
                    if isinstance(msg, dict) and msg.get("role") == "assistant"
                ]
                if assistant_messages:
                    formatted_completions.append([assistant_messages[-1]])

        if not formatted_completions:
            logger.warning("No valid completions to score")
            return None

        logger.warning(f"score: formatted_completions passed to reward_function: {formatted_completions}")
        try:
            reward_kwargs = {
                "solution": answer,
                "ground_truth": ground_truth,
                "item": self.current_item,
                "config": self.config,
            }

            all_rewards = self.reward_function(formatted_completions, **reward_kwargs)

            logger.info(f"Calculated rewards: {all_rewards}")

        except Exception as e:
            logger.error(f"Error applying reward functions: {e}")
            logger.exception(e)
            all_rewards = [0.0] * len(formatted_completions)

        for i, (trajectory, reward) in enumerate(zip(rollout_group_data, all_rewards)):
            try:
                tokenized = tokenize_for_trainer(self.tokenizer, trajectory)

                scores["tokens"].append(tokenized["tokens"])
                scores["masks"].append(tokenized["masks"])
                scores["scores"].append(reward)

                if self.config.include_messages_in_scoring:
                    if "messages" not in scores:
                        scores["messages"] = []
                    scores["messages"].append(trajectory)
                logger.warning(f"Scores: {scores['scores']}")
            except Exception as e:
                logger.error(f"Error processing trajectory {i}: {e}")
                logger.exception(e)

        if not scores["tokens"]:
            logger.warning("No valid scores generated")
            return None

        logger.info(f"Generated scores: {scores['scores']}")
        return scores

    async def evaluate(self):
        if (
            not hasattr(self.config, "eval_dataset_name")
            or not self.config.eval_dataset_name
        ):
            return

        if not hasattr(self, "eval_dataset"):
            self.eval_dataset = load_dataset(
                self.config.eval_dataset_name,
                self.config.eval_dataset_config,
                split=self.config.eval_split,
            )
            self.eval_dataset = self.eval_dataset.select(
                range(min(100, len(self.eval_dataset)))
            )

        eval_metrics = {}
        eval_tasks = []

        for i in range(min(self.config.max_eval_workers, len(self.eval_dataset))):
            item = self.eval_dataset[i]
            user_msg = {"role": "user", "content": item[self.config.prompt_field]}
            prompt = tuple([frozenset(user_msg.items())])

            answer = None
            if self.config.answer_field and self.config.answer_field in item:
                answer = item[self.config.answer_field]

            eval_tasks.append(self.collect_trajectory((prompt, answer)))

        eval_results = await asyncio.gather(*eval_tasks)

        eval_scores = []
        for result in eval_results:
            if result[0]:
                scored_data = await self.score(result[0])
                if scored_data and "scores" in scored_data:
                    eval_scores.extend(scored_data["scores"])

        if eval_scores:
            eval_metrics["eval/mean_score"] = sum(eval_scores) / len(eval_scores)
            eval_metrics["eval/max_score"] = max(eval_scores)
            eval_metrics["eval/min_score"] = min(eval_scores)

        await self.wandb_log(eval_metrics)

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        metrics = wandb_metrics or {}

        for key, values in self.metric_buffer.items():
            if values:
                metrics[f"train/{key}"] = sum(values) / len(values)

        self.metric_buffer = {k: [] for k in self.metric_buffer}

        if hasattr(self, "reward_function") and self.wandb:
            if hasattr(self.reward_function, "set_wandb_logger"):
                self.reward_function.set_wandb_logger(self.wandb)

        await super().wandb_log(metrics)

    @classmethod
    def config_init(
        cls, 
        config_name: Optional[str] = None,
        env_config_override: Optional[DatasetEnvConfig] = None  # Added to accept CLI overrides
    ) -> Tuple[DatasetEnvConfig, List[OpenaiConfig]]:
        """Load settings from the local configs directory, allowing for CLI overrides."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = config_name or "dataset_default.yaml"
        cfg_path = os.path.join(current_dir, "configs", config_file)

        raw_from_yaml = {}
        try:
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    raw_from_yaml = yaml.safe_load(f) or {}
                logger.info(f"Loaded config from {cfg_path}")
            else:
                # This case should ideally be handled by the fallback in the except block if it's critical
                logger.warning(
                    f"Config file not found at {cfg_path}. Will proceed with env_config_override or defaults."
                )
            
            # Merge CLI overrides: CLI values take precedence over YAML values
            if env_config_override:
                # Get non-default values from the override object provided by Tyro
                override_dict = env_config_override.model_dump(exclude_none=True) # exclude_none to avoid overwriting with None if CLI arg wasn't set
                raw_from_yaml.update(override_dict) # Update YAML data with CLI data

            # Ensure debug_mode exists before creating config if not set by CLI or YAML
            if 'debug_mode' not in raw_from_yaml:
                 raw_from_yaml['debug_mode'] = False # Default if missing

            env_conf = DatasetEnvConfig(**raw_from_yaml)

            # Validate that essential fields are loaded (either from YAML or CLI override)
            if env_conf.dataset_name is None:
                raise ValueError("dataset_name is required but was not provided by YAML or CLI.")
            if env_conf.prompt_field is None:
                raise ValueError("prompt_field is required but was not provided by YAML or CLI.")

            server_confs = []
            # Server config loading from the potentially merged raw_from_yaml
            for sc_data in raw_from_yaml.get("server_configs", []):
                # Get values directly from YAML data
                model_name = sc_data.get("model_name", os.getenv("OPENAI_MODEL", "NousResearch/DeepHermes-3-Llama-3-8B-Preview"))
                base_url = sc_data.get("base_url", os.getenv("OPENAI_API_BASE", "http://localhost:9004/v1"))
                num_requests = sc_data.get("num_requests_for_eval", 256)

                # Special handling for api_key: YAML -> Env Var -> "x"
                api_key = sc_data.get("api_key")
                if not api_key:
                    api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    api_key = "x"
                    logger.warning("API key not found in config or OPENAI_API_KEY env var. Defaulting to 'x'.")
                else:
                    masked_key = api_key[:4] + "****" + api_key[-4:] if api_key != "x" and len(api_key) > 8 else api_key
                    logger.debug(f"Using API key: {masked_key}")

                openai_config_args = {
                    "model_name": model_name,
                    "api_key": api_key,
                    "num_requests_for_eval": num_requests,
                    "base_url": base_url,
                }
                logger.info(f"Creating OpenaiConfig with args: model='{model_name}', base_url='{base_url}', key_present={api_key != 'x'}, requests={num_requests}")
                server_confs.append(OpenaiConfig(**openai_config_args))

            # Provide a default server config ONLY if server_configs was completely missing
            if "server_configs" not in raw_from_yaml:
                logger.warning("No 'server_configs' section found in YAML, creating default server config.")
                default_api_key = os.getenv("OPENAI_API_KEY")
                if not default_api_key:
                    default_api_key = "x"
                    logger.warning("Defaulting API key to 'x' for default server config.")

                server_confs = [
                    OpenaiConfig(
                        model_name=os.getenv("OPENAI_MODEL", "NousResearch/DeepHermes-3-Llama-3-8B-Preview"),
                        base_url=os.getenv("OPENAI_API_BASE", "http://localhost:9004/v1"),
                        api_key=default_api_key,
                        num_requests_for_eval=256,
                    )
                ]
                logger.info(f"Created default OpenaiConfig: model='{server_confs[0].model_name}', base_url='{server_confs[0].base_url}', key_present={server_confs[0].api_key != 'x'}")

            return env_conf, server_confs

        except Exception as e:
            logger.error(f"Error loading config: {e}. Using hardcoded fallback.")
            # Fallback if any error occurs, including validation errors from missing essential fields
            return DatasetEnvConfig(dataset_name="default_dataset_on_error", prompt_field="default_prompt_on_error", debug_mode=False), [
                OpenaiConfig(
                    model_name=os.getenv(
                        "OPENAI_MODEL", "NousResearch/DeepHermes-3-Llama-3-8B-Preview"
                    ),
                    base_url=os.getenv("OPENAI_API_BASE", "http://localhost:9004/v1"),
                    api_key=os.getenv("OPENAI_API_KEY", "x"),
                    num_requests_for_eval=256,
                )
            ]


if __name__ == "__main__":
    DatasetEnv.cli()
