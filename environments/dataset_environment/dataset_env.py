import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
from datasets import load_dataset
from pydantic import Field
import wandb

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataItem,
)
from atroposlib.envs.reward_fns import registry
from atroposlib.envs.reward_fns.combined_reward import CombinedReward
from atroposlib.type_definitions import Item
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DatasetEnvConfig(BaseEnvConfig):
    dataset_name: Optional[str] = Field(None, description="HuggingFace dataset name")
    dataset_config: Optional[str] = Field(
        None, description="Dataset configuration name"
    )
    split: str = Field("train", description="Dataset split to use")
    dataset_path: Optional[str] = Field(
        None, description="Local path to dataset (alternative to dataset_name)"
    )
    prompt_field: Optional[str] = Field(
        None, description="Field in dataset to use as prompt"
    )
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
        None,
        description="List of reward functions to apply (string names or full configs)",
    )
    temperature: float = Field(0.7, description="Temperature for generation")
    top_p: float = Field(0.9, description="Top-p for generation")
    max_tokens: int = Field(16384, description="Maximum tokens for generation")
    eval_dataset_name: Optional[str] = Field(
        None, description="Evaluation dataset name"
    )
    eval_dataset_config: Optional[str] = Field(
        None, description="Evaluation dataset config"
    )
    eval_split: Optional[str] = Field(None, description="Evaluation dataset split")
    debug_mode: bool = Field(False, description="Enable debug logging")
    use_wandb: bool = Field(False, description="Whether to use wandb for logging")


class DatasetEnv(BaseEnv):
    name = "dataset"

    def __init__(
        self, config: DatasetEnvConfig, server_configs, slurm=True, testing=False
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config = config

        if self.config.ground_truth_field == "None":
            logger.warning(
                "DatasetEnv.__init__: Configured 'ground_truth_field' was the string \"None\". Setting to Python None."
            )
            self.config.ground_truth_field = None

        self.dataset = None
        self.iter = 0
        self.metric_buffer = {}
        self.debug_mode = config.debug_mode

        if self.debug_mode:
            logger.setLevel(logging.DEBUG)
        else:
            if logger.level == logging.NOTSET or logger.level > logging.WARNING:
                logger.setLevel(logging.WARNING)

        self.reward_function = self._initialize_reward_function()
        logger.warning(
            f"DatasetEnv.__init__: Initialized reward_function type: {type(self.reward_function)}"
        )

        if hasattr(self.reward_function, "rewards") and isinstance(
            self.reward_function.rewards, list
        ):
            constituent_reward_types = [
                type(r).__name__ for r in self.reward_function.rewards
            ]
            logger.warning(
                f"DatasetEnv.__init__: Constituent reward function types: {constituent_reward_types}"
            )
            if self.config.reward_functions and isinstance(
                self.config.reward_functions, list
            ):
                logger.warning(
                    f"DatasetEnv.__init__: Original reward_functions config: {self.config.reward_functions}"
                )
        elif self.reward_function is not None:
            logger.warning(
                f"DatasetEnv.__init__: Single reward function type: {type(self.reward_function).__name__}"
            )
            if self.config.reward_functions and isinstance(
                self.config.reward_functions, list
            ):
                logger.warning(
                    f"DatasetEnv.__init__: Original reward_functions config: {self.config.reward_functions}"
                )
        else:
            logger.warning("DatasetEnv.__init__: self.reward_function is None.")

    def _initialize_reward_function(self):
        if self.config.reward_functions:
            if len(self.config.reward_functions) == 1:
                return registry.create(self.config.reward_functions[0])
            else:
                return CombinedReward(rewards=self.config.reward_functions)
        logger.warning(
            "No reward functions configured (field 'reward_functions' is None or list is empty)."
        )
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

        if self.config.shuffle_dataset:
            logger.info("Shuffling dataset.")
            self.dataset = self.dataset.shuffle(seed=42)

        if self.dataset:
            logger.info(f"Dataset loaded. Number of rows: {len(self.dataset)}")
            if len(self.dataset) > 0:
                logger.info(f"Sample item keys: {list(self.dataset[0].keys())}")
            else:
                logger.warning("Loaded dataset is empty.")
            logger.info(f"Dataset features: {self.dataset.features}")
        else:
            logger.error("Dataset could not be loaded.")

        self.iter = 0
        self.metric_buffer = {}

    async def get_next_item(self) -> Item:
        if not self.dataset:
            await self.setup()

        item_data = self.dataset[self.iter % len(self.dataset)]
        self.iter += 1
        user_msg = {"role": "user", "content": item_data[self.config.prompt_field]}
        prompt = tuple([frozenset(user_msg.items())])

        answer = None
        if self.config.answer_field and self.config.answer_field in item_data:
            answer = item_data[self.config.answer_field]

        ground_truth = None
        if (
            self.config.ground_truth_field
            and self.config.ground_truth_field in item_data
        ):
            ground_truth = item_data[self.config.ground_truth_field]

        return (prompt, answer, ground_truth)

    async def collect_trajectory(self, item: Item) -> Tuple[Optional[ScoredDataItem], List[Item]]:
        """
        Collects a single trajectory, scores it, and returns it as a ScoredDataItem.
        This method is called by the BaseEnv's collect_trajectories method.
        """
        logger.setLevel(logging.DEBUG)
        logger.warning(f"collect_trajectory: item: {item}")
        try:
            user_content = dict(item[0][0])["content"]
            answer = item[1] if len(item) > 1 and item[1] is not None else None
            ground_truth = item[2] if len(item) > 2 and item[2] is not None else None

            messages = []
            if self.config.system_prompt:
                messages.append({"role": "system", "content": self.config.system_prompt})

            messages.append({"role": "user", "content": user_content})

            if self.config.prefill:
                # Add prefill as the last part of the prompt for the model to complete from
                messages.append({"role": "assistant", "content": self.config.prefill})

            logger.warning(f"collect_trajectory: messages: {messages}")

            max_tokens = self.config.max_tokens

            async with self.server.dedicated_server() as server:
                completions_result = await server.chat_completion(
                    messages=messages,
                    n=1,  # Generate a single completion for this trajectory
                    max_tokens=max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                )

            if not completions_result.choices:
                logger.warning("collect_trajectory: No choices returned from server.chat_completion.")
                return None, []

            completion = completions_result.choices[0]
            # This is the text generated by the model AFTER any prefill provided in messages
            model_output_text = completion.message.content

            # Construct the full assistant response content
            response_content = model_output_text
            if self.config.prefill:
                response_content = self.config.prefill + model_output_text

            # Construct full messages list for this trajectory (for tokenization and reward func)
            full_messages_for_trajectory = []
            if self.config.system_prompt:
                full_messages_for_trajectory.append({"role": "system", "content": self.config.system_prompt})
            full_messages_for_trajectory.append({"role": "user", "content": user_content})
            full_messages_for_trajectory.append({"role": "assistant", "content": response_content})

            # Tokenize for trainer
            tokenized_trajectory = tokenize_for_trainer(self.tokenizer, full_messages_for_trajectory)

            # Prepare for reward calculation
            # The reward function expects a list of "formatted completions".
            # For a single trajectory, this is a list containing one item:
            # that item is a list containing the assistant's message.
            assistant_message_for_reward = {"role": "assistant", "content": response_content}
            formatted_single_completion = [[assistant_message_for_reward]]

            reward_value = 0.0
            if self.reward_function:
                try:
                    reward_kwargs = {
                        "solution": answer,
                        "ground_truth": ground_truth,
                        "item": item,
                        "config": self.config,
                    }
                    rewards_list = self.reward_function(formatted_single_completion, **reward_kwargs)
                    if rewards_list and isinstance(rewards_list, list) and rewards_list:
                        reward_value = float(rewards_list[0])
                    else:
                        logger.warning(f"collect_trajectory: Unexpected reward list format or empty list: {rewards_list}")
                except Exception as e:
                    logger.error(f"collect_trajectory: Error applying reward function: {e}")
                    logger.exception(e)
                    # Default to 0.0 if reward calculation fails
            else:
                logger.warning("collect_trajectory: No reward function configured. Defaulting reward to 0.0.")

            # Always populate messages_for_item with the full trajectory for consistency.
            # Downstream consumers (like the trainer via BaseEnvConfig.include_messages
            # or HTML generator) can then decide what to do with them.
            messages_for_item = full_messages_for_trajectory

            scored_data_item = ScoredDataItem(
                tokens=tokenized_trajectory["tokens"],
                masks=tokenized_trajectory["masks"],
                scores=reward_value,
                advantages=None,  # Base class will handle aggregation if needed
                ref_logprobs=None, # Base class will handle aggregation if needed
                messages=messages_for_item,
                group_overrides=None, # Not applicable for single item
                overrides=None        # Not applicable for single item
            )
            logger.debug(f"collect_trajectory: Produced ScoredDataItem with score {reward_value}")
            return scored_data_item, []

        except Exception as e:
            logger.error(f"collect_trajectory: Failed to collect trajectory for item {item}: {e}")
            logger.exception(e)
            return None, []

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

        # If using wandb and a reward function exists that can accept a wandb logger
        if self.config.use_wandb and hasattr(self, "reward_function") and \
           hasattr(self.reward_function, "set_wandb_logger"):
            # BaseEnv.process_manager (or setup_wandb in serve mode) is responsible for wandb.init()
            # We directly use the wandb module if it has an active run.
            if wandb.run:
                self.reward_function.set_wandb_logger(wandb) # Pass the wandb module
            else:
                logger.warning(
                    "DatasetEnv.wandb_log: Wandb configured (use_wandb=True) and reward_function has "
                    "set_wandb_logger, but wandb.run is not active. Skipping."
                )

        await super().wandb_log(metrics)

    @classmethod
    def config_init(
        cls,
        config_name: Optional[str] = None,
        env_config_override: Optional[
            DatasetEnvConfig
        ] = None,  # Added to accept CLI overrides
    ) -> Tuple[DatasetEnvConfig, List[APIServerConfig]]:
        """Load settings from the local configs directory, allowing for CLI overrides."""
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Determine the full config file path
        if config_name:
            if os.path.isabs(config_name) and config_name.endswith(".yaml"):
                cfg_path = config_name  # Already an absolute path to a yaml
            elif config_name.endswith(".yaml"):  # Relative path to a yaml
                # Try relative to CWD first, then relative to script's config dir
                cfg_path_cwd = os.path.join(os.getcwd(), config_name)
                if os.path.exists(cfg_path_cwd):
                    cfg_path = cfg_path_cwd
                else:
                    cfg_path = os.path.join(current_dir, "configs", config_name)
            else:  # It's a name like 'dataset_local', assume it's in script's config dir
                cfg_path = os.path.join(current_dir, "configs", config_name + ".yaml")
        else:  # Fallback to default
            cfg_path = os.path.join(current_dir, "configs", "dataset_default.yaml")

        logger.info(f"Attempting to load configuration from: {cfg_path}")

        raw_from_yaml = {}
        try:
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    raw_from_yaml = yaml.safe_load(f) or {}
                logger.info(f"Loaded config from {cfg_path}")
                logger.warning(
                    f"DatasetEnv.config_init: Raw reward_functions from YAML: {raw_from_yaml.get('reward_functions')}"
                )  # Added log line
            else:
                # This case should ideally be handled by the fallback in the except block if it's critical
                logger.warning(
                    f"Config file not found at {cfg_path}. Will proceed with env_config_override or defaults."
                )

            # Merge CLI overrides: CLI values take precedence over YAML values
            if env_config_override:
                # Get non-default values from the override object provided by Tyro
                override_dict = env_config_override.model_dump(
                    exclude_none=True
                )  # exclude_none to avoid overwriting with None if CLI arg wasn't set
                raw_from_yaml.update(override_dict)  # Update YAML data with CLI data

            # Ensure debug_mode exists before creating config if not set by CLI or YAML
            if "debug_mode" not in raw_from_yaml:
                raw_from_yaml["debug_mode"] = False  # Default if missing

            env_conf = DatasetEnvConfig(**raw_from_yaml)

            # Validate that essential fields are loaded (either from YAML or CLI override)
            if env_conf.dataset_name is None:
                raise ValueError(
                    "dataset_name is required but was not provided by YAML or CLI."
                )
            if env_conf.prompt_field is None:
                raise ValueError(
                    "prompt_field is required but was not provided by YAML or CLI."
                )

            server_confs = []
            # Server config loading from the potentially merged raw_from_yaml
            for sc_data in raw_from_yaml.get("server_configs", []):
                # Get values directly from YAML data
                model_name = sc_data.get(
                    "model_name",
                    os.getenv(
                        "OPENAI_MODEL", "NousResearch/DeepHermes-3-Llama-3-8B-Preview"
                    ),
                )
                base_url = sc_data.get(
                    "base_url", os.getenv("OPENAI_API_BASE", "http://localhost:9004/v1")
                )
                num_requests = sc_data.get("num_requests_for_eval", 256)

                # Special handling for api_key: YAML -> Env Var -> "x"
                api_key = sc_data.get("api_key")
                if not api_key:
                    api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    api_key = "x"
                    logger.warning(
                        "API key not found in config or OPENAI_API_KEY env var. Defaulting to 'x'."
                    )
                else:
                    masked_key = (
                        api_key[:4] + "****" + api_key[-4:]
                        if api_key != "x" and len(api_key) > 8
                        else api_key
                    )
                    logger.debug(f"Using API key: {masked_key}")

                openai_config_args = {
                    "model_name": model_name,
                    "api_key": api_key,
                    "num_requests_for_eval": num_requests,
                    "base_url": base_url,
                }
                logger.info(
                    f"Creating APIServerConfig with args: model='{model_name}', "
                    f"base_url='{base_url}', key_present={api_key != 'x'}, "
                    f"requests={num_requests}"
                )
                server_confs.append(APIServerConfig(**openai_config_args))

            # Provide a default server config ONLY if server_configs was completely missing
            if "server_configs" not in raw_from_yaml:
                logger.warning(
                    "No 'server_configs' section found in YAML, creating default server config."
                )
                default_api_key = os.getenv("OPENAI_API_KEY")
                if not default_api_key:
                    default_api_key = "x"
                    logger.warning(
                        "Defaulting API key to 'x' for default server config."
                    )

                server_confs = [
                    APIServerConfig(
                        model_name=os.getenv(
                            "OPENAI_MODEL",
                            "NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                        ),
                        base_url=os.getenv(
                            "OPENAI_API_BASE", "http://localhost:9004/v1"
                        ),
                        api_key=default_api_key,
                        num_requests_for_eval=256,
                    )
                ]
                logger.info(
                    f"Created default APIServerConfig: model='{server_confs[0].model_name}', "
                    f"base_url='{server_confs[0].base_url}', "
                    f"key_present={server_confs[0].api_key != 'x'}"
                )

            return env_conf, server_confs

        except Exception as e:
            logger.error(f"Error loading config: {e}. Using hardcoded fallback.")
            # Fallback if any error occurs, including validation errors from missing essential fields
            return DatasetEnvConfig(
                dataset_name="default_dataset_on_error",
                prompt_field="default_prompt_on_error",
                debug_mode=False,
            ), [
                APIServerConfig(
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
