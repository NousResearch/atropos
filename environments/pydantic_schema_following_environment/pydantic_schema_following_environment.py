"""
Pydantic Schema Following Environment

This environment trains models to generate JSON that adheres to Pydantic schemas.
It loads schemas dynamically from a HuggingFace dataset and validates model outputs.

Recent improvements (2025-06-01):
1. Fixed Pydantic ValidationError compatibility issues with proper exception handling
2. Enhanced JSON extraction with fallback methods for responses missing proper tags
3. Added comprehensive input validation for tokenization to prevent 'list' object errors
4. Improved system prompt with explicit formatting requirements and examples
5. Reduced max_token_length to prevent overly verbose responses
6. Added robust error handling throughout the scoring pipeline
7. Enhanced debug logging for better troubleshooting
8. MAJOR: Added strict thinking tag validation (similar to MCQA environment)
9. Enforces exactly one <think></think> section followed by <json_output></json_output>
10. Added detailed validation method with comprehensive error reporting

Key Features:
- Dynamic Pydantic model creation from dataset configurations
- Comprehensive data dumping for analysis
- Strict thinking tag validation for consistent response format
- Fallback JSON extraction for improved success rates
- Detailed debug logging and error handling
- Robust validation with detailed error messages

Response Format Requirements:
- Must use exactly ONE <think> opening tag and ONE </think> closing tag
- All reasoning must be inside the thinking tags
- JSON output must be in <json_output></json_output> tags after </think>
- No additional <think> tags allowed after the first </think> closing tag
"""

import asyncio
import json
import logging
import os
import random
import re
import uuid
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from uuid import UUID

import toml  # Added import
import wandb
import xmltodict  # Added
import yaml  # Added import
from datasets import load_dataset
from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    HttpUrl,
    ValidationError,
    field_validator,
    model_validator,
)
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

# Import editing functionality
try:
    from .error_introduction import (
        ErrorIntroductionConfig,
        introduce_error_for_pydantic,
    )
except ImportError:
    # Handle case when running as script or tests
    from error_introduction import ErrorIntroductionConfig, introduce_error_for_pydantic


class StructuredOutputFormat(Enum):
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    XML = "xml"


class OutputContainerFormat(Enum):
    TAGGED = "tagged"
    NONE = "none"
    MARKDOWN = "markdown"


class PydanticEnvConfig(BaseEnvConfig):
    """Custom config class for PydanticSchemaFollowingEnv with additional parameters."""

    dataset_name: str = Field(
        default="justus27/pydantic-adherance-test",
        description="Name of the HuggingFace dataset to load",
    )
    dataset_split: str = Field(
        default="train", description="Dataset split to use (train, test, validation)"
    )
    debug_logging: bool = Field(
        default=True, description="Enable detailed debug logging"
    )
    dump_rollouts: bool = Field(
        default=False,
        description="Whether to dump rollouts to JSONL files for analysis",
    )
    include_messages: bool = Field(
        default=True,
        description="Whether to include messages in the dataset for SFT data generation",
    )
    allowed_structured_formats: Optional[List[StructuredOutputFormat]] = Field(
        default=None,
        description="Optional list of StructuredOutputFormat enums to use for randomization. If None or empty, all supported formats are used.",  # noqa: E501
    )
    allowed_container_formats: Optional[List[OutputContainerFormat]] = Field(
        default=None,
        description="Optional list of OutputContainerFormat enums to use for randomization. If None or empty, all supported formats are used.",  # noqa: E501
    )
    eval_set_percentage: float = Field(
        default=0.1,
        description="Percentage of the dataset to use for the evaluation set (e.g., 0.1 for 10%).",
        ge=0.0,
        le=1.0,
    )

    # Field for task type
    task_type: str = Field(
        default="generation", description="Task type: 'generation' or 'editing'"
    )

    # Error introduction configuration
    error_introduction_seed: Optional[int] = Field(
        default=None, description="Seed for deterministic error introduction"
    )

    error_types_enabled: List[str] = Field(
        default=[
            "type_error",
            "format_error",
            "constraint_error",
            "enum_error",
            "required_field_missing",
        ],
        description="Types of errors to introduce in editing tasks",
    )

    error_introduction_probability: float = Field(
        default=1.0,
        description="Probability of introducing an error (for editing tasks)",
    )

    max_errors_per_item: int = Field(
        default=1, description="Maximum number of errors to introduce per editing task"
    )


class PydanticSchemaFollowingEnv(BaseEnv):
    env_config_cls = PydanticEnvConfig

    def __init__(
        self,
        config: PydanticEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)

        # Set up debug logging FIRST, as it's used by subsequent setup logic
        self.debug_logging = getattr(config, "debug_logging", True)
        if self.debug_logging:
            self.logger = logging.getLogger(f"{self.__class__.__name__}")
            self.logger.setLevel(logging.DEBUG)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
            self.logger.info("Debug logging enabled for PydanticSchemaFollowingEnv")
        else:
            self.logger = logging.getLogger(f"{self.__class__.__name__}")
            self.logger.addHandler(logging.NullHandler())

        self.percent_correct_buffer = list()
        self.eval_metrics = list()
        self.rollouts_for_wandb = []
        self.dataset_items: List[Dict[str, Any]] = []
        self.model_cache: Dict[str, Type[BaseModel]] = {}

        # Determine supported formats based on config or defaults
        if (
            config.allowed_structured_formats
            and len(config.allowed_structured_formats) > 0
        ):
            self.supported_structured_formats = config.allowed_structured_formats
            if self.debug_logging:
                self.logger.info(
                    f"Using configured structured formats: {[f.value for f in self.supported_structured_formats]}"
                )
        else:
            self.supported_structured_formats: List[StructuredOutputFormat] = [
                StructuredOutputFormat.JSON,
                StructuredOutputFormat.YAML,
                StructuredOutputFormat.TOML,
                StructuredOutputFormat.XML,
            ]
            if self.debug_logging:
                self.logger.info(
                    f"Using default structured formats: {[f.value for f in self.supported_structured_formats]}"
                )

        if (
            config.allowed_container_formats
            and len(config.allowed_container_formats) > 0
        ):
            self.supported_container_formats = config.allowed_container_formats
            if self.debug_logging:
                self.logger.info(
                    f"Using configured container formats: {[f.value for f in self.supported_container_formats]}"
                )
        else:
            self.supported_container_formats: List[OutputContainerFormat] = [
                OutputContainerFormat.TAGGED,
                OutputContainerFormat.NONE,
                OutputContainerFormat.MARKDOWN,
            ]
            if self.debug_logging:
                self.logger.info(
                    f"Using default container formats: {[f.value for f in self.supported_container_formats]}"
                )

        # Data dumping setup
        self.run_uuid = str(uuid.uuid4())

        # Buffer for saving rollouts - each item group contains rollouts for one dataset item
        # RolloutDetail: conversation, score, expected_json, model_name, problem_id
        RolloutDetail = Dict[str, Union[List[Dict[str, str]], float, str]]
        ItemGroup = Dict[str, Union[str, List[RolloutDetail]]]
        self.rollouts_to_save_buffer: List[ItemGroup] = []
        self.processed_item_count = 0

        # Create datadumps directory relative to this file
        self.datadumps_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "datadumps"
        )
        self.save_file_batch_num = 0

        if self.debug_logging:
            self.logger.info(
                f"Data dumping {'enabled' if config.dump_rollouts else 'disabled'}"
            )
            if config.dump_rollouts:
                self.logger.info(f"Rollouts will be saved to: {self.datadumps_dir}")

    def _generate_system_prompt(
        self,
        structured_format: StructuredOutputFormat,
        container_format: OutputContainerFormat,
    ) -> str:
        """Generates a system prompt tailored to the selected output and container formats."""
        prompt_lines = [
            f"You are an AI assistant that generates structured data in {structured_format.value.upper()} format according to Pydantic schemas.",  # noqa: E501
            "You may use extremely long chains of thought to deeply consider the problem and deliberate "
            "with yourself via systematic reasoning processes to help come to a correct solution prior to answering.",
            "You should enclose your thoughts and internal monologue inside <think> </think> tags.",
        ]

        example_output = ""
        if structured_format == StructuredOutputFormat.JSON:
            example_output = '{\\n  "field1": "value1",\\n  "field2": "value2"\\n}'
        elif structured_format == StructuredOutputFormat.YAML:
            example_output = "field1: value1\\nfield2: value2"
        elif structured_format == StructuredOutputFormat.TOML:
            example_output = 'field1 = "value1"\\nfield2 = "value2"'
        elif structured_format == StructuredOutputFormat.XML:
            example_output = "<YourModelName>\\n  <field1>value1</field1>\\n  <field2>value2</field2>\\n</YourModelName>"  # Assuming root tag is model name # noqa: E501

        if container_format == OutputContainerFormat.TAGGED:
            tag_name = f"{structured_format.value}_output"
            prompt_lines.extend(
                [
                    f"CRITICAL: Your final {structured_format.value.upper()} output MUST be enclosed within <{tag_name}> </{tag_name}> tags.",  # noqa: E501
                    f"The {structured_format.value.upper()} must be valid and complete. Do not include any text after the closing </{tag_name}> tag.",  # noqa: E501
                    "Example format:",
                    "<think>\\nMy reasoning here...\\n</think>\\n",
                    f"<{tag_name}>\\n{example_output}\\n</{tag_name}>",
                ]
            )
        elif container_format == OutputContainerFormat.MARKDOWN:
            prompt_lines.extend(
                [
                    f"CRITICAL: Your final {structured_format.value.upper()} output MUST be enclosed within a markdown code block (```).",  # noqa: E501
                    f"The {structured_format.value.upper()} must be valid and complete.",
                    "Example format:",
                    "<think>\\nMy reasoning here...\\n</think>\\n",
                    f"```{structured_format.value}\\n{example_output}\\n```",
                ]
            )
        elif container_format == OutputContainerFormat.NONE:
            prompt_lines.extend(
                [
                    f"CRITICAL: Your final {structured_format.value.upper()} output should be provided directly after the closing </think> tag, with no surrounding tags or markdown.",  # noqa: E501
                    f"The {structured_format.value.upper()} must be valid and complete.",
                    "Example format:",
                    "<think>\\nMy reasoning here...\\n</think>\\n",
                    example_output,
                ]
            )

        prompt_lines.extend(
            [
                f"Ensure the generated {structured_format.value.upper()} strictly adheres to the Pydantic model schema and any specific field "  # noqa: E501
                "requirements provided in the user prompt. Generate all required fields for the model, and "  # noqa: E501
                "include optional fields if they make sense in the context or are specified."  # noqa: E501
            ]
        )
        return "\\n".join(prompt_lines)

    @classmethod
    def config_init(cls) -> Tuple[PydanticEnvConfig, List[APIServerConfig]]:
        """Initialize configuration for the environment."""
        env_config = PydanticEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=16,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=250,
            batch_size=1024,
            steps_per_eval=20,
            max_num_workers=16,
            max_token_length=1024 * 12,
            inference_weight=1.0,
            wandb_name="pydantic_schema_following",
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.1,
            dataset_name="justus27/pydantic-adherance-test",
            dataset_split="train",
            debug_logging=False,
            dump_rollouts=False,
            allowed_structured_formats=[
                StructuredOutputFormat.JSON,
                StructuredOutputFormat.YAML,
                StructuredOutputFormat.TOML,
            ],
            allowed_container_formats=[
                OutputContainerFormat.TAGGED,
                OutputContainerFormat.NONE,
                OutputContainerFormat.MARKDOWN,
            ],
            eval_set_percentage=0.005,
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

    def _create_pydantic_model_from_code(
        self, pydantic_config: str, model_name: str
    ) -> Type[BaseModel]:
        """
        Dynamically create a Pydantic model from the provided configuration code.
        This executes the pydantic_config string and extracts the target model.
        """
        if self.debug_logging:
            self.logger.debug(f"Creating Pydantic model '{model_name}' from config")
            self.logger.debug(f"Config length: {len(pydantic_config)} characters")

        # Check cache first
        cache_key = f"{model_name}_{hash(pydantic_config)}"
        if cache_key in self.model_cache:
            if self.debug_logging:
                self.logger.debug(f"Model '{model_name}' found in cache")
            return self.model_cache[cache_key]

        if self.debug_logging:
            self.logger.debug(
                f"Model '{model_name}' not in cache, creating new instance"
            )

        # Create a namespace for executing the pydantic config
        namespace = {
            "BaseModel": BaseModel,
            "model_validator": model_validator,
            "ConfigDict": ConfigDict,
            "ValidationError": ValidationError,
            "HttpUrl": HttpUrl,
            "EmailStr": EmailStr,
            "Field": Field,
            "field_validator": field_validator,
            "List": List,
            "Dict": Dict,
            "Optional": Optional,
            "Union": Union,
            "Any": Any,
            "Literal": getattr(__import__("typing"), "Literal", None),
            "datetime": datetime,
            "date": date,
            "time": getattr(__import__("datetime"), "time"),
            "timedelta": timedelta,
            "Enum": Enum,
            "Decimal": Decimal,
            "UUID": UUID,
            # Add common imports that might be needed
            "typing": __import__("typing"),
            "json": json,
            "re": re,
        }

        try:
            # Execute the pydantic configuration code
            if self.debug_logging:
                self.logger.debug(f"Executing pydantic config for model '{model_name}'")
            exec(pydantic_config, namespace)

            # Extract the target model class
            if model_name in namespace:
                model_class = namespace[model_name]
                self.model_cache[cache_key] = model_class
                if self.debug_logging:
                    self.logger.debug(
                        f"Successfully created and cached model '{model_name}'"
                    )
                    self.logger.debug(
                        f"Model fields: {list(model_class.model_fields.keys())}"
                    )
                return model_class
            else:
                error_msg = (
                    f"Model '{model_name}' not found in the executed pydantic config"
                )
                if self.debug_logging:
                    self.logger.error(error_msg)
                    self.logger.debug(
                        f"Available classes in namespace: {[k for k in namespace.keys() if isinstance(namespace[k], type)]}"  # noqa: E501
                    )
                raise ValueError(error_msg)

        except Exception as e:
            error_msg = f"Error creating Pydantic model '{model_name}': {e}"
            if self.debug_logging:
                self.logger.error(error_msg)
                self.logger.debug(f"Pydantic config that failed:\n{pydantic_config}")
            print(error_msg)
            raise

    async def setup(self):
        """Load the dataset and prepare tasks."""
        if self.debug_logging:
            self.logger.info("Starting environment setup")

        try:
            # Load the dataset - you'll need to specify the correct dataset name/path
            dataset_name = getattr(
                self.config, "dataset_name", "justus27/pydantic-adherance-test"
            )
            dataset_split = getattr(self.config, "dataset_split", "train")

            if self.debug_logging:
                self.logger.info(
                    f"Loading dataset: {dataset_name}, split: {dataset_split}"
                )

            # Load your dataset
            dataset = load_dataset(dataset_name, split=dataset_split)

            if self.debug_logging:
                self.logger.info(
                    f"Dataset loaded successfully. Total items: {len(dataset)}"
                )

            # Convert to list for easier handling
            self.dataset_items = list(dataset)

            if self.debug_logging:
                self.logger.debug(
                    f"Sample dataset item keys: {list(self.dataset_items[0].keys()) if self.dataset_items else 'No items'}"  # noqa: E501
                )
                if self.dataset_items:
                    sample_item = self.dataset_items[0]
                    self.logger.debug(
                        f"Sample problem_id: {sample_item.get('problem_id', 'N/A')}"
                    )
                    self.logger.debug(
                        f"Sample task_type: {sample_item.get('task_type', 'N/A')}"
                    )
                    self.logger.debug(
                        f"Sample prompt length: {len(sample_item.get('prompt', ''))}"
                    )

            # Shuffle the dataset
            random.shuffle(self.dataset_items)
            if self.debug_logging:
                self.logger.debug("Dataset shuffled")

            # Split into train and test
            split_idx = int(
                len(self.dataset_items) * (1.0 - self.config.eval_set_percentage)
            )
            self.train_items = self.dataset_items[:split_idx]
            self.test_items = self.dataset_items[split_idx:]

            self.iter = 0

            if self.debug_logging:
                self.logger.info(
                    f"Dataset split complete: {len(self.train_items)} training items, {len(self.test_items)} test items"
                )
                self.logger.info("Environment setup complete")

            print(
                f"PydanticSchemaFollowingEnv setup complete. {len(self.train_items)} training items, {len(self.test_items)} test items."  # noqa: E501
            )

        except Exception as e:
            error_msg = f"Error during setup: {e}"
            if self.debug_logging:
                self.logger.error(error_msg)
            print(error_msg)
            # Fallback to empty lists if dataset loading fails
            self.train_items = []
            self.test_items = []
            self.iter = 0

    async def get_next_item(self) -> Tuple[Tuple[frozenset, ...], Dict[str, Any]]:
        """Get the next training item from the dataset."""
        if not self.train_items:
            error_msg = "No training items available. Setup might have failed or dataset is empty."
            if self.debug_logging:
                self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Get the next item cyclically
        dataset_item = self.train_items[self.iter % len(self.train_items)]

        if self.debug_logging:
            self.logger.debug(
                f"Getting item {self.iter % len(self.train_items)} (iteration {self.iter})"
            )
            self.logger.debug(
                f"Item problem_id: {dataset_item.get('problem_id', 'N/A')}"
            )
            self.logger.debug(f"Item task_type: {dataset_item.get('task_type', 'N/A')}")

            # Parse verification info to get model name
            try:
                verification_info = dataset_item.get("verification_info", "{}")
                verification_data = json.loads(verification_info)
                model_name = verification_data.get("model_name", "Unknown")
                self.logger.debug(f"Target model: {model_name}")
            except (json.JSONDecodeError, KeyError):
                self.logger.debug("Could not parse model name from verification_info")

        self.iter += 1

        # Check if this is an editing or generation task
        task_type = dataset_item.get("task_type", "generation")
        if task_type == "editing":
            return self._create_editing_item(dataset_item)
        else:
            return self._create_generation_item(dataset_item)

    def _extract_structured_data_response(
        self,
        text: str,
        container_format: OutputContainerFormat,
        structured_format: StructuredOutputFormat,
    ) -> Optional[str]:
        """Extracts structured data content based on container and structured format, with strict thinking tag validation."""  # noqa: E501
        if self.debug_logging:
            self.logger.debug(
                f"Extracting {structured_format.value} from response (length: {len(text)}), container: {container_format.value}"  # noqa: E501
            )

        # Ensure text is a string
        if not isinstance(text, str):
            if self.debug_logging:
                self.logger.warning(
                    f"Expected string but got {type(text)}, converting to string"
                )
            text = str(text)

        # 1. Validate thinking tags
        think_tags = re.findall(r"<think>", text, re.IGNORECASE)
        think_close_tags = re.findall(r"</think>", text, re.IGNORECASE)

        if len(think_tags) != 1 or len(think_close_tags) != 1:
            if self.debug_logging:
                self.logger.warning(
                    f"Invalid thinking tag structure: {len(think_tags)} open tags, {len(think_close_tags)} close tags. Full text: {text[:500]}"  # noqa: E501
                )
            return None

        # Split the text into thinking and response sections
        # Use a regex that captures the content before, between, and after think tags robustly
        match_think_block = re.match(
            r"(.*?)(<think>.*?</think>)(.*)", text, re.DOTALL | re.IGNORECASE
        )
        if not match_think_block:
            if self.debug_logging:
                self.logger.warning(
                    f"Could not find a complete <think>...</think> block. Full text: {text[:500]}"
                )
            return None

        # thinking_section_plus_prefix = match_think_block.group(1) # Content before <think>
        # thinking_block_content = match_think_block.group(2)  # <think>...</think>
        response_section = match_think_block.group(3)  # Content after </think>

        # Validate that the first <think> tag is indeed the one we captured (no <think> before it)
        if "<think>" in match_think_block.group(1).lower():
            if self.debug_logging:
                self.logger.warning(
                    f"Nested or malformed <think> tags detected before main block. Full text: {text[:500]}"
                )
            return None

        # Check if there are any thinking tags in the response section (after </think>)
        if "<think>" in response_section.lower():
            if self.debug_logging:
                self.logger.warning(
                    f"Found <think> tags in response section after </think>. Full text: {text[:500]}"
                )
            return None

        extracted_content: Optional[str] = None

        # 2. Extract based on container_format from response_section
        if container_format == OutputContainerFormat.TAGGED:
            tag_name = f"{structured_format.value}_output"
            # Loosen regex to allow for attributes in the opening tag if any, and handle varying whitespace
            pattern = rf"<{tag_name}[^>]*>\s*(.*?)\s*</{tag_name}>"
            match = re.search(pattern, response_section, re.DOTALL | re.IGNORECASE)
            if match:
                extracted_content = match.group(1)
                if self.debug_logging:
                    self.logger.debug(
                        f"Extracted using TAGGED ({tag_name}): {len(extracted_content)} chars"
                    )
            else:
                if self.debug_logging:
                    self.logger.warning(
                        f"TAGGED format: Could not find <{tag_name}>...</{tag_name}> tags in response section. Response section: {response_section[:300]}"  # noqa: E501
                    )

        elif container_format == OutputContainerFormat.MARKDOWN:
            # Pattern to match ```language ... ``` or just ``` ... ```
            # It captures the content within the backticks.
            # It optionally matches the language specifier.
            pattern = rf"^\s*```(?:{re.escape(structured_format.value)})?\s*\n(.*?)\n\s*```\s*$"
            match = re.search(
                pattern, response_section.strip(), re.DOTALL | re.IGNORECASE
            )
            if match:
                extracted_content = match.group(1)
                if self.debug_logging:
                    self.logger.debug(
                        f"Extracted using MARKDOWN: {len(extracted_content)} chars"
                    )
            else:
                # Fallback for markdown: if ```<format> content ``` is not found, try ``` content ```
                pattern_no_lang = r"^\s*```\s*\n(.*?)\n\s*```\s*$"
                match_no_lang = re.search(
                    pattern_no_lang, response_section.strip(), re.DOTALL | re.IGNORECASE
                )
                if match_no_lang:
                    extracted_content = match_no_lang.group(1)
                    if self.debug_logging:
                        self.logger.debug(
                            f"Extracted using MARKDOWN (no lang specified): {len(extracted_content)} chars"
                        )
                else:
                    if self.debug_logging:
                        self.logger.warning(
                            f"MARKDOWN format: Could not find ```...``` code block in response section. Response section: {response_section[:300]}"  # noqa: E501
                        )

        elif container_format == OutputContainerFormat.NONE:
            extracted_content = response_section
            if self.debug_logging:
                self.logger.debug(
                    f"Extracted using NONE: {len(extracted_content)} chars"
                )

        # 3. Post-processing
        if extracted_content is not None:
            extracted_content = extracted_content.strip()
            if not extracted_content:  # Empty after stripping
                if self.debug_logging:
                    self.logger.warning("Extracted content is empty after stripping.")
                return None
            if self.debug_logging:
                self.logger.debug(
                    f"Successfully extracted content ({len(extracted_content)} chars). First 100: {extracted_content[:100]}"  # noqa: E501
                )
            return extracted_content
        else:
            if self.debug_logging:
                self.logger.warning(
                    f"Extraction failed for container type {container_format.value}. No content extracted."
                )
            return None

    async def score(
        self,
        rollout_group_data: List[Tuple[Tuple[Dict[str, str], ...], Dict[str, Any]]],
    ) -> Optional[ScoredDataGroup]:
        """Score the rollouts based on Pydantic validation or other structural checks."""
        if self.debug_logging:
            self.logger.debug(f"Scoring {len(rollout_group_data)} rollouts")

        scores_obj = ScoredDataGroup()
        scores_obj["tokens"] = list()
        scores_obj["masks"] = list()
        scores_obj["scores"] = list()
        scores_obj["messages"] = list()

        if not rollout_group_data:
            if self.debug_logging:
                self.logger.warning("No rollout data to score")
            return None

        dataset_item = rollout_group_data[0][1]
        problem_id = dataset_item.get("problem_id", "N/A")
        selected_structured_format = dataset_item["selected_structured_format"]
        selected_container_format = dataset_item["selected_container_format"]

        if self.debug_logging:
            self.logger.debug(
                f"Scoring for problem_id: {problem_id}, structured_format: {selected_structured_format.value}, container_format: {selected_container_format.value}"  # noqa: E501
            )

        verification_info = dataset_item["verification_info"]
        try:
            verification_data = json.loads(verification_info)
            pydantic_config = verification_data["pydantic_config"]
            model_name = verification_data["model_name"]
            if self.debug_logging:
                self.logger.debug(f"Target Pydantic model for validation: {model_name}")
        except (json.JSONDecodeError, KeyError) as e:
            error_msg = f"Error parsing verification_info for {problem_id}: {e}"
            if self.debug_logging:
                self.logger.error(error_msg)
            print(error_msg)
            return None

        try:
            target_model_cls = self._create_pydantic_model_from_code(
                pydantic_config, model_name
            )
        except Exception as e:
            error_msg = (
                f"Error creating Pydantic model {model_name} for {problem_id}: {e}"
            )
            if self.debug_logging:
                self.logger.error(error_msg)
            print(error_msg)
            return None

        valid_count = 0
        invalid_count = 0
        extraction_failures = 0
        parsing_failures = 0

        random.shuffle(rollout_group_data)

        for i, (item_messages, _) in enumerate(rollout_group_data):

            messages_as_dicts = [dict(fs_message) for fs_message in item_messages]
            model_response_text = messages_as_dicts[-1]["content"]

            extracted_str = self._extract_structured_data_response(
                model_response_text,
                selected_container_format,
                selected_structured_format,
            )

            reward = 0.0
            parsed_data = None
            validation_error_msg = "Extraction failed"

            if extracted_str:
                try:
                    if selected_structured_format == StructuredOutputFormat.JSON:
                        parsed_data = json.loads(extracted_str)
                    elif selected_structured_format == StructuredOutputFormat.YAML:
                        parsed_data = yaml.safe_load(extracted_str)
                    elif selected_structured_format == StructuredOutputFormat.TOML:
                        parsed_data = toml.loads(extracted_str)
                    elif selected_structured_format == StructuredOutputFormat.XML:
                        try:
                            data_dict_outer = xmltodict.parse(extracted_str)
                            # Assumption: Pydantic model corresponds to the content *within* the single root XML tag.
                            # xmltodict.parse returns a dict like {'RootTag': actual_data_dict}.
                            # We extract actual_data_dict for validation. More complex XML might require
                            # specific xmltodict process_instructions or more sophisticated unwrapping.
                            if (
                                isinstance(data_dict_outer, dict)
                                and len(data_dict_outer) == 1
                            ):
                                root_key = list(data_dict_outer.keys())[0]
                                parsed_data = data_dict_outer[root_key]
                                if self.debug_logging:
                                    self.logger.debug(
                                        f"Eval item (problem {problem_id}): XML parsed, root key '{root_key}', data: {str(parsed_data)[:100]}..."  # noqa: E501
                                    )
                            else:
                                raise ValueError(
                                    f"XML from xmltodict.parse was not a dict with a single root key as expected. Got: {type(data_dict_outer)}, Keys: {list(data_dict_outer.keys()) if isinstance(data_dict_outer, dict) else 'N/A'}"  # noqa: E501
                                )
                        except xmltodict.expat.ExpatError as e_xml_parse:
                            # No reward variable here, score is returned directly
                            if self.debug_logging:
                                self.logger.debug(
                                    f"Eval item (problem {problem_id}): XML parsing failed (ExpatError): {e_xml_parse}. Extracted XML: {extracted_str[:100]}..."  # noqa: E501
                                )
                            return 0.0  # Return score directly
                        except Exception as e_xml_generic:
                            if self.debug_logging:
                                self.logger.debug(
                                    f"Eval item (problem {problem_id}): XML processing error: {e_xml_generic}. Extracted XML: {extracted_str[:100]}..."  # noqa: E501
                                )
                            return 0.0  # Return score directly

                    if parsed_data is not None and selected_structured_format in [
                        StructuredOutputFormat.JSON,
                        StructuredOutputFormat.YAML,
                        StructuredOutputFormat.TOML,
                        StructuredOutputFormat.XML,
                    ]:
                        is_valid, pydantic_error_msg = (
                            self._validate_parsed_data_against_model(
                                parsed_data, target_model_cls, problem_id
                            )
                        )
                        if is_valid:
                            reward = 1.0
                            valid_count += 1
                            validation_error_msg = None
                            if self.debug_logging and i < 3:
                                self.logger.debug(
                                    f"Rollout {i}: Pydantic validation successful for {selected_structured_format.value}"  # noqa: E501
                                )
                        else:
                            reward = 0.0
                            invalid_count += 1
                            validation_error_msg = pydantic_error_msg
                            if self.debug_logging and i < 3:
                                self.logger.debug(
                                    f"Rollout {i}: Pydantic validation failed for {selected_structured_format.value}. Error: {pydantic_error_msg}"  # noqa: E501
                                )

                except json.JSONDecodeError as e_json:
                    reward = 0.0
                    parsing_failures += 1
                    validation_error_msg = f"JSON parsing failed: {e_json}"
                    if self.debug_logging and i < 3:
                        self.logger.debug(
                            f"Rollout {i}: {validation_error_msg}. Extracted: {extracted_str[:100]}..."
                        )
                except yaml.YAMLError as e_yaml:
                    reward = 0.0
                    parsing_failures += 1
                    validation_error_msg = f"YAML parsing failed: {e_yaml}"
                    if self.debug_logging and i < 3:
                        self.logger.debug(
                            f"Rollout {i}: {validation_error_msg}. Extracted: {extracted_str[:100]}..."
                        )
                except toml.TomlDecodeError as e_toml:
                    reward = 0.0
                    parsing_failures += 1
                    validation_error_msg = f"TOML parsing failed: {e_toml}"
                    if self.debug_logging and i < 3:
                        self.logger.debug(
                            f"Rollout {i}: {validation_error_msg}. Extracted: {extracted_str[:100]}..."
                        )
                except Exception as e_parse:  # Catch any other parsing related errors
                    reward = 0.0
                    parsing_failures += 1
                    validation_error_msg = f"Generic parsing failed for {selected_structured_format.value}: {e_parse}"
                    if self.debug_logging and i < 3:
                        self.logger.debug(
                            f"Rollout {i}: {validation_error_msg}. Extracted: {extracted_str[:100]}..."
                        )
            else:
                reward = 0.0  # No structured data output found or extraction failed
                extraction_failures += 1
                # validation_error_msg is already "Extraction failed"
                if self.debug_logging and i < 3:
                    self.logger.debug(
                        f"Rollout {i}: Extraction failed for {selected_structured_format.value} with container {selected_container_format.value}"  # noqa: E501
                    )

            try:
                if not isinstance(messages_as_dicts, list):
                    if self.debug_logging:
                        self.logger.error(
                            f"Expected list for tokenization, got {type(messages_as_dicts)}"
                        )
                    continue
                for msg_idx, msg in enumerate(messages_as_dicts):
                    if not isinstance(msg, dict):
                        if self.debug_logging:
                            self.logger.error(
                                f"Message {msg_idx} is not a dict: {type(msg)}"
                            )
                        continue  # Skip this rollout if message format is incorrect
                    if "role" not in msg or "content" not in msg:
                        if self.debug_logging:
                            self.logger.error(
                                f"Message {msg_idx} missing required keys: {msg.keys()}"
                            )
                        continue  # Skip this rollout
                    if not isinstance(msg["content"], str):
                        if self.debug_logging:
                            self.logger.warning(
                                f"Converting content to string for message {msg_idx}"
                            )
                        msg["content"] = str(msg["content"])

                out_dict = tokenize_for_trainer(
                    self.tokenizer,
                    messages_as_dicts,
                    include_messages=self.config.include_messages,
                )
                tokens = out_dict["tokens"]
                masks = out_dict["masks"]

            except Exception as e:
                if self.debug_logging:
                    self.logger.error(
                        f"Tokenization failed for rollout {i} (problem: {problem_id}): {e}"
                    )
                    self.logger.debug(
                        f"Messages format: {[type(m) for m in messages_as_dicts]}"
                    )
                continue

            if len([1 for m_val in masks if m_val != -100]) < 10:  # Min context length
                if self.debug_logging:
                    self.logger.debug(
                        f"Skipping rollout {i} (problem: {problem_id}) due to insufficient context length after tokenization."  # noqa: E501
                    )
                continue

            scores_obj["tokens"].append(tokens)
            scores_obj["masks"].append(masks)
            scores_obj["scores"].append(reward)
            # Store original messages (converted to dicts) if available in out_dict, else the modified ones
            scores_obj["messages"].append(out_dict.get("messages", messages_as_dicts))

            self.percent_correct_buffer.append(1.0 if reward == 1.0 else 0.0)

            if len(scores_obj["tokens"]) >= self.config.group_size:
                break

        if self.debug_logging:
            self.logger.info(
                f"Scoring complete for {problem_id} (Format: {selected_structured_format.value}, Container: {selected_container_format.value}): "  # noqa: E501
                f"{valid_count} valid (Pydantic), {invalid_count} invalid (Pydantic), "
                f"{parsing_failures} parsing failures, {extraction_failures} extraction failures."
            )
            if scores_obj["scores"]:
                avg_score = sum(scores_obj["scores"]) / len(scores_obj["scores"])
                self.logger.info(
                    f"Average score for this batch ({problem_id}, Format: {selected_structured_format.value}, Container: {selected_container_format.value}): {avg_score:.3f}"  # noqa: E501
                )

        if not scores_obj["tokens"]:  # No valid examples processed
            if self.debug_logging:
                self.logger.warning(
                    f"No valid examples processed in this batch for {problem_id}"
                )
            return None

        # This condition might need adjustment if 0.0 is a valid signal for non-Pydantic formats
        if (
            all(scores_obj["scores"][0] == score for score in scores_obj["scores"])
            and scores_obj["scores"][0] != 1.0
        ):
            if self.debug_logging:
                self.logger.debug(
                    f"All scores are identical ({scores_obj['scores'][0]}) and not perfect for {problem_id}, returning None for learning signal."  # noqa: E501
                )
            return None

        if all(s == 1.0 for s in scores_obj["scores"]):
            avg_len = sum(len(t) for t in scores_obj["tokens"]) / len(
                scores_obj["tokens"]
            )
            if avg_len > self.config.max_token_length * 0.75:
                scores_obj["scores"] = [s * 0.9 for s in scores_obj["scores"]]
                if self.debug_logging:
                    self.logger.debug(
                        f"Applied length penalty for {problem_id}: avg_len={avg_len}, penalty_threshold={self.config.max_token_length * 0.75}"  # noqa: E501
                    )

        return scores_obj

    async def collect_trajectories(
        self, item: Item
    ) -> Tuple[Optional[ScoredDataGroup], List]:
        """Collect trajectories for a given item."""
        prompt_messages_tuple, dataset_item = item

        if self.debug_logging:
            self.logger.debug(
                f"Collecting trajectories for problem_id: {dataset_item.get('problem_id', 'N/A')}"
            )

        # Convert frozensets to dicts for the API call
        messages_for_api = [dict(fs_message) for fs_message in prompt_messages_tuple]

        prompt_str = self.tokenizer.apply_chat_template(
            messages_for_api, add_generation_prompt=True, tokenize=False
        )

        if self.debug_logging:
            self.logger.debug(f"Generated prompt length: {len(prompt_str)} characters")
            self.logger.debug(
                f"Requesting {self.config.group_size} completions with max_tokens={self.config.max_token_length}, temperature=0.9"  # noqa: E501
            )

        completions = await self.server.completion(
            prompt=prompt_str,
            n=self.config.group_size,
            max_tokens=self.config.max_token_length,
            temperature=0.9,
        )

        if self.debug_logging:
            self.logger.debug(
                f"Received {len(completions.choices)} completions from server"
            )

        to_score_list = []
        for i, choice in enumerate(completions.choices):
            if self.debug_logging and i < 3:  # Log first few completions
                self.logger.debug(
                    f"Completion {i} length: {len(choice.text)} characters"
                )

            # Create a full message list for this choice
            current_trajectory_messages = list(prompt_messages_tuple)
            current_trajectory_messages.append(
                frozenset({"role": "assistant", "content": choice.text}.items())
            )
            to_score_list.append((tuple(current_trajectory_messages), dataset_item))

        scored_data = await self.score(to_score_list)

        if self.debug_logging:
            if scored_data:
                self.logger.debug(
                    f"Scoring successful: {len(scored_data['scores'])} scored items"
                )
            else:
                self.logger.warning("Scoring returned None")

        # Log batch progress for data dumping
        current_batch_progress = self.processed_item_count % 100
        log_message_group_processed = (
            f"GROUP_PROC - Item Iter: {self.iter-1}, Scored Data Present: {bool(scored_data)}, "
            f"Dump Rollouts Cfg: {self.config.dump_rollouts}, "
            f"Total Items Processed (for save): {self.processed_item_count}, Batch Counter: {current_batch_progress}/99"
        )
        if self.debug_logging:
            self.logger.info(log_message_group_processed)

        # Data dumping logic
        if self.debug_logging:
            self.logger.info(
                f"COLLECT_TRAJ - dump_rollouts: {self.config.dump_rollouts}, "
                f"processed_item_count: {self.processed_item_count}, "
                f"current_buffer_size: {len(self.rollouts_to_save_buffer)}"
            )

        if scored_data and self.config.dump_rollouts:
            rollouts_for_current_item = []

            num_scored_rollouts = len(scored_data.get("scores", []))
            conversation_messages_batch = scored_data.get("messages", [])

            # Extract model info from dataset item
            try:
                verification_info = dataset_item.get("verification_info", "{}")
                verification_data = json.loads(verification_info)
                model_name = verification_data.get("model_name", "Unknown")
            except (json.JSONDecodeError, KeyError):
                model_name = "Unknown"

            for i in range(num_scored_rollouts):
                conversation_messages = (
                    conversation_messages_batch[i]
                    if i < len(conversation_messages_batch)
                    else []
                )
                score_for_rollout = scored_data["scores"][i]
                # Extract the generated JSON from the assistant's response
                if conversation_messages:
                    assistant_response = conversation_messages[-1].get("content", "")
                    generated_json = self._extract_structured_data_response(
                        assistant_response,
                        dataset_item["selected_container_format"],
                        dataset_item["selected_structured_format"],
                    )
                else:
                    generated_json = None

                rollouts_for_current_item.append(
                    {
                        "conversation": conversation_messages,
                        "score": score_for_rollout,
                        "expected_json": dataset_item.get("verification_info", ""),
                        "generated_json": generated_json,
                        "model_name": model_name,
                        "problem_id": dataset_item.get("problem_id", "N/A"),
                        "task_type": dataset_item.get("task_type", "N/A"),
                    }
                )

            if rollouts_for_current_item:
                # Use problem_id as the source item ID
                source_item_id = dataset_item.get("problem_id", f"item_{self.iter-1}")

                item_data_to_save = {
                    "item_id": source_item_id,
                    "rollouts": rollouts_for_current_item,
                }
                self.rollouts_to_save_buffer.append(item_data_to_save)
                self.processed_item_count += 1

                if self.debug_logging:
                    self.logger.debug(
                        f"Added {len(rollouts_for_current_item)} rollouts for item {source_item_id}"
                    )

                # Save batch every 100 processed items
                if (
                    self.config.dump_rollouts
                    and self.processed_item_count > 0
                    and self.processed_item_count % 100 == 0
                ):
                    log_msg = (
                        f"Reached {self.processed_item_count} processed items. "
                        f"Triggering save for {len(self.rollouts_to_save_buffer)} item groups."
                    )
                    if self.debug_logging:
                        self.logger.info(log_msg)
                    await self._save_rollouts_to_jsonl()

        return scored_data, []

    async def _save_rollouts_to_jsonl(self):
        """Saves the buffered rollouts to a JSONL file in the datadumps directory."""
        if not self.rollouts_to_save_buffer:
            if self.debug_logging:
                self.logger.info("No rollouts in buffer to save.")
            return

        try:
            if not os.path.exists(self.datadumps_dir):
                os.makedirs(self.datadumps_dir)
                if self.debug_logging:
                    self.logger.info(f"Created directory: {self.datadumps_dir}")
        except OSError as e:
            error_msg = f"Error creating directory {self.datadumps_dir}: {e}"
            if self.debug_logging:
                self.logger.error(error_msg)
            print(error_msg)
            return

        file_path = os.path.join(
            self.datadumps_dir,
            f"pydantic_rollouts_{self.run_uuid}_{self.save_file_batch_num:04d}.jsonl",
        )

        try:
            with open(file_path, "w") as f:
                for rollout_dict in self.rollouts_to_save_buffer:
                    json.dump(rollout_dict, f)
                    f.write("\n")

            success_msg = f"Successfully saved {len(self.rollouts_to_save_buffer)} rollouts to {file_path}"
            if self.debug_logging:
                self.logger.info(success_msg)
            print(success_msg)

            self.rollouts_to_save_buffer.clear()
            self.save_file_batch_num += 1

        except IOError as e:
            error_msg = f"Error writing rollouts to {file_path}: {e}"
            if self.debug_logging:
                self.logger.error(error_msg)
            print(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error saving rollouts to {file_path}: {e}"
            if self.debug_logging:
                self.logger.error(error_msg)
            print(error_msg)

    async def rollout_and_score_eval(self, dataset_item: Dict[str, Any]) -> float:
        """Evaluate a single item from the test set with randomized formats."""
        problem_id = dataset_item.get("problem_id", "N/A")

        # Randomly select formats for evaluation consistency with training
        selected_structured_format = random.choice(self.supported_structured_formats)
        selected_container_format = random.choice(self.supported_container_formats)
        # Store them in dataset_item if needed for logging or other parts, though not strictly for this function's direct logic # noqa: E501
        dataset_item["selected_structured_format"] = selected_structured_format
        dataset_item["selected_container_format"] = selected_container_format

        if self.debug_logging:
            self.logger.debug(
                f"Evaluating item: {problem_id}, structured: {selected_structured_format.value}, container: {selected_container_format.value}"  # noqa: E501
            )

        user_content = dataset_item["prompt"]

        current_system_prompt = self._generate_system_prompt(
            selected_structured_format, selected_container_format
        )

        messages = [
            {"role": "system", "content": current_system_prompt},
            {"role": "user", "content": user_content},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        if self.debug_logging:
            self.logger.debug(
                f"Eval prompt length for {problem_id}: {len(prompt)} characters"
            )

        completion = await self.server.completion(
            prompt=prompt,
            n=1,
            max_tokens=self.config.max_token_length,
            temperature=0.1,
            split="eval",
        )

        model_response_text = completion.choices[0].text

        if self.debug_logging:
            self.logger.debug(
                f"Eval response length for {problem_id}: {len(model_response_text)} characters"
            )

        extracted_str = self._extract_structured_data_response(
            model_response_text, selected_container_format, selected_structured_format
        )

        score = 0.0

        if not extracted_str:
            if self.debug_logging:
                self.logger.debug(
                    f"Eval extraction failed for {problem_id} (Format: {selected_structured_format.value}, Container: {selected_container_format.value})"  # noqa: E501
                )
            return 0.0

        # Attempt to parse and validate
        try:
            verification_info = dataset_item["verification_info"]
            verification_data = json.loads(verification_info)
            pydantic_config = verification_data["pydantic_config"]
            model_name = verification_data["model_name"]

            if self.debug_logging:
                self.logger.debug(
                    f"Eval target Pydantic model for {problem_id}: {model_name}"
                )

            target_model_cls = self._create_pydantic_model_from_code(
                pydantic_config, model_name
            )

            parsed_data = None
            if selected_structured_format == StructuredOutputFormat.JSON:
                parsed_data = json.loads(extracted_str)
            elif selected_structured_format == StructuredOutputFormat.YAML:
                parsed_data = yaml.safe_load(extracted_str)
            elif selected_structured_format == StructuredOutputFormat.TOML:
                parsed_data = toml.loads(extracted_str)
            elif selected_structured_format == StructuredOutputFormat.XML:
                try:
                    data_dict_outer = xmltodict.parse(extracted_str)
                    # Assumption: Pydantic model corresponds to the content *within* the single root XML tag.
                    # xmltodict.parse returns a dict like {'RootTag': actual_data_dict}.
                    # We extract actual_data_dict for validation. More complex XML might require
                    # specific xmltodict process_instructions or more sophisticated unwrapping.
                    if isinstance(data_dict_outer, dict) and len(data_dict_outer) == 1:
                        root_key = list(data_dict_outer.keys())[0]
                        parsed_data = data_dict_outer[root_key]
                        if self.debug_logging:
                            self.logger.debug(
                                f"Eval item (problem {problem_id}): XML parsed, root key '{root_key}', data: {str(parsed_data)[:100]}..."  # noqa: E501
                            )
                    else:
                        raise ValueError(
                            f"XML from xmltodict.parse was not a dict with a single root key as expected. Got: {type(data_dict_outer)}, Keys: {list(data_dict_outer.keys()) if isinstance(data_dict_outer, dict) else 'N/A'}"  # noqa: E501
                        )
                except xmltodict.expat.ExpatError as e_xml_parse:
                    # No reward variable here, score is returned directly
                    if self.debug_logging:
                        self.logger.debug(
                            f"Eval item (problem {problem_id}): XML parsing failed (ExpatError): {e_xml_parse}. Extracted XML: {extracted_str[:100]}..."  # noqa: E501
                        )
                    return 0.0  # Return score directly
                except Exception as e_xml_generic:
                    if self.debug_logging:
                        self.logger.debug(
                            f"Eval item (problem {problem_id}): XML processing error: {e_xml_generic}. Extracted XML: {extracted_str[:100]}..."  # noqa: E501
                        )
                    return 0.0  # Return score directly

            if parsed_data is not None and selected_structured_format in [
                StructuredOutputFormat.JSON,
                StructuredOutputFormat.YAML,
                StructuredOutputFormat.TOML,
                StructuredOutputFormat.XML,
            ]:
                is_valid, pydantic_error_msg = self._validate_parsed_data_against_model(
                    parsed_data, target_model_cls, problem_id
                )
                if is_valid:
                    score = 1.0
                    if self.debug_logging:
                        self.logger.debug(
                            f"Eval Pydantic validation successful for {problem_id} (Format: {selected_structured_format.value})"  # noqa: E501
                        )
                else:
                    score = 0.0  # Already 0.0 by default unless validation passes
                    if self.debug_logging:
                        self.logger.debug(
                            f"Eval Pydantic validation failed for {problem_id} (Format: {selected_structured_format.value}): {pydantic_error_msg}"  # noqa: E501
                        )

        except json.JSONDecodeError as je:
            if self.debug_logging:
                self.logger.debug(
                    f"Eval JSON parsing failed for {problem_id} (Format: {selected_structured_format.value}): {je}. Extracted: {extracted_str[:100]}..."  # noqa: E501
                )
            score = 0.0
        except yaml.YAMLError as ye:
            if self.debug_logging:
                self.logger.debug(
                    f"Eval YAML parsing failed for {problem_id} (Format: {selected_structured_format.value}): {ye}. Extracted: {extracted_str[:100]}..."  # noqa: E501
                )
            score = 0.0
        except toml.TomlDecodeError as te:
            if self.debug_logging:
                self.logger.debug(
                    f"Eval TOML parsing failed for {problem_id} (Format: {selected_structured_format.value}): {te}. Extracted: {extracted_str[:100]}..."  # noqa: E501
                )
            score = 0.0
        except (
            Exception
        ) as e:  # Catch errors in model creation, other parsing, or validation setup
            score = (
                0.0  # Ensure score is 0.0 for any other unexpected error in this block
            )
            if self.debug_logging:
                self.logger.error(
                    f"Error during eval scoring pipeline for {problem_id} (Format: {selected_structured_format.value}): {type(e).__name__}: {e}"  # noqa: E501
                )
        return score

    async def evaluate(self, *args, **kwargs):
        """Run evaluation on the test set."""
        if self.debug_logging:
            self.logger.info("Starting evaluation")

        if not self.test_items:
            warning_msg = "No test items available for evaluation."
            if self.debug_logging:
                self.logger.warning(warning_msg)
            print(warning_msg)
            self.eval_metrics.append(("eval/percent_correct", 0.0))
            return

        # Use a subset for faster evaluation
        items_to_eval = self.test_items[: min(len(self.test_items), 50)]

        if self.debug_logging:
            self.logger.info(f"Evaluating {len(items_to_eval)} items from test set")

        eval_results = await tqdm_asyncio.gather(
            *[self.rollout_and_score_eval(item) for item in items_to_eval]
        )

        # Calculate metrics
        perfect_scores = sum(1 for score in eval_results if score == 1.0)
        if eval_results:
            avg_score = sum(eval_results) / len(eval_results)
            percent_perfect = perfect_scores / len(eval_results)
        else:
            avg_score = 0.0
            percent_perfect = 0.0

        self.eval_metrics.append(("eval/avg_score", avg_score))
        self.eval_metrics.append(("eval/percent_perfect", percent_perfect))

        if self.debug_logging:
            self.logger.info(
                f"Evaluation complete: avg_score={avg_score:.3f}, percent_perfect={percent_perfect:.3f}"
            )
            self.logger.info(f"Perfect scores: {perfect_scores}/{len(eval_results)}")

        print(
            f"Evaluation complete. Avg Score: {avg_score:.3f}, Percent Perfect (1.0): {percent_perfect:.3f}"
        )

    async def add_rollouts_for_wandb(
        self,
        scored_data: Optional[ScoredDataGroup],
        item: Item = None,
    ):
        """Add rollouts to wandb logging."""
        if self.debug_logging:
            self.logger.debug("Adding rollouts for wandb logging")

        if scored_data is None or not scored_data["tokens"]:
            if self.debug_logging:
                self.logger.debug("No scored data to log to wandb")
            return

        # dataset_item contains selected_structured_format and selected_container_format
        dataset_item = item[1] if item and len(item) > 1 else {}
        problem_id = dataset_item.get("problem_id", "N/A")
        selected_structured_format = dataset_item.get(
            "selected_structured_format", StructuredOutputFormat.JSON
        )
        selected_container_format = dataset_item.get(
            "selected_container_format", OutputContainerFormat.TAGGED
        )

        if self.debug_logging:
            self.logger.debug(
                f"Logging rollouts for problem_id: {problem_id}, structured: {selected_structured_format.value}, container: {selected_container_format.value}"  # noqa: E501
            )

        num_keep = self.config.num_rollouts_per_group_for_logging
        if num_keep == -1:
            num_keep = len(scored_data["tokens"])
        else:
            num_keep = min(num_keep, len(scored_data["tokens"]))

        if self.debug_logging:
            self.logger.debug(
                f"Keeping {num_keep} rollouts for logging for {problem_id}"
            )

        rollout_batch = []
        for i in range(num_keep):
            # The full_convo_text comes from scored_data["messages"], which should be the tokenized version.
            # We need the raw model output for extraction.
            # scored_data["messages"][i] should be the list of message dicts for the i-th rollout

            # Reconstruct full_convo_text from messages if possible, or use decoded tokens as fallback
            raw_conversation_messages = scored_data["messages"][i]
            assistant_response_text = ""
            if (
                isinstance(raw_conversation_messages, list)
                and len(raw_conversation_messages) > 0
            ):
                # The last message should be the assistant's response
                if (
                    isinstance(raw_conversation_messages[-1], dict)
                    and raw_conversation_messages[-1].get("role") == "assistant"
                ):
                    assistant_response_text = raw_conversation_messages[-1].get(
                        "content", ""
                    )
                else:  # Try to find assistant message if not last, or if format is unexpected
                    for msg in reversed(raw_conversation_messages):
                        if isinstance(msg, dict) and msg.get("role") == "assistant":
                            assistant_response_text = msg.get("content", "")
                            break

            if not assistant_response_text:  # Fallback if proper message not found
                # This decodes the entire conversation, including system/user prompts
                assistant_response_text = self.tokenizer.decode(
                    scored_data["tokens"][i], skip_special_tokens=True
                )
                if self.debug_logging:
                    self.logger.warning(
                        f"WandB: Could not get raw assistant message for {problem_id}, using full decoded tokens for extraction."  # noqa: E501
                    )

            extracted_output = self._extract_structured_data_response(
                assistant_response_text,
                selected_container_format,
                selected_structured_format,
            )

            try:
                verification_info = dataset_item.get("verification_info", "{}")
                verification_data = json.loads(verification_info)
                pydantic_model_name = verification_data.get("model_name", "N/A")
                expected_schema_info = verification_data.get("pydantic_config", "N/A")
            except (json.JSONDecodeError, KeyError):
                pydantic_model_name = "N/A"
                expected_schema_info = "N/A"
                if self.debug_logging:
                    self.logger.debug(
                        f"Could not extract model name/schema for wandb logging for {problem_id}"
                    )

            # Construct the full conversation text for display from the messages list
            # This ensures the system prompt (which might be dynamic) is correctly shown.
            display_convo_text = ""
            if isinstance(raw_conversation_messages, list):
                try:
                    display_convo_text = self.tokenizer.apply_chat_template(
                        raw_conversation_messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                except Exception as e_tmpl:
                    if self.debug_logging:
                        self.logger.warning(
                            f"WandB: Error applying chat template for {problem_id}: {e_tmpl}. Falling back to joining content."  # noqa: E501
                        )
                    display_convo_text = "\n---\n".join(
                        [
                            str(msg.get("content", ""))
                            for msg in raw_conversation_messages
                        ]
                    )
            else:
                display_convo_text = assistant_response_text

            rollout_batch.append(
                (
                    display_convo_text,
                    scored_data["scores"][i],
                    pydantic_model_name,
                    problem_id,
                    dataset_item.get("task_type", "N/A"),
                    selected_structured_format.value,
                    selected_container_format.value,
                    (
                        extracted_output
                        if extracted_output
                        else "Extraction failed or no output"
                    ),
                    expected_schema_info[:200]
                    + ("..." if len(expected_schema_info) > 200 else ""),
                )
            )

        if rollout_batch:
            self.rollouts_for_wandb.append(rollout_batch)
            if self.debug_logging:
                self.logger.debug(
                    f"Added {len(rollout_batch)} rollouts to wandb queue for {problem_id}"
                )

        if len(self.rollouts_for_wandb) > self.config.num_rollouts_to_keep:
            removed_count = 0
            while len(self.rollouts_for_wandb) > self.config.num_rollouts_to_keep:
                self.rollouts_for_wandb.pop(0)
                removed_count += 1
            if self.debug_logging and removed_count > 0:
                self.logger.debug(
                    f"Removed {removed_count} oldest rollout batch(es) from wandb queue"
                )

    async def create_rollout_table(self, wandb_metrics: Dict) -> Dict:
        """Create wandb table for rollout visualization."""
        if self.debug_logging:
            self.logger.debug(
                f"Creating wandb rollout table with {len(self.rollouts_for_wandb)} batches"
            )

        if self.rollouts_for_wandb:
            table = wandb.Table(
                columns=[
                    "full_conversation",
                    "score",
                    "pydantic_model_name",
                    "problem_id",
                    "task_type",
                    "selected_structured_format",
                    "selected_container_format",
                    "extracted_output",
                    "expected_schema_preview",
                ]
            )
            total_entries = 0
            for group in self.rollouts_for_wandb:
                for entry in group:
                    table.add_data(*entry)
                    total_entries += 1
            wandb_metrics["train/rollouts"] = table

            if self.debug_logging:
                self.logger.debug(
                    f"Created wandb table with {total_entries} total entries"
                )

        self.rollouts_for_wandb = []
        return wandb_metrics

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Log metrics to wandb."""
        if self.debug_logging:
            self.logger.debug("Logging metrics to wandb")

        if wandb_metrics is None:
            wandb_metrics = {}

        if self.percent_correct_buffer:
            percent_perfect = sum(self.percent_correct_buffer) / len(
                self.percent_correct_buffer
            )
            wandb_metrics["train/percent_perfect"] = percent_perfect
            if self.debug_logging:
                self.logger.debug(
                    f"Train percent perfect: {percent_perfect:.3f} (from {len(self.percent_correct_buffer)} samples)"
                )
        else:
            wandb_metrics["train/percent_perfect"] = 0.0
            if self.debug_logging:
                self.logger.debug("No percent_correct_buffer data available")

        self.percent_correct_buffer = list()

        # Add eval metrics
        if self.eval_metrics:
            if self.debug_logging:
                self.logger.debug(
                    f"Adding {len(self.eval_metrics)} eval metrics to wandb"
                )
            for key, value in self.eval_metrics:
                wandb_metrics[key] = value
                if self.debug_logging:
                    self.logger.debug(f"Eval metric: {key} = {value}")

        self.eval_metrics = list()

        # Create rollout table (if any rollouts were collected)
        wandb_metrics = await self.create_rollout_table(wandb_metrics)

        if self.debug_logging:
            self.logger.debug(f"Final wandb metrics: {list(wandb_metrics.keys())}")

        await super().wandb_log(wandb_metrics)

    async def close(self):
        """Clean up and save any remaining rollouts before exiting."""
        if self.debug_logging:
            self.logger.info(
                "Closing PydanticSchemaFollowingEnv. Attempting to save any remaining rollouts..."
            )

        if self.config.dump_rollouts and self.rollouts_to_save_buffer:
            if self.debug_logging:
                self.logger.info(
                    f"Found {len(self.rollouts_to_save_buffer)} rollouts in buffer. Saving now."
                )
            await self._save_rollouts_to_jsonl()
        else:
            if self.debug_logging:
                self.logger.info("No rollouts in buffer to save upon closing.")

        # Call the superclass's close method more robustly
        base_close_method = getattr(super(), "close", None)
        if base_close_method and callable(base_close_method):
            try:
                if asyncio.iscoroutinefunction(base_close_method):
                    await base_close_method()
                else:
                    base_close_method()
            except Exception as e_super_close:
                if self.debug_logging:
                    self.logger.error(f"Error during super().close(): {e_super_close}")
        elif self.debug_logging:
            self.logger.debug(
                "No callable super().close() method found or it does not exist."
            )

        if self.debug_logging:
            self.logger.info("PydanticSchemaFollowingEnv closed.")

    def _validate_parsed_data_against_model(
        self, parsed_data: Any, model_cls: Type[BaseModel], problem_id: str = "N/A"
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate parsed data (e.g., a dictionary) against a Pydantic model and return detailed error info.

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            model_cls.model_validate(parsed_data)
            return True, None
        except ValidationError as ve:
            try:
                # Attempt to get structured error data, which is generally more robust
                error_details = json.dumps(ve.errors(), indent=2)
                # Truncate potentially long error details
                max_detail_len = 250
                if len(error_details) > max_detail_len:
                    error_details_str = f"{error_details[:max_detail_len]}..."
                else:
                    error_details_str = error_details
                error_msg = f"Pydantic validation failed for {problem_id} with {len(ve.errors())} error(s):\n{error_details_str}"  # noqa: E501
            except Exception as format_exc:
                # Fallback if formatting ve.errors() fails for some reason
                error_msg = f"Pydantic validation failed for {problem_id}. Error: {str(ve)[:250]}. (Additionally, formatting error details failed: {str(format_exc)[:50]})"  # noqa: E501

            if self.debug_logging:
                self.logger.debug(error_msg)
            return False, error_msg
        except TypeError as te:
            # Specifically check for the "model" keyword argument issue in ValidationError.__new__
            if (
                "ValidationError.__new__() got an unexpected keyword argument 'model'"
                in str(te)
            ):
                error_msg = f"Pydantic internal TypeError for {problem_id}: {str(te)[:200]}. This may indicate a Pydantic V1/V2 compatibility issue within the dynamic schema definition from the dataset."  # noqa: E501
                if self.debug_logging:
                    self.logger.error(
                        error_msg
                    )  # Log as error due to its specific nature
            else:
                # Handle other TypeErrors normally
                error_msg = f"Unexpected TypeError during validation for {problem_id}: {type(te).__name__}: {str(te)[:100]}"  # noqa: E501
                if self.debug_logging:
                    self.logger.debug(error_msg)
            return False, error_msg
        except Exception as e:
            # Catch any other unexpected exceptions during validation
            error_msg = f"Generic unexpected validation error for {problem_id}: {type(e).__name__}: {str(e)[:100]}"
            if self.debug_logging:
                self.logger.debug(error_msg)
            return False, error_msg

    def _generate_valid_data_for_model(
        self, model_cls: Type[BaseModel]
    ) -> Dict[str, Any]:
        """
        Generate realistic valid data for a Pydantic model using field constraints and semantic hints.

        This method analyzes the model's fields, their types, constraints, and names to generate
        appropriate data that will pass model validation.
        """
        valid_data = {}
        model_fields = model_cls.model_fields

        for field_name, field_info in model_fields.items():
            # Skip optional fields sometimes for variability
            if not field_info.is_required() and random.random() < 0.3:
                continue

            field_value = self._generate_field_value(field_name, field_info)
            if field_value is not None:
                valid_data[field_name] = field_value

        # Validate the generated data to ensure it's correct
        try:
            model_cls.model_validate(valid_data)
            return valid_data
        except ValidationError as e:
            # If validation fails, provide minimal valid data
            if self.debug_logging:
                self.logger.warning(f"Generated data failed validation: {e}")
            return self._generate_minimal_valid_data(model_cls)

    def _generate_field_value(self, field_name: str, field_info) -> Any:
        """Generate a realistic value for a specific field based on its type and constraints."""
        annotation = field_info.annotation

        # Handle Union types (like Optional[T])
        if hasattr(annotation, "__origin__") and annotation.__origin__ is Union:
            # Get the first non-None type
            for arg in annotation.__args__:
                if arg is not type(None):
                    annotation = arg
                    break

        # Handle List types
        if hasattr(annotation, "__origin__") and annotation.__origin__ is list:
            item_type = annotation.__args__[0] if annotation.__args__ else str
            return [
                self._generate_value_for_type(item_type, f"{field_name}_item")
                for _ in range(random.randint(1, 3))
            ]

        # Handle Dict types
        if hasattr(annotation, "__origin__") and annotation.__origin__ is dict:
            key_type = annotation.__args__[0] if len(annotation.__args__) > 0 else str
            value_type = annotation.__args__[1] if len(annotation.__args__) > 1 else str
            return {
                self._generate_value_for_type(
                    key_type, "key"
                ): self._generate_value_for_type(value_type, f"{field_name}_value")
                for _ in range(random.randint(1, 2))
            }

        # Generate value based on type
        return self._generate_value_for_type(annotation, field_name)

    def _generate_value_for_type(
        self, type_annotation: Type, field_name: str = ""
    ) -> Any:
        """Generate a value for a specific type, using field name as hint for semantic meaning."""
        field_name_lower = field_name.lower()

        # Special Pydantic types
        if type_annotation is EmailStr or "email" in field_name_lower:
            domains = ["example.com", "test.org", "sample.net", "demo.io"]
            names = ["user", "admin", "test", "demo", "sample"]
            return f"{random.choice(names)}{random.randint(1, 99)}@{random.choice(domains)}"

        if (
            type_annotation is HttpUrl
            or "url" in field_name_lower
            or "link" in field_name_lower
        ):
            domains = ["example.com", "test.org", "sample.net"]
            paths = ["api/v1", "users/profile", "data/export", "files/download"]
            return f"https://{random.choice(domains)}/{random.choice(paths)}"

        if (
            type_annotation is UUID
            or "uuid" in field_name_lower
            or "id" in field_name_lower
        ):
            return str(uuid.uuid4())

        # Date/Time types
        if type_annotation is datetime:
            return datetime.now() - timedelta(days=random.randint(0, 365))

        if type_annotation is date:
            return (datetime.now() - timedelta(days=random.randint(0, 365))).date()

        # Numeric types with semantic hints
        if type_annotation is int or type_annotation is float:
            if "age" in field_name_lower:
                return random.randint(18, 80)
            elif "year" in field_name_lower:
                return random.randint(2000, 2024)
            elif "score" in field_name_lower or "rating" in field_name_lower:
                return random.randint(1, 100)
            elif "price" in field_name_lower or "cost" in field_name_lower:
                return round(random.uniform(10.0, 1000.0), 2)
            elif "count" in field_name_lower or "quantity" in field_name_lower:
                return random.randint(1, 100)
            else:
                return (
                    random.randint(1, 100)
                    if type_annotation is int
                    else round(random.uniform(1.0, 100.0), 2)
                )

        if type_annotation is Decimal:
            return Decimal(str(round(random.uniform(1.0, 100.0), 2)))

        if type_annotation is bool:
            return random.choice([True, False])

        # String types with semantic hints
        if type_annotation is str:
            if "name" in field_name_lower:
                first_names = ["John", "Jane", "Alice", "Bob", "Charlie", "Diana"]
                last_names = [
                    "Smith",
                    "Johnson",
                    "Williams",
                    "Brown",
                    "Jones",
                    "Garcia",
                ]
                if "first" in field_name_lower:
                    return random.choice(first_names)
                elif "last" in field_name_lower:
                    return random.choice(last_names)
                else:
                    return f"{random.choice(first_names)} {random.choice(last_names)}"
            elif "title" in field_name_lower:
                titles = [
                    "Senior Developer",
                    "Product Manager",
                    "Data Scientist",
                    "UX Designer",
                ]
                return random.choice(titles)
            elif "description" in field_name_lower:
                descriptions = [
                    "A comprehensive solution",
                    "High-quality product",
                    "Innovative approach",
                    "User-friendly design",
                ]
                return random.choice(descriptions)
            elif "address" in field_name_lower:
                return f"{random.randint(100, 9999)} Main St, City, State {random.randint(10000, 99999)}"
            elif "phone" in field_name_lower:
                return f"+1-{random.randint(100, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
            else:
                return f"sample_{field_name_lower}_{random.randint(1, 999)}"

        # Enum types
        if hasattr(type_annotation, "__bases__") and Enum in type_annotation.__bases__:
            return random.choice(list(type_annotation)).value

        # Nested BaseModel
        if (
            hasattr(type_annotation, "__bases__")
            and BaseModel in type_annotation.__bases__
        ):
            return self._generate_valid_data_for_model(type_annotation)

        # Fallback for unknown types
        return f"generated_value_{random.randint(1, 999)}"

    def _generate_minimal_valid_data(
        self, model_cls: Type[BaseModel]
    ) -> Dict[str, Any]:
        """Generate minimal valid data when constraint-based generation fails."""
        valid_data = {}
        model_fields = model_cls.model_fields

        for field_name, field_info in model_fields.items():
            if field_info.is_required():
                annotation = field_info.annotation

                # Handle Union types (like Optional[T])
                if hasattr(annotation, "__origin__") and annotation.__origin__ is Union:
                    for arg in annotation.__args__:
                        if arg is not type(None):
                            annotation = arg
                            break

                # Generate simple fallback values
                if annotation is int:
                    valid_data[field_name] = 1
                elif annotation is float:
                    valid_data[field_name] = 1.0
                elif annotation is bool:
                    valid_data[field_name] = True
                elif annotation is str or annotation is EmailStr:
                    valid_data[field_name] = "valid_example"
                elif annotation is HttpUrl:
                    valid_data[field_name] = "https://example.com"
                elif annotation is UUID:
                    valid_data[field_name] = str(uuid.uuid4())
                elif annotation is datetime:
                    valid_data[field_name] = datetime.now()
                elif annotation is date:
                    valid_data[field_name] = date.today()
                else:
                    valid_data[field_name] = "fallback_value"

        return valid_data

    def _generate_editing_task_prompt(
        self,
        dataset_item: Dict[str, Any],
        structured_format: StructuredOutputFormat,
        container_format: OutputContainerFormat,
    ) -> str:
        """Generate prompt for editing tasks."""
        # Get Pydantic schema and erroneous data
        verification_info = dataset_item.get("verification_info", "{}")
        erroneous_data = dataset_item.get("erroneous_data")

        if not verification_info or erroneous_data is None:
            raise ValueError(
                "Editing task requires both 'verification_info' and 'erroneous_data' fields"
            )

        try:
            verification_data = json.loads(verification_info)
            pydantic_config = verification_data["pydantic_config"]
        except (json.JSONDecodeError, KeyError):
            raise ValueError("Invalid verification_info format for editing task")

        erroneous_data_str = json.dumps(erroneous_data, indent=2)

        base_prompt = self._generate_system_prompt(structured_format, container_format)

        editing_prompt = f"""
{base_prompt}

The following data contains errors and should be corrected to conform to the provided Pydantic model.

Pydantic Model:
```python
{pydantic_config}
```

Erroneous Data:
```json
{erroneous_data_str}
```

Please identify and fix the errors, then provide the corrected data in the requested format.
"""

        return editing_prompt

    def _create_editing_item(
        self, dataset_item: Dict[str, Any]
    ) -> Tuple[Tuple[frozenset, ...], Dict[str, Any]]:
        """Create an editing task item with appropriate prompts."""
        # Randomly select structured and container formats
        selected_structured_format = random.choice(self.supported_structured_formats)
        selected_container_format = random.choice(self.supported_container_formats)

        # Store the selections in the dataset_item
        dataset_item["selected_structured_format"] = selected_structured_format
        dataset_item["selected_container_format"] = selected_container_format

        if self.debug_logging:
            self.logger.debug(
                f"Creating editing task for problem_id: {dataset_item.get('problem_id', 'N/A')}"
            )
            self.logger.debug(
                f"Selected structured format: {selected_structured_format.value}"
            )
            self.logger.debug(
                f"Selected container format: {selected_container_format.value}"
            )

        # If erroneous_data is not already provided, generate it from a valid example
        if "erroneous_data" not in dataset_item:
            # Create valid data first, then introduce errors
            verification_info = dataset_item.get("verification_info", "{}")
            if verification_info:
                try:
                    verification_data = json.loads(verification_info)
                    pydantic_config = verification_data["pydantic_config"]
                    model_name = verification_data["model_name"]

                    # Create the Pydantic model
                    target_model_cls = self._create_pydantic_model_from_code(
                        pydantic_config, model_name
                    )

                    # Generate valid data using Pydantic constraints and field information
                    valid_data = self._generate_valid_data_for_model(target_model_cls)

                    # Introduce errors into the valid data using configuration
                    error_config = ErrorIntroductionConfig.from_env_config(
                        error_types_enabled=self.config.error_types_enabled,
                        max_errors_per_item=self.config.max_errors_per_item,
                        error_introduction_probability=self.config.error_introduction_probability,
                        error_introduction_seed=self.config.error_introduction_seed,
                    )

                    erroneous_data = introduce_error_for_pydantic(
                        valid_data, target_model_cls, config=error_config
                    )

                    if erroneous_data:
                        dataset_item["erroneous_data"] = erroneous_data
                    else:
                        # Fallback: use valid data with a simple error
                        dataset_item["erroneous_data"] = {
                            **valid_data,
                            "invalid_field": "should_not_exist",
                        }

                except Exception as e:
                    if self.debug_logging:
                        self.logger.warning(f"Could not generate erroneous data: {e}")
                    # Use a simple fallback
                    dataset_item["erroneous_data"] = {"error": "could_not_generate"}

        # Generate the editing task prompt
        prompt_content = self._generate_editing_task_prompt(
            dataset_item, selected_structured_format, selected_container_format
        )

        # Create the message structure
        prompt_messages = [
            frozenset({"role": "user", "content": prompt_content}.items()),
        ]

        return tuple(prompt_messages), dataset_item

    def _create_generation_item(
        self, dataset_item: Dict[str, Any]
    ) -> Tuple[Tuple[frozenset, ...], Dict[str, Any]]:
        """Create a generation task item (existing logic extracted for clarity)."""
        # Extract the prompt from the dataset item
        user_content = dataset_item["prompt"]

        if self.debug_logging:
            self.logger.debug(f"Prompt length: {len(user_content)} characters")

        # Randomly select structured and container formats
        selected_structured_format = random.choice(self.supported_structured_formats)
        selected_container_format = random.choice(self.supported_container_formats)

        # Store the selections in the dataset_item
        dataset_item["selected_structured_format"] = selected_structured_format
        dataset_item["selected_container_format"] = selected_container_format

        if self.debug_logging:
            self.logger.debug(
                f"Selected structured format: {selected_structured_format.value}"
            )
            self.logger.debug(
                f"Selected container format: {selected_container_format.value}"
            )

        # Generate the system prompt
        current_system_prompt = self._generate_system_prompt(
            selected_structured_format, selected_container_format
        )

        # Create the message structure
        prompt_messages = [
            frozenset({"role": "system", "content": current_system_prompt}.items()),
            frozenset({"role": "user", "content": user_content}.items()),
        ]

        return tuple(prompt_messages), dataset_item


if __name__ == "__main__":
    PydanticSchemaFollowingEnv.cli()
