import json
import logging
import os
import random
import uuid
from typing import Any, Dict, List, Optional, Tuple

import wandb
from pydantic import Field
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import BaseEnv, BaseEnvConfig, Item, ScoredDataGroup
from atroposlib.envs.server_handling.server_baseline import APIServerConfig
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

try:
    from jsonschema import validate
    from jsonschema.exceptions import ValidationError
except ImportError:
    raise ImportError(
        "Please install the required dependencies for the JSON Schema Conformance environment:"
        " `pip install jsonschema`"
    )

try:
    from jsf import JSF
except ImportError:
    raise ImportError(
        "Please install the required dependencies for the JSON Schema Conformance environment:"
        " `pip install jsf`"
    )


class JSONSchemaConformanceEnvConfig(BaseEnvConfig):
    """
    Configuration for the JSON Schema Conformance Environment.
    """

    # Task type controls
    task_type: str = Field(
        default="generation",
        description="Type of task to perform: 'generation' or 'editing'.",
    )

    # Complexity controls
    # The existing options can be adjusted to control the difficulty of the tasks.
    # For example, if the task is too easy, you could 10X the max nesting depth.
    max_properties: int = Field(
        default=5,
        description="Maximum number of properties in the generated JSON schema.",
    )
    max_nesting_depth: int = Field(
        default=2, description="Maximum nesting depth for objects in the schema."
    )
    property_required_prob: float = Field(
        default=0.5,
        description="Probability for a property in an object to be marked as required.",
    )
    string_pattern_prob: float = Field(
        default=0.2,
        description="Probability of adding a regex `pattern` constraint to a string schema.",
    )
    string_format_prob: float = Field(
        default=0.2,
        description="Probability of adding a `format` constraint (e.g., 'date-time', 'email') to a string schema.",
    )
    number_range_prob: float = Field(
        default=0.3,
        description="Probability of adding `minimum` and `maximum` constraints to a number schema.",
    )
    array_items_range_prob: float = Field(
        default=0.3,
        description="Probability of adding `minItems` and `maxItems` constraints to an array schema.",
    )
    enum_prob: float = Field(
        default=0.1,
        description="Probability of converting a simple type into an `enum` with a few choices.",
    )

    # Logging and evaluation controls
    debug_logging: bool = Field(
        default=False, description="Enable detailed debug logging."
    )
    dump_rollouts: bool = Field(
        default=False,
        description="Whether to dump successful rollouts to JSONL files for analysis.",
    )
    dump_failed_rollouts: bool = Field(
        default=False,
        description="Whether to dump failed rollouts (all scores < 0.2) to JSONL files for debugging.",
    )
    rollout_save_score_threshold: float = Field(
        default=0.9,
        description="Minimum score threshold for saving rollouts (1.0 saves only perfect rollouts).",
        ge=0.0,
        le=1.0,
    )
    eval_set_size: int = Field(
        default=50,
        description="Number of fixed schemas to generate for the evaluation set.",
        ge=10,
    )


class JSONSchemaConformanceServer(BaseEnv):
    """
    An environment for testing an LLM's ability to generate JSON that conforms to a given JSON schema.

    The environment generates a random JSON schema and asks the model to produce a valid JSON object
    that adheres to it. Additionally, the environment can generate editing tasks, where the model is given
    a schema and a valid JSON object that does not conform to it, and must fix the error.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set up debug logging
        self.debug_logging = getattr(self.config, "debug_logging", False)
        if self.debug_logging:
            self.logger = logging.getLogger(f"{self.__class__.__name__}")
            self.logger.setLevel(logging.DEBUG)
            self.logger.propagate = False
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
            self.logger.info("Debug logging enabled for JSONSchemaConformanceServer")
        else:
            # Still need a logger, but it won't output anything unless level is changed
            self.logger = logging.getLogger(f"{self.__class__.__name__}")
            self.logger.addHandler(logging.NullHandler())
            self.logger.propagate = False

        self.common_patterns = {
            "zip_code": r"^\d{5}(-\d{4})?$",
            "phone_number": r"^\+?1?\s?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}$",
            "hex_color": r"^#([a-fA-F0-9]{6}|[a-fA-F0-9]{3})$",
        }
        self.common_formats = ["date-time", "date", "time", "email", "ipv4", "uri"]
        self.common_enums = {
            "status": ["active", "inactive", "pending", "deleted"],
            "role": ["admin", "user", "guest", "editor"],
            "level": ["easy", "medium", "hard"],
        }
        self.fixed_eval_set: List[Dict[str, Any]] = []
        self.eval_metrics: List[Tuple[str, float]] = []
        self.rollouts_for_wandb: List[List[Tuple]] = []

        # Statistics tracking
        self.conformance_buffer: List[float] = []
        self.task_type_counts = {"generation": 0, "editing": 0}
        self.schema_complexity_stats = {
            "properties": [],
            "depth": [],
            "constraints": [],
        }

        # Data dumping setup
        self.run_uuid = str(uuid.uuid4())
        self.rollouts_to_save_buffer: List[Dict[str, Any]] = []
        self.failed_rollouts_to_save_buffer: List[Dict[str, Any]] = []
        self.processed_item_count = 0
        self.save_file_batch_num = 0
        self.failed_save_file_batch_num = 0
        self.datadumps_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "datadumps"
        )

    @classmethod
    def config_init(
        cls,
    ) -> Tuple[JSONSchemaConformanceEnvConfig, List[APIServerConfig]]:
        """
        Initializes the environment and server configurations.
        """
        env_config = JSONSchemaConformanceEnvConfig(
            data_path_to_save_groups="json_schema_conformance_rollouts.jsonl",
            group_size=8,
            use_wandb=False,
            wandb_name="json-schema-conformance",
            debug_logging=False,
            dump_rollouts=True,
        )

        # Load server configs from llm_config.json if it exists
        server_configs = []
        script_dir = os.path.dirname(os.path.abspath(__file__))
        llm_config_path = os.path.join(script_dir, "llm_config.json")
        if os.path.exists(llm_config_path):
            with open(llm_config_path, "r") as f:
                configs = json.load(f)
                for config_dict in configs:
                    server_configs.append(APIServerConfig(**config_dict))

        # Fallback to default if config file is not found or empty
        if not server_configs:
            server_configs = [
                APIServerConfig(model_name="gpt-4o-mini-2024-07-18", timeout=30)
            ]

        return env_config, server_configs

    async def setup(self):
        """
        Performs one-time setup, including generating a fixed evaluation set.
        """
        self.logger.info("Setting up JSONSchemaConformanceServer environment...")
        self._generate_fixed_eval_set()
        self.logger.info(
            f"Generated a fixed evaluation set of {len(self.fixed_eval_set)} items."
        )
        if self.config.dump_rollouts or self.config.dump_failed_rollouts:
            try:
                if not os.path.exists(self.datadumps_dir):
                    os.makedirs(self.datadumps_dir)
                self.logger.info(
                    f"Rollout dump directory ensured at: {self.datadumps_dir}"
                )
            except OSError as e:
                self.logger.error(f"Could not create datadumps directory: {e}")

    def _generate_fixed_eval_set(self):
        """Generates a fixed set of schemas for consistent evaluation."""
        # Use a fixed seed for reproducibility of the evaluation set
        rng_state = random.getstate()
        random.seed(42)

        self.fixed_eval_set = []
        for _ in range(self.config.eval_set_size):
            task_type = random.choices(
                ["generation", "editing"], weights=[0.7, 0.3], k=1
            )[0]
            if task_type == "editing":
                item = self._generate_editing_task_item()
            else:
                item = {
                    "schema": self._generate_random_schema(),
                    "task_type": "generation",
                }
            self.fixed_eval_set.append(item)

        random.setstate(rng_state)

    def _track_schema_complexity(self, schema: Dict[str, Any]):
        """Helper to track complexity metrics of a generated schema."""
        num_properties = len(schema.get("properties", {}))
        # A simple way to estimate depth and constraints
        depth = json.dumps(schema).count('"type": "object"')
        constraints = 0
        if "pattern" in schema:
            constraints += 1
        if "format" in schema:
            constraints += 1
        if "minimum" in schema:
            constraints += 1
        if "minItems" in schema:
            constraints += 1
        if "enum" in schema:
            constraints += 1

        self.schema_complexity_stats["properties"].append(num_properties)
        self.schema_complexity_stats["depth"].append(depth)
        self.schema_complexity_stats["constraints"].append(constraints)

    def _generate_random_schema(
        self,
        depth: int = 0,
        force_type: Optional[str] = None,
        ensure_required_property: bool = False,
    ) -> Dict[str, Any]:
        """
        Generates a random JSON schema with varying types and constraints,
        controlled by the environment's configuration.
        Can force a top-level type for specific use cases like editing tasks.
        """
        if force_type and depth == 0:
            schema_type = force_type
        elif depth >= self.config.max_nesting_depth:
            # At max depth, only allow simple types
            schema_type = random.choice(["string", "number", "integer", "boolean"])
        else:
            schema_type = random.choice(
                ["object", "string", "number", "integer", "boolean", "array"]
            )

        schema: Dict[str, Any] = {"type": schema_type}

        # Potentially convert to an enum, which is a very strict constraint
        if (
            schema_type in ["string", "number", "integer"]
            and random.random() < self.config.enum_prob
        ):
            _, enum_values = random.choice(list(self.common_enums.items()))
            # Ensure enum values match the schema type
            if schema_type == "string":
                schema["enum"] = enum_values
                return schema

        # Add other type-specific constraints
        if schema_type == "string":
            if random.random() < self.config.string_pattern_prob:
                _, pattern_regex = random.choice(list(self.common_patterns.items()))
                schema["pattern"] = pattern_regex
            # To avoid conflicting constraints, don't add format if pattern is already there
            elif random.random() < self.config.string_format_prob:
                schema["format"] = random.choice(self.common_formats)

        elif schema_type in ["number", "integer"]:
            if random.random() < self.config.number_range_prob:
                schema["minimum"] = random.randint(0, 50)
                schema["maximum"] = random.randint(51, 100)

        elif schema_type == "object":
            num_properties = random.randint(1, self.config.max_properties)
            properties = {}
            required = []
            for i in range(num_properties):
                prop_name = f"property_{i+1}"
                properties[prop_name] = self._generate_random_schema(depth + 1)
                if random.random() < self.config.property_required_prob:
                    required.append(prop_name)

            schema["properties"] = properties

            # If we must have a required property for editing, ensure one exists.
            if ensure_required_property and not required and properties:
                required.append(random.choice(list(properties.keys())))

            if required:
                schema["required"] = required

        elif schema_type == "array":
            schema["items"] = self._generate_random_schema(depth + 1)
            if random.random() < self.config.array_items_range_prob:
                schema["minItems"] = random.randint(1, 3)
                schema["maxItems"] = random.randint(3, 5)

        if depth == 0:
            # Only track complexity for top-level schemas
            self._track_schema_complexity(schema)

        return schema

    def _create_error_in_json(
        self, json_data: Any, schema: Dict[str, Any]
    ) -> Optional[Any]:
        """
        Intentionally introduces a single, simple error into a valid JSON object
        based on its schema. Tries to find an easy-to-fix error.
        Returns the modified JSON object with an error, or None if an error
        could not be introduced.
        """
        if not isinstance(json_data, dict) or not schema.get("properties"):
            # Can only reliably edit objects with properties
            if self.debug_logging:
                self.logger.debug(
                    "Cannot create error: not a dictionary or no properties in schema."
                )
            return None  # Indicate failure

        candidates = list(json_data.keys())
        if not candidates:
            if self.debug_logging:
                self.logger.debug("Cannot create error: JSON data has no keys.")
            return None  # Indicate failure

        random.shuffle(candidates)

        for prop_to_alter in candidates:
            prop_schema = schema.get("properties", {}).get(prop_to_alter, {})
            if not prop_schema:
                continue

            prop_type = prop_schema.get("type")

            # Try to introduce a type error
            if prop_type == "string":
                json_data[prop_to_alter] = 12345
                return json_data
            elif prop_type in ["number", "integer"]:
                json_data[prop_to_alter] = "not-a-number"
                return json_data
            elif prop_type == "boolean":
                json_data[prop_to_alter] = "true"  # String, not boolean
                return json_data
            elif prop_type == "array":
                json_data[prop_to_alter] = {"error": "this should be an array"}
                return json_data
            elif prop_type == "object":
                json_data[prop_to_alter] = "just a string now"
                return json_data

        # If no type error was introduced, try to introduce a 'required' violation
        # by deleting a required property.
        required_properties = schema.get("required", [])
        # Find required properties that are actually in the generated json_data
        available_to_delete = [
            prop for prop in required_properties if prop in json_data
        ]

        if available_to_delete:
            prop_to_delete = random.choice(available_to_delete)
            del json_data[prop_to_delete]
            return json_data

        # If all else fails, we couldn't introduce an error
        if self.debug_logging:
            self.logger.debug(
                f"Could not introduce a simple error for schema: {schema}"
            )
        return None

    def _build_prompt(self, item: Dict[str, Any]) -> str:
        """
        Constructs the prompt that will be sent to the LLM.
        The `item` for this synthetic environment is the generated schema itself.
        """
        schema_str = json.dumps(item["schema"], indent=2)

        if item["task_type"] == "editing":
            erroneous_json_str = json.dumps(item["erroneous_json"], indent=2)
            prompt = (
                "The following JSON object should conform to the provided schema, but it contains an error. "
                "Please correct the object and return only the fixed, valid JSON in your response.\n\n"
                f"JSON Schema:\n```json\n{schema_str}\n```\n\n"
                f"Erroneous JSON:\n```json\n{erroneous_json_str}\n```"
            )
        else:  # 'generation'
            prompt = (
                "Please generate a valid JSON object that strictly conforms to the following JSON schema."
                " Your response should only be the JSON object itself, with no other text or explanations.\n\n"
                f"JSON Schema:\n```json\n{schema_str}\n```"
            )
        return prompt

    async def get_next_item(self) -> Dict[str, Any]:
        """
        Generates a single item for the environment, which can be either a
        generation task or an editing task based on the configuration.
        """
        if self.config.task_type == "editing":
            return self._generate_editing_task_item()

        # Default to generation task
        schema = self._generate_random_schema()
        return {"schema": schema, "task_type": "generation"}

    def _generate_editing_task_item(self) -> Dict[str, Any]:
        """
        Creates an item for the schema-aware editing task by generating a schema,
        a valid JSON object, and then introducing an error into it.
        Will retry up to 10 times to create a valid editing task.
        If `jsf` is not installed, it will fall back to a generation task.
        """
        if JSF is None:
            self.logger.warning(
                "`jsf` library not found, required for 'editing' task. "
                "Falling back to a 'generation' task. Run `pip install jsf` to enable editing tasks."
            )
            return {"schema": self._generate_random_schema(), "task_type": "generation"}

        for _ in range(10):  # Retry loop
            # For editing tasks, we must start with an object schema that has at least
            # one required property to guarantee jsf generates a non-empty object.
            schema = self._generate_random_schema(
                force_type="object", ensure_required_property=True
            )
            valid_json = JSF(schema).generate()

            erroneous_json = self._create_error_in_json(valid_json, schema)

            if erroneous_json is not None:
                return {
                    "schema": schema,
                    "erroneous_json": erroneous_json,
                    "task_type": "editing",
                }

        self.logger.warning(
            "Failed to generate a valid editing task after 10 attempts. Falling back to a generation task."
        )
        return {"schema": self._generate_random_schema(), "task_type": "generation"}

    async def evaluate(self, *args, **kwargs):
        """
        Runs evaluation against the fixed set of test schemas and logs the result.
        """
        if self.debug_logging:
            self.logger.debug(
                f"Starting evaluation on {len(self.fixed_eval_set)} items..."
            )

        if not self.fixed_eval_set:
            self.logger.warning("Evaluation set is empty. Skipping evaluation.")
            self.eval_metrics.append(("eval/avg_conformance_score", 0.0))
            return

        eval_results = await tqdm_asyncio.gather(
            *[self.rollout_and_score_eval(item) for item in self.fixed_eval_set]
        )

        if eval_results:
            avg_score = sum(eval_results) / len(eval_results)
        else:
            avg_score = 0.0

        self.eval_metrics.append(("eval/avg_conformance_score", avg_score))

        if self.debug_logging:
            self.logger.debug(
                f"Evaluation complete. Average conformance score: {avg_score:.3f}"
            )

    async def collect_trajectories(
        self, item: Dict[str, Any]
    ) -> Tuple[Optional[ScoredDataGroup], List[Any]]:
        """
        Overrides the base class method to orchestrate the prompt creation,
        model completion, and scoring for this specific environment.
        """
        prompt_text = self._build_prompt(item)
        self.task_type_counts[item["task_type"]] = (
            self.task_type_counts.get(item["task_type"], 0) + 1
        )

        if self.debug_logging:
            self.logger.debug(
                f"Requesting completion from server for task type: {item['task_type']}"
            )
            self.logger.debug(f"Prompt: {prompt_text[:500]}...")

        # Get completions from the model server
        try:
            completions = await self.server.completion(
                prompt=prompt_text,
                n=self.config.group_size,
                max_tokens=self.config.max_token_length,
                temperature=0.7,
            )
        except Exception as e:
            self.logger.error(
                f"An exception occurred during the completion call: {e}", exc_info=True
            )
            return None, []

        if self.debug_logging:
            self.logger.debug("Received completion from server.")

        rollouts = []
        for choice in completions.choices:
            # For scoring and tokenizing, we need the full conversation history
            full_conversation = [
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": choice.text},
            ]
            rollouts.append(full_conversation)

        scores = self.score(rollouts=rollouts, items=[item] * len(rollouts))

        # Prepare the data for the trainer
        scored_data_group = ScoredDataGroup(tokens=[], masks=[], scores=[])
        if scores:
            for i, rollout in enumerate(rollouts):
                score = scores[i]
                if score is None:
                    continue

                tokenized = tokenize_for_trainer(
                    self.tokenizer, rollout, self.config.max_token_length
                )
                scored_data_group["tokens"].append(tokenized["tokens"])
                scored_data_group["masks"].append(tokenized["masks"])
                scored_data_group["scores"].append(score)

        if not scored_data_group["tokens"]:
            return None, []

        await self.add_rollouts_for_wandb(scored_data_group, item)
        await self._dump_rollouts_if_needed(scored_data_group, rollouts, item)

        return scored_data_group, []

    def score(
        self, rollouts: List[List[Dict[str, str]]], items: List[Dict[str, Any]]
    ) -> List[float | None]:
        """
        Calculates the reward for the model's responses.

        The score is 1.0 if the response is a valid JSON object that conforms
        to the schema, 0.2 if it is valid JSON but does not conform, and 0.0 otherwise.
        """
        scores = []
        success_count = 0
        invalid_json_count = 0
        validation_error_count = 0
        example_validation_error = None

        for i, rollout in enumerate(rollouts):
            # The last message in the rollout is the assistant's response
            response_text = rollout[-1]["content"]
            schema = items[i]["schema"]

            # Attempt to parse the response as JSON
            try:
                # The model might wrap the JSON in markdown, so we extract it.
                if "```json" in response_text:
                    json_str = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    # Generic markdown block
                    json_str = response_text.split("```")[1].strip()
                else:
                    json_str = response_text

                parsed_json = json.loads(json_str)
            except (json.JSONDecodeError, IndexError):
                invalid_json_count += 1
                scores.append(0.0)
                continue

            # Validate the parsed JSON against the schema
            try:
                validate(instance=parsed_json, schema=schema)
                success_count += 1
                scores.append(1.0)
            except ValidationError as e:
                validation_error_count += 1
                if not example_validation_error:
                    example_validation_error = e.message
                scores.append(0.2)

        if self.debug_logging:
            log_msg = (
                f"Scoring group -> Success: {success_count}, "
                f"Invalid JSON: {invalid_json_count}, "
                f"Validation Errors: {validation_error_count}"
            )
            if example_validation_error:
                log_msg += f" (Example Error: {example_validation_error})"
            self.logger.debug(log_msg)

        self.conformance_buffer.extend(scores)

        if self.config.ensure_scores_are_not_same and len(set(scores)) <= 1:
            return [None] * len(scores)

        return scores

    async def add_rollouts_for_wandb(
        self,
        scored_data: Optional[ScoredDataGroup],
        item: Item = None,
    ):
        """Add rollouts to wandb logging buffer."""
        if scored_data is None or not scored_data["tokens"]:
            return

        dataset_item = item
        selected_format = dataset_item.get("task_type", "generation")

        num_keep = self.config.num_rollouts_per_group_for_logging
        if num_keep == -1:
            num_keep = len(scored_data["tokens"])
        else:
            num_keep = min(num_keep, len(scored_data["tokens"]))

        rollout_batch = []
        for i in range(num_keep):
            display_convo_text = self.tokenizer.decode(
                scored_data["tokens"][i], skip_special_tokens=True
            )

            rollout_batch.append(
                (
                    display_convo_text,
                    scored_data["scores"][i],
                    selected_format,
                    dataset_item.get("schema", {}),
                )
            )

        if rollout_batch:
            self.rollouts_for_wandb.append(rollout_batch)

        if len(self.rollouts_for_wandb) > self.config.num_rollouts_to_keep:
            self.rollouts_for_wandb.pop(0)

    async def create_rollout_table(self, wandb_metrics: Dict) -> Dict:
        """Create wandb table for rollout visualization."""
        if self.rollouts_for_wandb:
            table = wandb.Table(
                columns=["full_conversation", "score", "task_type", "schema"]
            )
            for group in self.rollouts_for_wandb:
                for entry in group:
                    # wandb tables don't like dicts, so we json.dumps the schema
                    table.add_data(
                        entry[0], entry[1], entry[2], json.dumps(entry[3], indent=2)
                    )
            wandb_metrics["train/rollouts"] = table

        self.rollouts_for_wandb = []
        return wandb_metrics

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Log metrics to wandb with clear, intuitive categories."""
        if wandb_metrics is None:
            wandb_metrics = {}

        # --- Core Performance ---
        if self.conformance_buffer:
            avg_conformance = sum(self.conformance_buffer) / len(
                self.conformance_buffer
            )
            wandb_metrics["performance/avg_conformance_score"] = avg_conformance
            # Log how many were perfect (1.0), partial (0.2), and fail (0.0)
            wandb_metrics["performance/perfect_rate"] = self.conformance_buffer.count(
                1.0
            ) / len(self.conformance_buffer)
            wandb_metrics["performance/partial_rate"] = self.conformance_buffer.count(
                0.2
            ) / len(self.conformance_buffer)
            wandb_metrics["performance/fail_rate"] = self.conformance_buffer.count(
                0.0
            ) / len(self.conformance_buffer)
        self.conformance_buffer = []

        # --- Evaluation Metrics ---
        for key, value in self.eval_metrics:
            wandb_metrics[key] = value
        self.eval_metrics = []

        # --- Environment-Specific Stats ---
        total_tasks = sum(self.task_type_counts.values())
        if total_tasks > 0:
            wandb_metrics["env/generation_task_ratio"] = (
                self.task_type_counts.get("generation", 0) / total_tasks
            )
            wandb_metrics["env/editing_task_ratio"] = (
                self.task_type_counts.get("editing", 0) / total_tasks
            )

        if self.schema_complexity_stats["properties"]:
            wandb_metrics["env/avg_properties"] = sum(
                self.schema_complexity_stats["properties"]
            ) / len(self.schema_complexity_stats["properties"])
            wandb_metrics["env/avg_depth"] = sum(
                self.schema_complexity_stats["depth"]
            ) / len(self.schema_complexity_stats["depth"])
            wandb_metrics["env/avg_constraints"] = sum(
                self.schema_complexity_stats["constraints"]
            ) / len(self.schema_complexity_stats["constraints"])
            # Clear after logging
            self.schema_complexity_stats = {
                "properties": [],
                "depth": [],
                "constraints": [],
            }

        # --- Rollout Table ---
        wandb_metrics = await self.create_rollout_table(wandb_metrics)

        # Use the super-class's wandb_log to actually log the metrics
        await super().wandb_log(wandb_metrics)

    async def rollout_and_score_eval(self, test_item: Dict[str, Any]) -> float:
        """Evaluate a single test item."""
        prompt = self._build_prompt(test_item)

        completion = await self.server.completion(
            prompt=prompt,
            n=1,
            max_tokens=self.config.max_token_length,
            temperature=0.0,  # Use greedy decoding for eval
            split="eval",
        )

        response_text = completion.choices[0].text

        # Re-use the scoring logic
        rollout = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response_text},
        ]
        scores = self.score(rollouts=[rollout], items=[test_item])
        return scores[0] if scores else 0.0

    async def _dump_rollouts_if_needed(
        self,
        scored_data: ScoredDataGroup,
        rollouts: List[List[Dict[str, str]]],
        item: Dict[str, Any],
    ):
        """Saves rollouts to files based on configuration."""
        if not self.config.dump_rollouts and not self.config.dump_failed_rollouts:
            return

        group_scores = scored_data.get("scores", [])
        if not group_scores:
            return

        is_success_group = any(
            score >= self.config.rollout_save_score_threshold for score in group_scores
        )
        is_failure_group = all(score < 0.2 for score in group_scores)

        # --- Save Successful Rollouts ---
        if self.config.dump_rollouts and is_success_group:
            successful_rollouts_to_save = []
            for i, score in enumerate(group_scores):
                if score >= self.config.rollout_save_score_threshold:
                    successful_rollouts_to_save.append(
                        {
                            "conversation": rollouts[i],
                            "score": score,
                        }
                    )
            if successful_rollouts_to_save:
                item_data_to_save = {
                    "item_id": f"item_{self.processed_item_count}",
                    "task_type": item["task_type"],
                    "schema": item["schema"],
                    "rollouts": successful_rollouts_to_save,
                }
                self.rollouts_to_save_buffer.append(item_data_to_save)

        # --- Save Failed Rollouts ---
        if self.config.dump_failed_rollouts and is_failure_group:
            failed_rollouts_this_group = []
            for i, score in enumerate(group_scores):
                failed_rollouts_this_group.append(
                    {
                        "conversation": rollouts[i],
                        "score": score,
                    }
                )
            item_data_to_save = {
                "item_id": f"item_{self.processed_item_count}",
                "task_type": item["task_type"],
                "schema": item["schema"],
                "rollouts": failed_rollouts_this_group,
            }
            self.failed_rollouts_to_save_buffer.append(item_data_to_save)

        self.processed_item_count += 1
        # Save every 100 items
        if self.processed_item_count > 0 and self.processed_item_count % 100 == 0:
            await self._save_rollouts_to_jsonl()
            await self._save_failed_rollouts_to_jsonl()

    async def _save_rollouts_to_jsonl(self):
        """Save successful rollouts to a JSONL file."""
        if not self.rollouts_to_save_buffer:
            return
        file_path = os.path.join(
            self.datadumps_dir,
            f"json_schema_rollouts_{self.run_uuid}_{self.save_file_batch_num:04d}.jsonl",
        )
        try:
            with open(file_path, "w") as f:
                for rollout_dict in self.rollouts_to_save_buffer:
                    json.dump(rollout_dict, f)
                    f.write("\n")
            if self.debug_logging:
                self.logger.debug(
                    f"Saved {len(self.rollouts_to_save_buffer)} rollouts to {file_path}"
                )
            self.rollouts_to_save_buffer.clear()
            self.save_file_batch_num += 1
        except Exception as e:
            self.logger.error(f"Error saving rollouts: {e}")

    async def _save_failed_rollouts_to_jsonl(self):
        """Save failed rollouts to a JSONL file."""
        if not self.failed_rollouts_to_save_buffer:
            return
        file_path = os.path.join(
            self.datadumps_dir,
            f"json_schema_FAILED_rollouts_{self.run_uuid}_{self.failed_save_file_batch_num:04d}.jsonl",
        )
        try:
            with open(file_path, "w") as f:
                for rollout_dict in self.failed_rollouts_to_save_buffer:
                    json.dump(rollout_dict, f)
                    f.write("\n")
            if self.debug_logging:
                self.logger.debug(
                    f"Saved {len(self.failed_rollouts_to_save_buffer)} failed rollouts to {file_path}"
                )
            self.failed_rollouts_to_save_buffer.clear()
            self.failed_save_file_batch_num += 1
        except Exception as e:
            self.logger.error(f"Error saving failed rollouts: {e}")

    async def close(self):
        """Clean up and save any remaining rollouts before exiting."""
        self.logger.info(
            "Closing JSONSchemaConformanceServer and saving remaining rollouts..."
        )
        await self._save_rollouts_to_jsonl()
        await self._save_failed_rollouts_to_jsonl()
        if hasattr(super(), "close"):
            await super().close()


if __name__ == "__main__":
    JSONSchemaConformanceServer.cli()
