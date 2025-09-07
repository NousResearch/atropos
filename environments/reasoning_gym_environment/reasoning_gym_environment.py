import json
import logging
import os
import pkgutil
import random
import re
import sys
import uuid
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import wandb
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

# Add the local reasoning-gym submodule to Python's path
# This allows `import reasoning_gym` to find the local submodule
_SUBMODULE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "reasoning-gym")
)
if _SUBMODULE_DIR not in sys.path:
    sys.path.insert(0, _SUBMODULE_DIR)

# Attempt to import reasoning_gym. If not found, a warning will be issued in _get_task_names.
try:
    import reasoning_gym
    from reasoning_gym.utils import extract_answer
except ImportError as e:
    print(e)
    reasoning_gym = None
    extract_answer = None


system_prompt = (
    "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the "
    "problem and deliberate with yourself via systematic reasoning processes to help come to a correct "
    "solution prior to answering. You should enclose your thoughts and internal monologue inside <think> "
    "</think> tags, and then provide your solution or response to the problem. After your thinking, "
    "make sure to clearly provide your final answer inside <answer></answer> tags. You can provide "
    "context, explanation, etc before and after your answer tags/answer, but you must provide a single "
    "answer and place it inside <answer> tags. You must provide a single answer and place it inside "
    "<answer> tags."
)


class ReasoningGymEnvConfig(BaseEnvConfig):
    """Extended configuration for ReasoningGymEnv with additional fields."""

    dump_rollouts: bool = Field(
        default=False,
        description="Whether to dump successful rollouts (above threshold) to JSONL files.",
    )
    dump_failed_rollouts: bool = Field(
        default=False,
        description="Whether to dump failed rollouts (all 0 scores) to JSONL files for debugging.",
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility.",
    )
    debug_logging: bool = Field(
        default=False,
        description="Enable debug-level logging for more verbose output.",
    )
    suppress_base_env_logs: bool = Field(
        default=True,
        description="Suppress verbose base environment logs (like status dict updates).",
    )
    mask_too_long_completions: bool = Field(
        default=True,
        description="Whether to mask too long completions.",
    )
    rollout_save_score_threshold: float = Field(
        default=0.7,
        description="Minimum score threshold for saving rollouts to data dumps. Only groups with at least one rollout above this threshold will be saved.",  # noqa: E501
    )
    num_eval_samples_per_task: int = Field(
        default=5,
        description="Number of evaluation samples to generate per task for the test set.",
    )
    eval_seed: int = Field(
        default=123,
        description="Seed for generating fixed evaluation set to ensure reproducibility.",
    )
    complexity_mode: Optional[Literal["curriculum", "random"]] = Field(
        default=None,
        description="Complexity control mode: None (default params), 'curriculum' (adaptive difficulty), or 'random' (randomized complexity).",  # noqa: E501
    )
    curriculum_target_accuracy: float = Field(
        default=0.7,
        description="Target accuracy for curriculum mode - difficulty adjusts to maintain this accuracy.",
    )

    def validate_config(self):
        """Validate configuration parameters."""
        if not (0.0 <= self.rollout_save_score_threshold <= 1.0):
            raise ValueError(
                f"rollout_save_score_threshold must be between 0.0 and 1.0, got {self.rollout_save_score_threshold}"
            )
        if self.rollout_save_score_threshold == 1.0:
            print(
                f"Warning: rollout_save_score_threshold is {self.rollout_save_score_threshold}, which may be too strict and result in no saved rollouts."  # noqa: E501
            )
        if self.num_eval_samples_per_task <= 0:
            raise ValueError(
                f"num_eval_samples_per_task must be positive, got {self.num_eval_samples_per_task}"
            )


class ReasoningGymEnv(BaseEnv):
    name = "reasoning_gym"
    env_config_cls = ReasoningGymEnvConfig

    def __init__(
        self,
        config: ReasoningGymEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        # Validate configuration before proceeding
        config.validate_config()
        super().__init__(config, server_configs, slurm, testing)

        # Initialize the logger like swe_rl_env.py
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            # Add a basic stream handler if no handlers are configured
            _handler = logging.StreamHandler()
            _formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            _handler.setFormatter(_formatter)
            self.logger.addHandler(_handler)
            # Set logging level based on config
            log_level = logging.DEBUG if self.config.debug_logging else logging.INFO
            self.logger.setLevel(log_level)
        # Ensure the logger itself is enabled
        self.logger.disabled = False

        # Suppress base environment logs if requested
        if self.config.suppress_base_env_logs:
            # Set the base environment logger to WARNING level to suppress INFO logs
            base_logger = logging.getLogger("atroposlib.envs.base")
            base_logger.setLevel(logging.WARNING)

        # Set max_token_len for base class compatibility

        self.percent_correct_buffer = list()
        self.eval_metrics = list()
        self.rollouts_for_wandb: List[List[Tuple[str, float, str, str]]] = []
        self.task_names: List[str] = []
        self.test_items_with_scorers: List[Tuple[Dict[str, Any], Any]] = []
        self.rng = random.Random()

        self.run_uuid = str(uuid.uuid4())
        self.rollouts_to_save_buffer: List[
            Dict[str, Union[str, List[Dict[str, Union[List[Dict[str, str]], float]]]]]
        ] = []
        self.processed_item_count = 0
        self.datadumps_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data_dumps"
        )
        self.save_file_batch_num = 0

        # For saving failed rollouts (all 0 scores) for debugging
        self.failed_rollouts_to_save_buffer: List[
            Dict[str, Union[str, List[Dict[str, Union[List[Dict[str, str]], float]]]]]
        ] = []
        self.failed_processed_item_count = 0
        self.failed_save_file_batch_num = 0

        # Initialize complexity and curriculum management
        self.task_complexity_levels: Dict[str, float] = (
            {}
        )  # Task -> current complexity level (0.0-1.0)
        self.task_performance_history: Dict[str, List[float]] = (
            {}
        )  # Task -> recent scores
        self.task_curricula: Dict[str, Any] = (
            {}
        )  # Task -> curriculum object (if available)
        self.task_group_counts: Dict[str, int] = (
            {}
        )  # Task -> number of groups processed

    @classmethod
    def config_init(cls) -> Tuple[ReasoningGymEnvConfig, List[APIServerConfig]]:
        env_config = ReasoningGymEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=8,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=250,
            seed=1918,
            batch_size=1024,
            steps_per_eval=25,
            max_token_length=1024 * 8,
            inference_weight=4.0,
            wandb_name="reasoning_gym_think",  # Specific name for reasoning gym
            eval_handling=EvalHandlingEnum.NONE,
            eval_limit_ratio=0.1,
            num_rollouts_per_group_for_logging=4,
            num_rollouts_to_keep=50,
            dump_rollouts=False,
            dump_failed_rollouts=False,
            debug_logging=False,
            suppress_base_env_logs=True,
            rollout_save_score_threshold=0.51,
            num_eval_samples_per_task=5,
            eval_seed=123,
            complexity_mode="random",  # Options: None, "curriculum", "random"
            curriculum_target_accuracy=0.7,
            min_batch_allocation=0.1,
        )
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9004/v1",
                api_key="x",
                num_max_requests_at_once=128,
                num_requests_for_eval=256,
            ),
        ]
        return env_config, server_configs

    def _get_task_names(self) -> List[str]:
        """
        Gets task names from the reasoning_gym DATASETS registry.
        This is more reliable than dynamic discovery via pkgutil.
        Falls back to a predefined list if registry access fails.
        """
        if reasoning_gym is None:
            print(
                "ERROR: The local reasoning-gym submodule could not be imported. "
                "Ensure it is present in 'atropos/environments/reasoning_gym_environment/reasoning-gym' "
                "and is a valid Python package (e.g., has __init__.py)."
            )
            return ["algebra/linear_1d", "arithmetic/add_or_subtract", "leg_counting"]

        discovered_tasks = []
        try:
            # Access the DATASETS registry directly from reasoning_gym.factory
            from reasoning_gym.factory import DATASETS

            discovered_tasks = list(DATASETS.keys())
            print(
                f"Discovered {len(discovered_tasks)} tasks from DATASETS registry: {discovered_tasks[:10]}{'...' if len(discovered_tasks) > 10 else ''}"  # noqa: E501
            )

        except Exception as e:
            print(
                f"WARNING: Could not access DATASETS registry: {e}. "
                "Falling back to manual discovery and validation."
            )

            # Fallback to pkgutil discovery if registry access fails
            try:
                package = reasoning_gym
                if not hasattr(package, "__path__"):
                    raise AttributeError("'reasoning_gym' is not a package")

                base_module_path_parts = package.__name__.split(".")

                print(
                    f"Attempting to discover tasks in package: {package.__name__} from path: {package.__path__}"
                )

                for _, modname, ispkg in pkgutil.walk_packages(
                    path=package.__path__,
                    prefix=package.__name__ + ".",
                    onerror=lambda name: print(
                        f"Error importing module during task discovery: {name}"
                    ),
                ):
                    if not ispkg:
                        module_parts = modname.split(".")
                        if (
                            len(module_parts) > len(base_module_path_parts)
                            and module_parts[: len(base_module_path_parts)]
                            == base_module_path_parts
                        ):

                            task_specific_parts = module_parts[
                                len(base_module_path_parts) :
                            ]

                            # Filter out potential private/internal modules
                            if any(
                                part.startswith("_") for part in task_specific_parts
                            ):
                                continue

                            task_name = "/".join(task_specific_parts)
                            discovered_tasks.append(task_name)

                if discovered_tasks:
                    print(
                        f"Dynamically discovered {len(discovered_tasks)} potential task names from submodule."
                    )

            except Exception as e2:
                print(
                    f"WARNING: Pkgutil discovery also failed: {e2}. Using fallback list."
                )

        if not discovered_tasks:
            print(
                "WARNING: All discovery methods failed. Using fallback list of known tasks."
            )
            # Complete fallback list with all available reasoning-gym tasks
            fallback_tasks = [
                "ab",
                "acre",
                "advanced_geometry",
                "aiw",
                "arc_1d",
                "arc_agi",
                "base_conversion",
                "basic_arithmetic",
                "bf",
                "binary_alternation",
                "binary_matrix",
                "bitwise_arithmetic",
                "boxnet",
                "caesar_cipher",
                "calendar_arithmetic",
                "chain_sum",
                "circuit_logic",
                "codeio",
                "color_cube_rotation",
                "complex_arithmetic",
                "composite",
                "count_bits",
                "count_primes",
                "countdown",
                "course_schedule",
                "cryptarithm",
                "decimal_arithmetic",
                "decimal_chain_sum",
                "dice",
                "emoji_mystery",
                "family_relationships",
                "figlet_font",
                "fraction_simplification",
                "futoshiki",
                "game_of_life",
                "game_of_life_halting",
                "gcd",
                "graph_color",
                "group_anagrams",
                "gsm_symbolic",
                "intermediate_integration",
                "isomorphic_strings",
                "jugs",
                "knight_swap",
                "knights_knaves",
                "largest_island",
                "lcm",
                "leg_counting",
                "letter_counting",
                "letter_jumble",
                "list_functions",
                "mahjong_puzzle",
                "manipulate_matrix",
                "maze",
                "mini_sudoku",
                "modulo_grid",
                "n_queens",
                "needle_haystack",
                "number_filtering",
                "number_format",
                "number_sequence",
                "number_sorting",
                "palindrome_generation",
                "palindrome_partitioning",
                "polynomial_equations",
                "polynomial_multiplication",
                "pool_matrix",
                "power_function",
                "prime_factorization",
                "products",
                "propositional_logic",
                "puzzle24",
                "quantum_lock",
                "ransom_note",
                "rearc",
                "rectangle_count",
                "rotate_matrix",
                "rotten_oranges",
                "rubiks_cube",
                "rush_hour",
                "self_reference",
                "sentence_reordering",
                "shortest_path",
                "simple_equations",
                "simple_geometry",
                "simple_integration",
                "sokoban",
                "spell_backward",
                "spiral_matrix",
                "string_insertion",
                "string_manipulation",
                "string_splitting",
                "string_synthesis",
                "sudoku",
                "syllogism",
                "time_intervals",
                "tower_of_hanoi",
                "tsumego",
                "word_ladder",
                "word_sequence_reversal",
                "word_sorting",
                "zebra_puzzles",
            ]
            return self._validate_discovered_tasks(
                fallback_tasks, reasoning_gym if reasoning_gym else None
            )

        # Validate all discovered tasks
        return self._validate_discovered_tasks(discovered_tasks, reasoning_gym)

    def _validate_discovered_tasks(
        self, task_names_to_validate: List[str], rg_package_or_none: Any
    ) -> List[str]:
        """
        Validates a list of task names by attempting to create a dataset for each.
        Args:
            task_names_to_validate: List of task names (e.g., "domain/task").
            rg_package_or_none: The imported reasoning_gym package, or None if import failed.
        Returns:
            A list of task names that were successfully validated.
        """
        if rg_package_or_none is None:
            print(
                "Validation SKIPPED: reasoning_gym package not available for validation."
            )
            return []

        valid_tasks = []
        print(
            f"Validating {len(task_names_to_validate)} discovered/fallback task names..."
        )
        for task_name in task_names_to_validate:
            try:
                # Use the potentially imported reasoning_gym directly for create_dataset
                _ = rg_package_or_none.create_dataset(task_name, size=1, seed=0)
                valid_tasks.append(task_name)
            except Exception as e:
                print(
                    f"Note: Task '{task_name}' could not be loaded (validation failed): {type(e).__name__} - {e}"
                )  # noqa: E501
                pass

        if not valid_tasks and task_names_to_validate:
            print(
                "WARNING: Validation of discovered/fallback tasks failed for all. This might indicate a systematic issue."  # noqa: E501
            )
            # If validation fails for all, it's safer to return an empty list or a minimal known-good set.
            # However, if task_names_to_validate was non-empty, returning it raw might be a last resort if validation itself is flawed. # noqa: E501
            # For safety, let's prefer an empty list if validation fails completely.
            print("No tasks passed validation.")
            return []

        if not valid_tasks:
            print(
                "CRITICAL WARNING: No valid reasoning-gym tasks could be loaded. Environment may not function."
            )
            # Return an absolute minimal, known-good task if everything else fails.
            # This specific task must exist and be loadable in reasoning_gym.
            try:
                _ = rg_package_or_none.create_dataset("leg_counting", size=1, seed=0)
                print("Falling back to absolute minimal task: 'leg_counting'")
                return ["leg_counting"]
            except Exception:
                print("Absolute fallback 'leg_counting' also failed to load.")
                return []

        print(f"Validated {len(valid_tasks)} tasks for use.")
        return valid_tasks

    async def setup(self):
        # The reasoning_gym import is now handled at the top with sys.path modification.
        if reasoning_gym is None:
            raise ImportError(
                "reasoning-gym library could not be imported from the local submodule. "
                "This environment cannot function. Check submodule presence and integrity."
            )

        self.logger.info("Setting up ReasoningGym environment...")

        self.task_names = (
            self._get_task_names()
        )  # _get_task_names now uses self._validate_discovered_tasks
        if not self.task_names:
            raise ValueError(
                "No reasoning_gym tasks could be loaded. Environment setup failed."
            )

        self.logger.info(
            f"ReasoningGymEnv: Initialized with {len(self.task_names)} tasks."
        )
        self.logger.info(f"Sample tasks: {self.task_names[:5]}")

        # Seed for main RNG used in get_next_item
        # The seed for reasoning_gym dataset creation will be self.iter
        self.rng.seed(self.config.seed)
        self.iter = 0

        # Create a fixed test set for evaluation
        self.logger.info("Generating fixed test set for evaluation...")
        eval_tasks_sample = self.rng.sample(
            self.task_names, min(len(self.task_names), 20)
        )  # Sample 20 tasks for eval

        for task_name in tqdm_asyncio(eval_tasks_sample, desc="Creating eval dataset"):
            try:
                # Each task gets its own dataset instance for evaluation
                # Using a fixed seed for reproducibility of the test set
                dataset = reasoning_gym.create_dataset(
                    task_name,
                    size=self.config.num_eval_samples_per_task,
                    seed=self.config.eval_seed,
                )
                for item in dataset:
                    self.test_items_with_scorers.append((item, dataset))
            except Exception as e:
                self.logger.warning(
                    f"Could not create eval dataset for task '{task_name}': {e}"
                )

        if not self.test_items_with_scorers:
            self.logger.warning(
                "No evaluation items could be generated. Evaluation might be skipped or fail."
            )
        else:
            self.logger.info(
                f"Generated {len(self.test_items_with_scorers)} items for the evaluation test set."
            )

        # Initialize complexity mapping after task names are loaded
        self._initialize_complexity_mapping()

        self.logger.info(
            "ReasoningGym environment setup complete. Ready to start training!"
        )
        self.logger.info(
            f"Configuration: group_size={self.config.group_size}, max_token_length={self.config.max_token_length}, steps_per_eval={self.config.steps_per_eval}"  # noqa: E501
        )
        if self.config.complexity_mode:
            self.logger.info(
                f"Complexity mode: {self.config.complexity_mode} "
                f"(target accuracy: {self.config.curriculum_target_accuracy})"
            )
        if self.config.dump_rollouts:
            self.logger.info(
                f"Data dumping enabled with score threshold: {self.config.rollout_save_score_threshold}"
            )
        self.logger.info(
            "Using strict <answer> tag enforcement: models must use exactly one <answer> tag or receive 0 score"
        )
        self.logger.info(
            "Using dual-format scoring for valid answers: trying both raw answers and <answer>-tagged answers, using higher score"  # noqa: E501
        )

    async def get_next_item(self) -> Optional[Item]:
        """
        Get the next training item by randomly selecting a reasoning_gym task and generating a sample.
        Returns:
            A tuple: (prompt_messages, reasoning_gym_item, reasoning_gym_dataset_object)
            prompt_messages: Formatted for the language model.
            reasoning_gym_item: The raw item from reasoning_gym (dict with 'question', 'answer', 'metadata').
            reasoning_gym_dataset_object: The dataset object from which the item was generated (for scoring).
        """
        if not self.task_names:
            return None  # Should not happen if setup is correct

        selected_task_name = self.rng.choice(self.task_names)

        try:
            # Create a new dataset instance with complexity control
            current_seed = self.config.seed + self.iter  # Vary seed per item
            dataset_obj = self._create_dataset_with_complexity(
                selected_task_name, current_seed
            )
            rg_item = next(iter(dataset_obj))  # Get the single item

            # Log task selection and complexity every 10 items to avoid spam
            if self.iter % 10 == 0:
                complexity_level = self._get_task_complexity_level(selected_task_name)
                complexity_info = (
                    f" (complexity: {complexity_level:.2f})"
                    if self.config.complexity_mode
                    else ""
                )
                self.logger.info(
                    f"Selected task: {selected_task_name}{complexity_info} (iteration {self.iter})"
                )

        except Exception as e:
            self.logger.warning(
                f"Error generating item for task {selected_task_name} with seed {current_seed}: {e}"
            )
            return None  # Skip this item if generation fails

        self.iter += 1

        question_text = rg_item["question"]

        # Construct prompt messages
        prompt_messages = [
            frozenset({"role": "system", "content": system_prompt}.items()),
            frozenset({"role": "user", "content": question_text}.items()),
        ]

        # The 'answer' here is the data needed for scoring later
        return (tuple(prompt_messages), rg_item, dataset_obj)

    def _extract_final_answer(self, text: str) -> Optional[str]:
        """
        Extracts the content from <answer> tags using reasoning-gym's extract_answer function.
        Enforces strict compliance with answer tag instructions.
        Returns None if model doesn't follow instructions properly (no answer tags or multiple answer tags outside think).
        """  # noqa: E501
        if extract_answer is None:
            # If reasoning-gym not available, we can't enforce the format
            if self.config.debug_logging:
                self.logger.debug(
                    "reasoning-gym extract_answer not available, cannot enforce answer tag format"
                )
            return None

        # Check for multiple <answer> tags outside of <think> blocks
        # First, remove all <think>...</think> content to check only the "outside" content
        think_pattern = r"<think>.*?</think>"
        text_outside_think = re.sub(
            think_pattern, "", text, flags=re.DOTALL | re.IGNORECASE
        )

        # Count <answer> tags in the content outside <think> blocks
        answer_pattern = r"<answer>.*?</answer>"
        answer_matches_outside_think = re.findall(
            answer_pattern, text_outside_think, flags=re.DOTALL | re.IGNORECASE
        )

        if len(answer_matches_outside_think) > 1:
            if self.config.debug_logging:
                self.logger.debug(
                    f"Model provided {len(answer_matches_outside_think)} answer tags outside think blocks - failing for not following single answer instruction"  # noqa: E501
                )
            return None

        # Try to extract from <answer> tags using reasoning-gym's function
        answer_content = extract_answer(text, tag_name="answer", strip=True)
        if answer_content is not None:
            if self.config.debug_logging:
                self.logger.debug(
                    f"Successfully extracted answer from <answer> tags: '{answer_content[:100]}{'...' if len(answer_content) > 100 else ''}'"  # noqa: E501
                )
            return answer_content

        # No valid answer tags found - model failed to follow instructions
        if self.config.debug_logging:
            self.logger.debug(
                "No <answer> tags found - model failed to follow answer format instructions"
            )
        return None

    def _score_answer_with_both_formats(
        self, model_answer: str, rg_item: dict, dataset_obj: Any
    ) -> float:
        """
        Score the answer using both formats (with and without <answer> tags) and return the higher score.
        This handles verifiers that expect different formats.
        """
        # Format 1: Just the answer content
        try:
            score1 = dataset_obj.score_answer(answer=model_answer, entry=rg_item)
            score1 = max(0.0, min(1.0, float(score1)))
        except Exception as e:
            task_name = rg_item.get("metadata", {}).get("source_dataset", "unknown")
            self.logger.debug(
                f"Error scoring answer format 1 for task {task_name}: {e}"
            )
            score1 = 0.0

        # Format 2: Answer wrapped in <answer> tags
        answer_with_tags = f"<answer>{model_answer}</answer>"
        try:
            score2 = dataset_obj.score_answer(answer=answer_with_tags, entry=rg_item)
            score2 = max(0.0, min(1.0, float(score2)))
        except Exception as e:
            task_name = rg_item.get("metadata", {}).get("source_dataset", "unknown")
            self.logger.debug(
                f"Error scoring answer format 2 for task {task_name}: {e}"
            )
            score2 = 0.0

        # Return the higher score
        final_score = max(score1, score2)

        # Log which format worked better (only in debug mode)
        if self.config.debug_logging and score1 != score2:
            task_name = rg_item.get("metadata", {}).get("source_dataset", "unknown")
            if score1 > score2:
                self.logger.debug(
                    f"Task {task_name}: Raw answer format scored higher ({score1:.3f} vs {score2:.3f})"
                )
            else:
                self.logger.debug(
                    f"Task {task_name}: Tagged answer format scored higher ({score2:.3f} vs {score1:.3f})"
                )

        return final_score

    async def score(
        self,
        rollout_group_data: List[Tuple[Tuple[Dict[str, str]], Dict[str, Any], Any]],
    ) -> Optional[ScoredDataGroup]:
        """
        Scores a group of rollouts using reasoning_gym's score_answer method.
        Args:
            rollout_group_data: A list of tuples, where each tuple contains:
                - trajectory_messages: The full conversation history for the rollout.
                - rg_item: The original reasoning_gym item (contains 'question', 'answer', 'metadata').
                - dataset_obj: The reasoning_gym dataset object used to generate and score the item.
        Returns:
            ScoredDataGroup with scores between 0.0 and 1.0, or None if no valid items
        """
        scores_container = ScoredDataGroup()
        scores_container["tokens"] = list()
        scores_container["masks"] = list()
        scores_container["scores"] = list()
        scores_container["overrides"] = list()

        if not rollout_group_data:
            return None

        rg_item_for_group = rollout_group_data[0][1]
        dataset_obj_for_group = rollout_group_data[0][2]

        # Shuffle to avoid bias in selection
        random.shuffle(rollout_group_data)

        for trajectory_messages, _, _, finish_reason in rollout_group_data:
            model_full_response = trajectory_messages[-1]["content"]

            # Extract the part of the response that should be the answer
            model_answer_to_score = self._extract_final_answer(model_full_response)

            # If extraction failed (model didn't follow instructions), give 0 score
            if model_answer_to_score is None:
                reward_0_to_1 = 0.0
                if self.config.debug_logging:
                    task_name = rg_item_for_group.get("metadata", {}).get(
                        "source_dataset", "unknown"
                    )
                    self.logger.debug(
                        f"Task {task_name}: Giving 0 score due to failed answer extraction (didn't follow format)"
                    )
            else:
                # Use our dual-format scoring method that tries both raw answer and tagged answer
                reward_0_to_1 = self._score_answer_with_both_formats(
                    model_answer_to_score, rg_item_for_group, dataset_obj_for_group
                )

            # Tokenize the conversation for learning
            out_dict = tokenize_for_trainer(self.tokenizer, trajectory_messages)
            tokens = out_dict["tokens"]
            masks = out_dict["masks"]

            # Remove examples with insufficient context
            if len([1 for i in masks if i != -100]) < 10:
                continue

            scores_container["tokens"].append(tokens)
            scores_container["masks"].append(masks)
            scores_container["scores"].append(reward_0_to_1)
            scores_container["overrides"].append(dict())
            if finish_reason == "length":
                if self.config.mask_too_long_completions:
                    scores_container["overrides"][-1]["set_advantage_to_zero"] = True

            if len(scores_container["tokens"]) >= self.config.group_size:
                break

        if not scores_container["tokens"]:
            self.logger.warning(
                "No valid items were scored in this group - all items had insufficient context or failed scoring"
            )
            return None

        # Record success rate metrics (convert to binary for percent_correct tracking)
        for score_val in scores_container["scores"]:
            self.percent_correct_buffer.append(1.0 if score_val >= 0.5 else 0.0)

        # Update task performance for curriculum adjustment
        task_name = rg_item_for_group.get("metadata", {}).get(
            "source_dataset", "unknown"
        )
        if task_name != "unknown" and self.config.complexity_mode == "curriculum":
            # Use average score of the group for curriculum feedback
            avg_score = sum(scores_container["scores"]) / len(
                scores_container["scores"]
            )
            self._update_task_performance(task_name, avg_score)

        # Calculate and log average score for the current group
        current_scores = scores_container.get("scores", [])
        if current_scores:
            average_score = sum(current_scores) / len(current_scores)
            task_name = rg_item_for_group.get("metadata", {}).get(
                "source_dataset", "unknown_task"
            )
            log_message_main = (
                f"Task: {task_name} | Group average score: {average_score:.4f}"
            )
            if all(s >= 0.5 for s in current_scores):
                self.logger.info(f"{log_message_main} (All correct in this group!)")
            elif all(s == 0.0 for s in current_scores):
                self.logger.info(f"{log_message_main} (All failed - no valid answers!)")
            elif all(s < 0.5 for s in current_scores):
                self.logger.info(
                    f"{log_message_main} (All incorrect but some partial credit!)"
                )
            else:
                self.logger.info(log_message_main)

        # Apply length penalty if all responses are correct (score >= 0.5)
        if all(s >= 0.5 for s in scores_container["scores"]):
            # Calculate token lengths
            token_lengths = [len(token) for token in scores_container["tokens"]]
            if max(token_lengths) == 0:
                return None

            # Get max allowed token length from config
            max_allowed_length = self.config.max_token_length
            # Set threshold at 75% of max_token_length
            length_threshold = max_allowed_length * 0.75

            # Apply modified length penalty with threshold
            new_scores = []
            penalties_applied = 0
            for i, length in enumerate(token_lengths):
                original_score = scores_container["scores"][i]
                if length <= length_threshold:
                    new_scores.append(original_score)
                else:
                    # Calculate how far we are between threshold and max as a percentage
                    percentage_of_range = (length - length_threshold) / (
                        max_allowed_length - length_threshold
                    )
                    percentage_of_range = min(percentage_of_range, 1.0)
                    # Apply linear penalty scaling from original_score down to 0.0
                    penalized_score = original_score * (1.0 - percentage_of_range)
                    new_scores.append(penalized_score)
                    penalties_applied += 1

            if penalties_applied > 0:
                avg_length = sum(token_lengths) / len(token_lengths)
                self.logger.debug(
                    f"Applied length penalty to {penalties_applied}/{len(token_lengths)} responses (avg length: {avg_length:.0f}, threshold: {length_threshold:.0f})"  # noqa: E501
                )

            scores_container["scores"] = new_scores

        # Check if all scores are the same (no learning signal)
        if all(
            scores_container["scores"][0] == score
            for score in scores_container["scores"]
        ):
            self.logger.debug(
                f"All scores in group are identical ({scores_container['scores'][0]:.4f}) - no learning signal, skipping group"  # noqa: E501
            )

            # Before returning None, check if this is a completely failed group (all 0.0 scores) for debugging
            if self.config.dump_failed_rollouts and all(
                score == 0.0 for score in scores_container["scores"]
            ):
                self.logger.debug(
                    "Saving failed group (all 0 scores) for debugging analysis"
                )
                await self._save_failed_group_for_debugging(
                    rollout_group_data, scores_container
                )

            return None

        return scores_container

    async def _save_failed_group_for_debugging(
        self, rollout_group_data, scores_container
    ):
        """Helper method to save failed groups (all 0 scores) for debugging analysis."""
        failed_rollouts_with_scores_to_save = []

        # Build the failed rollouts data structure
        for i, (trajectory_messages, rg_item, dataset_obj) in enumerate(
            rollout_group_data
        ):
            if i < len(scores_container["scores"]):
                score_for_rollout = scores_container["scores"][i]
                failed_rollouts_with_scores_to_save.append(
                    {
                        "conversation": trajectory_messages,  # Full conversation history
                        "score": score_for_rollout,
                    }
                )

        if failed_rollouts_with_scores_to_save:
            # Extract item info for logging - get from first rollout
            _, rg_item, _ = rollout_group_data[0]
            item_id = rg_item.get("metadata", {}).get("source_dataset", "unknown_task")

            failed_item_data_to_save = {
                "item_id": item_id,
                "rollouts": failed_rollouts_with_scores_to_save,
            }
            self.failed_rollouts_to_save_buffer.append(failed_item_data_to_save)
            self.failed_processed_item_count += 1

            # Calculate progress toward next failed save
            failed_batch_progress = (
                self.failed_processed_item_count % 50
            )  # Save failed every 50 items
            if failed_batch_progress == 0:
                failed_batch_progress = (
                    50  # Show 50/50 instead of 0/50 when we hit the threshold
                )

            # Log progress every 10 failed items or when we hit the save threshold
            if failed_batch_progress % 10 == 0 or failed_batch_progress == 50:
                self.logger.info(
                    f"Failed rollouts progress: {failed_batch_progress}/50 items buffered "
                    f"(Total failed processed: {self.failed_processed_item_count}, Failed buffer size: {len(self.failed_rollouts_to_save_buffer)})"  # noqa: E501
                )

            # Check if it's time to save a batch of failed rollouts (every 50 instead of 100)
            if (
                self.config.dump_failed_rollouts
                and self.failed_processed_item_count % 50 == 0
                and self.failed_processed_item_count > 0
            ):
                failed_log_msg = (
                    f"Reached {self.failed_processed_item_count} failed items. "
                    f"Triggering save for {len(self.failed_rollouts_to_save_buffer)} failed items "
                    f"(each with multiple failed rollouts)."
                )
                self.logger.info(failed_log_msg)
                await self._save_failed_rollouts_to_jsonl()

    async def collect_trajectories(
        self, item: Item
    ) -> Tuple[Optional[ScoredDataGroup], List]:
        """
        Generate model responses for the given item and then score them.
        Args:
            item: A tuple from get_next_item: (prompt_messages, rg_item, dataset_obj)
        """
        prompt_messages_tuple, rg_item, dataset_obj = item

        # Apply chat template to convert messages to a single string
        # Ensure prompt_messages_tuple is correctly formatted list of dicts
        messages_for_template = [dict(msg_fset) for msg_fset in prompt_messages_tuple]

        prompt_str = self.tokenizer.apply_chat_template(
            messages_for_template, add_generation_prompt=True, tokenize=False
        )

        # Calculate max_tokens like tool_calling_server
        prompt_tokens = len(self.tokenizer.encode(prompt_str))
        max_tokens = self.config.max_token_length - prompt_tokens
        if max_tokens <= 0:
            return None, []

        completions = await self.server.completion(
            prompt=prompt_str,
            n=self.config.group_size,
            max_tokens=max_tokens,
            temperature=1.0,
            top_p=0.95,
        )

        to_score_list = []
        for choice in completions.choices:
            self.completion_lengths.append(len(choice.text))

            # Create full trajectory messages for this choice
            current_trajectory_messages = list(messages_for_template)
            current_trajectory_messages.append(
                {"role": "assistant", "content": choice.text}
            )

            to_score_list.append(
                (
                    tuple(current_trajectory_messages),
                    rg_item,
                    dataset_obj,
                    choice.finish_reason,
                )
            )

        scored_data_group = await self.score(to_score_list)

        # If rollouts were generated and scored, and data dumping is enabled, prepare them for saving
        if scored_data_group and self.config.dump_rollouts:
            # Only save groups that have at least one rollout with score > threshold
            group_scores = scored_data_group.get("scores", [])
            threshold = self.config.rollout_save_score_threshold
            if any(score > threshold for score in group_scores):
                self.logger.debug(
                    f"Saving group with scores: {[f'{s:.3f}' for s in group_scores]} (has high-quality rollout, threshold: {threshold})"  # noqa: E501
                )
                rollouts_with_scores_to_save = []

                num_scored_rollouts = len(group_scores)
                for i in range(num_scored_rollouts):
                    conversation_messages = to_score_list[i][0]
                    score_for_rollout = group_scores[i]
                    rollouts_with_scores_to_save.append(
                        {
                            "conversation": conversation_messages,
                            "score": score_for_rollout,
                        }
                    )

                if rollouts_with_scores_to_save:
                    # Extract item info for logging
                    _, rg_item, _ = item
                    item_id = rg_item.get("metadata", {}).get(
                        "source_dataset", "unknown_task"
                    )

                    item_data_to_save = {
                        "item_id": item_id,
                        "rollouts": rollouts_with_scores_to_save,
                    }
                    self.rollouts_to_save_buffer.append(item_data_to_save)
                    self.processed_item_count += 1

                # Calculate progress toward next save
                current_batch_progress = self.processed_item_count % 100
                if current_batch_progress == 0:
                    current_batch_progress = 100

                # Log progress every 10 items or when we hit the save threshold
                if current_batch_progress % 10 == 0 or current_batch_progress == 100:
                    self.logger.info(
                        f"Data dump progress: {current_batch_progress}/100 items buffered "
                        f"(Total processed: {self.processed_item_count}, Buffer size: {len(self.rollouts_to_save_buffer)})"  # noqa: E501
                    )

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
            else:
                max_score = max(group_scores) if group_scores else 0.0
                self.logger.debug(
                    f"Skipping group save - no high-quality rollouts (max score: {max_score:.3f}, threshold: {threshold})"  # noqa: E501
                )

        to_backlog = []
        return scored_data_group, to_backlog

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
            return

        file_path = os.path.join(
            self.datadumps_dir,
            f"reasoning_gym_environment_rollouts_{self.run_uuid}_{self.save_file_batch_num:04d}.jsonl",
        )

        try:
            with open(file_path, "w") as f:
                for rollout_dict in self.rollouts_to_save_buffer:
                    json.dump(rollout_dict, f)
                    f.write("\n")
            self.logger.info(
                f"Successfully saved {len(self.rollouts_to_save_buffer)} rollouts to {file_path}"
            )
            self.rollouts_to_save_buffer.clear()
            self.save_file_batch_num += 1
        except IOError as e:
            self.logger.error(f"Error writing rollouts to {file_path}: {e}")
        except Exception as e:
            self.logger.error(
                f"An unexpected error occurred while saving rollouts to {file_path}: {e}"
            )

    async def _save_failed_rollouts_to_jsonl(self):
        """Saves the buffered failed rollouts (all 0 scores) to a JSONL file for debugging."""
        if not self.failed_rollouts_to_save_buffer:
            self.logger.info("No failed rollouts in buffer to save.")
            return

        try:
            if not os.path.exists(self.datadumps_dir):
                os.makedirs(self.datadumps_dir)
                self.logger.info(f"Created directory: {self.datadumps_dir}")
        except OSError as e:
            self.logger.error(f"Error creating directory {self.datadumps_dir}: {e}")
            return

        file_path = os.path.join(
            self.datadumps_dir,
            f"reasoning_gym_environment_FAILED_rollouts_{self.run_uuid}_{self.failed_save_file_batch_num:04d}.jsonl",
        )

        try:
            with open(file_path, "w") as f:
                for rollout_dict in self.failed_rollouts_to_save_buffer:
                    json.dump(rollout_dict, f)
                    f.write("\n")
            self.logger.info(
                f"Successfully saved {len(self.failed_rollouts_to_save_buffer)} FAILED rollouts to {file_path}"
            )
            self.failed_rollouts_to_save_buffer.clear()
            self.failed_save_file_batch_num += 1
        except IOError as e:
            self.logger.error(f"Error writing failed rollouts to {file_path}: {e}")
        except Exception as e:
            self.logger.error(
                f"An unexpected error occurred while saving failed rollouts to {file_path}: {e}"
            )

    async def rollout_and_score_eval(
        self, test_data_tuple: Tuple[Dict[str, Any], Any]
    ) -> float:
        """
        Performs a rollout for a single evaluation item and scores it.
        Args:
            test_data_tuple: A tuple (rg_item, dataset_obj) from self.test_items_with_scorers.
        Returns:
            Score (1.0 for correct, 0.0 for incorrect/error).
        """
        rg_item, dataset_obj = test_data_tuple
        question_text = rg_item["question"]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question_text},
        ]

        prompt_str = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        # Calculate max_tokens like tool_calling_server
        prompt_tokens = len(self.tokenizer.encode(prompt_str))
        max_tokens = (2 * self.config.max_token_length) - prompt_tokens
        if max_tokens < 0:
            return 0.0

        completion = await self.server.completion(
            prompt=prompt_str,
            n=1,
            max_tokens=max_tokens,
            temperature=0.1,
            split="eval",
        )

        model_full_response = completion.choices[0].text
        model_answer_to_score = self._extract_final_answer(model_full_response)

        # If extraction failed (model didn't follow instructions), give 0 score
        if model_answer_to_score is None:
            if self.config.debug_logging:
                task_name = rg_item.get("metadata", {}).get("source_dataset", "unknown")
                self.logger.debug(
                    f"Eval - Task {task_name}: Giving 0 score due to failed answer extraction (didn't follow format)"
                )
            return 0.0

        # Use our dual-format scoring method for evaluation as well
        return self._score_answer_with_both_formats(
            model_answer_to_score, rg_item, dataset_obj
        )

    async def evaluate(self, *args, **kwargs):
        self.logger.info("Starting evaluation...")
        if not self.test_items_with_scorers:
            self.logger.warning("No test items available for evaluation. Skipping.")
            self.eval_metrics.append(("eval/percent_correct", 0.0))
            return

        eval_tasks = [
            self.rollout_and_score_eval(item_tuple)
            for item_tuple in self.test_items_with_scorers
        ]

        self.logger.info(
            f"Starting evaluation on {len(self.test_items_with_scorers)} items..."
        )
        scores = await tqdm_asyncio.gather(*eval_tasks, desc="Evaluating")

        if not scores:
            percent_correct = 0.0
        else:
            percent_correct = sum(scores) / len(scores)

        self.eval_metrics.append(("eval/percent_correct", percent_correct))
        self.logger.info(f"Evaluation finished. Percent correct: {percent_correct:.4f}")

    async def add_rollouts_for_wandb(
        self,
        scored_data: Union[ScoredDataGroup, List[ScoredDataGroup]],
        item: Item = None,
    ):
        if item is None or scored_data is None or not scored_data.get("tokens"):
            return

        _, rg_item, _ = item
        expected_answer = str(rg_item.get("answer", "N/A"))
        task_name = str(
            rg_item.get("metadata", {}).get(
                "source_dataset", rg_item.get("task_name", "unknown_task")
            )
        )

        # save rollout to trajectory
        num_keep = self.config.num_rollouts_per_group_for_logging
        if num_keep == -1:
            num_keep = self.config.group_size

        # Make sure there's data to log
        num_keep = min(num_keep, len(scored_data["tokens"]))
        if num_keep == 0:
            return

        current_rollouts = []
        for i in range(num_keep):
            # Ensure tokens and scores have the same length
            if i < len(scored_data["tokens"]) and i < len(scored_data["scores"]):
                # Decode the full trajectory including prompt and model response
                full_text = self.tokenizer.decode(
                    scored_data["tokens"][i], skip_special_tokens=True
                )
                score_val = scored_data["scores"][i]
                current_rollouts.append(
                    (full_text, score_val, expected_answer, task_name)
                )
            else:
                print(
                    f"Warning: Mismatch in lengths of tokens/scores for wandb logging at index {i}."
                )

        self.rollouts_for_wandb.append(current_rollouts)

        if len(self.rollouts_for_wandb) > self.config.num_rollouts_to_keep:
            self.rollouts_for_wandb.pop(0)

    async def create_rollout_table(self, wandb_metrics: Dict) -> Dict:
        if len(self.rollouts_for_wandb) > 0:
            table = wandb.Table(
                columns=["full_text", "score", "expected_answer", "task_name"]
            )
            for group_rollouts in self.rollouts_for_wandb:
                for rollout_tuple in group_rollouts:
                    # Ensure rollout_tuple has exactly 4 elements as defined in columns
                    if len(rollout_tuple) == 4:
                        table.add_data(*rollout_tuple)
                    else:
                        print(
                            f"Warning: Skipping malformed rollout_tuple for wandb table: {rollout_tuple}"
                        )
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
            pass

        self.percent_correct_buffer = list()

        # Add eval metrics
        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = list()

        # Add curriculum statistics if in curriculum mode
        if self.config.complexity_mode == "curriculum":
            curriculum_stats = self.get_curriculum_stats()
            wandb_metrics["curriculum/total_tasks_tracked"] = curriculum_stats[
                "total_tasks_tracked"
            ]
            wandb_metrics["curriculum/tasks_with_adjustments"] = curriculum_stats[
                "tasks_with_adjustments"
            ]
            wandb_metrics["curriculum/target_accuracy"] = curriculum_stats[
                "target_accuracy"
            ]

            # Log average complexity across all tasks
            complexities = [
                details["complexity"]
                for details in curriculum_stats["task_details"].values()
            ]
            if complexities:
                wandb_metrics["curriculum/avg_complexity"] = sum(complexities) / len(
                    complexities
                )
                wandb_metrics["curriculum/min_complexity"] = min(complexities)
                wandb_metrics["curriculum/max_complexity"] = max(complexities)

            # Log average recent accuracy for tasks that have enough data
            recent_accuracies = [
                details["recent_accuracy"]
                for details in curriculum_stats["task_details"].values()
                if details["recent_accuracy"] is not None
            ]
            if recent_accuracies:
                wandb_metrics["curriculum/avg_recent_accuracy"] = sum(
                    recent_accuracies
                ) / len(recent_accuracies)

        # Add rollout table
        wandb_metrics = await self.create_rollout_table(wandb_metrics)

        # Call superclass wandb_log
        await super().wandb_log(wandb_metrics)

    def save_checkpoint(self, step, data=None):
        """Save checkpoint including current iteration number, completion lengths, data dumping state, and complexity system."""  # noqa: E501
        if data is None:
            data = {}
        data["iter"] = self.iter
        data["rng_state"] = self.rng.getstate()
        data["completion_lengths"] = self.completion_lengths
        data["processed_item_count"] = self.processed_item_count
        data["save_file_batch_num"] = self.save_file_batch_num
        data["failed_processed_item_count"] = self.failed_processed_item_count
        data["failed_save_file_batch_num"] = self.failed_save_file_batch_num
        # Save complexity system state
        data["task_complexity_levels"] = self.task_complexity_levels
        data["task_performance_history"] = self.task_performance_history
        data["task_group_counts"] = self.task_group_counts
        super().save_checkpoint(step, data)

    def load_checkpoint(self):
        """Load checkpoint including iteration number, completion lengths, data dumping state, and complexity system."""
        # Call the base class method first to load the data
        super().load_checkpoint()

        # The base class loads data into attributes, so we can access them directly
        # if they were saved in save_checkpoint
        if hasattr(self, "iter"):
            # Restore complexity system state if available
            if hasattr(self, "task_complexity_levels") and self.task_complexity_levels:
                self.logger.info(
                    f"Restored complexity levels for {len(self.task_complexity_levels)} tasks"
                )
            if (
                hasattr(self, "task_performance_history")
                and self.task_performance_history
            ):
                total_scores = sum(
                    len(scores) for scores in self.task_performance_history.values()
                )
                self.logger.info(
                    f"Restored performance history with {total_scores} total scores"
                )
            if hasattr(self, "task_group_counts") and self.task_group_counts:
                total_groups = sum(self.task_group_counts.values())
                self.logger.info(
                    f"Restored group counts with {total_groups} total groups processed"
                )
            # Data was loaded successfully
            pass

    async def close(self):
        """Clean up and save any remaining rollouts before exiting."""
        self.logger.info(
            "Closing ReasoningGymEnv. Attempting to save any remaining rollouts..."
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

        # Also save any remaining failed rollouts
        if self.config.dump_failed_rollouts and self.failed_rollouts_to_save_buffer:
            self.logger.info(
                f"Found {len(self.failed_rollouts_to_save_buffer)} failed rollouts in buffer. Saving now."
            )
            await self._save_failed_rollouts_to_jsonl()
        else:
            self.logger.info("No failed rollouts in buffer to save upon closing.")

        # Call the superclass's close method if it exists
        if hasattr(super(), "close"):
            await super().close()
        self.logger.info("ReasoningGymEnv closed.")

    def _initialize_complexity_mapping(self):
        """Initialize complexity mappings for all reasoning-gym tasks."""
        self.logger.info(
            f"Initializing complexity mapping with mode: {self.config.complexity_mode}"
        )

        # Initialize all tasks with default complexity level
        for task_name in self.task_names:
            self.task_complexity_levels[task_name] = 0.3  # Start at 30% complexity
            self.task_performance_history[task_name] = []
            self.task_group_counts[task_name] = 0

            # Try to initialize curriculum if available and mode is curriculum
            if self.config.complexity_mode == "curriculum":
                try:
                    if (
                        reasoning_gym
                        and hasattr(reasoning_gym, "has_curriculum")
                        and reasoning_gym.has_curriculum(task_name)
                    ):
                        curriculum = reasoning_gym.create_curriculum(task_name)
                        self.task_curricula[task_name] = curriculum
                        self.logger.debug(
                            f"Initialized curriculum for task: {task_name}"
                        )
                except Exception as e:
                    self.logger.debug(
                        f"Could not initialize curriculum for {task_name}: {e}"
                    )

    def _get_complexity_params_for_task(
        self, task_name: str, complexity_level: float
    ) -> Dict[str, Any]:
        """
        Map a normalized complexity level (0.0-1.0) to task-specific parameters.

        These mappings are based on examining actual reasoning-gym task implementations
        and their default parameter ranges. The complexity ranges use reasonable
        variations around the documented defaults (typically 1-2 standard deviations).

        Args:
            task_name: Name of the reasoning-gym task
            complexity_level: Normalized complexity (0.0 = easiest, 1.0 = hardest)

        Returns:
            Dict of parameters to pass to create_dataset()
        """
        if self.config.complexity_mode is None:
            return {}  # Use default parameters

        # Realistic complexity mappings based on actual reasoning-gym defaults
        # Using reasonable ranges around the documented defaults
        complexity_mappings = {
            # Arithmetic tasks (based on basic_arithmetic.py defaults)
            "basic_arithmetic": {
                "min_terms": int(2 + complexity_level * 4),  # 2-6 (default range)
                "max_terms": int(2 + complexity_level * 4),
                "min_digits": int(1 + complexity_level * 3),  # 1-4 (default range)
                "max_digits": int(1 + complexity_level * 3),
                "allow_parentheses": complexity_level > 0.3,
                "allow_negation": complexity_level > 0.5,
            },
            # Based on leg_counting.py defaults
            "leg_counting": {
                "min_animals": int(3 + complexity_level * 7),  # 3-10 (default range)
                "max_animals": int(3 + complexity_level * 7),
                "min_instances": int(1 + complexity_level * 14),  # 1-15 (default range)
                "max_instances": int(1 + complexity_level * 14),
            },
            "decimal_arithmetic": {
                "min_terms": int(
                    2 + complexity_level * 4
                ),  # Similar to basic_arithmetic
                "max_terms": int(2 + complexity_level * 4),
                "min_digits": int(1 + complexity_level * 3),
                "max_digits": int(1 + complexity_level * 3),
            },
            "complex_arithmetic": {
                "min_terms": int(2 + complexity_level * 4),
                "max_terms": int(2 + complexity_level * 4),
                "min_magnitude": int(1 + complexity_level * 9),  # 1-10
                "max_magnitude": int(1 + complexity_level * 9),
            },
            "fraction_simplification": {
                "min_numerator": int(
                    1 + complexity_level * 49
                ),  # 1-50 (reasonable range)
                "max_numerator": int(1 + complexity_level * 49),
                "min_denominator": int(2 + complexity_level * 48),  # 2-50
                "max_denominator": int(2 + complexity_level * 48),
            },
            "bitwise_arithmetic": {
                "difficulty": int(1 + complexity_level * 9),  # 1-10
            },
            "chain_sum": {
                "min_length": int(3 + complexity_level * 7),  # 3-10
                "max_length": int(3 + complexity_level * 7),
                "min_value": int(1 + complexity_level * 99),  # 1-100
                "max_value": int(1 + complexity_level * 99),
            },
            "decimal_chain_sum": {
                "min_length": int(3 + complexity_level * 7),
                "max_length": int(3 + complexity_level * 7),
                "min_value": 1 + complexity_level * 99,  # 1.0-100.0
                "max_value": 1 + complexity_level * 99,
            },
            "count_bits": {
                "min_value": int(
                    1 + complexity_level * 255
                ),  # 1-256 (reasonable for bit counting)
                "max_value": int(1 + complexity_level * 255),
            },
            "gcd": {
                "min_value": int(1 + complexity_level * 199),  # 1-200
                "max_value": int(1 + complexity_level * 199),
            },
            "lcm": {
                "min_value": int(1 + complexity_level * 199),
                "max_value": int(1 + complexity_level * 199),
            },
            "prime_factorization": {
                "min_value": int(2 + complexity_level * 198),  # 2-200
                "max_value": int(2 + complexity_level * 198),
            },
            "power_function": {
                "min_base": int(1 + complexity_level * 19),  # 1-20
                "max_base": int(1 + complexity_level * 19),
                "min_exponent": int(1 + complexity_level * 9),  # 1-10
                "max_exponent": int(1 + complexity_level * 9),
            },
            "products": {
                "min_terms": int(2 + complexity_level * 3),  # 2-5
                "max_terms": int(2 + complexity_level * 3),
                "min_value": int(1 + complexity_level * 99),  # 1-100
                "max_value": int(1 + complexity_level * 99),
            },
            "time_intervals": {
                "min_intervals": int(2 + complexity_level * 8),  # 2-10
                "max_intervals": int(2 + complexity_level * 8),
            },
            "calendar_arithmetic": {
                "min_days": int(1 + complexity_level * 364),  # 1-365
                "max_days": int(1 + complexity_level * 364),
            },
            "dice": {
                "min_dice": int(1 + complexity_level * 9),  # 1-10
                "max_dice": int(1 + complexity_level * 9),
                "min_sides": int(4 + complexity_level * 16),  # 4-20
                "max_sides": int(4 + complexity_level * 16),
            },
            "number_format": {
                "min_digits": int(1 + complexity_level * 9),  # 1-10
                "max_digits": int(1 + complexity_level * 9),
            },
            # Games (based on n_queens.py defaults)
            "n_queens": {
                "n": int(4 + complexity_level * 8),  # 4-12 (actual range from code)
                "min_remove": int(1 + complexity_level * 6),  # 1-7
                "max_remove": int(1 + complexity_level * 6),
            },
            "sudoku": {
                "min_empty": int(30 + complexity_level * 20),  # 30-50
                "max_empty": int(30 + complexity_level * 20),
            },
            "mini_sudoku": {
                "min_empty": int(5 + complexity_level * 10),  # 5-15
                "max_empty": int(5 + complexity_level * 10),
            },
            "futoshiki": {
                "min_board_size": int(4 + complexity_level * 5),  # 4-9
                "max_board_size": int(4 + complexity_level * 5),
                "min_difficulty": int(complexity_level * 3),  # 0-3
                "max_difficulty": int(complexity_level * 3),
            },
            "tower_of_hanoi": {
                "min_disks": int(3 + complexity_level * 5),  # 3-8
                "max_disks": int(3 + complexity_level * 5),
            },
            "maze": {
                "min_size": int(5 + complexity_level * 15),  # 5-20
                "max_size": int(5 + complexity_level * 15),
            },
            "sokoban": {
                "min_size": int(4 + complexity_level * 6),  # 4-10
                "max_size": int(4 + complexity_level * 6),
            },
            "rush_hour": {
                "board_size": int(6 + complexity_level * 2),  # 6-8
                "min_cars": int(6 + complexity_level * 9),  # 6-15
                "max_cars": int(6 + complexity_level * 9),
            },
            "puzzle24": {
                "min_numbers": 4,  # Fixed at 4
                "max_numbers": 4,
                "min_value": int(1 + complexity_level * 12),  # 1-13
                "max_value": int(1 + complexity_level * 12),
            },
            "countdown": {
                "min_target": int(
                    100 + complexity_level * 400
                ),  # 100-500 (more reasonable)
                "max_target": int(100 + complexity_level * 400),
                "num_numbers": int(6 + complexity_level * 2),  # 6-8
            },
            "tsumego": {
                "board_size": int(9 + complexity_level * 10),  # 9-19
                "difficulty": int(1 + complexity_level * 4),  # 1-5
            },
            "knight_swap": {
                "board_size": int(3 + complexity_level * 2),  # 3-5
            },
            "emoji_mystery": {
                "min_equations": int(3 + complexity_level * 4),  # 3-7
                "max_equations": int(3 + complexity_level * 4),
                "min_symbols": int(3 + complexity_level * 4),  # 3-7
                "max_symbols": int(3 + complexity_level * 4),
            },
            "mahjong_puzzle": {
                "difficulty": int(1 + complexity_level * 4),  # 1-5
            },
            "boxnet": {
                "min_size": int(3 + complexity_level * 4),  # 3-7
                "max_size": int(3 + complexity_level * 4),
            },
            # Logic tasks
            "self_reference": {
                "difficulty": int(1 + complexity_level * 9),  # 1-10
            },
            "propositional_logic": {
                "min_variables": int(2 + complexity_level * 6),  # 2-8
                "max_variables": int(2 + complexity_level * 6),
                "min_clauses": int(3 + complexity_level * 12),  # 3-15
                "max_clauses": int(3 + complexity_level * 12),
            },
            "knights_knaves": {
                "min_people": int(2 + complexity_level * 6),  # 2-8
                "max_people": int(2 + complexity_level * 6),
                "min_statements": int(2 + complexity_level * 8),  # 2-10
                "max_statements": int(2 + complexity_level * 8),
            },
            "syllogism": {
                "min_premises": int(2 + complexity_level * 3),  # 2-5
                "max_premises": int(2 + complexity_level * 3),
            },
            "circuit_logic": {
                "min_gates": int(3 + complexity_level * 12),  # 3-15
                "max_gates": int(3 + complexity_level * 12),
                "min_inputs": int(2 + complexity_level * 6),  # 2-8
                "max_inputs": int(2 + complexity_level * 6),
            },
            "zebra_puzzles": {
                "num_people": int(3 + complexity_level * 2),  # 3-5
                "num_attributes": int(3 + complexity_level * 2),  # 3-5
            },
            "aiw": {
                "min_characters": int(3 + complexity_level * 4),  # 3-7
                "max_characters": int(3 + complexity_level * 4),
            },
            # Algorithmic tasks (based on graph_color.py defaults)
            "graph_color": {
                "min_num_vertices": int(4 + complexity_level * 16),  # 4-20
                "max_num_vertices": int(4 + complexity_level * 16),
                "num_colors": int(3 + complexity_level * 4),  # 3-7
                "edge_probability": 0.1
                + complexity_level * 0.4,  # 0.1-0.5 (reasonable range)
            },
            "shortest_path": {
                "min_nodes": int(4 + complexity_level * 16),  # 4-20
                "max_nodes": int(4 + complexity_level * 16),
            },
            "largest_island": {
                "min_size": int(5 + complexity_level * 15),  # 5-20
                "max_size": int(5 + complexity_level * 15),
            },
            "course_schedule": {
                "min_courses": int(3 + complexity_level * 12),  # 3-15
                "max_courses": int(3 + complexity_level * 12),
            },
            "rotten_oranges": {
                "min_size": int(3 + complexity_level * 7),  # 3-10
                "max_size": int(3 + complexity_level * 7),
            },
            "word_ladder": {
                "min_length": int(3 + complexity_level * 4),  # 3-7
                "max_length": int(3 + complexity_level * 4),
                "min_steps": int(2 + complexity_level * 8),  # 2-10
                "max_steps": int(2 + complexity_level * 8),
            },
            "binary_matrix": {
                "min_size": int(3 + complexity_level * 7),  # 3-10
                "max_size": int(3 + complexity_level * 7),
            },
            "spiral_matrix": {
                "min_size": int(3 + complexity_level * 7),  # 3-10
                "max_size": int(3 + complexity_level * 7),
            },
            "rotate_matrix": {
                "min_size": int(3 + complexity_level * 7),  # 3-10
                "max_size": int(3 + complexity_level * 7),
            },
            "pool_matrix": {
                "min_size": int(3 + complexity_level * 7),  # 3-10
                "max_size": int(3 + complexity_level * 7),
                "min_pool_size": int(2 + complexity_level * 3),  # 2-5
                "max_pool_size": int(2 + complexity_level * 3),
            },
            "manipulate_matrix": {
                "min_size": int(3 + complexity_level * 7),  # 3-10
                "max_size": int(3 + complexity_level * 7),
            },
            "string_manipulation": {
                "min_length": int(5 + complexity_level * 25),  # 5-30 (more reasonable)
                "max_length": int(5 + complexity_level * 25),
            },
            "string_synthesis": {
                "min_length": int(5 + complexity_level * 15),  # 5-20
                "max_length": int(5 + complexity_level * 15),
            },
            "string_insertion": {
                "min_length": int(5 + complexity_level * 15),  # 5-20
                "max_length": int(5 + complexity_level * 15),
            },
            "string_splitting": {
                "min_parts": int(2 + complexity_level * 6),  # 2-8
                "max_parts": int(2 + complexity_level * 6),
            },
            "palindrome_generation": {
                "min_length": int(3 + complexity_level * 12),  # 3-15
                "max_length": int(3 + complexity_level * 12),
            },
            "palindrome_partitioning": {
                "min_length": int(3 + complexity_level * 12),  # 3-15
                "max_length": int(3 + complexity_level * 12),
            },
            "letter_counting": {
                "min_length": int(
                    10 + complexity_level * 40
                ),  # 10-50 (more reasonable)
                "max_length": int(10 + complexity_level * 40),
            },
            "letter_jumble": {
                "min_words": int(3 + complexity_level * 7),  # 3-10
                "max_words": int(3 + complexity_level * 7),
                "min_length": int(3 + complexity_level * 7),  # 3-10
                "max_length": int(3 + complexity_level * 7),
            },
            "word_sorting": {
                "min_words": int(3 + complexity_level * 12),  # 3-15
                "max_words": int(3 + complexity_level * 12),
            },
            "word_sequence_reversal": {
                "min_words": int(3 + complexity_level * 7),  # 3-10
                "max_words": int(3 + complexity_level * 7),
            },
            "sentence_reordering": {
                "min_sentences": int(3 + complexity_level * 7),  # 3-10
                "max_sentences": int(3 + complexity_level * 7),
            },
            "spell_backward": {
                "min_words": int(3 + complexity_level * 7),  # 3-10
                "max_words": int(3 + complexity_level * 7),
            },
            "group_anagrams": {
                "min_words": int(3 + complexity_level * 12),  # 3-15
                "max_words": int(3 + complexity_level * 12),
            },
            "isomorphic_strings": {
                "min_length": int(3 + complexity_level * 12),  # 3-15
                "max_length": int(3 + complexity_level * 12),
            },
            "ransom_note": {
                "min_note_length": int(10 + complexity_level * 40),  # 10-50
                "max_note_length": int(10 + complexity_level * 40),
                "min_magazine_length": int(20 + complexity_level * 80),  # 20-100
                "max_magazine_length": int(20 + complexity_level * 80),
            },
            "number_sorting": {
                "min_numbers": int(5 + complexity_level * 15),  # 5-20
                "max_numbers": int(5 + complexity_level * 15),
                "min_value": int(1 + complexity_level * 199),  # 1-200
                "max_value": int(1 + complexity_level * 199),
            },
            "number_filtering": {
                "min_numbers": int(5 + complexity_level * 15),  # 5-20
                "max_numbers": int(5 + complexity_level * 15),
            },
            "base_conversion": {
                "min_base": int(2 + complexity_level * 14),  # 2-16
                "max_base": int(2 + complexity_level * 14),
                "min_value": int(1 + complexity_level * 199),  # 1-200
                "max_value": int(1 + complexity_level * 199),
            },
            "count_primes": {
                "min_limit": int(
                    10 + complexity_level * 190
                ),  # 10-200 (more reasonable)
                "max_limit": int(10 + complexity_level * 190),
            },
            "binary_alternation": {
                "min_length": int(4 + complexity_level * 16),  # 4-20
                "max_length": int(4 + complexity_level * 16),
            },
            "cryptarithm": {
                "min_letters": int(4 + complexity_level * 6),  # 4-10
                "max_letters": int(4 + complexity_level * 6),
            },
            "caesar_cipher": {
                "min_length": int(10 + complexity_level * 40),  # 10-50
                "max_length": int(10 + complexity_level * 40),
                "min_shift": int(1 + complexity_level * 24),  # 1-25
                "max_shift": int(1 + complexity_level * 24),
            },
            "game_of_life": {
                "min_size": int(5 + complexity_level * 15),  # 5-20
                "max_size": int(5 + complexity_level * 15),
                "min_steps": int(1 + complexity_level * 19),  # 1-20
                "max_steps": int(1 + complexity_level * 19),
            },
            "game_of_life_halting": {
                "difficulty": int(1 + complexity_level * 9),  # 1-10
                "grid_size_x": int(
                    10 + complexity_level * 20
                ),  # 10-30 (more reasonable)
                "grid_size_y": int(10 + complexity_level * 20),  # 10-30
                "max_simulation_steps": int(50 + complexity_level * 150),  # 50-200
            },
            "ab": {
                "min_length": int(5 + complexity_level * 25),  # 5-30
                "max_length": int(5 + complexity_level * 25),
            },
            "jugs": {
                "num_jugs": int(3 + complexity_level * 2),  # 3-5
                "difficulty": int(5 + complexity_level * 15),  # 5-20
            },
            # Cognition tasks
            "needle_haystack": {
                "min_haystack_length": int(
                    100 + complexity_level * 400
                ),  # 100-500 (more reasonable)
                "max_haystack_length": int(100 + complexity_level * 400),
                "min_needle_length": int(5 + complexity_level * 15),  # 5-20
                "max_needle_length": int(5 + complexity_level * 15),
            },
            "number_sequence": {
                "min_length": int(5 + complexity_level * 10),  # 5-15
                "max_length": int(5 + complexity_level * 10),
                "sequence_type": (
                    "arithmetic" if complexity_level < 0.5 else "geometric"
                ),
            },
            "rectangle_count": {
                "min_size": int(2 + complexity_level * 6),  # 2-8
                "max_size": int(2 + complexity_level * 6),
            },
            "modulo_grid": {
                "min_size": int(3 + complexity_level * 7),  # 3-10
                "max_size": int(3 + complexity_level * 7),
                "min_modulo": int(2 + complexity_level * 8),  # 2-10
                "max_modulo": int(2 + complexity_level * 8),
            },
            "figlet_font": {
                "min_length": int(3 + complexity_level * 12),  # 3-15
                "max_length": int(3 + complexity_level * 12),
            },
            "color_cube_rotation": {
                "cube_size": int(2 + complexity_level * 3),  # 2-5
                "num_rotations": int(1 + complexity_level * 9),  # 1-10
            },
            "rubiks_cube": {
                "cube_size": int(
                    2 + complexity_level
                ),  # 2-3 (only 2x2 and 3x3 supported)
                "num_moves": int(5 + complexity_level * 15),  # 5-20
            },
            # Algebra tasks
            "simple_equations": {
                "min_terms": int(2 + complexity_level * 3),  # 2-5
                "max_terms": int(2 + complexity_level * 3),
                "max_coefficient": int(5 + complexity_level * 15),  # 5-20
            },
            "polynomial_equations": {
                "min_degree": int(1 + complexity_level * 3),  # 1-4
                "max_degree": int(1 + complexity_level * 3),
                "max_coefficient": int(5 + complexity_level * 15),  # 5-20
            },
            "polynomial_multiplication": {
                "min_degree": int(1 + complexity_level * 3),  # 1-4
                "max_degree": int(1 + complexity_level * 3),
                "max_coefficient": int(5 + complexity_level * 15),  # 5-20
            },
            "simple_integration": {
                "min_degree": int(1 + complexity_level * 4),  # 1-5
                "max_degree": int(1 + complexity_level * 4),
                "max_coefficient": int(5 + complexity_level * 15),  # 5-20
            },
            "intermediate_integration": {
                "min_complexity": int(1 + complexity_level * 4),  # 1-5
                "max_complexity": int(1 + complexity_level * 4),
            },
            # Geometry tasks
            "simple_geometry": {
                "shape_complexity": int(1 + complexity_level * 4),  # 1-5
            },
            "advanced_geometry": {
                "shape_complexity": int(1 + complexity_level * 4),  # 1-5
                "dimension": int(2 + complexity_level),  # 2-3
            },
            # Code tasks
            "bf": {
                "difficulty": int(1 + complexity_level * 4),  # 1-5
            },
            "codeio": {
                "difficulty": int(1 + complexity_level * 9),  # 1-10
            },
            # Graph tasks
            "family_relationships": {
                "min_people": int(3 + complexity_level * 7),  # 3-10
                "max_people": int(3 + complexity_level * 7),
                "min_relationships": int(3 + complexity_level * 12),  # 3-15
                "max_relationships": int(3 + complexity_level * 12),
            },
            "quantum_lock": {
                "difficulty": int(1 + complexity_level * 9),  # 1-10
            },
            # ARC tasks
            "arc_1d": {
                "min_length": int(3 + complexity_level * 12),  # 3-15
                "max_length": int(3 + complexity_level * 12),
            },
            "arc_agi": {
                "difficulty": int(1 + complexity_level * 4),  # 1-5
            },
            "rearc": {
                "pso_difficulty_weights": [1.0 - complexity_level]
                + [complexity_level / 6] * 6,
                "rng_difficulty_weights": [1.0 - complexity_level]
                + [complexity_level / 6] * 6,
            },
            # GSM Symbolic
            "gsm_symbolic": {
                "difficulty": complexity_level,  # 0.0-1.0
            },
            # Induction tasks (ACRE, List Functions)
            "acre": {
                "difficulty": int(1 + complexity_level * 4),  # 1-5
            },
            "list_functions": {
                "min_length": int(3 + complexity_level * 12),  # 3-15
                "max_length": int(3 + complexity_level * 12),
            },
            # Composite task - dynamically includes all available tasks (handled specially in _create_dataset_with_complexity) # noqa: E501
            "composite": {
                "use_all_tasks": True,  # Special flag to indicate we want all tasks
                "exclude_tasks": ["composite"],  # Avoid infinite recursion
                "default_weight": 1.0,  # Equal weight for all tasks
            },
        }

        return complexity_mappings.get(task_name, {})

    def _get_task_complexity_level(self, task_name: str) -> float:
        """Get the current complexity level for a task based on the complexity mode."""
        if self.config.complexity_mode is None:
            return 0.5  # Default middle complexity
        elif self.config.complexity_mode == "random":
            return self.rng.random()  # Random complexity 0.0-1.0
        elif self.config.complexity_mode == "curriculum":
            return self.task_complexity_levels.get(
                task_name, 0.3
            )  # Curriculum-managed complexity
        else:
            return 0.5

    def _update_task_performance(self, task_name: str, score: float):
        """Update performance history for a task and adjust curriculum immediately if needed."""
        if self.config.complexity_mode != "curriculum":
            return

        # Add score to history
        if task_name not in self.task_performance_history:
            self.task_performance_history[task_name] = []
            self.task_group_counts[task_name] = 0

        self.task_performance_history[task_name].append(score)
        self.task_group_counts[task_name] += 1

        # Keep only recent history (last 10 groups for faster adaptation)
        if len(self.task_performance_history[task_name]) > 10:
            self.task_performance_history[task_name] = self.task_performance_history[
                task_name
            ][-10:]

        # Adjust complexity for this specific task based on its performance
        self._adjust_task_complexity(task_name)

    def _adjust_task_complexity(self, task_name: str):
        """Adjust complexity level for a specific task based on its recent performance."""
        if self.config.complexity_mode != "curriculum":
            return

        scores = self.task_performance_history.get(task_name, [])
        group_count = self.task_group_counts.get(task_name, 0)

        # Need at least 3 groups to make adjustments, but be more responsive
        if len(scores) < 3:
            return

        target_accuracy = self.config.curriculum_target_accuracy
        current_complexity = self.task_complexity_levels.get(task_name, 0.3)

        # Use recent performance (last 5 groups for responsiveness)
        recent_scores = scores[-5:]
        recent_accuracy = sum(recent_scores) / len(recent_scores)

        # Calculate variance to detect if performance is stable
        if len(recent_scores) >= 3:
            mean_score = recent_accuracy
            variance = sum((score - mean_score) ** 2 for score in recent_scores) / len(
                recent_scores
            )
            std_dev = variance**0.5
            # If performance is very unstable (high variance), be more conservative
            stability_factor = max(0.5, 1.0 - std_dev)  # Reduce adjustment if unstable
        else:
            stability_factor = 1.0

        # More aggressive adjustment thresholds for faster adaptation
        adjustment_threshold = 0.05  # Smaller threshold for more responsive adjustments
        complexity_step = 0.05 * stability_factor  # Adjust step size based on stability

        # Prevent complexity from changing too rapidly (max 0.2 change per adjustment)
        max_change = 0.2

        # Adjust complexity based on performance vs target
        if (
            recent_accuracy > target_accuracy + adjustment_threshold
        ):  # Too easy, increase complexity
            change = min(complexity_step, max_change)
            new_complexity = min(1.0, current_complexity + change)
            if new_complexity != current_complexity:
                self.task_complexity_levels[task_name] = new_complexity
                self.logger.info(
                    f"â†‘ {task_name}: complexity {current_complexity:.2f} -> {new_complexity:.2f} "
                    f"(accuracy: {recent_accuracy:.2f}, stability: {stability_factor:.2f}, groups: {group_count})"
                )

        elif (
            recent_accuracy < target_accuracy - adjustment_threshold
        ):  # Too hard, decrease complexity
            change = min(complexity_step, max_change)
            new_complexity = max(0.0, current_complexity - change)
            if new_complexity != current_complexity:
                self.task_complexity_levels[task_name] = new_complexity
                self.logger.info(
                    f"â†“ {task_name}: complexity {current_complexity:.2f} -> {new_complexity:.2f} "
                    f"(accuracy: {recent_accuracy:.2f}, stability: {stability_factor:.2f}, groups: {group_count})"
                )

        # Special case: If accuracy is very high (>90%) on easy problems, jump complexity faster
        # But only if performance is stable
        if (
            recent_accuracy > 0.9
            and current_complexity < 0.5
            and stability_factor > 0.8
        ):
            change = min(0.1 * stability_factor, max_change)
            new_complexity = min(1.0, current_complexity + change)
            if new_complexity != current_complexity:
                self.task_complexity_levels[task_name] = new_complexity
                self.logger.info(
                    f"âš¡ {task_name}: fast complexity jump {current_complexity:.2f} -> {new_complexity:.2f} "
                    f"(high accuracy: {recent_accuracy:.2f}, stable performance)"
                )

        # Special case: If accuracy is very low (<30%) on hard problems, drop complexity faster
        # But only if performance is stable (consistently low, not just unlucky)
        elif (
            recent_accuracy < 0.3
            and current_complexity > 0.5
            and stability_factor > 0.8
        ):
            change = min(0.1 * stability_factor, max_change)
            new_complexity = max(0.0, current_complexity - change)
            if new_complexity != current_complexity:
                self.task_complexity_levels[task_name] = new_complexity
                self.logger.info(
                    f"ðŸ”» {task_name}: fast complexity drop {current_complexity:.2f} -> {new_complexity:.2f} "
                    f"(low accuracy: {recent_accuracy:.2f}, stable performance)"
                )

        # Log when we're in the target zone and stable
        elif (
            abs(recent_accuracy - target_accuracy) <= adjustment_threshold
            and stability_factor > 0.9
        ):
            if group_count % 10 == 0:  # Log every 10 groups when stable
                self.logger.debug(
                    f"ðŸŽ¯ {task_name}: stable at complexity {current_complexity:.2f} "
                    f"(accuracy: {recent_accuracy:.2f}, target: {target_accuracy:.2f})"
                )

    def _create_dataset_with_complexity(self, task_name: str, current_seed: int) -> Any:
        """Create a dataset with appropriate complexity parameters."""
        try:
            # Get complexity level for this task
            complexity_level = self._get_task_complexity_level(task_name)

            # Get task-specific complexity parameters
            complexity_params = self._get_complexity_params_for_task(
                task_name, complexity_level
            )

            # Special handling for composite task
            if task_name == "composite" and complexity_params.get("use_all_tasks"):
                # Dynamically build datasets from all available tasks
                from reasoning_gym.composite import DatasetSpec

                datasets_list = []

                exclude_tasks = set(
                    complexity_params.get("exclude_tasks", ["composite"])
                )
                default_weight = complexity_params.get("default_weight", 1.0)

                # Include all tasks we have complexity mappings for
                for available_task in self.task_names:
                    if available_task not in exclude_tasks:
                        # Get the complexity parameters for this task at current complexity level
                        task_complexity_params = self._get_complexity_params_for_task(
                            available_task, complexity_level
                        )

                        datasets_list.append(
                            DatasetSpec(
                                name=available_task,
                                weight=default_weight,
                                config=task_complexity_params,
                            )
                        )

                # Replace the complexity_params with the actual datasets configuration
                complexity_params = {"datasets": datasets_list}

                if self.config.debug_logging and self.iter % 100 == 0:
                    self.logger.debug(
                        f"Composite task using {len(datasets_list)} tasks at complexity {complexity_level:.2f}"
                    )

            # Try to use reasoning-gym's curriculum system if available and mode is curriculum
            if (
                self.config.complexity_mode == "curriculum"
                and task_name in self.task_curricula
                and self.task_curricula[task_name] is not None
            ):
                try:
                    curriculum = self.task_curricula[task_name]
                    # Set curriculum level based on complexity_level (0.0-1.0 -> curriculum levels)
                    max_level = getattr(curriculum, "max_level", 10)
                    curriculum_level = int(complexity_level * max_level)
                    curriculum.set_global_level(curriculum_level)
                    config = curriculum.get_config()

                    dataset_obj = reasoning_gym.create_dataset(
                        task_name, config=config, size=1, seed=current_seed
                    )

                    if self.config.debug_logging and self.iter % 100 == 0:
                        self.logger.debug(
                            f"Used curriculum for {task_name}: level {curriculum_level}/{max_level} "
                            f"(complexity: {complexity_level:.2f})"
                        )

                    return dataset_obj

                except Exception as e:
                    if self.config.debug_logging:
                        self.logger.debug(
                            f"Curriculum failed for {task_name}, falling back to manual params: {e}"
                        )

            # Fallback to manual complexity parameters
            dataset_obj = reasoning_gym.create_dataset(
                task_name, size=1, seed=current_seed, **complexity_params
            )

            # Log complexity usage periodically
            if self.config.debug_logging and self.iter % 100 == 0 and complexity_params:
                self.logger.debug(
                    f"Manual complexity for {task_name}: {complexity_level:.2f} -> {complexity_params}"
                )

            return dataset_obj

        except Exception as e:
            # Final fallback to default parameters
            if self.config.debug_logging:
                self.logger.debug(
                    f"All complexity methods failed for {task_name}, using defaults: {e}"
                )
            return reasoning_gym.create_dataset(task_name, size=1, seed=current_seed)

    def get_curriculum_stats(self) -> Dict[str, Any]:
        """Get current curriculum statistics for monitoring and debugging."""
        if self.config.complexity_mode != "curriculum":
            return {"curriculum_mode": False}

        stats = {
            "curriculum_mode": True,
            "target_accuracy": self.config.curriculum_target_accuracy,
            "total_tasks_tracked": len(self.task_complexity_levels),
            "tasks_with_adjustments": sum(
                1 for count in self.task_group_counts.values() if count >= 3
            ),
            "task_details": {},
        }

        for task_name in self.task_complexity_levels:
            complexity = self.task_complexity_levels[task_name]
            history = self.task_performance_history.get(task_name, [])
            group_count = self.task_group_counts.get(task_name, 0)

            recent_accuracy = (
                sum(history[-5:]) / len(history[-5:]) if len(history) >= 3 else None
            )

            stats["task_details"][task_name] = {
                "complexity": complexity,
                "groups_processed": group_count,
                "recent_accuracy": recent_accuracy,
                "history_length": len(history),
                "adjustable": len(history) >= 3,
            }

        return stats


if __name__ == "__main__":
    ReasoningGymEnv.cli()
