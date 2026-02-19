"""
Letter Counting Environment with Adaptive Difficulty (Curriculum Learning)

This environment trains models to count letter occurrences in words and strings
using a 10-tier difficulty system that automatically adjusts based on performance.

Evaluation uses a static external dataset from HuggingFace: NousResearch/Letter-Counting-Eval
"""

import json
import logging
import os
import random
import re
import sys
import threading
import uuid
from typing import Dict, List, Optional, Tuple, Union

import wandb
import yaml
from pydantic import Field
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    Item,
    ScoredDataGroup,
)

# Default path to the YAML configuration file
logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "config.yaml"


def get_config_path() -> str:
    """
    Get the config path from CLI --config argument or use default.

    Usage: python letter_counting_environment.py --config /path/to/config.yaml

    Returns:
        Config file path (from --config arg or DEFAULT_CONFIG_PATH)
    """
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith("--config="):
            return arg.split("=", 1)[1]
    return DEFAULT_CONFIG_PATH


# Import NLTK words corpus for training word list
try:
    import nltk
    from nltk.corpus import words

    try:
        words.words()
    except LookupError:
        nltk.download("words")
except ImportError:
    logger.warning("NLTK not available. Please install with: pip install nltk")
    words = None

# Import datasets for loading external eval dataset
try:
    from datasets import load_dataset
except ImportError:
    logger.warning("datasets library not available. Install with: pip install datasets")
    load_dataset = None


# =============================================================================
# DIFFICULTY TIERS CONFIGURATION
# =============================================================================
# This dict can be modified to adjust difficulty levels.
# Each tier defines text length ranges, letter count ranges, and whether to use random strings.
# Levels 1-5 use real English words, levels 6-10 use random letter strings.

DIFFICULTY_TIERS = {
    # Tier 1: Very Easy - Short words, always single letter
    1: {
        "min_word_length": 3,
        "max_word_length": 8,
        "multi_letter_probability": 0.0,
        "min_letters_to_count": 1,
        "max_letters_to_count": 1,
        "use_random_string": False,
    },
    # Tier 2: Easy - Short-medium words, 50% multi (1-2 letters)
    2: {
        "min_word_length": 5,
        "max_word_length": 12,
        "multi_letter_probability": 0.5,
        "min_letters_to_count": 1,
        "max_letters_to_count": 2,
        "use_random_string": False,
    },
    # Tier 3: Medium-Easy - Medium words, 60% multi (2-3 letters)
    3: {
        "min_word_length": 8,
        "max_word_length": 18,
        "multi_letter_probability": 0.6,
        "min_letters_to_count": 2,
        "max_letters_to_count": 3,
        "use_random_string": False,
    },
    # Tier 4: Medium - Longer words, 80% multi (3-5 letters)
    4: {
        "min_word_length": 10,
        "max_word_length": 25,
        "multi_letter_probability": 0.8,
        "min_letters_to_count": 3,
        "max_letters_to_count": 5,
        "use_random_string": False,
    },
    # Tier 5: Medium-Hard - Long words, 90% multi (5-8 letters)
    5: {
        "min_word_length": 15,
        "max_word_length": 35,
        "multi_letter_probability": 0.9,
        "min_letters_to_count": 5,
        "max_letters_to_count": 8,
        "use_random_string": False,
    },
    # Tier 6: Hard - Random strings, 100% multi (8-12 letters)
    6: {
        "min_word_length": 40,
        "max_word_length": 80,
        "multi_letter_probability": 1.0,
        "min_letters_to_count": 8,
        "max_letters_to_count": 12,
        "use_random_string": True,
    },
    # Tier 7: Very Hard - Long random strings, 100% multi (12-20 letters)
    7: {
        "min_word_length": 80,
        "max_word_length": 150,
        "multi_letter_probability": 1.0,
        "min_letters_to_count": 12,
        "max_letters_to_count": 20,
        "use_random_string": True,
    },
    # Tier 8: Expert - Very long strings, 100% multi (18-30 letters)
    8: {
        "min_word_length": 150,
        "max_word_length": 250,
        "multi_letter_probability": 1.0,
        "min_letters_to_count": 18,
        "max_letters_to_count": 30,
        "use_random_string": True,
    },
    # Tier 9: Master - Extreme strings, 100% multi (25-40 letters)
    9: {
        "min_word_length": 250,
        "max_word_length": 400,
        "multi_letter_probability": 1.0,
        "min_letters_to_count": 25,
        "max_letters_to_count": 40,
        "use_random_string": True,
    },
    # Tier 10: Maximum - 500 chars, 100% multi (35-50 letters)
    10: {
        "min_word_length": 400,
        "max_word_length": 500,
        "multi_letter_probability": 1.0,
        "min_letters_to_count": 35,
        "max_letters_to_count": 50,
        "use_random_string": True,
    },
}


class LetterCountingConfig(BaseEnvConfig):
    """
    Configuration class for Letter Counting Environment.

    This environment uses adaptive difficulty (curriculum learning) with 10 difficulty tiers.
    Difficulty automatically adjusts based on model performance.
    """

    # Generation configuration
    generation_temperature: float = Field(
        1.0, description="Temperature for training generation"
    )
    eval_temperature: float = Field(
        0.6, description="Temperature for evaluation generation"
    )
    max_generation_tokens: int = Field(
        1024 * 15, description="Maximum tokens for model generation"
    )

    # Adaptive difficulty / curriculum learning configuration (10 tiers)
    difficulty_window_size: int = Field(
        150, description="Number of recent groups to track for difficulty adjustment"
    )
    difficulty_increase_threshold: float = Field(
        0.8, description="If success rate > this, increase difficulty (skip group)"
    )
    difficulty_decrease_threshold: float = Field(
        0.2, description="If success rate < this, decrease difficulty (skip group)"
    )
    min_difficulty_level: int = Field(
        1, description="Minimum difficulty level (1 = easiest)"
    )
    max_difficulty_level: int = Field(
        10, description="Maximum difficulty level (10 = hardest)"
    )
    starting_difficulty_level: int = Field(
        3, description="Starting difficulty level (1-10)"
    )

    # Logging configuration
    debug_logging: bool = Field(
        True, description="Enable debug-level logging for more verbose output"
    )
    suppress_base_env_logs: bool = Field(
        True, description="Suppress verbose base environment logs"
    )

    # Data dumping configuration (for creating offline training datasets)
    dump_rollouts: bool = Field(
        False, description="Whether to dump successful rollouts to JSONL files"
    )
    dump_batch_size: int = Field(
        100, description="Number of groups to accumulate before saving to disk"
    )


class LetterCountingEnv(BaseEnv):
    """
    Letter Counting Environment for training models to count letters in words and strings.

    This environment uses adaptive difficulty (curriculum learning) with 10 difficulty tiers:
    - Levels 1-5: Real English words with increasing length and multi-letter counting
    - Levels 6-10: Random letter strings with extreme length and many letters

    The model is presented with questions like:
    - "How many 'a's are in the string banana?" (single letter)
    - "Count the occurrences of the letters 'e', 'o', 't' in the string..." (multi-letter)

    Expected response format:
    - <think>reasoning</think><answer>3</answer> (single letter)
    - <think>reasoning</think><answer>{"e": 4, "o": 4, "t": 2}</answer> (multi-letter)

    Training filters:
    - Groups with >80% success rate are skipped (too easy)
    - Groups with <20% success rate are skipped (too hard)
    - Difficulty automatically adjusts based on rolling window performance

    Evaluation uses static dataset from HuggingFace: NousResearch/Letter-Counting-Eval
    """

    name = "letter_counting"
    env_config_cls = LetterCountingConfig

    # Fixed parameters (not configurable)
    TRAIN_TEST_SPLIT = 0.95
    RANDOM_SEED = 42
    MIN_WORD_LENGTH = 3
    MAX_WORD_LENGTH = 30
    EVAL_DATASET_HF = "NousResearch/Letter-Counting-Eval"

    def __init__(
        self,
        config: LetterCountingConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        """
        Initialize the Letter Counting environment.

        Args:
            config: Configuration for the environment
            server_configs: List of server configurations for OpenAI API
            slurm: Whether to use Slurm for distributed training
            testing: Whether in testing mode
        """
        super().__init__(config, server_configs, slurm, testing)

        # Run UUID for tracking
        self.run_uuid = str(uuid.uuid4())

        # Data dumping infrastructure
        self.rollouts_to_save_buffer: List[Dict] = []
        self.processed_item_count = 0
        self.datadumps_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data_dumps"
        )
        self.save_file_batch_num = 0

        # Metrics tracking
        self.letter_distribution_stats: Dict[str, int] = {}
        self.word_length_stats: Dict[int, int] = {}
        self.answer_format_errors = 0
        self.think_format_errors = 0

        # Adaptive difficulty / curriculum learning state
        # Thread-safe lock for concurrent workers
        self._difficulty_lock = threading.Lock()
        self.current_difficulty_level = self.config.starting_difficulty_level
        self.recent_scores: List[float] = []  # Rolling window of recent group scores
        self.difficulty_history: List[Tuple[int, int, float]] = []

        # Fixed evaluation dataset (loaded from HuggingFace during setup)
        self.eval_dataset: List[Dict] = []

        # Initialize the logger
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            log_level = logging.DEBUG if self.config.debug_logging else logging.INFO
            self.logger.setLevel(log_level)
        self.logger.disabled = False

        # Suppress base environment logs if requested
        if self.config.suppress_base_env_logs:
            base_logger = logging.getLogger("atroposlib.envs.base")
            base_logger.setLevel(logging.WARNING)

        # Log initialization
        self.logger.info(
            f"LetterCountingEnv initialized with run UUID: {self.run_uuid}"
        )
        self.logger.info(
            f"Adaptive difficulty: starting at L{self.current_difficulty_level}/10 "
            f"(range: {self.config.min_difficulty_level}-{self.config.max_difficulty_level}, "
            f"window: {self.config.difficulty_window_size}, "
            f"thresholds: >{self.config.difficulty_increase_threshold:.0%} to increase, "
            f"<{self.config.difficulty_decrease_threshold:.0%} to decrease)"
        )
        if self.config.dump_rollouts:
            self.logger.info(
                f"Data dumping ENABLED: saving to {self.datadumps_dir}, "
                f"batch size: {self.config.dump_batch_size}"
            )

        self.percent_correct_buffer = []
        self.eval_metrics = []
        self.rollouts_for_wandb: List[List[Tuple]] = []

    @classmethod
    def config_init(cls) -> Tuple[LetterCountingConfig, List[APIServerConfig]]:
        """
        Load configuration from YAML file, with fallback to defaults if not found.

        Config path priority:
        1. CLI argument: --config /path/to/config.yaml
        2. Default: config.yaml (DEFAULT_CONFIG_PATH)

        If config file doesn't exist or can't be read, uses sensible defaults.
        """
        config_path = get_config_path()

        # Try to load config from file, fall back to empty dicts if not found
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
            logger.info(f"Loaded config from {config_path}")
        except FileNotFoundError:
            logger.warning(f"Config file not found at {config_path}, using defaults")
            config = {}
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}, using defaults")
            config = {}

        env = config.get("env", {})
        openai_configs = config.get("openai", [])

        # Build LetterCountingConfig from YAML
        env_config = LetterCountingConfig(
            # Base environment config
            tokenizer_name=env.get(
                "tokenizer_name", "meta-llama/Llama-3.1-8B-Instruct"
            ),
            group_size=env.get("group_size", 16),
            batch_size=env.get("batch_size", 256),
            max_batches_offpolicy=env.get("max_batches_offpolicy", 3),
            use_wandb=env.get("use_wandb", True),
            rollout_server_url=env.get("rollout_server_url", "http://localhost:8000"),
            wandb_name=env.get("wandb_name", "tinker-letter-counting-env"),
            ensure_scores_are_not_same=env.get("ensure_scores_are_not_same", True),
            max_token_length=env.get("max_token_length", 16864),
            max_num_workers=env.get("max_num_workers", 24),
            total_steps=env.get("total_steps", 5000),
            steps_per_eval=env.get("steps_per_eval", 100),
            inference_weight=env.get("inference_weight", 1.0),
            data_path_to_save_groups=env.get("data_path_to_save_groups", None),
            eval_limit_ratio=env.get("eval_limit_ratio", 0.1),
            # Generation config
            generation_temperature=env.get("generation_temperature", 1.0),
            eval_temperature=env.get("eval_temperature", 0.6),
            max_generation_tokens=env.get("max_generation_tokens", 15360),
            # Adaptive difficulty config
            difficulty_window_size=env.get("difficulty_window_size", 150),
            difficulty_increase_threshold=env.get("difficulty_increase_threshold", 0.8),
            difficulty_decrease_threshold=env.get("difficulty_decrease_threshold", 0.2),
            min_difficulty_level=env.get("min_difficulty_level", 1),
            max_difficulty_level=env.get("max_difficulty_level", 10),
            starting_difficulty_level=env.get("starting_difficulty_level", 3),
            # Logging config
            debug_logging=env.get("debug_logging", True),
            suppress_base_env_logs=env.get("suppress_base_env_logs", True),
            # Data dumping config
            dump_rollouts=env.get("dump_rollouts", False),
            dump_batch_size=env.get("dump_batch_size", 100),
        )

        # Build server configs from YAML
        server_configs = []
        for openai_cfg in openai_configs:
            server_configs.append(
                APIServerConfig(
                    model_name=openai_cfg.get(
                        "model_name", "meta-llama/Llama-3.1-8B-Instruct"
                    ),
                    base_url=openai_cfg.get("base_url", "http://localhost:8001/v1"),
                    api_key=openai_cfg.get("api_key", "x"),
                    num_requests_for_eval=openai_cfg.get("num_requests_for_eval", 256),
                    server_type=openai_cfg.get("server_type"),
                )
            )

        # Default server config if none provided
        if not server_configs:
            server_configs = [
                APIServerConfig(
                    model_name="meta-llama/Llama-3.1-8B-Instruct",
                    base_url="http://localhost:8001/v1",
                    api_key="x",
                    num_requests_for_eval=256,
                )
            ]

        return env_config, server_configs

    async def setup(self):
        """Set up the environment by loading word dataset and eval dataset."""
        if words is None:
            raise ImportError(
                "NLTK is required for this environment. Please install with: pip install nltk"
            )

        # Set random seed for reproducibility
        random.seed(self.RANDOM_SEED)

        self.logger.info("Setting up word dataset from NLTK...")

        # Get all English words from NLTK and filter
        all_words = words.words()
        filtered_words = [
            word.lower()
            for word in all_words
            if word.isalpha()
            and self.MIN_WORD_LENGTH <= len(word) <= self.MAX_WORD_LENGTH
        ]

        # Shuffle for randomness
        random.shuffle(filtered_words)

        # Create train/test split (test words used as fallback, eval uses HF dataset)
        split_point = int(self.TRAIN_TEST_SPLIT * len(filtered_words))
        self.train_words = filtered_words[:split_point]
        self.test_words = filtered_words[split_point:]

        self.logger.info(f"Loaded {len(filtered_words)} words total")
        self.logger.info(f"Training words: {len(self.train_words)}")
        self.logger.info(f"Test words: {len(self.test_words)}")
        self.logger.info(f"Example words: {self.train_words[:10]}")

        # Initialize iteration counter
        self.iter = 0

        # Load evaluation dataset from HuggingFace
        self.eval_dataset = self._load_eval_dataset()

        self.logger.info("Letter counting environment setup complete")

    def _load_eval_dataset(self) -> List[Dict]:
        """
        Load the static evaluation dataset from HuggingFace.

        Dataset: NousResearch/Letter-Counting-Eval
        Schema: Prompt, Answer, DifficultyLevel, _text, _target_letters, _expected_counts
        """
        if load_dataset is None:
            self.logger.warning(
                "datasets library not available. Evaluation will be skipped. "
                "Install with: pip install datasets"
            )
            return []

        self.logger.info(
            f"Loading eval dataset from HuggingFace: {self.EVAL_DATASET_HF}"
        )

        try:
            dataset = load_dataset(self.EVAL_DATASET_HF, split="train")
        except Exception as e:
            self.logger.error(f"Failed to load eval dataset: {e}")
            return []

        eval_items = []
        for item in dataset:
            # Handle both formats from the schema
            eval_item = {
                "prompt": item.get("Prompt"),
                "answer": item.get("Answer"),
                "difficulty_level": item.get("DifficultyLevel", 0),
            }

            # Include raw data if available for detailed analysis
            if "_text" in item:
                eval_item["text"] = item["_text"]
            if "_target_letters" in item:
                eval_item["target_letters"] = item["_target_letters"]
            if "_expected_counts" in item:
                eval_item["expected_counts"] = item["_expected_counts"]

            eval_items.append(eval_item)

        # Log distribution by difficulty level
        level_counts = {}
        for item in eval_items:
            level = item.get("difficulty_level", 0)
            level_counts[level] = level_counts.get(level, 0) + 1

        level_str = ", ".join(f"L{k}={v}" for k, v in sorted(level_counts.items()))
        self.logger.info(f"Loaded eval dataset: {len(eval_items)} items ({level_str})")

        return eval_items

    def _get_difficulty_params(self, level: Optional[int] = None) -> Dict:
        """
        Get difficulty parameters for the specified or current difficulty level.

        Args:
            level: Optional difficulty level. If None, uses current_difficulty_level.

        Returns:
            Dict with difficulty parameters from DIFFICULTY_TIERS
        """
        if level is None:
            level = self.current_difficulty_level

        # Clamp to valid range
        level = max(
            self.config.min_difficulty_level,
            min(self.config.max_difficulty_level, level),
        )
        return DIFFICULTY_TIERS.get(level, DIFFICULTY_TIERS[3])

    def _generate_random_letter_string(self, min_length: int, max_length: int) -> str:
        """
        Generate a random string of lowercase letters for high-difficulty tiers.

        Args:
            min_length: Minimum string length
            max_length: Maximum string length

        Returns:
            Random string of lowercase letters
        """
        length = random.randint(min_length, max_length)
        return "".join(
            random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(length)
        )

    def _select_target_letters(self, text: str, num_letters: int) -> List[str]:
        """
        Select target letters with randomized bias toward letters present in the text.

        The bias is randomized each time (between 30% and 90%) to prevent the model
        from learning that a fixed percentage of letters are always absent.
        For long strings with many letters requested, the bias automatically adjusts
        to what's actually available.

        Args:
            text: The text to analyze for present letters
            num_letters: Number of letters to select

        Returns:
            List of selected target letters
        """
        all_letters = list("abcdefghijklmnopqrstuvwxyz")

        # Find letters present/absent in text
        text_lower = text.lower()
        present_letters = [ch for ch in all_letters if ch in text_lower]
        absent_letters = [ch for ch in all_letters if ch not in text_lower]

        # Randomize the present letter bias between 30% and 90%
        # This prevents the model from learning a fixed pattern
        present_bias = random.uniform(0.3, 0.9)

        selected_letters = []
        present_pool = present_letters.copy()
        absent_pool = absent_letters.copy()

        for _ in range(num_letters):
            # Calculate effective bias based on what's available
            # If we've exhausted one pool, use the other
            if not present_pool and not absent_pool:
                break
            elif not present_pool:
                effective_bias = 0.0  # Must use absent
            elif not absent_pool:
                effective_bias = 1.0  # Must use present
            else:
                effective_bias = present_bias

            # Select based on bias
            if random.random() < effective_bias and present_pool:
                chosen = random.choice(present_pool)
                present_pool.remove(chosen)
            elif absent_pool:
                chosen = random.choice(absent_pool)
                absent_pool.remove(chosen)
            elif present_pool:
                # Fallback if absent is empty
                chosen = random.choice(present_pool)
                present_pool.remove(chosen)
            else:
                break

            selected_letters.append(chosen)

        return selected_letters

    def _update_difficulty(self, group_score: float, item_difficulty_level: int):
        """
        Update difficulty level based on recent performance.

        Key behaviors:
        1. Only counts scores from items at the CURRENT difficulty level
        2. Resets the rolling window when difficulty changes
        3. Requires min_samples before making any adjustment decision

        Thread-safe: uses a lock to prevent race conditions.

        Args:
            group_score: Average score for the most recent group (0.0 to 1.0)
            item_difficulty_level: The difficulty level when the item was generated
        """
        with self._difficulty_lock:
            # Only count scores from items at the CURRENT difficulty level
            if item_difficulty_level != self.current_difficulty_level:
                self.logger.debug(
                    f"Ignoring score from L{item_difficulty_level} item "
                    f"(current: L{self.current_difficulty_level})"
                )
                return

            # Add to rolling window
            self.recent_scores.append(group_score)

            # Keep only the most recent scores
            if len(self.recent_scores) > self.config.difficulty_window_size:
                self.recent_scores = self.recent_scores[
                    -self.config.difficulty_window_size :
                ]

            # Need at least half the window size before adjusting
            min_samples = max(10, self.config.difficulty_window_size // 2)
            if len(self.recent_scores) < min_samples:
                self.logger.debug(
                    f"Not enough samples at L{self.current_difficulty_level}: "
                    f"{len(self.recent_scores)}/{min_samples} required"
                )
                return

            # Calculate recent success rate
            recent_success_rate = sum(self.recent_scores) / len(self.recent_scores)

            old_level = self.current_difficulty_level
            level_changed = False

            # Adjust difficulty
            if recent_success_rate > self.config.difficulty_increase_threshold:
                if self.current_difficulty_level < self.config.max_difficulty_level:
                    self.current_difficulty_level += 1
                    level_changed = True
                    self.logger.info(
                        f"DIFFICULTY INCREASED: L{old_level} -> L{self.current_difficulty_level} "
                        f"(success: {recent_success_rate:.1%} > {self.config.difficulty_increase_threshold:.1%})"
                    )
            elif recent_success_rate < self.config.difficulty_decrease_threshold:
                if self.current_difficulty_level > self.config.min_difficulty_level:
                    self.current_difficulty_level -= 1
                    level_changed = True
                    self.logger.info(
                        f"DIFFICULTY DECREASED: L{old_level} -> L{self.current_difficulty_level} "
                        f"(success: {recent_success_rate:.1%} < {self.config.difficulty_decrease_threshold:.1%})"
                    )

            # Reset window when difficulty changes
            if level_changed:
                self.recent_scores = []
                self.logger.info(
                    f"Window reset - collecting fresh samples at L{self.current_difficulty_level}"
                )

            # Track difficulty history
            self.difficulty_history.append(
                (self.iter, self.current_difficulty_level, recent_success_rate)
            )

    def _reconstruct_message_with_thinking(self, completion_choice) -> str:
        """
        Reconstruct a message by combining reasoning_content and content.

        Modern reasoning models return reasoning in a separate `reasoning_content` field.
        This method wraps that in <think> tags for consistent processing.
        """
        # Handle text completion API
        if hasattr(completion_choice, "text"):
            raw_content = completion_choice.text or ""
            reasoning_content = None

            if hasattr(completion_choice, "reasoning_content"):
                reasoning_content = completion_choice.reasoning_content

            if reasoning_content is None and hasattr(completion_choice, "message"):
                message = completion_choice.message
                if message and hasattr(message, "reasoning_content"):
                    reasoning_content = message.reasoning_content

            if reasoning_content:
                return f"<think>\n{reasoning_content}\n</think>{raw_content}"
            return raw_content

        # Handle chat completion API
        elif hasattr(completion_choice, "message"):
            message = completion_choice.message
            content = getattr(message, "content", "") or ""
            reasoning_content = getattr(message, "reasoning_content", None)

            if reasoning_content:
                return f"<think>\n{reasoning_content}\n</think>{content}"
            return content

        return ""

    async def _save_rollouts_to_jsonl(self):
        """Save buffered rollouts to a JSONL file in the data_dumps directory."""
        if not self.rollouts_to_save_buffer:
            self.logger.warning("_save_rollouts_to_jsonl called but buffer is empty!")
            return

        buffer_size = len(self.rollouts_to_save_buffer)
        self.logger.info(f"Starting save of {buffer_size} groups to JSONL file...")

        try:
            if not os.path.exists(self.datadumps_dir):
                os.makedirs(self.datadumps_dir)
                self.logger.info(f"Created directory: {self.datadumps_dir}")
        except OSError as e:
            self.logger.error(f"Error creating directory {self.datadumps_dir}: {e}")
            return

        file_path = os.path.join(
            self.datadumps_dir,
            f"letter_counting_rollouts_{self.run_uuid}_{self.save_file_batch_num:04d}.jsonl",
        )

        try:
            with open(file_path, "w") as f:
                for rollout_dict in self.rollouts_to_save_buffer:
                    json.dump(rollout_dict, f)
                    f.write("\n")
            self.logger.info(
                f"Successfully saved {buffer_size} groups to {file_path} "
                f"(batch #{self.save_file_batch_num})"
            )
            self.rollouts_to_save_buffer.clear()
            self.save_file_batch_num += 1
        except IOError as e:
            self.logger.error(f"Error writing rollouts to {file_path}: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error saving rollouts: {e}")

    def save_checkpoint(self, step, data=None):
        """Save checkpoint including iteration and difficulty state."""
        if data is None:
            data = {}
        data["iter"] = self.iter
        data["current_difficulty_level"] = self.current_difficulty_level
        data["recent_scores"] = self.recent_scores
        data["processed_item_count"] = self.processed_item_count
        data["save_file_batch_num"] = self.save_file_batch_num
        data["letter_distribution_stats"] = self.letter_distribution_stats
        data["word_length_stats"] = self.word_length_stats
        data["answer_format_errors"] = self.answer_format_errors
        data["think_format_errors"] = self.think_format_errors
        super().save_checkpoint(step, data)

    def load_checkpoint(self):
        """Load checkpoint including iteration and difficulty state."""
        super().load_checkpoint()

        if hasattr(self, "current_difficulty_level"):
            self.logger.info(
                f"Restored difficulty level: L{self.current_difficulty_level}"
            )
        if hasattr(self, "recent_scores") and self.recent_scores:
            self.logger.info(f"Restored {len(self.recent_scores)} recent scores")
        if hasattr(self, "save_file_batch_num"):
            self.logger.info(
                f"Restored save_file_batch_num: {self.save_file_batch_num}"
            )

    async def close(self):
        """Clean up and save any remaining rollouts before exiting."""
        self.logger.info("Closing LetterCountingEnv...")

        if self.config.dump_rollouts and self.rollouts_to_save_buffer:
            self.logger.info(
                f"FINAL SAVE: {len(self.rollouts_to_save_buffer)} groups in buffer. "
                f"Saving to disk..."
            )
            await self._save_rollouts_to_jsonl()

        if hasattr(super(), "close"):
            await super().close()

        self.logger.info("LetterCountingEnv closed.")

    async def get_next_item(self):
        """
        Get the next training item from the dataset.

        Uses adaptive difficulty to determine text length and letter count.
        """
        # Get difficulty parameters for current level
        diff_params = self._get_difficulty_params()
        min_len = diff_params["min_word_length"]
        max_len = diff_params["max_word_length"]
        multi_prob = diff_params["multi_letter_probability"]
        min_letters = diff_params.get("min_letters_to_count", 1)
        max_letters = diff_params["max_letters_to_count"]
        use_random_string = diff_params.get("use_random_string", False)

        # Get text based on difficulty level
        if use_random_string:
            # Higher difficulty tiers use generated random strings
            text = self._generate_random_letter_string(min_len, max_len)
        else:
            # Find a word that matches the difficulty-appropriate length
            attempts = 0
            max_attempts = 100
            while attempts < max_attempts:
                candidate = self.train_words[
                    (self.iter + attempts) % len(self.train_words)
                ]
                if min_len <= len(candidate) <= max_len:
                    text = candidate
                    break
                attempts += 1
            else:
                # Fallback: generate random string if no matching word found
                text = self._generate_random_letter_string(min_len, max_len)

        # Decide number of letters to count
        if random.random() < multi_prob:
            num_letters = random.randint(max(2, min_letters), max_letters)
        else:
            num_letters = 1

        target_letters = self._select_target_letters(text, num_letters)

        # Count occurrences for each target letter
        correct_counts = {
            letter: text.lower().count(letter) for letter in target_letters
        }

        # Log item selection
        text_preview = text[:50] + "..." if len(text) > 50 else text
        letters_str = ", ".join(target_letters)
        counts_str = ", ".join(
            f"{letter}:{correct_counts[letter]}" for letter in target_letters
        )
        present_count = sum(
            1 for letter in target_letters if correct_counts[letter] > 0
        )

        self.logger.info(
            f"[L{self.current_difficulty_level}/10] '{text_preview}' | "
            f"Letters: [{letters_str}] | Counts: [{counts_str}] | "
            f"Present: {present_count}/{len(target_letters)} (iter {self.iter})"
        )

        self.iter += 1

        # Create the question based on single/multiple letters
        if len(target_letters) == 1:
            target_letter = target_letters[0]
            question_text = f"How many {target_letter}s are in the string {text}?"
            question_with_instruction = (
                f"{question_text}\n\n"
                f"Provide your answer in the format: <answer>{{number}}</answer>"
            )
        else:
            letters_str = (
                ", ".join(f"'{letter}'" for letter in target_letters[:-1])
                + f", and '{target_letters[-1]}'"
            )
            question_text = f"Count the occurrences of the letters {letters_str} in the string {text}"
            example_json = (
                "{" + ", ".join(f'"{letter}": 0' for letter in target_letters) + "}"
            )
            question_with_instruction = (
                f"{question_text}\n\n"
                f"Provide your answer as JSON in the format: <answer>{example_json}</answer>"
            )

        # Create prompt (user message only, no system prompt)
        prompt = [
            frozenset({"role": "user", "content": question_with_instruction}.items())
        ]

        # Return prompt, correct counts, text, target letters, and difficulty level
        return (
            tuple(prompt),
            correct_counts,
            text,
            target_letters,
            self.current_difficulty_level,
        )

    async def collect_trajectories(self, item) -> Tuple[ScoredDataGroup, List]:
        """Generate and collect model responses for scoring."""
        # Extract messages from the item
        messages = [dict(role_dict) for role_dict in item[0]]

        try:
            async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
                completions = await managed.chat_completion(
                    messages=messages,
                    n=self.config.group_size,
                    max_tokens=self.config.max_generation_tokens,
                    temperature=self.config.generation_temperature,
                    stop=[self.tokenizer.eos_token_id],
                )

                state = managed.get_state()
                nodes = state["nodes"]
        except Exception as e:
            import traceback

            text_preview = item[2][:50] if len(item[2]) > 50 else item[2]
            self.logger.error(
                f"API call failed for '{text_preview}' (iter {self.iter}): {e}"
            )
            traceback.print_exc()
            raise

        to_score = []

        for i, completion_choice in enumerate(completions.choices):
            trajectory_messages = [dict(role_dict) for role_dict in item[0]]
            reconstructed_response = self._reconstruct_message_with_thinking(
                completion_choice
            )
            trajectory_messages.append(
                {"role": "assistant", "content": reconstructed_response}
            )

            to_score.append(
                {
                    "messages": tuple(trajectory_messages),
                    "correct_counts": item[1],
                    "text": item[2],
                    "target_letters": item[3],
                    "difficulty_level": item[4],
                    "finish_reason": completion_choice.finish_reason,
                    "tokens": nodes[i].tokens,
                    "masks": nodes[i].masked_tokens,
                    "logprobs": nodes[i].logprobs,
                }
            )

        scored_data = await self.score(to_score)

        # Data dumping logic
        if self.config.dump_rollouts and scored_data is not None:
            await self._handle_data_dumping(to_score, scored_data, item)

        return scored_data, []

    async def _handle_data_dumping(
        self, to_score: List, scored_data: ScoredDataGroup, item
    ):
        """Handle data dumping for creating offline training datasets."""
        if not scored_data.get("scores"):
            return

        scores = scored_data["scores"]
        group_average = sum(scores) / len(scores)

        # Only dump groups that have learning signal (not all same scores)
        if all(scores[0] == s for s in scores):
            return

        # Extract successful rollouts for dumping
        rollouts_to_dump = []
        for i, score_val in enumerate(scores):
            if score_val > 0:  # Only save successful rollouts
                rollouts_to_dump.append(
                    {
                        "conversation": to_score[i]["messages"],
                        "score": score_val,
                        "expected_counts": to_score[i]["correct_counts"],
                        "text": to_score[i]["text"],
                        "target_letters": to_score[i]["target_letters"],
                        "difficulty_level": to_score[i]["difficulty_level"],
                        "group_average_score": group_average,
                    }
                )

        if rollouts_to_dump:
            text = item[2]
            target_letters = item[3]
            text_preview = (
                text[:30].replace(" ", "_")
                if len(text) > 30
                else text.replace(" ", "_")
            )
            letters_str = "_".join(target_letters)
            item_id = f"{text_preview}_{letters_str}"

            self.rollouts_to_save_buffer.append(
                {
                    "item_id": item_id,
                    "rollouts": rollouts_to_dump,
                }
            )
            self.processed_item_count += 1

            self.logger.debug(
                f"BUFFER ADD: Added '{item_id}' to buffer. "
                f"Buffer: {len(self.rollouts_to_save_buffer)}/{self.config.dump_batch_size}"
            )

            # Save when buffer reaches batch size
            if len(self.rollouts_to_save_buffer) >= self.config.dump_batch_size:
                await self._save_rollouts_to_jsonl()

    def _extract_answer(self, text: str, expected_format: str = "single"):
        """
        Extract the answer from model response.

        Only allows one valid answer format - multiple formats result in score of 0.

        Args:
            text: The model's response text
            expected_format: "single" for number, "multi" for JSON

        Returns:
            Extracted answer (int for single, dict for multi) or None if invalid
        """
        # Check for exactly one <think> opening and closing tag
        think_tags = re.findall(r"<think>", text, re.IGNORECASE)
        if len(think_tags) != 1:
            return None

        think_close_tags = re.findall(r"</think>", text, re.IGNORECASE)
        if len(think_close_tags) != 1:
            return None

        # Split into thinking and answer sections
        parts = re.split(r"</think>", text, flags=re.IGNORECASE, maxsplit=1)
        if len(parts) != 2:
            return None

        thinking_section, answer_section = parts

        # Validate thinking section contains opening tag
        if "<think>" not in thinking_section.lower():
            return None

        # No additional think tags in answer section
        if "<think>" in answer_section.lower():
            return None

        # Extract answer based on format
        if expected_format == "single":
            answer_pattern = r"<answer>\s*(\d+)\s*</answer>"
            matches = re.findall(answer_pattern, answer_section, re.IGNORECASE)
            if len(matches) != 1:
                return None
            try:
                return int(matches[0])
            except ValueError:
                return None
        else:
            answer_pattern = r"<answer>\s*(\{[^}]+\})\s*</answer>"
            matches = re.findall(answer_pattern, answer_section, re.IGNORECASE)
            if len(matches) != 1:
                return None
            try:
                answer_dict = json.loads(matches[0])
                if not isinstance(answer_dict, dict):
                    return None
                for key, value in answer_dict.items():
                    if not isinstance(key, str) or not isinstance(value, int):
                        return None
                return answer_dict
            except (json.JSONDecodeError, ValueError):
                return None

    async def score(self, rollout_group_data: List) -> Optional[ScoredDataGroup]:
        """Score the generated model responses against expected letter counts."""
        scores = ScoredDataGroup()
        scores["tokens"] = []
        scores["masks"] = []
        scores["scores"] = []
        scores["inference_logprobs"] = []

        if not rollout_group_data:
            return None

        expected_counts = rollout_group_data[0]["correct_counts"]
        text = rollout_group_data[0]["text"]
        target_letters = rollout_group_data[0]["target_letters"]
        item_difficulty_level = rollout_group_data[0].get("difficulty_level", 0)

        # Track statistics
        for target_letter in target_letters:
            if target_letter not in self.letter_distribution_stats:
                self.letter_distribution_stats[target_letter] = 0
            self.letter_distribution_stats[target_letter] += 1

        text_len = len(text)
        if text_len not in self.word_length_stats:
            self.word_length_stats[text_len] = 0
        self.word_length_stats[text_len] += 1

        # Shuffle to avoid selection bias
        random.shuffle(rollout_group_data)

        format_errors_in_group = 0
        think_errors_in_group = 0

        for item in rollout_group_data:
            model_response = item["messages"][-1]["content"]
            stop_reason = item["finish_reason"]

            if stop_reason == "length":
                reward = 0
            else:
                expected_format = "single" if len(target_letters) == 1 else "multi"
                model_answer = self._extract_answer(model_response, expected_format)

                if model_answer is None:
                    reward = 0
                    format_errors_in_group += 1
                    if (
                        "<think>" not in model_response.lower()
                        or "</think>" not in model_response.lower()
                    ):
                        think_errors_in_group += 1
                else:
                    if expected_format == "single":
                        expected_single_count = expected_counts[target_letters[0]]
                        reward = 1 if model_answer == expected_single_count else 0
                    else:
                        if set(model_answer.keys()) == set(target_letters) and all(
                            model_answer.get(letter, -1) == expected_counts[letter]
                            for letter in target_letters
                        ):
                            reward = 1
                        else:
                            reward = 0

            tokens = item["tokens"]
            masks = item["masks"]
            logprobs = item["logprobs"]

            # Remove examples with insufficient context
            if len([1 for i in masks if i != -100]) < 10:
                continue

            scores["tokens"].append(tokens)
            scores["masks"].append(masks)
            scores["inference_logprobs"].append(logprobs)
            scores["scores"].append(1.0 if reward else 0.0)

            if len(scores["tokens"]) >= self.config.group_size:
                break

        if not scores["tokens"]:
            self.logger.warning(f"No valid items scored for '{text[:50]}...'")
            return None

        # Update global error counters
        self.answer_format_errors += format_errors_in_group
        self.think_format_errors += think_errors_in_group

        # Record success rate
        for score in scores["scores"]:
            self.percent_correct_buffer.append(score)

        # Calculate group average and check training thresholds
        current_scores = scores.get("scores", [])
        if not current_scores:
            return None

        average_score = sum(current_scores) / len(current_scores)

        # Update adaptive difficulty
        self._update_difficulty(average_score, item_difficulty_level)

        # Log group results
        text_preview = text[:50] + "..." if len(text) > 50 else text
        letters_str = ", ".join(target_letters)
        expected_str = (
            str(expected_counts)
            if len(target_letters) > 1
            else str(expected_counts[target_letters[0]])
        )

        self.logger.info(
            f"'{text_preview}' | Letters: '{letters_str}' | Expected: {expected_str} | "
            f"Score: {average_score:.2f} | L{item_difficulty_level} (current: L{self.current_difficulty_level})"
        )

        # CRITICAL: Training difficulty filtering
        # Skip groups that are too easy (>80%) or too hard (<20%)
        if average_score > self.config.difficulty_increase_threshold:
            self.logger.debug(
                f"Skipping group - too easy (avg: {average_score:.2f} > "
                f"{self.config.difficulty_increase_threshold})"
            )
            return None

        if average_score < self.config.difficulty_decrease_threshold:
            self.logger.debug(
                f"Skipping group - too hard (avg: {average_score:.2f} < "
                f"{self.config.difficulty_decrease_threshold})"
            )
            return None

        # Skip if all scores are identical (no learning signal)
        if all(scores["scores"][0] == score for score in scores["scores"]):
            self.logger.debug(
                f"All scores identical ({scores['scores'][0]:.2f}) - skipping group"
            )
            return None

        return scores

    async def rollout_and_score_eval(self, eval_item: Dict) -> Tuple[int, int]:
        """
        Generate and score model response for a single eval item.

        Uses the pre-formatted prompt from the HuggingFace dataset.

        Args:
            eval_item: Item from the eval dataset with 'prompt' and 'answer' fields

        Returns:
            Tuple of (score, difficulty_level)
        """
        prompt = eval_item["prompt"]
        expected_answer = eval_item["answer"]
        difficulty_level = eval_item.get("difficulty_level", 0)

        # Determine expected format from answer
        try:
            # Try to parse as JSON (multi-letter)
            parsed_answer = json.loads(expected_answer)
            if isinstance(parsed_answer, dict):
                expected_format = "multi"
                expected_counts = parsed_answer
            else:
                expected_format = "single"
                expected_counts = int(expected_answer)
        except (json.JSONDecodeError, ValueError, TypeError):
            # Single number
            expected_format = "single"
            try:
                expected_counts = int(expected_answer)
            except (ValueError, TypeError):
                # Can't parse answer, skip this item
                self.logger.warning(f"Could not parse eval answer: {expected_answer}")
                return (0, difficulty_level)

        # Create messages and get completion
        messages = [{"role": "user", "content": prompt}]

        async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
            completion = await managed.chat_completion(
                messages=messages,
                n=1,
                max_tokens=self.config.max_generation_tokens,
                temperature=self.config.eval_temperature,
                split="eval",
                stop=[self.tokenizer.eos_token_id],
            )

        model_response = self._reconstruct_message_with_thinking(completion.choices[0])
        model_answer = self._extract_answer(model_response, expected_format)

        # Score based on format
        if model_answer is None:
            score = 0
        elif expected_format == "single":
            score = 1 if model_answer == expected_counts else 0
        else:
            # Multi-letter: compare dicts
            score = 1 if model_answer == expected_counts else 0

        return (score, difficulty_level)

    async def evaluate(self, *args, **kwargs):
        """
        Evaluate the model on the static HuggingFace eval dataset.

        Tracks and logs accuracy for each difficulty level separately.
        """
        self.logger.info("Starting evaluation...")

        if not self.eval_dataset:
            self.logger.warning("No eval dataset available. Skipping evaluation.")
            self.eval_metrics.append(("eval/percent_correct", 0.0))
            return

        self.logger.info(
            f"Evaluating on {len(self.eval_dataset)} items from {self.EVAL_DATASET_HF}..."
        )

        eval_tasks = [
            self.rollout_and_score_eval(eval_item) for eval_item in self.eval_dataset
        ]
        results = await tqdm_asyncio.gather(*eval_tasks, desc="Evaluating")

        if not results:
            self.eval_metrics.append(("eval/percent_correct", 0.0))
            self.logger.warning("No evaluation results returned.")
            return

        # Calculate overall and per-difficulty scores
        all_scores = [r[0] for r in results]
        level_scores = {}

        for score, difficulty_level in results:
            if difficulty_level not in level_scores:
                level_scores[difficulty_level] = []
            level_scores[difficulty_level].append(score)

        # Overall accuracy
        percent_correct = sum(all_scores) / len(all_scores) if all_scores else 0.0
        self.eval_metrics.append(("eval/percent_correct", percent_correct))

        # Per-difficulty accuracy
        level_accuracies = {}
        for level in sorted(level_scores.keys()):
            if level_scores[level]:
                level_acc = sum(level_scores[level]) / len(level_scores[level])
                level_accuracies[level] = level_acc
                self.eval_metrics.append((f"eval/percent_correct_L{level}", level_acc))

        level_summary = " | ".join(
            f"L{lvl}: {level_accuracies.get(lvl, 0):.0%}"
            for lvl in sorted(level_accuracies.keys())
        )
        self.logger.info(
            f"Evaluation finished. Overall: {percent_correct:.1%} | {level_summary}"
        )

    async def add_rollouts_for_wandb(
        self,
        scored_data: Union[ScoredDataGroup, List[ScoredDataGroup]],
        item: Item = None,
    ):
        """Add rollouts for WandB logging."""
        if item is None or scored_data is None or not scored_data.get("tokens"):
            return

        expected_counts = item[1]
        text = item[2]
        target_letters = item[3]

        num_keep = self.config.num_rollouts_per_group_for_logging
        if num_keep == -1:
            num_keep = self.config.group_size

        num_keep = min(num_keep, len(scored_data["tokens"]))
        if num_keep == 0:
            return

        group_scores = scored_data.get("scores", [])
        group_average_score = (
            sum(group_scores) / len(group_scores) if group_scores else 0.0
        )

        current_rollouts = []
        for i in range(num_keep):
            if i < len(scored_data["tokens"]) and i < len(scored_data["scores"]):
                full_text = self.tokenizer.decode(
                    scored_data["tokens"][i], skip_special_tokens=True
                )
                score_val = scored_data["scores"][i]
                expected_str = (
                    str(expected_counts)
                    if len(target_letters) > 1
                    else str(expected_counts[target_letters[0]])
                )
                letters_str = ", ".join(target_letters)
                current_rollouts.append(
                    (
                        full_text,
                        score_val,
                        expected_str,
                        text[:100],
                        letters_str,
                        group_average_score,
                    )
                )

        self.rollouts_for_wandb.append(current_rollouts)

        if len(self.rollouts_for_wandb) > self.config.num_rollouts_to_keep:
            self.rollouts_for_wandb.pop(0)

    async def create_rollout_table(self, wandb_metrics):
        """Create a WandB table with rollout examples."""
        if len(self.rollouts_for_wandb) > 0:
            table = wandb.Table(
                columns=[
                    "full_text",
                    "score",
                    "expected_counts",
                    "text",
                    "target_letters",
                    "group_average_score",
                ]
            )
            for group in self.rollouts_for_wandb:
                for item in group:
                    if len(item) >= 6:
                        table.add_data(
                            item[0], item[1], item[2], item[3], item[4], item[5]
                        )
                    else:
                        table.add_data(item[0], item[1], item[2], item[3], item[4], 0.0)
            wandb_metrics["train/rollouts"] = table
        self.rollouts_for_wandb = []
        return wandb_metrics

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Log metrics to WandB."""
        if wandb_metrics is None:
            wandb_metrics = {}

        # Training accuracy
        try:
            wandb_metrics["train/percent_correct"] = sum(
                self.percent_correct_buffer
            ) / len(self.percent_correct_buffer)
        except ZeroDivisionError:
            pass

        self.percent_correct_buffer = []

        # Eval metrics
        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = []

        # Letter distribution stats
        if self.letter_distribution_stats:
            total_letters_asked = sum(self.letter_distribution_stats.values())
            wandb_metrics["stats/total_letters_asked"] = total_letters_asked

            most_common = max(
                self.letter_distribution_stats, key=self.letter_distribution_stats.get
            )
            least_common = min(
                self.letter_distribution_stats, key=self.letter_distribution_stats.get
            )
            wandb_metrics["stats/most_common_letter_count"] = (
                self.letter_distribution_stats[most_common]
            )
            wandb_metrics["stats/least_common_letter_count"] = (
                self.letter_distribution_stats[least_common]
            )

            import math

            entropy = -sum(
                (count / total_letters_asked) * math.log2(count / total_letters_asked)
                for count in self.letter_distribution_stats.values()
                if count > 0
            )
            wandb_metrics["stats/letter_distribution_entropy"] = entropy

        # Word length stats
        if self.word_length_stats:
            total_words = sum(self.word_length_stats.values())
            wandb_metrics["stats/total_words_asked"] = total_words

            avg_length = (
                sum(length * count for length, count in self.word_length_stats.items())
                / total_words
            )
            wandb_metrics["stats/avg_word_length"] = avg_length
            wandb_metrics["stats/min_word_length"] = min(self.word_length_stats.keys())
            wandb_metrics["stats/max_word_length"] = max(self.word_length_stats.keys())

        # Error rates
        if self.processed_item_count > 0:
            wandb_metrics["errors/answer_format_error_rate"] = (
                self.answer_format_errors
                / (self.processed_item_count * self.config.group_size)
            )
            wandb_metrics["errors/think_format_error_rate"] = (
                self.think_format_errors
                / (self.processed_item_count * self.config.group_size)
            )
            wandb_metrics["errors/total_format_errors"] = (
                self.answer_format_errors + self.think_format_errors
            )

        # Adaptive difficulty metrics
        wandb_metrics["curriculum/difficulty_level"] = self.current_difficulty_level
        if self.recent_scores:
            wandb_metrics["curriculum/recent_success_rate"] = sum(
                self.recent_scores
            ) / len(self.recent_scores)
        wandb_metrics["curriculum/samples_in_window"] = len(self.recent_scores)

        # Data dumping metrics
        if self.config.dump_rollouts:
            wandb_metrics["data_dumps/buffer_size"] = len(self.rollouts_to_save_buffer)
            wandb_metrics["data_dumps/batches_saved"] = self.save_file_batch_num

        # Rollout table
        wandb_metrics = await self.create_rollout_table(wandb_metrics)

        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    LetterCountingEnv.cli()
