#!/usr/bin/env python3
"""
Recursive Tool Improvement Environment

This environment trains language models to create, improve, and recursively
refine tool compositions to solve complex problems. The environment enables
models to:

1. Create initial tool compositions for a given problem
2. Execute the compositions and observe results
3. Critique their approach
4. Improve the compositions based on self-critique
5. Repeat iteratively, creating a feedback loop of continuous improvement

This addresses a novel research direction focused on higher-order tool
composition, longitudinal improvement patterns, and meta-cognitive evaluation.
"""

import asyncio
import json
import logging
import os
import random
import re
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Literal
from collections import Counter

# Load environment variables from .env file if it exists
from dotenv import load_dotenv
load_dotenv()

import wandb
from pydantic import Field
from tqdm.asyncio import tqdm_asyncio

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
    Item,
    ScoredDataGroup,
    ScoredDataItem,
)
from atroposlib.type_definitions import Message
from atroposlib.utils.message_history_utils import (
    ensure_trajectory_token_limit,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

from environments.hack0.recursive_tool_improvement.tool_registry import ToolRegistry, default_registry
from environments.hack0.recursive_tool_improvement.execution_engine import ExecutionEngine, ExecutionResult
from environments.hack0.recursive_tool_improvement.reward_functions.binary_verification import BinaryVerificationReward


logger = logging.getLogger(__name__)


class ReasoningMode(str, Enum):
    """
    Reasoning modes supported by the environment.

    Based on Absolute-Zero-Reasoner's three reasoning modes:
    """

    DEDUCTION = "deduction"  # Given tool composition and input, predict output
    ABDUCTION = "abduction"  # Given input and output, create a tool composition
    INDUCTION = "induction"  # Generate examples demonstrating tool composition behavior
    ALL = "all"  # Support all reasoning modes


class RecursiveToolImprovementConfig(BaseEnvConfig):
    """
    Configuration for the Recursive Tool Improvement Environment.
    """

    # Basic operational parameters
    max_iterations: int = Field(
        default=3, description="Number of improvement iterations per trajectory"
    )
    tool_set: str = Field(
        default="basic", description="Which tool set to use"
    )
    timeout: int = Field(
        default=5, description="Execution timeout in seconds"
    )
    memory_limit: int = Field(
        default=100 * 1024 * 1024, description="Memory limit in bytes (100 MB default)"
    )
    improvement_threshold: float = Field(
        default=0.2, description="Minimum improvement to continue iterations"
    )

    # Environment operational modes
    reasoning_mode: ReasoningMode = Field(
        default=ReasoningMode.DEDUCTION, description="Reasoning mode to use"
    )
    verification_type: str = Field(
        default="binary", description="Verification type (binary or graduated)"
    )

    # System prompt configuration
    use_thinking_tags: bool = Field(
        default=True, description="Whether to use thinking tags in prompts"
    )
    critique_format: str = Field(
        default="structured", description="Format for critiques (structured or free-form)"
    )

    # Problem settings
    data_path: str = Field(
        default="data/problems", description="Path to problem dataset"
    )
    eval_ratio: float = Field(
        default=0.1, description="Percentage of problems reserved for evaluation"
    )

    # Overrides for process and demonstration
    wandb_name: str = Field(
        default="recursive_tool_improvement", description="WandB run name"
    )
    group_size: int = Field(
        default=8, description="Number of responses per problem"
    )
    allow_partial_credit: bool = Field(
        default=False, description="Whether to allow partial credit for partial solutions"
    )


class ConversationState(Enum):
    """
    Represents the current state of a conversation trajectory.

    Simplified to three core states for the recursive improvement cycle.
    """

    INITIAL = "initial"        # Initial problem statement and first solution
    IMPROVEMENT = "improvement"  # Improvement cycle (critique and improved solution)
    FINAL = "final"            # Final results after iterations


class ProblemDefinition:
    """
    Defines a single problem for the tool improvement environment.

    Attributes:
        problem_id: Unique identifier for the problem
        title: Short descriptive title
        description: Detailed problem description
        input_data: Sample input data for testing
        expected_outputs: Expected outputs for verification
        difficulty: Problem difficulty level (easy, medium, hard)
        domain: Problem domain or category
        available_tools: Tools that should be available for this problem
        reasoning_mode: The reasoning mode this problem is designed for
    """

    def __init__(
        self,
        problem_id: str,
        title: str,
        description: str,
        input_data: Any,
        expected_outputs: Any,
        difficulty: str = "medium",
        domain: str = "text_processing",
        available_tools: Optional[List[str]] = None,
        reasoning_mode: str = "deduction"
    ):
        self.problem_id = problem_id
        self.title = title
        self.description = description
        self.input_data = input_data
        self.expected_outputs = expected_outputs
        self.difficulty = difficulty
        self.domain = domain
        self.available_tools = available_tools or []
        self.reasoning_mode = reasoning_mode

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProblemDefinition":
        """Create a ProblemDefinition from a dictionary"""
        return cls(
            problem_id=data["problem_id"],
            title=data["title"],
            description=data["description"],
            input_data=data["input_data"],
            expected_outputs=data["expected_outputs"],
            difficulty=data.get("difficulty", "medium"),
            domain=data.get("domain", "text_processing"),
            available_tools=data.get("available_tools", []),
            reasoning_mode=data.get("reasoning_mode", "deduction")
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary"""
        return {
            "problem_id": self.problem_id,
            "title": self.title,
            "description": self.description,
            "input_data": self.input_data,
            "expected_outputs": self.expected_outputs,
            "difficulty": self.difficulty,
            "domain": self.domain,
            "available_tools": self.available_tools,
            "reasoning_mode": self.reasoning_mode
        }


class TrajectoryState:
    """
    Represents the state of a single trajectory in the environment.

    This class keeps track of conversation history, execution results,
    and other state needed for the recursive improvement cycle.
    """

    def __init__(self, problem: ProblemDefinition, conversation_id: str):
        self.problem = problem
        self.conversation_id = conversation_id
        self.message_history: List[Dict[str, str]] = []
        self.current_state = ConversationState.INITIAL
        self.current_iteration = 0
        self.execution_results: List[ExecutionResult] = []
        self.rewards: List[float] = []
        self.critiques: List[str] = []  # Store critiques for analysis
        # Add tracking for best composition
        self.best_composition: Optional[str] = None
        self.best_reward: float = 0.0
        self.best_iteration: int = 0
        # Simplified metadata with only essential tracking
        self.metadata: Dict[str, Any] = {
            "problem_id": problem.problem_id,
            "iterations": 0,
            "final_reward": 0.0,
            "converged": False,
            "improvement_trend": [],
            "early_stop_reason": None,  # Why the trajectory stopped (max_iterations, convergence, perfect_score)
            "best_found_at_iteration": 0  # Track when the best composition was found
        }

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history"""
        self.message_history.append({"role": role, "content": content})

    def get_last_assistant_message(self) -> Optional[str]:
        """Get the content of the last assistant message"""
        for msg in reversed(self.message_history):
            if msg["role"] == "assistant":
                return msg["content"]
        return None

    def update_state(self, new_state: ConversationState) -> None:
        """Update the conversation state"""
        self.current_state = new_state

        # Update iteration counter if we're moving to the improvement state
        if new_state == ConversationState.IMPROVEMENT:
            self.current_iteration += 1
            self.metadata["iterations"] = self.current_iteration

    def add_execution_result(self, result: ExecutionResult) -> None:
        """Add an execution result"""
        self.execution_results.append(result)

    def add_reward(self, reward: float) -> None:
        """Add a reward for the current iteration"""
        self.rewards.append(reward)

        if len(self.rewards) > 1:
            # Calculate improvement from previous iteration
            improvement = self.rewards[-1] - self.rewards[-2]
            self.metadata["improvement_trend"].append(improvement)

        # Update final reward
        self.metadata["final_reward"] = self.rewards[-1]

    def add_critique(self, critique: Optional[str]) -> None:
        """Add a critique for tracking and analysis"""
        if critique:
            self.critiques.append(critique)
        else:
            self.critiques.append("")  # Empty string if no critique found

    def update_execution_metrics(self, result: ExecutionResult) -> None:
        """Update execution metrics based on the latest execution result"""
        # Simplified metrics tracking - just track success/failure
        if not result.success and "error_counts" not in self.metadata:
            self.metadata["error_counts"] = 0

        if not result.success:
            self.metadata["error_counts"] += 1
            
    def update_best_composition(self, composition: str, reward: float) -> bool:
        """
        Store composition if it's better than previous best.
        
        Args:
            composition: The composition code to consider
            reward: The reward value of this composition
            
        Returns:
            True if this was an improvement and stored as best, False otherwise
        """
        if reward > self.best_reward:
            self.best_composition = composition
            self.best_reward = reward
            self.best_iteration = self.current_iteration
            self.metadata["best_found_at_iteration"] = self.current_iteration
            return True
        return False


class RecursiveToolImprovementEnv(BaseEnv):
    """
    Environment for training LLMs to create and improve tool compositions.

    This environment implements a recursive tool improvement cycle for language models,
    enabling them to learn higher-order tool composition skills through a process of
    self-critique and continuous refinement.

    The core interaction loop involves:
    1. Create initial tool compositions for problems
    2. Execute the compositions and observe results
    3. Critique their approach
    4. Improve the compositions based on self-critique
    5. Repeat iteratively

    This environment builds on concepts from Absolute-Zero-Reasoner (self-proposal mechanism),
    Tool-N1 (binary verification reward), and BespokeLabs (multi-turn tool learning techniques).

    Key Features:
    - ToolRegistry with configurable tool sets for problem-solving
    - Sandboxed ExecutionEngine for safe code execution
    - Binary verification for clean reward signals
    - Multi-turn conversation management
    - Support for different reasoning modes (deduction, abduction, induction)
    - Progressive improvement tracking
    - WandB integration for metrics visualization

    This environment can be used with the Atropos framework for both online training
    (via 'serve' command) and offline trajectory generation (via 'process' command).
    """

    name = "recursive_tool_improvement"
    env_config_cls = RecursiveToolImprovementConfig

    def __init__(
        self,
        config: RecursiveToolImprovementConfig,
        server_configs: Union[List[APIServerConfig], Any],
        slurm: bool = True,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)

        # Initialize empty dataset containers
        self.train_problems: List[ProblemDefinition] = []
        self.eval_problems: List[ProblemDefinition] = []

        # Initialize training state
        self.current_problem_idx = 0
        self.active_trajectories: Dict[str, TrajectoryState] = {}

        # Initialize metrics tracking
        self.success_rate_buffer = []
        self.improvement_rate_buffer = []
        self.convergence_speed_buffer = []

        # Configuration for environment tools
        self.registry = default_registry  # Use default tool registry for now
        self.execution_engine = ExecutionEngine(
            registry=self.registry,
            timeout=config.timeout,
            memory_limit=config.memory_limit
        )

        # Initialize reward function
        self.reward_fn = BinaryVerificationReward(
            semantic_equivalence=True
        )

        # Initialize placeholder for evaluation metrics to avoid AttributeError
        # when wandb_log is called before any evaluation pass has populated
        # this list.
        self.eval_metrics: list[tuple[str, float]] = []

    @classmethod
    def config_init(cls) -> Tuple[RecursiveToolImprovementConfig, List[APIServerConfig]]:
        """
        Initialize the configuration for the environment.
        """
        env_config = RecursiveToolImprovementConfig(
            max_iterations=3,
            tool_set="basic",
            timeout=5,
            verification_type="binary",
            reasoning_mode=ReasoningMode.DEDUCTION,
            use_thinking_tags=True,
            eval_ratio=0.1,
            group_size=8,
            max_token_length=8192,
            include_messages=True
        )

        # Use a single API server for now
        server_configs = [
            APIServerConfig(
                model_name="gpt-4.1-mini",  # Default model
                base_url=None,  # Will use OpenAI's base URL
                api_key=None,   # Will use environment variable
                num_max_requests_at_once=16,
                num_requests_for_eval=32,
            )
        ]

        return env_config, server_configs

    async def setup(self):
        """
        Set up the environment by loading problems and initializing tools.
        """
        logger.info("Setting up Recursive Tool Improvement Environment")

        # Create problem dataset directory if it doesn't exist
        os.makedirs(os.path.join(os.path.dirname(__file__), self.config.data_path), exist_ok=True)

        # Load problem datasets or create default examples if none exist
        await self._load_problems()

        # If no problems found or created, raise an error
        if not self.train_problems and not self.eval_problems:
            raise ValueError("No problems found or created for the environment")

        logger.info(f"Loaded {len(self.train_problems)} training problems and {len(self.eval_problems)} evaluation problems")

    async def _load_problems(self):
        """
        Load problems from JSON files in the data directory.
        If no files exist, create default test problems.
        """
        problem_dir = os.path.join(os.path.dirname(__file__), self.config.data_path)
        problem_files = list(Path(problem_dir).glob("*.json"))

        all_problems = []

        # Load problems from files
        for file_path in problem_files:
            try:
                with open(file_path, "r") as f:
                    problem_data = json.load(f)
                    problem = ProblemDefinition.from_dict(problem_data)
                    all_problems.append(problem)
            except Exception as e:
                logger.warning(f"Error loading problem file {file_path}: {e}")

        # If no problems found, create default test problems
        if not all_problems:
            logger.info("No problem files found, creating default test problems")
            all_problems = self._create_default_problems()

            # Save default problems to files
            try:
                logger.info(f"Saving {len(all_problems)} default problems to {problem_dir}")
                for problem in all_problems:
                    file_path = os.path.join(problem_dir, f"{problem.problem_id}.json")
                    # Make sure the directory exists
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, "w") as f:
                        # Use custom serializable format for any non-JSON-serializable objects
                        problem_dict = problem.to_dict()
                        json.dump(problem_dict, f, indent=2)
                    logger.info(f"Saved problem {problem.problem_id} to {file_path}")
            except Exception as e:
                logger.error(f"Error saving default problems to disk: {e}")
                # Continue execution even if saving fails

        # Shuffle problems with a fixed seed for reproducibility
        random.seed(42)
        random.shuffle(all_problems)

        # Split into train and evaluation sets
        eval_count = max(1, int(len(all_problems) * self.config.eval_ratio))
        self.eval_problems = all_problems[:eval_count]
        self.train_problems = all_problems[eval_count:]

    def _create_default_problems(self) -> List[ProblemDefinition]:
        """
        Create a default set of tool composition problems.

        Implemented as a 4-tier curriculum of problems with increasing difficulty:
        - Tier 1: Ultra-simple, single-tool problems (success rate target: 90%+)
        - Tier 2: Two-tool simple compositions (success rate target: 60%+)
        - Tier 3: Multi-step medium complexity (success rate target: 30%+)
        - Tier 4: Complex edge-case handling (success rate target: 10%+)

        Returns:
            A list of default problems
        """
        problems = []

        # ===== TIER 1: Ultra-simple single-tool problems =====
        # Problem 1: Simple Text Transformation (single tool use)
        problems.append(ProblemDefinition(
            problem_id="tier1-uppercase",
            title="Simple Text Transformation",
            description="Convert the input text to uppercase.",
            input_data="hello world",
            expected_outputs="HELLO WORLD",
            difficulty="easy",
            domain="text_processing",
            available_tools=["uppercase_text"],
            reasoning_mode="deduction"
        ))

        # Problem 2: Simple Math Operation (single tool use)
        problems.append(ProblemDefinition(
            problem_id="tier1-count",
            title="Character Count",
            description="Count the number of characters in the input string.",
            input_data="Programming is fun!",
            expected_outputs=18,
            difficulty="easy",
            domain="text_analysis",
            available_tools=["count_chars"],
            reasoning_mode="deduction"
        ))

        # ===== TIER 2: Two-tool simple compositions =====
        # Problem 3: Two-Step Text Processing
        problems.append(ProblemDefinition(
            problem_id="tier2-reverse-uppercase",
            title="Reverse and Uppercase",
            description="Reverse the input string and convert it to uppercase.",
            input_data="hello world",
            expected_outputs="DLROW OLLEH",
            difficulty="easy",
            domain="text_processing",
            available_tools=["reverse_text", "uppercase_text"],
            reasoning_mode="deduction"
        ))

        # Problem 4: Simple List Operation
        problems.append(ProblemDefinition(
            problem_id="tier2-filter-sort",
            title="Filter and Sort",
            description="Filter out numbers less than 10 from the input list, then sort in ascending order.",
            input_data=[5, 15, 3, 20, 8, 12, 7, 25],
            expected_outputs=[12, 15, 20, 25],
            difficulty="easy",
            domain="data_processing",
            available_tools=["filter_list", "sort_list"],
            reasoning_mode="deduction"
        ))

        # ===== TIER 3: Multi-step medium complexity =====
        # Problem 1: Word Extraction and Formatting
        problems.append(ProblemDefinition(
            problem_id="tier3-extract-01",
            title="Extract and Format Words",
            description=(
                "Create a tool composition that: "
                "1. Extracts all words containing the letter 'a' "
                "2. Converts them to uppercase "
                "3. Joins them with commas"
            ),
            input_data="The quick brown fox jumps over the lazy dog",
            expected_outputs="LAZY",
            difficulty="medium",
            domain="text_processing",
            available_tools=[
                "split_text", "filter_list", "uppercase_text", "join_text", 
                "extract_regex", "sort_list", "remove_duplicates"
            ],
            reasoning_mode="deduction"
        ))

        # Problem 2: Multi-Step Data Transformation
        # This requires multiple steps with precise formatting that's easy to get wrong initially
        problems.append(ProblemDefinition(
            problem_id="tier3-multi-step-01",
            title="Multi-Step Data Transformation",
            description=(
                "Create a tool composition that processes the input data by: "
                "1. Extracting all words containing 'a' or 'e' "
                "2. Converting them to title case (first letter uppercase, rest lowercase) "
                "3. Sorting them by length (shortest first) "
                "4. Joining them with semicolons "
                "5. Adding 'Result: ' as a prefix"
            ),
            input_data="The quick brown fox jumps over the lazy dog and runs away from the hunter",
            expected_outputs="Result: The;And;Away;Lazy;Over;Jumps;Hunter",
            difficulty="medium",
            domain="text_processing",
            available_tools=[
                "split_text", "filter_list", "join_text", "replace_text", 
                "regex_replace", "sort_list", "titlecase_text"
            ],
            reasoning_mode="deduction"
        ))

        # ===== TIER 4: Complex edge-case handling =====
        # Problem 3: Deliberately Ambiguous Problem
        # This problem has ambiguous requirements that will likely need clarification in iterations
        problems.append(ProblemDefinition(
            problem_id="tier4-ambiguous-01",
            title="Process Structured Data",
            description=(
                "Create a tool composition that extracts and formats data from the input. "
                "The output should be properly formatted and include all relevant information."
            ),
            input_data={
                "users": [
                    {"id": 1, "name": "Alice Smith", "email": "alice@example.com", "active": True},
                    {"id": 2, "name": "Bob Jones", "email": "bob@example.com", "active": False},
                    {"id": 3, "name": "Charlie Brown", "email": "charlie@example.com", "active": True}
                ]
            },
            expected_outputs="Active users: Alice Smith (alice@example.com), Charlie Brown (charlie@example.com)",
            difficulty="hard",
            domain="data_processing",
            available_tools=[
                "parse_json", "format_json", "filter_list", "join_text"
            ],
            reasoning_mode="deduction"
        ))

        # Problem 4: Exact Format Matching
        # This problem requires precise output formatting that's likely to need refinement
        problems.append(ProblemDefinition(
            problem_id="tier4-format-matching-01",
            title="Format Data Precisely",
            description=(
                "Create a tool composition that extracts dates from the input text and formats them "
                "according to ISO 8601 standard (YYYY-MM-DDThh:mm:ssZ). Return them as a JSON array."
            ),
            input_data=(
                "Meeting on 05/20/2023 at 2:30 PM, Deadline on 04/15/2023 at 11:59 PM, "
                "Launch on 07/01/2023 at 9:00 AM."
            ),
            expected_outputs='["2023-05-20T14:30:00Z", "2023-04-15T23:59:00Z", "2023-07-01T09:00:00Z"]',
            difficulty="hard",
            domain="data_extraction",
            available_tools=[
                "extract_regex", "regex_replace", "format_json", "format_date"
            ],
            reasoning_mode="deduction"
        ))

        return problems

    def _generate_system_prompt(self, problem: ProblemDefinition) -> str:
        """
        Generate the system prompt for a given problem.

        Args:
            problem: The problem to generate a prompt for

        Returns:
            The system prompt text
        """
        # Format available tools for the problem
        tool_descriptions = self.registry.format_for_prompt()

        # Format input data summary (handle different types)
        if isinstance(problem.input_data, dict):
            input_summary = json.dumps(problem.input_data, indent=2)
        elif isinstance(problem.input_data, list):
            input_summary = json.dumps(problem.input_data, indent=2)
        else:
            input_summary = str(problem.input_data)

        # Format expected outputs
        if isinstance(problem.expected_outputs, dict):
            output_summary = json.dumps(problem.expected_outputs, indent=2)
        elif isinstance(problem.expected_outputs, list):
            output_summary = json.dumps(problem.expected_outputs, indent=2)
        else:
            output_summary = str(problem.expected_outputs)

        # Build the prompt
        prompt = (
            "You are an expert in solving problems by creating and improving tool compositions.\n\n"
            "Available Tools:\n"
            f"{tool_descriptions}\n\n"
            "Problem:\n"
            f"{problem.description}\n\n"
            "Input Data:\n"
            f"{input_summary}\n\n"
            "Expected Outputs:\n"
            f"{output_summary}\n\n"
        )

        # Add reasoning mode
        if problem.reasoning_mode:
            prompt += f"Reasoning Mode: {problem.reasoning_mode}\n\n"

        # Add task instructions
        prompt += (
            "Your task is to:\n"
            "1. Create a tool composition to solve this problem\n"
            "2. Execute the composition and observe the results\n"
            "3. Critique your solution, identifying specific limitations and improvement opportunities\n"
            "4. Create an improved tool composition based on your critique\n\n"
        )

        # Add format instructions with clearer and more explicit guidance
        prompt += (
            "IMPORTANT: Format your response exactly as follows:\n\n"
        )

        if self.config.use_thinking_tags:
            prompt += (
                "<thinking>\n"
                "Step 1: Analyze the problem requirements\n"
                "Step 2: Identify which tools to use\n"
                "Step 3: Design a solution approach\n"
                "</thinking>\n\n"
            )

        prompt += (
            "<composition>\n"
            "def solve(input_data):\n"
            "    # Your tool composition code here\n"
            "    # This function MUST be named 'solve' and take 'input_data' as parameter\n"
            "    # Use ONLY the available tools defined above\n"
            "    # The function MUST return a result matching the expected output format\n"
            "    # PUT YOUR COMPLETE SOLUTION BETWEEN THESE TAGS\n"
            "</composition>\n\n"
        )
        
        # Add examples of correct vs incorrect formats
        prompt += (
            "CRITICAL FORMAT REQUIREMENTS:\n"
            "1. Your solution MUST be inside <composition> tags\n"
            "2. Your function MUST be named 'solve' with 'input_data' parameter\n"
            "3. Do not omit the opening or closing tags\n"
            "4. The function must return a result that matches the expected output format\n\n"
            
            "Example of CORRECT format:\n"
            "<composition>\n"
            "def solve(input_data):\n"
            "    result = uppercase_text(input_data)\n"
            "    return result\n"
            "</composition>\n\n"
            
            "Example of INCORRECT format (missing tags):\n"
            "def solve(input_data):\n"
            "    result = uppercase_text(input_data)\n"
            "    return result\n\n"
        )

        return prompt

    def _generate_initial_problem_prompt(self, problem: ProblemDefinition) -> str:
        """
        Generate the initial user prompt for a given problem.

        Args:
            problem: The problem to generate a prompt for

        Returns:
            The user prompt text
        """
        return (
            f"Please solve this {problem.difficulty} problem in {problem.domain}.\n\n"
            "Create a tool composition that uses the available tools to process the input data "
            "and produce the expected output. Show your thinking process and then provide "
            "your solution as a Python function named 'solve' that takes 'input_data' as a parameter."
        )

    def _generate_execution_result_prompt(self, result: ExecutionResult, expected_output: Any) -> str:
        """
        Generate a prompt showing the execution results of a tool composition.

        Enhanced version with more detailed feedback to help the model improve.

        Args:
            result: The execution result
            expected_output: The expected output for comparison

        Returns:
            The execution result prompt text
        """
        if result.success:
            output_str = f"Execution successful with result: {repr(result.result)}"

            if result.result == expected_output:
                output_str += "\n\n✅ Your result MATCHES the expected output. Great job!"
            else:
                output_str += f"\n\n❌ Your result does NOT match the expected output: {repr(expected_output)}"

                # Add more detailed comparison for better feedback
                if isinstance(result.result, str) and isinstance(expected_output, str):
                    # For string outputs, show a character-by-character comparison
                    if len(result.result) != len(expected_output):
                        output_str += f"\n\nLength difference: Your result has {len(result.result)} characters, expected output has {len(expected_output)} characters."

                    # Find the first difference
                    for i, (a, b) in enumerate(zip(result.result, expected_output)):
                        if a != b:
                            output_str += f"\n\nFirst difference at position {i}: '{a}' vs '{b}'"
                            break

                elif isinstance(result.result, (list, tuple)) and isinstance(expected_output, (list, tuple)):
                    # For list outputs, show length and content differences
                    if len(result.result) != len(expected_output):
                        output_str += f"\n\nLength difference: Your result has {len(result.result)} items, expected output has {len(expected_output)} items."

                    # Find differences in items
                    common = set(result.result) & set(expected_output)
                    only_in_result = set(result.result) - set(expected_output)
                    only_in_expected = set(expected_output) - set(result.result)

                    if common:
                        output_str += f"\n\nCorrect items: {common}"
                    if only_in_result:
                        output_str += f"\n\nExtra items in your result: {only_in_result}"
                    if only_in_expected:
                        output_str += f"\n\nMissing items from your result: {only_in_expected}"
        else:
            output_str = f"Execution failed with error: {result.error}"

            # Add more helpful error analysis
            if "NameError" in str(result.error):
                output_str += "\n\nThis looks like you're using a variable or function that doesn't exist. Check for typos or missing definitions."
            elif "TypeError" in str(result.error):
                output_str += "\n\nThis looks like you're using the wrong type of data with a function. Check the parameters you're passing to functions."
            elif "IndexError" in str(result.error):
                output_str += "\n\nThis looks like you're trying to access an index that doesn't exist in a list or string."
            elif "KeyError" in str(result.error):
                output_str += "\n\nThis looks like you're trying to access a key that doesn't exist in a dictionary."
            elif "AttributeError" in str(result.error):
                output_str += "\n\nThis looks like you're trying to use a method or property that doesn't exist for this type of object."
            elif "SyntaxError" in str(result.error):
                output_str += "\n\nThere's a syntax error in your code. Check for missing parentheses, quotes, or other syntax elements."

        # Add execution stats
        output_str += f"\n\nExecution time: {result.execution_time:.3f} seconds"
        output_str += f"\nNumber of tool calls: {len(result.tool_calls)}"

        # Add tool call trace if available
        if result.tool_calls:
            output_str += "\n\nTool call trace:"
            for i, call in enumerate(result.tool_calls):
                output_str += f"\n{i+1}. {call['tool']}(args={call['args']}, kwargs={call['kwargs']})"

        # Add stdout/stderr if available
        if result.stdout.strip():
            output_str += f"\n\nStandard output:\n{result.stdout.strip()}"
        if result.stderr.strip():
            output_str += f"\n\nStandard error:\n{result.stderr.strip()}"

        # Add suggestions for improvement
        output_str += "\n\nSuggestions for improvement:"
        if not result.success:
            output_str += "\n1. Fix the error in your code before making other improvements."
            output_str += "\n2. Make sure you're using the correct tool functions with the right parameters."
            output_str += "\n3. Check for proper error handling in your code."
        elif result.result != expected_output:
            output_str += "\n1. Compare your output format with the expected output format."
            output_str += "\n2. Check if you need to process the data differently to match the expected output."
            output_str += "\n3. Consider if you're using the right tools for this task."
        else:
            output_str += "\n1. Your solution works! Consider if it can be made more efficient or elegant."
            output_str += "\n2. Check if there are edge cases your solution might not handle."
            output_str += "\n3. Consider if you can reduce the number of tool calls or execution time."

        return f"<execution>\n{output_str}\n</execution>"

    def _generate_critique_prompt(self) -> str:
        """
        Generate a prompt asking for critique of the solution.

        Highly structured template with specific questions to guide
        the model toward actionable critique.

        Returns:
            The critique prompt text
        """
        return (
            "Analyze your solution by answering these specific questions:\n\n"
            "## CORRECTNESS\n"
            "- Does the output exactly match the expected format? If not, what's different?\n"
            "- Are there any logical errors in your implementation?\n\n"
            "## EFFICIENCY\n"
            "- Are there unnecessary steps or redundant operations?\n"
            "- Could any operations be combined or simplified?\n\n"
            "## ROBUSTNESS\n"
            "- Would this solution handle edge cases?\n"
            "- Is there any input validation that's missing?\n\n"
            "## NEXT STEPS\n"
            "- What specific change would most improve this solution?\n"
            "- What tool or technique would make this solution better?\n\n"
            "Format your critique with these exact headings, then provide your improved solution in a <composition> block.\n\n"
            "IMPORTANT: After your critique, you MUST include your improved solution like this:\n\n"
            "<composition>\n"
            "def solve(input_data):\n"
            "    # Your improved code here\n"
            "    # Make sure to address the issues identified in your critique\n"
            "    return result  # Must match expected output format\n"
            "</composition>\n\n"
            "Make sure to enclose your improved solution within the <composition> tags exactly as shown above."
        )

    def _generate_improvement_prompt(self, iteration: int, max_iterations: int) -> str:
        """
        Generate a prompt asking for an improved solution.

        Args:
            iteration: Current iteration number
            max_iterations: Maximum number of iterations

        Returns:
            The improvement prompt text
        """
        return (
            f"Based on your critique, please provide an improved version of your tool composition. "
            f"This is iteration {iteration+1} of {max_iterations}.\n\n"
            f"Provide your improved solution in a <composition> block and explain the changes "
            f"you made to improve it."
        )

    def _extract_composition(self, text: str) -> Optional[str]:
        """
        Extract the composition code from the model's response.
        
        Enhanced with better extraction heuristics and fallback mechanisms
        to handle various LLM code formatting patterns.

        Args:
            text: The model's response text

        Returns:
            The extracted composition code, or None if not found
        """
        # Look for composition inside tags (first priority)
        composition_pattern = r"<composition>(.*?)</composition>"
        composition_match = re.search(composition_pattern, text, re.DOTALL)

        if composition_match:
            code = composition_match.group(1).strip()
            # Verify it has a solve function
            if "def solve" in code:
                return code
            # If no solve function but there's code, try to create a wrapper
            elif code and not code.startswith("def "):
                return f"def solve(input_data):\n    {code.replace('\n', '\n    ')}"

        # Look for code blocks with triple backticks (second priority)
        code_block_pattern = r"```(?:python)?(.*?)```"
        code_blocks = re.findall(code_block_pattern, text, re.DOTALL)
        
        for block in code_blocks:
            block = block.strip()
            if "def solve" in block:
                return block

        # Fallback to looking for solve function directly (third priority)
        function_pattern = r"def\s+solve\s*\(.*?\).*?:"
        match = re.search(function_pattern, text, re.DOTALL)

        if match:
            # Get the starting position of the function definition
            start_pos = match.start()

            # Extract the function body (handling indentation)
            lines = text[start_pos:].split('\n')
            function_lines = [lines[0]]

            # Find the indentation of the first line of the function body
            base_indent = ""
            for i in range(1, len(lines)):
                if re.match(r'^\s*$', lines[i]):  # Skip empty lines
                    function_lines.append(lines[i])
                    continue

                if i == 1:  # First non-empty line after function definition
                    indent_match = re.match(r'^(\s+)', lines[i])
                    if not indent_match:  # No indentation, not a valid function
                        function_lines.append("    pass")  # Add minimal valid body
                        break
                    base_indent = indent_match.group(1)
                    function_lines.append(lines[i])
                elif lines[i].startswith(base_indent):  # Line has base indentation
                    function_lines.append(lines[i])
                elif not re.match(r'^\s', lines[i]):  # New unindented line - end of function
                    break
                else:  # Line has different indentation - likely part of nested block
                    indent_match = re.match(r'^(\s+)', lines[i])
                    if indent_match and len(indent_match.group(1)) > len(base_indent):
                        function_lines.append(lines[i])
                    else:
                        break

            return '\n'.join(function_lines)
            
        # Last resort - if we find some code that looks like it might be useful,
        # wrap it in a solve function
        if "input_data" in text and ("return" in text or "print" in text):
            # Look for code-like lines
            code_lines = []
            in_code_section = False
            
            for line in text.split('\n'):
                # Skip empty lines at the beginning
                if not code_lines and not line.strip():
                    continue
                    
                # Code section markers
                if any(marker in line.lower() for marker in ["here's my code", "here is the code", "improved solution", "my solution"]):
                    in_code_section = True
                    continue
                    
                # Skip lines that are clearly explanations
                if line.strip() and not line.startswith('#') and not re.match(r'^[0-9]+\.', line.strip()):
                    if (any(keyword in line for keyword in ["input_data", "return ", "=", "if ", "for ", "while ", "import ", 
                                                          "+", "-", "*", "/", "filter", "map", "sort"]) or 
                        any(tool in line for tool in self.registry.tools.keys())):
                        code_lines.append(line)
                        in_code_section = True
                    elif in_code_section and line.strip():
                        code_lines.append(line)
            
            if code_lines:
                # Determine indentation
                min_indent = min((len(line) - len(line.lstrip())) for line in code_lines if line.strip())
                
                # Remove common indentation
                code_lines = [line[min_indent:] if line.strip() else line for line in code_lines]
                
                # Create a solve function
                code = "def solve(input_data):\n"
                if not any("return" in line for line in code_lines):
                    # Add a return if none exists
                    code_lines.append("    return input_data")
                
                # Indent and join
                code += "\n".join(f"    {line}" for line in code_lines)
                return code
        
        # No valid code found
        logger.warning(f"No valid composition found in response. Response start: {text[:100]}...")
        return None

    def _extract_critique(self, text: str) -> Optional[str]:
        """
        Extract the critique from the model's response.

        Args:
            text: The model's response text

        Returns:
            The extracted critique, or None if not found
        """
        critique_pattern = r"<critique>(.*?)</critique>"
        critique_match = re.search(critique_pattern, text, re.DOTALL)

        if critique_match:
            return critique_match.group(1).strip()

        # If no tags, look for critique section in free text
        if "<composition>" in text:
            # Look for text between execution results and composition
            execution_end = text.find("</execution>")
            composition_start = text.find("<composition>")

            if execution_end != -1 and composition_start != -1 and execution_end < composition_start:
                critique = text[execution_end + 12:composition_start].strip()
                if critique:
                    return critique

        return None

    async def get_next_item(self) -> Optional[Item]:
        """
        Get the next problem to solve from the training problem set.

        Simplified implementation that creates a new trajectory for a problem.

        Returns:
            An Item object containing (conversation_id, problem_id), or None if no problems available
        """
        # Choose the next problem from training set
        if self.current_problem_idx >= len(self.train_problems):
            self.current_problem_idx = 0

        problem = self.train_problems[self.current_problem_idx]
        self.current_problem_idx += 1

        # Generate a unique conversation ID
        conversation_id = f"{problem.problem_id}-{int(time.time())}-{random.randint(1000, 9999)}"

        # Create the initial message sequence
        system_prompt = self._generate_system_prompt(problem)
        user_prompt = self._generate_initial_problem_prompt(problem)

        # Initialize trajectory state (already in INITIAL state)
        trajectory = TrajectoryState(problem, conversation_id)
        trajectory.add_message("system", system_prompt)
        trajectory.add_message("user", user_prompt)

        # Store the active trajectory
        self.active_trajectories[conversation_id] = trajectory

        # Return the conversation ID and problem ID
        return (conversation_id, problem.problem_id)

    async def collect_trajectory(self, item: Item) -> Tuple[Optional[ScoredDataItem], List[Item]]:
        """
        Collect a single trajectory for a problem, handling the full improvement cycle.

        Simplified implementation with three states:
        1. INITIAL: Get initial solution and execute it
        2. IMPROVEMENT: Get critique, improved solution, and execute it
        3. FINAL: Trajectory is complete

        Args:
            item: Tuple containing (conversation_id, problem_id) that uniquely
                  identifies the current trajectory

        Returns:
            Tuple containing:
              - ScoredDataItem (if trajectory is complete) or None (if still in progress)
              - List of backlog items to process (typically the same item if continuing)
        """
        # Extract conversation_id from the item
        conversation_id, _ = item

        # Get the active trajectory or return None if not found
        if conversation_id not in self.active_trajectories:
            logger.error(f"No active trajectory found for conversation_id: {conversation_id}")
            return None, []

        trajectory = self.active_trajectories[conversation_id]
        problem = trajectory.problem

        # Handle the current state of the conversation
        if trajectory.current_state == ConversationState.INITIAL:
            # Get the initial solution from the model
            messages = trajectory.message_history.copy()
            response = await self._get_model_response(messages)
            trajectory.add_message("assistant", response)

            # Extract the composition from the response
            composition = self._extract_composition(response)
            if composition is None:
                logger.error("No composition found in model response")
                # Add detailed error message with clear instructions on how to format the response
                error_message = (
                    "I couldn't find a valid tool composition in your response. "
                    "Please provide your solution following this exact format:\n\n"
                    "<composition>\n"
                    "def solve(input_data):\n"
                    "    # Your solution code here using the available tools\n"
                    "    result = some_tool(input_data)  # Example\n"
                    "    return result  # Must return the expected output format\n"
                    "</composition>\n\n"
                    "Make sure your code includes:\n"
                    "1. The <composition> opening and closing tags\n"
                    "2. A function named 'solve' that takes 'input_data' as a parameter\n"
                    "3. Proper use of the available tools\n"
                    "4. A return statement that produces the expected output format\n"
                )
                trajectory.add_message("user", error_message)
                return None, [item]  # Retry the same conversation
            
            # Execute the composition
            execution_result = self.execution_engine.execute(
                composition, problem.input_data
            )
            trajectory.add_execution_result(execution_result)

            # Calculate reward for the initial solution
            reward = self._calculate_reward(execution_result, problem.expected_outputs)
            trajectory.add_reward(reward)
            
            # Track this as the best composition so far
            trajectory.update_best_composition(composition, reward)

            # If reward is 1.0, we've found a perfect solution, so we can skip to FINAL
            if reward >= 1.0:
                trajectory.update_state(ConversationState.FINAL)
                trajectory.metadata["early_stop_reason"] = "perfect_score"
                logger.info(f"Perfect solution found on first attempt for {conversation_id}")
            else:
                # Add the execution result and critique request to the conversation
                execution_prompt = self._generate_execution_result_prompt(
                    execution_result, problem.expected_outputs
                )
                critique_prompt = self._generate_critique_prompt()
                trajectory.add_message("user", execution_prompt + "\n\n" + critique_prompt)
                trajectory.update_state(ConversationState.IMPROVEMENT)

        elif trajectory.current_state == ConversationState.IMPROVEMENT:
            # Get the critique and improved solution from the model
            messages = trajectory.message_history.copy()
            response = await self._get_model_response(messages)
            trajectory.add_message("assistant", response)

            # Extract the critique and improved composition
            critique = self._extract_critique(response)
            trajectory.add_critique(critique)

            improved_composition = self._extract_composition(response)
            if improved_composition is None:
                logger.error("No improved composition found in model response")
                # Add detailed error message specific to the improvement phase
                error_message = (
                    "I couldn't find your improved solution in the expected format. "
                    "After your critique, please provide your improved solution following this exact format:\n\n"
                    "<composition>\n"
                    "def solve(input_data):\n"
                    "    # Your improved solution code here\n"
                    "    # Make sure to address the issues identified in your critique\n"
                    "    return result  # Must return the expected output format\n"
                    "</composition>\n\n"
                    "Remember to include both your critique AND your improved solution in your response.\n"
                )
                trajectory.add_message("user", error_message)
                return None, [item]  # Retry the same conversation
                
            # Execute the improved composition
            execution_result = self.execution_engine.execute(
                improved_composition, problem.input_data
            )
            trajectory.add_execution_result(execution_result)

            # Calculate reward for the improved solution
            reward = self._calculate_reward(execution_result, problem.expected_outputs)
            trajectory.add_reward(reward)
            
            # Update best composition if this is better
            trajectory.update_best_composition(improved_composition, reward)
            
            # Calculate whether we should continue or stop
            should_continue = True
            stop_reason = None

            # Check if we've reached max iterations
            if trajectory.current_iteration >= self.config.max_iterations:
                should_continue = False
                stop_reason = "max_iterations"

            # Check if we've found a perfect solution
            elif reward >= 1.0:
                should_continue = False
                stop_reason = "perfect_score"

            # Check if improvement is below threshold
            elif len(trajectory.rewards) >= 2:
                improvement = trajectory.rewards[-1] - trajectory.rewards[-2]
                if improvement < self.config.improvement_threshold and trajectory.current_iteration >= 2:
                    should_continue = False
                    stop_reason = "below_improvement_threshold"

            if should_continue:
                # Add the execution result and critique request to the conversation
                execution_prompt = self._generate_execution_result_prompt(
                    execution_result, problem.expected_outputs
                )
                critique_prompt = self._generate_critique_prompt()
                trajectory.add_message("user", execution_prompt + "\n\n" + critique_prompt)
            else:
                # Trajectory is complete, move to FINAL state
                trajectory.update_state(ConversationState.FINAL)
                trajectory.metadata["early_stop_reason"] = stop_reason
                
                # For the final message, make sure we add the best composition ever found if the last one wasn't the best
                if trajectory.best_reward > reward and trajectory.best_composition:
                    # Add a note about using the best composition instead of the last one
                    trajectory.add_message("system", f"USING BEST COMPOSITION FROM ITERATION {trajectory.best_iteration} WITH REWARD {trajectory.best_reward} INSTEAD OF FINAL COMPOSITION WITH REWARD {reward}")
                
                # Log metrics for this trajectory
                logger.info(f"Trajectory {conversation_id} complete. Final reward: {reward}, "
                            f"Iterations: {trajectory.current_iteration}, "
                            f"Best reward: {trajectory.best_reward} at iteration {trajectory.best_iteration}")
                
                # Add to success rate buffer if final solution is correct
                self.success_rate_buffer.append(1.0 if trajectory.best_reward >= 1.0 else 0.0)
                
                # Add to improvement rate buffer if multiple iterations
                if len(trajectory.rewards) > 1:
                    self.improvement_rate_buffer.append(trajectory.best_reward - trajectory.rewards[0])
                
                # Add to convergence speed buffer if converged to correct solution
                if trajectory.best_reward >= 1.0:
                    self.convergence_speed_buffer.append(trajectory.best_iteration)

                # Record iteration count and rewards for analysis
                trajectory.metadata["iteration_info"] = f"Stopping at iteration {trajectory.current_iteration}"

        # If the trajectory is complete, convert it to a ScoredDataItem and return
        if trajectory.current_state == ConversationState.FINAL:
            # Clean up the active trajectory
            scored_data = self._create_scored_data(trajectory)
            del self.active_trajectories[conversation_id]
            return scored_data, []
        
        # If the trajectory is not complete, return None and the same item as backlog
        return None, [item]

    async def _get_model_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Get a response from the language model using the server.

        Enhanced implementation with better error handling and fallback responses.

        Args:
            messages: The conversation history as a list of message dicts

        Returns:
            The model's response as a string
        """
        # Create a better fallback response that follows the expected format
        user_message = messages[-1]["content"] if messages and messages[-1]["role"] == "user" else ""
        
        # Check if this is the initial request or an improvement request
        is_improvement_request = "critique" in user_message.lower() or "improve" in user_message.lower()
        
        # Create appropriate fallback response based on request type
        if is_improvement_request:
            default_response = """
<thinking>
I need to provide a critique of my previous solution and then offer an improved version that addresses the issues I identify.
</thinking>

## CORRECTNESS
- The solution may not correctly match the expected output format.
- There might be logical errors in the implementation.

## EFFICIENCY
- The solution likely has unnecessary operations that could be simplified.
- There are probably opportunities to combine steps for better performance.

## ROBUSTNESS
- The solution might not handle all possible edge cases.
- Input validation could be improved.

## NEXT STEPS
- I should make sure the output format matches exactly what's expected.
- I should simplify the code to be more efficient.

<composition>
def solve(input_data):
    # Simple fallback implementation when API call fails
    if isinstance(input_data, str):
        # String processing fallback
        result = input_data.upper()  # Convert to uppercase as a simple operation
        return result
    elif isinstance(input_data, list):
        # List processing fallback
        if all(isinstance(x, (int, float)) for x in input_data):
            return sorted(input_data)  # Sort numbers
        else:
            return input_data  # Return as is
    elif isinstance(input_data, dict):
        # Dict processing fallback
        return input_data  # Return as is
    else:
        # Default fallback
        return str(input_data)
</composition>
"""
        else:
            default_response = """
<thinking>
I'll analyze this problem step by step to create a solution using the available tools.
</thinking>

<composition>
def solve(input_data):
    # Simple fallback implementation when API call fails
    if isinstance(input_data, str):
        # String processing fallback
        result = input_data.upper()  # Convert to uppercase as a simple operation
        return result
    elif isinstance(input_data, list):
        # List processing fallback
        if all(isinstance(x, (int, float)) for x in input_data):
            return sorted(input_data)  # Sort numbers
        else:
            return input_data  # Return as is
    elif isinstance(input_data, dict):
        # Dict processing fallback
        return input_data  # Return as is
    else:
        # Default fallback
        return str(input_data)
</composition>
"""

        # Add a special formatting hint to the last message to help with formatting
        last_message_idx = -1
        for i, msg in enumerate(messages):
            if msg["role"] == "user":
                last_message_idx = i
                
        if last_message_idx >= 0:
            if is_improvement_request:
                # Add a format hint for improvement requests
                messages[last_message_idx]["content"] += "\n\nIMPORTANT: Your response must include a <composition> section with your improved solution."
            else:
                # Add a format hint for initial solution requests
                messages[last_message_idx]["content"] += "\n\nIMPORTANT: Your response must include a <composition> section with your solution."

        # Generate the chat prompt
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Try to get a response from the model
                completion = await self.server.completion(
                    prompt=prompt,
                    max_tokens=self.config.max_token_length // 2,
                    temperature=0.7,
                    stop=None
                )
                
                response_text = completion.choices[0].text
                
                # Verify the response contains some expected format elements
                if "<composition>" in response_text and "def solve" in response_text:
                    return response_text
                elif attempt < max_retries - 1:
                    # If format is incorrect but we have retries left, log and continue
                    logger.warning(f"Response missing expected format elements (attempt {attempt+1}/{max_retries})")
                    continue
                else:
                    # On final attempt, if format is still incorrect, add tags to make it valid
                    if "def solve" in response_text and "<composition>" not in response_text:
                        # Extract the solve function and wrap it in composition tags
                        function_pattern = r"def\s+solve\s*\(.*?\).*?:"
                        match = re.search(function_pattern, response_text, re.DOTALL)
                        if match:
                            # Try to extract and format the function
                            start_pos = match.start()
                            lines = response_text[start_pos:].split('\n')
                            function_lines = []
                            capture = False
                            for line in lines:
                                if "def solve" in line:
                                    capture = True
                                if capture:
                                    function_lines.append(line)
                                    if line.strip() == "" and len(function_lines) > 3:
                                        # Empty line after function body, likely end of function
                                        break
                            if function_lines:
                                return f"<composition>\n{''.join(function_lines)}\n</composition>"
                    
                    # If we couldn't fix it, log and return default response
                    logger.error(f"Model response missing required format elements, using fallback")
                    return default_response

            except Exception as e:
                # Log the error and retry or return default response on final attempt
                logger.error(f"Error getting model response (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)  # Small delay before retry
                    continue
                else:
                    logger.error(f"All {max_retries} attempts failed, using fallback response")
                    return default_response

        # If all retries failed or other issues occurred
        return default_response

    def _calculate_reward(self, execution_result: ExecutionResult, expected_output: Any) -> float:
        """
        Calculate the reward for an execution result.

        Uses the BinaryVerificationReward directly for consistent rewards based on
        functional correctness. This follows the Tool-N1 research showing that binary
        rewards provide a cleaner learning signal.

        Args:
            execution_result: The result of executing a tool composition
            expected_output: The expected output for comparison

        Returns:
            The calculated reward value (0.0 or 1.0)
        """
        # Early return for execution failures
        if not execution_result.success:
            return 0.0

        # Use the binary verification reward directly
        completion = [{
            "success": execution_result.success,
            "result": execution_result.result
        }]
        
        rewards = self.reward_fn.compute(
            completions=completion,
            expected_output=expected_output
        )
        
        # Log why verification failed if it did
        if rewards[0] == 0.0 and execution_result.success:
            # Check different aspects of failure for debugging
            result = execution_result.result
            if type(result) != type(expected_output):
                logger.info(f"Reward 0: Type mismatch. Got {type(result)}, expected {type(expected_output)}")
            else:
                logger.info(f"Reward 0: Output did not match expected result")
        
        return rewards[0]

    def _create_scored_data(self, trajectory: TrajectoryState) -> ScoredDataItem:
        """
        Create a ScoredDataItem from a completed trajectory.

        Simplified version that focuses on essential data required by Atropos.

        Args:
            trajectory: The completed trajectory with all iterations

        Returns:
            A ScoredDataItem with the structure expected by Atropos
        """
        # Use the entire conversation history
        messages = trajectory.message_history.copy()

        # Add iteration information to the messages for better tracking
        if "iteration_info" in trajectory.metadata:
            iteration_info = trajectory.metadata["iteration_info"]
            # Add as a system message at the end
            messages.append({
                "role": "system",
                "content": f"ITERATION INFO: {iteration_info}. Total iterations: {trajectory.current_iteration}. Rewards: {trajectory.rewards}"
            })

        # Tokenize the conversation
        tokenized = tokenize_for_trainer(self.tokenizer, messages)

        # Use final reward as the score
        score = trajectory.rewards[-1] if trajectory.rewards else 0.0

        # Create a ScoredDataItem with minimal required fields
        # This follows the exact pattern from the working Blackjack example
        scored_data = ScoredDataItem(
            messages=messages if self.config.include_messages else None,
            tokens=tokenized["tokens"],
            masks=tokenized["masks"],
            scores=score,
            advantages=None,
            ref_logprobs=None,
            group_overrides=None,
            overrides={
                "problem_id": trajectory.problem.problem_id,
                "iterations": trajectory.current_iteration,
                "rewards": trajectory.rewards
            }
        )

        # Log what we're returning to help with debugging
        logger.info(f"Created ScoredDataItem with tokens length: {len(tokenized['tokens'])}")
        logger.info(f"Created ScoredDataItem with score: {score}")
        logger.info(f"Created ScoredDataItem with iterations: {trajectory.current_iteration}")
        logger.info(f"Created ScoredDataItem with rewards: {trajectory.rewards}")

        return scored_data

    async def evaluate(self, *args, **kwargs):
        """
        Evaluate the environment on the evaluation problem set.

        This method runs a complete evaluation cycle on the reserved evaluation problems.
        For each problem, it:
        1. Creates an initial solution
        2. Goes through multiple improvement iterations (up to max_iterations)
        3. Records success rates, improvement metrics, and convergence speeds

        The evaluation results are recorded for logging to WandB, tracking:
        - Success rate: Percentage of problems solved correctly
        - Average improvement: Average score increase from first to final solution
        - Average convergence speed: Average iterations needed to reach optimal solution
        - Per-problem performance metrics

        This is called periodically during training based on the steps_per_eval config
        to assess how well the model is learning to perform recursive tool improvement.

        Args:
            *args: Additional positional arguments (not used)
            **kwargs: Additional keyword arguments (not used)

        Note:
            This method pauses normal trajectory collection to perform evaluation,
            controlled by the eval_handling setting in the configuration.
        """
        logger.info("Starting evaluation")

        eval_results = []
        success_count = 0
        improvement_rates = []
        convergence_speeds = []

        # Process each evaluation problem
        for problem in self.eval_problems:
            # Create a trajectory for this problem
            conversation_id = f"eval-{problem.problem_id}-{int(time.time())}"
            trajectory = TrajectoryState(problem, conversation_id)

            # Add initial messages
            system_prompt = self._generate_system_prompt(problem)
            user_prompt = self._generate_initial_problem_prompt(problem)
            trajectory.add_message("system", system_prompt)
            trajectory.add_message("user", user_prompt)

            # Get initial solution
            messages = trajectory.message_history.copy()
            response = await self._get_model_response(messages)
            trajectory.add_message("assistant", response)

            # Execute initial solution
            initial_result = self.execution_engine.execute_from_text(
                response, problem.input_data
            )
            trajectory.add_execution_result(initial_result)

            # Calculate initial reward
            initial_reward = self._calculate_reward(initial_result, problem.expected_outputs)
            trajectory.add_reward(initial_reward)

            # Perform improvement iterations
            current_reward = initial_reward
            for i in range(self.config.max_iterations):
                # Stop if we already have a perfect solution
                if current_reward >= 1.0:
                    break

                # Add execution result and critique request
                execution_prompt = self._generate_execution_result_prompt(
                    trajectory.execution_results[-1], problem.expected_outputs
                )
                critique_prompt = self._generate_critique_prompt()
                trajectory.add_message("user", execution_prompt + "\n\n" + critique_prompt)

                # Get critique and improved solution
                messages = trajectory.message_history.copy()
                response = await self._get_model_response(messages)
                trajectory.add_message("assistant", response)

                # Execute improved solution
                improved_result = self.execution_engine.execute_from_text(
                    response, problem.input_data
                )
                trajectory.add_execution_result(improved_result)

                # Calculate reward for improved solution
                improved_reward = self._calculate_reward(improved_result, problem.expected_outputs)
                trajectory.add_reward(improved_reward)
                current_reward = improved_reward

                # Update iteration counter
                trajectory.current_iteration += 1

            # Record metrics for this problem
            final_reward = trajectory.rewards[-1] if trajectory.rewards else 0.0
            success = final_reward >= 1.0

            if success:
                success_count += 1
                convergence_speeds.append(trajectory.current_iteration)

            if len(trajectory.rewards) > 1:
                improvement = trajectory.rewards[-1] - trajectory.rewards[0]
                improvement_rates.append(improvement)

            # Add to results
            eval_results.append({
                "problem_id": problem.problem_id,
                "success": success,
                "initial_reward": initial_reward,
                "final_reward": final_reward,
                "iterations": trajectory.current_iteration,
                "improvement": final_reward - initial_reward if len(trajectory.rewards) > 1 else 0.0
            })

        # Calculate overall metrics
        success_rate = success_count / len(self.eval_problems) if self.eval_problems else 0.0
        avg_improvement = sum(improvement_rates) / len(improvement_rates) if improvement_rates else 0.0
        avg_convergence = sum(convergence_speeds) / len(convergence_speeds) if convergence_speeds else 0.0

        # Log metrics to wandb
        self.eval_metrics = [
            ("eval/success_rate", success_rate),
            ("eval/avg_improvement", avg_improvement),
            ("eval/avg_convergence_speed", avg_convergence),
            ("eval/problems_evaluated", len(self.eval_problems))
        ]

        logger.info(f"Evaluation complete. Success rate: {success_rate:.2f}, Average improvement: {avg_improvement:.2f}")

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """
        Log metrics to wandb.
        
        Enhanced with specific improvement-tracking metrics to better analyze
        the patterns of tool composition refinement across iterations.
        """
        if wandb_metrics is None:
            wandb_metrics = {}

        # Add training metrics if available
        if self.success_rate_buffer:
            wandb_metrics["train/success_rate"] = sum(self.success_rate_buffer) / len(self.success_rate_buffer)
            self.success_rate_buffer = []

        if self.improvement_rate_buffer:
            wandb_metrics["train/avg_improvement"] = sum(self.improvement_rate_buffer) / len(self.improvement_rate_buffer)
            self.improvement_rate_buffer = []

        if self.convergence_speed_buffer:
            wandb_metrics["train/avg_convergence_speed"] = sum(self.convergence_speed_buffer) / len(self.convergence_speed_buffer)
            self.convergence_speed_buffer = []
            
        # Add detailed improvement tracking metrics
        # These metrics help understand how the improvement process is working
        
        # Calculate first iteration success rate
        if hasattr(self, 'active_trajectories') and self.active_trajectories:
            first_iter_successes = sum(1 for t in self.active_trajectories.values() 
                                      if t.rewards and len(t.rewards) > 0 and t.rewards[0] >= 1.0)
            total_trajectories = len(self.active_trajectories)
            
            if total_trajectories > 0:
                wandb_metrics["improvement/first_iteration_success_rate"] = first_iter_successes / total_trajectories
                
                # Calculate final iteration success rate
                final_iter_successes = sum(1 for t in self.active_trajectories.values() 
                                          if t.rewards and t.rewards[-1] >= 1.0)
                wandb_metrics["improvement/final_iteration_success_rate"] = final_iter_successes / total_trajectories
                
                # Calculate positive trajectory rate (number of trajectories where reward improved)
                positive_trajectories = sum(1 for t in self.active_trajectories.values() 
                                           if len(t.rewards) > 1 and t.rewards[-1] > t.rewards[0])
                wandb_metrics["improvement/positive_trajectory_rate"] = positive_trajectories / total_trajectories
                
                # Calculate negative trajectory rate (number of trajectories where reward decreased)
                negative_trajectories = sum(1 for t in self.active_trajectories.values() 
                                           if len(t.rewards) > 1 and t.rewards[-1] < t.rewards[0])
                wandb_metrics["improvement/negative_trajectory_rate"] = negative_trajectories / total_trajectories
                
                # Calculate best iteration distribution (at which iteration was the best solution found)
                best_iter_counts = Counter([t.best_iteration for t in self.active_trajectories.values() 
                                           if hasattr(t, 'best_iteration')])
                for iter_num, count in best_iter_counts.items():
                    wandb_metrics[f"improvement/best_at_iteration_{iter_num}"] = count / total_trajectories
                
                # Track tier-specific success rates
                tiers = {"tier1": [], "tier2": [], "tier3": [], "tier4": []}
                for t in self.active_trajectories.values():
                    if t.rewards and t.problem and t.problem.problem_id:
                        for tier in tiers:
                            if t.problem.problem_id.startswith(tier):
                                tiers[tier].append(1.0 if t.rewards[-1] >= 1.0 else 0.0)
                
                for tier, results in tiers.items():
                    if results:
                        wandb_metrics[f"curriculum/{tier}_success_rate"] = sum(results) / len(results)

        # Add evaluation metrics **if any have been recorded**
        if self.eval_metrics:
            for key, value in self.eval_metrics:
                wandb_metrics[key] = value
            # Clear after logging to avoid duplicate entries in future calls
            self.eval_metrics = []

        # Call parent method to log metrics
        await super().wandb_log(wandb_metrics)

    async def collect_trajectories(self, item: Item):
        """Collect *one* group of completed trajectories.

        We synchronously iterate each trajectory (re-invoking ``collect_trajectory``
        on the same *item* until it returns a ``ScoredDataItem``) so that every
        trajectory in the returned group is finished.  This avoids the repeated
        *Failed to process group, retrying…* loop seen in **process** mode where
        partially-completed rollouts caused `None` to be returned.
        """

        async def _run_to_completion(start_item: Item) -> Tuple[ScoredDataItem, List[Item]]:
            """Iteratively call ``collect_trajectory`` until a ScoredDataItem is
            produced.  Any backlog produced along the way is merged so we don't
            lose pending work."""
            pending_item: Item = start_item
            backlog_accum: List[Item] = []
            while True:
                result, backlog = await self.collect_trajectory(pending_item)
                backlog_accum.extend(backlog)
                if result is not None:
                    return result, backlog_accum
                # When the rollout isn't finished, the env returns the same
                # conversation *item* in the backlog so we continue working on
                # it.  If for some reason the backlog is empty (shouldn't
                # happen), we reuse the original *start_item* to keep going.
                pending_item = backlog[0] if backlog else start_item

        completed_items: List[ScoredDataItem] = []
        backlog: List[Item] = []

        # Collect exactly ``group_size`` finished trajectories.
        for _ in range(self.config.group_size):
            scored_item, extra_backlog = await _run_to_completion(item)
            completed_items.append(scored_item)
            backlog.extend(extra_backlog)

        # Package into ScoredDataGroup as expected by the trainer.
        grouped: ScoredDataGroup = {
            "tokens": [ci["tokens"] for ci in completed_items],
            "masks": [ci["masks"] for ci in completed_items],
            "scores": [ci["scores"] for ci in completed_items],
            "advantages": [],
            "ref_logprobs": [],
            "messages": [],
            "group_overrides": {},
            "overrides": [],
        }

        for ci in completed_items:
            if ci.get("advantages") is not None:
                grouped["advantages"].append(ci["advantages"])
            if ci.get("ref_logprobs") is not None:
                grouped["ref_logprobs"].append(ci["ref_logprobs"])
            if ci.get("messages") is not None:
                # Convert nested message dicts into a single markdown string for
                # each trajectory.  This keeps the JSONL compatible with the
                # `jsonl2html` renderer which expects a **string** per message.
                message_str = "\n\n".join(
                    [
                        f"**{m['role']}**: {m['content']}"
                        if isinstance(m, dict) and 'role' in m and 'content' in m
                        else str(m)
                        for m in ci["messages"]
                    ]
                )
                grouped["messages"].append(message_str)
            if ci.get("group_overrides") is not None:
                grouped["group_overrides"].update(ci["group_overrides"])
            if ci.get("overrides") is not None:
                grouped["overrides"].append(ci["overrides"])

        return grouped, backlog


if __name__ == "__main__":
    RecursiveToolImprovementEnv.cli()