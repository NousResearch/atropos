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
from typing import Any, Dict, List, Optional, Tuple, Union

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
from atroposlib.utils.message_history_utils import (
    ensure_trajectory_token_limit,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

from .tool_registry import ToolRegistry, default_registry
from .execution_engine import ExecutionEngine, ExecutionResult
from .reward_functions.binary_verification import BinaryVerificationReward


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
    
    This helps track where we are in the recursive improvement cycle.
    """
    
    PROBLEM_PRESENTED = "problem_presented"  # Initial problem statement
    INITIAL_SOLUTION = "initial_solution"    # First solution provided
    EXECUTION_RESULT = "execution_result"    # Results of execution
    CRITIQUE = "critique"                    # Self-critique of solution
    IMPROVED_SOLUTION = "improved_solution"  # Improved solution based on critique
    FINAL_RESULT = "final_result"            # Final results after iterations


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
        self.current_state = ConversationState.PROBLEM_PRESENTED
        self.current_iteration = 0
        self.execution_results: List[ExecutionResult] = []
        self.rewards: List[float] = []
        self.metadata: Dict[str, Any] = {
            "problem_id": problem.problem_id,
            "iterations": 0,
            "final_reward": 0.0,
            "converged": False,
            "improvement_trend": []
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
        
        # Update iteration counter if we've completed a cycle
        if new_state == ConversationState.IMPROVED_SOLUTION:
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


class RecursiveToolImprovementEnv(BaseEnv):
    """
    Environment for training LLMs to create and improve tool compositions.
    
    This environment focuses on a multi-turn interaction where models:
    1. Create initial tool compositions for problems
    2. Execute the compositions and observe results
    3. Critique their approach
    4. Improve the compositions based on self-critique
    5. Repeat iteratively
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
            for problem in all_problems:
                file_path = os.path.join(problem_dir, f"{problem.problem_id}.json")
                with open(file_path, "w") as f:
                    json.dump(problem.to_dict(), f, indent=2)
        
        # Shuffle problems with a fixed seed for reproducibility
        random.seed(42)
        random.shuffle(all_problems)
        
        # Split into train and evaluation sets
        eval_count = max(1, int(len(all_problems) * self.config.eval_ratio))
        self.eval_problems = all_problems[:eval_count]
        self.train_problems = all_problems[eval_count:]
    
    def _create_default_problems(self) -> List[ProblemDefinition]:
        """
        Create a set of default test problems for the environment.
        
        Returns:
            List of ProblemDefinition objects
        """
        problems = []
        
        # Problem 1: Text Processing - Extract and Format Emails
        problems.append(ProblemDefinition(
            problem_id="text-extraction-01",
            title="Extract and Format Email Addresses",
            description=(
                "Create a tool composition that extracts all email addresses from the input text, "
                "converts them to lowercase, removes any duplicates, and returns them as a sorted "
                "comma-separated list."
            ),
            input_data=(
                "Contact us at Support@Example.com or sales@example.com. For urgent matters, "
                "reach out to support@example.com or EMERGENCY@example.com."
            ),
            expected_outputs="emergency@example.com,sales@example.com,support@example.com",
            difficulty="easy",
            domain="text_processing",
            available_tools=[
                "extract_regex", "to_lowercase", "remove_duplicates", 
                "sort_list", "join_text"
            ],
            reasoning_mode="deduction"
        ))
        
        # Problem 2: Text Transformation - Format JSON Data
        problems.append(ProblemDefinition(
            problem_id="json-processing-01",
            title="Extract and Format JSON Data",
            description=(
                "Create a tool composition that extracts the names and ages from the input JSON data, "
                "formats them as 'Name: X, Age: Y' strings, sorts them alphabetically by name, "
                "and joins them with newlines."
            ),
            input_data={
                "people": [
                    {"name": "Alice", "age": 29, "city": "New York"},
                    {"name": "Bob", "age": 42, "city": "Chicago"},
                    {"name": "Charlie", "age": 35, "city": "Los Angeles"}
                ]
            },
            expected_outputs="Name: Alice, Age: 29\nName: Bob, Age: 42\nName: Charlie, Age: 35",
            difficulty="medium",
            domain="data_processing",
            available_tools=[
                "parse_json", "format_json", "sort_list", "join_text"
            ],
            reasoning_mode="deduction"
        ))
        
        # Problem 3: Text Cleaning - Clean and Format Data
        problems.append(ProblemDefinition(
            problem_id="text-cleaning-01",
            title="Clean and Format Messy Text",
            description=(
                "Create a tool composition that takes messy text input with extra whitespace, "
                "normalizes it by removing extra spaces, converts to lowercase, "
                "and replaces all instances of 'color' with 'colour'."
            ),
            input_data="  The   COLOR of  the  SKY  is  BLUE. The COLOR of  GRASS  is GREEN.  ",
            expected_outputs="the colour of the sky is blue. the colour of grass is green.",
            difficulty="easy",
            domain="text_processing",
            available_tools=[
                "trim_whitespace", "to_lowercase", "replace_text", "regex_replace"
            ],
            reasoning_mode="deduction"
        ))
        
        # Problem 4: Data Extraction - Parse Structured Data
        problems.append(ProblemDefinition(
            problem_id="data-extraction-01",
            title="Extract Structured Data",
            description=(
                "Create a tool composition that extracts all dates in MM/DD/YYYY format from the input, "
                "converts them to YYYY-MM-DD format, and returns them as a sorted array."
            ),
            input_data=(
                "Important dates: Meeting on 05/20/2023, Deadline on 04/15/2023, "
                "Launch on 07/01/2023, Review on 05/20/2023."
            ),
            expected_outputs=["2023-04-15", "2023-05-20", "2023-07-01"],
            difficulty="medium",
            domain="data_extraction",
            available_tools=[
                "extract_regex", "regex_replace", "sort_list", "remove_duplicates"
            ],
            reasoning_mode="deduction"
        ))
        
        # Problem 5: Counting and Analysis
        problems.append(ProblemDefinition(
            problem_id="text-analysis-01",
            title="Count Word Frequencies",
            description=(
                "Create a tool composition that counts the frequency of each word in the input text, "
                "ignoring case, and returns a sorted list of words with at least 2 occurrences, "
                "in the format 'word: count'."
            ),
            input_data=(
                "The quick brown fox jumps over the lazy dog. The fox is quick and the dog is lazy."
            ),
            expected_outputs=["fox: 2", "lazy: 2", "quick: 2", "the: 4"],
            difficulty="hard",
            domain="text_analysis",
            available_tools=[
                "to_lowercase", "split_text", "count_occurrences"
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
        
        # Add format instructions
        prompt += (
            "Format your response as follows:\n\n"
        )
        
        if self.config.use_thinking_tags:
            prompt += (
                "<thinking>\n"
                "[Your detailed problem analysis and planning]\n"
                "</thinking>\n\n"
            )
        
        prompt += (
            "<composition>\n"
            "def solve(input_data):\n"
            "    # Your tool composition code\n"
            "    # Use the available tools to solve the problem\n"
            "    # Return results in the expected format\n"
            "</composition>\n\n"
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
        
        Args:
            result: The execution result
            expected_output: The expected output for comparison
            
        Returns:
            The execution result prompt text
        """
        if result.success:
            output_str = f"Execution successful with result: {repr(result.result)}"
            if result.result == expected_output:
                output_str += "\n\nYour result MATCHES the expected output."
            else:
                output_str += f"\n\nYour result does NOT match the expected output: {repr(expected_output)}"
        else:
            output_str = f"Execution failed with error: {result.error}"
        
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
        
        return f"<execution>\n{output_str}\n</execution>"
    
    def _generate_critique_prompt(self) -> str:
        """
        Generate a prompt asking for critique of the solution.
        
        Returns:
            The critique prompt text
        """
        if self.config.critique_format == "structured":
            return (
                "Based on the execution results, please critique your solution. "
                "Identify specific limitations, inefficiencies, or bugs, and suggest "
                "concrete ways to improve it.\n\n"
                "Structure your critique as follows:\n"
                "1. What works in your solution\n"
                "2. What doesn't work or could be improved\n"
                "3. Specific improvement ideas\n\n"
                "Then provide an improved version of your composition that addresses these issues."
            )
        else:
            return (
                "Based on the execution results, please critique your solution. "
                "Identify what worked well, what didn't work, and how you could improve it. "
                "Then provide an improved version of your composition that addresses these issues."
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
        
        Args:
            text: The model's response text
            
        Returns:
            The extracted composition code, or None if not found
        """
        # Look for composition inside tags
        composition_pattern = r"<composition>(.*?)</composition>"
        composition_match = re.search(composition_pattern, text, re.DOTALL)
        
        if composition_match:
            return composition_match.group(1).strip()
        
        # Fallback to looking for solve function
        function_pattern = r"def\s+solve\s*\(.*?\).*?:"
        match = re.search(function_pattern, text, re.DOTALL)
        
        if not match:
            return None
        
        # Get the starting position of the function definition
        start_pos = match.start()
        
        # Extract the function body (handling indentation)
        lines = text[start_pos:].split('\n')
        function_lines = [lines[0]]
        
        # Find the indentation of the first line of the function body
        for i in range(1, len(lines)):
            if re.match(r'^\s*$', lines[i]):  # Skip empty lines
                function_lines.append(lines[i])
                continue
                
            if i == 1:  # First non-empty line after function definition
                indent_match = re.match(r'^(\s+)', lines[i])
                if not indent_match:  # No indentation, not a valid function
                    return None
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
        Get the next problem to solve.
        
        Returns:
            An Item object containing the problem, or None if no more problems
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
        
        # Initialize trajectory state
        trajectory = TrajectoryState(problem, conversation_id)
        trajectory.add_message("system", system_prompt)
        trajectory.add_message("user", user_prompt)
        
        # Store the active trajectory
        self.active_trajectories[conversation_id] = trajectory
        
        # Return the conversation history as the item
        return (conversation_id, problem.problem_id)
    
    async def collect_trajectory(self, item: Item) -> Tuple[Dict, List]:
        """
        Collect a single trajectory for a problem, handling the full improvement cycle.
        
        This method implements the core recursive improvement loop:
        1. Get initial solution
        2. Execute and verify
        3. Request critique
        4. Get improved solution
        5. Repeat for multiple iterations
        
        Args:
            item: Tuple containing conversation_id and problem_id
            
        Returns:
            Tuple containing the scored data and any backlog items
        """
        conversation_id, problem_id = item
        
        # Retrieve the trajectory state
        trajectory = self.active_trajectories.get(conversation_id)
        if not trajectory:
            logger.error(f"No active trajectory found for conversation {conversation_id}")
            return None, []
        
        # Get the problem definition
        problem = trajectory.problem
        
        # Process the trajectory based on its current state
        if trajectory.current_state == ConversationState.PROBLEM_PRESENTED:
            # Get initial solution from the model
            messages = trajectory.message_history.copy()
            response = await self._get_model_response(messages)
            
            # Extract the composition from the response
            composition = self._extract_composition(response)
            if not composition:
                logger.warning(f"No valid composition found in response for {conversation_id}")
                # Fall back to using the entire response
                composition = response
            
            # Add the assistant's response to the conversation
            trajectory.add_message("assistant", response)
            
            # Execute the composition
            execution_result = self.execution_engine.execute_from_text(
                response, problem.input_data
            )
            trajectory.add_execution_result(execution_result)
            
            # Generate execution result prompt
            execution_prompt = self._generate_execution_result_prompt(
                execution_result, problem.expected_outputs
            )
            trajectory.add_message("user", execution_prompt)
            
            # Calculate reward for this solution
            reward = self._calculate_reward(execution_result, problem.expected_outputs)
            trajectory.add_reward(reward)
            
            # Update state to execution result
            trajectory.update_state(ConversationState.EXECUTION_RESULT)
            
            # Request critique
            critique_prompt = self._generate_critique_prompt()
            trajectory.add_message("user", critique_prompt)
            
            # Update state to critique requested
            trajectory.update_state(ConversationState.CRITIQUE)
            
            # Add to backlog to continue the trajectory
            return None, [item]
            
        elif trajectory.current_state == ConversationState.CRITIQUE:
            # Get critique from the model
            messages = trajectory.message_history.copy()
            response = await self._get_model_response(messages)
            
            # Add the assistant's critique to the conversation
            trajectory.add_message("assistant", response)
            
            # Extract critique and improved composition
            critique = self._extract_critique(response)
            composition = self._extract_composition(response)
            
            if composition:
                # We already have an improved composition in the critique response
                # Execute the improved composition
                execution_result = self.execution_engine.execute_from_text(
                    response, problem.input_data
                )
                trajectory.add_execution_result(execution_result)
                
                # Generate execution result prompt
                execution_prompt = self._generate_execution_result_prompt(
                    execution_result, problem.expected_outputs
                )
                trajectory.add_message("user", execution_prompt)
                
                # Calculate reward for this solution
                reward = self._calculate_reward(execution_result, problem.expected_outputs)
                trajectory.add_reward(reward)
                
                # Update state to improved solution
                trajectory.update_state(ConversationState.IMPROVED_SOLUTION)
                
                # Check if we should continue with more iterations
                if (
                    trajectory.current_iteration < self.config.max_iterations and
                    (trajectory.rewards[-1] < 1.0 or len(trajectory.rewards) < 2)
                ):
                    # Request another improvement
                    improvement_prompt = self._generate_improvement_prompt(
                        trajectory.current_iteration,
                        self.config.max_iterations
                    )
                    trajectory.add_message("user", improvement_prompt)
                    
                    # Update state to critique for next iteration
                    trajectory.update_state(ConversationState.CRITIQUE)
                    
                    # Add to backlog to continue the trajectory
                    return None, [item]
                else:
                    # Reached maximum iterations or achieved perfect score
                    trajectory.update_state(ConversationState.FINAL_RESULT)
                    trajectory.metadata["converged"] = trajectory.rewards[-1] >= 1.0
                    
                    # Calculate improvement metrics
                    if len(trajectory.rewards) > 1:
                        improvement = trajectory.rewards[-1] - trajectory.rewards[0]
                        self.improvement_rate_buffer.append(improvement)
                    
                    # Calculate convergence speed
                    if trajectory.rewards[-1] >= 1.0:
                        self.convergence_speed_buffer.append(trajectory.current_iteration)
                    
                    # Calculate final success rate
                    self.success_rate_buffer.append(1.0 if trajectory.rewards[-1] >= 1.0 else 0.0)
                    
                    # Create ScoredDataItem from the trajectory
                    return self._create_scored_data(trajectory), []
            else:
                # No improved composition in the critique, request one explicitly
                improvement_prompt = self._generate_improvement_prompt(
                    trajectory.current_iteration,
                    self.config.max_iterations
                )
                trajectory.add_message("user", improvement_prompt)
                
                # Keep the state as critique to get an improved solution
                return None, [item]
                
        elif trajectory.current_state == ConversationState.IMPROVED_SOLUTION:
            # We're waiting for an improved solution after critique
            # Similar logic to critique state, but explicitly expecting a composition
            
            messages = trajectory.message_history.copy()
            response = await self._get_model_response(messages)
            
            # Add the assistant's improved solution to the conversation
            trajectory.add_message("assistant", response)
            
            # Extract the improved composition
            composition = self._extract_composition(response)
            if not composition:
                logger.warning(f"No valid improved composition found in response for {conversation_id}")
                # Fall back to using the entire response
                composition = response
            
            # Execute the improved composition
            execution_result = self.execution_engine.execute_from_text(
                response, problem.input_data
            )
            trajectory.add_execution_result(execution_result)
            
            # Generate execution result prompt
            execution_prompt = self._generate_execution_result_prompt(
                execution_result, problem.expected_outputs
            )
            trajectory.add_message("user", execution_prompt)
            
            # Calculate reward for this solution
            reward = self._calculate_reward(execution_result, problem.expected_outputs)
            trajectory.add_reward(reward)
            
            # Update state to improved solution (increments iteration counter)
            trajectory.update_state(ConversationState.IMPROVED_SOLUTION)
            
            # Check if we should continue with more iterations
            if (
                trajectory.current_iteration < self.config.max_iterations and
                (trajectory.rewards[-1] < 1.0 or len(trajectory.rewards) < 2) and
                (len(trajectory.rewards) < 2 or trajectory.rewards[-1] - trajectory.rewards[-2] >= self.config.improvement_threshold)
            ):
                # Request another critique
                critique_prompt = self._generate_critique_prompt()
                trajectory.add_message("user", critique_prompt)
                
                # Update state to critique for next iteration
                trajectory.update_state(ConversationState.CRITIQUE)
                
                # Add to backlog to continue the trajectory
                return None, [item]
            else:
                # Reached maximum iterations, achieved perfect score, or no improvement
                trajectory.update_state(ConversationState.FINAL_RESULT)
                trajectory.metadata["converged"] = trajectory.rewards[-1] >= 1.0
                
                # Calculate improvement metrics
                if len(trajectory.rewards) > 1:
                    improvement = trajectory.rewards[-1] - trajectory.rewards[0]
                    self.improvement_rate_buffer.append(improvement)
                
                # Calculate convergence speed
                if trajectory.rewards[-1] >= 1.0:
                    self.convergence_speed_buffer.append(trajectory.current_iteration)
                
                # Calculate final success rate
                self.success_rate_buffer.append(1.0 if trajectory.rewards[-1] >= 1.0 else 0.0)
                
                # Create ScoredDataItem from the trajectory
                return self._create_scored_data(trajectory), []
        
        # If we reach here, something went wrong or the state is unexpected
        logger.warning(f"Unexpected state {trajectory.current_state} for conversation {conversation_id}")
        return None, []
    
    async def _get_model_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Get a response from the language model using the server.
        
        Args:
            messages: The conversation history as a list of message dicts
            
        Returns:
            The model's response as a string
        """
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        
        try:
            completion = await self.server.completion(
                prompt=prompt,
                max_tokens=self.config.max_token_length // 2,
                temperature=0.7,
                stop=None
            )
            
            return completion.choices[0].text
        except Exception as e:
            logger.error(f"Error getting model response: {e}")
            return f"Error: Unable to generate response due to {str(e)}"
    
    def _calculate_reward(self, execution_result: ExecutionResult, expected_output: Any) -> float:
        """
        Calculate the reward for an execution result.
        
        Args:
            execution_result: The result of executing a tool composition
            expected_output: The expected output for comparison
            
        Returns:
            The calculated reward value
        """
        if self.config.verification_type == "binary":
            # Simple binary reward: 1.0 for correct, 0.0 for incorrect
            if not execution_result.success:
                return 0.0
                
            reward = self.reward_fn.compute(
                [execution_result.to_dict()], 
                expected_output=expected_output
            )[0]
            
            return reward
        else:
            # More graduated reward with partial credit
            if not execution_result.success:
                return 0.0
                
            # Start with binary correctness check
            correct = (execution_result.result == expected_output)
            
            # Base reward for correct result
            if correct:
                reward = 1.0
            else:
                # Allow partial credit if enabled
                if self.config.allow_partial_credit:
                    # Try to calculate similarity for partial credit
                    try:
                        if isinstance(expected_output, (list, tuple)) and isinstance(execution_result.result, (list, tuple)):
                            # List similarity
                            similarity = len(set(execution_result.result) & set(expected_output)) / max(1, len(set(expected_output)))
                            reward = 0.5 * similarity
                        elif isinstance(expected_output, str) and isinstance(execution_result.result, str):
                            # String similarity (rough approximation)
                            if len(expected_output) > 0:
                                # Very basic similarity: % of characters that match
                                matching_chars = sum(1 for a, b in zip(expected_output, execution_result.result) if a == b)
                                similarity = matching_chars / len(expected_output)
                                reward = 0.5 * similarity
                            else:
                                reward = 0.0
                        else:
                            # No partial credit for other types
                            reward = 0.0
                    except Exception:
                        reward = 0.0
                else:
                    reward = 0.0
            
            return reward
    
    def _create_scored_data(self, trajectory: TrajectoryState) -> Dict:
        """
        Create a ScoredDataItem from a completed trajectory.
        
        Args:
            trajectory: The completed trajectory
            
        Returns:
            ScoredDataItem with tokens, masks, scores, and messages
        """
        # Use the entire conversation history
        messages = trajectory.message_history.copy()
        
        # Tokenize the conversation
        tokenized = tokenize_for_trainer(self.tokenizer, messages)
        
        # Ensure the trajectory is within token limits
        if len(tokenized["tokens"]) > self.config.max_trajectory_tokens:
            result = ensure_trajectory_token_limit(
                tokenized["tokens"],
                tokenized["masks"],
                self.config.max_trajectory_tokens
            )
            tokens = result["tokens"]
            masks = result["masks"]
        else:
            tokens = tokenized["tokens"]
            masks = tokenized["masks"]
        
        # Use final reward as the score
        score = trajectory.rewards[-1] if trajectory.rewards else 0.0
        
        # Create the scored data item
        scored_data = {
            "tokens": tokens,
            "masks": masks,
            "scores": score,
            "messages": messages if self.config.include_messages else None,
            "overrides": {
                "problem_id": trajectory.problem.problem_id,
                "iterations": trajectory.current_iteration,
                "rewards": trajectory.rewards,
                "converged": trajectory.metadata.get("converged", False)
            }
        }
        
        return scored_data
    
    async def evaluate(self, *args, **kwargs):
        """
        Evaluate the environment on the evaluation problem set.
        
        This runs through the evaluation problems and calculates success metrics.
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
        
        # Add evaluation metrics
        for key, value in self.eval_metrics:
            wandb_metrics[key] = value
        self.eval_metrics = []
        
        # Call parent method to log metrics
        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    RecursiveToolImprovementEnv.cli()