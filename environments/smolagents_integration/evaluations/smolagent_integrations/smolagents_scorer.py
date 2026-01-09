"""
Scoring utilities for SmolaGents agent trajectories.

This module contains functions for evaluating different aspects of agent
trajectories, including message format compliance, final answer detection,
execution error detection, and efficiency metrics.
"""

import re
from typing import Dict, List, Optional


def check_format_adherence(memory_content: str) -> float:
    """
    Check if memory content follows the required CodeAgent format.

    The expected format includes:
    - "Thought:" section with reasoning
    - "Code:" section with a Python code block
    - Code blocks with triple backticks and "<end_code>" marker

    Args:
        memory_content: The content of a memory step to check

    Returns:
        float: Score between 0.0 and 1.0 indicating format compliance
    """
    thought_pattern = r"Thought: .+"
    code_pattern = r"Code:\s*```py\s*[\s\S]*?```<end_code>"

    # Check if both patterns exist in the content
    has_thought = bool(re.search(thought_pattern, memory_content))
    has_code = bool(re.search(code_pattern, memory_content))

    if has_thought and has_code:
        return 1.0
    elif has_thought or has_code:
        return 0.5
    else:
        return 0.0


def check_final_answer_usage(memory_content: str) -> bool:
    """
    Check if the final_answer tool was used appropriately.

    Args:
        memory_content: The content of a memory step to check

    Returns:
        bool: True if final_answer tool was used, False otherwise
    """
    final_answer_pattern = r"final_answer\(.*?\)"
    return bool(re.search(final_answer_pattern, memory_content))


def extract_execution_errors(agent_memory: List[Dict]) -> List[Dict]:
    """
    Extract execution errors from agent memory.

    Looks for error patterns in the observations field of each memory step.

    Args:
        agent_memory: List of memory steps from agent execution

    Returns:
        List[Dict]: List of errors with step number and error message
    """
    execution_errors = []

    if not agent_memory:
        return execution_errors

    for step in agent_memory:
        # In SmolaGents ActionStep, observations field contains execution output
        if (
            isinstance(step, dict)
            and "observations" in step
            and isinstance(step["observations"], str)
        ):
            observation = step["observations"]

            # Look for error patterns
            error_patterns = [
                r"Error: .*",
                r"Exception: .*",
                r"Traceback \(most recent call last\).*",
                r".*Error: .*",
                r".*Exception: .*",
            ]

            for pattern in error_patterns:
                matches = re.findall(pattern, observation, re.DOTALL)
                if matches:
                    # Record step number and error
                    execution_errors.append(
                        {"step": step.get("step_number", 0), "error": matches[0]}
                    )

    return execution_errors


def calculate_efficiency_score(
    steps_count: int,
    max_steps: int,
    execution_time: float = None,  # Parameter kept for backward compatibility but not used
    execution_times_history: Optional[
        List[float]
    ] = None,  # Parameter kept for backward compatibility but not used
) -> float:
    """
    Calculate efficiency score based on steps used only.
    Execution time is no longer considered in the score calculation.

    Args:
        steps_count: Number of steps taken by the agent
        max_steps: Maximum allowed steps
        execution_time: Not used, kept for backward compatibility
        execution_times_history: Not used, kept for backward compatibility

    Returns:
        float: Efficiency score between 0.0 and 1.0
    """
    # Start with full efficiency score
    efficiency_score = 1.0

    # Penalty for excessive steps (above 75% of max)
    step_penalty = 1.0
    if steps_count > (max_steps * 0.75):
        step_penalty = max(
            0.5, 1.0 - ((steps_count - max_steps * 0.75) / (max_steps * 0.25))
        )
        efficiency_score *= step_penalty

    # Note: Execution time penalty has been removed

    return efficiency_score


def calculate_execution_score(agent_memory: List[Dict]) -> float:
    """
    Calculate execution success score by detecting errors in agent memory.

    Args:
        agent_memory: List of memory steps from agent execution

    Returns:
        float: Execution score between 0.0 and 1.0
    """
    execution_errors = extract_execution_errors(agent_memory)

    if not agent_memory:
        return 0.0

    total_steps = len(agent_memory)
    error_steps = len(execution_errors)

    if total_steps > 0:
        # Penalize proportionally to the number of steps with errors
        execution_score = max(0, 1.0 - (error_steps / total_steps))
    else:
        execution_score = 0.0

    return execution_score
