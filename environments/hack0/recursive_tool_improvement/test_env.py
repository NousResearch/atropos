#!/usr/bin/env python3
"""
Test script for the Recursive Tool Improvement Environment.

This script tests the environment by creating a simple trajectory and verifying
that the ScoredDataItem is correctly formatted.
"""

import asyncio
import logging
import os
import sys
from typing import Dict, Any

# Add the parent directory to the path so we can import the environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from environments.hack0.recursive_tool_improvement.recursive_tool_improvement import (
    RecursiveToolImprovementEnv,
    RecursiveToolImprovementConfig,
    TrajectoryState,
    ProblemDefinition,
    ConversationState
)
from atroposlib.envs.base import APIServerConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_create_scored_data():
    """Test the _create_scored_data method."""
    # Create a simple environment
    config = RecursiveToolImprovementConfig(
        max_iterations=2,
        tool_set="basic",
        timeout=5,
        verification_type="binary",
        reasoning_mode="deduction",
        use_thinking_tags=True,
        eval_ratio=0.1,
        group_size=1,
        max_token_length=2048,
        include_messages=True,
        wandb_name="test_env"
    )
    
    server_configs = [
        APIServerConfig(
            model_name="gpt-4o",
            base_url=None,
            api_key=None,
            num_max_requests_at_once=16,
            num_requests_for_eval=32,
        )
    ]
    
    env = RecursiveToolImprovementEnv(config, server_configs, slurm=False, testing=True)
    
    # Create a simple problem
    problem = ProblemDefinition(
        problem_id="test-problem",
        title="Test Problem",
        description="A simple test problem",
        input_data="test input",
        expected_outputs="test output",
        difficulty="easy",
        domain="test",
        available_tools=[],
        reasoning_mode="deduction"
    )
    
    # Create a trajectory
    trajectory = TrajectoryState(problem, "test-conversation")
    trajectory.add_message("system", "System prompt")
    trajectory.add_message("user", "User prompt")
    trajectory.add_message("assistant", "Assistant response")
    trajectory.add_reward(1.0)
    trajectory.update_state(ConversationState.FINAL_RESULT)
    
    # Create a scored data item
    scored_data = env._create_scored_data(trajectory)
    
    # Verify that the scored data is correctly formatted
    logger.info(f"Scored data: {scored_data}")
    
    # Check that the required keys are present
    required_keys = ["tokens", "masks", "scores"]
    for key in required_keys:
        if key not in scored_data:
            logger.error(f"Missing required key '{key}' in scored_data")
            return False
    
    logger.info("All required keys are present in scored_data")
    return True

async def test_collect_trajectory():
    """Test the collect_trajectory method."""
    # Create a simple environment
    config = RecursiveToolImprovementConfig(
        max_iterations=2,
        tool_set="basic",
        timeout=5,
        verification_type="binary",
        reasoning_mode="deduction",
        use_thinking_tags=True,
        eval_ratio=0.1,
        group_size=1,
        max_token_length=2048,
        include_messages=True,
        wandb_name="test_env"
    )
    
    server_configs = [
        APIServerConfig(
            model_name="gpt-4o",
            base_url=None,
            api_key=None,
            num_max_requests_at_once=16,
            num_requests_for_eval=32,
        )
    ]
    
    env = RecursiveToolImprovementEnv(config, server_configs, slurm=False, testing=True)
    
    # Create a simple problem
    problem = ProblemDefinition(
        problem_id="test-problem",
        title="Test Problem",
        description="A simple test problem",
        input_data="test input",
        expected_outputs="test output",
        difficulty="easy",
        domain="test",
        available_tools=[],
        reasoning_mode="deduction"
    )
    
    # Create a trajectory
    conversation_id = "test-conversation"
    trajectory = TrajectoryState(problem, conversation_id)
    trajectory.add_message("system", "System prompt")
    trajectory.add_message("user", "User prompt")
    trajectory.update_state(ConversationState.FINAL_RESULT)
    trajectory.add_reward(1.0)
    
    # Add the trajectory to the environment
    env.active_trajectories[conversation_id] = trajectory
    
    # Call collect_trajectory
    item = (conversation_id, problem.problem_id)
    result, backlog = await env.collect_trajectory(item)
    
    # Verify that the result is correctly formatted
    logger.info(f"Result: {result}")
    logger.info(f"Backlog: {backlog}")
    
    # Check that the result is a dictionary or None
    if result is not None and not isinstance(result, dict):
        logger.error(f"Result is not a dictionary or None: {type(result)}")
        return False
    
    # Check that the backlog is a list
    if not isinstance(backlog, list):
        logger.error(f"Backlog is not a list: {type(backlog)}")
        return False
    
    logger.info("collect_trajectory returned the correct types")
    return True

async def main():
    """Run the tests."""
    logger.info("Testing _create_scored_data...")
    create_scored_data_result = await test_create_scored_data()
    
    logger.info("Testing collect_trajectory...")
    collect_trajectory_result = await test_collect_trajectory()
    
    if create_scored_data_result and collect_trajectory_result:
        logger.info("All tests passed!")
        return 0
    else:
        logger.error("Some tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
