"""
Unit tests for the binary verification reward function.
"""

import unittest
import sys
import os
import inspect
import inspect
import copy

# Import the mock RewardFunction
from tests.mock_reward_function import RewardFunction

# Create a simplified version of the binary verification reward function for testing
class BinaryVerificationReward(RewardFunction):
    """
    Mock binary verification reward for testing.
    """
    
    def __init__(self, weight=1.0, name="binary_verification", semantic_equivalence=True, 
                 equality_fn=None, **kwargs):
        super().__init__(weight, name, **kwargs)
        self.semantic_equivalence = semantic_equivalence
        self.equality_fn = equality_fn
    
    def compute(self, completions, **kwargs):
        """
        Compute reward scores for the given completions.
        """
        if "expected_output" not in kwargs:
            raise ValueError("Expected output must be provided to compute binary verification reward")
        
        expected_output = kwargs["expected_output"]
        rewards = []
        
        for completion in completions:
            # If execution failed, score is 0.0
            if not completion.get("success", False):
                rewards.append(0.0)
                continue
            
            actual_output = completion.get("result")
            
            # Check if outputs match
            if self._are_outputs_equal(actual_output, expected_output):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        
        return rewards
    
    def _are_outputs_equal(self, actual, expected):
        """
        Determine if the actual output matches the expected output.
        """
        if self.equality_fn is not None:
            try:
                return self.equality_fn(actual, expected)
            except Exception:
                pass
                
        # Check for exact equality first
        if actual == expected:
            return True
        
        # If semantic equivalence is enabled, perform additional checks
        if self.semantic_equivalence:
            # Handle None specially
            if actual is None or expected is None:
                return actual is expected
            
            # Compare strings with normalization
            if isinstance(actual, str) and isinstance(expected, str):
                actual_norm = ' '.join(actual.strip().lower().split())
                expected_norm = ' '.join(expected.strip().lower().split())
                return actual_norm == expected_norm
            
            # Compare lists/tuples with order-insensitive comparison
            if ((isinstance(actual, list) and isinstance(expected, list)) or
                (isinstance(actual, tuple) and isinstance(expected, tuple))):
                if len(actual) != len(expected):
                    return False
                
                # Try order-insensitive comparison for simple types
                try:
                    return sorted(actual) == sorted(expected)
                except TypeError:
                    pass
        
        # If no equivalence found
        return False


class ImprovementReward(RewardFunction):
    """
    Mock improvement reward for testing.
    """
    
    def __init__(self, weight=1.0, name="improvement", **kwargs):
        super().__init__(weight, name, **kwargs)
    
    def compute(self, completions, **kwargs):
        """
        Compute improvement-based reward scores for the given completions.
        """
        if "previous_result" not in kwargs:
            raise ValueError("Previous result must be provided to compute improvement reward")
        
        if "expected_output" not in kwargs:
            raise ValueError("Expected output must be provided to compute improvement reward")
        
        previous_result = kwargs["previous_result"]
        expected_output = kwargs["expected_output"]
        rewards = []
        
        for completion in completions:
            # Base reward
            reward = 0.0
            
            # If current execution failed, low reward
            if not completion.get("success", False):
                rewards.append(reward)
                continue
            
            # Calculate correctness improvement
            prev_correct = (previous_result.get("result") == expected_output)
            curr_correct = (completion.get("result") == expected_output)
            
            if curr_correct and not prev_correct:
                # Major improvement: from incorrect to correct
                reward += 0.6
            elif curr_correct and prev_correct:
                # Already correct
                reward += 0.3
            
            # Calculate efficiency improvement
            if previous_result.get("success", False) and completion.get("success", False):
                # Time improvement
                prev_time = previous_result.get("execution_time", float('inf'))
                curr_time = completion.get("execution_time", float('inf'))
                if curr_time < prev_time:
                    reward += 0.2
                
                # Tool call reduction
                prev_calls = len(previous_result.get("tool_calls", []))
                curr_calls = len(completion.get("tool_calls", []))
                if curr_calls < prev_calls:
                    reward += 0.1
            
            rewards.append(reward)
        
        return rewards


class TestBinaryVerificationReward(unittest.TestCase):
    """Tests for the BinaryVerificationReward class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.reward_fn = BinaryVerificationReward(weight=1.0)
    
    def test_exact_match_reward(self):
        """Test reward calculation for exact match."""
        # Create a sample completion with correct output
        completion = {
            "success": True,
            "result": "test output",
            "execution_time": 0.5,
            "tool_calls": []
        }
        
        rewards = self.reward_fn.compute([completion], expected_output="test output")
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_no_match_reward(self):
        """Test reward calculation for no match."""
        # Create a sample completion with incorrect output
        completion = {
            "success": True,
            "result": "wrong output",
            "execution_time": 0.5,
            "tool_calls": []
        }
        
        rewards = self.reward_fn.compute([completion], expected_output="test output")
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)
    
    def test_execution_failure_reward(self):
        """Test reward calculation for execution failure."""
        # Create a sample completion with execution failure
        completion = {
            "success": False,
            "result": None,
            "error": "Runtime error",
            "execution_time": 0.5,
            "tool_calls": []
        }
        
        rewards = self.reward_fn.compute([completion], expected_output="test output")
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)
    
    def test_semantic_equivalence(self):
        """Test semantic equivalence for reward calculation."""
        # Create reward function with semantic equivalence enabled
        reward_fn = BinaryVerificationReward(semantic_equivalence=True)
        
        # Test list equivalence (same elements, different order)
        completion1 = {
            "success": True,
            "result": [1, 2, 3],
            "execution_time": 0.5,
            "tool_calls": []
        }
        
        rewards = reward_fn.compute([completion1], expected_output=[3, 1, 2])
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
        
        # Test string normalization (case, whitespace)
        completion2 = {
            "success": True,
            "result": "Hello  World",
            "execution_time": 0.5,
            "tool_calls": []
        }
        
        rewards = reward_fn.compute([completion2], expected_output="hello world")
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_multiple_completions(self):
        """Test reward calculation for multiple completions."""
        # Create multiple completions
        completions = [
            {
                "success": True,
                "result": "test output",
                "execution_time": 0.5,
                "tool_calls": []
            },
            {
                "success": True,
                "result": "wrong output",
                "execution_time": 0.5,
                "tool_calls": []
            },
            {
                "success": False,
                "result": None,
                "error": "Runtime error",
                "execution_time": 0.5,
                "tool_calls": []
            }
        ]
        
        rewards = self.reward_fn.compute(completions, expected_output="test output")
        
        self.assertEqual(len(rewards), 3)
        self.assertEqual(rewards[0], 1.0)  # Correct
        self.assertEqual(rewards[1], 0.0)  # Incorrect
        self.assertEqual(rewards[2], 0.0)  # Failed
    
    def test_custom_weight(self):
        """Test reward calculation with custom weight."""
        # Create reward function with custom weight
        reward_fn = BinaryVerificationReward(weight=0.5)
        
        # Create a sample completion with correct output
        completion = {
            "success": True,
            "result": "test output",
            "execution_time": 0.5,
            "tool_calls": []
        }
        
        # Use __call__ to apply weight
        rewards = reward_fn([completion], expected_output="test output")
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.5)  # Weight of 0.5 applied to 1.0 reward


class TestImprovementReward(unittest.TestCase):
    """Tests for the ImprovementReward class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.reward_fn = ImprovementReward(weight=1.0)
    
    def test_correctness_improvement(self):
        """Test reward for improving from incorrect to correct."""
        # Create a previous incorrect result
        previous_result = {
            "success": True,
            "result": "wrong output",
            "execution_time": 1.0,
            "tool_calls": [{"tool": "tool1"}, {"tool": "tool2"}]
        }
        
        # Create a current correct result
        current_completion = {
            "success": True,
            "result": "test output",
            "execution_time": 0.5,
            "tool_calls": [{"tool": "tool1"}]
        }
        
        rewards = self.reward_fn.compute(
            [current_completion], 
            previous_result=previous_result,
            expected_output="test output"
        )
        
        self.assertEqual(len(rewards), 1)
        self.assertTrue(rewards[0] > 0.5)  # Should get a high reward for correctness improvement
    
    def test_efficiency_improvement(self):
        """Test reward for improving efficiency while already correct."""
        # Create a previous correct but inefficient result
        previous_result = {
            "success": True,
            "result": "test output",
            "execution_time": 1.0,
            "tool_calls": [{"tool": "tool1"}, {"tool": "tool2"}, {"tool": "tool3"}]
        }
        
        # Create a current correct and more efficient result
        current_completion = {
            "success": True,
            "result": "test output",
            "execution_time": 0.5,
            "tool_calls": [{"tool": "tool1"}]
        }
        
        rewards = self.reward_fn.compute(
            [current_completion], 
            previous_result=previous_result,
            expected_output="test output"
        )
        
        self.assertEqual(len(rewards), 1)
        self.assertTrue(0 < rewards[0] < 1.0)  # Should get a moderate reward for efficiency improvement
    
    def test_no_improvement(self):
        """Test reward for no improvement."""
        # Create a previous incorrect result
        previous_result = {
            "success": True,
            "result": "wrong output",
            "execution_time": 0.5,
            "tool_calls": [{"tool": "tool1"}]
        }
        
        # Create a current incorrect result (still wrong)
        current_completion = {
            "success": True,
            "result": "another wrong output",
            "execution_time": 0.5,
            "tool_calls": [{"tool": "tool1"}]
        }
        
        rewards = self.reward_fn.compute(
            [current_completion], 
            previous_result=previous_result,
            expected_output="test output"
        )
        
        self.assertEqual(len(rewards), 1)
        self.assertTrue(rewards[0] < 0.5)  # Should get a low reward for no improvement


if __name__ == "__main__":
    unittest.main()