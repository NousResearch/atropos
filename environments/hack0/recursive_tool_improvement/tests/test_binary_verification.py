"""
Unit tests for the binary verification reward function.
"""

import unittest
from ..reward_functions.binary_verification import BinaryVerificationReward, ImprovementReward


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