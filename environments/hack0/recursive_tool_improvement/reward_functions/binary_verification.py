"""
Binary verification reward function inspired by Tool-N1.

This module implements a simple binary reward function (0 or 1) based on 
functional correctness, following the principles of the Tool-N1 research.
It also supports semantic equivalence validation for outputs.
"""

import json
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

from atroposlib.envs.reward_fns.reward_function import RewardFunction


class BinaryVerificationReward(RewardFunction):
    """
    Binary verification reward that gives a score of 1.0 for correct tool compositions
    and 0.0 for incorrect compositions.
    
    This reward is based on the Tool-N1 research showing that binary rewards provide
    a cleaner learning signal for tool learning compared to more complex reward schemes.
    """
    
    def __init__(
        self,
        weight: float = 1.0,
        name: Optional[str] = "binary_verification",
        semantic_equivalence: bool = True,
        equality_fn: Optional[Callable[[Any, Any], bool]] = None,
        **kwargs
    ):
        """
        Initialize the binary verification reward.
        
        Args:
            weight: Weight of this reward function when combined with others
            name: Optional custom name for this reward function
            semantic_equivalence: Whether to use semantic equivalence checking
            equality_fn: Optional custom function to determine if outputs are equal
            **kwargs: Additional configuration parameters
        """
        super().__init__(weight, name, **kwargs)
        self.semantic_equivalence = semantic_equivalence
        self.equality_fn = equality_fn
        
    def compute(self, completions: List[Dict], **kwargs) -> List[float]:
        """
        Compute reward scores for the given completions.
        
        Args:
            completions: List of completion results to evaluate
                Each completion should be a dictionary with at least:
                - 'success': Whether execution was successful
                - 'result': The output of the execution
            **kwargs: Additional context, must include:
                - 'expected_output': The expected output for correctness checking
        
        Returns:
            List of reward scores (0.0 or 1.0 for each completion)
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
    
    def _are_outputs_equal(self, actual: Any, expected: Any) -> bool:
        """
        Determine if the actual output matches the expected output.
        
        Args:
            actual: The actual output from execution
            expected: The expected correct output
            
        Returns:
            True if outputs are considered equal, False otherwise
        """
        # Use custom equality function if provided
        if self.equality_fn is not None:
            try:
                return self.equality_fn(actual, expected)
            except Exception:
                # If custom comparison fails, fall back to standard comparison
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
                return self._compare_strings(actual, expected)
            
            # Compare lists/tuples with order-insensitive comparison if needed
            if ((isinstance(actual, list) and isinstance(expected, list)) or
                (isinstance(actual, tuple) and isinstance(expected, tuple))):
                return self._compare_sequences(actual, expected)
            
            # Compare dicts with semantic equivalence
            if isinstance(actual, dict) and isinstance(expected, dict):
                return self._compare_dicts(actual, expected)
            
            # Try JSON comparison for complex objects
            try:
                actual_json = json.dumps(actual, sort_keys=True)
                expected_json = json.dumps(expected, sort_keys=True)
                return actual_json == expected_json
            except (TypeError, ValueError):
                # If JSON serialization fails, they're not semantically equivalent
                pass
                
        # If no equivalence found or semantic equivalence is disabled
        return False
    
    def _compare_strings(self, actual: str, expected: str) -> bool:
        """Compare strings with normalization for semantic equivalence"""
        # Normalize for comparison (removing extra whitespace, case insensitive, etc.)
        actual_norm = ' '.join(actual.strip().lower().split())
        expected_norm = ' '.join(expected.strip().lower().split())
        
        return actual_norm == expected_norm
    
    def _compare_sequences(self, actual: Union[List, Tuple], expected: Union[List, Tuple]) -> bool:
        """Compare sequences (lists/tuples) for semantic equivalence"""
        # Same length check first for efficiency
        if len(actual) != len(expected):
            return False
            
        # Try order-sensitive comparison first (faster)
        all_equal = True
        for a, e in zip(actual, expected):
            if not self._are_outputs_equal(a, e):
                all_equal = False
                break
        
        if all_equal:
            return True
            
        # If order-sensitive comparison fails, try order-insensitive
        # (only for simple types where order might not matter)
        try:
            # Check if all elements are of comparable simple types
            simple_types = (str, int, float, bool, type(None))
            if all(isinstance(x, simple_types) for x in actual + list(expected)):
                # Sort and compare (assuming elements are comparable)
                return sorted(actual) == sorted(expected)
        except TypeError:
            # If elements are not comparable, they're not semantically equivalent
            pass
            
        return False
    
    def _compare_dicts(self, actual: Dict, expected: Dict) -> bool:
        """Compare dictionaries for semantic equivalence"""
        # Same keys check
        if set(actual.keys()) != set(expected.keys()):
            return False
            
        # Check each key-value pair
        for key in actual:
            if not self._are_outputs_equal(actual[key], expected[key]):
                return False
                
        return True


class ImprovementReward(RewardFunction):
    """
    Measures improvement between iterations of tool compositions.
    
    This reward function compares the current solution with a previous solution
    and rewards based on improvement metrics.
    """
    
    def __init__(
        self,
        weight: float = 1.0,
        name: Optional[str] = "improvement",
        **kwargs
    ):
        """
        Initialize the improvement reward function.
        
        Args:
            weight: Weight of this reward function when combined with others
            name: Optional custom name for this reward function
            **kwargs: Additional configuration parameters
        """
        super().__init__(weight, name, **kwargs)
    
    def compute(self, completions: List[Dict], **kwargs) -> List[float]:
        """
        Compute improvement-based reward scores for the given completions.
        
        Args:
            completions: List of completion results to evaluate
            **kwargs: Additional context, must include:
                - 'previous_result': The result from the previous iteration
                - 'expected_output': The expected output for correctness checking
        
        Returns:
            List of improvement reward scores for each completion
        """
        if "previous_result" not in kwargs:
            raise ValueError("Previous result must be provided to compute improvement reward")
        
        if "expected_output" not in kwargs:
            raise ValueError("Expected output must be provided to compute improvement reward")
        
        previous_result = kwargs["previous_result"]
        expected_output = kwargs["expected_output"]
        rewards = []
        
        for completion in completions:
            # Base reward on execution success (higher weight)
            if not completion.get("success", False):
                rewards.append(0.0)
                continue
            
            # Get current result
            current_result = completion.get("result")
            
            # Track various improvement metrics
            metrics = {
                "correctness_improved": 0.0,
                "efficiency_improved": 0.0,
                "error_reduced": 0.0
            }
            
            # Calculate efficiency improvement (if both executions succeeded)
            if (previous_result.get("success", False) and 
                completion.get("success", False)):
                
                # Compare execution time
                prev_time = previous_result.get("execution_time", float('inf'))
                curr_time = completion.get("execution_time", float('inf'))
                
                if curr_time < prev_time:
                    # Reward proportional to time improvement, up to 0.5
                    time_factor = min(0.5, (prev_time - curr_time) / prev_time)
                    metrics["efficiency_improved"] = time_factor
                
                # Compare number of tool calls
                prev_calls = len(previous_result.get("tool_calls", []))
                curr_calls = len(completion.get("tool_calls", []))
                
                if curr_calls < prev_calls:
                    # Reward proportional to reduction in tool calls, up to 0.3
                    calls_factor = min(0.3, (prev_calls - curr_calls) / max(1, prev_calls))
                    metrics["efficiency_improved"] += calls_factor
            
            # Calculate correctness improvement
            prev_correct = (previous_result.get("result") == expected_output)
            curr_correct = (current_result == expected_output)
            
            if curr_correct and not prev_correct:
                # Major improvement: from incorrect to correct
                metrics["correctness_improved"] = 1.0
            elif curr_correct and prev_correct:
                # Already correct, but might have improved in other ways
                metrics["correctness_improved"] = 0.5
            elif not curr_correct and not prev_correct:
                # Both incorrect, but check if closer to correct answer
                # This would require domain-specific comparison logic
                metrics["correctness_improved"] = 0.0
            
            # Calculate error reduction
            if not previous_result.get("success", False) and completion.get("success", False):
                # Major improvement: from execution failure to success
                metrics["error_reduced"] = 1.0
            
            # Combine metrics into final reward
            reward = (
                0.6 * metrics["correctness_improved"] +
                0.3 * metrics["efficiency_improved"] +
                0.1 * metrics["error_reduced"]
            )
            
            rewards.append(reward)
        
        return rewards