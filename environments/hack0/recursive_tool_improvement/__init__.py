"""
Recursive Tool Improvement Environment

This environment trains language models to create, improve, and recursively
refine tool compositions to solve complex problems.
"""

from .recursive_tool_improvement import RecursiveToolImprovementEnv, RecursiveToolImprovementConfig
from .tool_registry import ToolRegistry, Tool, default_registry
from .execution_engine import ExecutionEngine, ExecutionResult
from .reward_functions.binary_verification import BinaryVerificationReward, ImprovementReward