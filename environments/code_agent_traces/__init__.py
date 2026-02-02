"""
Code Agent Traces Environment.

Pipeline for generating structured agent traces with Planning-Action-Reflection
reasoning using Ollama Cloud (DeepSeek V3.2) with logprobs support.

Trace Structure:
1. PLANNING - Problem analysis and approach planning
2. ACTION - Code generation
3. REFLECTION - Result analysis and iteration

Local code execution is provided when Modal is not available.
"""

from .agent_trace_env import AgentTraceConfig, AgentTraceEnv, OllamaAgentTraceEnv
from .local_executor import LocalCodeExecutor, execute_code_safe, run_test_local
from .structured_agent_env import StructuredAgentConfig, StructuredAgentEnv

__all__ = [
    # Basic trace environment
    "AgentTraceConfig",
    "AgentTraceEnv",
    "OllamaAgentTraceEnv",
    # Structured reasoning environment
    "StructuredAgentConfig",
    "StructuredAgentEnv",
    # Local execution
    "LocalCodeExecutor",
    "execute_code_safe",
    "run_test_local",
]
