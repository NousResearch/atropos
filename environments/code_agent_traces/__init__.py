"""
Code Agent Traces Environment.

Pipeline for generating agent traces for code generation tasks
using Ollama Cloud (DeepSeek V3.2) with logprobs support.

Local code execution is provided for testing when Modal is not available.
"""

from .agent_trace_env import AgentTraceConfig, AgentTraceEnv, OllamaAgentTraceEnv
from .local_executor import LocalCodeExecutor, execute_code_safe, run_test_local

__all__ = [
    "AgentTraceConfig",
    "AgentTraceEnv",
    "OllamaAgentTraceEnv",
    "LocalCodeExecutor",
    "execute_code_safe",
    "run_test_local",
]
