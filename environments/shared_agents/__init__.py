"""
Shared agents for Atropos environments.

These agents can be used across different game environments like TextWorld and Diplomacy.
"""

from .atropos_agent import AtroposAgent, AtroposAgentConfig
from .atropos_memory_manager import (
    AtroposMemoryManager,
    MEMORY_SYSTEM_PREREQUISITES_AVAILABLE,
    SentenceEmbeddingHelper
)
from .utils.memory_parser import extract_memory_block, validate_memory_content
from .types import AtroposAgentAction, AtroposAgentTurn, AtroposAgentActionLog

__all__ = [
    "AtroposAgent",
    "AtroposAgentConfig", 
    "AtroposMemoryManager",
    "MEMORY_SYSTEM_PREREQUISITES_AVAILABLE",
    "SentenceEmbeddingHelper",
    "extract_memory_block",
    "validate_memory_content",
    "AtroposAgentAction",
    "AtroposAgentTurn", 
    "AtroposAgentActionLog",
]