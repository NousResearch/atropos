"""
IFEval Instruction Checking Module

Ported from lighteval (google/IFEval) for standalone use in Atropos.
"""

from .instructions_registry import INSTRUCTION_CONFLICTS, INSTRUCTION_DICT

__all__ = ["INSTRUCTION_DICT", "INSTRUCTION_CONFLICTS"]
