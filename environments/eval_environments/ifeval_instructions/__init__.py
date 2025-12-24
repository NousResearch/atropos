"""
IFEval Instruction Checking Module

Ported from lighteval (google/IFEval) for standalone use in Atropos.
"""

from .instructions_registry import INSTRUCTION_DICT, INSTRUCTION_CONFLICTS

__all__ = ["INSTRUCTION_DICT", "INSTRUCTION_CONFLICTS"]

