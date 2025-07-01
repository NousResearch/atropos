"""
Diplomacy Training Environment for Atropos RL Framework

This environment integrates the AI_Diplomacy game engine with Atropos
to enable multi-agent reinforcement learning for diplomatic negotiation
and strategic planning.
"""

from .diplomacy_env import DiplomacyEnv, DiplomacyEnvConfig

__all__ = ["DiplomacyEnv", "DiplomacyEnvConfig"]