"""
DynastAI - Atropos-Compatible Medieval Kingdom Management Game

This package implements the DynastAI game environment:
- A medieval kingdom management card game
- Atropos-compatible Python RL environment
- FastAPI REST API endpoints
- HTML/CSS/JS web frontend
"""

__version__ = "1.0.0"

# Import main classes for easier access
from .dynastai_env import DynastAIEnv, DynastAIEnvConfig
from .game_logic import GameState, generate_card, apply_choice_effects