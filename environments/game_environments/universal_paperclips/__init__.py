"""
Universal Paperclips Atropos Environment

This package provides an Atropos-compatible environment for training RL agents
to play the Universal Paperclips incremental game.

Key components:
- PaperclipsAtroposEnv: Main environment class compatible with Atropos BaseEnv
- PaperclipsEnvConfig: Configuration for the environment
- EpisodeContext: Manages isolated browser contexts for parallel episodes

Usage:
    # CLI usage (serve mode)
    python -m environments.game_environments.universal_paperclips.atropos_env serve

    # Testing
    python -m environments.game_environments.universal_paperclips.test_atropos_env --all
"""

from .config import (
    PAPERCLIPS_SYSTEM_PROMPT,
    PaperclipsEnvConfig,
    get_action_prompt,
)
from .universal_paperclips_env import (
    EpisodeContext,
    GameState,
    PaperclipsAtroposEnv,
)

__all__ = [
    "PaperclipsAtroposEnv",
    "PaperclipsEnvConfig",
    "EpisodeContext",
    "GameState",
    "PAPERCLIPS_SYSTEM_PROMPT",
    "get_action_prompt",
]
