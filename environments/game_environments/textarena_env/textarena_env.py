#!/usr/bin/env python3
"""
TextarenaEnv: Trainer environment for Textarena games with VR-CLI

A trainer environment that wraps Textarena game environments to train LLMs
using best-of-n pattern with function-call style actions. Uses VR-CLI
(Verifiable Rewards via Completion Likelihood Improvement) to score predictions
based on how well the model anticipates action outcomes.
"""

import json
import logging
import math
import os
import random
import re
import tempfile
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import textarena as ta
from pydantic import Field
from textarena.envs.registration import ENV_REGISTRY

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.type_definitions import Message
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer
from atroposlib.utils.tool_call_parser import parse_tool_call
from environments.game_environments.textarena_env.agents.textarena_agent import (
    TextarenaAgent,
    TextarenaAgentConfig,
)
from environments.game_environments.textarena_env.textarena_registry import (
    create_textarena_registry,
)

logger = logging.getLogger(__name__)


class TextarenaEnvConfig(BaseEnvConfig):
    """Configuration for the Textarena environment trainer with VR-CLI."""

    env_name: str = "Textarena"
    max_steps: int = 50

    # Environment selection configuration
    registry_mode: str = Field(
        default="random", description="Registry mode: random, challenge, specific"
    )
    registry_difficulty: Optional[str] = Field(
        default=None, description="Difficulty: easy, medium, hard, expert, random"
    )
    registry_game_type: Optional[str] = Field(
        default=None,
        description="Game type: 1-player, 2-player, 3-player, multiplayer, mixed",
    )
    registry_max_players: Optional[int] = Field(
        default=None, description="Maximum number of players for selected games"
    )
    registry_min_players: Optional[int] = Field(
        default=None, description="Minimum number of players for selected games"
    )

    # VR-CLI specific configurations
    vrcli_enabled: bool = Field(
        default=True, description="Use VR-CLI scoring for action predictions"
    )
    vrcli_weight: float = Field(
        default=0.3, description="Weight for VR-CLI score in combined reward"
    )
    vrcli_discount_factor: float = Field(
        default=0.99,
        description="Discount factor for credit assignment in sparse reward setting",
    )

    # Format reward configuration
    format_reward_enabled: bool = Field(
        default=True, description="Apply format rewards for proper response structure"
    )
    format_reward_weight: float = Field(
        default=0.1, description="Weight for format reward in combined scoring"
    )
    format_memory_reward: float = Field(
        default=0.5, description="Reward for including a proper <memory> block"
    )
    format_thinking_reward: float = Field(
        default=0.5, description="Reward for including a proper <thinking> block"
    )
    format_strict_structure: bool = Field(
        default=True,
        description="Enforce strict structure: exactly 1 think, 1 memory, 1 tool_call in order",
    )
    format_wrong_order_penalty: float = Field(
        default=0.5, description="Penalty multiplier for blocks in wrong order"
    )
    format_extra_blocks_penalty: float = Field(
        default=0.2,
        description="Penalty for each extra block beyond the expected count",
    )

    # Token length penalty configuration
    token_length_penalty_enabled: bool = Field(
        default=True, description="Apply token length penalty/bonus to rewards"
    )
    token_length_penalty_weight: float = Field(
        default=0.1, description="Weight for token length penalty"
    )
    token_length_baseline: int = Field(
        default=500, description="Baseline token count for neutral penalty"
    )
    token_length_penalty_scale: float = Field(
        default=0.0002, description="Scale factor for token length penalty"
    )

    # LaTRo specific configurations
    latro_enabled: bool = Field(
        default=False,
        description="Use LaTRo scoring for action quality (disabled for now)",
    )
    latro_weight: float = Field(default=0.0, description="Weight for LaTRo score")

    debug_mode: bool = False

    enable_policy_thinking_summarization: bool = Field(
        default=True,
        description="Whether to use LLM-based summarization for thinking blocks",
    )
    max_policy_thinking_summary_tokens: int = Field(
        default=128, description="Maximum tokens for LLM-summarized thinking blocks"
    )

    # Registry configuration
    use_registry: bool = Field(
        default=True, description="Whether to use the registry for game selection"
    )
    registry_generation_ratio: float = Field(
        default=0.0, description="Ratio of generated games vs pre-built"
    )

    # Environment filtering
    registry_include_envs: Optional[List[str]] = Field(
        default=None,
        description="Specific environments to include in registry selection",
    )
    registry_exclude_envs: Optional[List[str]] = Field(
        default=None,
        description="Specific environments to exclude from registry selection",
    )

    default_server_config: APIServerConfig = Field(
        default_factory=lambda: APIServerConfig(
            server_type="openai", model_name="gpt-3.5-turbo"
        )
    )
    policy_agent_server_config: Optional[APIServerConfig] = None

    textarena_agent_config: TextarenaAgentConfig = Field(
        default_factory=TextarenaAgentConfig
    )

    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True


class TextarenaEpisodeState:
    """Stores per-episode state for a Textarena game using VR-CLI."""

    def __init__(
        self,
        episode_id: str,
        textarena_env_instance: ta.Env,
        initial_obs: List[Tuple[int, str, ta.ObservationType]],
        max_steps: int,
        num_players: int,
    ):
        self.episode_id: str = episode_id
        self.textarena_env: ta.Env = textarena_env_instance
        self.initial_formatted_obs: List[Tuple[int, str, ta.ObservationType]] = (
            initial_obs
        )
        self.max_turns: int = max_steps
        self.num_players: int = num_players

        self.policy_step_data: List[ScoredDataGroup] = []

        self.cumulative_rewards: Dict[int, float] = {
            pid: 0.0 for pid in range(num_players)
        }
        self.max_turns: int = max_steps

        self.last_rewards: Dict[int, float] = {}
        self.current_player_id: Optional[int] = None
        self.game_over: bool = False
        self.last_env_observations: Optional[
            List[Tuple[int, str, ta.ObservationType]]
        ] = None

        self.canonical_rewards: List[Dict[int, float]] = []
        self.canonical_chosen_alternative_indices: List[int] = []

        # Episode-level cache for thinking block summarizations
        self.thinking_block_cache: Dict[int, str] = {}

    def get_current_player_id(self) -> Optional[int]:
        """Get the current player ID from the environment."""
        return (
            self.textarena_env.state.current_player_id
            if hasattr(self.textarena_env, "state")
            else None
        )
