"""
Diplomacy Environment for Atropos RL Training

This module implements the main environment class that integrates AI_Diplomacy
with the Atropos training framework.
"""

import asyncio
import json
import logging
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import Field

from atroposlib.envs.base import (
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
    ScoredDataItem,
)
from atroposlib.type_definitions import Item, Message

from .diplomacy_agent import DiplomacyAgent, DiplomacyAgentConfig
from .diplomacy_game_manager import DiplomacyGameManager
from .diplomacy_scoring import DiplomacyScorer
from .diplomacy_types import (
    DiplomacyAction,
    DiplomacyEpisodeState,
    DiplomacyState,
    PowerState,
)

logger = logging.getLogger(__name__)


class DiplomacyEnvConfig(BaseEnvConfig):
    """Configuration for the Diplomacy environment."""

    # Game configuration
    powers_per_episode: int = Field(
        default=7,
        description="Number of powers (countries) in each game",
    )

    powers_controlled_by_model: List[str] = Field(
        default_factory=lambda: ["ENGLAND", "FRANCE", "GERMANY"],
        description="Which powers are controlled by the model (others use baseline)",
    )

    game_variant: str = Field(
        default="standard",
        description="Diplomacy game variant to use",
    )

    max_turns_per_phase: int = Field(
        default=5,
        description="Maximum number of negotiation rounds per phase",
    )

    max_game_years: int = Field(
        default=10,
        description="Maximum number of game years before ending",
    )

    # Agent configuration
    agent_config: DiplomacyAgentConfig = Field(
        default_factory=DiplomacyAgentConfig,
        description="Configuration for Diplomacy agents",
    )

    # Integration configuration
    use_submodule: bool = Field(
        default=True,
        description="Whether to use AI_Diplomacy as submodule (vs external API)",
    )

    diplomacy_engine_path: str = Field(
        default="./AI_Diplomacy",
        description="Path to AI_Diplomacy installation",
    )

    diplomacy_api_url: str = Field(
        default="http://localhost:8432",
        description="URL for AI_Diplomacy API if not using submodule",
    )

    # Scoring configuration
    enable_vrcli_scoring: bool = Field(
        default=True,
        description="Whether to use VR-CLI scoring for action predictions",
    )

    vrcli_weight: float = Field(
        default=0.3,
        description="Weight for VR-CLI component in composite score",
    )

    outcome_weight: float = Field(
        default=0.5,
        description="Weight for game outcome component in composite score",
    )

    negotiation_weight: float = Field(
        default=0.2,
        description="Weight for negotiation quality component in composite score",
    )

    # Memory configuration
    use_memory_system: bool = Field(
        default=True,
        description="Whether to use episodic memory system",
    )

    memory_top_k: int = Field(
        default=5,
        description="Number of similar memories to retrieve",
    )

    max_memories_per_episode: int = Field(
        default=100,
        description="Maximum memories to store per episode",
    )

    # Registry configuration
    use_scenario_registry: bool = Field(
        default=True,
        description="Whether to use scenario registry for varied starting positions",
    )

    scenario_distribution: Dict[str, float] = Field(
        default_factory=lambda: {
            "standard": 0.4,
            "historical": 0.3,
            "balanced": 0.2,
            "random": 0.1,
        },
        description="Distribution of scenario types",
    )


class DiplomacyEnv(BaseEnv):
    """
    Diplomacy environment for multi-agent reinforcement learning.

    This environment supports:
    - Multiple simultaneous games with different power assignments
    - Negotiation and order phases
    - Memory systems for long-term strategy
    - VR-CLI scoring for action prediction quality
    - Both rejection sampling and online RL modes
    """

    def __init__(self, config: DiplomacyEnvConfig):
        super().__init__(config)
        self.config: DiplomacyEnvConfig = config

        # Initialize game manager
        self.game_manager = DiplomacyGameManager(
            engine_path=self.config.diplomacy_engine_path,
            use_submodule=self.config.use_submodule,
            api_url=self.config.diplomacy_api_url,
        )

        # Initialize scorer
        self.scorer = DiplomacyScorer(
            vrcli_weight=self.config.vrcli_weight,
            outcome_weight=self.config.outcome_weight,
            negotiation_weight=self.config.negotiation_weight,
        )

        # Episode tracking
        self.active_episodes: Dict[str, DiplomacyEpisodeState] = {}

        # Initialize scenario registry if enabled
        if self.config.use_scenario_registry:
            from .diplomacy_registry import DiplomacyScenarioRegistry

            self.scenario_registry = DiplomacyScenarioRegistry(
                distribution=self.config.scenario_distribution
            )
        else:
            self.scenario_registry = None

    def generate_items(self) -> Tuple[List[Item], List[Item]]:
        """Generate items for training (episodes with power assignments)."""
        train_items = []
        eval_items = []

        # Generate episode configurations
        for i in range(self.config.total_item_number):
            episode_id = str(uuid.uuid4())

            # Get scenario from registry or use default
            if self.scenario_registry:
                scenario = self.scenario_registry.get_scenario()
            else:
                scenario = {"variant": "standard", "starting_position": None}

            # Assign powers to control
            power_assignment = self._generate_power_assignment()

            item = Item(
                id=episode_id,
                data={
                    "episode_id": episode_id,
                    "scenario": scenario,
                    "power_assignment": power_assignment,
                    "max_game_years": self.config.max_game_years,
                },
            )

            # Simple train/eval split
            if i < int(self.config.total_item_number * 0.9):
                train_items.append(item)
            else:
                eval_items.append(item)

        return train_items, eval_items

    def _generate_power_assignment(self) -> Dict[str, str]:
        """Generate assignment of powers to model vs baseline."""
        all_powers = [
            "ENGLAND",
            "FRANCE",
            "GERMANY",
            "ITALY",
            "AUSTRIA",
            "RUSSIA",
            "TURKEY",
        ]

        assignment = {}
        for power in all_powers:
            if power in self.config.powers_controlled_by_model:
                assignment[power] = "model"
            else:
                assignment[power] = "baseline"

        return assignment

    async def get_next_item(self, mode: str = "train") -> Optional[Item]:
        """Get the next episode configuration."""
        items = self.train_items if mode == "train" else self.eval_items

        if not items:
            return None

        # Simple round-robin for now
        item = items[self.current_train_idx % len(items)]
        self.current_train_idx += 1

        return item

    async def collect_trajectories(
        self,
        item: Item,
        num_copies: int,
        *,
        generation_kwargs: Dict[str, Any],
    ) -> ScoredDataGroup:
        """
        Collect trajectories for a Diplomacy episode.

        This involves:
        1. Setting up the game state
        2. Running negotiation and order phases
        3. Collecting alternative actions for each decision
        4. Scoring based on game outcomes
        """
        episode_id = item.id
        episode_data = item.data

        # Initialize episode state
        episode_state = DiplomacyEpisodeState(
            episode_id=episode_id,
            scenario=episode_data["scenario"],
            power_assignment=episode_data["power_assignment"],
            game_state=None,  # Will be set by game manager
            agents={},
            memories={},
            turn_history=[],
        )

        # Initialize game through manager
        game_state = await self.game_manager.initialize_game(
            scenario=episode_data["scenario"],
            power_assignment=episode_data["power_assignment"],
        )
        episode_state.game_state = game_state

        # Create agents for each power
        for power, control_type in episode_data["power_assignment"].items():
            agent_config = self.config.agent_config.copy()
            agent_config.power = power
            agent_config.is_baseline = control_type == "baseline"

            agent = DiplomacyAgent(
                config=agent_config,
                episode_state=episode_state,
            )
            episode_state.agents[power] = agent

        # Store episode state
        self.active_episodes[episode_id] = episode_state

        # Collect trajectories for the episode
        scored_groups = []

        try:
            # Run the game
            while not self._is_game_over(episode_state):
                # Negotiation phase
                negotiation_groups = await self._run_negotiation_phase(
                    episode_state, num_copies, generation_kwargs
                )
                scored_groups.extend(negotiation_groups)

                # Order phase
                order_groups = await self._run_order_phase(
                    episode_state, num_copies, generation_kwargs
                )
                scored_groups.extend(order_groups)

                # Execute orders and update game state
                await self.game_manager.execute_turn(episode_state)

        finally:
            # Cleanup
            del self.active_episodes[episode_id]

        # Combine all scored groups from the episode
        return self._combine_scored_groups(scored_groups)

    def _is_game_over(self, episode_state: DiplomacyEpisodeState) -> bool:
        """Check if the game should end."""
        game_state = episode_state.game_state

        # Check victory conditions
        for power, num_centers in game_state.supply_centers.items():
            if num_centers >= 18:  # Standard victory condition
                return True

        # Check max years
        if game_state.year >= self.config.max_game_years + 1901:
            return True

        # Check if only one power remains
        active_powers = [p for p, n in game_state.supply_centers.items() if n > 0]
        if len(active_powers) <= 1:
            return True

        return False

    async def _run_negotiation_phase(
        self,
        episode_state: DiplomacyEpisodeState,
        num_alternatives: int,
        generation_kwargs: Dict[str, Any],
    ) -> List[ScoredDataGroup]:
        """Run a negotiation phase and collect alternative messages."""
        scored_groups = []

        # For each controlled power, generate negotiation alternatives
        for power, agent in episode_state.agents.items():
            if agent.config.is_baseline:
                continue

            # Generate multiple negotiation strategies
            alternatives = await agent.generate_negotiation_alternatives(
                num_alternatives=num_alternatives,
                **generation_kwargs,
            )

            # Create scored group for this decision point
            group = self._create_negotiation_scored_group(
                episode_state, power, alternatives
            )
            scored_groups.append(group)

        return scored_groups

    async def _run_order_phase(
        self,
        episode_state: DiplomacyEpisodeState,
        num_alternatives: int,
        generation_kwargs: Dict[str, Any],
    ) -> List[ScoredDataGroup]:
        """Run an order phase and collect alternative orders."""
        scored_groups = []

        # For each controlled power, generate order alternatives
        for power, agent in episode_state.agents.items():
            if agent.config.is_baseline:
                continue

            # Generate multiple order strategies
            alternatives = await agent.generate_order_alternatives(
                num_alternatives=num_alternatives,
                **generation_kwargs,
            )

            # Create scored group for this decision point
            group = self._create_order_scored_group(episode_state, power, alternatives)
            scored_groups.append(group)

        return scored_groups

    def _create_negotiation_scored_group(
        self,
        episode_state: DiplomacyEpisodeState,
        power: str,
        alternatives: List[Dict[str, Any]],
    ) -> ScoredDataGroup:
        """Create a scored group for negotiation alternatives."""
        # TODO: Implement negotiation scoring
        # This will involve:
        # 1. Tokenizing the negotiation messages
        # 2. Creating masks for the generated portions
        # 3. Initial scoring based on coherence/strategy

        raise NotImplementedError("Negotiation scoring not yet implemented")

    def _create_order_scored_group(
        self,
        episode_state: DiplomacyEpisodeState,
        power: str,
        alternatives: List[Dict[str, Any]],
    ) -> ScoredDataGroup:
        """Create a scored group for order alternatives."""
        # TODO: Implement order scoring
        # This will involve:
        # 1. Tokenizing the order sets
        # 2. Creating masks for the generated portions
        # 3. Initial scoring based on validity/strategy

        raise NotImplementedError("Order scoring not yet implemented")

    def _combine_scored_groups(self, groups: List[ScoredDataGroup]) -> ScoredDataGroup:
        """Combine multiple scored groups into one."""
        if not groups:
            return ScoredDataGroup(
                tokens=[],
                masks=[],
                scores=[],
                advantages=None,
                ref_logprobs=None,
                messages=None,
            )

        # For now, just return the first group
        # TODO: Implement proper combination logic
        return groups[0]

    @classmethod
    def cli(cls, args=None) -> None:
        """CLI entry point for the Diplomacy environment."""
        cmd = Cmd(
            model=cls.get_pydantic_model(),
            default_config=cls.get_default_config_path(),
        )

        try:
            run_and_exit(cmd, args)
        except FailedExecutionException:
            logger.exception("Failed to run Diplomacy environment")
            raise


if __name__ == "__main__":
    DiplomacyEnv.cli()
