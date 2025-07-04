"""
DiplomacyEnvGRPO - GRPO-compatible Diplomacy Environment

This environment implements proper trajectory collection for GRPO training
by intercepting LLM calls and performing best-of-N selection at each
decision point.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from pydantic import Field

# Load environment variables from .env file
load_dotenv()

# Add AI_Diplomacy to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "AI_Diplomacy"))

from atropos_client import AtroposClient, register_atropos_models

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

# Import AI_Diplomacy components - delayed to prevent early initialization

logger = logging.getLogger(__name__)


class DiplomacyEnvGRPOConfig(BaseEnvConfig):
    """Configuration for GRPO Diplomacy environment."""

    # Training configuration
    training_power: str = Field(
        default="FRANCE", description="Which power to train (others will be fixed LLMs)"
    )

    # Game configuration
    max_game_years: int = Field(
        default=5, description="Maximum years to play (each year has multiple phases)"
    )
    negotiation_rounds: int = Field(
        default=1, description="Number of negotiation rounds per turn"
    )

    # Opponent models
    opponent_models: Dict[str, str] = Field(
        default={
            "AUSTRIA": "gpt-4o-mini",
            "ENGLAND": "gpt-4o-mini",
            "GERMANY": "gpt-4o-mini",
            "ITALY": "gpt-4o-mini",
            "RUSSIA": "gpt-4o",
            "TURKEY": "gpt-4o",
        },
        description="Models to use for non-training powers",
    )

    # Scoring configuration
    temperature: float = 0.7
    top_p: float = 0.9

    # Environment settings
    wandb_name: str = "diplomacy_grpo"
    game_log_dir: str = "./game_logs"

    # Credit assignment
    discount_factor: float = Field(
        default=0.99, description="Discount factor for Monte Carlo returns"
    )

    # LaTRo reward configuration
    use_latro_rewards: bool = Field(
        default=True,
        description="Use LaTRo (cross-entropy) rewards for best-of-N selection",
    )
    latro_beta: float = Field(
        default=0.05,
        description="KL penalty coefficient for LaTRo rewards (not currently used)",
    )


class DiplomacyEnvGRPO(BaseEnv):
    """
    GRPO-compatible Diplomacy environment.

    Uses InterceptingAtroposClient to implement best-of-N selection
    at each LLM decision point while the game runs normally.
    """

    name = "diplomacy_grpo"
    env_config_cls = DiplomacyEnvGRPOConfig

    def __init__(
        self,
        config: DiplomacyEnvGRPOConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = False,
        testing: bool = False,
    ):
        logger.info("[DiplomacyEnvGRPO.__init__] Starting initialization")
        super().__init__(config, server_configs, slurm, testing)
        self.config: DiplomacyEnvGRPOConfig = config

        # Create game log directory
        Path(self.config.game_log_dir).mkdir(parents=True, exist_ok=True)

        # Track active games
        self.active_games = {}

        # Register atropos models with GRPO support
        logger.info(
            "[DiplomacyEnvGRPO.__init__] About to register atropos models with GRPO support"
        )
        register_atropos_models(self)
        logger.info("[DiplomacyEnvGRPO.__init__] Registration complete")

    async def setup(self):
        """Initialize the environment."""
        logger.info("Setting up Diplomacy GRPO environment")

        # Note: We'll register intercepting models per-episode in collect_trajectories
        # since they need a reference to the env instance

    async def get_next_item(self) -> Dict[str, Any]:
        """Get configuration for next game episode."""
        import random

        seed = random.randint(0, 1000000)
        return {
            "seed": seed,
            "game_id": f"grpo-game-{seed}",
            "training_power": self.config.training_power,
        }

    async def collect_trajectories(
        self, item: Dict[str, Any]
    ) -> Tuple[List[ScoredDataGroup], List[Dict]]:
        """
        Collect trajectories for one complete game episode.

        This method:
        1. Sets up intercepting client for the training agent
        2. Runs a complete game using lm_game
        3. Collects ScoredDataGroups from the intercepting client
        4. Applies credit assignment based on final game outcome
        """
        seed = item["seed"]
        game_id = item["game_id"]
        training_power = item["training_power"]

        # Store training power for the intercepting client to use
        self.training_power = training_power

        logger.info(
            f"Starting trajectory collection for game {game_id}, "
            f"training power: {training_power}"
        )

        # Set up power configurations
        power_configs = {}
        for power in [
            "AUSTRIA",
            "ENGLAND",
            "FRANCE",
            "GERMANY",
            "ITALY",
            "RUSSIA",
            "TURKEY",
        ]:
            if power == training_power:
                # Use atropos client for training
                # Use the same pattern as the working version
                power_configs[power] = {
                    "model": "atropos-training-policy",
                    "is_training": True,
                }
            else:
                # Use fixed LLM models for opponents
                power_configs[power] = {
                    "model": self.config.opponent_models.get(power, "gpt-4o-mini"),
                    "is_training": False,
                }

        # Import lm_game here to delay AI_Diplomacy initialization
        import argparse

        sys.path.append(os.path.join(os.path.dirname(__file__), "AI_Diplomacy"))
        import lm_game

        # Build model string (comma-separated)
        model_list = []
        for power in [
            "AUSTRIA",
            "ENGLAND",
            "FRANCE",
            "GERMANY",
            "ITALY",
            "RUSSIA",
            "TURKEY",
        ]:
            model_list.append(power_configs[power]["model"])
        models_str = ",".join(model_list)

        # Store original argv to restore later
        original_argv = sys.argv

        # Set up command line args like the working version
        sys.argv = [
            "lm_game.py",
            "--max_year",
            str(1901 + self.config.max_game_years - 1),
            "--models",
            models_str,
            "--output",
            os.path.join(self.config.game_log_dir, f"{game_id}.json"),
            "--max_tokens",
            "16000",
            "--num_negotiation_rounds",
            str(self.config.negotiation_rounds),
        ]

        try:
            # Run the game
            logger.info(f"Starting AI_Diplomacy game {game_id}")

            # Store reference to training client
            training_client = None

            # Monkey-patch the model loading to capture our client
            from ai_diplomacy import clients

            original_load = clients.load_model_client

            def capture_training_client(model_id: str):
                client = original_load(model_id)
                if (
                    model_id == "atropos-training-policy"
                    and isinstance(client, AtroposClient)
                    and client.is_training
                ):
                    nonlocal training_client
                    training_client = client
                    logger.info(
                        f"[DiplomacyEnvGRPO] Captured training client for {client.power}"
                    )
                return client

            clients.load_model_client = capture_training_client

            # Run the game
            await lm_game.main()

            # Restore original functions
            clients.load_model_client = original_load
            sys.argv = original_argv

            if training_client is None:
                logger.error("Failed to capture training client!")
                return [], []

            # Get trajectory data from intercepting client
            trajectory_data = training_client.get_trajectory_data()

            # Load game result from saved file
            output_path = os.path.join(self.config.game_log_dir, f"{game_id}.json")
            if os.path.exists(output_path):
                with open(output_path, "r") as f:
                    game_result = json.load(f)
            else:
                logger.error(f"Game output file not found: {output_path}")
                game_result = {}

            logger.info(
                f"Game {game_id} completed. Collected {len(trajectory_data)} decision points"
            )

            # Apply credit assignment based on final game outcome
            final_score = self._compute_final_score(game_result, training_power)
            trajectory_data = self._apply_credit_assignment(
                trajectory_data, final_score, self.config.discount_factor
            )

            return trajectory_data, []

        except Exception as e:
            logger.error(f"Error running game {game_id}: {e}", exc_info=True)
            # Make sure to restore sys.argv
            sys.argv = original_argv
            return [], []

    def _compute_final_score(self, game_result: Dict, training_power: str) -> float:
        """
        Compute final score for the training power.

        This can be based on:
        - Supply center count
        - Survival
        - Relative ranking
        - Victory
        """
        try:
            # Get final supply center counts
            final_sc = game_result.get("supply_centers", {})
            training_sc = len(final_sc.get(training_power, []))

            # Get winner if any
            winner = game_result.get("winner")

            # Compute score
            if winner == training_power:
                # Victory bonus
                score = 10.0
            elif training_sc == 0:
                # Eliminated
                score = -5.0
            else:
                # Score based on supply centers (0-18 centers -> -2 to +2)
                score = (training_sc - 9) / 4.5

            logger.info(
                f"Final score for {training_power}: {score:.2f} "
                f"(SCs: {training_sc}, Winner: {winner})"
            )

            return score

        except Exception as e:
            logger.error(f"Error computing final score: {e}")
            return 0.0

    def _apply_credit_assignment(
        self,
        trajectory_data: List[ScoredDataGroup],
        final_score: float,
        discount_factor: float,
    ) -> List[ScoredDataGroup]:
        """
        Apply credit assignment to trajectory data.

        Uses Monte Carlo returns with discounting to propagate
        final game outcome back through the trajectory.
        """
        if not trajectory_data:
            return []

        logger.info(
            f"Applying credit assignment to {len(trajectory_data)} steps "
            f"with final score {final_score:.2f}"
        )

        # Work backwards through trajectory
        cumulative_return = final_score

        for i in range(len(trajectory_data) - 1, -1, -1):
            group = trajectory_data[i]

            # Get current step scores (per-alternative)
            step_scores = group.get("scores", [])

            # Update scores with discounted returns
            # Each alternative gets: step_score + γ * future_return
            updated_scores = []
            for j, step_score in enumerate(step_scores):
                # For now, all alternatives get the same future return
                # In future, we could give bonus to alternatives that
                # produced the same action as the selected one
                new_score = step_score + discount_factor * cumulative_return
                updated_scores.append(new_score)

            # Update the group
            group["scores"] = updated_scores

            # Discount return for next step
            cumulative_return *= discount_factor

        return trajectory_data

    async def evaluate(self, *args, **kwargs):
        """Evaluation logic for Diplomacy."""
        # TODO: Implement evaluation games
        logger.info("Evaluation not yet implemented for Diplomacy GRPO")

    @classmethod
    def config_init(cls) -> Tuple[DiplomacyEnvGRPOConfig, List[APIServerConfig]]:
        """Initialize default configuration."""
        env_config = DiplomacyEnvGRPOConfig(
            tokenizer_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
            group_size=4,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            max_token_length=2048,
            wandb_name="diplomacy_grpo",
            training_power="FRANCE",
            max_game_years=3,
            negotiation_rounds=1,
            discount_factor=0.99,
        )

        # Server config for the training policy
        server_configs = [
            APIServerConfig(
                model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
                base_url="http://localhost:8001/v1",
                api_key="none",
                num_requests_for_eval=16,
            )
        ]

        return env_config, server_configs

    @classmethod
    def cli(cls):
        super().cli()


if __name__ == "__main__":
    DiplomacyEnvGRPO.cli()
