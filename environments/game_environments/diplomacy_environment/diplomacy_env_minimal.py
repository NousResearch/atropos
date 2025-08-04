"""
Minimal Diplomacy Environment for Atropos

A simplified implementation focusing on:
- Basic game integration with AI_Diplomacy
- Parallel rollouts with group_size
- LLM proxy interception via AtroposClient
- Simple supply center based scoring
"""

import asyncio
import json
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from atroposlib.envs.base import APIServerConfig, BaseEnv, BaseEnvConfig, ScoredDataItem
from atroposlib.type_definitions import Item
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

# Add AI_Diplomacy to path
sys.path.append(os.path.join(os.path.dirname(__file__), "AI_Diplomacy"))

logger = logging.getLogger(__name__)

# Diplomacy constants
POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]
STARTING_SUPPLY_CENTERS = {
    "AUSTRIA": 3,
    "ENGLAND": 3,
    "FRANCE": 3,
    "GERMANY": 3,
    "ITALY": 3,
    "RUSSIA": 4,
    "TURKEY": 3,
}


class DiplomacyEnvMinimalConfig(BaseEnvConfig):
    """Configuration for the minimal Diplomacy environment."""

    env_name: str = "diplomacy_minimal"

    # Game settings
    max_game_turns: int = 10  # Keep games short for faster iteration
    training_power: str = "FRANCE"  # Which power the RL agent controls
    total_steps: int = 10  # Low for initial testing

    # Scoring
    supply_center_weight: float = 1.0
    survival_bonus: float = 0.1
    win_bonus: float = 5.0

    # Process management
    diplomacy_server_port: int = 8432
    start_diplomacy_server: bool = True

    # Logging
    save_game_logs: bool = True
    game_logs_dir: str = "./game_logs"

    # Evaluation
    eval_episodes: int = 10


class DiplomacyEnvMinimal(BaseEnv):
    name = "diplomacy_minimal"
    env_config_cls = DiplomacyEnvMinimalConfig

    def __init__(
        self,
        config: DiplomacyEnvMinimalConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = False,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: DiplomacyEnvMinimalConfig = config
        self.game_server_process: Optional[subprocess.Popen] = None
        self.game_outcomes_buffer: List[Dict] = []
        self.eval_metrics_custom: List[Tuple[str, float]] = []

        # Ensure game logs directory exists
        if config.save_game_logs:
            Path(config.game_logs_dir).mkdir(exist_ok=True)

        # Simple system prompt for the training agent
        self.system_prompt = (
            f"You are playing Diplomacy as {config.training_power}. "
            "Analyze the game state and respond with your strategy and orders."
        )

    @classmethod
    def config_init(cls) -> Tuple[DiplomacyEnvMinimalConfig, List[APIServerConfig]]:
        """Initialize default configuration."""
        env_config = DiplomacyEnvMinimalConfig(
            tokenizer_name="NousResearch/Hermes-4-Qwen3-14B-1-e3",
            group_size=4,  # Run 4 parallel games
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            max_token_length=4096,
            wandb_name=cls.name,
            steps_per_eval=20,
        )
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/Hermes-4-Qwen3-14B-1-e3",
                base_url="http://localhost:9004/v1",
                api_key="x",
                num_requests_for_eval=128,
            ),
            APIServerConfig(
                model_name="NousResearch/Hermes-4-Qwen3-14B-1-e3",
                base_url="http://localhost:9005/v1",
                api_key="x",
                num_requests_for_eval=128,
            ),
            APIServerConfig(
                model_name="NousResearch/Hermes-4-Qwen3-14B-1-e3",
                base_url="http://localhost:9006/v1",
                api_key="x",
                num_requests_for_eval=128,
            ),
            APIServerConfig(
                model_name="NousResearch/Hermes-4-Qwen3-14B-1-e3",
                base_url="http://localhost:9007/v1",
                api_key="x",
                num_requests_for_eval=128,
            ),
        ]
        return env_config, server_configs

    async def setup(self):
        """Set up the environment."""
        logger.info(f"Setting up {self.name} environment")

        # Register our AtroposClient proxy
        await self._register_atropos_client()

        # Start Diplomacy server if requested
        if self.config.start_diplomacy_server:
            await self._start_diplomacy_server()

    async def _register_atropos_client(self):
        """Register AtroposClient with AI_Diplomacy."""
        try:
            from atropos_client_minimal import register_atropos_models

            # Get the first server config from the server manager
            if (
                hasattr(self, "server")
                and hasattr(self.server, "servers")
                and self.server.servers
            ):
                server_config = self.server.servers[0].config
                register_atropos_models(server_config)
            else:
                logger.error(
                    "No server configuration available for AtroposClient registration"
                )
            logger.info("Registered AtroposClient proxy")
        except Exception as e:
            logger.error(f"Failed to register AtroposClient: {e}")

    async def _start_diplomacy_server(self):
        """Start the AI_Diplomacy game server."""
        try:
            logger.info(
                f"Starting Diplomacy server on port {self.config.diplomacy_server_port}"
            )
            self.game_server_process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "diplomacy.server.run",
                    "--port",
                    str(self.config.diplomacy_server_port),
                ],
                cwd=os.path.join(os.path.dirname(__file__), "AI_Diplomacy"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            # Give server time to start
            await asyncio.sleep(3)
            logger.info(
                f"Diplomacy server started at http://localhost:{self.config.diplomacy_server_port}"
            )
        except Exception as e:
            logger.error(f"Failed to start Diplomacy server: {e}")

    async def collect_trajectory(
        self, item: Item
    ) -> Tuple[Optional[ScoredDataItem], List[Item]]:
        """
        Run parallel Diplomacy games and collect best trajectory.

        This implements the key RL training pattern:
        1. Run group_size parallel games with the same seed
        2. Each game explores different action sequences
        3. Score each trajectory based on game outcome
        4. Return the best trajectory for training
        """
        base_game_id = item.get("game_id", f"game-{int(time.time())}")
        seed = item.get("seed", random.randint(0, 1_000_000))

        logger.info(
            f"Starting {self.config.group_size} parallel games with seed {seed}"
        )

        # Run parallel games
        game_tasks = []
        for i in range(self.config.group_size):
            game_id = f"{base_game_id}-{i}"
            # Use same seed but different trajectory_id to get different rollouts
            task = self._run_single_game(game_id, seed, trajectory_id=i)
            game_tasks.append(task)

        # Wait for all games to complete
        results = await asyncio.gather(*game_tasks, return_exceptions=True)

        # Find best trajectory
        best_score = float("-inf")
        best_trajectory = None

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Game {i} failed: {result}")
                continue

            if result and result[0]:  # Check if we got a valid ScoredDataItem
                score = result[0]["scores"]
                if score > best_score:
                    best_score = score
                    best_trajectory = result[0]

        if best_trajectory:
            logger.info(f"Best trajectory score: {best_score:.2f}")
            return best_trajectory, []
        else:
            logger.error("No valid trajectories collected")
            return None, []

    async def _run_single_game(
        self, game_id: str, seed: int, trajectory_id: int
    ) -> Tuple[Optional[ScoredDataItem], None]:
        """
        Run a single Diplomacy game and return scored trajectory.
        """
        try:
            # Run game using AI_Diplomacy
            game_result = await self._run_diplomacy_game(game_id, seed, trajectory_id)

            if not game_result:
                logger.error(f"Game {game_id} failed to complete")
                return None, None

            # Calculate score for training power
            score = self._calculate_score(game_result, self.config.training_power)

            # Extract trajectory from our client
            # In a full implementation, we'd get actual game interactions here
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Playing Diplomacy game {game_id}"},
                {
                    "role": "assistant",
                    "content": f"Game completed with score {score:.2f}",
                },
            ]

            # Tokenize
            tokenization_result = tokenize_for_trainer(
                tokenizer=self.tokenizer,
                chat=messages,
                train_on_all_assistant_turns=True,
            )

            scored_data_item = ScoredDataItem(
                messages=messages if self.config.include_messages else None,
                tokens=tokenization_result["tokens"],
                masks=tokenization_result["masks"],
                scores=score,
            )

            # Store outcome for metrics
            self.game_outcomes_buffer.append(
                {
                    "game_id": game_id,
                    "score": score,
                    "winner": game_result.get("winner"),
                    "turns": game_result.get("turns_played", 0),
                    "final_centers": game_result.get("final_centers", {}),
                }
            )

            return scored_data_item, None

        except Exception as e:
            logger.error(f"Error in game {game_id}: {e}", exc_info=True)
            return None, None

    async def _run_diplomacy_game(
        self, game_id: str, seed: int, trajectory_id: int = 0
    ) -> Optional[Dict]:
        """Run a Diplomacy game using AI_Diplomacy's lm_game module."""
        # Import lm_game
        sys.path.append(os.path.join(os.path.dirname(__file__), "AI_Diplomacy"))
        import lm_game

        # Configure game
        game_output_path = os.path.join(self.config.game_logs_dir, f"{game_id}.json")

        # Build models list - training power uses atropos, others use diverse LLMs
        models = []
        # Mix of OpenAI and Anthropic models for variety
        opponent_models = [
            "gpt-4o-mini",  # OpenAI
            "anthropic:claude-sonnet-4-20250514",  # Anthropic Sonnet
            "o3",  # OpenAI o3
            "anthropic:claude-opus-4-20250514",  # Anthropic Opus
            "gpt-4o-mini",  # OpenAI
            "anthropic:claude-sonnet-4-20250514",  # Anthropic Sonnet
        ]
        opponent_idx = 0

        for power in POWERS:
            if power == self.config.training_power:
                models.append("atropos-training-policy")
            else:
                # Assign from opponent models
                models.append(opponent_models[opponent_idx])
                opponent_idx += 1

        # Save original argv
        original_argv = sys.argv

        try:
            # Set up arguments for lm_game
            sys.argv = [
                "lm_game.py",
                "--models",
                ",".join(models),
                "--max_year",
                str(1900 + self.config.max_game_turns),
                "--output",
                game_output_path,
                "--seed",
                str(
                    seed + trajectory_id
                ),  # Vary seed by trajectory for different rollouts
                "--num_negotiation_rounds",
                "0",  # No negotiation for minimal version
            ]

            # Run the game
            await lm_game.main()

            # Load and parse results
            # The output path is actually a directory, the game file is inside
            actual_game_file = os.path.join(game_output_path, "lmvsgame.json")
            if os.path.exists(actual_game_file):
                with open(actual_game_file, "r") as f:
                    saved_game = json.load(f)

                # Extract key info
                phases = saved_game.get("phases", [])
                last_phase = phases[-1] if phases else {}

                result = {
                    "winner": saved_game.get("winner"),
                    "turns_played": len(phases),
                    "final_centers": {},
                }

                # Get final supply centers
                for power in POWERS:
                    centers = (
                        last_phase.get("state", {}).get("centers", {}).get(power, [])
                    )
                    result["final_centers"][power] = len(centers)

                return result
            else:
                logger.error(f"Game output not found: {actual_game_file}")
                return None

        finally:
            # Restore argv
            sys.argv = original_argv

    def _calculate_score(self, game_result: Dict, power: str) -> float:
        """Calculate score for a power based on game outcome."""
        # Supply center score
        start_centers = STARTING_SUPPLY_CENTERS[power]
        end_centers = game_result["final_centers"].get(power, 0)
        center_score = (end_centers - start_centers) * self.config.supply_center_weight

        # Survival bonus
        survival_score = self.config.survival_bonus * game_result["turns_played"]

        # Win bonus
        win_score = self.config.win_bonus if game_result["winner"] == power else 0.0

        total_score = center_score + survival_score + win_score

        logger.info(
            f"{power} score: centers={center_score:.2f}, "
            f"survival={survival_score:.2f}, win={win_score:.2f}, "
            f"total={total_score:.2f}"
        )

        return total_score

    async def get_next_item(self) -> Item:
        """Generate configuration for the next game."""
        return {
            "game_id": f"game-{int(time.time())}-{random.randint(1000, 9999)}",
            "seed": random.randint(0, 1_000_000),
        }

    async def evaluate(self, *args, **kwargs):
        """Run evaluation games."""
        logger.info(f"Starting evaluation with {self.config.eval_episodes} episodes")

        eval_scores = []
        wins = 0

        for i in range(self.config.eval_episodes):
            item = await self.get_next_item()
            item["is_eval"] = True

            scored_item_tuple = await self.collect_trajectory(item)
            if scored_item_tuple and scored_item_tuple[0]:
                score = scored_item_tuple[0]["scores"]
                eval_scores.append(score)

                # Check if training power won
                if (
                    self.game_outcomes_buffer
                    and self.game_outcomes_buffer[-1]["winner"]
                    == self.config.training_power
                ):
                    wins += 1

        if eval_scores:
            avg_score = sum(eval_scores) / len(eval_scores)
            win_rate = wins / self.config.eval_episodes

            self.eval_metrics_custom = [
                (f"{self.name}_eval/avg_score", avg_score),
                (f"{self.name}_eval/win_rate", win_rate),
                (f"{self.name}_eval/num_completed", len(eval_scores)),
            ]

            logger.info(
                f"Evaluation completed: avg_score={avg_score:.2f}, win_rate={win_rate:.2%}"
            )

    async def wandb_log(self, wandb_metrics: Optional[Dict[str, float]] = None):
        """Log metrics to wandb."""
        if wandb_metrics is None:
            wandb_metrics = {}

        if self.game_outcomes_buffer:
            # Calculate aggregate metrics
            total_games = len(self.game_outcomes_buffer)
            avg_score = sum(g["score"] for g in self.game_outcomes_buffer) / total_games
            wins = sum(
                1
                for g in self.game_outcomes_buffer
                if g["winner"] == self.config.training_power
            )
            win_rate = wins / total_games
            avg_turns = sum(g["turns"] for g in self.game_outcomes_buffer) / total_games

            wandb_metrics.update(
                {
                    f"{self.name}_train/avg_score": avg_score,
                    f"{self.name}_train/win_rate": win_rate,
                    f"{self.name}_train/avg_turns": avg_turns,
                    f"{self.name}_train/num_games": total_games,
                }
            )

        # Clear buffer
        self.game_outcomes_buffer = []

        # Add eval metrics
        for key, value in self.eval_metrics_custom:
            wandb_metrics[key] = value
        self.eval_metrics_custom = []

        await super().wandb_log(wandb_metrics)

    def __del__(self):
        """Clean up server process on exit."""
        if self.game_server_process:
            self.game_server_process.terminate()
            self.game_server_process.wait()


if __name__ == "__main__":
    DiplomacyEnvMinimal.cli()
