"""
Diplomacy Environment for Atropos - No Thinking Version

This environment integrates AI_Diplomacy with Atropos for training RL policies
in the game of Diplomacy. Supports mixed agent games where the training policy
plays against strong LLMs, other policies, or humans.
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
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel

# Add AI_Diplomacy to path
sys.path.append(os.path.join(os.path.dirname(__file__), "AI_Diplomacy"))

# Import our integration components
from atropos_client import AtroposClient, register_atropos_models

# Import AI_Diplomacy components
from diplomacy import Game

from atroposlib.envs.base import APIServerConfig, BaseEnv, BaseEnvConfig, ScoredDataItem
from atroposlib.type_definitions import Item, Message
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer
from atroposlib.utils.tool_call_parser import parse_tool_call

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


class PowerConfig(BaseModel):
    """Configuration for a single power in the game."""

    type: str = "llm"  # "atropos", "llm", or "human"
    model: str = "gpt-4o"
    is_training: bool = False  # Whether this power contributes to training data


class DiplomacyEnvNoThinkingConfig(BaseEnvConfig):
    """Configuration for the Diplomacy environment."""

    # Game settings
    max_game_turns: int = 20
    game_deadline_seconds: int = 300  # 5 minutes per phase

    # Power configurations - which agents play which powers
    powers_config: Dict[str, PowerConfig] = {
        "FRANCE": {"type": "atropos", "model": "training-policy", "is_training": True},
        "ENGLAND": {
            "type": "llm",
            "model": "claude-3-5-sonnet-20241022",
            "is_training": False,
        },
        "GERMANY": {"type": "llm", "model": "gpt-4o", "is_training": False},
        "ITALY": {"type": "llm", "model": "gemini-2.0-flash-exp", "is_training": False},
        "AUSTRIA": {"type": "llm", "model": "gpt-4o", "is_training": False},
        "RUSSIA": {
            "type": "llm",
            "model": "claude-3-5-sonnet-20241022",
            "is_training": False,
        },
        "TURKEY": {"type": "llm", "model": "gpt-4o", "is_training": False},
    }

    # Server settings
    launch_web_server: bool = True
    web_server_port: int = 8432
    atropos_server_url: str = "http://localhost:8000"

    # Scoring settings
    score_by_supply_centers: bool = True
    survival_bonus: float = 0.1  # Bonus per turn survived
    win_bonus: float = 10.0  # Bonus for winning

    # Logging
    save_game_logs: bool = True
    game_logs_dir: str = "./game_logs"

    # Evaluation
    eval_episodes: int = 10
    eval_opponent_models: List[str] = ["claude-3-5-sonnet-20241022", "gpt-4o"]


class DiplomacyEnvNoThinking(BaseEnv):
    name = "diplomacy_no_thinking"
    env_config_cls = DiplomacyEnvNoThinkingConfig

    def __init__(
        self,
        config: DiplomacyEnvNoThinkingConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = False,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: DiplomacyEnvNoThinkingConfig = config
        self.game_outcomes_buffer: List[Dict[str, float]] = []
        self.eval_metrics_custom: List[Tuple[str, float]] = []
        self.web_server_process: Optional[subprocess.Popen] = None

        # Register Atropos models with AI_Diplomacy
        register_atropos_models()

        # Set Atropos server URL
        os.environ["ATROPOS_SERVER_URL"] = config.atropos_server_url

        # Ensure game logs directory exists
        if config.save_game_logs:
            Path(config.game_logs_dir).mkdir(exist_ok=True)

        # Define the tool for submitting orders
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "submit_orders",
                    "description": "Submit diplomatic orders for the current turn",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "orders": {
                                "type": "object",
                                "description": "Dictionary mapping unit positions to orders",
                                "additionalProperties": {"type": "string"},
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Brief explanation of your strategy",
                            },
                        },
                        "required": ["orders"],
                    },
                },
            }
        ]

        tools_json = json.dumps(self.tools)
        self.system_prompt = (
            "You are playing Diplomacy as {power}. You need to submit orders for your units.\n\n"
            f"<tools>\n{tools_json}\n</tools>\n\n"
            "For your function call, return a JSON object with function name and arguments "
            "within <tool_call> </tool_call> tags with the following schema:\n"
            '<tool_call>\n{"arguments": {"orders": {"A PAR": "A PAR - BUR", "F BRE": "F BRE - MAO"}, '
            '"reasoning": "Moving to secure Burgundy and expand naval presence"}, "name": "submit_orders"}\n</tool_call>\n\n'
            "Your response should contain ONLY the tool call, no thinking or explanation outside of it."
        )

    @classmethod
    def config_init(cls) -> Tuple[DiplomacyEnvNoThinkingConfig, List[APIServerConfig]]:
        """Initialize default configuration."""
        env_config = DiplomacyEnvNoThinkingConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=1,  # One trajectory per item since it's a full game
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            max_token_length=4096,
            wandb_name=cls.name,
            steps_per_eval=10,
        )
        server_configs = [
            APIServerConfig(
                model_name="training-policy",
                base_url="http://localhost:8000/v1",
                api_key="x",
                num_requests_for_eval=10,
            ),
        ]
        return env_config, server_configs

    async def setup(self):
        """Set up the environment, including optional web server."""
        logger.info(f"Setting up {self.name} environment")

        if self.config.launch_web_server:
            await self._start_web_server()

    async def _start_web_server(self):
        """Start the AI_Diplomacy web server for game visualization."""
        try:
            logger.info(
                f"Starting Diplomacy web server on port {self.config.web_server_port}"
            )
            self.web_server_process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "diplomacy.server.run",
                    "--port",
                    str(self.config.web_server_port),
                ],
                cwd=os.path.join(os.path.dirname(__file__), "AI_Diplomacy"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            # Give server time to start
            await asyncio.sleep(2)
            logger.info(
                f"Diplomacy web server started at http://localhost:{self.config.web_server_port}"
            )
        except Exception as e:
            logger.error(f"Failed to start web server: {e}")

    def _get_training_powers(self) -> List[str]:
        """Get list of powers that are being trained."""
        return [
            power
            for power, config in self.config.powers_config.items()
            if config.is_training
        ]

    def _create_agent_configs(self, training_episode: bool = True) -> Dict[str, str]:
        """Create model configuration for each power."""
        models = {}
        for power, power_config in self.config.powers_config.items():
            if power_config.type == "atropos":
                # Use atropos- prefix for our custom client
                models[power] = f"atropos-{power_config.model}"
            else:
                # Use standard model names for LLMs
                models[power] = power_config.model
        return models

    async def collect_trajectory(
        self, item: Item
    ) -> Tuple[Optional[ScoredDataItem], List[Item]]:
        """
        Run a single Diplomacy game and collect trajectory for training powers.
        """
        game_id = item.get(
            "game_id", f"game-{int(time.time())}-{random.randint(1000, 9999)}"
        )
        seed = item.get("seed", random.randint(0, 1_000_000))

        logger.info(f"Starting Diplomacy game {game_id} with seed {seed}")

        # Get training powers
        training_powers = self._get_training_powers()
        if not training_powers:
            logger.warning("No training powers configured!")
            return None, []

        # Create agent configurations
        models = self._create_agent_configs()

        # Messages for the training agent(s)
        all_messages: Dict[str, List[Message]] = {
            power: [] for power in training_powers
        }
        game_scores: Dict[str, float] = {power: 0.0 for power in training_powers}

        try:
            # Run a full Diplomacy game using AI_Diplomacy's main function
            import argparse

            sys.path.append(os.path.join(os.path.dirname(__file__), "AI_Diplomacy"))
            import lm_game

            # Prepare arguments for lm_game.main()
            # We'll simulate command line args
            original_argv = sys.argv

            # Build model string for --models argument
            model_list = []
            for power in POWERS:
                power_config = self.config.powers_config[power]
                if power_config.type == "atropos":
                    # Use atropos- prefix for our custom client
                    model_list.append(f"atropos-{power_config.model}")
                else:
                    model_list.append(power_config.model)

            models_str = ",".join(model_list)

            # Set up args for lm_game
            game_output_path = (
                os.path.join(self.config.game_logs_dir, f"{game_id}.json")
                if self.config.save_game_logs
                else f"/tmp/{game_id}.json"
            )

            sys.argv = [
                "lm_game.py",
                "--max_year",
                str(1900 + self.config.max_game_turns),  # Convert turns to years
                "--models",
                models_str,
                "--output",
                game_output_path,
                "--max_tokens",
                "2000",
                "--num_negotiation_rounds",
                "1",  # Enable some negotiation
            ]

            # Monkey-patch argparse to avoid system exit
            original_parse_args = argparse.ArgumentParser.parse_args

            def custom_parse_args(self, args=None, namespace=None):
                if args is None:
                    args = sys.argv[1:]
                return original_parse_args(self, args, namespace)

            argparse.ArgumentParser.parse_args = custom_parse_args

            try:
                # Run the game
                logger.info(f"Running Diplomacy game with models: {models_str}")
                await lm_game.main()

                # Load the game result
                if os.path.exists(game_output_path):
                    with open(game_output_path, "r") as f:
                        saved_game = json.load(f)

                    # Extract game result info
                    phases = saved_game.get("phases", [])
                    game_result = {
                        "winner": saved_game.get("winner"),
                        "phase_history": [p["name"] for p in phases],
                        "supply_centers": {},
                    }

                    # Get final supply centers from last phase
                    if phases:
                        last_phase = phases[-1]
                        for power in POWERS:
                            game_result["supply_centers"][power] = (
                                last_phase.get("state", {})
                                .get("centers", {})
                                .get(power, [])
                            )
                else:
                    logger.error(f"Game output file not found: {game_output_path}")
                    return None, []

            finally:
                # Restore original argv and parse_args
                sys.argv = original_argv
                argparse.ArgumentParser.parse_args = original_parse_args

            # Calculate scores for training powers
            final_centers = game_result.get("supply_centers", {})
            winner = game_result.get("winner")
            turns_played = len(game_result.get("phase_history", []))

            for power in training_powers:
                # Supply center score
                start_centers = STARTING_SUPPLY_CENTERS[power]
                end_centers = len(final_centers.get(power, []))
                center_score = (end_centers - start_centers) / 10.0  # Normalize

                # Survival bonus
                survival_score = self.config.survival_bonus * turns_played

                # Win bonus
                win_score = self.config.win_bonus if winner == power else 0.0

                # Total score
                game_scores[power] = center_score + survival_score + win_score

                logger.info(
                    f"{power} score: centers={center_score:.2f}, survival={survival_score:.2f}, "
                    f"win={win_score:.2f}, total={game_scores[power]:.2f}"
                )

            # Store outcomes for metrics
            self.game_outcomes_buffer.append(
                {
                    "game_id": game_id,
                    "training_powers": training_powers,
                    "scores": game_scores,
                    "winner": winner,
                    "turns": turns_played,
                    "final_centers": {
                        p: len(centers) for p, centers in final_centers.items()
                    },
                }
            )

        except Exception as e:
            logger.error(f"Error running Diplomacy game {game_id}: {e}", exc_info=True)
            return None, []

        # For now, return trajectory for the first training power
        # In a full implementation, we'd handle multiple training powers properly
        primary_power = training_powers[0]

        # Create messages with system prompt and a dummy exchange
        primary_messages = [
            {
                "role": "system",
                "content": self.system_prompt.format(power=primary_power),
            },
            {
                "role": "user",
                "content": f"Game {game_id} completed. You played as {primary_power}. Final score: {game_scores[primary_power]:.2f}",
            },
            {
                "role": "assistant",
                "content": '<tool_call>\n{"arguments": {"orders": {"A PAR": "A PAR H"}, "reasoning": "Game completed"}, "name": "submit_orders"}\n</tool_call>',
            },
        ]

        # Tokenize messages
        tokenization_result = tokenize_for_trainer(
            tokenizer=self.tokenizer,
            chat=primary_messages,
            train_on_all_assistant_turns=True,
        )

        tokens = tokenization_result["tokens"]
        masks = tokenization_result["masks"]

        scored_data_item = ScoredDataItem(
            messages=primary_messages if self.config.include_messages else None,
            tokens=tokens,
            masks=masks,
            scores=game_scores[primary_power],
        )

        return scored_data_item, []

    async def get_next_item(self) -> Item:
        """Generate configuration for the next game."""
        return {
            "game_id": f"game-{int(time.time())}-{random.randint(1000, 9999)}",
            "seed": random.randint(0, 1_000_000),
        }

    async def evaluate(self, *args, **kwargs):
        """Evaluate the training policy against strong opponents."""
        logger.info(
            f"Starting evaluation for {self.name} with {self.config.eval_episodes} episodes"
        )

        eval_scores = []
        wins = 0
        total_games = self.config.eval_episodes

        for i in range(total_games):
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
                    in self._get_training_powers()
                ):
                    wins += 1
            else:
                logger.warning(f"Evaluation episode {i+1} failed to produce data")

        if eval_scores:
            avg_score = sum(eval_scores) / len(eval_scores)
            win_rate = wins / total_games

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
            total_score = sum(
                score
                for game in self.game_outcomes_buffer
                for score in game["scores"].values()
            )
            avg_score = total_score / (total_games * len(self._get_training_powers()))

            wins = sum(
                1
                for game in self.game_outcomes_buffer
                if game["winner"] in game["training_powers"]
            )
            win_rate = wins / total_games if total_games > 0 else 0

            avg_turns = (
                sum(game["turns"] for game in self.game_outcomes_buffer) / total_games
            )

            wandb_metrics.update(
                {
                    f"{self.name}_train/avg_score": avg_score,
                    f"{self.name}_train/win_rate": win_rate,
                    f"{self.name}_train/avg_game_turns": avg_turns,
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
        """Clean up web server on exit."""
        if self.web_server_process:
            self.web_server_process.terminate()
            self.web_server_process.wait()


if __name__ == "__main__":
    DiplomacyEnvNoThinking.cli()
