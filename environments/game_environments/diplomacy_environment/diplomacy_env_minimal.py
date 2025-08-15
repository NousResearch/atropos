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

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
    ScoredDataItem,
)
from atroposlib.type_definitions import Item
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer
from environments.game_environments.diplomacy_environment.atropos_client_minimal import (
    clear_game_interactions,
    current_game_context,
    get_game_interactions,
    register_atropos_models_globally,
)
from environments.game_environments.diplomacy_environment.queue_manager import (
    PolicyRequest,
    PolicyResponse,
    get_queue_manager,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "AI_Diplomacy"))
import lm_game  # noqa: E402

logger = logging.getLogger(__name__)

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

    # Game settings (minimal implementation, no randomisation stuff)
    max_game_turns: int = 10
    training_power: str = "FRANCE"
    total_steps: int = 10

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

    # Opponent models (None = use SGLang endpoints from server_configs)
    opponent_models: Optional[List[str]] = None


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

        self.queue_manager = get_queue_manager()
        self.active_games: Dict[str, Dict] = {}

        if config.save_game_logs:
            Path(config.game_logs_dir).mkdir(exist_ok=True)

        self.system_prompt = (
            f"You are playing Diplomacy as {config.training_power}. "
            "Analyze the game state and respond with your strategy and orders."
        )

        if config.opponent_models:
            self.opponent_models = config.opponent_models
        else:
            self.opponent_models = []
            for server_config in server_configs:
                model_spec = f"openai:{server_config.model_name}@{server_config.base_url}#{server_config.api_key}"
                self.opponent_models.append(model_spec)

    @classmethod
    def config_init(cls) -> Tuple[DiplomacyEnvMinimalConfig, List[APIServerConfig]]:
        """Initialize default configuration."""
        env_config = DiplomacyEnvMinimalConfig(
            tokenizer_name="NousResearch/Hermes-4-Qwen3-14B-1-e3",
            group_size=4,
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

        register_atropos_models_globally(self.queue_manager)

        if self.config.start_diplomacy_server:
            await self._start_diplomacy_server()

        asyncio.create_task(self._poll_request_queues())

    async def _poll_request_queues(self):
        """Poll request queues and handle policy requests."""
        while True:
            try:
                for game_id in list(self.active_games.keys()):
                    queue_pair = self.queue_manager.get_queue_pair(game_id)
                    if not queue_pair:
                        continue

                    try:
                        request = queue_pair.request_queue.get_nowait()
                        asyncio.create_task(self._handle_policy_request(request))
                    except asyncio.QueueEmpty:
                        pass

                await asyncio.sleep(0.01)

            except Exception as e:
                logger.error(f"Error in queue polling: {e}")
                await asyncio.sleep(1)

    async def _handle_policy_request(self, request: PolicyRequest):
        """Handle a single policy request by sampling from SGLang."""
        try:
            logger.info(
                f"Handling request {request.request_id} for {request.power} in game {request.game_id}"
            )

            messages = [{"role": "system", "content": self.system_prompt}]

            for interaction in request.trajectory:
                messages.append({"role": "user", "content": interaction["prompt"]})
                messages.append(
                    {"role": "assistant", "content": interaction["response"]}
                )

            messages.append({"role": "user", "content": request.prompt})

            async with self.server.dedicated_server() as server:
                response = await server.chat_completion(
                    messages=messages,
                    n=1,
                    temperature=request.temperature,
                    max_tokens=2000,
                )

            response_text = response.choices[0].message.content.strip()

            policy_response = PolicyResponse(
                request_id=request.request_id,
                response=response_text,
                metadata={
                    "power": request.power,
                    "phase": request.phase,
                },
            )

            await self.queue_manager.put_response(request.game_id, policy_response)
            logger.debug(f"Sent response for request {request.request_id}")

        except Exception as e:
            logger.error(f"Error handling policy request: {e}")
            error_response = PolicyResponse(
                request_id=request.request_id,
                response="Error: Failed to generate response",
                metadata={"error": str(e)},
            )
            await self.queue_manager.put_response(request.game_id, error_response)

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

    async def collect_trajectories(
        self, item: Item
    ) -> Tuple[ScoredDataGroup, List[Item]]:
        """
        Run parallel Diplomacy games and collect all trajectories.

        This implements the key RL training pattern:
        1. Run group_size parallel games with the same seed
        2. Each game explores different action sequences
        3. Score each trajectory based on game outcome
        4. Return all trajectories as a ScoredDataGroup for training
        """
        logger.warning(
            f"[DiplomacyEnvMinimal] collect_trajectories called with item: {item}"
        )
        base_game_id = item.get("game_id", f"game-{int(time.time())}")
        seed = item.get("seed", random.randint(0, 1_000_000))

        logger.info(
            f"Starting {self.config.group_size} parallel games with seed {seed}"
        )

        # Run parallel games w/ same seed
        game_tasks = []
        for i in range(self.config.group_size):
            game_id = f"{base_game_id}-{i}"
            task = self._run_single_game(game_id, seed, trajectory_id=i)
            game_tasks.append(task)

        results = await asyncio.gather(*game_tasks, return_exceptions=True)

        scored_items = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Game {i} failed: {result}")
                continue

            if result and result[0]:
                scored_items.append(result[0])

        logger.warning(
            f"[DiplomacyEnvMinimal] Collected {len(scored_items)} scored items"
        )
        if not scored_items:
            logger.error("No valid trajectories collected")
            return (
                ScoredDataGroup(
                    tokens=[],
                    masks=[],
                    scores=[],
                    messages=[],
                    advantages=None,
                    ref_logprobs=None,
                    group_overrides={},
                    overrides=None,
                    images=None,
                ),
                [],
            )

        sdg = ScoredDataGroup(
            tokens=[],
            masks=[],
            scores=[],
            messages=[],
            advantages=None,
            ref_logprobs=None,
            group_overrides={},
            overrides=None,
            images=None,
        )

        for scored_item in scored_items:
            sdg["tokens"].append(scored_item["tokens"])
            sdg["masks"].append(scored_item["masks"])
            sdg["scores"].append(scored_item["scores"])
            if self.config.include_messages and scored_item.get("messages"):
                sdg["messages"].append(scored_item["messages"])

        logger.info(f"Collected {len(scored_items)} trajectories")
        logger.warning(
            f"[DiplomacyEnvMinimal] Returning ScoredDataGroup with {len(sdg['tokens'])} "
            f"tokens, {len(sdg['scores'])} scores"
        )
        logger.warning(
            f"[DiplomacyEnvMinimal] First few scores: {sdg['scores'][:5] if sdg['scores'] else 'None'}"
        )

        for i in range(self.config.group_size):
            game_id = f"{base_game_id}-{i}"
            if game_id in self.active_games:
                del self.active_games[game_id]
            try:
                await self.queue_manager.remove_game_queues(game_id)
            except Exception as e:
                logger.debug(f"Error cleaning up queues for {game_id}: {e}")

        return sdg, []

    async def _run_single_game(
        self, game_id: str, seed: int, trajectory_id: int
    ) -> Tuple[Optional[ScoredDataItem], None]:
        """
        Run a single Diplomacy game and return scored trajectory.
        """
        try:
            queue_pair = await self.queue_manager.create_game_queues(game_id)

            self.active_games[game_id] = {
                "queue_pair": queue_pair,
                "start_time": time.time(),
                "interactions": [],
            }

            token = current_game_context.set(game_id)

            try:
                game_result = await self._run_diplomacy_game(
                    game_id, seed, trajectory_id
                )
            finally:
                current_game_context.reset(token)

            if not game_result:
                logger.error(f"Game {game_id} failed to complete")
                return None, None

            score = self._calculate_score(game_result, self.config.training_power)

            interactions = get_game_interactions(game_id)

            training_interactions = [
                i for i in interactions if i.get("power") == self.config.training_power
            ]

            if training_interactions:
                messages = [{"role": "system", "content": self.system_prompt}]

                for interaction in training_interactions:
                    messages.append({"role": "user", "content": interaction["prompt"]})
                    messages.append(
                        {"role": "assistant", "content": interaction["response"]}
                    )

                logger.info(
                    f"Collected {len(training_interactions)} interactions for {self.config.training_power}"
                )
            else:
                logger.warning(
                    f"No interactions found for {self.config.training_power} in game {game_id}"
                )
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Playing Diplomacy game {game_id}"},
                    {
                        "role": "assistant",
                        "content": f"Game completed with score {score:.2f}",
                    },
                ]

            clear_game_interactions(game_id)

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

        game_output_path = os.path.join(self.config.game_logs_dir, f"{game_id}.json")

        models = []
        opponent_idx = 0

        for power in POWERS:
            if power == self.config.training_power:
                models.append("atropos-training-policy")
            else:
                models.append(
                    self.opponent_models[opponent_idx % len(self.opponent_models)]
                )
                opponent_idx += 1

        original_argv = sys.argv

        try:
            sys.argv = [
                "lm_game.py",
                "--models",
                ",".join(models),
                "--max_year",
                str(1900 + self.config.max_game_turns),
                "--output",
                game_output_path,
                "--seed",
                str(seed + trajectory_id),
                "--num_negotiation_rounds",
                "0",
            ]

            await lm_game.main()

            actual_game_file = os.path.join(game_output_path, "lmvsgame.json")
            if os.path.exists(actual_game_file):
                with open(actual_game_file, "r") as f:
                    saved_game = json.load(f)

                phases = saved_game.get("phases", [])
                last_phase = phases[-1] if phases else {}

                result = {
                    "winner": saved_game.get("winner"),
                    "turns_played": len(phases),
                    "final_centers": {},
                }

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
            sys.argv = original_argv

    def _calculate_score(self, game_result: Dict, power: str) -> float:
        """Calculate score for a power based on game outcome."""
        start_centers = STARTING_SUPPLY_CENTERS[power]
        end_centers = game_result["final_centers"].get(power, 0)
        center_score = (end_centers - start_centers) * self.config.supply_center_weight

        survival_score = self.config.survival_bonus * game_result["turns_played"]

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

            scored_data_group, _ = await self.collect_trajectories(item)
            if scored_data_group and scored_data_group["scores"]:
                avg_score = sum(scored_data_group["scores"]) / len(
                    scored_data_group["scores"]
                )
                eval_scores.append(avg_score)

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

        self.game_outcomes_buffer = []

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
