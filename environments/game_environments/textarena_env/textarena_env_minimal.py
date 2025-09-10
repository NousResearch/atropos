#!/usr/bin/env python3
"""
Minimal TextArena Environment for Atropos

A simplified implementation focusing on:
- Random game selection from 99+ TextArena environments
- Multi-agent support with single agent training (GRPO)
- Simple opponent agents for multi-player games
"""

import asyncio
import logging
import random
import traceback
from typing import Any, Dict, List, Optional, Tuple

import textarena as ta
from dotenv import load_dotenv

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
    ScoredDataItem,
)
from atroposlib.type_definitions import Item, Message
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer
from environments.game_environments.textarena_env.textarena_registry import (
    create_textarena_registry,
)

load_dotenv()

logger = logging.getLogger(__name__)


class TextArenaEnvMinimalConfig(BaseEnvConfig):
    """Configuration for the minimal TextArena environment."""

    env_name: str = "TextArena"
    wandb_name: str = "textarena-trainer-minimal"
    group_size: int = 4
    max_num_workers: int = 16
    total_steps: int = 500
    max_steps: int = 20
    max_token_length: int = 8192

    # Game selection
    game_filter: str = "all"
    exclude_games: List[str] = []
    max_players: Optional[int] = None
    min_players: Optional[int] = None
    max_player_detection: int = 4
    specific_game: Optional[str] = None

    # Registry cache controls
    use_textarena_registry_cache: bool = True
    reset_textarena_registry_cache: bool = False
    textarena_registry_cache_path: Optional[str] = None
    validate_registry_selection: bool = False

    # Debugging / logging
    debug_rollout_logging: bool = False

    # Training configuration
    training_player_index: int = 0  # Which player position to train (0-indexed)

    # Opponent configuration for multi-player games (no true self play yet)
    opponent_temperature: float = 0.7
    opponent_max_tokens: int = 500


class TextArenaEnvMinimal(BaseEnv):
    """Minimal TextArena environment for training LLMs."""

    name = "textarena_minimal"
    env_config_cls = TextArenaEnvMinimalConfig

    _registry = None

    def __init__(
        self,
        config: TextArenaEnvMinimalConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = True,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: TextArenaEnvMinimalConfig = config
        self.registry = None

        # Tracking
        self.episode_outcomes_buffer = []
        self.episode_rewards_buffer = []
        self.episode_steps_buffer = []
        self.episode_game_types = []
        self.eval_metrics_custom = []
        # Note: nerfed this to use the same model for now, sooooo expensive otherwise
        self.opponent_clients = []

        self.system_prompt = (
            "You are an expert game player. You will be shown the current state of a text-based game.\n"
            "Your task is to play the game by choosing actions that will help you win.\n\n"
            "IMPORTANT RULES:\n"
            "1. Read the game state and instructions carefully\n"
            "2. Respond with ONLY the action you want to take\n"
            "3. Do not provide explanations or reasoning\n"
            "4. Use the exact format specified in the game instructions\n"
            "5. Be strategic and try to win the game\n\n"
            "Example responses: 'Up', 'Down', '5', 'red', '[1 2 3]', 'fold', etc.\n"
            "Your response should be a single action, nothing more."
        )

    async def setup(self):
        """Initialize the environment and game registry."""
        try:
            if TextArenaEnvMinimal._registry is None:
                logger.info("Creating TextArena registry singleton...")
                TextArenaEnvMinimal._registry = create_textarena_registry(
                    seed=None,
                    max_player_detection=self.config.max_player_detection,
                    cache_path=self.config.textarena_registry_cache_path,
                    use_cache=self.config.use_textarena_registry_cache,
                    reset_cache=self.config.reset_textarena_registry_cache,
                )
                logger.info(
                    f"Discovering games (max_player_detection={self.config.max_player_detection})..."
                )
                TextArenaEnvMinimal._registry.discover_games()

            self.registry = TextArenaEnvMinimal._registry

            available_games = self.registry.list_available_games(
                self.config.game_filter
            )
            logger.info(
                f"Using TextArena registry with {len(available_games)} games "
                f"(filter: {self.config.game_filter})"
            )

            logger.info("TextArenaEnvMinimal setup completed successfully")

        except Exception as e:
            logger.error(f"Failed to setup TextArena environment: {e}")
            logger.error(traceback.format_exc())
            raise

    async def get_next_item(self) -> Item:
        """Get the next game configuration."""
        env_id, metadata = self.registry.get_random_game(
            game_filter=self.config.game_filter,
            exclude_games=self.config.exclude_games,
            max_players=self.config.max_players,
            min_players=self.config.min_players,
            specific_game=self.config.specific_game,
            validate_on_select=self.config.validate_registry_selection,
        )

        if metadata["min_players"] == metadata["max_players"]:
            num_players = metadata["min_players"]
        else:
            num_players = random.randint(
                metadata["min_players"], metadata["max_players"]
            )

        return {
            "env_id": env_id,
            "num_players": num_players,
            "metadata": metadata,
            "game_type": metadata["game_type"],
        }

    async def collect_trajectories(
        self, item: Item
    ) -> Tuple[List[ScoredDataGroup], List[Item]]:
        """Collect parallel trajectories from the same game.

        Returns a list of ScoredDataGroups, one for each player across all games.
        """
        env_id = item["env_id"]
        num_players = item["num_players"]
        metadata = item["metadata"]

        logger.warning(
            f"Starting {self.config.group_size} parallel games of {env_id} "
            f"with {num_players} players"
        )

        # Collect player trajectories from all games
        # Structure: game_idx -> player_id -> ScoredDataItem
        all_game_results = []

        tasks = []
        for i in range(self.config.group_size):
            task = self._collect_single_trajectory(env_id, num_players, i, metadata)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Game {i} failed: {result}")
                continue
            if result and isinstance(result, dict):
                all_game_results.append(result)

        if not all_game_results:
            logger.error("No valid games completed")
            # Return empty ScoredDataGroup list
            return [], []

        # Organize by player across all games
        # Create one ScoredDataGroup per player
        player_groups = {}

        for game_result in all_game_results:
            for player_id, scored_item in game_result.items():
                if scored_item is None:
                    continue

                if player_id not in player_groups:
                    player_groups[player_id] = ScoredDataGroup(
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

                player_groups[player_id]["tokens"].append(scored_item["tokens"])
                player_groups[player_id]["masks"].append(scored_item["masks"])
                player_groups[player_id]["scores"].append(scored_item["scores"])
                if self.config.include_messages and scored_item.get("messages"):
                    player_groups[player_id]["messages"].append(scored_item["messages"])

        # Convert to list of ScoredDataGroups
        scored_data_groups = list(player_groups.values())

        logger.info(
            f"Collected {len(all_game_results)} games for {env_id} "
            f"with {len(scored_data_groups)} player groups"
        )

        return scored_data_groups, []

    async def _collect_single_trajectory(
        self,
        env_id: str,
        num_players: int,
        trajectory_idx: int,
        metadata: Dict[str, Any],
    ) -> Dict[int, Optional[ScoredDataItem]]:
        """Collect trajectories for all players in a single game.

        Returns a dictionary mapping player_index -> ScoredDataItem.
        """
        # Initialize message lists for each player
        player_messages: Dict[int, List[Message]] = {}
        for i in range(num_players):
            player_messages[i] = [{"role": "system", "content": self.system_prompt}]

        try:
            env = ta.make(env_id)

            agents = {}
            for i in range(num_players):
                if i == self.config.training_player_index:
                    agents[i] = None
                else:
                    agents[i] = "opponent"

            try:
                env.reset(num_players=num_players)
            except TypeError:
                env.reset()

            initial_obs = env.get_observation()
            if initial_obs is None or len(initial_obs) < 2:
                logger.error(f"Failed to get initial observation for {env_id}")
                return {}

            done = False
            player_rewards = {i: 0.0 for i in range(num_players)}
            step_count = 0
            player_turns = {i: 0 for i in range(num_players)}

            while not done and step_count < self.config.max_steps:
                if num_players == 1:
                    current_player = 0
                else:
                    current_player = (
                        env.state.current_player_id
                        if hasattr(env.state, "current_player_id")
                        else 0
                    )

                current_obs = env.get_observation()
                obs_str = (
                    current_obs[:200]
                    if isinstance(current_obs, (str, tuple))
                    else current_obs
                )
                logger.warning(
                    f"Step {step_count} - env.get_observation() returned: "
                    f"type={type(current_obs)}, value={obs_str}"
                )

                if current_obs is None:
                    logger.error(
                        f"Failed to get observation at step {step_count} - got None"
                    )
                    break

                # Extract the formatted observation string
                if isinstance(current_obs, tuple) and len(current_obs) >= 2:
                    obs_text = str(current_obs[1])
                elif isinstance(current_obs, str):
                    obs_text = current_obs
                else:
                    logger.error(f"Unexpected observation format: {type(current_obs)}")
                    obs_text = str(current_obs)

                if not obs_text or obs_text == "[]":
                    logger.error(f"Empty observation text at step {step_count}!")
                    # TODO: Could try stingifying the state instead as fallback
                    # but not sure if that works well in multi-player games
                    break

                if self.config.debug_rollout_logging:
                    _obs_preview = obs_text[:200].replace("\n", " ")
                    logger.info(
                        f"[Traj {trajectory_idx}] Step {step_count} | Player {current_player} | Obs: {_obs_preview}..."
                    )

                # Add observation to current player's messages
                player_messages[current_player].append(
                    {"role": "user", "content": obs_text}
                )

                if current_player == self.config.training_player_index:

                    async with self.server.dedicated_server() as server:
                        try:
                            logger.warning(
                                f"[Training Agent] Sending messages to server: {obs_str}"
                            )
                            for i, msg in enumerate(player_messages[current_player]):
                                logger.warning(
                                    f"  Message {i}: role={msg['role']}, content={msg['content'][:200]}..."
                                )

                            response = await server.chat_completion(
                                messages=player_messages[current_player],
                                n=1,
                                max_tokens=self.config.opponent_max_tokens,
                                temperature=0.7,
                            )
                            if response is None:
                                logger.error(
                                    "Training agent got None response object from policy server"
                                )
                                raise RuntimeError(
                                    "Policy server returned None response"
                                )

                            if not hasattr(response, "choices") or not response.choices:
                                logger.error(
                                    f"Training agent response has no choices: {response}"
                                )
                                raise RuntimeError(
                                    f"Policy server response has no choices: {response}"
                                )

                            if response.choices[0].message.content is not None:
                                action = response.choices[0].message.content.strip()
                                logger.debug(
                                    f"Turn {step_count}: Training agent action: {action}"
                                )
                            else:
                                logger.error(
                                    f"Training agent got None content from policy server. Full response: {response}"
                                )
                                raise RuntimeError(
                                    "Policy server returned None content in response"
                                )
                        except Exception as e:
                            logger.error(f"Training agent CRITICAL error: {e}")
                            logger.error(f"Error type: {type(e).__name__}")
                            logger.error(f"Server info: {server}")
                            raise
                        if self.config.debug_rollout_logging:
                            logger.info(
                                f"[Traj {trajectory_idx}] Step {step_count} | Training action: {action}"
                            )

                    player_messages[current_player].append(
                        {"role": "assistant", "content": action}
                    )
                    player_turns[current_player] += 1

                else:
                    # Opponent's turn
                    action = await self._get_opponent_action(
                        obs_text, current_player, trajectory_idx
                    )
                    # Track opponent's action in their message history
                    player_messages[current_player].append(
                        {"role": "assistant", "content": action}
                    )
                    player_turns[current_player] += 1

                    if self.config.debug_rollout_logging:
                        logger.info(
                            f"[Traj {trajectory_idx}] Step {step_count} | Opponent {current_player} action: {action}"
                        )

                # Execute action
                try:
                    # Handle various step return formats robustly
                    step_result = env.step(action)
                    done, info = self._parse_step_result(step_result)

                    # Get rewards from state or info for all players
                    if hasattr(env, "state") and hasattr(env.state, "rewards"):
                        rewards = env.state.rewards
                        if isinstance(rewards, dict):
                            for player_id, reward in rewards.items():
                                if player_id in player_rewards:
                                    player_rewards[player_id] += reward
                        else:
                            # Single player reward
                            if current_player in player_rewards:
                                player_rewards[current_player] += (
                                    rewards if rewards else 0
                                )
                    elif isinstance(info, dict) and "rewards" in info:
                        rewards = info["rewards"]
                        if isinstance(rewards, dict):
                            for player_id, reward in rewards.items():
                                if player_id in player_rewards:
                                    player_rewards[player_id] += reward
                        else:
                            # Single player reward
                            if current_player in player_rewards:
                                player_rewards[current_player] += (
                                    rewards if rewards else 0
                                )

                    step_count += 1

                    if self.config.debug_rollout_logging:
                        logger.info(
                            f"[Traj {trajectory_idx}] Step {step_count} | Done={done} "
                            f"| TotalReward={player_rewards[current_player]}"
                        )

                except Exception as e:
                    logger.error(f"Environment step error: {e}")
                    break

            env.close()

            # Determine outcome for each player
            player_outcomes = {}
            if done and isinstance(info, dict):
                winner = info.get("winner", None)
                for player_id in range(num_players):
                    if winner == player_id:
                        player_outcomes[player_id] = 1.0
                    elif winner is None:
                        player_outcomes[player_id] = 0.0  # Draw
                    else:
                        player_outcomes[player_id] = -1.0  # Loss
            elif step_count >= self.config.max_steps:
                # Episode ended due to max steps - treat as draw
                for player_id in range(num_players):
                    player_outcomes[player_id] = 0.0
                logger.warning(
                    f"Episode hit max_steps ({self.config.max_steps}), treating as draw"
                )
            else:
                for player_id in range(num_players):
                    player_outcomes[player_id] = 0.0
                logger.warning(
                    f"Episode ended unexpectedly: done={done}, steps={step_count}"
                )

            logger.warning(
                f"[Episode Complete] Game: {env_id}, Steps: {step_count}, Done: {done}"
            )

            # Track training player stats
            training_outcome = player_outcomes[self.config.training_player_index]
            training_reward = player_rewards[self.config.training_player_index]
            self.episode_outcomes_buffer.append(training_outcome)
            self.episode_rewards_buffer.append(training_reward)
            self.episode_steps_buffer.append(step_count)
            self.episode_game_types.append(metadata["game_type"])

            # Create ScoredDataItems for each player
            player_scored_items = {}
            for player_id in range(num_players):
                if player_turns[player_id] == 0:
                    logger.warning(f"Player {player_id} never got a turn in {env_id}")
                    continue

                tokenization_result = tokenize_for_trainer(
                    tokenizer=self.tokenizer,
                    chat=player_messages[player_id],
                    train_on_all_assistant_turns=True,
                )

                # Calculate final score: actual rewards + bonus for wins
                win_bonus = 1.0 if player_outcomes[player_id] == 1.0 else 0.0
                final_score = player_rewards[player_id] + win_bonus

                player_scored_items[player_id] = ScoredDataItem(
                    messages=(
                        player_messages[player_id]
                        if self.config.include_messages
                        else None
                    ),
                    tokens=tokenization_result["tokens"],
                    masks=tokenization_result["masks"],
                    scores=final_score,
                    metadata={
                        "trajectory_idx": trajectory_idx,
                        "env_id": env_id,
                        "num_players": num_players,
                        "player_id": player_id,
                        "outcome": player_outcomes[player_id],
                        "total_reward": player_rewards[player_id],
                        "steps": step_count,
                        "player_turns": player_turns[player_id],
                    },
                )

                if self.config.debug_rollout_logging:
                    logger.info(
                        f"[Traj {trajectory_idx}] Player {player_id} Finished | steps={step_count} | "
                        f"outcome={player_outcomes[player_id]} | total_reward={player_rewards[player_id]} | "
                        f"tokens={len(player_scored_items[player_id]['tokens'])} | "
                        f"masks={len(player_scored_items[player_id]['masks'])} | "
                        f"messages={len(player_messages[player_id])}"
                    )

            return player_scored_items

        except Exception as e:
            logger.error(f"Trajectory {trajectory_idx} fatal error: {e}")
            logger.error(traceback.format_exc())
            return {}

    async def _get_opponent_action(
        self, observation: str, player_id: int, trajectory_idx: int
    ) -> str:
        """Get action from an opponent agent using SGLang servers."""
        async with self.server.dedicated_server() as server:
            try:
                opponent_system_prompt = (
                    "You are playing a text-based game. "
                    "Read the game state and respond with only your action. "
                    "Be strategic and try to win."
                )

                opponent_messages = [
                    {"role": "system", "content": opponent_system_prompt},
                    {"role": "user", "content": observation},
                ]

                logger.warning(f"[Opponent {player_id}] Sending messages to server:")
                for i, msg in enumerate(opponent_messages):
                    logger.warning(
                        f"  Message {i}: role={msg['role']}, content={msg['content'][:200]}..."
                    )

                response = await server.chat_completion(
                    messages=opponent_messages,
                    n=1,
                    max_tokens=self.config.opponent_max_tokens,
                    temperature=self.config.opponent_temperature,
                )
                # handling for weird errors when observations are empty
                if response is None:
                    logger.error("Opponent got None response object from SGLang")
                    raise RuntimeError(
                        "SGLang server returned None response for opponent"
                    )

                if not hasattr(response, "choices") or not response.choices:
                    logger.error(f"Opponent response has no choices: {response}")
                    raise RuntimeError(
                        f"SGLang response has no choices for opponent: {response}"
                    )

                if response.choices[0].message.content is not None:
                    return response.choices[0].message.content.strip()
                else:
                    logger.error(
                        f"Opponent got None content from SGLang. Full response: {response}"
                    )
                    raise RuntimeError("SGLang returned None content for opponent")
            except Exception as e:
                logger.error(f"Opponent SGLang CRITICAL error: {e}")
                logger.error(f"Error type: {type(e).__name__}")
                logger.error(f"Server info: {server}")
                raise

    def _parse_step_result(self, step_result: Any) -> Tuple[bool, Dict[str, Any]]:
        """Handle different step return signatures and normalize to (done, info)."""
        # Common: (done, info)
        if isinstance(step_result, tuple):
            if len(step_result) == 2 and isinstance(step_result[0], (bool, int)):
                done = bool(step_result[0])
                info = step_result[1] if isinstance(step_result[1], dict) else {}
                return done, info
            if len(step_result) == 3:
                # (obs, done, info)
                done = bool(step_result[1])
                info = step_result[2] if isinstance(step_result[2], dict) else {}
                return done, info
        if isinstance(step_result, bool):
            return bool(step_result), {}
        return False, {}

    @classmethod
    def config_init(cls) -> Tuple[TextArenaEnvMinimalConfig, List[APIServerConfig]]:
        """Initialize default configuration."""
        env_config = TextArenaEnvMinimalConfig(
            tokenizer_name="NousResearch/Hermes-4-Qwen3-14B-1-e3",
            group_size=4,
            batch_size=8,
            use_wandb=True,
            wandb_name=cls.name,
            max_token_length=8192,
            total_steps=100,
            game_filter="all",
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

    async def evaluate(self, num_items: int) -> Dict[str, Any]:
        """Evaluate the model - simplified version."""
        logger.warning(
            "Evaluation not fully implemented in minimal TextArena environment"
        )
        return {"message": "Evaluation not implemented"}

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Log episode statistics to wandb."""
        if wandb_metrics is None:
            wandb_metrics = {}

        # Log training episode outcomes
        if self.episode_outcomes_buffer:
            wins = sum(1 for outcome in self.episode_outcomes_buffer if outcome > 0)
            losses = sum(1 for outcome in self.episode_outcomes_buffer if outcome < 0)
            draws = sum(1 for outcome in self.episode_outcomes_buffer if outcome == 0)
            total_episodes = len(self.episode_outcomes_buffer)

            win_rate = (wins / total_episodes) * 100 if total_episodes > 0 else 0.0
            loss_rate = (losses / total_episodes) * 100 if total_episodes > 0 else 0.0
            draw_rate = (draws / total_episodes) * 100 if total_episodes > 0 else 0.0

            avg_steps = (
                sum(self.episode_steps_buffer) / len(self.episode_steps_buffer)
                if self.episode_steps_buffer
                else 0
            )
            avg_outcome = (
                sum(self.episode_outcomes_buffer) / len(self.episode_outcomes_buffer)
                if self.episode_outcomes_buffer
                else 0
            )
            avg_reward = (
                sum(self.episode_rewards_buffer) / len(self.episode_rewards_buffer)
                if self.episode_rewards_buffer
                else 0
            )

            wandb_metrics[f"{self.name}/train/total_episodes"] = total_episodes
            wandb_metrics[f"{self.name}/train/win_rate_percent"] = win_rate
            wandb_metrics[f"{self.name}/train/loss_rate_percent"] = loss_rate
            wandb_metrics[f"{self.name}/train/draw_rate_percent"] = draw_rate
            wandb_metrics[f"{self.name}/train/avg_episode_steps"] = avg_steps
            wandb_metrics[f"{self.name}/train/avg_outcome"] = avg_outcome
            wandb_metrics[f"{self.name}/train/avg_reward_score"] = avg_reward

            # Per game type stats
            game_type_stats = {}
            for outcome, game_type in zip(
                self.episode_outcomes_buffer, self.episode_game_types
            ):
                if game_type not in game_type_stats:
                    game_type_stats[game_type] = {
                        "wins": 0,
                        "losses": 0,
                        "draws": 0,
                        "total": 0,
                    }

                game_type_stats[game_type]["total"] += 1
                if outcome > 0:
                    game_type_stats[game_type]["wins"] += 1
                elif outcome < 0:
                    game_type_stats[game_type]["losses"] += 1
                else:
                    game_type_stats[game_type]["draws"] += 1

            for game_type, stats in game_type_stats.items():
                total = stats["total"]
                wandb_metrics[f"{self.name}/train/{game_type}/episodes_count"] = total
                wandb_metrics[f"{self.name}/train/{game_type}/win_rate_percent"] = (
                    (stats["wins"] / total) * 100 if total > 0 else 0
                )
                wandb_metrics[f"{self.name}/train/{game_type}/loss_rate_percent"] = (
                    (stats["losses"] / total) * 100 if total > 0 else 0
                )
                wandb_metrics[f"{self.name}/train/{game_type}/draw_rate_percent"] = (
                    (stats["draws"] / total) * 100 if total > 0 else 0
                )

            self.episode_outcomes_buffer = []
            self.episode_rewards_buffer = []
            self.episode_steps_buffer = []
            self.episode_game_types = []

        for key, value in self.eval_metrics_custom:
            wandb_metrics[key] = value
        self.eval_metrics_custom = []

        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    TextArenaEnvMinimal.cli()
