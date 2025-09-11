#!/usr/bin/env python3
"""
TextArena GenRM: LLM best-of-n with LLM-as-judge and self-play (multi-agent)

This environment mirrors the GenRM approach used in TextWorld/Factorio:
- Generate multiple alternative actions per decision point (best-of-n)
- Score alternatives with an LLM judge (consensus ranking)
- Select the best alternative to advance the environment
- Track separate trajectories per player in multiplayer games (self-play)
- Return all ScoredDataGroups across all players flattened in a single list (for trainer)
- Save each trajectory (per player) as one line in a JSONL file with reconstructable metadata
"""

import asyncio
import json
import logging
import os
import random
import traceback
import uuid
from datetime import datetime
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


class TextArenaEnvGenRMConfig(BaseEnvConfig):
    """Configuration for the TextArena GenRM environment."""

    env_name: str = "textarena_genrm"
    wandb_name: str = "textarena-genrm"

    # Best-of-n generation
    group_size: int = 8
    max_num_workers: int = 16
    total_steps: int = 500
    max_steps: int = 40
    max_token_length: int = 16384

    # Judge configuration
    judge_model_name: str = "Hermes-4-405B"  # Use large judge by default
    judge_group_size: int = 3
    judge_temperature: float = 0.7
    use_same_model_for_judge: bool = False

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

    # Data saving: one trajectory per JSONL line
    trajectories_output_path: Optional[str] = None

    # Debugging
    debug_rollout_logging: bool = False


class TextArenaEnvGenRM(BaseEnv):
    """TextArena GenRM environment supporting multi-agent self-play and LLM judging."""

    name = "textarena_genrm"
    env_config_cls = TextArenaEnvGenRMConfig

    _registry = None

    def __init__(
        self,
        config: TextArenaEnvGenRMConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = True,
        testing: bool = False,
    ):
        # If we're configured to use Hermes-405B as the judge explicitly, optionally ensure
        # a server config exists for judge usage. Primary generation still uses self.server.
        if config.judge_model_name == "Hermes-4-405B":
            api_key = os.getenv("NOUS_API_KEY") or os.getenv("HERMES_API_KEY")
            if api_key:
                # We don't override generation servers here; judge flow uses direct HTTP
                pass
        super().__init__(config, server_configs, slurm, testing)
        self.config: TextArenaEnvGenRMConfig = config
        self.registry = None

        # Tracking buffers for wandb
        self.episode_outcomes_buffer: List[float] = []
        self.episode_rewards_buffer: List[float] = []
        self.episode_steps_buffer: List[int] = []
        self.episode_game_types: List[str] = []
        self.eval_metrics_custom: List[Tuple[str, float]] = []

        # Simple instruction to constrain outputs to actions
        self.system_prompt = (
            "You are an expert TextArena game player.\n"
            "Respond with ONLY the action to take, no explanations.\n"
            "Use the exact format specified by the game.\n"
        )

    async def setup(self):
        """Initialize the environment and TextArena registry."""
        try:
            if TextArenaEnvGenRM._registry is None:
                logger.info("Creating TextArena registry singleton...")
                TextArenaEnvGenRM._registry = create_textarena_registry(
                    seed=None,
                    max_player_detection=self.config.max_player_detection,
                    cache_path=self.config.textarena_registry_cache_path,
                    use_cache=self.config.use_textarena_registry_cache,
                    reset_cache=self.config.reset_textarena_registry_cache,
                )
                TextArenaEnvGenRM._registry.discover_games()
            self.registry = TextArenaEnvGenRM._registry
            available_games = self.registry.list_available_games(
                self.config.game_filter
            )
            logger.warning(
                f"TextArena GenRM setup with {len(available_games)} games (filter: {self.config.game_filter})"
            )
        except Exception as e:
            logger.error(f"Failed to setup TextArena GenRM: {e}")
            logger.error(traceback.format_exc())
            raise

    async def get_next_item(self) -> Item:
        """Select a random game and number of players from the registry."""
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
            "game_type": metadata.get("game_type", "unknown"),
        }

    async def collect_trajectories(
        self, item: Item
    ) -> Tuple[List[ScoredDataGroup], List[Item]]:
        """Run a single TextArena game with self-play and return all step groups.

        Returns a flattened list of ScoredDataGroups across all players' trajectories
        so the trainer can consume a single list, mirroring online RL.
        """
        env_id = item["env_id"]
        num_players: int = item["num_players"]
        metadata: Dict[str, Any] = item["metadata"]

        logger.warning(
            f"[GenRM] Starting game {env_id} with {num_players} player(s)"
        )

        # Initialize per-player state
        player_messages: Dict[int, List[Message]] = {}
        for pid in range(num_players):
            player_messages[pid] = [{"role": "system", "content": self.system_prompt}]

        # Per-player trajectory storage
        player_step_groups: Dict[int, List[ScoredDataGroup]] = {pid: [] for pid in range(num_players)}
        player_selected_indices: Dict[int, List[int]] = {pid: [] for pid in range(num_players)}
        player_actions_taken: Dict[int, List[str]] = {pid: [] for pid in range(num_players)}

        # Track a unique game id so saved trajectories can be grouped
        game_uuid = str(uuid.uuid4())

        try:
            env = ta.make(env_id)
            try:
                env.reset(num_players=num_players)
            except TypeError:
                env.reset()

            # Initial observation (may be shared or per-turn)
            initial_obs = env.get_observation()
            if initial_obs is None:
                logger.error(f"Failed to get initial observation for {env_id}")
                return [], []

            done: bool = False
            step_count: int = 0
            # Track winner(s) if the env returns such info
            winners: Optional[List[int]] = None

            while not done and step_count < self.config.max_steps:
                step_count += 1

                # Identify current player
                if num_players == 1:
                    current_player = 0
                else:
                    current_player = (
                        env.state.current_player_id
                        if hasattr(env.state, "current_player_id")
                        else 0
                    )

                # Pull the latest observation text
                current_obs = env.get_observation()
                if current_obs is None:
                    logger.error(f"Observation is None at step {step_count}")
                    break
                if isinstance(current_obs, tuple) and len(current_obs) >= 2:
                    obs_text = str(current_obs[1])
                elif isinstance(current_obs, str):
                    obs_text = current_obs
                else:
                    obs_text = str(current_obs)
                if not obs_text or obs_text == "[]":
                    logger.error(f"Empty observation at step {step_count}")
                    break

                # Build the message context for the current player
                messages = list(player_messages[current_player])
                messages.append({"role": "user", "content": obs_text})

                # Generate alternatives
                alternatives = await self._generate_alternatives_for_step(messages, step_count)
                if not alternatives:
                    logger.warning(f"No alternatives generated at step {step_count}")
                    break

                # Score and package alternatives for this decision point
                step_group = await self._create_step_scored_group(
                    alternatives, messages, step_count
                )
                if step_group is None:
                    logger.warning(f"Failed to build step group at step {step_count}")
                    break

                # Select best alternative
                selected_idx = self._select_best_alternative_idx(alternatives)
                selected_alt = alternatives[selected_idx]
                action_text = selected_alt.get("parsed_action", "").strip()
                if not action_text:
                    # Fallback to raw response
                    action_text = selected_alt.get("response", "").strip()

                # Execute in env
                try:
                    step_result = env.step(action_text)
                except Exception as e:
                    logger.error(f"env.step failed for action '{action_text}': {e}")
                    break

                done, info = self._parse_step_result(step_result)

                # Record per-player decision
                player_step_groups[current_player].append(step_group)
                player_selected_indices[current_player].append(selected_idx)
                player_actions_taken[current_player].append(action_text)

                # Extend the player's conversation with the chosen action
                player_messages[current_player].append(
                    {"role": "assistant", "content": action_text}
                )

                # Track winners if provided at terminal
                if done and isinstance(info, dict):
                    if "winner" in info and isinstance(info["winner"], int):
                        winners = [int(info["winner"])]
                    elif "winners" in info and isinstance(info["winners"], list):
                        winners = [int(w) for w in info["winners"] if isinstance(w, (int, str))]

            # Apply terminal credit assignment based on winners
            if winners is None:
                winners = []
            for pid in range(num_players):
                self._apply_terminal_credit_assignment(
                    player_step_groups[pid], player_selected_indices[pid], pid in winners
                )

            # For wandb metrics, record per-game simple outcome as: 1 if any winner, else 0
            outcome_val = 1.0 if winners else 0.0
            self.episode_outcomes_buffer.append(outcome_val)
            self.episode_rewards_buffer.append(0.0)  # Not all games provide scalar rewards
            self.episode_steps_buffer.append(step_count)
            self.episode_game_types.append(item.get("game_type", "unknown"))

            # Save trajectories per player if configured
            if self.config.trajectories_output_path:
                for pid in range(num_players):
                    await self._save_single_player_trajectory(
                        output_path=self.config.trajectories_output_path,
                        game_uuid=game_uuid,
                        env_id=env_id,
                        num_players=num_players,
                        player_id=pid,
                        is_winner=pid in winners,
                        actions=player_actions_taken[pid],
                        step_groups=player_step_groups[pid],
                        selected_indices=player_selected_indices[pid],
                        metadata=metadata,
                    )

            # Flatten all players' step groups for the trainer
            flattened: List[ScoredDataGroup] = []
            for pid in range(num_players):
                flattened.extend(player_step_groups[pid])

            return flattened, []

        except Exception as e:
            logger.error(f"Fatal error during TextArena GenRM rollout: {e}")
            logger.error(traceback.format_exc())
            return [], []

    async def _generate_alternatives_for_step(
        self, messages: List[Message], turn: int
    ) -> List[Dict[str, Any]]:
        """Generate exactly group_size alternative actions for the current step."""
        alternatives: List[Dict[str, Any]] = []
        try:
            # Ask for direct action-only outputs
            response = await self.server.chat_completion(
                messages=messages,
                n=self.config.group_size,
                max_tokens=64,
                temperature=0.8,
                top_p=0.95,
            )
            for i, choice in enumerate(response.choices):
                try:
                    content = choice.message.content.strip()
                    # Some models add quotes or explanations; keep only first line
                    action_text = content.splitlines()[0].strip()
                    alternatives.append(
                        {
                            "response": action_text,
                            "parsed_action": action_text,
                            "logprobs": getattr(choice, "logprobs", None),
                            "index": i,
                        }
                    )
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"Turn {turn}: Failed to generate alternatives: {e}")
            return []
        return alternatives

    async def _create_step_scored_group(
        self, alternatives: List[Dict[str, Any]], messages: List[Message], turn: int
    ) -> Optional[ScoredDataGroup]:
        """Tokenize alternatives and score them with an LLM judge."""
        if not alternatives:
            return None

        tokens_list: List[List[int]] = []
        masks_list: List[List[int]] = []
        messages_list: Optional[List[List[Message]]] = [] if self.config.include_messages else None

        # Tokenize each alternative appended as assistant turn
        for alt in alternatives:
            alt_messages = list(messages)
            alt_messages.append({"role": "assistant", "content": alt["response"]})
            tokenization_result = tokenize_for_trainer(
                tokenizer=self.tokenizer,
                chat=alt_messages,
                train_on_all_assistant_turns=True,
            )
            tokens_list.append(tokenization_result["tokens"])
            masks_list.append(tokenization_result["masks"])
            if messages_list is not None:
                messages_list.append(alt_messages)

        # Score with judge
        judge_scores = await self._score_actions_with_llm_judge(alternatives, messages, turn)
        scores_list: List[float] = [0.0] * len(alternatives)

        # Add a mild length regularizer to encourage concise action text
        lengths = [len(toks) for toks in tokens_list]
        avg_len = sum(lengths) / len(lengths) if lengths else 1.0
        for i in range(len(alternatives)):
            length_penalty = 0.2 * (avg_len - lengths[i]) / max(1.0, avg_len)
            scores_list[i] = max(0.0, min(1.0, judge_scores[i] + length_penalty))

        return ScoredDataGroup(
            tokens=tokens_list,
            masks=masks_list,
            scores=scores_list,
            messages=messages_list,
            advantages=None,
            ref_logprobs=None,
            group_overrides={},
            overrides=None,
            images=None,
        )

    async def _score_actions_with_llm_judge(
        self, alternatives: List[Dict[str, Any]], messages: List[Message], turn: int
    ) -> List[float]:
        """Rank alternatives with LLM-as-judge and return normalized scores in [0,1]."""
        actions = [alt.get("parsed_action", alt.get("response", "")) for alt in alternatives]
        context = messages[-1]["content"] if messages else ""
        judge_prompt = self._create_judge_prompt(context, actions)

        all_rankings: List[List[int]] = []
        for judge_idx in range(self.config.judge_group_size):
            ranking = await self._get_single_judge_ranking(judge_prompt, len(actions), judge_idx)
            if ranking:
                all_rankings.append(ranking)
        scores = self._calculate_consensus_scores(all_rankings, len(actions))
        logger.warning(
            f"Turn {turn}: Judge consensus scores: {[f'{s:.3f}' for s in scores]}"
        )
        return scores

    def _create_judge_prompt(self, context: str, actions: List[str]) -> str:
        prompt = (
            "You are evaluating candidate actions in a text-based game.\n\n"
            f"Current observation:\n{context}\n\n"
            "Actions to rank (best to worst):\n"
        )
        for i, a in enumerate(actions):
            prompt += f"\n{i+1}. {a}"
        prompt += (
            "\n\nRank as a comma-separated list of numbers (e.g., '3,1,4,2').\n"
            "Ranking:"
        )
        return prompt

    async def _get_single_judge_ranking(
        self, prompt: str, num_actions: int, judge_idx: int
    ) -> Optional[List[int]]:
        try:
            # If using Hermes-405B judge, call Nous API directly
            if self.config.judge_model_name == "Hermes-4-405B":
                import aiohttp

                api_key = os.getenv("NOUS_API_KEY") or os.getenv("HERMES_API_KEY")
                if not api_key:
                    logger.warning("Missing NOUS_API_KEY/HERMES_API_KEY for judge")
                    return None
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                payload = {
                    "model": "Hermes-4-405B",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 50,
                    "temperature": self.config.judge_temperature,
                }
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "https://inference-api.nousresearch.com/v1/chat/completions",
                        headers=headers,
                        json=payload,
                    ) as resp:
                        if resp.status != 200:
                            logger.warning(f"Hermes-405B judge HTTP {resp.status}")
                            return None
                        result = await resp.json()
                        response = result["choices"][0]["message"]["content"]
            else:
                # Use local servers for judging as well
                response_obj = await self.server.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    n=1,
                    max_tokens=50,
                    temperature=self.config.judge_temperature,
                )
                response = response_obj.choices[0].message.content

            ranking_text = (response or "").strip()
            ranking = self._parse_ranking(ranking_text, num_actions)
            return ranking
        except Exception as e:
            logger.warning(f"Judge {judge_idx} failed: {e}")
            return None

    def _parse_ranking(self, ranking_text: str, num_actions: int) -> Optional[List[int]]:
        try:
            import re
            numbers = re.findall(r"\d+", ranking_text)
            if not numbers:
                return None
            ranking = [int(n) - 1 for n in numbers]
            if len(ranking) != num_actions:
                return None
            if set(ranking) != set(range(num_actions)):
                return None
            return ranking
        except Exception:
            return None

    def _calculate_consensus_scores(
        self, all_rankings: List[List[int]], num_actions: int
    ) -> List[float]:
        if not all_rankings:
            return [0.5] * num_actions
        scores = [0.0] * num_actions
        for ranking in all_rankings:
            for position, idx in enumerate(ranking):
                scores[idx] += num_actions - position
        max_possible = num_actions * len(all_rankings)
        if max_possible > 0:
            scores = [s / max_possible for s in scores]
        return scores

    def _select_best_alternative_idx(self, alternatives: List[Dict[str, Any]]) -> int:
        if not alternatives:
            raise ValueError("No alternatives to select from")
        best_idx = 0
        best_score = float("-inf")
        for i, alt in enumerate(alternatives):
            score = alt.get("judge_score", 0.0)
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    def _apply_terminal_credit_assignment(
        self,
        step_groups: List[ScoredDataGroup],
        selected_indices: List[int],
        is_winner: bool,
    ) -> None:
        """Simple terminal credit assignment using discounted terminal outcome.

        If the player won, add a small positive bonus to selected alternatives across steps,
        otherwise a small negative penalty. This mirrors GenRM credit assignment patterns
        without relying on per-step numeric rewards.
        """
        if not step_groups or not selected_indices:
            return
        gamma = 0.99
        terminal = 1.0 if is_winner else -1.0
        T = len(step_groups)
        for t in range(T):
            ret = terminal * (gamma ** (T - 1 - t))
            sdg = step_groups[t]
            chosen = selected_indices[t] if t < len(selected_indices) else 0
            if 0 <= chosen < len(sdg["scores"]):
                sdg["scores"][chosen] = float(sdg["scores"][chosen]) + 0.1 * ret

    async def _save_single_player_trajectory(
        self,
        output_path: str,
        game_uuid: str,
        env_id: str,
        num_players: int,
        player_id: int,
        is_winner: bool,
        actions: List[str],
        step_groups: List[ScoredDataGroup],
        selected_indices: List[int],
        metadata: Dict[str, Any],
    ) -> None:
        """Append a single player's trajectory as one JSONL line with metadata."""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # Build compact per-step info (omit tokens/masks)
            steps: List[Dict[str, Any]] = []
            for i, sdg in enumerate(step_groups):
                step_info: Dict[str, Any] = {
                    "turn": i + 1,
                    "selected_index": selected_indices[i] if i < len(selected_indices) else 0,
                    "selected_action": actions[i] if i < len(actions) else "",
                    "scores": list(sdg.get("scores", [])),
                }
                if self.config.include_messages and sdg.get("messages") is not None:
                    # Only include assistant message text for each alternative
                    msgs = sdg.get("messages") or []
                    alts = []
                    for alt_msgs in msgs:
                        if alt_msgs and alt_msgs[-1]["role"] == "assistant":
                            alts.append(alt_msgs[-1]["content"])
                        else:
                            alts.append("")
                    step_info["alternatives"] = alts
                steps.append(step_info)

            payload = {
                "schema": 1,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "game_uuid": game_uuid,
                "env_id": env_id,
                "num_players": num_players,
                "player_id": player_id,
                "winner": bool(is_winner),
                "metadata": metadata,
                "steps": steps,
            }

            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception as e:
            logger.error(f"Failed to save player trajectory: {e}")

    def _parse_step_result(self, step_result: Any) -> Tuple[bool, Dict[str, Any]]:
        """Normalize TextArena step result to (done, info)."""
        if isinstance(step_result, tuple):
            if len(step_result) == 2 and isinstance(step_result[0], (bool, int)):
                done = bool(step_result[0])
                info = step_result[1] if isinstance(step_result[1], dict) else {}
                return done, info
            if len(step_result) == 3:
                done = bool(step_result[1])
                info = step_result[2] if isinstance(step_result[2], dict) else {}
                return done, info
        if isinstance(step_result, bool):
            return bool(step_result), {}
        return False, {}

    @classmethod
    def config_init(cls) -> Tuple[TextArenaEnvGenRMConfig, List[APIServerConfig]]:
        """Initialize a sensible default configuration for local development."""
        env_config = TextArenaEnvGenRMConfig(
            tokenizer_name="NousResearch/Hermes-4-Qwen3-14B-1-e3",
            group_size=8,
            use_wandb=True,
            wandb_name=cls.name,
            max_token_length=16384,
            total_steps=200,
            game_filter="all",
        )
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/Hermes-4-Qwen3-14B-1-e3",
                base_url="http://localhost:9004/v1",
                api_key="x",
                num_requests_for_eval=64,
            ),
            APIServerConfig(
                model_name="NousResearch/Hermes-4-Qwen3-14B-1-e3",
                base_url="http://localhost:9005/v1",
                api_key="x",
                num_requests_for_eval=64,
            ),
        ]
        return env_config, server_configs

    async def evaluate(self, num_items: int) -> Dict[str, Any]:
        logger.warning("Evaluation not implemented in TextArena GenRM environment")
        return {"message": "Evaluation not implemented"}

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}
        if self.episode_outcomes_buffer:
            wins = sum(1 for x in self.episode_outcomes_buffer if x > 0)
            losses = sum(1 for x in self.episode_outcomes_buffer if x < 0)
            draws = sum(1 for x in self.episode_outcomes_buffer if x == 0)
            total = len(self.episode_outcomes_buffer)
            wandb_metrics[f"{self.name}/train/total_episodes"] = total
            wandb_metrics[f"{self.name}/train/win_rate_percent"] = (wins / total) * 100 if total else 0
            wandb_metrics[f"{self.name}/train/loss_rate_percent"] = (losses / total) * 100 if total else 0
            wandb_metrics[f"{self.name}/train/draw_rate_percent"] = (draws / total) * 100 if total else 0
            wandb_metrics[f"{self.name}/train/avg_episode_steps"] = (
                sum(self.episode_steps_buffer) / len(self.episode_steps_buffer)
                if self.episode_steps_buffer
                else 0
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
    TextArenaEnvGenRM.cli()
