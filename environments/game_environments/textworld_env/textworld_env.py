#!/usr/bin/env python3
"""
TextWorldEnv: Minimalist trainer environment for Microsoft TextWorld

A simple trainer environment that wraps TextWorld game generator and Gym interface
to train LLMs. The LLM outputs actions in plain text and receives only environment rewards.
No thinking tokens, memory, format rewards, or complex scoring - just pure environment interaction.
"""

print("TEXTWORLD_ENV: Script is being executed", flush=True)

import logging
import os
import random
import tempfile
import traceback
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import textworld
import textworld.challenges
import textworld.gym

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
    ScoredDataItem,
)
from atroposlib.type_definitions import Item, Message
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

logger = logging.getLogger(__name__)


class TextWorldEnvConfig(BaseEnvConfig):
    """Configuration for the minimalist TextWorld environment trainer."""

    env_name: str = "TextWorld"
    wandb_name: str = "textworld-trainer-minimal"
    group_size: int = 16
    max_num_workers: int = 16
    total_steps: int = 500
    max_steps: int = 300  # max steps per episode (matches coin_collector max)
    max_token_length: int = 32768

    # Challenge settings
    challenge_names: List[str] = [
        "tw-simple",
        "tw-cooking",
        "tw-coin_collector",
        "tw-treasure_hunter",
    ]
    randomize_challenge_settings: bool = (
        True  # Randomize settings within each challenge
    )


class TextWorldEnv(BaseEnv):
    """Minimalist TextWorld environment for training LLMs."""

    name = "textworld_minimal"
    env_config_cls = TextWorldEnvConfig

    def __init__(
        self,
        config: TextWorldEnvConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = True,
        testing: bool = False,
    ):
        logger.warning(f"Initializing TextWorldEnv with config: {config}")
        logger.warning(f"Server configs: {server_configs}")
        logger.warning(f"Slurm: {slurm}, Testing: {testing}")
        super().__init__(config, server_configs, slurm, testing)
        self.config: TextWorldEnvConfig = config
        self.challenge_registry = None
        logger.warning("TextWorldEnv initialized successfully")

        # Track generated game files for cleanup
        self._generated_files = set()

        # Create temp directory for game files
        self._temp_dir = tempfile.mkdtemp(prefix="textworld_minimal_")
        logger.warning(f"Created temp directory for game files: {self._temp_dir}")
        
        # Tracking buffers for wandb logging
        self.episode_outcomes_buffer = []  # Store episode outcomes (-1, 0, 1)
        self.episode_rewards_buffer = []  # Store actual reward scores
        self.episode_steps_buffer = []  # Store number of steps per episode  
        self.episode_challenge_types = []  # Track which challenges were played
        self.eval_metrics_custom = []

        # Simple system prompt - just ask for actions
        self.system_prompt = (
            "You are playing a text-based adventure game. "
            "Read the game state and respond with ONLY the action you want to take. "
            "Do not include any explanation or reasoning, just the action command itself.\n\n"
            "Examples of valid actions:\n"
            "- go north\n"
            "- take key\n"
            "- open door\n"
            "- examine table\n"
            "- inventory\n\n"
            "Respond with a single action only."
        )

    async def setup(self):
        """Initialize the environment and challenge registry."""
        logger.warning(f"Starting TextWorldEnv setup...")
        logger.warning(f"Setting up {self.name} environment.")

        # Import registry creation from local module
        from environments.game_environments.textworld_env.textworld_registry import (
            create_textworld_registry,
        )

        try:
            self.challenge_registry = create_textworld_registry()
            logger.warning(
                f"Initialized TextWorld challenge registry with challenges: {self.config.challenge_names}"
            )
            logger.warning("TextWorldEnv setup completed successfully")
        except Exception as e:
            logger.error(f"Failed to create TextWorld registry: {e}")
            logger.error(traceback.format_exc())
            raise

    async def get_next_item(self) -> Item:
        """Get the next game configuration."""
        # Randomly select a challenge
        if len(self.config.challenge_names) == 1:
            challenge_name = self.config.challenge_names[0]
        else:
            challenge_name = random.choice(self.config.challenge_names)

        # Get challenge settings
        challenge_name, settings = self.challenge_registry.get_challenge(
            challenge_name, randomize_settings=self.config.randomize_challenge_settings
        )

        return {
            "challenge_name": challenge_name,
            "settings": settings,
            "game_type": "challenge",
        }

    def _create_game(self, challenge_name: str, settings: Dict[str, Any]) -> str:
        """Create a TextWorld game and save it to a file.

        Returns:
            Path to the saved game file (.z8)
        """
        # Create default options
        options = textworld.GameOptions()
        options.seeds = settings.get("seed", random.randint(0, 1000000))

        if challenge_name == "tw-simple":
            # Simple challenge expects rewards, goal, test settings
            game_settings = {
                "rewards": settings.get("rewards", "balanced"),
                "goal": settings.get("goal", "detailed"),
                "test": str(settings.get("test", False)).lower(),
            }
            game = textworld.challenges.simple.make(game_settings, options=options)
        elif challenge_name == "tw-cooking":
            # Cooking challenge expects individual parameters as integers/booleans
            game_settings = {
                "recipe": settings.get("recipe", 1),  # Number of ingredients
                "take": settings.get("take", 1),  # Number to find
                "cook": settings.get("cook", False),  # Whether to cook
                "open": settings.get("open", False),  # Whether to open containers
                "drop": settings.get("drop", False),  # Whether limited inventory
                "go": settings.get("go", 1),  # Number of locations
                "recipe_seed": settings.get(
                    "recipe-seed",
                    settings.get("recipe_seed", random.randint(0, 1000000)),
                ),
                "split": "train",  # cooking challenge expects this
            }
            logger.debug(f"Cooking game settings: {game_settings}")
            game = textworld.challenges.cooking.make(game_settings, options=options)
        elif challenge_name == "tw-coin_collector":
            # Coin collector expects level as integer
            game_settings = {"level": settings.get("level", 1)}
            game = textworld.challenges.coin_collector.make(
                game_settings, options=options
            )
        elif challenge_name == "tw-treasure_hunter":
            # Treasure hunter expects level as integer
            game_settings = {"level": settings.get("level", 1)}
            game = textworld.challenges.treasure_hunter.make(
                game_settings, options=options
            )
        else:
            raise ValueError(f"Unknown challenge: {challenge_name}")

        # Save game to file
        game_file = os.path.join(
            self._temp_dir,
            f"{challenge_name}_{settings.get('seed', random.randint(0, 1000000))}.z8",
        )
        options.path = game_file
        options.file_ext = ".z8"
        game_file = textworld.generator.compile_game(game, options)

        # Track for cleanup
        self._generated_files.add(game_file)
        logger.debug(f"Generated game file: {game_file}")

        return game_file

    async def collect_trajectories(
        self, item: Item
    ) -> Tuple[ScoredDataGroup, List[Item]]:
        """Collect parallel trajectories from the same game."""
        # Get challenge info
        challenge_name = item["challenge_name"]
        settings = item["settings"]

        # Create the game file using our helper method
        game_file = self._create_game(challenge_name, settings)

        # Register the game
        request_infos = textworld.EnvInfos(
            description=True,
            inventory=True,
            objective=True,
            admissible_commands=True,
            won=True,
            lost=True,
            score=True,
            moves=True,
            max_score=True,
        )
        env_id = textworld.gym.register_game(game_file, request_infos)

        # Collect trajectories in parallel
        scored_items = []

        for i in range(self.config.group_size):
            try:
                scored_item = await self._collect_single_trajectory(env_id, i, challenge_name)
                if scored_item:
                    scored_items.append(scored_item)
            except Exception as e:
                logger.error(f"Error collecting trajectory {i}: {e}")
                continue

        if not scored_items:
            logger.error("No successful trajectories collected")
            # Return empty ScoredDataGroup with correct structure
            return ScoredDataGroup(
                tokens=[],
                masks=[],
                scores=[],
                messages=[],
                advantages=None,
                ref_logprobs=None,
                group_overrides={},  # Should be empty dict, not None
                overrides=None,
                images=None,
            ), []

        # Create ScoredDataGroup by aggregating all trajectories
        sdg = ScoredDataGroup(
            tokens=[],
            masks=[],
            scores=[],
            messages=[],
            advantages=None,
            ref_logprobs=None,
            group_overrides={},  # Should be empty dict, not None
            overrides=None,
            images=None,
        )
        
        # Aggregate data from all scored items
        for scored_item in scored_items:
            sdg["tokens"].append(scored_item["tokens"])
            sdg["masks"].append(scored_item["masks"])
            sdg["scores"].append(scored_item["scores"])
            if self.config.include_messages and scored_item.get("messages"):
                sdg["messages"].append(scored_item["messages"])
                
            # Track outcomes for wandb logging
            metadata = scored_item.get("metadata", {})
            final_score = metadata.get("final_score", 0)
            won = metadata.get("won", False)
            lost = metadata.get("lost", False)
            moves = metadata.get("moves", 0)
            
            # Convert to outcome: 1 for win, -1 for loss, 0 for neither
            if won:
                outcome = 1.0
                logger.warning(f"Episode WON! Score: {final_score}, Moves: {moves}, Challenge: {metadata.get('challenge_name', 'unknown')}")
            elif lost:
                outcome = -1.0
                logger.warning(f"Episode LOST! Score: {final_score}, Moves: {moves}, Challenge: {metadata.get('challenge_name', 'unknown')}")
            else:
                outcome = 0.0
                logger.warning(f"Episode DRAW! Score: {final_score}, Moves: {moves}, Challenge: {metadata.get('challenge_name', 'unknown')}")
                
            self.episode_outcomes_buffer.append(outcome)
            self.episode_rewards_buffer.append(final_score)
            self.episode_steps_buffer.append(moves)
            self.episode_challenge_types.append(metadata.get("challenge_name", "unknown"))
        
        logger.warning(f"DEBUG: Created ScoredDataGroup with {len(sdg['tokens'])} trajectories")

        # Clean up the game file
        self._cleanup_game_file(game_file)

        return sdg, []

    async def _collect_single_trajectory(
        self, env_id: str, trajectory_idx: int, challenge_name: str = "unknown"
    ) -> Optional[ScoredDataItem]:
        """Collect a single trajectory for the game."""
        messages: List[Message] = []

        try:
            # Create and reset environment using TextWorld's gym
            env = textworld.gym.make(env_id)
            obs, info = env.reset()

            # Initial messages
            messages.append({"role": "system", "content": self.system_prompt})

            # Format initial observation
            obs_text = self._format_observation(obs, info)
            messages.append({"role": "user", "content": obs_text})

            done = False
            total_reward = 0.0

            # Use dedicated server for this trajectory
            async with self.server.dedicated_server() as server:
                while not done and len(messages) < self.config.max_steps * 2:
                    # Check token limit
                    current_tokens = len(
                        self.tokenizer.apply_chat_template(messages, tokenize=True)
                    )
                    if current_tokens > self.config.max_token_length - 500:
                        logger.warning(
                            f"Trajectory {trajectory_idx}: Approaching token limit, ending episode"
                        )
                        logger.info(f"Token usage: {current_tokens} tokens")
                        logger.info(f"Number of messages: {len(messages)}")
                        logger.info(
                            f"Last observation length: {len(messages[-1]['content']) if messages else 0}"
                        )
                        break

                    # Get action from LLM
                    try:
                        response = await server.chat_completion(
                            messages=messages,
                            n=1,
                            max_tokens=50,  # Actions should be short
                            temperature=0.7,
                        )
                        action = response.choices[0].message.content.strip()
                        logger.debug(f"Trajectory {trajectory_idx}: Action: {action}")
                    except Exception as e:
                        logger.error(f"Trajectory {trajectory_idx}: LLM error: {e}")
                        break

                    messages.append({"role": "assistant", "content": action})

                    # Execute action
                    try:
                        obs, reward, done, info = env.step(action)
                        total_reward += reward
                    except Exception as e:
                        logger.error(
                            f"Trajectory {trajectory_idx}: Environment error: {e}"
                        )
                        break

                    # Add observation if not done
                    if not done:
                        obs_text = self._format_observation(obs, info)
                        messages.append({"role": "user", "content": obs_text})

            env.close()

            # Tokenize for training
            tokenization_result = tokenize_for_trainer(
                tokenizer=self.tokenizer,
                chat=messages,
                train_on_all_assistant_turns=True,
            )

            # Create scored data item with only environment reward
            return ScoredDataItem(
                messages=messages if self.config.include_messages else None,
                tokens=tokenization_result["tokens"],
                masks=tokenization_result["masks"],
                scores=total_reward,
                metadata={
                    "trajectory_idx": trajectory_idx,
                    "final_score": total_reward,
                    "won": info.get("won", False),
                    "lost": info.get("lost", False),
                    "moves": info.get("moves", 0),
                    "challenge_name": challenge_name,
                },
            )

        except Exception as e:
            logger.error(f"Trajectory {trajectory_idx}: Fatal error: {e}")
            logger.error(traceback.format_exc())
            return None

    def _format_observation(self, obs: str, info: Dict[str, Any]) -> str:
        """Format game observation for the LLM."""
        parts = []

        # Main observation
        parts.append(obs)

        # Add objective if available
        if "objective" in info and info["objective"]:
            parts.append(f"\nObjective: {info['objective']}")

        # Add score info
        if "score" in info and "max_score" in info:
            parts.append(f"\nScore: {info['score']}/{info['max_score']}")

        # Add inventory if not empty
        if "inventory" in info and info["inventory"]:
            parts.append(f"\nInventory: {info['inventory']}")

        return "\n".join(parts)

    @classmethod
    def config_init(cls) -> Tuple[TextWorldEnvConfig, List[APIServerConfig]]:
        """Initialize default configuration."""
        env_config = TextWorldEnvConfig(
            tokenizer_name="NousResearch/Hermes-4-Qwen3-14B-1-e3",
            group_size=16,
            use_wandb=True,
            wandb_name=cls.name,
            max_token_length=32768,
            total_steps=500,
            challenge_names=[
                "tw-simple",
                "tw-cooking",
                "tw-coin_collector",
                "tw-treasure_hunter",
            ],
            randomize_challenge_settings=True,
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
        """Evaluate the model - not implemented for this minimal environment."""
        logger.warning("Evaluation not implemented in minimal TextWorld environment")
        return {"message": "Evaluation not implemented"}
    
    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Log episode statistics to wandb."""
        if wandb_metrics is None:
            wandb_metrics = {}
        
        # Log training episode outcomes
        if self.episode_outcomes_buffer:
            # Calculate overall statistics
            wins = sum(1 for outcome in self.episode_outcomes_buffer if outcome > 0)
            losses = sum(1 for outcome in self.episode_outcomes_buffer if outcome < 0)
            draws = sum(1 for outcome in self.episode_outcomes_buffer if outcome == 0)
            total_episodes = len(self.episode_outcomes_buffer)
            
            # Calculate rates (percentages)
            win_rate = (wins / total_episodes) * 100 if total_episodes > 0 else 0.0
            loss_rate = (losses / total_episodes) * 100 if total_episodes > 0 else 0.0
            draw_rate = (draws / total_episodes) * 100 if total_episodes > 0 else 0.0
            
            # Average metrics
            avg_steps = sum(self.episode_steps_buffer) / len(self.episode_steps_buffer) if self.episode_steps_buffer else 0
            avg_outcome = sum(self.episode_outcomes_buffer) / len(self.episode_outcomes_buffer) if self.episode_outcomes_buffer else 0
            avg_reward = sum(self.episode_rewards_buffer) / len(self.episode_rewards_buffer) if self.episode_rewards_buffer else 0
            max_reward = max(self.episode_rewards_buffer) if self.episode_rewards_buffer else 0
            min_reward = min(self.episode_rewards_buffer) if self.episode_rewards_buffer else 0
            
            # Log overall metrics with clear labels
            wandb_metrics[f"{self.name}/train/total_episodes"] = total_episodes
            wandb_metrics[f"{self.name}/train/win_count_absolute"] = wins
            wandb_metrics[f"{self.name}/train/loss_count_absolute"] = losses  
            wandb_metrics[f"{self.name}/train/draw_count_absolute"] = draws
            wandb_metrics[f"{self.name}/train/win_rate_percent"] = win_rate
            wandb_metrics[f"{self.name}/train/loss_rate_percent"] = loss_rate
            wandb_metrics[f"{self.name}/train/draw_rate_percent"] = draw_rate
            wandb_metrics[f"{self.name}/train/avg_episode_steps"] = avg_steps
            wandb_metrics[f"{self.name}/train/avg_outcome"] = avg_outcome  # -1, 0, 1 average
            wandb_metrics[f"{self.name}/train/avg_reward_score"] = avg_reward  # Actual game score
            wandb_metrics[f"{self.name}/train/max_reward_score"] = max_reward
            wandb_metrics[f"{self.name}/train/min_reward_score"] = min_reward
            
            # Per-challenge statistics
            challenge_stats = {}
            for i, (outcome, reward, steps, challenge) in enumerate(zip(
                self.episode_outcomes_buffer,
                self.episode_rewards_buffer,
                self.episode_steps_buffer, 
                self.episode_challenge_types
            )):
                if challenge not in challenge_stats:
                    challenge_stats[challenge] = {
                        'outcomes': [],
                        'rewards': [],
                        'steps': [],
                        'count': 0
                    }
                challenge_stats[challenge]['outcomes'].append(outcome)
                challenge_stats[challenge]['rewards'].append(reward)
                challenge_stats[challenge]['steps'].append(steps)
                challenge_stats[challenge]['count'] += 1
            
            # Log per-challenge metrics
            for challenge, stats in challenge_stats.items():
                challenge_wins = sum(1 for o in stats['outcomes'] if o > 0)
                challenge_losses = sum(1 for o in stats['outcomes'] if o < 0)
                challenge_draws = sum(1 for o in stats['outcomes'] if o == 0)
                challenge_total = stats['count']
                
                # Counts
                wandb_metrics[f"{self.name}/train/{challenge}/episodes_count"] = challenge_total
                wandb_metrics[f"{self.name}/train/{challenge}/wins_count"] = challenge_wins
                wandb_metrics[f"{self.name}/train/{challenge}/losses_count"] = challenge_losses
                wandb_metrics[f"{self.name}/train/{challenge}/draws_count"] = challenge_draws
                
                # Rates
                wandb_metrics[f"{self.name}/train/{challenge}/win_rate_percent"] = (challenge_wins / challenge_total) * 100 if challenge_total > 0 else 0
                wandb_metrics[f"{self.name}/train/{challenge}/loss_rate_percent"] = (challenge_losses / challenge_total) * 100 if challenge_total > 0 else 0
                wandb_metrics[f"{self.name}/train/{challenge}/draw_rate_percent"] = (challenge_draws / challenge_total) * 100 if challenge_total > 0 else 0
                
                # Average metrics
                wandb_metrics[f"{self.name}/train/{challenge}/avg_steps"] = sum(stats['steps']) / len(stats['steps']) if stats['steps'] else 0
                wandb_metrics[f"{self.name}/train/{challenge}/avg_outcome"] = sum(stats['outcomes']) / len(stats['outcomes']) if stats['outcomes'] else 0
                wandb_metrics[f"{self.name}/train/{challenge}/avg_reward_score"] = sum(stats['rewards']) / len(stats['rewards']) if stats['rewards'] else 0
                wandb_metrics[f"{self.name}/train/{challenge}/max_reward_score"] = max(stats['rewards']) if stats['rewards'] else 0
                wandb_metrics[f"{self.name}/train/{challenge}/min_reward_score"] = min(stats['rewards']) if stats['rewards'] else 0
            
            # Clear buffers
            self.episode_outcomes_buffer = []
            self.episode_rewards_buffer = []
            self.episode_steps_buffer = []
            self.episode_challenge_types = []
        
        # Log eval metrics if any
        for key, value in self.eval_metrics_custom:
            wandb_metrics[key] = value
        self.eval_metrics_custom = []
        
        # Call parent wandb_log
        await super().wandb_log(wandb_metrics)

    def _cleanup_game_file(self, game_file: str):
        """Clean up a generated game file and its associated files."""
        if game_file in self._generated_files:
            try:
                # Remove .z8 file
                if os.path.exists(game_file):
                    os.remove(game_file)
                    logger.debug(f"Removed game file: {game_file}")

                # Remove .ni file if it exists
                ni_file = game_file.replace(".z8", ".ni")
                if os.path.exists(ni_file):
                    os.remove(ni_file)
                    logger.debug(f"Removed ni file: {ni_file}")

                # Remove .json file if it exists
                json_file = game_file.replace(".z8", ".json")
                if os.path.exists(json_file):
                    os.remove(json_file)
                    logger.debug(f"Removed json file: {json_file}")

                self._generated_files.remove(game_file)
            except OSError as e:
                logger.warning(f"Failed to clean up game file {game_file}: {e}")

    async def cleanup(self):
        """Clean up all resources - called after each episode by base class."""
        # Don't clean up temp directory here since env is singleton
        # Just log that cleanup was called
        logger.debug(
            f"Cleanup called, {len(self._generated_files)} files still tracked"
        )

    def __del__(self):
        """Ensure cleanup on deletion - this is when we actually clean the temp dir."""
        # Clean up any remaining game files
        files_to_clean = list(self._generated_files)
        for game_file in files_to_clean:
            self._cleanup_game_file(game_file)

        # Clean up temp directory
        if hasattr(self, "_temp_dir") and os.path.exists(self._temp_dir):
            import shutil

            try:
                shutil.rmtree(self._temp_dir)
                logger.debug(f"Cleaned up temp directory: {self._temp_dir}")
            except Exception:
                pass


# Add main block to catch errors
if __name__ == "__main__":
    print("TEXTWORLD_ENV: Main block executing", flush=True)
    try:
        TextWorldEnv.cli()
    except Exception as e:
        print(f"TEXTWORLD_ENV: Fatal error in main: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise
