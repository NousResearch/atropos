#!/usr/bin/env python3
"""
TextWorldEnv: Minimalist trainer environment for Microsoft TextWorld

A simple trainer environment that wraps TextWorld game generator and Gym interface
to train LLMs. The LLM outputs actions in plain text and receives only environment rewards.
No thinking tokens, memory, format rewards, or complex scoring - just pure environment interaction.
"""

import logging
import random
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
    max_steps: int = 20  # max steps per episode
    max_token_length: int = 8192

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
        super().__init__(config, server_configs, slurm, testing)
        self.config: TextWorldEnvConfig = config
        self.challenge_registry = None

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
        logger.info(f"Setting up {self.name} environment.")

        # Import registry creation from local module
        from environments.game_environments.textworld_env.textworld_registry import (
            create_textworld_registry,
        )

        self.challenge_registry = create_textworld_registry()
        logger.info(
            f"Initialized TextWorld challenge registry with challenges: {self.config.challenge_names}"
        )

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

    def _create_game(self, challenge_name: str, settings: Dict[str, Any]) -> Any:
        """Create a TextWorld game based on challenge name and settings."""
        # Create default options
        options = textworld.GameOptions()
        options.seeds = settings.get('seed', random.randint(0, 1000000))
        
        if challenge_name == "tw-simple":
            # Simple challenge expects rewards, goal, test settings
            game_settings = {
                "rewards": settings.get('rewards', 'balanced'),
                "goal": settings.get('goal', 'detailed'),
                "test": str(settings.get('test', False)).lower()
            }
            game = textworld.challenges.simple.make(game_settings, options=options)
        elif challenge_name == "tw-cooking":
            # Cooking challenge expects "recipe[1-4]+take[1-6]+open+drop+go[1-12]"
            recipe = settings.get('recipe', 1)
            take = settings.get('take', 2)
            go = settings.get('go', 1)
            game_settings = {"recipe": f"recipe{recipe}+take{take}+open+drop+go{go}"}
            game = textworld.challenges.cooking.make(game_settings, options=options)
        elif challenge_name == "tw-coin_collector":
            # Coin collector expects level as integer
            game_settings = {"level": settings.get('level', 1)}
            game = textworld.challenges.coin_collector.make(game_settings, options=options)
        elif challenge_name == "tw-treasure_hunter":
            # Treasure hunter expects level as integer  
            game_settings = {"level": settings.get('level', 1)}
            game = textworld.challenges.treasure_hunter.make(game_settings, options=options)
        else:
            raise ValueError(f"Unknown challenge: {challenge_name}")
            
        return game

    async def collect_trajectories(
        self, item: Item
    ) -> Tuple[ScoredDataGroup, List[Item]]:
        """Collect parallel trajectories from the same game."""
        # Get challenge info
        challenge_name = item["challenge_name"]
        settings = item["settings"]

        # Create the game using our helper method
        game = self._create_game(challenge_name, settings)

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
        env_id = textworld.gym.register_game(game, request_infos)

        # Collect trajectories in parallel
        scored_items = []

        for i in range(self.config.group_size):
            try:
                scored_item = await self._collect_single_trajectory(env_id, i)
                if scored_item:
                    scored_items.append(scored_item)
            except Exception as e:
                logger.error(f"Error collecting trajectory {i}: {e}")
                continue

        if not scored_items:
            logger.error("No successful trajectories collected")
            return ScoredDataGroup(scored_data_items=[]), []

        # Create ScoredDataGroup with all trajectories
        sdg = ScoredDataGroup(
            scored_data_items=scored_items,
            metadata={
                "game_type": item.get("game_type", "unknown"),
                "difficulty": item.get("difficulty", "unknown"),
                "num_trajectories": len(scored_items),
            },
        )

        return sdg, []

    async def _collect_single_trajectory(
        self, env_id: str, trajectory_idx: int
    ) -> Optional[ScoredDataItem]:
        """Collect a single trajectory for the game."""
        messages: List[Message] = []

        try:
            # Create and reset environment
            env = gym.make(env_id)
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
                        self.tokenizer.apply_chat_template(messages, tokenize=False)
                    )
                    if current_tokens > self.config.max_token_length - 500:
                        logger.warning(
                            f"Trajectory {trajectory_idx}: Approaching token limit, ending episode"
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
            tokenizer_name="NousResearch/Hermes-3-Llama-3.1-8B",
            group_size=16,
            use_wandb=True,
            wandb_name=cls.name,
            max_token_length=8192,
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
                model_name="NousResearch/Hermes-3-Llama-3.1-8B",
                base_url="http://localhost:8001/v1",
                api_key="x",
                num_requests_for_eval=128,
            ),
        ]

        return env_config, server_configs

    async def evaluate(self, num_items: int) -> Dict[str, Any]:
        """Evaluate the model - not implemented for this minimal environment."""
        logger.warning("Evaluation not implemented in minimal TextWorld environment")
        return {"message": "Evaluation not implemented"}
