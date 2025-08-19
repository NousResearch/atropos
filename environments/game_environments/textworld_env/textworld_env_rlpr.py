#!/usr/bin/env python3
"""
TextWorldEnv: Trainer environment for Microsoft TextWorld with VR-CLI

A trainer environment that wraps TextWorld game generator and Gym interface
to train LLMs using best-of-n pattern with function-call style actions.
Uses VR-CLI (Verifiable Rewards via Completion Likelihood Improvement) to score
predictions based on how well the model anticipates action outcomes.
"""

import asyncio
import logging
import math
import os
import random
import traceback
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import textworld
import textworld.challenges
import textworld.gym
from pydantic import Field
from textworld.gym.envs import TextworldGymEnv

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.type_definitions import Message
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer
from atroposlib.utils.tokenizers.qwen_fixed_tokenizer import QwenFixedTokenizer
from atroposlib.utils.tool_call_parser import parse_tool_call
from environments.game_environments.agents.atropos_agent import (
    AtroposAgent,
    AtroposAgentConfig,
    _convert_messages_for_api,
)
from environments.game_environments.agents.atropos_memory_manager import (
    MEMORY_SYSTEM_PREREQUISITES_AVAILABLE,
    AtroposMemoryManager,
)
from environments.game_environments.textworld_env.scoring.entropy_calculator import (
    confidence_score,
)
from environments.game_environments.textworld_env.textworld_registry import (
    create_textworld_registry,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TextWorldEnvConfig(BaseEnvConfig):
    """Configuration for the TextWorld environment trainer with VR-CLI."""

    env_name: str = "TextWorld"
    wandb_name: str = "textworld-trainer"
    group_size: int = 16
    max_num_workers: int = 16
    total_steps: int = 500
    max_steps: int = 20  # prevent infinite loops
    # Challenge setup (align with minimal env)
    challenge_names: List[str] = [
        "tw-simple",
        "tw-cooking",
        "tw-coin_collector",
        "tw-treasure_hunter",
    ]
    randomize_challenge_settings: bool = True

    # Top-k/bottom-k selection
    use_topk_bottomk_selection: bool = True
    topk: int = 8
    bottomk: int = 8
    allow_variable_group_size: bool = True
    max_token_length: int = 16384
    max_trajectory_tokens: int = 32768

    # VR-CLI specific configurations
    vrcli_weight: float = Field(
        default=0.3,
        description="Weight for VR-CLI score in combined reward",
    )
    vrcli_discount_factor: float = Field(
        default=0.99,
        description="Discount factor for credit assignment in sparse reward setting",
    )

    # Format reward configuration
    format_reward_enabled: bool = Field(
        default=True,
        description="Apply format rewards for proper response structure (memory and thinking blocks)",
    )
    format_reward_weight: float = Field(
        default=0.1,
        description="Weight for format reward in combined scoring (0.1 = up to 10% adjustment)",
    )
    format_memory_reward: float = Field(
        default=0.05, description="Reward for including a proper <memory> block"
    )
    format_thinking_reward: float = Field(
        default=0.05, description="Reward for including a proper <think> block"
    )
    format_strict_structure: bool = Field(
        default=True,
        description="Enforce strict structure: exactly 1 think, 1 memory, 1 tool_call in order",
    )
    format_wrong_order_penalty: float = Field(
        default=0.5,
        description="Penalty multiplier for blocks in wrong order (0.5 = 50% penalty)",
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
        default=0.1,
        description="Weight for token length penalty (0.1 = up to 10% adjustment)",
    )
    token_length_baseline: int = Field(
        default=500,
        description="Baseline token count for neutral penalty (no bonus/penalty)",
    )
    token_length_penalty_scale: float = Field(
        default=0.0002,
        description="Scale factor for token length penalty (penalty per token over baseline)",
    )

    # Deprecated/unused fields from older variant removed

    default_server_config: APIServerConfig = Field(
        default_factory=lambda: APIServerConfig(
            server_type="openai", model_name="gpt-3.5-turbo"
        )
    )
    policy_agent_server_config: Optional[APIServerConfig] = None

    atropos_agent_config: AtroposAgentConfig = Field(default_factory=AtroposAgentConfig)

    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True


class TextWorldEpisodeState:
    """Stores per-episode state for a TextWorld game using VR-CLI."""

    def __init__(
        self,
        episode_id: str,
        game_file: str,
        textworld_env_instance: TextworldGymEnv,
        initial_obs: str,
        initial_infos: Dict[str, Any],
        max_steps: int,
    ):
        self.episode_id: str = episode_id
        self.game_file: str = game_file
        self.textworld_env: TextworldGymEnv = textworld_env_instance
        self.initial_formatted_obs: str = initial_obs
        self.initial_infos: Dict[str, Any] = initial_infos

        self.policy_step_data: List[ScoredDataGroup] = []

        self.cumulative_reward: float = 0.0
        self.max_turns: int = max_steps

        self.last_score: float = initial_infos.get("score", 0.0)
        self.moves: int = initial_infos.get("moves", 0)
        self.won: bool = False
        self.lost: bool = False
        self.done: bool = False
        self.last_env_raw_observation: Optional[str] = None
        self.last_env_infos: Optional[Dict[str, Any]] = None
        self.last_formatted_obs: str = initial_obs

        self.canonical_rewards: List[float] = []
        self.canonical_chosen_alternative_indices: List[int] = []

        # Episode-level cache for think block summarizations
        self.think_block_cache: Dict[int, str] = {}


class TextWorldEnv(BaseEnv):
    """Trainer environment for TextWorld using VR-CLI for action outcome prediction."""

    name = "TextWorld"  # Default name for wandb

    def __init__(
        self,
        config: TextWorldEnvConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = True,
        testing: bool = False,
    ):
        logger.info("Initializing TextWorldEnv with configuration:")
        logger.info(f"  - max_steps: {config.max_steps}")
        logger.info(f"  - group_size: {config.group_size}")
        logger.info(f"  - tokenizer: {config.tokenizer_name}")
        logger.info(f"  - rollout_server_url: {config.rollout_server_url}")
        logger.info(f"  - num_servers: {len(server_configs)}")
        logger.info(f"  - slurm: {slurm}")

        try:
            super().__init__(config, server_configs, slurm, testing)
        except Exception as e:
            logger.error(f"Exception in super().__init__: {type(e).__name__}: {e}")
            traceback.print_exc()
            raise
        self.config: TextWorldEnvConfig = config
        self.episodes: Dict[str, TextWorldEpisodeState] = {}

        # Wandb logging buffers
        self.completed_episodes_buffer = []
        self.step_metrics_buffer = []
        self.last_logged_episodes = 0

        # Override tokenizer for Qwen models with our fixed tokenizer
        if "qwen" in config.tokenizer_name.lower():
            logger.info(
                f"Detected Qwen model, using QwenFixedTokenizer for {config.tokenizer_name}"
            )
            self.tokenizer = QwenFixedTokenizer(config.tokenizer_name)
        # Track generated game files for cleanup and create a temp dir
        self._generated_files: set[str] = set()
        import tempfile

        self._temp_dir = tempfile.mkdtemp(prefix="textworld_rlpr_")

        # Memory manager will be created per-agent in _create_agent_for_episode()
        # Define TextWorld command execution tool with outcome prediction
        self.textworld_tools = [
            {
                "type": "function",
                "function": {
                    "name": "execute_command",
                    "description": "Execute a text command in the adventure game and predict the outcome.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The command to execute in the game.",
                            },
                            "expected_outcome": {
                                "type": "string",
                                "description": (
                                    "What you expect to observe after executing this command. "
                                    "Be specific about changes to the environment, your location, "
                                    "inventory, or game state."
                                ),
                            },
                        },
                        "required": ["command", "expected_outcome"],
                    },
                },
            }
        ]
        # Policy agent system prompt with outcome prediction
        constructed_system_prompt = (
            "You are an AI agent playing a text-based adventure game who uses extreme long chains of thought "
            "to carefully plan your actions and predict their outcomes. Your goal is to follow the objective "
            "described at the start of the game. You interact with the world by providing text commands and "
            "predicting their outcomes."
            "\\n\\n"
            "You should:\\n"
            "1. Enclose your thoughts and internal monologue inside <think> </think> tags. Use extremely "
            "long chains of thought to carefully consider the game state, your objectives, and the likely "
            "outcomes of your actions.\\n"
            "2. Generate a memory summary inside <memory> </memory> tags that captures key information from "
            "this turn. Your memory should:\\n"
            "   - Build upon previous memories shown in 'Relevant Memories' if present\\n"
            "   - Note the outcome of your last action (did it match your prediction?)\\n"
            "   - Update your understanding of the game state, location, and inventory\\n"
            "   - Track progress toward objectives and any multi-step plans\\n"
            "   - Be concise but comprehensive (1-3 sentences)\\n"
            "3. Provide your action using the execute_command function call."
            "\\n\\n"
            "For your function call, return a JSON object with function name and arguments "
            "within <tool_call> </tool_call> tags with the following schema:\\n"
            '<tool_call>\\n{"name": "execute_command", "arguments": {"command": "go north", '
            '"expected_outcome": "I move north to a new room"}}\\n</tool_call>\\n\\n'
            "EXAMPLE RESPONSE 1:\\n"
            "<think>\\n"
            "I'm in the kitchen. I see a stove and a fridge. The objective says to cook something. "
            "Let me check what's in the fridge first to see what ingredients are available."
            "\\n</think>\\n"
            "<memory>\\n"
            "Kitchen has stove and fridge. Main objective is cooking. Need to find ingredients."
            "\\n</memory>\\n"
            "<tool_call>\\n"
            '{"name": "execute_command", "arguments": {"command": "open fridge", '
            '"expected_outcome": "The fridge opens, revealing its contents. I expect to see various '
            'food items or ingredients inside that I can take and use for cooking."}}'
            "\\n</tool_call>\\n\\n"
            "EXAMPLE RESPONSE 2 (with previous memories):\\n"
            "<think>\\n"
            "Looking at my previous memories, I was exploring the kitchen to find cooking ingredients. "
            "I successfully opened the fridge and found eggs, milk, and flour. My goal is still to "
            "cook something. Now I need to take these ingredients and find a recipe or mixing bowl. "
            "The previous action of opening the fridge worked as expected."
            "\\n</think>\\n"
            "<memory>\\n"
            "Found eggs, milk, and flour in kitchen fridge. Still need mixing bowl or recipe to cook. "
            "Previous exploration of kitchen successful - have stove and ingredients located."
            "\\n</memory>\\n"
            "<tool_call>\\n"
            '{"name": "execute_command", "arguments": {"command": "take eggs", '
            '"expected_outcome": "I take the eggs from the fridge and add them to my inventory"}}'
            "\\n</tool_call>\\n\\n"
            "EXAMPLE RESPONSE 3:\\n"
            "<think>\\n"
            "There's a locked door here and I have a key in my inventory. I should try using the key "
            "on the door."
            "\\n</think>\\n"
            "<memory>\\n"
            "Found locked door in current room. Have key in inventory that might open it."
            "\\n</memory>\\n"
            "<tool_call>\\n"
            '{"name": "execute_command", "arguments": {"command": "unlock door with key", '
            '"expected_outcome": "The key turns in the lock and the door unlocks. I should now be '
            'able to open the door and go through it."}}'
            "\\n</tool_call>\\n\\n"
            "Remember: Your entire response must be exactly three XML blocks: <think>...</think> "
            "followed by <memory>...</memory> followed by <tool_call>...</tool_call>\\n\\n"
            "FINAL REMINDER: After your <think> block and <memory> block, you MUST wrap your JSON "
            "function call in <tool_call> tags. The JSON goes INSIDE the <tool_call> tags, not after them."
        )
        # Ensure AtroposAgentConfig is instantiated
        agent_cfg = (
            self.config.atropos_agent_config
            if self.config.atropos_agent_config is not None
            else AtroposAgentConfig()
        )
        agent_cfg.system_prompt = constructed_system_prompt
        if (
            self.config.policy_agent_server_config
            and self.config.policy_agent_server_config.model_name
        ):
            agent_cfg.model_id = self.config.policy_agent_server_config.model_name
        else:
            agent_cfg.model_id = self.config.default_server_config.model_name

        # Store agent configuration for later use in creating episode-specific agents
        self.agent_config = agent_cfg

        # Initialize challenge registry (same as minimal env)
        self.challenge_registry = create_textworld_registry(
            seed=getattr(self.config, "seed", None)
        )

    def _create_game(self, challenge_name: str, settings: Dict[str, Any]) -> str:
        """Create a TextWorld game using TextWorld challenges (mirrors minimal env)."""
        options = textworld.GameOptions()
        options.seeds = settings.get("seed", random.randint(0, 1000000))

        if challenge_name == "tw-simple":
            game_settings = {
                "rewards": settings.get("rewards", "balanced"),
                "goal": settings.get("goal", "detailed"),
                "test": str(settings.get("test", False)).lower(),
            }
            game = textworld.challenges.simple.make(game_settings, options=options)
        elif challenge_name == "tw-cooking":
            game_settings = {
                "recipe": settings.get("recipe", 1),
                "take": settings.get("take", 1),
                "cook": settings.get("cook", False),
                "open": settings.get("open", False),
                "drop": settings.get("drop", False),
                "go": settings.get("go", 1),
                "recipe_seed": settings.get(
                    "recipe-seed",
                    settings.get("recipe_seed", random.randint(0, 1000000)),
                ),
                "split": "train",
            }
            game = textworld.challenges.cooking.make(game_settings, options=options)
        elif challenge_name == "tw-coin_collector":
            game_settings = {"level": settings.get("level", 1)}
            game = textworld.challenges.coin_collector.make(
                game_settings, options=options
            )
        elif challenge_name == "tw-treasure_hunter":
            game_settings = {"level": settings.get("level", 1)}
            game = textworld.challenges.treasure_hunter.make(
                game_settings, options=options
            )
        else:
            raise ValueError(f"Unknown challenge: {challenge_name}")

        game_file = os.path.join(
            self._temp_dir,
            f"{challenge_name}_{settings.get('seed', random.randint(0, 1000000))}.z8",
        )
        options.path = game_file
        options.file_ext = ".z8"
        game_file = textworld.generator.compile_game(game, options)
        self._generated_files.add(game_file)
        return game_file

    def _create_agent_for_episode(self) -> AtroposAgent:
        """Create a new agent instance for an episode with its own memory manager."""
        # Create a new memory manager for this agent
        episode_memory_manager = None
        if MEMORY_SYSTEM_PREREQUISITES_AVAILABLE:
            episode_memory_manager = AtroposMemoryManager(
                embedding_dim_config_val=self.agent_config.embedding_dim,
                player_id_for_logging=f"Agent_{uuid.uuid4().hex[:8]}",
            )
            logger.info("Created new memory manager for episode agent")

        # Create new agent with its own memory manager
        agent = AtroposAgent(
            server_client=self.server,
            tokenizer=self.tokenizer,
            config=self.agent_config,
            memory_manager=episode_memory_manager,
        )
        return agent

    async def setup(self):
        for i, server in enumerate(self.server.servers):
            if hasattr(server, "config"):
                logger.info(
                    f"  Server {i}: {server.config.base_url} "
                    f"(api_key={'set' if server.config.api_key else 'empty'})"
                )
            else:
                logger.info(f"  Server {i}: {type(server).__name__}")

    async def register_env(self):
        """Register environment with rollout server."""
        logger.info(
            f"Registering TextWorldEnv with rollout server at {self.config.rollout_server_url}"
        )
        try:
            await super().register_env()
            # Override max token length to allow full trajectories
            self.max_token_len = 40960
            logger.info(
                f"Successfully registered with rollout server, "
                f"max_token_len set to {self.max_token_len}"
            )
        except Exception as e:
            logger.error(f"Failed to register with rollout server: {e}", exc_info=True)
            raise

    async def get_server_info(self):
        """Get server information after registration."""
        logger.info("Getting server information...")
        try:
            await super().get_server_info()
            logger.info("Server information retrieved successfully")
            logger.info(
                f"TextWorldEnv is ready to receive requests on {self.config.rollout_server_url}"
            )
        except Exception as e:
            logger.error(f"Failed to get server info: {e}", exc_info=True)
            raise

    def _format_observation(self, obs: str, infos: Dict[str, Any]) -> str:
        """Format TextWorld observation and additional info for the LLM."""
        objective = infos.get("objective", "No objective provided.")
        inventory = infos.get("inventory", "Your inventory is empty.")
        description = infos.get("description", obs)
        feedback = infos.get("feedback", "")

        formatted_obs = f"Objective: {objective}\n\n"
        formatted_obs += f"Current Location & State:\n{description}\n\n"
        formatted_obs += f"Inventory: {inventory}\n\n"
        if infos.get("last_action"):
            formatted_obs += (
                f"Feedback from last action ('{infos['last_action']}'):\n{feedback}\n"
            )

        return formatted_obs.strip()

    async def _get_or_create_episode(
        self,
        challenge_name: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> Optional[TextWorldEpisodeState]:
        """Create a new TextWorld episode using the simplified challenge flow."""
        episode_id = f"textworld-episode-{uuid.uuid4().hex}"
        # Pick a challenge if not provided
        if challenge_name is None or settings is None:
            challenge_name, settings = self.challenge_registry.get_challenge(
                random.choice(self.config.challenge_names),
                randomize_settings=self.config.randomize_challenge_settings,
            )
        logger.info(
            f"[Episode: {episode_id}] Challenge: {challenge_name} | Settings: {settings}"
        )

        try:
            game_file_path = self._create_game(challenge_name, settings)
        except Exception as e:
            logger.error(f"Failed to create game for episode {episode_id}: {e}")
            return None

        requested_infos = textworld.EnvInfos(
            description=True,
            inventory=True,
            objective=True,
            score=True,
            max_score=True,
            won=True,
            lost=True,
            facts=True,
            last_action=True,
            feedback=True,
            moves=True,
            admissible_commands=True,
        )

        registered_env_id = textworld.gym.register_game(
            game_file_path,
            requested_infos,
            max_episode_steps=self.config.max_steps,
            name=episode_id,
        )

        try:
            env = textworld.gym.make(registered_env_id)
            raw_obs, infos = env.reset()
            formatted_initial_obs = self._format_observation(raw_obs, infos)

            ep_state = TextWorldEpisodeState(
                episode_id=episode_id,
                game_file=game_file_path,
                textworld_env_instance=env,
                initial_obs=formatted_initial_obs,
                initial_infos=infos,
                max_steps=self.config.max_steps,
            )
            self.episodes[episode_id] = ep_state
            ep_state.last_env_raw_observation = raw_obs
            ep_state.last_env_infos = infos
            return ep_state
        except Exception as e:
            logger.error(
                f"Failed to setup gym environment for episode {episode_id}: {e}"
            )
            if os.path.exists(game_file_path):
                try:
                    os.remove(game_file_path)
                except OSError:
                    pass
            return None

    async def get_next_item(self) -> Optional[Dict[str, Any]]:
        """Create and return a new initialized TextWorldEpisodeState for trajectory collection."""
        # Select a challenge via the registry (same as minimal env)
        challenge_name = random.choice(self.config.challenge_names)
        challenge_name, settings = self.challenge_registry.get_challenge(
            challenge_name, randomize_settings=self.config.randomize_challenge_settings
        )
        episode_state = await self._get_or_create_episode(challenge_name, settings)
        if episode_state is None:
            logger.error("Failed to create new TextWorld episode.")
            return None

        return {"episode_state": episode_state, "episode_id": episode_state.episode_id}

    def _parse_action_with_prediction(
        self, agent_response_text: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Parse agent response to extract TextWorld command and expected outcome from tool call."""
        if not agent_response_text:
            return None, None

        tool_name, arguments, is_error = parse_tool_call(
            response=agent_response_text, preferred_tags=["tool_call"]
        )

        if is_error or tool_name != "execute_command":
            return None, None

        parsed_command = arguments.get("command")
        expected_outcome = arguments.get("expected_outcome", "")

        if not parsed_command or not isinstance(parsed_command, str):
            return None, None

        parsed_command = parsed_command.strip()
        expected_outcome = (
            expected_outcome.strip() if isinstance(expected_outcome, str) else ""
        )

        return (parsed_command if parsed_command else None), expected_outcome

    async def _calculate_perplexity_from_server(
        self, messages: List[Dict[str, str]]
    ) -> float:
        """Calculate perplexity using logprobs from the inference server."""
        # Import AtroposAgent to access TOOLS constant
        from environments.game_environments.agents.atropos_agent import (
            AtroposAgent,
        )

        # Apply chat template to get the full formatted text with tools
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tools=AtroposAgent.TOOLS,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Use completion mode to get logprobs for the entire sequence
        response = await self.server.completion(
            prompt=full_text,
            max_tokens=0,  # We're not generating, just getting logprobs
            echo=True,  # Return logprobs for the input
            logprobs=1,  # Return log probability of the selected tokens (must be > 0 for SGLang)
            temperature=0.0,
        )

        # Find where the last message starts in the tokenized sequence
        # We need to tokenize the messages without the last one to find the boundary
        if len(messages) > 1:
            prefix_messages = messages[:-1]
            prefix_text = self.tokenizer.apply_chat_template(
                prefix_messages,
                tools=AtroposAgent.TOOLS,
                tokenize=False,
                add_generation_prompt=True,
            )
            prefix_tokens = self.tokenizer.encode(prefix_text, add_special_tokens=False)

            # Extract logprobs for the last message (the completion we're evaluating)
            if (
                response.choices[0].logprobs is None
                or response.choices[0].logprobs.token_logprobs is None
            ):
                return float("inf")
            all_logprobs = response.choices[0].logprobs.token_logprobs
            completion_logprobs = all_logprobs[len(prefix_tokens) :]
        else:
            # If there's only one message, evaluate the whole thing
            if (
                response.choices[0].logprobs is None
                or response.choices[0].logprobs.token_logprobs is None
            ):
                return float("inf")
            completion_logprobs = response.choices[0].logprobs.token_logprobs

        # Calculate perplexity: exp(-mean(log_probs))
        if completion_logprobs:
            mean_logprob = sum(completion_logprobs) / len(completion_logprobs)
            return math.exp(-mean_logprob)
        return float("inf")

    async def _get_assistant_logprobs_from_server(
        self, messages: List[Dict[str, str]], assistant_response: str
    ) -> List[float]:
        """Get logprobs for assistant response using echo mode from the inference server."""
        # Import AtroposAgent to access TOOLS constant
        from environments.game_environments.agents.atropos_agent import (
            AtroposAgent,
        )

        try:
            # Create the full conversation including the assistant response
            full_messages = messages + [
                {"role": "assistant", "content": assistant_response}
            ]

            # Apply chat template to get the full formatted text with tools
            full_text = self.tokenizer.apply_chat_template(
                full_messages,
                tools=AtroposAgent.TOOLS,
                tokenize=False,
                add_generation_prompt=False,
            )

            # Use completion mode to get logprobs for the entire sequence
            response = await self.server.completion(
                prompt=full_text,
                max_tokens=0,  # We're not generating, just getting logprobs
                echo=True,  # Return logprobs for the input
                logprobs=1,  # Return log probability of the selected tokens (must be > 0 for SGLang)
                temperature=0.0,
            )

            # Find where the assistant message starts in the tokenized sequence
            # Tokenize the messages without the assistant response to find the boundary
            prefix_text = self.tokenizer.apply_chat_template(
                messages,
                tools=AtroposAgent.TOOLS,
                tokenize=False,
                add_generation_prompt=True,
            )
            prefix_tokens = self.tokenizer.encode(prefix_text, add_special_tokens=False)

            # Extract logprobs for the assistant message (the completion we're evaluating)
            if (
                response.choices[0].logprobs is None
                or response.choices[0].logprobs.token_logprobs is None
            ):
                return []

            all_logprobs = response.choices[0].logprobs.token_logprobs
            assistant_logprobs = all_logprobs[len(prefix_tokens) :]

            return assistant_logprobs

        except Exception as e:
            logger.error(
                f"Error getting logprobs for assistant response: {e}",
                exc_info=True,
            )
            return []

    async def _vrcli_score(
        self,
        current_obs: str,
        actual_action: str,
        predicted_outcome: str,
        actual_outcome: str,
    ) -> float:
        """Score prediction quality using perplexity improvement with discrete reward levels.

        Following the VR-CLI paper, we calculate percentage improvement and map to discrete rewards:
        - 0.0: improvement < 0.05 (negligible)
        - 0.5: 0.05 ≤ improvement < 1 (small improvement)
        - 0.9: 1 ≤ improvement < 2 (moderate improvement)
        - 1.0: improvement ≥ 2 (significant improvement)

        TODO: try other ways of conditioning the predictions, like including history,
        full samples, memories, etc
        Plus, should the base contain action or not, ablations needed on this
        seems to work tho
        """
        if not predicted_outcome:
            return 0.0

        # Create message lists for base and prediction-conditioned perplexity
        base_messages = [
            {
                "role": "user",
                "content": (
                    f"Current state:\n{current_obs}\n\n"
                    f"Action Taken: {actual_action}\n\n"
                    "What happens next?"
                ),
            },
            {"role": "assistant", "content": actual_outcome},
        ]
        prediction_messages = [
            {
                "role": "user",
                "content": (
                    f"Current state:\n{current_obs}\n\n"
                    f"Action Taken: {actual_action}\n\n"
                    f"Predicted outcome: {predicted_outcome}\n\n"
                    "What actually happens?"
                ),
            },
            {"role": "assistant", "content": actual_outcome},
        ]

        # Calculate perplexities using server
        base_ppl = await self._calculate_perplexity_from_server(base_messages)
        pred_ppl = await self._calculate_perplexity_from_server(prediction_messages)

        logger.debug(
            f"VR-CLI perplexity calculation: base_ppl={base_ppl:.3f}, pred_ppl={pred_ppl:.3f}"
        )

        # Calculate percentage improvement using VR-CLI formula
        # Improvement = [1 - PPL(y|x,a)/PPL(y|x)] × 100
        if base_ppl == 0 or base_ppl == float("inf"):
            logger.debug("VR-CLI: Base perplexity is invalid, returning 0.0")
            return 0.0

        improvement = (1 - pred_ppl / base_ppl) * 100

        logger.debug(
            f"VR-CLI improvement: {improvement:.2f}% (ratio: {pred_ppl/base_ppl:.3f})"
        )

        # Map to discrete reward levels
        if improvement < 0.05:
            reward = 0.0  # Negligible improvement
        elif improvement < 1.0:
            reward = 0.5  # Small improvement
        elif improvement < 2.0:
            reward = 0.9  # Moderate improvement
        else:
            reward = 1.0  # Significant improvement

        logger.debug(f"VR-CLI final reward: {reward} (improvement: {improvement:.2f}%)")
        return reward

    def _calculate_format_reward(self, response_text: str) -> float:
        """Calculate format reward based on adherence to required structure.

        Expected structure (in exact order):
        1. Exactly one <think>...</think> block with content
        2. Exactly one <memory>...</memory> block with content
        3. Exactly one <tool_call>...</tool_call> block with valid JSON

        Args:
            response_text: The full response text from the agent

        Returns:
            Format reward score (0.0 to 0.1) - gives partial credit for good blocks
        """
        if not self.config.format_reward_enabled:
            return 0.0

        # Parse the response into structured components and always return partial score
        _, structure_score = self._validate_response_structure(response_text)
        # length score
        # length_scores = [math.tanh((length_mean - len(tok)) / length_mean) / 2.0 for tok in scores['tokens']]
        # Return partial score even if structure is considered "invalid"
        return structure_score

    def _validate_response_structure(self, response_text: str) -> Tuple[bool, float]:
        """Validate the exact structure of the response.

        Returns:
            Tuple of (is_valid, partial_score)
        """
        import re

        # Find all think blocks
        think_pattern = r"<think>(.*?)</think>"
        think_matches = re.findall(think_pattern, response_text, re.DOTALL)

        # Find all memory blocks
        memory_pattern = r"<memory>(.*?)</memory>"
        memory_matches = re.findall(memory_pattern, response_text, re.DOTALL)

        # Find all tool_call blocks
        tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
        tool_call_matches = re.findall(tool_call_pattern, response_text, re.DOTALL)

        # Check exact counts
        has_exactly_one_think = len(think_matches) == 1 and think_matches[0].strip()
        has_exactly_one_memory = len(memory_matches) == 1 and memory_matches[0].strip()
        has_exactly_one_tool_call = (
            len(tool_call_matches) == 1 and tool_call_matches[0].strip()
        )

        # Check ordering by finding positions
        think_pos = response_text.find("<think>")
        memory_pos = response_text.find("<memory>")
        tool_call_pos = response_text.find("<tool_call>")

        # All blocks must be present
        blocks_present = think_pos != -1 and memory_pos != -1 and tool_call_pos != -1
        correct_order = blocks_present and think_pos < memory_pos < tool_call_pos

        # Calculate partial score based on what's correct
        score = 0.0
        max_score = (
            self.config.format_thinking_reward + self.config.format_memory_reward
        )

        # Award points for each correctly structured block
        if has_exactly_one_think:
            score += self.config.format_thinking_reward

        if has_exactly_one_memory:
            score += self.config.format_memory_reward

        # Apply strict structure enforcement if enabled
        # Want to see if mixing it up or letting it generate multiple changes stuffs
        if self.config.format_strict_structure:
            # Bonus for correct ordering (only if all blocks present)
            if (
                correct_order
                and has_exactly_one_think
                and has_exactly_one_memory
                and has_exactly_one_tool_call
            ):
                score = max_score  # Full score for perfect structure
            elif not correct_order and score > 0:
                score *= (
                    self.config.format_wrong_order_penalty
                )  # Configurable penalty for wrong order

            # Check for extra blocks (penalty)
            extra_blocks_penalty = 0.0
            if len(think_matches) > 1:
                extra_blocks_penalty += self.config.format_extra_blocks_penalty
            if len(memory_matches) > 1:
                extra_blocks_penalty += self.config.format_extra_blocks_penalty
            if len(tool_call_matches) > 1:
                extra_blocks_penalty += self.config.format_extra_blocks_penalty

            score = max(0.0, score - extra_blocks_penalty)
        else:
            pass

        # Consider response valid if it has the basic structure, even if not perfect
        is_valid = (
            blocks_present
            and len(think_matches) >= 1
            and len(memory_matches) >= 1
            and len(tool_call_matches) >= 1
        )

        return is_valid, score

    async def _score_alternatives_hybrid(
        self,
        ep_state: TextWorldEpisodeState,
        candidates: List[
            Tuple[Optional[str], Optional[str], str, Optional[List[Dict[str, Any]]]]
        ],
        conversation_history: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """
        Hybrid scoring system:
        1. Use entropy/varentropy to select best alternative for execution
        2. Execute selected alternative in real environment
        3. Score selected alternative + same-action alternatives with VR-CLI
        4. Score rejected alternatives with entropy confidence scores

        TODO: Try keeping entropy score + VR-CLI, see what combinations work best
        Also compare VR-CLI against a GenRM (once we have a good one)
        """
        evaluations = []

        # Step 1: Calculate confidence scores for all alternatives
        confidence_scores = []
        for i, (action, prediction, response_text, logprobs_data) in enumerate(
            candidates
        ):
            if action is None:
                action = "look"  # Default action

            # Get logprobs for this alternative using the new method
            try:
                assistant_logprobs = await self._get_assistant_logprobs_from_server(
                    conversation_history, response_text
                )

                # Convert to format expected by entropy calculator
                if assistant_logprobs:
                    logprobs_data = []
                    for logprob in assistant_logprobs:
                        logprobs_data.append(
                            {
                                "token": "",
                                "logprob": logprob,
                                "top_logprobs": [{"token": "", "logprob": logprob}],
                            }
                        )
                    entropy_confidence = confidence_score(logprobs_data)
                else:
                    entropy_confidence = (
                        0.0  # No logprobs available, eg openai or something
                    )

                logger.debug(
                    f"[Episode: {ep_state.episode_id}] Alternative {i}: action='{action}', "
                    f"entropy_confidence={entropy_confidence:.3f}, logprobs_count={len(assistant_logprobs)}"
                )
            except Exception as e:
                logger.warning(
                    f"[Episode: {ep_state.episode_id}] Failed to get logprobs for alternative {i}: {e}"
                )
                entropy_confidence = 0.0

            confidence_scores.append(entropy_confidence)

        # Step 2: Select best alternative (highest confidence)
        if confidence_scores:
            logger.debug(
                f"[Episode: {ep_state.episode_id}] Confidence scores for all alternatives: "
                f"{[f'alt{i}={score:.3f}' for i, score in enumerate(confidence_scores)]}"
            )
            selected_idx = max(
                range(len(confidence_scores)), key=lambda i: confidence_scores[i]
            )
        else:
            selected_idx = 0

        selected_action = candidates[selected_idx][0]
        if selected_action is None:
            selected_action = "look"

        confidence_value = confidence_scores[selected_idx] if confidence_scores else 0.0
        logger.info(
            f"[Episode: {ep_state.episode_id}] Selected alternative {selected_idx} with action '{selected_action}' "
            f"(confidence: {confidence_value:.3f})"
        )

        # Step 3: Execute selected action in main environment
        # Save previous observation for VR-CLI scoring
        previous_obs = ep_state.last_formatted_obs

        try:
            obs, reward, done, info = ep_state.textworld_env.step(selected_action)
            actual_outcome = self._format_observation(obs, info)

            # Update episode state
            ep_state.last_env_raw_observation = obs
            ep_state.last_env_infos = info
            ep_state.last_formatted_obs = actual_outcome
            ep_state.won = info.get("won", False)
            ep_state.lost = info.get("lost", False)
            ep_state.done = done

            logger.info(
                f"[Episode: {ep_state.episode_id}] Executed action '{selected_action}', "
                f"reward={reward}, done={done}"
            )

        except Exception as e:
            logger.error(
                f"[Episode: {ep_state.episode_id}] Error executing selected action: {e}"
            )
            actual_outcome = f"Error executing action: {e}"
            reward = -1.0
            done = True
            info = {"error": True}

        # Step 4: Group alternatives by action to identify same-action alternatives
        action_groups = {}
        for i, (action, prediction, response_text, logprobs_data) in enumerate(
            candidates
        ):
            if action is None:
                action = "look"
            if action not in action_groups:
                action_groups[action] = []
            action_groups[action].append(i)

        # Step 5: Score all alternatives
        for i, (action, prediction, response_text, logprobs_data) in enumerate(
            candidates
        ):
            if action is None:
                action = "look"

            # Calculate format reward
            format_score = self._calculate_format_reward(response_text)

            # Apply token length penalty/bonus if enabled
            response_tokens = 0
            token_length_adjustment = 0.0
            if self.config.token_length_penalty_enabled:
                response_tokens = len(
                    self.tokenizer.encode(response_text, add_special_tokens=False)
                )
                token_deviation = response_tokens - self.config.token_length_baseline
                token_length_adjustment = (
                    -token_deviation * self.config.token_length_penalty_scale
                )
                max_adjustment = self.config.token_length_penalty_weight
                token_length_adjustment = max(
                    -max_adjustment, min(max_adjustment, token_length_adjustment)
                )

            if action == selected_action:
                # Use VR-CLI for selected action and same-action alternatives
                try:
                    # Use the observation from before the step for VR-CLI scoring
                    logger.debug(
                        f"[Episode: {ep_state.episode_id}] Calculating VR-CLI score for alt {i}: "
                        f"action='{action}', prediction='{prediction[:50] if prediction else 'None'}...'"
                    )
                    vrcli_score = await self._vrcli_score(
                        previous_obs,
                        action,
                        prediction or "",
                        actual_outcome,
                    )
                    logger.debug(
                        f"[Episode: {ep_state.episode_id}] VR-CLI score for alt {i}: {vrcli_score:.3f}"
                    )
                except Exception as e:
                    logger.error(
                        f"Error calculating VR-CLI score for alternative {i}: {e}"
                    )
                    vrcli_score = 0.0

                # Combine scores with proper weighting
                env_weight = (
                    1.0 - self.config.vrcli_weight - self.config.format_reward_weight
                )
                combined_score = (
                    self.config.vrcli_weight * vrcli_score
                    + self.config.format_reward_weight * format_score
                    + env_weight * reward
                )

                # Apply token length adjustment
                if self.config.token_length_penalty_enabled:
                    combined_score *= 1.0 + token_length_adjustment

                evaluations.append(
                    {
                        "index": i,
                        "action": action,
                        "prediction": prediction,
                        "vrcli_score": vrcli_score,
                        "entropy_score": confidence_scores[i],
                        "latro_score": 0.0,
                        "format_score": format_score,
                        "env_reward": reward,
                        "combined_score": combined_score,
                        "response_text": response_text,
                        "response_tokens": response_tokens,
                        "token_length_adjustment": token_length_adjustment,
                        "actual_outcome": actual_outcome,
                        "done": done,
                        "info": info,
                        "selected": i == selected_idx,
                        "scoring_method": "vrcli",
                    }
                )
            else:
                # Use entropy confidence for rejected alternatives
                entropy_confidence = confidence_scores[i]

                # Use entropy score as primary score for rejected alternatives
                combined_score = (
                    0.7 * entropy_confidence  # Entropy confidence as main score
                    + 0.3 * format_score  # Format reward for structure
                )

                # Apply token length adjustment
                if self.config.token_length_penalty_enabled:
                    combined_score *= 1.0 + token_length_adjustment

                evaluations.append(
                    {
                        "index": i,
                        "action": action,
                        "prediction": prediction,
                        "vrcli_score": 0.0,  # Not calculated for rejected alternatives
                        "entropy_score": entropy_confidence,
                        "latro_score": 0.0,
                        "format_score": format_score,
                        "env_reward": 0.0,  # Not applicable
                        "combined_score": combined_score,
                        "response_text": response_text,
                        "response_tokens": response_tokens,
                        "token_length_adjustment": token_length_adjustment,
                        "actual_outcome": "",  # Not executed
                        "done": False,
                        "info": {},
                        "selected": False,
                        "scoring_method": "entropy",
                    }
                )

        return evaluations

    async def _next_step(
        self,
        ep_state: TextWorldEpisodeState,
        current_turn_num: int,
        agent: AtroposAgent,
    ) -> Tuple[Optional[ScoredDataGroup], bool]:
        """Execute one step of the TextWorld episode using VR-CLI evaluation."""
        logger.info(
            f"[DEBUG] _next_step called for episode {ep_state.episode_id}, turn {current_turn_num}"
        )

        # Early termination check: if episode is already done, don't process
        if ep_state.done:
            logger.info(
                f"[DEBUG] Episode {ep_state.episode_id} - Episode already marked as done, skipping step"
            )
            # Return empty ScoredDataGroup with score 0 instead of None
            empty_sdg = {
                "tokens": [[]],
                "masks": [[]],
                "scores": [0.0],
                "messages": [[]],
                "advantages": None,
                "ref_logprobs": None,
                "group_overrides": None,
                "overrides": None,
                "images": None,
            }
            return empty_sdg, True

        # 1. Get current observation
        if current_turn_num == 0:
            current_observation = ep_state.initial_formatted_obs
        else:
            raw_obs = ep_state.last_env_raw_observation
            infos = ep_state.last_env_infos
            if raw_obs is None or infos is None:
                logger.error(
                    f"[Episode: {ep_state.episode_id}] Missing observation data for turn {current_turn_num}"
                )
                # Return ScoredDataGroup with score 0 for missing observation
                error_sdg = {
                    "tokens": [[]],
                    "masks": [[]],
                    "scores": [0.0],
                    "messages": [[]],
                    "advantages": None,
                    "ref_logprobs": None,
                    "group_overrides": None,
                    "overrides": None,
                    "images": None,
                }
                return error_sdg, True
            current_observation = self._format_observation(raw_obs, infos)

        ep_state.last_formatted_obs = current_observation
        logger.debug(
            f"[DEBUG] Episode {ep_state.episode_id} - Observation length: {len(current_observation)}"
        )

        # 2. Generate candidate actions with predictions
        logger.debug(
            f"[DEBUG] Episode {ep_state.episode_id} - Calling agent.generate_action with "
            f"n={self.config.group_size}"
        )
        try:
            group_actions = await agent.generate_action(
                current_observation, n=self.config.group_size
            )
            logger.debug(
                f"[DEBUG] Episode {ep_state.episode_id} - Received "
                f"{len(group_actions) if group_actions else 0} actions from agent"
            )
        except Exception as e:
            logger.error(
                f"[Episode: {ep_state.episode_id}] Error getting actions from agent: {e}",
                exc_info=True,
            )
            # Return ScoredDataGroup with score 0 for agent error
            error_sdg = {
                "tokens": [[]],
                "masks": [[]],
                "scores": [0.0],
                "messages": [[]],
                "advantages": None,
                "ref_logprobs": None,
                "group_overrides": None,
                "overrides": None,
                "images": None,
            }
            return error_sdg, True

        if not group_actions:
            logger.error(
                f"[Episode: {ep_state.episode_id}] No actions received from agent"
            )
            # Return ScoredDataGroup with score 0 for no actions
            error_sdg = {
                "tokens": [[]],
                "masks": [[]],
                "scores": [0.0],
                "messages": [[]],
                "advantages": None,
                "ref_logprobs": None,
                "group_overrides": None,
                "overrides": None,
                "images": None,
            }
            return error_sdg, True

        # 3. Parse actions and predictions
        candidates = []
        for i, action_response in enumerate(group_actions):
            response_text = action_response["action_text"]
            logprobs_data = action_response.get("logprobs")
            logger.info(f"\n=== Alternative {i} Full Response ===")
            logger.info(response_text)
            logger.info("=== End Response ===\n")

            # Validate structure before parsing
            structure_valid, structure_score = self._validate_response_structure(
                response_text
            )
            if not structure_valid:
                logger.warning(
                    f"[Episode: {ep_state.episode_id}] Alternative {i} has invalid structure. "
                    f"Structure score: {structure_score:.3f}"
                )

            # Check for multiple tool calls specifically
            tool_call_count = response_text.count("<tool_call>")
            if tool_call_count > 1:
                logger.warning(
                    f"[Episode: {ep_state.episode_id}] Alternative {i} has {tool_call_count} tool calls! "
                    f"This violates the expected structure of exactly 1 tool call."
                )

            action, prediction = self._parse_action_with_prediction(response_text)
            candidates.append((action, prediction, response_text, logprobs_data))

        if not candidates:
            logger.error(f"[Episode: {ep_state.episode_id}] No valid actions parsed")
            # Return ScoredDataGroup with score 0 for no valid candidates
            error_sdg = {
                "tokens": [[]],
                "masks": [[]],
                "scores": [0.0],
                "messages": [[]],
                "advantages": None,
                "ref_logprobs": None,
                "group_overrides": None,
                "overrides": None,
                "images": None,
            }
            return error_sdg, True

        # 4. Get conversation history for entropy calculation (reconstruct same as agent)

        current_history_atropos = agent._reconstruct_canonical_history()

        observation_message = Message(
            role="user", content=current_observation, reward=None
        )
        current_history_atropos.append(observation_message)
        conversation_history = _convert_messages_for_api(current_history_atropos)

        # 4. Score alternatives using hybrid entropy + VR-CLI system
        evaluations = await self._score_alternatives_hybrid(
            ep_state, candidates, conversation_history
        )

        if not evaluations:
            logger.error(f"[Episode: {ep_state.episode_id}] No evaluations returned")
            # Return ScoredDataGroup with score 0 for no evaluations
            error_sdg = {
                "tokens": [[]],
                "masks": [[]],
                "scores": [0.0],
                "messages": [[]],
                "advantages": None,
                "ref_logprobs": None,
                "group_overrides": None,
                "overrides": None,
                "images": None,
            }
            return error_sdg, True

        # Track step-level metrics for wandb
        selected_eval = next((e for e in evaluations if e.get("selected", False)), None)
        if selected_eval:
            step_metrics = {
                "vrcli_score": selected_eval.get("vrcli_score", 0.0),
                "format_score": selected_eval.get("format_score", 0.0),
                "confidence_score": selected_eval.get("entropy_score", 0.0),
                "combined_score": selected_eval.get("combined_score", 0.0),
                "env_reward": selected_eval.get("env_reward", 0.0),
                "action": selected_eval.get("action", "unknown"),
                "response_tokens": selected_eval.get("response_tokens", 0),
                "selected_rank": sorted(
                    evaluations, key=lambda x: x.get("entropy_score", 0), reverse=True
                ).index(selected_eval),
            }
            self.step_metrics_buffer.append(step_metrics)

        # 5. Select best action based on combined score
        best_eval = max(evaluations, key=lambda x: x["combined_score"])
        best_idx = best_eval["index"]

        # 6. Record selected action with agent
        try:
            await agent.record_selected_action_and_learn_from_turn(
                selected_action_index=best_idx
            )
        except Exception as e:
            logger.error(
                f"[Episode: {ep_state.episode_id}] Error recording selected action: {e}"
            )

        # 7. Execute best action in main environment
        try:
            obs, reward, done, info = ep_state.textworld_env.step(best_eval["action"])
            ep_state.last_env_raw_observation = obs
            ep_state.last_env_infos = info
            ep_state.last_formatted_obs = self._format_observation(obs, info)
        except Exception as e:
            logger.error(
                f"[Episode: {ep_state.episode_id}] Error stepping environment: {e}"
            )
            # Return ScoredDataGroup with score 0 for environment error
            error_sdg = {
                "tokens": [[]],
                "masks": [[]],
                "scores": [0.0],
                "messages": [[]],
                "advantages": None,
                "ref_logprobs": None,
                "group_overrides": None,
                "overrides": None,
                "images": None,
            }
            return error_sdg, True

        # 8. Update episode state
        ep_state.cumulative_reward += reward
        ep_state.canonical_rewards.append(reward)
        ep_state.canonical_chosen_alternative_indices.append(best_idx)
        ep_state.done = done
        ep_state.last_score = info.get("score", ep_state.last_score)
        ep_state.moves = info.get("moves", ep_state.moves)

        if done:
            ep_state.won = info.get("won", False)
            ep_state.lost = info.get("lost", False)
            logger.info(
                f"[Episode: {ep_state.episode_id}] Game completed on turn {current_turn_num + 1}. "
                f"Won: {ep_state.won}, Lost: {ep_state.lost}, Score: {ep_state.last_score}"
            )

        # Additional check: if the best evaluation shows done, we should also be done
        # This handles cases where the TextWorld environment might not properly report done=True
        if best_eval.get("done", False) and not ep_state.done:
            logger.info(
                f"[Episode: {ep_state.episode_id}] Evaluation environment indicates done=True, "
                f"but main environment doesn't. Forcing episode termination."
            )
            ep_state.done = True
            ep_state.won = best_eval.get("info", {}).get("won", False)
            ep_state.lost = best_eval.get("info", {}).get("lost", False)

        # 9. Prepare training data
        sg_tokens = []
        sg_masks = []
        sg_messages = []
        scores = []

        # Get canonical history for all alternatives
        canonical_history = agent.get_final_canonical_dialogue()

        for i, eval_data in enumerate(evaluations):
            # Build message history for this alternative
            alternative_history = canonical_history.copy()
            alternative_history.append(
                Message(
                    role="assistant", content=eval_data["response_text"], reward=None
                )
            )
            # Tokenize
            try:
                tokenized = tokenize_for_trainer(self.tokenizer, alternative_history)
                sg_tokens.append(tokenized["tokens"])
                sg_masks.append(tokenized["masks"])
                sg_messages.append(alternative_history)
                scores.append(eval_data["combined_score"])
            except Exception as e:
                # Had a couple issues with a weird sample from sglang
                # So just skipping that for now (will be thrown out & not sent to trainer)
                logger.error(f"Error tokenizing alternative {i}: {e}")
                sg_tokens.append([])
                sg_masks.append([])
                sg_messages.append(alternative_history)
                scores.append(0.0)

        # Apply top-k/bottom-k selection if enabled
        # Idea is to keep the extreme ends of the distribution, but not the middle
        # Just for fun, I saw it on x somewhere, thought it was worth trying
        # TODO: Also add this: https://arxiv.org/html/2506.02177v1
        # Not sure how that interacts with multistep training like this tho
        if self.config.use_topk_bottomk_selection and len(scores) > (
            self.config.topk + self.config.bottomk
        ):
            # Create list of (index, score, has_valid_tokens) tuples
            indexed_scores = []
            for i, score in enumerate(scores):
                has_valid_tokens = len(sg_tokens[i]) > 0
                if has_valid_tokens:  # Only consider alternatives with valid tokens
                    indexed_scores.append((i, score))

            # Sort by score (descending)
            indexed_scores.sort(key=lambda x: x[1], reverse=True)

            # Select top-k and bottom-k indices
            selected_indices = []
            if len(indexed_scores) >= 2:  # Ensure we have at least 2 alternatives
                # Take top-k
                selected_indices.extend(
                    [idx for idx, _ in indexed_scores[: self.config.topk]]
                )
                # Take bottom-k (if we have enough)
                if len(indexed_scores) > self.config.topk:
                    bottom_start = max(
                        self.config.topk, len(indexed_scores) - self.config.bottomk
                    )
                    selected_indices.extend(
                        [idx for idx, _ in indexed_scores[bottom_start:]]
                    )

                # Sort selected indices to maintain order
                selected_indices.sort()

                # Filter arrays to keep only selected indices
                sg_tokens = [sg_tokens[i] for i in selected_indices]
                sg_masks = [sg_masks[i] for i in selected_indices]
                sg_messages = [sg_messages[i] for i in selected_indices]
                scores = [scores[i] for i in selected_indices]

                logger.info(
                    f"[Episode: {ep_state.episode_id}] Selected {len(selected_indices)} alternatives "
                    f"(top-{self.config.topk}, bottom-{self.config.bottomk}) "
                    f"from {len(indexed_scores)} valid alternatives"
                )
            else:
                logger.warning(
                    f"[Episode: {ep_state.episode_id}] Only "
                    f"{len(indexed_scores)} valid alternatives available, "
                    "keeping all (need at least 2 for training)"
                )
        else:
            # Log original behavior
            logger.info(
                f"[Episode: {ep_state.episode_id}] Creating ScoredDataGroup with {len(evaluations)} alternatives "
                f"(all {self.config.group_size} generated alternatives included, including any with parsing errors)"
            )

        # Create scored data group
        # If we're using top-k/bottom-k selection, we need to override the group size
        group_overrides = None
        if (
            self.config.use_topk_bottomk_selection
            and len(sg_tokens) != self.config.group_size
        ):
            group_overrides = {"group_size": len(sg_tokens)}

        scored_data_group = {
            "tokens": sg_tokens,
            "masks": sg_masks,
            "scores": scores,
            "messages": sg_messages,
            "advantages": None,
            "ref_logprobs": None,
            "group_overrides": group_overrides,
            "overrides": None,
            "images": None,
        }

        ep_state.policy_step_data.append(scored_data_group)

        return scored_data_group, ep_state.done

    async def collect_trajectories(self, item: Dict[str, Any]) -> Tuple[
        Union[
            Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]], List[Any | None]
        ],
        List[Any],
    ]:
        """Run a full TextWorld episode collecting data for each step."""
        # Ensure logger is at INFO level for debugging
        logger.setLevel(logging.INFO)

        logger.info(
            f"[DEBUG] collect_trajectories called with item: {item.get('episode_id', 'unknown')}"
        )

        if not item or "episode_state" not in item:
            logger.error("Invalid item received - missing 'episode_state'")
            return [], []

        ep_state: TextWorldEpisodeState = item["episode_state"]
        if not ep_state:
            logger.error("Episode state is None")
            return [], []

        policy_sdgs_for_episode: List[Optional[ScoredDataGroup]] = []

        # Create a new agent for this episode with its own memory
        agent = self._create_agent_for_episode()
        agent.new_game()
        logger.info(
            f"[Episode: {ep_state.episode_id}] Created new agent with isolated memory"
        )

        try:
            # Execute episode turns
            for current_turn_num in range(ep_state.max_turns):
                logger.info(
                    f"[DEBUG] Episode {ep_state.episode_id} - Starting turn {current_turn_num + 1}/{ep_state.max_turns}"
                )

                if ep_state.done:
                    logger.info(
                        f"[DEBUG] Episode {ep_state.episode_id} - Episode done, breaking"
                    )
                    break

                scored_data_group_for_turn, episode_is_done_after_step = (
                    await self._next_step(ep_state, current_turn_num, agent)
                )

                # Always append the ScoredDataGroup - it should never be None now
                policy_sdgs_for_episode.append(scored_data_group_for_turn)
                logger.warning(
                    f"[DEBUG] Episode {ep_state.episode_id} - Turn {current_turn_num + 1} completed "
                    f"with {len(scored_data_group_for_turn['scores'])} scores"
                )

                if episode_is_done_after_step:
                    logger.info(
                        f"[DEBUG] Episode {ep_state.episode_id} - Episode done after step, breaking"
                    )
                    break

                if current_turn_num == ep_state.max_turns - 1 and not ep_state.done:
                    logger.info(
                        f"[Episode: {ep_state.episode_id}] Reached max turns ({ep_state.max_turns}), "
                        f"forcing episode termination"
                    )
                    ep_state.done = True

        except Exception as e:
            logger.error(
                f"[Episode: {ep_state.episode_id}] Error during trajectory collection: {e}",
                exc_info=True,
            )
            ep_state.done = True
            return [], []
        finally:
            self._apply_credit_assignment(ep_state, policy_sdgs_for_episode)
            # Clean up episode resources
            # TEMPORARILY DISABLED: Cleanup happening too early, before candidate evaluation completes
            # await self._cleanup_episode_resources(ep_state)

        logger.info(
            f"[Episode: {ep_state.episode_id}] Episode complete! Final status: "
            f"done={ep_state.done}, won={ep_state.won}, lost={ep_state.lost}, "
            f"score={ep_state.last_score}, turns_taken={len(policy_sdgs_for_episode)}/{ep_state.max_turns}, "
            f"cumulative_reward={ep_state.cumulative_reward:.3f}"
        )

        return policy_sdgs_for_episode, []

    def _apply_credit_assignment(
        self,
        ep_state: TextWorldEpisodeState,
        policy_sdgs_for_episode: List[Optional[ScoredDataGroup]],
    ) -> None:
        """Apply credit assignment for sparse rewards using discounted returns.

        Also assigns credit to unselected alternatives that produced the same action
        as the selected candidate, as they would have led to the same outcome.
        """
        if not policy_sdgs_for_episode:
            return

        # Calculate final outcome reward
        final_outcome_reward = 0.0
        if ep_state.won:
            final_outcome_reward = 1.0
        elif ep_state.lost:
            final_outcome_reward = -1.0

        num_steps = len(ep_state.canonical_rewards)
        if num_steps == 0 or len(policy_sdgs_for_episode) != num_steps:
            logger.warning(
                f"[Episode: {ep_state.episode_id}] Mismatch in step counts for credit assignment"
            )
            return

        # Calculate discounted returns backwards (Monte Carlo returns)
        discounted_return = final_outcome_reward
        for t in range(num_steps - 1, -1, -1):
            immediate_reward = ep_state.canonical_rewards[t]

            discounted_return = (
                immediate_reward + self.config.vrcli_discount_factor * discounted_return
            )

            sdg = policy_sdgs_for_episode[t]
            # SDGs should never be None now
            chosen_idx = ep_state.canonical_chosen_alternative_indices[t]

            if 0 <= chosen_idx < len(sdg["scores"]):
                chosen_action = None
                if (
                    "messages" in sdg
                    and sdg["messages"] is not None
                    and chosen_idx < len(sdg["messages"])
                ):
                    chosen_messages = sdg["messages"][chosen_idx]
                    if chosen_messages and chosen_messages[-1]["role"] == "assistant":
                        response_text = chosen_messages[-1]["content"]
                        if isinstance(response_text, str):
                            action, _ = self._parse_action_with_prediction(
                                response_text
                            )
                            chosen_action = action

                new_scores = list(sdg["scores"])
                future_return = self.config.vrcli_discount_factor * (
                    discounted_return - immediate_reward
                )

                new_scores[chosen_idx] += future_return

                if (
                    chosen_action is not None
                    and "messages" in sdg
                    and sdg["messages"] is not None
                ):
                    for alt_idx in range(len(sdg["messages"])):
                        if alt_idx != chosen_idx and alt_idx < len(new_scores):
                            alt_messages = sdg["messages"][alt_idx]
                            if alt_messages and alt_messages[-1]["role"] == "assistant":
                                alt_response = alt_messages[-1]["content"]
                                if isinstance(alt_response, str):
                                    alt_action, _ = self._parse_action_with_prediction(
                                        alt_response
                                    )
                                    if alt_action == chosen_action:
                                        # This alternative would have led to the same outcome
                                        new_scores[alt_idx] += future_return
                                        logger.debug(
                                            f"[Episode: {ep_state.episode_id}] Turn {t}: "
                                            f"Alternative {alt_idx} also gets future return "
                                            f"for same action '{chosen_action}'"
                                        )

                sdg["scores"] = new_scores

                logger.debug(
                    f"[Episode: {ep_state.episode_id}] Turn {t}: "
                    f"immediate_reward={immediate_reward:.3f}, "
                    f"future_return={future_return:.3f}, "
                    f"total_return={new_scores[chosen_idx]:.3f}"
                )

        # Track episode completion metrics for wandb
        episode_metrics = {
            "won": ep_state.won,
            "lost": ep_state.lost,
            "completed": ep_state.done,
            "final_score": ep_state.last_score,
            "num_steps": len(ep_state.canonical_rewards),
            "cumulative_reward": ep_state.cumulative_reward,
            "final_outcome_reward": final_outcome_reward,
        }
        self.completed_episodes_buffer.append(episode_metrics)

    async def _cleanup_episode_resources(self, ep_state: TextWorldEpisodeState) -> None:
        """Clean up episode resources including environment and game files."""
        if ep_state.textworld_env:
            try:
                ep_state.textworld_env.close()
            except Exception as e:
                logger.warning(
                    f"[Episode: {ep_state.episode_id}] Error closing TextWorld environment: {e}"
                )

        # Clean up game file
        if ep_state.game_file and os.path.exists(ep_state.game_file):
            try:
                os.remove(ep_state.game_file)
                json_file = ep_state.game_file.replace(".z8", ".json")
                if os.path.exists(json_file):
                    os.remove(json_file)
            except OSError as e:
                logger.warning(
                    f"[Episode: {ep_state.episode_id}] Error removing game files: {e}"
                )

    async def postprocess_histories(
        self,
        trajectories: Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]],
    ) -> Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]:
        """Process and return all ScoredDataGroups for training on all episode turns."""

        # If trajectories is empty or None, return None
        if not trajectories:
            return None

        # Ensure we have a list
        if not isinstance(trajectories, list):
            trajectories = [trajectories]

        # Filter out None values and create minimal ScoredDataGroups
        processed_groups = []
        for sdg in trajectories:
            if sdg is None:
                continue

            # Create minimal ScoredDataGroup with only the fields needed for training
            tokens = sdg.get("tokens", [])
            masks = sdg.get("masks", [])
            scores = sdg.get("scores", [])

            # Skip empty groups
            if not tokens or not masks or not scores:
                continue

            minimal_sdg = {"tokens": tokens, "masks": masks, "scores": scores}
            processed_groups.append(minimal_sdg)

        if not processed_groups:
            return None

        # Return the list of all ScoredDataGroups
        return processed_groups

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Log TextWorld-specific metrics to wandb."""
        if wandb_metrics is None:
            wandb_metrics = {}

        # Episode-level metrics
        if self.completed_episodes_buffer:
            num_episodes = len(self.completed_episodes_buffer)

            # Win/loss/completion rates
            win_rate = (
                sum(1 for e in self.completed_episodes_buffer if e["won"])
                / num_episodes
            )
            loss_rate = (
                sum(1 for e in self.completed_episodes_buffer if e["lost"])
                / num_episodes
            )
            completion_rate = (
                sum(1 for e in self.completed_episodes_buffer if e["completed"])
                / num_episodes
            )

            wandb_metrics["textworld_train/episode_win_rate"] = win_rate
            wandb_metrics["textworld_train/episode_loss_rate"] = loss_rate
            wandb_metrics["textworld_train/episode_completion_rate"] = completion_rate

            # Episode performance
            avg_reward = (
                sum(e["cumulative_reward"] for e in self.completed_episodes_buffer)
                / num_episodes
            )
            avg_steps = (
                sum(e["num_steps"] for e in self.completed_episodes_buffer)
                / num_episodes
            )
            avg_final_score = (
                sum(e["final_score"] for e in self.completed_episodes_buffer)
                / num_episodes
            )

            wandb_metrics["textworld_train/avg_episode_reward"] = avg_reward
            wandb_metrics["textworld_train/avg_episode_steps"] = avg_steps
            wandb_metrics["textworld_train/avg_final_score"] = avg_final_score

            # Episode efficiency
            winning_episodes = [e for e in self.completed_episodes_buffer if e["won"]]
            if winning_episodes:
                avg_steps_to_win = sum(e["num_steps"] for e in winning_episodes) / len(
                    winning_episodes
                )
                wandb_metrics["textworld_train/avg_steps_to_win"] = avg_steps_to_win

            # Learning progress
            total_episodes = self.last_logged_episodes + num_episodes
            wandb_metrics["textworld_train/episodes_completed"] = total_episodes
            self.last_logged_episodes = total_episodes

            # Clear buffer
            self.completed_episodes_buffer = []

        # Step-level metrics
        if self.step_metrics_buffer:
            num_steps = len(self.step_metrics_buffer)

            # Action quality metrics
            avg_vrcli = (
                sum(s["vrcli_score"] for s in self.step_metrics_buffer) / num_steps
            )
            avg_format = (
                sum(s["format_score"] for s in self.step_metrics_buffer) / num_steps
            )
            avg_confidence = (
                sum(s["confidence_score"] for s in self.step_metrics_buffer) / num_steps
            )
            avg_combined = (
                sum(s["combined_score"] for s in self.step_metrics_buffer) / num_steps
            )

            wandb_metrics["textworld_train/avg_vrcli_score"] = avg_vrcli
            wandb_metrics["textworld_train/avg_format_score"] = avg_format
            wandb_metrics["textworld_train/avg_confidence_score"] = avg_confidence
            wandb_metrics["textworld_train/avg_combined_score"] = avg_combined

            # Response quality
            avg_response_length = (
                sum(s["response_tokens"] for s in self.step_metrics_buffer) / num_steps
            )
            wandb_metrics["textworld_train/avg_response_length"] = avg_response_length

            # Action selection quality
            avg_selected_rank = (
                sum(s["selected_rank"] for s in self.step_metrics_buffer) / num_steps
            )
            top_confidence_selected = (
                sum(1 for s in self.step_metrics_buffer if s["selected_rank"] == 0)
                / num_steps
            )

            wandb_metrics["textworld_train/avg_selected_alternative_rank"] = (
                avg_selected_rank
            )
            wandb_metrics["textworld_train/top_confidence_selected_rate"] = (
                top_confidence_selected
            )

            wandb_metrics["textworld_train/steps_taken"] = num_steps

            # Clear buffer
            self.step_metrics_buffer = []

        # Call parent wandb_log
        await super().wandb_log(wandb_metrics)

    async def evaluate(self, *args, **kwargs):
        """Evaluation method - implementation pending."""
        pass

    async def cleanup(self):
        """Clean up resources and temporary files."""
        # Remove generated game files
        for fp in list(self._generated_files):
            try:
                if os.path.exists(fp):
                    os.remove(fp)
            except Exception:
                pass
            finally:
                self._generated_files.discard(fp)
        # Remove temp dir if empty
        try:
            if os.path.isdir(self._temp_dir) and not os.listdir(self._temp_dir):
                os.rmdir(self._temp_dir)
        except Exception:
            pass

    def __del__(self):
        """Best-effort cleanup of temp files."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.cleanup())
            else:
                loop.run_until_complete(self.cleanup())
        except Exception:
            pass

    @classmethod
    def config_init(cls) -> Tuple[TextWorldEnvConfig, List[APIServerConfig]]:
        """Initialize default configuration for TextWorldEnv."""
        config = TextWorldEnvConfig(
            tokenizer_name="NousResearch/Hermes-4-Qwen3-14B-1-e3",
            wandb_name="textworld-qwen3-14b",
        )
        # When running with SLURM, configure to use SGLang servers on ports 9004-9007
        server_configs = [
            APIServerConfig(
                api_key="x",  # SGLang requires non-empty API key
                base_url=f"http://localhost:{port}/v1",  # Add /v1 for OpenAI API compatibility
                model_name="NousResearch/Hermes-4-Qwen3-14B-1-e3",
                server_type="openai",  # SGLang is OpenAI API compatible
                timeout=1200,
                num_max_requests_at_once=512,
                num_requests_for_eval=64,
                health_check=True,
            )
            for port in [9004, 9005, 9006, 9007]
        ]
        return config, server_configs


if __name__ == "__main__":
    import sys

    logger.info(f"TextWorld module started with args: {sys.argv}")
    TextWorldEnv.cli()
