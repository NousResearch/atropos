#!/usr/bin/env python3
"""
TextWorldEnv: Trainer environment for Microsoft TextWorld with VR-CLI

A trainer environment that wraps TextWorld game generator and Gym interface
to train LLMs using best-of-n pattern with function-call style actions.
Uses VR-CLI (Verifiable Rewards via Completion Likelihood Improvement) to score
predictions based on how well the model anticipates action outcomes.
"""

import asyncio
import copy
import json
import logging
import math
import os
import random
import shutil
import tempfile
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import Field

import textworld
import textworld.challenges
import textworld.gym
from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.type_definitions import Message
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer
from atroposlib.utils.tool_call_parser import parse_tool_call
from environments.game_environments.textworld.generation_utils import (
    generate_textworld_game,
)
from environments.game_environments.textworld.textworld_registry import (
    create_textworld_registry,
)
from textworld import EnvInfos, GameOptions
from textworld.gym.envs import TextworldGymEnv

from .agents.atropos_agent import AtroposAgent, AtroposAgentConfig
from .agents.atropos_memory_manager import (
    MEMORY_SYSTEM_PREREQUISITES_AVAILABLE,
    AtroposMemoryManager,
)

logger = logging.getLogger(__name__)


class TextWorldEnvConfig(BaseEnvConfig):
    """Configuration for the TextWorld environment trainer with VR-CLI."""

    env_name: str = "TextWorld"
    max_steps: int = 50
    challenge_name: str = "tw-simple"
    challenge_rewards: str = "sparse"  # Changed to sparse for VR-CLI
    challenge_goal: str = "detailed"
    challenge_test_mode: bool = False
    nb_rooms: int = 5
    nb_objects: int = 10
    quest_min_length: int = 3
    quest_max_length: int = 3
    quest_max_depth: int = 3
    grammar_theme: str = "house"
    grammar_include_adj: bool = True
    game_seed: Optional[int] = None
    max_token_length: int = 16384
    max_trajectory_tokens: int = 24576

    # VR-CLI specific configurations
    vrcli_enabled: bool = Field(
        default=True, description="Use VR-CLI scoring for action predictions"
    )
    vrcli_weight: float = Field(
        default=0.7,
        description="Weight for combining VR-CLI score with environment reward",
    )
    vrcli_discount_factor: float = Field(
        default=0.99,
        description="Discount factor for credit assignment in sparse reward setting",
    )

    debug_mode: bool = False

    enable_policy_thinking_summarization: bool = Field(
        default=True,
        description="Whether to use LLM-based summarization for thinking blocks in policy training data.",
    )
    max_policy_thinking_summary_tokens: int = Field(
        default=128, description="Maximum tokens for LLM-summarized thinking blocks."
    )

    game_generation_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "max_score": 10,
            "nb_rooms": 5,
            "nb_objects": 10,
            "quest_length": 3,
            "quest_breadth": 2,
            "include_take_action": True,
            "include_open_action": True,
            "include_drop_action": True,
            "include_go_action": True,
            "include_examine_action": True,
            "include_inventory_action": True,
            "include_look_action": True,
        }
    )
    game_file_path: Optional[str] = None
    
    # Registry configuration
    use_registry: bool = Field(
        default=True,
        description="Whether to use the registry for game selection"
    )
    registry_mode: str = Field(
        default="random",
        description="Registry mode: random, generated, challenge"
    )
    registry_generation_ratio: float = Field(
        default=0.7,
        description="Ratio of generated games vs pre-built (0.0 to 1.0)"
    )
    registry_difficulty: Optional[str] = Field(
        default="random",
        description="Difficulty: easy, medium, hard, expert, random"
    )
    registry_game_type: Optional[str] = Field(
        default=None,
        description="Game type: quest, puzzle, navigation, mixed (None for random)"
    )

    default_server_config: APIServerConfig = Field(
        default_factory=lambda: APIServerConfig(
            api_server_type="openai", model_name="gpt-3.5-turbo"
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

        # Episode-level cache for thinking block summarizations
        self.thinking_block_cache: Dict[int, str] = {}


class TextWorldEnv(BaseEnv):
    """Trainer environment for TextWorld using VR-CLI for action outcome prediction."""

    def __init__(
        self,
        config: TextWorldEnvConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = True,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: TextWorldEnvConfig = config
        self.episodes: Dict[str, TextWorldEpisodeState] = {}
        self._temp_dir = tempfile.mkdtemp(prefix="textworld_env_")

        # Initialize Memory Manager
        self.memory_manager = None
        if (
            self.config.atropos_agent_config.enable_memory
            and MEMORY_SYSTEM_PREREQUISITES_AVAILABLE
        ):
            self.memory_manager = AtroposMemoryManager(
                embedding_dim_config_val=self.config.atropos_agent_config.embedding_dim,
                player_id_for_logging=f"{self.config.atropos_agent_config.player_id_for_logging}_Memory",
            )
        elif self.config.atropos_agent_config.enable_memory:
            logger.warning(
                "Memory is enabled in config, but prerequisites are not met. Memory disabled."
            )

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
                                "description": "What you expect to observe after executing this command. Be specific about changes to the environment, your location, inventory, or game state.",
                            },
                        },
                        "required": ["command", "expected_outcome"],
                    },
                },
            }
        ]
        tools_json = json.dumps(self.textworld_tools, indent=2)

        # Policy agent system prompt with outcome prediction
        constructed_system_prompt = (
            "You are an AI agent playing a text-based adventure game who uses extreme long chains of thought "
            "to carefully plan your actions and predict their outcomes. Your goal is to follow the objective described "
            "at the start of the game. You interact with the world by providing text commands and predicting their outcomes."
            "\\n\\n"
            "You should:\\n"
            "1. Enclose your thoughts and internal monologue inside <think> </think> tags. Use extremely long chains "
            "of thought to carefully consider the game state, your objectives, and the likely outcomes of your actions.\\n"
            "2. Generate a memory summary inside <memory> </memory> tags that captures key information from this turn. "
            "Your memory should:\\n"
            "   - Build upon previous memories shown in 'Relevant Memories' if present\\n"
            "   - Note the outcome of your last action (did it match your prediction?)\\n"
            "   - Update your understanding of the game state, location, and inventory\\n"
            "   - Track progress toward objectives and any multi-step plans\\n"
            "   - Be concise but comprehensive (1-3 sentences)\\n"
            "3. Provide your action using the execute_command function call."
            "\\n\\n"
            f"<tools>\\n{tools_json}\\n</tools>\\n\\n"
            "For your function call, return a JSON object with function name and arguments "
            "within <tool_call> </tool_call> tags with the following schema:\\n"
            '<tool_call>\\n{"name": "execute_command", "arguments": {"command": "go north", "expected_outcome": "I move north to a new room"}}\\n</tool_call>\\n\\n'
            "EXAMPLE RESPONSE 1:\\n"
            "<think>\\n"
            "I'm in the kitchen. I see a stove and a fridge. The objective says to cook something. "
            "Let me check what's in the fridge first to see what ingredients are available."
            "\\n</think>\\n"
            "<memory>\\n"
            "Kitchen has stove and fridge. Main objective is cooking. Need to find ingredients."
            "\\n</memory>\\n"
            "<tool_call>\\n"
            """{"name": "execute_command", "arguments": {"command": "open fridge", "expected_outcome": "The fridge opens, revealing its contents. I expect to see various food items or ingredients inside that I can take and use for cooking."}}"""
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
            """{"name": "execute_command", "arguments": {"command": "take eggs", "expected_outcome": "I take the eggs from the fridge and add them to my inventory"}}"""
            "\\n</tool_call>\\n\\n"
            "EXAMPLE RESPONSE 3:\\n"
            "<think>\\n"
            "There's a locked door here and I have a key in my inventory. I should try using the key on the door."
            "\\n</think>\\n"
            "<memory>\\n"
            "Found locked door in current room. Have key in inventory that might open it."
            "\\n</memory>\\n"
            "<tool_call>\\n"
            """{"name": "execute_command", "arguments": {"command": "unlock door with key", "expected_outcome": "The key turns in the lock and the door unlocks. I should now be able to open the door and go through it."}}"""
            "\\n</tool_call>\\n\\n"
            "Remember: Your entire response must be exactly three XML blocks: <think>...</think> followed by <memory>...</memory> followed by <tool_call>...</tool_call>\\n\\n"
            "FINAL REMINDER: After your <think> block and <memory> block, you MUST wrap your JSON function call in <tool_call> tags. "
            "The JSON goes INSIDE the <tool_call> tags, not after them."
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

        self.agent = AtroposAgent(
            server_client=self.server,
            tokenizer=self.tokenizer,
            config=agent_cfg,
            memory_manager=self.memory_manager,  # Pass the memory manager
        )

        if self.config.debug_mode:
            logger.setLevel(logging.DEBUG)
            
        # Initialize registry if enabled
        self.registry = None
        if self.config.use_registry:
            self.registry = create_textworld_registry(
                generation_ratio=self.config.registry_generation_ratio,
                seed=self.config.seed if hasattr(self.config, 'seed') else None
            )

    async def setup(self):
        """Ensure prerequisites are met for TextWorld."""
        try:
            import textworld
        except ImportError:
            logger.error(
                "TextWorld library not found. Please install it to use TextWorldEnv."
            )
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
        self, episode_seed: Optional[int] = None
    ) -> Optional[TextWorldEpisodeState]:
        """Generate a new TextWorld game and initialize episode state."""
        episode_id = f"textworld-episode-{uuid.uuid4().hex}"
        current_game_seed = (
            episode_seed if episode_seed is not None else random.randint(0, 0xFFFFFFFF)
        )

        # Generate or select game using registry if enabled
        if self.config.use_registry and self.registry:
            try:
                game_file_path, game_config = self.registry.get_environment(
                    mode=self.config.registry_mode,
                    difficulty=self.config.registry_difficulty,
                    game_type=self.config.registry_game_type
                )
                
                if not game_file_path or not os.path.exists(game_file_path):
                    logger.error(f"Registry failed to generate game for episode {episode_id}")
                    return None
                    
                logger.info(f"Generated game from registry: {game_config}")
                
            except Exception as e:
                logger.error(f"Error using registry for game generation: {e}")
                return None
        else:
            # Use original generation logic
            options = GameOptions()
            options.seeds = current_game_seed
            options.nb_rooms = self.config.nb_rooms
            options.nb_objects = self.config.nb_objects
            options.chaining.min_length = self.config.quest_min_length
            options.chaining.max_length = self.config.quest_max_length
            options.chaining.max_depth = self.config.quest_max_depth
            options.grammar.theme = self.config.grammar_theme
            options.grammar.include_adj = self.config.grammar_include_adj

            challenge_settings = {
                "seed": current_game_seed,
                "rewards": self.config.challenge_rewards,
                "goal": self.config.challenge_goal,
                "test": self.config.challenge_test_mode,
            }

            try:
                game_file_path, game_object = generate_textworld_game(
                    challenge_name=self.config.challenge_name,
                    settings=challenge_settings,
                    options=options,
                    output_folder=self._temp_dir,
                    filename_prefix=f"{self.config.challenge_name}_ep{current_game_seed}",
                )

                if not game_file_path or not os.path.exists(game_file_path):
                    logger.error(f"Failed to generate game file for episode {episode_id}")
                    return None

            except Exception as e:
                logger.error(
                    f"Error generating game for {self.config.challenge_name} challenge: {e}"
                )
                return None

        requested_infos = EnvInfos(
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

            self.agent.new_game()

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
        """Provide a new, initialized TextWorldEpisodeState for trajectory collection."""
        episode_state = await self._get_or_create_episode(
            episode_seed=self.config.game_seed
        )
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
        # Apply chat template to get the full formatted text
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
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
                prefix_messages, tokenize=False, add_generation_prompt=True
            )
            prefix_tokens = self.tokenizer.encode(prefix_text, add_special_tokens=False)

            # Extract logprobs for the last message (the completion we're evaluating)
            all_logprobs = response.choices[0].logprobs.token_logprobs
            completion_logprobs = all_logprobs[len(prefix_tokens) :]
        else:
            # If there's only one message, evaluate the whole thing
            completion_logprobs = response.choices[0].logprobs.token_logprobs

        # Calculate perplexity: exp(-mean(log_probs))
        if completion_logprobs:
            mean_logprob = sum(completion_logprobs) / len(completion_logprobs)
            return math.exp(-mean_logprob)
        return float("inf")

    async def _vrcli_score(
        self, current_obs: str, predicted_outcome: str, actual_outcome: str
    ) -> float:
        """Score prediction quality using perplexity improvement."""
        if not predicted_outcome:
            return 0.0

        # Create message lists for base and prediction-conditioned perplexity
        base_messages = [
            {
                "role": "user",
                "content": f"Current state:\n{current_obs}\n\nWhat happens next?",
            },
            {"role": "assistant", "content": actual_outcome},
        ]

        prediction_messages = [
            {
                "role": "user",
                "content": f"Current state:\n{current_obs}\n\nPredicted outcome: {predicted_outcome}\n\nWhat actually happens?",
            },
            {"role": "assistant", "content": actual_outcome},
        ]

        # Calculate perplexities using server
        base_ppl = await self._calculate_perplexity_from_server(base_messages)
        pred_ppl = await self._calculate_perplexity_from_server(prediction_messages)

        # Score is improvement in perplexity
        return max(0.0, (base_ppl - pred_ppl) / base_ppl)

    def _get_action_history_from_agent(self) -> List[str]:
        """Extract the canonical action history from the agent."""
        action_history = []
        
        for turn_data in self.agent.game_log['turn']:
            if turn_data['selected_alternative'] is not None:
                selected_idx = turn_data['selected_alternative']
                if 0 <= selected_idx < len(turn_data['alternatives']):
                    choice = turn_data['alternatives'][selected_idx]
                    if not choice['api_error'] and choice['action_text']:
                        # Extract the actual command from the tool call
                        action_text = choice['action_text']
                        tool_name, arguments, is_error = parse_tool_call(
                            response=action_text, preferred_tags=["tool_call"]
                        )
                        if not is_error and tool_name == "execute_command":
                            command = arguments.get("command")
                            if command:
                                action_history.append(command.strip())
        
        return action_history
    
    def _create_env_copy_with_replay(self, ep_state: TextWorldEpisodeState) -> Optional[TextworldGymEnv]:
        """Create a new environment instance and replay actions to reach current state."""
        try:
            # Get the action history from the agent
            action_history = self._get_action_history_from_agent()
            
            # Register a new environment instance with the same game file
            requested_infos = textworld.EnvInfos(
                objective=True,
                inventory=True,
                description=True,
                score=True,
                won=True,
                lost=True,
                facts=True,
                last_action=True,
                feedback=True,
                moves=True,
                admissible_commands=True,
            )
            
            # Create a unique ID for this copy
            copy_env_id = f"{ep_state.episode_id}_copy_{random.randint(1000, 9999)}"
            
            registered_env_id = textworld.gym.register_game(
                ep_state.game_file,
                requested_infos,
                max_episode_steps=self.config.max_steps,
                name=copy_env_id,
            )
            
            # Create the environment
            env_copy = textworld.gym.make(registered_env_id)
            
            # Reset the environment
            obs, infos = env_copy.reset()
            
            # Replay all actions to reach the current state
            for past_action in action_history:
                obs, reward, done, infos = env_copy.step(past_action)
                if done:
                    break
                    
            return env_copy
        except Exception as e:
            logger.error(f"Failed to create environment copy with replay: {e}")
            return None

    async def _evaluate_candidates(
        self, ep_state: TextWorldEpisodeState, candidates: List[Tuple[str, str, str]]
    ) -> List[Dict[str, Any]]:
        """Evaluate each candidate by executing in copied environment."""
        evaluations = []

        for i, (action, prediction, response_text) in enumerate(candidates):
            if action is None:
                action = "look"  # Default action

            # Create environment copy with replay
            env_copy = self._create_env_copy_with_replay(ep_state)
            if env_copy is None:
                logger.error(f"Failed to create environment copy for candidate {i}")
                evaluations.append({
                    "action": action,
                    "prediction": prediction,
                    "vrcli_score": 0.0,
                    "env_reward": 0.0,
                    "combined_score": 0.0,
                    "error": True
                })
                continue

            try:
                # Execute action
                obs, reward, done, info = env_copy.step(action)
                actual_outcome = self._format_observation(obs, info)

                # Calculate VR-CLI score
                vrcli_score = await self._vrcli_score(
                    ep_state.last_formatted_obs, prediction, actual_outcome
                )

                # Combine scores
                combined_score = (
                    self.config.vrcli_weight * vrcli_score
                    + (1 - self.config.vrcli_weight) * reward
                )

                evaluations.append(
                    {
                        "index": i,
                        "action": action,
                        "prediction": prediction,
                        "vrcli_score": vrcli_score,
                        "env_reward": reward,
                        "combined_score": combined_score,
                        "response_text": response_text,
                        "actual_outcome": actual_outcome,
                        "done": done,
                        "info": info,
                    }
                )
            except Exception as e:
                logger.error(f"Error evaluating candidate {i}: {e}")
                evaluations.append(
                    {
                        "index": i,
                        "action": action,
                        "prediction": prediction,
                        "vrcli_score": 0.0,
                        "env_reward": 0.0,
                        "combined_score": 0.0,
                        "response_text": response_text,
                        "actual_outcome": "",
                        "done": False,
                        "info": {},
                    }
                )
            finally:
                # Clean up copied environment
                env_copy.close()

        return evaluations

    async def _next_step(
        self, ep_state: TextWorldEpisodeState, current_turn_num: int
    ) -> Tuple[Optional[ScoredDataGroup], bool]:
        """Execute one step of the TextWorld episode using VR-CLI evaluation."""

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
                return None, True
            current_observation = self._format_observation(raw_obs, infos)

        ep_state.last_formatted_obs = current_observation

        # 2. Generate candidate actions with predictions
        try:
            group_actions = await self.agent.generate_action(
                current_observation, n=self.config.group_size
            )
        except Exception as e:
            logger.error(
                f"[Episode: {ep_state.episode_id}] Error getting actions from agent: {e}"
            )
            return None, True

        if not group_actions:
            logger.error(
                f"[Episode: {ep_state.episode_id}] No actions received from agent"
            )
            return None, True

        # 3. Parse actions and predictions
        candidates = []
        for i, action_response in enumerate(group_actions):
            response_text = action_response["action_text"]
            print(f"\n=== Alternative {i} Full Response ===")
            print(response_text)
            print("=== End Response ===\n")
            action, prediction = self._parse_action_with_prediction(response_text)
            candidates.append((action, prediction, response_text))

        if not candidates:
            logger.error(f"[Episode: {ep_state.episode_id}] No valid actions parsed")
            return None, True

        # 4. Evaluate candidates using VR-CLI
        evaluations = await self._evaluate_candidates(ep_state, candidates)

        if not evaluations:
            logger.error(f"[Episode: {ep_state.episode_id}] No evaluations returned")
            return None, True

        # 5. Select best action based on combined score
        best_eval = max(evaluations, key=lambda x: x["combined_score"])
        best_idx = best_eval["index"]

        # 6. Record selected action with agent
        try:
            await self.agent.record_selected_action_and_learn_from_turn(
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
            return None, True

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

        # 9. Prepare training data
        sg_tokens = []
        sg_masks = []
        sg_messages = []
        scores = []

        # Get canonical history for all alternatives
        canonical_history = self.agent.get_final_canonical_dialogue()

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
                logger.error(f"Error tokenizing alternative {i}: {e}")
                sg_tokens.append([])
                sg_masks.append([])
                sg_messages.append(alternative_history)
                scores.append(0.0)

        # Create scored data group
        scored_data_group = ScoredDataGroup(
            tokens=sg_tokens,
            masks=sg_masks,
            scores=scores,
            messages=sg_messages,
            metadata={
                "turn_number": current_turn_num,
                "chosen_alternative_index": best_idx,
                "episode_id": ep_state.episode_id,
                "type": "policy_training_data",
                "vrcli_scores": [e["vrcli_score"] for e in evaluations],
                "env_rewards": [e["env_reward"] for e in evaluations],
            },
        )

        ep_state.policy_step_data.append(scored_data_group)

        return scored_data_group, ep_state.done

    async def collect_trajectories(
        self, item: Dict[str, Any]
    ) -> Tuple[List[ScoredDataGroup], List[Dict[str, Any]]]:
        """Run a full TextWorld episode collecting data for each step."""
        if not item or "episode_state" not in item:
            logger.error("Invalid item received - missing 'episode_state'")
            return [], []

        ep_state: TextWorldEpisodeState = item["episode_state"]
        if not ep_state:
            logger.error("Episode state is None")
            return [], []

        policy_sdgs_for_episode: List[ScoredDataGroup] = []

        try:
            # Execute episode turns
            for current_turn_num in range(ep_state.max_turns):
                if ep_state.done:
                    break

                scored_data_group_for_turn, episode_is_done_after_step = (
                    await self._next_step(ep_state, current_turn_num)
                )

                if scored_data_group_for_turn:
                    policy_sdgs_for_episode.append(scored_data_group_for_turn)

                if episode_is_done_after_step:
                    break

                if current_turn_num == ep_state.max_turns - 1 and not ep_state.done:
                    ep_state.done = True

        except Exception as e:
            logger.error(
                f"[Episode: {ep_state.episode_id}] Error during trajectory collection: {e}"
            )
            ep_state.done = True
        finally:
            # Apply credit assignment for sparse rewards
            self._apply_credit_assignment(ep_state, policy_sdgs_for_episode)
            # Clean up episode resources
            await self._cleanup_episode_resources(ep_state)

        logger.info(
            f"[Episode: {ep_state.episode_id}] Episode complete. Score: {ep_state.last_score}, Won: {ep_state.won}, Lost: {ep_state.lost}, Turns: {len(policy_sdgs_for_episode)}"
        )

        return policy_sdgs_for_episode, []

    def _apply_credit_assignment(
        self,
        ep_state: TextWorldEpisodeState,
        policy_sdgs_for_episode: List[ScoredDataGroup],
    ) -> None:
        """Apply credit assignment for sparse rewards using discounted returns."""
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
            # Get immediate reward from this step
            immediate_reward = ep_state.canonical_rewards[t]

            # Update discounted return
            discounted_return = (
                immediate_reward + self.config.vrcli_discount_factor * discounted_return
            )

            # Update scores for the chosen action
            sdg = policy_sdgs_for_episode[t]
            chosen_idx = ep_state.canonical_chosen_alternative_indices[t]

            if 0 <= chosen_idx < len(sdg["scores"]):
                # The scores already contain VR-CLI + immediate reward combination
                # Now we add the future discounted return to the chosen action
                new_scores = list(sdg["scores"])
                # Add the future return (not including immediate reward to avoid double counting)
                future_return = self.config.vrcli_discount_factor * (
                    discounted_return - immediate_reward
                )
                new_scores[chosen_idx] += future_return
                sdg["scores"] = new_scores

                # Log credit assignment info
                logger.debug(
                    f"[Episode: {ep_state.episode_id}] Turn {t}: "
                    f"immediate_reward={immediate_reward:.3f}, "
                    f"future_return={future_return:.3f}, "
                    f"total_return={new_scores[chosen_idx]:.3f}"
                )

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
        if ep_state.game_file:
            if self.registry:
                # Use registry cleanup which also handles JSON files
                self.registry.cleanup_game_file(ep_state.game_file)
            elif os.path.exists(ep_state.game_file):
                try:
                    os.remove(ep_state.game_file)
                    # Also remove JSON file if it exists
                    json_file = ep_state.game_file.replace('.z8', '.json')
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
        """Post-process policy agent trajectories with thinking block summarization."""
        if not trajectories:
            return trajectories

        if not isinstance(trajectories, list):
            trajectories = [trajectories] if trajectories is not None else []

        # Find episode state for cache
        episode_state = None
        episode_id = None
        for sdg in trajectories:
            if sdg and sdg.get("metadata", {}).get("episode_id"):
                episode_id = sdg["metadata"]["episode_id"]
                if episode_id in self.episodes:
                    episode_state = self.episodes[episode_id]
                    break

        thinking_block_cache = (
            episode_state.thinking_block_cache if episode_state else {}
        )

        # Process each ScoredDataGroup
        processed_trajectories = []
        for sdg_idx, sdg in enumerate(trajectories):
            if sdg is None:
                processed_trajectories.append(None)
                continue

            # Only process policy training data with thinking summarization
            # RM training data is already processed with thinking stripped
            if sdg.get("metadata", {}).get("type") == "policy_training_data":
                try:
                    processed_policy_sdg = await self._process_policy_training_data(
                        sdg, thinking_block_cache
                    )
                    processed_trajectories.append(processed_policy_sdg)
                except Exception as e:
                    logger.error(
                        f"Error processing policy ScoredDataGroup {sdg_idx}: {e}"
                    )
                    processed_trajectories.append(sdg)
            else:
                # Pass through RM training data unchanged
                processed_trajectories.append(sdg)

        # Log cache efficiency
        if thinking_block_cache:
            total_messages = sum(
                len(sdg.get("messages", [])) for sdg in trajectories if sdg
            )
            logger.info(
                f"Episode-level thinking block summarization: {len(thinking_block_cache)} unique blocks "
                f"processed across {len(trajectories)} turns with {total_messages} total alternatives"
            )

        # Clean up episode state
        if episode_id and episode_id in self.episodes:
            del self.episodes[episode_id]

        return processed_trajectories

    async def _process_policy_training_data(
        self, sdg: ScoredDataGroup, thinking_block_cache: Dict[int, str]
    ) -> ScoredDataGroup:
        """Process policy training data with thinking block summarization while keeping latest message raw."""
        if not self.config.enable_policy_thinking_summarization:
            return sdg

        processed_messages = []
        processed_tokens = []
        processed_masks = []

        for alt_idx, alt_messages in enumerate(sdg["messages"]):
            if not alt_messages:
                processed_messages.append(alt_messages)
                processed_tokens.append(
                    sdg["tokens"][alt_idx] if alt_idx < len(sdg["tokens"]) else []
                )
                processed_masks.append(
                    sdg["masks"][alt_idx] if alt_idx < len(sdg["masks"]) else []
                )
                continue

            alt_processed_messages = []
            for msg_idx, msg in enumerate(alt_messages):
                if msg_idx == len(alt_messages) - 1:
                    # Keep the last message raw for training
                    alt_processed_messages.append(msg.copy())
                elif (
                    msg["role"] == "assistant"
                    and self.config.enable_policy_thinking_summarization
                ):
                    # Check cache for thinking block summarization
                    original_content = msg["content"]
                    cache_key = hash(original_content)

                    if cache_key in thinking_block_cache:
                        processed_content = thinking_block_cache[cache_key]
                    else:
                        # Summarize thinking blocks
                        try:
                            from atroposlib.utils.message_history_utils import (
                                summarize_thinking_block,
                            )

                            processed_content = await summarize_thinking_block(
                                original_content,
                                self.server,
                                self.tokenizer,
                                self.config.max_policy_thinking_summary_tokens,
                            )
                            thinking_block_cache[cache_key] = processed_content
                        except Exception as e:
                            logger.warning(f"Failed to summarize thinking block: {e}")
                            from atroposlib.utils.message_history_utils import (
                                strip_thinking,
                            )

                            processed_content = strip_thinking(original_content)
                            thinking_block_cache[cache_key] = processed_content

                    processed_msg = msg.copy()
                    processed_msg["content"] = processed_content
                    alt_processed_messages.append(processed_msg)
                else:
                    alt_processed_messages.append(msg.copy())

            # Re-tokenize processed messages
            try:
                tokenized_output = tokenize_for_trainer(
                    self.tokenizer, alt_processed_messages
                )
                processed_messages.append(alt_processed_messages)
                processed_tokens.append(tokenized_output["tokens"])
                processed_masks.append(tokenized_output["masks"])
            except Exception as e:
                logger.error(
                    f"Error re-tokenizing processed policy messages for alt {alt_idx}: {e}"
                )
                processed_messages.append(alt_messages)
                processed_tokens.append(
                    sdg["tokens"][alt_idx] if alt_idx < len(sdg["tokens"]) else []
                )
                processed_masks.append(
                    sdg["masks"][alt_idx] if alt_idx < len(sdg["masks"]) else []
                )

        return ScoredDataGroup(
            tokens=processed_tokens,
            masks=processed_masks,
            scores=sdg["scores"],
            messages=processed_messages,
            metadata=sdg["metadata"],
        )

    async def evaluate(self, *args, **kwargs):
        """Evaluation method - implementation pending."""
        pass

    async def cleanup(self):
        """Clean up resources."""
        # Clean up any remaining registry files
        if self.registry:
            self.registry.cleanup_all()
            
        if (
            hasattr(self, "_temp_dir")
            and self._temp_dir
            and os.path.exists(self._temp_dir)
        ):
            try:
                shutil.rmtree(self._temp_dir)
            except Exception as e:
                logger.warning(
                    f"Error cleaning up temporary directory {self._temp_dir}: {e}"
                )

    def __del__(self):
        """Ensure cleanup runs even if explicit cleanup isn't called."""
        try:
            import asyncio

            if (
                hasattr(self, "_temp_dir")
                and self._temp_dir
                and os.path.exists(self._temp_dir)
            ):
                shutil.rmtree(self._temp_dir)
        except Exception:
            pass

    @classmethod
    def config_init(cls) -> Tuple[TextWorldEnvConfig, List[APIServerConfig]]:
        """Initialize default configuration for TextWorldEnv."""
        config = TextWorldEnvConfig()
        server_configs = [config.default_server_config]
        return config, server_configs


if __name__ == "__main__":
    TextWorldEnv.cli()
