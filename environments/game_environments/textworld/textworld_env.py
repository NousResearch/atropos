#!/usr/bin/env python3
"""
TextWorldEnv: Trainer environment for Microsoft TextWorld

A trainer environment that wraps TextWorld game generator and Gym interface 
to train LLMs using best-of-n pattern with function-call style actions.
Integrates with AtroposAgent and AtroposRM for policy and reward modeling.
"""

import asyncio
import json
import logging
import os
import random
import shutil
import tempfile
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import textworld
import textworld.challenges
import textworld.gym
from textworld import EnvInfos, GameOptions
from textworld.gym.envs import TextworldGymEnv
from pydantic import Field

from atroposlib.envs.base import (
    BaseEnv,
    BaseEnvConfig,
    APIServerConfig,
    ScoredDataGroup,
)
from atroposlib.type_definitions import Message, AtroposAgentAction
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer
from atroposlib.utils.tool_call_parser import parse_tool_call
from atroposlib.utils.message_history_utils import prepare_reward_model_input

from environments.game_environments.textworld.generation_utils import generate_textworld_game
from .agents.atropos_agent import AtroposAgent, AtroposAgentConfig
from .agents.atropos_rm import AtroposRM, AtroposRMConfig, RMJudgementLog
from .agents.atropos_memory_manager import AtroposMemoryManager, MEMORY_SYSTEM_PREREQUISITES_AVAILABLE

logger = logging.getLogger(__name__)


class TextWorldEnvConfig(BaseEnvConfig):
    """Configuration for the TextWorld environment trainer."""

    env_name: str = "TextWorld"
    max_steps: int = 50
    challenge_name: str = "tw-simple"
    challenge_rewards: str = "dense"
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
    agent_config: Optional[AtroposAgentConfig] = None
    rm_config: Optional[AtroposRMConfig] = None
    rm_reward_discount_factor: float = Field(
        default=0.99, 
        description="Discount factor for RM Q-value accuracy scoring."
    )
    debug_mode: bool = False

    enable_policy_thinking_summarization: bool = Field(
        default=True, 
        description="Whether to use LLM-based summarization for thinking blocks in policy training data."
    )
    max_policy_thinking_summary_tokens: int = Field(
        default=128,
        description="Maximum tokens for LLM-summarized thinking blocks."
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

    default_server_config: APIServerConfig = Field(
        default_factory=lambda: APIServerConfig(api_server_type="openai", model_name="gpt-3.5-turbo")
    )
    policy_agent_server_config: Optional[APIServerConfig] = None
    rm_agent_server_config: Optional[APIServerConfig] = None

    atropos_agent_config: AtroposAgentConfig = Field(default_factory=AtroposAgentConfig) 
    atropos_rm_config: AtroposRMConfig = Field(default_factory=AtroposRMConfig) 

    class Config:
        extra = 'forbid'
        arbitrary_types_allowed = True


class TextWorldEpisodeState:
    """Stores per-episode state for a TextWorld game when using AtroposAgent and AtroposRM."""
    
    def __init__(self, episode_id: str, game_file: str, textworld_env_instance: TextworldGymEnv, 
                 initial_obs: str, initial_infos: Dict[str, Any], max_steps: int):
        self.episode_id: str = episode_id
        self.game_file: str = game_file
        self.textworld_env: TextworldGymEnv = textworld_env_instance
        self.initial_formatted_obs: str = initial_obs
        self.initial_infos: Dict[str, Any] = initial_infos
        
        self.rm_judgement_history: List[RMJudgementLog] = [] 
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

        self.canonical_rewards: List[float] = []
        self.canonical_chosen_alternative_indices: List[int] = []
        
        # Episode-level cache for thinking block summarizations
        self.thinking_block_cache: Dict[int, str] = {}


class TextWorldEnv(BaseEnv):
    """Trainer environment for TextWorld integrating AtroposAgent and AtroposRM."""

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
        if self.config.atropos_agent_config.enable_memory and MEMORY_SYSTEM_PREREQUISITES_AVAILABLE:
            self.memory_manager = AtroposMemoryManager(
                embedding_dim_config_val=self.config.atropos_agent_config.embedding_dim,
                player_id_for_logging=f"{self.config.atropos_agent_config.player_id_for_logging}_Memory"
            )
        elif self.config.atropos_agent_config.enable_memory:
            logger.warning("Memory is enabled in config, but prerequisites are not met. Memory disabled.")

        # Define TextWorld command execution tool
        self.textworld_tools = [
            {
                "type": "function",
                "function": {
                    "name": "execute_command",
                    "description": "Execute a text command in the adventure game.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                             "command": {
                                "type": "string", 
                                "description": "The command to execute in the game."
                            }
                        },
                        "required": ["command"]
                    }
                }
            }
        ]
        tools_json = json.dumps(self.textworld_tools, indent=2)

        # Policy agent system prompt
        constructed_system_prompt = (
            "You are a long-thinking AI agent playing a text-based adventure game. Your goal is to follow the objective described "
            "at the start of the game. You interact with the world by providing text commands."
            "\\n\\n"
            "Carefully observe the room descriptions, your inventory, and any feedback from your previous actions. "
            "Think step-by-step about how to achieve the objective."
            "\\n\\n"
            "You MUST first output your thoughts and reasoning process within <think> </think> XML tags. " # Emphasize thinking block
            "After your thoughts, you MUST call the 'execute_command' function to provide your chosen text command. "
            "Do NOT output the command directly as text. Use the tool."
            "\\n\\n"
            f"<tools>\\n{tools_json}\\n</tools>\\n\\n"
            "Your function call should be a JSON object with the function name ('execute_command') and the 'command' argument, "
            "enclosed within <tool_call> </tool_call> tags. Example format:"
            "\\n\\n"
            "<think>\\n"
            "The player is in a dark room. There's a door to the north. The objective is to find the treasure. "
            "I should try opening the door first, or perhaps look for a light source. "
            "Going north seems like a direct approach to explore further. "
            "I will try to go north."
            "\\n</think>\\n"
            "<tool_call>\\n"
            '''{"name": "execute_command", "arguments": {"command": "go north"}}'''
            "\\n</tool_call>\\n\\n" # Corrected example arguments to match tool schema
            "Your response MUST follow this format exactly: <think>...</think> followed by <tool_call>...</tool_call>."
        )
        # Ensure AtroposAgentConfig is instantiated
        agent_cfg = self.config.atropos_agent_config if self.config.atropos_agent_config is not None else AtroposAgentConfig()
        agent_cfg.system_prompt = constructed_system_prompt 
        if self.config.policy_agent_server_config and self.config.policy_agent_server_config.model_name:
            agent_cfg.model_id = self.config.policy_agent_server_config.model_name
        else:
            agent_cfg.model_id = self.config.default_server_config.model_name
        
        self.agent = AtroposAgent(
            server_client=self.server, 
            tokenizer=self.tokenizer, 
            config=agent_cfg,
            memory_manager=self.memory_manager # Pass the memory manager
        )
        # Store the system prompt that will be used for the policy agent message history
        # TODO: Remove this? -> self.agent.system_prompt_content should be the source of truth
        # self.policy_agent_system_prompt_content = agent_cfg.system_prompt # REMOVED - Redundant
 
        # Ensure AtroposRMConfig is instantiated
        rm_cfg = self.config.atropos_rm_config if self.config.atropos_rm_config is not None else AtroposRMConfig()
        # Set RM model_id, defaulting to policy agent's model if not specified
        if self.config.rm_agent_server_config and self.config.rm_agent_server_config.model_name:
            rm_cfg.model_id = self.config.rm_agent_server_config.model_name
        elif agent_cfg.model_id: # Default to policy agent's model_id if RM specific is not set
            rm_cfg.model_id = agent_cfg.model_id
        else: # Fallback to default server config if policy also didn't have one (should not happen with current logic)
            rm_cfg.model_id = self.config.default_server_config.model_name

        self.rm = AtroposRM(
            server_client=self.server,
            tokenizer=self.tokenizer,
            config=rm_cfg
        )
 
        if self.config.debug_mode:
            logger.setLevel(logging.DEBUG)

    async def setup(self):
        """Ensure prerequisites are met for TextWorld."""
        try:
            import textworld
        except ImportError:
            logger.error("TextWorld library not found. Please install it to use TextWorldEnv.")
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
            formatted_obs += f"Feedback from last action ('{infos['last_action']}'):\n{feedback}\n"
        
        return formatted_obs.strip()

    async def _get_or_create_episode(self, episode_seed: Optional[int] = None) -> Optional[TextWorldEpisodeState]:
        """Generate a new TextWorld game and initialize episode state."""
        episode_id = f"textworld-episode-{uuid.uuid4().hex}"
        current_game_seed = episode_seed if episode_seed is not None else random.randint(0, 0xFFFFFFFF)

        # Create GameOptions
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
            'seed': current_game_seed,
            'rewards': self.config.challenge_rewards,
            'goal': self.config.challenge_goal,
            'test': self.config.challenge_test_mode
        }

        try:
            game_file_path, game_object = generate_textworld_game(
                challenge_name=self.config.challenge_name,
                settings=challenge_settings,
                options=options,
                output_folder=self._temp_dir,
                filename_prefix=f"{self.config.challenge_name}_ep{current_game_seed}"
            )
            
            if not game_file_path or not os.path.exists(game_file_path):
                logger.error(f"Failed to generate game file for episode {episode_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating game for {self.config.challenge_name} challenge: {e}")
            return None

        requested_infos = EnvInfos(
            description=True, inventory=True, objective=True, score=True, 
            max_score=True, won=True, lost=True, facts=True, 
            last_action=True, feedback=True, moves=True, admissible_commands=True
        )
        
        registered_env_id = textworld.gym.register_game(
            game_file_path, requested_infos, max_episode_steps=self.config.max_steps, name=episode_id
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
            logger.error(f"Failed to setup gym environment for episode {episode_id}: {e}")
            if os.path.exists(game_file_path):
                try: 
                    os.remove(game_file_path)
                except OSError: 
                    pass
            return None

    async def get_next_item(self) -> Optional[Dict[str, Any]]:
        """Provide a new, initialized TextWorldEpisodeState for trajectory collection."""
        episode_state = await self._get_or_create_episode(episode_seed=self.config.game_seed)
        if episode_state is None:
            logger.error("Failed to create new TextWorld episode.")
            return None
        
        return {"episode_state": episode_state, "episode_id": episode_state.episode_id}

    def _parse_action(self, agent_response_text: str) -> Optional[str]:
        """Parse agent response to extract TextWorld command from tool call."""
        if not agent_response_text:
            return None

        tool_name, arguments, is_error = parse_tool_call(
            response=agent_response_text,
            preferred_tags=["tool_call"]
        )

        if is_error or tool_name != "execute_command":
            return None

        parsed_command = arguments.get("command")
        if not parsed_command or not isinstance(parsed_command, str):
            return None
        
        parsed_command = parsed_command.strip()
        return parsed_command if parsed_command else None

    async def _next_step(
        self, ep_state: TextWorldEpisodeState, current_turn_num: int
    ) -> Tuple[Optional[ScoredDataGroup], bool]:
        """Execute one step of the TextWorld episode using AtroposAgent and AtroposRM."""
        
        # 1. Construct observation for agent
        if current_turn_num == 0:
            current_observation = ep_state.initial_formatted_obs
        else:
            raw_obs = ep_state.last_env_raw_observation
            infos = ep_state.last_env_infos
            if raw_obs is None or infos is None:
                logger.error(f"[Episode: {ep_state.episode_id}] Missing observation data for turn {current_turn_num + 1}")
                return None, True
            current_observation = self._format_observation(raw_obs, infos)

        # 2. Get alternative actions from Agent
        try:
            group_actions: List[AtroposAgentAction] = await self.agent.generate_action(
                current_observation, n=self.config.group_size
            )
        except Exception as e:
            logger.error(f"[Episode: {ep_state.episode_id}] Error getting actions from agent: {e}")
            return None, True

        if not group_actions:
            logger.error(f"[Episode: {ep_state.episode_id}] No actions received from agent")
            return None, True

        # 3. Evaluate alternatives with RM
        policy_alternatives_for_rm_eval = []
        for alt_action in group_actions:
            parsed_cmd = self._parse_action(alt_action['action_text'])
            
            # Get the canonical history up to this point
            history_for_rm_alt = self.agent.get_final_canonical_dialogue()
            # Add the alternative action to evaluate
            history_for_rm_alt.append(Message(role="assistant", content=alt_action['action_text'], reward=None))
            
            policy_alternatives_for_rm_eval.append({
                "parsed_command": parsed_cmd,
                "raw_agent_response": alt_action['action_text'],
                "agent_history_for_rm": history_for_rm_alt 
            })

        # Evaluate alternatives with RM
        alternative_rm_scores: List[float] = []
        all_rm_judgements_this_step: List[RMJudgementLog] = []
        rm_evaluation_tasks = []

        for i, policy_alt_data in enumerate(policy_alternatives_for_rm_eval):
            rm_evaluation_tasks.append(self.rm.generate_g_judgements(
                num_judgements_g=1,
                game_history_window=policy_alt_data["agent_history_for_rm"],
                game_seed_for_logging=self.config.game_seed,
                turn_idx_for_logging=current_turn_num,
                policy_action_candidate_idx_for_logging=i
            ))
        
        if rm_evaluation_tasks:
            try:
                rm_judgement_log_groups = await asyncio.gather(*rm_evaluation_tasks)
            except Exception as e:
                logger.error(f"[Episode: {ep_state.episode_id}] Error during RM evaluation: {e}")
                rm_judgement_log_groups = [[RMJudgementLog(api_error=True, parsed_q_value=0.0)] * 1] * len(policy_alternatives_for_rm_eval)
        else:
            rm_judgement_log_groups = [] 

        for i, rm_judgements_for_this_alt in enumerate(rm_judgement_log_groups):
            all_rm_judgements_this_step.extend(rm_judgements_for_this_alt)
            valid_q_values = [
                j["parsed_q_value"] for j in rm_judgements_for_this_alt 
                if not j["api_error"] and not j["q_value_parse_error"] and j["parsed_q_value"] is not None
            ]
            if not valid_q_values:
                alternative_rm_scores.append(0.0) 
            else:
                mean_q_value = sum(valid_q_values) / len(valid_q_values)
                alternative_rm_scores.append(mean_q_value)

        ep_state.rm_judgement_history.extend(all_rm_judgements_this_step)

        # 4. Select best action
        if not alternative_rm_scores: 
            logger.error(f"[Episode: {ep_state.episode_id}] No RM scores available")
            return None, True
            
        best_alternative_idx = alternative_rm_scores.index(max(alternative_rm_scores))
        chosen_policy_alt_data = policy_alternatives_for_rm_eval[best_alternative_idx]
        chosen_action_command = chosen_policy_alt_data["parsed_command"]

        if chosen_action_command is None:
            chosen_action_command = "look"

        # 5. Record selected action with Agent
        try:
            await self.agent.record_selected_action_and_learn_from_turn(selected_action_index=best_alternative_idx)
        except Exception as e:
            logger.error(f"[Episode: {ep_state.episode_id}] Error recording selected action: {e}")
            return None, True

        # 6. Execute action in TextWorld
        try:
            raw_obs_next, immediate_score_from_env, done_from_env, infos_next = ep_state.textworld_env.step(chosen_action_command)
        except Exception as e:
            logger.error(f"[Episode: {ep_state.episode_id}] Error stepping TextWorld environment: {e}")
            ep_state.done = True
            ep_state.canonical_rewards.append(0.0)
            ep_state.canonical_chosen_alternative_indices.append(best_alternative_idx)
            return None, True 

        # 7. Update episode state
        ep_state.cumulative_reward += immediate_score_from_env
        ep_state.canonical_rewards.append(immediate_score_from_env)
        ep_state.canonical_chosen_alternative_indices.append(best_alternative_idx)
        ep_state.done = done_from_env 
        ep_state.last_score = infos_next.get("score", ep_state.last_score)
        ep_state.moves = infos_next.get("moves", ep_state.moves)
        
        if ep_state.done:
            ep_state.won = infos_next.get("won", False)
            ep_state.lost = infos_next.get("lost", False)
        
        ep_state.last_env_raw_observation = raw_obs_next
        ep_state.last_env_infos = infos_next

        # 8. Prepare ScoredDataGroup for policy agent
        sg_tokens: List[List[int]] = []
        sg_masks: List[List[int]] = []
        sg_messages: List[List[Message]] = [] 

        for policy_alt_data in policy_alternatives_for_rm_eval:
            history_to_tokenize = policy_alt_data["agent_history_for_rm"]
            
            try:
                tokenized_output = tokenize_for_trainer(self.tokenizer, history_to_tokenize)
                sg_tokens.append(tokenized_output["tokens"])
                sg_masks.append(tokenized_output["masks"])
                sg_messages.append(history_to_tokenize)
            except Exception as e:
                logger.error(f"[Episode: {ep_state.episode_id}] Error tokenizing history: {e}")
                sg_tokens.append([])
                sg_masks.append([])
                sg_messages.append(history_to_tokenize) 
        
        current_step_scored_data = ScoredDataGroup(
            tokens=sg_tokens,
            masks=sg_masks,
            scores=list(alternative_rm_scores), 
            messages=sg_messages,
            metadata={
                "turn_number": current_turn_num, 
                "chosen_alternative_index": best_alternative_idx,
                "episode_id": ep_state.episode_id,
                "type": "policy_training_data"
            }
        )
        ep_state.policy_step_data.append(current_step_scored_data)
        
        return current_step_scored_data, ep_state.done

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
                
                scored_data_group_for_turn, episode_is_done_after_step = await self._next_step(
                    ep_state, current_turn_num
                )

                if scored_data_group_for_turn:
                    policy_sdgs_for_episode.append(scored_data_group_for_turn)
                
                if episode_is_done_after_step:
                    break
                
                if current_turn_num == ep_state.max_turns - 1 and not ep_state.done:
                    ep_state.done = True

        except Exception as e:
            logger.error(f"[Episode: {ep_state.episode_id}] Error during trajectory collection: {e}")
            ep_state.done = True
        finally:
            # Post-episode processing
            await self._process_episode_completion(ep_state, policy_sdgs_for_episode)

        logger.info(f"[Episode: {ep_state.episode_id}] Finalizing episode. Score: {ep_state.last_score}, Won: {ep_state.won}, Lost: {ep_state.lost}, Processed Turns: {len(policy_sdgs_for_episode)}")
        
        # Combine policy and RM training data
        all_scored_data_groups = []
        
        # Add policy training data
        all_scored_data_groups.extend(policy_sdgs_for_episode)
        
        # Add RM training data if available
        if hasattr(ep_state, 'rm_training_data') and ep_state.rm_training_data:
            all_scored_data_groups.extend(ep_state.rm_training_data)
            logger.info(f"[Episode: {ep_state.episode_id}] Returning {len(policy_sdgs_for_episode)} policy + {len(ep_state.rm_training_data)} RM ScoredDataGroups")
        
        return all_scored_data_groups, []

    async def _process_episode_completion(self, ep_state: TextWorldEpisodeState, policy_sdgs_for_episode: List[ScoredDataGroup]) -> None:
        """Process episode completion including reward calculations, RM data generation, and cleanup."""
        if not policy_sdgs_for_episode:
            await self._cleanup_episode_resources(ep_state)
            return

        # Calculate final outcome reward
        final_outcome_reward = self._calculate_final_outcome_reward(ep_state)
        
        # Update policy training data with discounted returns
        self._update_policy_returns(policy_sdgs_for_episode, ep_state, final_outcome_reward)
        
        # Generate and send RM training data
        await self._generate_and_send_rm_training_data(policy_sdgs_for_episode, ep_state, final_outcome_reward)
        
        # Clean up episode resources
        await self._cleanup_episode_resources(ep_state)

    def _calculate_final_outcome_reward(self, ep_state: TextWorldEpisodeState) -> float:
        """Calculate the final outcome reward based on episode completion status."""
        if ep_state.won:
            return 1.0
        elif ep_state.lost:
            return -1.0
        return 0.0

    def _update_policy_returns(self, policy_sdgs_for_episode: List[ScoredDataGroup], 
                              ep_state: TextWorldEpisodeState, final_outcome_reward: float) -> None:
        """Update policy training data with Monte Carlo discounted returns."""
        num_canonical_steps = len(ep_state.canonical_rewards)
        
        if not policy_sdgs_for_episode or num_canonical_steps == 0:
            return

        if len(policy_sdgs_for_episode) != num_canonical_steps:
            logger.error(f"[Episode: {ep_state.episode_id}] Mismatch between policy SDGs and canonical rewards")

        # Calculate discounted returns backwards
        discounted_return = final_outcome_reward
        for t in range(num_canonical_steps - 1, -1, -1):
            current_step_reward = ep_state.canonical_rewards[t]
            discounted_return = current_step_reward + self.config.rm_reward_discount_factor * discounted_return
            
            if t < len(policy_sdgs_for_episode):
                sdg_t = policy_sdgs_for_episode[t]
                chosen_idx = ep_state.canonical_chosen_alternative_indices[t]
                
                if 0 <= chosen_idx < len(sdg_t["scores"]):
                    new_scores = list(sdg_t["scores"])
                    new_scores[chosen_idx] = discounted_return
                    sdg_t["scores"] = new_scores

    async def _generate_and_send_rm_training_data(self, policy_sdgs_for_episode: List[ScoredDataGroup], 
                                                 ep_state: TextWorldEpisodeState, final_outcome_reward: float) -> None:
        """Generate RM training data and store it for later processing."""
        rm_training_sdgs = self._generate_rm_training_data(policy_sdgs_for_episode, ep_state, final_outcome_reward)
        
        if rm_training_sdgs:
            logger.info(f"[Episode: {ep_state.episode_id}] Total {len(rm_training_sdgs)} RM ScoredDataGroups prepared")
            # Store RM training data in episode state for now
            # TODO: Determine proper handling of RM training data within the base class pattern
            ep_state.rm_training_data = rm_training_sdgs

    def _generate_rm_training_data(self, policy_sdgs_for_episode: List[ScoredDataGroup], 
                                  ep_state: TextWorldEpisodeState, final_outcome_reward: float) -> List[ScoredDataGroup]:
        """Generate RM training data from policy episode data."""
        num_canonical_steps = len(ep_state.canonical_rewards)
        if not policy_sdgs_for_episode or num_canonical_steps == 0:
            return []

        # Calculate canonical path discounted returns
        canonical_discounted_returns = self._calculate_canonical_returns(ep_state, final_outcome_reward)
        
        rm_training_sdgs = []
        for turn_idx, policy_sdg_for_turn in enumerate(policy_sdgs_for_episode):
            if not (0 <= turn_idx < num_canonical_steps):
                continue

            num_alternatives_this_turn = len(policy_sdg_for_turn["messages"])
            if num_alternatives_this_turn != self.config.group_size:
                continue

            # Create RM training targets
            rm_target_scores = self._create_rm_target_scores(
                policy_sdg_for_turn, ep_state, turn_idx, canonical_discounted_returns
            )
            
            # Process RM training data
            try:
                processed_rm_sdg = self._process_rm_training_sdg(
                    policy_sdg_for_turn, rm_target_scores, turn_idx, ep_state.episode_id
                )
                rm_training_sdgs.append(processed_rm_sdg)
            except Exception as e:
                logger.error(f"[Episode: {ep_state.episode_id}] Error processing RM training data: {e}")

        return rm_training_sdgs

    def _calculate_canonical_returns(self, ep_state: TextWorldEpisodeState, final_outcome_reward: float) -> List[float]:
        """Calculate discounted returns for the canonical path."""
        num_canonical_steps = len(ep_state.canonical_rewards)
        canonical_discounted_returns = [0.0] * num_canonical_steps
        current_discounted_return = final_outcome_reward
        
        for t_idx in range(num_canonical_steps - 1, -1, -1):
            current_step_reward = ep_state.canonical_rewards[t_idx]
            current_discounted_return = current_step_reward + self.config.rm_reward_discount_factor * current_discounted_return
            canonical_discounted_returns[t_idx] = current_discounted_return
            
        return canonical_discounted_returns

    def _create_rm_target_scores(self, policy_sdg_for_turn: ScoredDataGroup, ep_state: TextWorldEpisodeState, 
                                turn_idx: int, canonical_discounted_returns: List[float]) -> List[float]:
        """Create target scores for RM training."""
        num_alternatives = len(policy_sdg_for_turn["messages"])
        chosen_alternative_idx = ep_state.canonical_chosen_alternative_indices[turn_idx]
        
        rm_target_scores = []
        for alt_idx in range(num_alternatives):
            if alt_idx == chosen_alternative_idx:
                target_score = canonical_discounted_returns[turn_idx]
            else:
                target_score = policy_sdg_for_turn["scores"][alt_idx]
            rm_target_scores.append(target_score)
            
        return rm_target_scores

    def _process_rm_training_sdg(self, policy_sdg_for_turn: ScoredDataGroup, rm_target_scores: List[float], 
                                turn_idx: int, episode_id: str) -> ScoredDataGroup:
        """Process a single RM training ScoredDataGroup."""
        raw_rm_sdg = ScoredDataGroup(
            tokens=policy_sdg_for_turn["tokens"],
            masks=policy_sdg_for_turn["masks"],
            scores=rm_target_scores,
            messages=policy_sdg_for_turn["messages"],
            metadata={
                "turn_number": turn_idx,
                "episode_id": episode_id,
                "type": "rm_training_data_raw"
            }
        )
        
        processed_rm_sdg = prepare_reward_model_input(
            scored_data_group=raw_rm_sdg,
            tokenizer=self.tokenizer,
            max_tokens=self.config.max_token_length,
            strip_thinking_from_history=True,
            summarize_thinking_with_llm=False,
            server_client=None,
            max_thinking_summary_tokens=0
        )
        
        processed_rm_sdg["metadata"]["type"] = "rm_training_data_processed"
        return processed_rm_sdg

    async def _cleanup_episode_resources(self, ep_state: TextWorldEpisodeState) -> None:
        """Clean up episode resources including environment and game files."""
        if ep_state.textworld_env:
            try:
                ep_state.textworld_env.close()
            except Exception as e:
                logger.warning(f"[Episode: {ep_state.episode_id}] Error closing TextWorld environment: {e}")

        if ep_state.game_file and os.path.exists(ep_state.game_file):
            try:
                os.remove(ep_state.game_file)
            except OSError as e:
                logger.warning(f"[Episode: {ep_state.episode_id}] Error removing game file: {e}")

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
        
        thinking_block_cache = episode_state.thinking_block_cache if episode_state else {}
        
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
                    processed_policy_sdg = await self._process_policy_training_data(sdg, thinking_block_cache)
                    processed_trajectories.append(processed_policy_sdg)
                except Exception as e:
                    logger.error(f"Error processing policy ScoredDataGroup {sdg_idx}: {e}")
                    processed_trajectories.append(sdg)
            else:
                # Pass through RM training data unchanged
                processed_trajectories.append(sdg)
        
        # Log cache efficiency
        if thinking_block_cache:
            total_messages = sum(len(sdg.get("messages", [])) for sdg in trajectories if sdg)
            logger.info(
                f"Episode-level thinking block summarization: {len(thinking_block_cache)} unique blocks "
                f"processed across {len(trajectories)} turns with {total_messages} total alternatives"
            )
        
        # Clean up episode state
        if episode_id and episode_id in self.episodes:
            del self.episodes[episode_id]
        
        return processed_trajectories

    async def _process_policy_training_data(self, sdg: ScoredDataGroup, thinking_block_cache: Dict[int, str]) -> ScoredDataGroup:
        """Process policy training data with thinking block summarization while keeping latest message raw."""
        if not self.config.enable_policy_thinking_summarization:
            return sdg
            
        processed_messages = []
        processed_tokens = []
        processed_masks = []
        
        for alt_idx, alt_messages in enumerate(sdg["messages"]):
            if not alt_messages:
                processed_messages.append(alt_messages)
                processed_tokens.append(sdg["tokens"][alt_idx] if alt_idx < len(sdg["tokens"]) else [])
                processed_masks.append(sdg["masks"][alt_idx] if alt_idx < len(sdg["masks"]) else [])
                continue
                
            alt_processed_messages = []
            for msg_idx, msg in enumerate(alt_messages):
                if msg_idx == len(alt_messages) - 1:
                    # Keep the last message raw for training
                    alt_processed_messages.append(msg.copy())
                elif msg["role"] == "assistant" and self.config.enable_policy_thinking_summarization:
                    # Check cache for thinking block summarization
                    original_content = msg["content"]
                    cache_key = hash(original_content)
                    
                    if cache_key in thinking_block_cache:
                        processed_content = thinking_block_cache[cache_key]
                    else:
                        # Summarize thinking blocks
                        try:
                            from atroposlib.utils.message_history_utils import summarize_thinking_block
                            processed_content = await summarize_thinking_block(
                                original_content, 
                                self.server, 
                                self.tokenizer, 
                                self.config.max_policy_thinking_summary_tokens
                            )
                            thinking_block_cache[cache_key] = processed_content
                        except Exception as e:
                            logger.warning(f"Failed to summarize thinking block: {e}")
                            from atroposlib.utils.message_history_utils import strip_thinking
                            processed_content = strip_thinking(original_content)
                            thinking_block_cache[cache_key] = processed_content
                    
                    processed_msg = msg.copy()
                    processed_msg["content"] = processed_content
                    alt_processed_messages.append(processed_msg)
                else:
                    alt_processed_messages.append(msg.copy())
            
            # Re-tokenize processed messages
            try:
                tokenized_output = tokenize_for_trainer(self.tokenizer, alt_processed_messages)
                processed_messages.append(alt_processed_messages)
                processed_tokens.append(tokenized_output["tokens"])
                processed_masks.append(tokenized_output["masks"])
            except Exception as e:
                logger.error(f"Error re-tokenizing processed policy messages for alt {alt_idx}: {e}")
                processed_messages.append(alt_messages)
                processed_tokens.append(sdg["tokens"][alt_idx] if alt_idx < len(sdg["tokens"]) else [])
                processed_masks.append(sdg["masks"][alt_idx] if alt_idx < len(sdg["masks"]) else [])
        
        return ScoredDataGroup(
            tokens=processed_tokens,
            masks=processed_masks,
            scores=sdg["scores"],
            messages=processed_messages,
            metadata=sdg["metadata"]
        )

    async def evaluate(self, *args, **kwargs):
        """Evaluation method - implementation pending."""
        pass

    async def cleanup(self):
        """Clean up resources."""
        if hasattr(self, '_temp_dir') and self._temp_dir and os.path.exists(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir)
            except Exception as e:
                logger.warning(f"Error cleaning up temporary directory {self._temp_dir}: {e}")

    def __del__(self):
        """Ensure cleanup runs even if explicit cleanup isn't called."""
        try:
            import asyncio
            if hasattr(self, '_temp_dir') and self._temp_dir and os.path.exists(self._temp_dir):
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
    async def main_cli():
        await TextWorldEnv.cli()
    asyncio.run(main_cli())