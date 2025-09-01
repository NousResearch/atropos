#!/usr/bin/env python3
"""
TextWorldEnvRLPRv3: Per-step ScoredDataGroup returns with credit assignment

This version returns one ScoredDataGroup per step instead of merging them for the whole episode.
Each step's ScoredDataGroup contains all alternatives generated for that step, with appropriate
credit assignment based on which alternatives produced the same action as the selected one.
"""

import asyncio
import logging
import math
import os
import random
import tempfile
import traceback
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym  # noqa: F401
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
from environments.game_environments.textworld_env.textworld_registry import (
    create_textworld_registry,
)  # noqa: F401

logger = logging.getLogger(__name__)


class TextWorldEnvRLPRv3Config(BaseEnvConfig):
    """Configuration for the TextWorld RLPR v3 environment."""

    env_name: str = "TextWorldRLPRv3"
    wandb_name: str = "textworld-rlpr-v3"
    group_size: int = 16  # Alternatives per step
    max_num_workers: int = 16
    total_steps: int = 500
    max_steps: int = 10  # Max turns per episode - reduced for debugging
    max_token_length: int = 32768

    # VR-CLI perplexity reward settings
    vrcli_weight: float = 0.3
    vrcli_discount_factor: float = 0.99

    # Format reward settings
    format_reward_enabled: bool = True
    format_reward_weight: float = 0.1
    format_memory_reward: float = 0.05
    format_thinking_reward: float = 0.05

    # Token length management
    token_length_penalty_enabled: bool = True
    token_length_baseline: int = 500
    token_length_penalty_scale: float = 0.0002

    # Challenge settings - Start with simple for debugging
    challenge_names: List[str] = [
        "tw-simple",
        # "tw-cooking", 
        # "tw-coin_collector",
        # "tw-treasure_hunter",
    ]
    randomize_challenge_settings: bool = False  # Use fixed settings for debugging


class TextWorldEnvRLPRv3(BaseEnv):
    """TextWorld RLPR v3 environment with per-step ScoredDataGroup returns."""

    name = "textworld_rlpr_v3"
    env_config_cls = TextWorldEnvRLPRv3Config

    def __init__(
        self,
        config: TextWorldEnvRLPRv3Config,
        server_configs: List[APIServerConfig],
        slurm: bool = True,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: TextWorldEnvRLPRv3Config = config
        self.challenge_registry = None

        # Track generated game files for cleanup
        self._generated_files = set()

        # Create temp directory for game files
        self._temp_dir = tempfile.mkdtemp(prefix="textworld_rlpr_v3_")

        # wandb logging
        self.episode_outcomes_buffer = []
        self.episode_rewards_buffer = []
        self.episode_steps_buffer = []
        self.episode_challenge_types = []
        self.eval_metrics_custom = []

        self.system_prompt = (
            "You are playing a text-based adventure game. "
            "You must respond in the following exact format:\n\n"
            "<think>\n"
            "Your detailed reasoning about the current situation, objectives, "
            "and likely outcomes of potential actions.\n"
            "</think>\n\n"
            "<memory>\n"
            "Concise summary building on previous memories, noting the outcome "
            "of the last action, current game state, inventory, location, "
            "and progress toward objectives.\n"
            "</memory>\n\n"
            "<tool_call>\n"
            '{"name": "execute_command", "arguments": {"command": "go north", "expected_outcome": "I expect to move north to a new room."}}\n'
            "</tool_call>\n\n"
            "Use exactly one of each block type, in this order."
        )

    async def setup(self):
        """Initialize the environment and challenge registry."""
        try:
            self.challenge_registry = create_textworld_registry()
            logger.warning(
                f"Initialized TextWorld RLPR v3 challenge registry with challenges: {self.config.challenge_names}"
            )
            logger.warning("TextWorldEnvRLPRv3 setup completed successfully")
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
            game_settings = {
                "rewards": settings.get("rewards", "balanced"),
                "goal": settings.get("goal", "detailed"),
                "test": str(settings.get("test", False)).lower(),
            }
            game = textworld.challenges.simple.make(game_settings, options=options)
        elif challenge_name == "tw-cooking":
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
                "split": "train",
            }
            logger.debug(f"Cooking game settings: {game_settings}")
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

        # Save gamefile
        game_file = os.path.join(
            self._temp_dir,
            f"{challenge_name}_{settings.get('seed', random.randint(0, 1000000))}.z8",
        )
        options.path = game_file
        options.file_ext = ".z8"
        game_file = textworld.generator.compile_game(game, options)

        # Track for cleanup
        self._generated_files.add(game_file)

        return game_file

    async def collect_trajectories(
        self, item: Item
    ) -> Tuple[List[ScoredDataGroup], List[Item]]:
        """Collect trajectories using per-step ScoredDataGroup generation.
        
        Returns:
            A list of ScoredDataGroups, one for each step in the episode.
        """
        challenge_name = item["challenge_name"]
        settings = item["settings"]

        game_file = self._create_game(challenge_name, settings)

        # Register the gamefile
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

        # Collect episode using step-by-step generation with per-step ScoredDataGroups
        scored_data_groups = await self._collect_episode_step_by_step(
            env_id, challenge_name
        )

        self._cleanup_game_file(game_file)
        
        # Return list of ScoredDataGroups (one per step)
        return scored_data_groups, []

    async def _collect_episode_step_by_step(
        self, env_id: str, challenge_name: str = "unknown"
    ) -> List[ScoredDataGroup]:
        """Collect an episode returning one ScoredDataGroup per step."""
        logger.warning(f"Starting per-step episode collection for {challenge_name}")
        
        scored_data_groups = []
        
        try:
            env = textworld.gym.make(env_id)
            obs, info = env.reset()

            # Initialize conversation with system prompt
            messages: List[Message] = [
                {"role": "system", "content": self.system_prompt}
            ]
            
            obs_text = self._format_observation(obs, info)
            messages.append({"role": "user", "content": obs_text})

            done = False
            total_reward = 0.0
            turn = 0
            cumulative_vrcli = 0.0

            async with self.server.dedicated_server() as server:
                while not done and turn < self.config.max_steps:
                    turn += 1
                    
                    # Check token limit
                    current_tokens = len(
                        self.tokenizer.apply_chat_template(messages, tokenize=True)
                    )
                    if current_tokens > self.config.max_token_length - 1000:
                        logger.warning(
                            f"Step {turn}: Approaching token limit, ending episode"
                        )
                        break

                    # Generate alternatives for this step
                    alternatives = await self._generate_step_alternatives(
                        messages, server, turn
                    )
                    if not alternatives:
                        logger.warning(f"Step {turn}: No alternatives generated")
                        break

                    # TODO: Restore entropy-based selection instead of random
                    # For now, randomly select one alternative
                    selected_idx = random.randint(0, len(alternatives) - 1)
                    selected_alternative = alternatives[selected_idx]
                    logger.warning(
                        f"Step {turn}: Randomly selected alternative {selected_idx} (TODO: restore entropy selection)"
                    )

                    # Extract action from selected alternative
                    selected_action = selected_alternative["parsed_action"]["command"]
                    logger.debug(f"Step {turn}: Selected action: {selected_action}")

                    # Execute action in environment
                    try:
                        obs, reward, done, info = env.step(selected_action)
                        total_reward += reward
                        logger.warning(
                            f"Step {turn}: Action '{selected_action}' -> reward={reward}, done={done}"
                        )
                    except Exception as e:
                        logger.error(f"Step {turn}: Environment error: {e}")
                        # Still create ScoredDataGroup for failed step
                        reward = 0.0
                        done = True

                    # Create ScoredDataGroup for this step with credit assignment
                    step_sdg = await self._create_step_scored_data_group(
                        messages=messages.copy(),  # Current conversation state
                        alternatives=alternatives,
                        selected_idx=selected_idx,
                        selected_action=selected_action,
                        step_reward=reward,
                        cumulative_reward=total_reward,
                        done=done,
                        won=info.get("won", False),
                        lost=info.get("lost", False),
                        turn=turn,
                    )
                    
                    if step_sdg:
                        scored_data_groups.append(step_sdg)
                    
                    # Add selected response to conversation (strip thinking blocks)
                    selected_response = selected_alternative["response"]
                    stripped_response = self._strip_thinking_blocks(selected_response)
                    messages.append({"role": "assistant", "content": stripped_response})

                    # Add next observation if not done
                    if not done:
                        obs_text = self._format_observation(obs, info)
                        messages.append({"role": "user", "content": obs_text})

            env.close()

            # Log episode statistics
            self.episode_outcomes_buffer.append(
                1.0 if info.get("won", False) else (-1.0 if info.get("lost", False) else 0.0)
            )
            self.episode_rewards_buffer.append(total_reward)
            self.episode_steps_buffer.append(turn)
            self.episode_challenge_types.append(challenge_name)

            logger.warning(
                f"Episode completed: {len(scored_data_groups)} steps, total_reward={total_reward:.2f}"
            )
            return scored_data_groups

        except Exception as e:
            logger.error(f"Fatal error in episode collection: {e}")
            logger.error(traceback.format_exc())
            return scored_data_groups

    async def _generate_step_alternatives(
        self, messages: List[Message], server, turn: int
    ) -> List[Dict[str, Any]]:
        """Generate group_size alternatives for the current step."""
        alternatives = []
        
        # Generate group_size alternatives in parallel
        tasks = []
        for i in range(self.config.group_size):
            task = server.chat_completion(
                messages=messages,
                n=1,
                max_tokens=200,
                temperature=0.8,
                logprobs=True,  # Need logprobs for future entropy calculation
            )
            tasks.append(task)

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.warning(f"Step {turn} Alternative {i} failed: {response}")
                # Still include failed alternatives for GRPO training
                alternatives.append({
                    "response": "",
                    "parsed_action": {"command": "invalid", "expected_outcome": ""},
                    "logprobs": None,
                    "index": i,
                    "failed": True,
                })
                continue

            try:
                content = response.choices[0].message.content
                logprobs = getattr(response.choices[0], 'logprobs', None)

                # Parse response format
                parsed_action = self._parse_response_format(content)
                if not parsed_action:
                    logger.debug(f"Step {turn} Alternative {i}: Failed to parse response format")
                    # Include with invalid action for GRPO
                    parsed_action = {"command": "invalid", "expected_outcome": ""}

                alternatives.append({
                    "response": content,
                    "parsed_action": parsed_action,
                    "logprobs": logprobs,
                    "index": i,
                    "failed": False,
                })

            except Exception as e:
                logger.debug(f"Step {turn} Alternative {i}: Error processing response: {e}")
                alternatives.append({
                    "response": "",
                    "parsed_action": {"command": "invalid", "expected_outcome": ""},
                    "logprobs": None,
                    "index": i,
                    "failed": True,
                })

        return alternatives

    async def _create_step_scored_data_group(
        self,
        messages: List[Message],
        alternatives: List[Dict[str, Any]],
        selected_idx: int,
        selected_action: str,
        step_reward: float,
        cumulative_reward: float,
        done: bool,
        won: bool,
        lost: bool,
        turn: int,
    ) -> Optional[ScoredDataGroup]:
        """Create a ScoredDataGroup for a single step with credit assignment.
        
        Credit assignment strategy:
        - The selected alternative (canonical) gets the full reward
        - Other alternatives that produced the same action also get credit
        - Alternatives with invalid/different actions get negative/zero credit
        """
        if not alternatives:
            return None
            
        scored_items = []
        
        for i, alt in enumerate(alternatives):
            # Create messages for this alternative
            alt_messages = messages.copy()
            
            # Only add non-empty responses
            if alt["response"]:
                stripped_response = self._strip_thinking_blocks(alt["response"])
                alt_messages.append({"role": "assistant", "content": stripped_response})
            else:
                # Failed alternative - add placeholder
                alt_messages.append({"role": "assistant", "content": "invalid"})
            
            # Tokenize the conversation
            tokenization_result = tokenize_for_trainer(
                tokenizer=self.tokenizer,
                chat=alt_messages,
                train_on_all_assistant_turns=True,
            )
            
            # Calculate score for this alternative
            alt_score = self._calculate_alternative_score(
                alternative=alt,
                is_selected=(i == selected_idx),
                selected_action=selected_action,
                step_reward=step_reward,
                cumulative_reward=cumulative_reward,
                done=done,
                won=won,
                lost=lost,
            )
            
            scored_items.append(ScoredDataItem(
                messages=alt_messages if self.config.include_messages else None,
                tokens=tokenization_result["tokens"],
                masks=tokenization_result["masks"],
                scores=alt_score,
                metadata={
                    "step": turn,
                    "alternative_idx": i,
                    "is_selected": (i == selected_idx),
                    "action": alt["parsed_action"]["command"],
                    "selected_action": selected_action,
                    "step_reward": step_reward,
                    "alternative_score": alt_score,
                }
            ))
        
        # Apply token length-based diversity scoring to encourage efficiency
        if len(scored_items) > 1:
            # Get token lengths for all alternatives
            token_lengths = [len(item["tokens"]) for item in scored_items]
            length_mean = sum(token_lengths) / len(token_lengths)
            
            # Calculate length-based adjustments using tanh penalty
            length_adjustments = [
                math.tanh((length_mean - tok_len) / length_mean) / 2.0 
                for tok_len in token_lengths
            ]
            
            # Apply adjustments to scores
            for i, adjustment in enumerate(length_adjustments):
                original_score = scored_items[i]["scores"]
                scored_items[i]["scores"] = original_score + adjustment
                # Update metadata to track the adjustment
                scored_items[i]["metadata"]["length_adjustment"] = adjustment
                scored_items[i]["metadata"]["token_length"] = token_lengths[i]
                scored_items[i]["metadata"]["length_mean"] = length_mean
        
        # Create ScoredDataGroup from all alternatives
        sdg = ScoredDataGroup(
            tokens=[item["tokens"] for item in scored_items],
            masks=[item["masks"] for item in scored_items],
            scores=[item["scores"] for item in scored_items],
            messages=[item.get("messages", []) for item in scored_items] if self.config.include_messages else [],
            advantages=None,
            ref_logprobs=None,
            group_overrides={},
            overrides=None,
            images=None,
        )
        
        logger.warning(
            f"Step {turn}: Created ScoredDataGroup with {len(scored_items)} alternatives, "
            f"selected={selected_idx}, reward={step_reward:.2f}"
        )
        
        return sdg

    def _calculate_alternative_score(
        self,
        alternative: Dict[str, Any],
        is_selected: bool,
        selected_action: str,
        step_reward: float,
        cumulative_reward: float,
        done: bool,
        won: bool,
        lost: bool,
    ) -> float:
        """Calculate score for a single alternative with credit assignment."""
        
        # Base score depends on whether this alternative matches the selected action
        alt_action = alternative["parsed_action"]["command"]
        
        if alternative.get("failed", False) or alt_action == "invalid":
            # Failed to generate or parse - negative score
            base_score = -0.5
        elif alt_action == selected_action:
            # This alternative produced the same action as selected - gets credit
            base_score = step_reward
        else:
            # Different action - small negative score
            base_score = -0.1
        
        # Add VR-CLI style rewards
        vrcli_score = self._calculate_vrcli_score_for_alternative(
            alternative, step_reward, cumulative_reward, won, lost
        )
        
        # Add format rewards
        format_score = 0.0
        if alternative["response"]:
            format_score = self._calculate_format_reward(alternative["response"])
        
        # Token length adjustment
        token_adj = 0.0
        if alternative["response"] and self.config.token_length_penalty_enabled:
            token_adj = self._calculate_token_length_adjustment(alternative["response"])
        
        # Composite score
        final_score = (
            base_score +
            vrcli_score * self.config.vrcli_weight +
            format_score * self.config.format_reward_weight +
            token_adj
        )
        
        # Boost score if this was the selected alternative and it succeeded
        if is_selected and step_reward > 0:
            final_score *= 1.2
        
        return final_score

    def _calculate_vrcli_score_for_alternative(
        self, alternative: Dict[str, Any], step_reward: float, cumulative_reward: float, won: bool, lost: bool
    ) -> float:
        """Calculate VR-CLI-style score for an alternative."""
        expected_outcome = alternative["parsed_action"].get("expected_outcome", "")
        
        # Simplified VR-CLI: reward based on outcome prediction accuracy
        positive_words = ["succeed", "win", "complete", "find", "get", "take", "open", "solve"]
        negative_words = ["fail", "lose", "stuck", "cannot", "blocked", "impossible"]
        
        has_positive_expectation = any(word in expected_outcome.lower() for word in positive_words)
        has_negative_expectation = any(word in expected_outcome.lower() for word in negative_words)
        
        vrcli_score = 0.0
        
        if step_reward > 0 and has_positive_expectation:
            # Correctly predicted positive outcome for this step
            vrcli_score = 0.5
        elif step_reward < 0 and has_negative_expectation:
            # Correctly predicted negative outcome
            vrcli_score = 0.2
        elif step_reward > 0 and has_negative_expectation:
            # Incorrectly predicted negative when positive happened
            vrcli_score = -0.2
        elif step_reward < 0 and has_positive_expectation:
            # Incorrectly predicted positive when negative happened  
            vrcli_score = -0.2
            
        # Apply discount based on overall trajectory outcome if episode is done
        if won:
            vrcli_score *= 1.5
        elif lost:
            vrcli_score *= 0.5
            
        return vrcli_score

    def _calculate_format_reward(self, response: str) -> float:
        """Calculate reward for following the structured response format."""
        if not self.config.format_reward_enabled:
            return 0.0
            
        score = 0.0
        
        # Check for required blocks
        has_think = "<think>" in response and "</think>" in response
        has_memory = "<memory>" in response and "</memory>" in response
        has_tool_call = "<tool_call>" in response and "</tool_call>" in response
        
        if has_think:
            score += self.config.format_thinking_reward
        if has_memory:
            score += self.config.format_memory_reward
        if has_tool_call:
            score += 0.05  # Tool call reward
            
        # Penalty for wrong structure
        if not (has_think and has_memory and has_tool_call):
            score *= 0.5
            
        return score

    def _calculate_token_length_adjustment(self, response: str) -> float:
        """Calculate token length penalty/bonus."""
        if not self.config.token_length_penalty_enabled:
            return 0.0
            
        token_count = len(self.tokenizer.encode(response))
        deviation = token_count - self.config.token_length_baseline
        adjustment = -deviation * self.config.token_length_penalty_scale
        
        return adjustment

    def _parse_response_format(self, response: str) -> Optional[Dict[str, str]]:
        """Parse the structured response format to extract action and expected outcome."""
        import json
        import re
        
        try:
            # Extract tool_call JSON
            tool_call_match = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', response, re.DOTALL)
            if not tool_call_match:
                return None
                
            tool_call_json = tool_call_match.group(1).strip()
            tool_call = json.loads(tool_call_json)
            
            if tool_call.get("name") != "execute_command":
                return None
                
            args = tool_call.get("arguments", {})
            command = args.get("command", "")
            expected_outcome = args.get("expected_outcome", "")
            
            if not command:
                return None
                
            return {
                "command": command,
                "expected_outcome": expected_outcome
            }
            
        except (json.JSONDecodeError, KeyError) as e:
            return None

    def _strip_thinking_blocks(self, response: str) -> str:
        """Strip thinking blocks from response but keep memory blocks."""
        import re
        
        # Remove <think>...</think> blocks
        stripped = re.sub(r'<think>.*?</think>\s*', '', response, flags=re.DOTALL)
        
        return stripped.strip()

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
    def config_init(cls) -> Tuple[TextWorldEnvRLPRv3Config, List[APIServerConfig]]:
        """Initialize default configuration."""
        env_config = TextWorldEnvRLPRv3Config(
            tokenizer_name="NousResearch/Hermes-4-Qwen3-14B",
            group_size=16,
            use_wandb=True,
            wandb_name=cls.name,
            max_token_length=32768,
            total_steps=500,
            challenge_names=[
                "tw-simple",
            ],
            randomize_challenge_settings=False,
            max_steps=10,
        )

        server_configs = [
            APIServerConfig(
                api_key="x",
                base_url="http://localhost:9001/v1",
                model_name="NousResearch/Hermes-4-Qwen3-14B",
                server_type="openai",
                timeout=1200,
                num_max_requests_at_once=512,
                num_requests_for_eval=64,
                health_check=True,
            )
        ]

        return env_config, server_configs

    # TODO: implement evaluation properly re eval changes
    async def evaluate(self, num_items: int) -> Dict[str, Any]:
        """Evaluate the model - not implemented for this environment."""
        logger.warning("Evaluation not implemented in TextWorld RLPR v3 environment")
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
            max_reward = (
                max(self.episode_rewards_buffer) if self.episode_rewards_buffer else 0
            )
            min_reward = (
                min(self.episode_rewards_buffer) if self.episode_rewards_buffer else 0
            )

            wandb_metrics[f"{self.name}/train/total_episodes"] = total_episodes
            wandb_metrics[f"{self.name}/train/win_count_absolute"] = wins
            wandb_metrics[f"{self.name}/train/loss_count_absolute"] = losses
            wandb_metrics[f"{self.name}/train/draw_count_absolute"] = draws
            wandb_metrics[f"{self.name}/train/win_rate_percent"] = win_rate
            wandb_metrics[f"{self.name}/train/loss_rate_percent"] = loss_rate
            wandb_metrics[f"{self.name}/train/draw_rate_percent"] = draw_rate
            wandb_metrics[f"{self.name}/train/avg_episode_steps"] = avg_steps
            wandb_metrics[f"{self.name}/train/avg_outcome"] = (
                avg_outcome  # -1, 0, 1 average
            )
            wandb_metrics[f"{self.name}/train/avg_reward_score"] = (
                avg_reward  # Actual game score
            )
            wandb_metrics[f"{self.name}/train/max_reward_score"] = max_reward
            wandb_metrics[f"{self.name}/train/min_reward_score"] = min_reward

            # Per-challenge statistics
            challenge_stats = {}
            for i, (outcome, reward, steps, challenge) in enumerate(
                zip(
                    self.episode_outcomes_buffer,
                    self.episode_rewards_buffer,
                    self.episode_steps_buffer,
                    self.episode_challenge_types,
                )
            ):
                if challenge not in challenge_stats:
                    challenge_stats[challenge] = {
                        "outcomes": [],
                        "rewards": [],
                        "steps": [],
                        "count": 0,
                    }
                challenge_stats[challenge]["outcomes"].append(outcome)
                challenge_stats[challenge]["rewards"].append(reward)
                challenge_stats[challenge]["steps"].append(steps)
                challenge_stats[challenge]["count"] += 1

            for challenge, stats in challenge_stats.items():
                challenge_wins = sum(1 for o in stats["outcomes"] if o > 0)
                challenge_losses = sum(1 for o in stats["outcomes"] if o < 0)
                challenge_draws = sum(1 for o in stats["outcomes"] if o == 0)
                challenge_total = stats["count"]

                wandb_metrics[f"{self.name}/train/{challenge}/episodes_count"] = (
                    challenge_total
                )
                wandb_metrics[f"{self.name}/train/{challenge}/wins_count"] = (
                    challenge_wins
                )
                wandb_metrics[f"{self.name}/train/{challenge}/losses_count"] = (
                    challenge_losses
                )
                wandb_metrics[f"{self.name}/train/{challenge}/draws_count"] = (
                    challenge_draws
                )

                wandb_metrics[f"{self.name}/train/{challenge}/win_rate_percent"] = (
                    (challenge_wins / challenge_total) * 100
                    if challenge_total > 0
                    else 0
                )
                wandb_metrics[f"{self.name}/train/{challenge}/loss_rate_percent"] = (
                    (challenge_losses / challenge_total) * 100
                    if challenge_total > 0
                    else 0
                )
                wandb_metrics[f"{self.name}/train/{challenge}/draw_rate_percent"] = (
                    (challenge_draws / challenge_total) * 100
                    if challenge_total > 0
                    else 0
                )

                wandb_metrics[f"{self.name}/train/{challenge}/avg_steps"] = (
                    sum(stats["steps"]) / len(stats["steps"]) if stats["steps"] else 0
                )
                wandb_metrics[f"{self.name}/train/{challenge}/avg_outcome"] = (
                    sum(stats["outcomes"]) / len(stats["outcomes"])
                    if stats["outcomes"]
                    else 0
                )
                wandb_metrics[f"{self.name}/train/{challenge}/avg_reward_score"] = (
                    sum(stats["rewards"]) / len(stats["rewards"])
                    if stats["rewards"]
                    else 0
                )
                wandb_metrics[f"{self.name}/train/{challenge}/max_reward_score"] = (
                    max(stats["rewards"]) if stats["rewards"] else 0
                )
                wandb_metrics[f"{self.name}/train/{challenge}/min_reward_score"] = (
                    min(stats["rewards"]) if stats["rewards"] else 0
                )

            self.episode_outcomes_buffer = []
            self.episode_rewards_buffer = []
            self.episode_steps_buffer = []
            self.episode_challenge_types = []

        # Log eval metrics if any
        for key, value in self.eval_metrics_custom:
            wandb_metrics[key] = value
        self.eval_metrics_custom = []

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

    def __del__(self):
        """Ensure cleanup on deletion."""
        # Clean up any remaining game files
        files_to_clean = list(self._generated_files)
        for game_file in files_to_clean:
            self._cleanup_game_file(game_file)

        # Clean up local temp directory
        if hasattr(self, "_temp_dir") and os.path.exists(self._temp_dir):
            import shutil

            try:
                shutil.rmtree(self._temp_dir)
                logger.debug(f"Cleaned up temp directory: {self._temp_dir}")
            except Exception:
                pass


if __name__ == "__main__":
    TextWorldEnvRLPRv3.cli()