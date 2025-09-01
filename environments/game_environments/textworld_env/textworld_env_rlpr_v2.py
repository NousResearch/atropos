#!/usr/bin/env python3
"""
TextWorldEnvRLPRv2: Streamlined RLPR environment for Microsoft TextWorld

A streamlined implementation of VR-CLI perplexity rewards without complex dependencies.
Features step-by-step generation with parallel alternatives, entropy-based selection,
thinking block stripping, and inline memory management.
"""

import asyncio
import logging
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


class TextWorldEnvRLPRv2Config(BaseEnvConfig):
    """Configuration for the streamlined TextWorld RLPR environment."""

    env_name: str = "TextWorldRLPRv2"
    wandb_name: str = "textworld-rlpr-v2"
    group_size: int = 16  # Alternatives per step
    max_num_workers: int = 16
    total_steps: int = 500
    max_steps: int = 20  # Max turns per episode
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

    # Challenge settings
    challenge_names: List[str] = [
        "tw-simple",
        "tw-cooking", 
        "tw-coin_collector",
        "tw-treasure_hunter",
    ]
    randomize_challenge_settings: bool = True


class TextWorldEnvRLPRv2(BaseEnv):
    """Streamlined TextWorld RLPR environment with step-by-step generation."""

    name = "textworld_rlpr_v2"
    env_config_cls = TextWorldEnvRLPRv2Config

    def __init__(
        self,
        config: TextWorldEnvRLPRv2Config,
        server_configs: List[APIServerConfig],
        slurm: bool = True,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: TextWorldEnvRLPRv2Config = config
        self.challenge_registry = None

        # Track generated game files for cleanup
        self._generated_files = set()

        # Create temp directory for game files
        self._temp_dir = tempfile.mkdtemp(prefix="textworld_rlpr_v2_")

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
            "{\"name\": \"execute_command\", \"arguments\": {\"command\": \"go north\", \"expected_outcome\": \"I expect to move north to a new room.\"}}\n"
            "</tool_call>\n\n"
            "Use exactly one of each block type, in this order."
        )

    async def setup(self):
        """Initialize the environment and challenge registry."""
        try:
            self.challenge_registry = create_textworld_registry()
            logger.warning(
                f"Initialized TextWorld RLPR v2 challenge registry with challenges: {self.config.challenge_names}"
            )
            logger.warning("TextWorldEnvRLPRv2 setup completed successfully")
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
        """Collect trajectories using step-by-step generation with parallel alternatives."""
        challenge_name = item["challenge_name"]
        settings = item["settings"]
        logger.warning(f"Creating game for challenge: {challenge_name} with settings: {settings}")

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

        # Collect episode using step-by-step generation
        scored_data_groups = await self._collect_episode_step_by_step(
            env_id, challenge_name
        )

        self._cleanup_game_file(game_file)
        logger.warning("Episode complete, cleaning up")
        return scored_data_groups, []

    async def _collect_episode_step_by_step(
        self, env_id: str, challenge_name: str = "unknown"
    ) -> List[ScoredDataGroup]:
        """Collect ONE episode with group_size alternatives at EACH step.
        
        Returns a List[ScoredDataGroup] where each group contains alternatives for one step.
        This enables step-level training where the model learns good decisions at each step.
        """
        logger.warning(f"Starting step-by-step episode collection for {challenge_name}")
        
        try:
            env = textworld.gym.make(env_id)
            obs, info = env.reset()
            
            messages: List[Message] = []
            messages.append({"role": "system", "content": self.system_prompt})
            
            obs_text = self._format_observation(obs, info)
            messages.append({"role": "user", "content": obs_text})
            
            done = False
            total_reward = 0.0
            turn = 0
            scored_data_groups = []  # One ScoredDataGroup per step
            
            while not done and turn < self.config.max_steps:
                turn += 1
                current_tokens = len(
                    self.tokenizer.apply_chat_template(messages, tokenize=True)
                )
                if current_tokens > self.config.max_token_length - 1000:
                    logger.warning(f"Approaching token limit at turn {turn}, ending episode")
                    break
                
                # Generate group_size alternatives for this step
                alternatives = await self._generate_alternatives_for_step(
                    messages, turn
                )
                if not alternatives:
                    logger.warning(f"No alternatives generated at turn {turn}")
                    break
                
                # Score and package alternatives into a ScoredDataGroup for this step
                step_group = await self._create_step_scored_group(
                    alternatives, messages, env, turn
                )
                if step_group:
                    scored_data_groups.append(step_group)
                    
                    # Select best alternative to continue the episode
                    selected_idx = self._select_best_alternative_idx(alternatives)
                    selected_alt = alternatives[selected_idx]
                    
                    # Execute the selected action
                    action = selected_alt["parsed_action"]["command"]
                    obs, reward, done, info = env.step(action)
                    total_reward += reward
                    
                    logger.warning(f"Turn {turn}: Selected alt {selected_idx}, action '{action}' -> reward={reward}, done={done}")
                    
                    # Add selected response to conversation for next step
                    stripped_response = self._strip_thinking_blocks(selected_alt["response"])
                    messages.append({"role": "assistant", "content": stripped_response})
                    
                    if not done:
                        obs_text = self._format_observation(obs, info)
                        messages.append({"role": "user", "content": obs_text})
            
            env.close()
            
            # Track episode metrics
            outcome = 1.0 if info.get("won", False) else (-1.0 if info.get("lost", False) else 0.0)
            self.episode_outcomes_buffer.append(outcome)
            self.episode_rewards_buffer.append(total_reward)
            self.episode_steps_buffer.append(turn)
            self.episode_challenge_types.append(challenge_name)
            
            logger.warning(f"Episode completed: {len(scored_data_groups)} steps, total_reward={total_reward:.2f}")
            return scored_data_groups
            
        except Exception as e:
            logger.error(f"Fatal error in episode collection: {e}")
            logger.error(traceback.format_exc())
            return []


    async def _generate_alternatives_for_step(
        self, messages: List[Message], turn: int
    ) -> List[Dict[str, Any]]:
        """Generate group_size alternative responses for the current step.
        
        Each alternative represents a different possible action at this step.
        All alternatives will be scored and included in the ScoredDataGroup for training.
        """
        alternatives = []
        
        # Generate exactly group_size alternatives for this step
        tasks = []
        for i in range(self.config.group_size):
            task = self.server.chat_completion(
                messages=messages,
                n=1,
                max_tokens=200,
                temperature=0.8,
                logprobs=True,  # Need logprobs for entropy calculation
            )
            tasks.append(task)

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.warning(f"Turn {turn} Alternative {i} failed: {response}")
                continue

            try:
                content = response.choices[0].message.content
                logprobs = getattr(response.choices[0], 'logprobs', None)

                # Parse response format
                parsed_action = self._parse_response_format(content)
                if not parsed_action:
                    logger.debug(f"Turn {turn} Alternative {i}: Failed to parse response format")
                    # Include failed alternatives for GRPO training
                    parsed_action = {"command": "invalid", "expected_outcome": ""}

                alternatives.append({
                    "response": content,
                    "parsed_action": parsed_action,
                    "logprobs": logprobs,
                    "index": i,
                })

            except Exception as e:
                logger.debug(f"Turn {turn} Alternative {i}: Error processing response: {e}")
                continue

        return alternatives


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

    async def _create_step_scored_group(
        self, alternatives: List[Dict[str, Any]], messages: List[Message], 
        env, turn: int
    ) -> Optional[ScoredDataGroup]:
        """Create a ScoredDataGroup for one step containing all alternatives.
        
        Each alternative is tokenized, scored, and packaged together.
        This represents the decision point at one step of the episode.
        """
        import math
        
        if not alternatives:
            return None
            
        tokens_list = []
        masks_list = []
        scores_list = []
        messages_list = [] if self.config.include_messages else None
        
        for alt in alternatives:
            # Create messages with this alternative's response
            alt_messages = messages.copy()
            alt_messages.append({"role": "assistant", "content": alt["response"]})
            
            # Tokenize
            tokenization_result = tokenize_for_trainer(
                tokenizer=self.tokenizer,
                chat=alt_messages,
                train_on_all_assistant_turns=True,
            )
            
            tokens_list.append(tokenization_result["tokens"])
            masks_list.append(tokenization_result["masks"])
            
            if self.config.include_messages:
                messages_list.append(alt_messages)
        
        # Calculate length penalty to encourage concise thinking
        token_lengths = [len(tokens) for tokens in tokens_list]
        length_mean = sum(token_lengths) / len(token_lengths) if token_lengths else 1.0
        
        # Apply tanh-based length penalty
        for i, tok_len in enumerate(token_lengths):
            length_penalty = math.tanh((length_mean - tok_len) / length_mean) / 2.0
            scores_list.append(length_penalty)
        
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
    
    def _select_best_alternative_idx(self, alternatives: List[Dict[str, Any]]) -> int:
        """Select the best alternative index. For now, random selection.
        
        TODO: Implement value function-based selection once we have a value model.
        """
        if not alternatives:
            raise ValueError("No alternatives to select from")
        
        # For now, random selection
        import random
        selected_idx = random.randint(0, len(alternatives) - 1)
        return selected_idx
    
    def _select_best_alternative(self, alternatives: List[Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
        """Select the best alternative using entropy-based confidence scoring."""
        if not alternatives:
            raise ValueError("No alternatives to select from")
            
        best_idx = 0
        best_confidence = float('-inf')
        
        for i, alt in enumerate(alternatives):
            confidence = self._calculate_entropy_confidence(alt.get("logprobs"))
            if confidence > best_confidence:
                best_confidence = confidence
                best_idx = i
                
        logger.warning(f"Entropy selection: chose alternative {best_idx} with confidence {best_confidence:.4f}")
        return best_idx, alternatives[best_idx]

    def _calculate_entropy_confidence(self, logprobs) -> float:
        """Calculate confidence score based on token entropy."""
        if not logprobs or not hasattr(logprobs, 'content'):
            return random.random()  # Fallback to random if no logprobs
            
        try:
            import math
            total_entropy = 0.0
            token_count = 0
            
            for token_logprob in logprobs.content:
                if hasattr(token_logprob, 'top_logprobs'):
                    # Calculate entropy from top logprobs
                    entropy = 0.0
                    for top_logprob in token_logprob.top_logprobs:
                        prob = math.exp(top_logprob.logprob)
                        entropy -= prob * top_logprob.logprob
                    total_entropy += entropy
                    token_count += 1
                    
            if token_count == 0:
                return random.random()
                
            avg_entropy = total_entropy / token_count
            # Convert entropy to confidence (lower entropy = higher confidence)
            confidence = math.exp(-avg_entropy)
            return confidence
            
        except Exception as e:
            logger.debug(f"Error calculating entropy: {e}")
            return random.random()

    def _calculate_vrcli_score_for_trajectory(
        self, alternative: Dict[str, Any], total_reward: float, won: bool, lost: bool
    ) -> float:
        """Calculate a simplified VR-CLI-style score based on trajectory outcome."""
        # Simplified VR-CLI: reward trajectories that perform better than expected
        expected_outcome = alternative["parsed_action"].get("expected_outcome", "")
        
        # Basic heuristic: if the expected outcome mentions positive words and we won, boost score
        positive_words = ["succeed", "win", "complete", "find", "get", "take", "open", "solve"]
        negative_words = ["fail", "lose", "stuck", "cannot", "blocked", "impossible"]
        
        has_positive_expectation = any(word in expected_outcome.lower() for word in positive_words)
        has_negative_expectation = any(word in expected_outcome.lower() for word in negative_words)
        
        base_vrcli_score = 0.0
        
        if won and has_positive_expectation:
            # Model correctly predicted positive outcome
            base_vrcli_score = 1.0
            logger.warning(f"VR-CLI reward: Correct positive prediction -> {base_vrcli_score}")
        elif lost and has_negative_expectation:
            # Model correctly predicted negative outcome
            base_vrcli_score = 0.5  
            logger.warning(f"VR-CLI reward: Correct negative prediction -> {base_vrcli_score}")
        elif won and has_negative_expectation:
            # Model incorrectly predicted negative outcome
            base_vrcli_score = 0.1
            logger.warning(f"VR-CLI penalty: Incorrect negative prediction -> {base_vrcli_score}")
        elif total_reward > 0:
            # Positive trajectory outcome
            base_vrcli_score = 0.3
        else:
            base_vrcli_score = 0.0
            
        return base_vrcli_score

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
    def config_init(cls) -> Tuple[TextWorldEnvRLPRv2Config, List[APIServerConfig]]:
        """Initialize default configuration."""
        env_config = TextWorldEnvRLPRv2Config(
            tokenizer_name="NousResearch/Hermes-4-Qwen3-14B",
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
        """Evaluate the model - not implemented for this streamlined environment."""
        logger.warning("Evaluation not implemented in TextWorld RLPR v2 environment")
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
    TextWorldEnvRLPRv2.cli()