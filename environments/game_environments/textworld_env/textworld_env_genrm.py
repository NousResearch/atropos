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
from datetime import datetime

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
# No longer using entropy-based scoring - using LLM-as-judge instead
from environments.game_environments.textworld_env.textworld_registry import (
    create_textworld_registry,
)  # noqa: F401

logger = logging.getLogger(__name__)


class TextWorldEnvGenRMConfig(BaseEnvConfig):
    """Configuration for the TextWorld GenRM environment with LLM-as-judge."""

    env_name: str = "TextWorldGenRM"
    wandb_name: str = "textworld-genrm"
    
    # Judge configuration
    judge_model_name: str = "Hermes-4-405B"
    judge_group_size: int = 3  # Number of judges for consensus
    judge_temperature: float = 0.7
    use_same_model_for_judge: bool = False
    group_size: int = 16  # Alternatives per step
    max_num_workers: int = 16
    total_steps: int = 1000  # Collect 1000 winning episodes
    max_steps: int = 100  # Max turns per episode
    max_token_length: int = 65536  # Support 32k output + context

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
    output_path: Optional[str] = None  # Path to save episodes (overrides hardcoded path)
    
    # Data saving
    data_path_to_save_groups: Optional[str] = None


class TextWorldEnvGenRM(BaseEnv):
    """TextWorld GenRM environment with LLM-as-judge scoring."""

    name = "textworld_genrm"
    env_config_cls = TextWorldEnvGenRMConfig

    def __init__(
        self,
        config: TextWorldEnvGenRMConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = True,
        testing: bool = False,
    ):
        # Override server configs for Hermes-405B API access
        if config.judge_model_name == "Hermes-4-405B" or "Hermes-4-405B" in str(config):
            import os
            api_key = os.getenv("HERMES_API_KEY", "sk-CRs4gcGL5Jai3ojQ2BKxxA")
            server_configs = [
                APIServerConfig(
                    model_name="Hermes-4-405B",
                    base_url="https://inference-api.nousresearch.com/v1",
                    api_key=api_key,
                    num_requests_for_eval=128,
                )
            ]
        
        super().__init__(config, server_configs, slurm, testing)
        self.config: TextWorldEnvGenRMConfig = config
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
            "You are an AI agent playing a text-based adventure game who uses extreme long chains of thought "
            "to carefully plan your actions and predict their outcomes. Your goal is to follow the objective "
            "described at the start of the game. You interact with the world by providing text commands and "
            "predicting their outcomes."
            "\n\n"
            "You should:\n"
            "1. Enclose your thoughts and internal monologue inside <think> </think> tags. Use extremely "
            "long chains of thought to carefully consider the game state, your objectives, and the likely "
            "outcomes of your actions.\n"
            "2. Generate a memory summary inside <memory> </memory> tags that captures key information from "
            "this turn. Your memory should:\n"
            "   - Build upon previous memories shown in 'Relevant Memories' if present\n"
            "   - Note the outcome of your last action (did it match your prediction?)\n"
            "   - Update your understanding of the game state, location, and inventory\n"
            "   - Track progress toward objectives and any multi-step plans\n"
            "   - Be concise but comprehensive (1-3 sentences)\n"
            "3. Provide your action using the execute_command function call."
            "\n\n"
            "For your function call, return a JSON object with function name and arguments "
            "within <tool_call> </tool_call> tags with the following schema:\n"
            '<tool_call>\n{"name": "execute_command", "arguments": {"command": "go north", '
            '"expected_outcome": "I move north to a new room"}}\n</tool_call>\n\n'
            "EXAMPLE RESPONSE 1:\n"
            "<think>\n"
            "I'm in the kitchen. I see a stove and a fridge. The objective says to cook something. "
            "Let me check what's in the fridge first to see what ingredients are available."
            "\n</think>\n"
            "<memory>\n"
            "Kitchen has stove and fridge. Main objective is cooking. Need to find ingredients."
            "\n</memory>\n"
            "<tool_call>\n"
            '{"name": "execute_command", "arguments": {"command": "open fridge", '
            '"expected_outcome": "The fridge opens, revealing its contents. I expect to see various '
            'food items or ingredients inside that I can take and use for cooking."}}'
            "\n</tool_call>\n\n"
            "EXAMPLE RESPONSE 2 (with previous memories):\n"
            "<think>\n"
            "Looking at my previous memories, I was exploring the kitchen to find cooking ingredients. "
            "I successfully opened the fridge and found eggs, milk, and flour. My goal is still to "
            "cook something. Now I need to take these ingredients and find a recipe or mixing bowl. "
            "The previous action of opening the fridge worked as expected."
            "\n</think>\n"
            "<memory>\n"
            "Found eggs, milk, and flour in kitchen fridge. Still need mixing bowl or recipe to cook. "
            "Previous exploration of kitchen successful - have stove and ingredients located."
            "\n</memory>\n"
            "<tool_call>\n"
            '{"name": "execute_command", "arguments": {"command": "take eggs", '
            '"expected_outcome": "I take the eggs from the fridge and add them to my inventory"}}'
            "\n</tool_call>\n\n"
            "EXAMPLE RESPONSE 3:\n"
            "<think>\n"
            "There's a locked door here and I have a key in my inventory. I should try using the key "
            "on the door."
            "\n</think>\n"
            "<memory>\n"
            "Found locked door in current room. Have key in inventory that might open it."
            "\n</memory>\n"
            "<tool_call>\n"
            '{"name": "execute_command", "arguments": {"command": "unlock door with key", '
            '"expected_outcome": "The key turns in the lock and the door unlocks. I should now be '
            'able to open the door and go through it."}}'
            "\n</tool_call>\n\n"
            "Remember: Your entire response must be exactly three XML blocks: <think>...</think> "
            "followed by <memory>...</memory> followed by <tool_call>...</tool_call>\n\n"
            "FINAL REMINDER: After your <think> block and <memory> block, you MUST wrap your JSON "
            "function call in <tool_call> tags. The JSON goes INSIDE the <tool_call> tags, not after them."
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
        import time
        
        # Generate a unique seed for this game instance
        unique_seed = int(time.time() * 1000000) + random.randint(0, 1000000)
        
        # Randomly select a challenge
        if len(self.config.challenge_names) == 1:
            challenge_name = self.config.challenge_names[0]
        else:
            challenge_name = random.choice(self.config.challenge_names)

        # Get challenge settings with unique seed
        challenge_name, settings = self.challenge_registry.get_challenge(
            challenge_name, 
            randomize_settings=self.config.randomize_challenge_settings,
            seed=unique_seed
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
        from textworld.generator.chaining import QuestGenerationError
        
        # Retry up to 10 times with different seeds for quest generation failures
        max_retries = 10
        original_seed = settings.get("seed", random.randint(0, 1000000))
        
        for retry_count in range(max_retries):
            try:
                # Create default options with current seed
                options = textworld.GameOptions()
                current_seed = original_seed + retry_count * 1000  # Use different seeds for retries
                options.seeds = current_seed

                if challenge_name == "tw-simple":
                    game_settings = {
                        "rewards": settings["rewards"],
                        "goal": settings["goal"],
                        "test": str(settings["test"]).lower(),
                    }
                    game = textworld.challenges.simple.make(game_settings, options=options)
                elif challenge_name == "tw-cooking":
                    game_settings = {
                        "recipe": settings["recipe"],  # Number of ingredients
                        "take": settings["take"],  # Number to find
                        "cook": settings["cook"],  # Whether to cook
                        "open": settings["open"],  # Whether to open containers
                        "drop": settings["drop"],  # Whether limited inventory
                        "go": settings["go"],  # Number of locations
                        "recipe_seed": settings.get("recipe-seed", settings.get("recipe_seed", random.randint(0, 1000000))),
                        "split": "train",
                    }
                    logger.debug(f"Cooking game settings: {game_settings}")
                    game = textworld.challenges.cooking.make(game_settings, options=options)
                elif challenge_name == "tw-coin_collector":
                    game_settings = {"level": settings["level"]}
                    game = textworld.challenges.coin_collector.make(
                        game_settings, options=options
                    )
                elif challenge_name == "tw-treasure_hunter":
                    game_settings = {"level": settings["level"]}
                    game = textworld.challenges.treasure_hunter.make(
                        game_settings, options=options
                    )
                else:
                    raise ValueError(f"Unknown challenge: {challenge_name}")

                # Save gamefile
                game_file = os.path.join(
                    self._temp_dir,
                    f"{challenge_name}_{current_seed}.z8",
                )
                options.path = game_file
                options.file_ext = ".z8"
                
                try:
                    game_file = textworld.generator.compile_game(game, options)
                except Exception as e:
                    logger.error(f"Failed to compile game {challenge_name} with settings {settings}: {e}")
                    # If it's a size overflow error for coin_collector or treasure_hunter, retry with lower level
                    if "exceeds version-8 limit" in str(e) and challenge_name in ["tw-coin_collector", "tw-treasure_hunter"]:
                        logger.warning(f"Retrying {challenge_name} with level 10 instead of {settings.get('level', 'unknown')}")
                        settings["level"] = 10
                        if challenge_name == "tw-coin_collector":
                            game_settings = {"level": 10}
                            game = textworld.challenges.coin_collector.make(
                                game_settings, options=options
                            )
                        elif challenge_name == "tw-treasure_hunter":
                            game_settings = {"level": 10}
                            game = textworld.challenges.treasure_hunter.make(
                                game_settings, options=options
                            )
                        game_file = textworld.generator.compile_game(game, options)
                    else:
                        raise

                # Track for cleanup
                self._generated_files.add(game_file)

                return game_file
                
            except QuestGenerationError as e:
                # Quest generation failed, retry with different seed
                if retry_count < max_retries - 1:
                    logger.warning(f"Quest generation failed with seed {current_seed}: {e}, retrying with different seed...")
                    continue
                else:
                    logger.error(f"Failed to generate quest after {max_retries} attempts")
                    raise
        
        # Should never reach here
        raise RuntimeError(f"Failed to create game after {max_retries} attempts")

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
            step_rewards = []  # Track rewards for each step
            selected_indices = []  # Track selected alternative for each step
            episode_data = []  # Track data for saving
            
            while not done and turn < self.config.max_steps:
                turn += 1
                
                # Implement sliding window for token management
                MAX_GENERATION_TOKENS = 32768  # DeepSeek-R1 can generate 32k
                available_budget = self.config.max_token_length - MAX_GENERATION_TOKENS
                
                current_tokens = self._count_tokens(messages)
                
                if current_tokens > available_budget:
                    logger.warning(f"Turn {turn}: Token count {current_tokens} exceeds budget {available_budget}, applying sliding window")
                    
                    # Keep system prompt (messages[0]) and recent context
                    # Drop oldest messages until under budget
                    while len(messages) > 3 and current_tokens > available_budget:
                        # Remove the second message (keeping system prompt at index 0)
                        dropped_msg = messages.pop(1)
                        logger.debug(f"Dropped message: {dropped_msg['role']} - {dropped_msg['content'][:50]}...")
                        
                        # Recalculate tokens
                        current_tokens = self._count_tokens(messages)
                    
                    logger.warning(f"After sliding window: {len(messages)} messages, {current_tokens} tokens")
                
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
                    
                    # Save data for this step if configured
                    if self.config.data_path_to_save_groups:
                        step_data = {
                            "turn": turn,
                            "messages": messages.copy() if self.config.include_messages else None,
                            "alternatives": [
                                {
                                    "response": alt["response"],
                                    "judge_score": alt.get("judge_score", 0.0),
                                    "parsed_action": alt.get("parsed_action", {})
                                }
                                for alt in alternatives
                            ],
                            "selected_idx": selected_idx,
                        }
                        episode_data.append(step_data)
                    selected_alt = alternatives[selected_idx]
                    
                    # Execute the selected action
                    action = selected_alt["parsed_action"]["command"]
                    obs, reward, done, info = env.step(action)
                    total_reward += reward
                    
                    # Track for credit assignment
                    step_rewards.append(reward)
                    selected_indices.append(selected_idx)
                    
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
            
            # Apply credit assignment to scored data groups
            self._apply_credit_assignment(
                scored_data_groups, 
                step_rewards, 
                selected_indices,
                info.get("won", False),
                info.get("lost", False)
            )
            
            logger.warning(f"Episode completed: {len(scored_data_groups)} steps, total_reward={total_reward:.2f}")
            
            # Save episode data only if we won
            won = info.get("won", False)
            if won and scored_data_groups:
                self._save_winning_episode(scored_data_groups, challenge_name, total_reward)
            
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
        
        # Convert messages to prompt format for completions endpoint
        # This avoids SGLang's tool execution behavior
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Add prefill to guide response format
        prefill = "<think>\n"
        prompt = prompt + prefill
        
        # Generate exactly group_size alternatives in a single call
        try:
            response = await self.server.completion(
                prompt=prompt,
                n=self.config.group_size,  # Get all alternatives at once
                max_tokens=32768,  # DeepSeek-R1 supports 32k context
                temperature=0.8,
                top_p=0.95,
                logprobs=5,  # Get top 5 logprobs for entropy calculation
                stop=["</tool_call>", "<|im_end|>", "<|endoftext|>"],
            )
            
            # Process each choice/alternative
            for i, choice in enumerate(response.choices):
                try:
                    # Get the generated text and prepend the prefill
                    generated_text = choice.text.strip()
                    content = prefill + generated_text
                    
                    # Extract logprobs if available
                    logprobs = getattr(choice, 'logprobs', None)
                    
                    # If we have an opening tool_call tag but no closing tag (due to stop token), append it
                    if "<tool_call>" in content and "</tool_call>" not in content:
                        content += "\n</tool_call>"
                    
                    # Debug: Log first few alternatives' content
                    if i < 3:
                        logger.warning(f"Turn {turn} Alt {i} FULL response: {content}")
                        if logprobs:
                            logger.warning(f"Turn {turn} Alt {i} has logprobs: {logprobs is not None}")
                        else:
                            logger.warning(f"Turn {turn} Alt {i} has NO logprobs data")

                    # Parse response format
                    parsed_action = self._parse_response_format(content)
                    if not parsed_action:
                        if i < 3:  # Debug first few parse failures
                            logger.warning(f"Turn {turn} Alternative {i}: Failed to parse response format")
                        # Include failed alternatives for GRPO training
                        parsed_action = {"command": "invalid", "expected_outcome": ""}
                    else:
                        if i < 3:  # Debug successful parses
                            logger.warning(f"Turn {turn} Alt {i} parsed action: {parsed_action['command']}")

                    alternatives.append({
                        "response": content,
                        "parsed_action": parsed_action,
                        "logprobs": logprobs,
                        "index": i,
                    })

                except Exception as e:
                    logger.debug(f"Turn {turn} Alternative {i}: Error processing response: {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"Turn {turn}: Failed to generate alternatives: {e}")
            return []

        return alternatives


    def _parse_response_format(self, response: str) -> Optional[Dict[str, str]]:
        """Parse the structured response format to extract action and expected outcome."""
        import json
        import re
        
        try:
            # Extract tool_call JSON
            tool_call_match = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', response, re.DOTALL)
            if not tool_call_match:
                # Debug: Log if no tool_call tags found
                if "<tool_call>" not in response:
                    logger.debug(f"Parse fail: No <tool_call> tags in response")
                return None
                
            tool_call_json = tool_call_match.group(1).strip()
            logger.debug(f"Extracted tool_call JSON: {tool_call_json[:100]}...")
            
            tool_call = json.loads(tool_call_json)
            
            if tool_call.get("name") != "execute_command":
                logger.debug(f"Parse fail: Tool name is '{tool_call.get('name')}', not 'execute_command'")
                return None
                
            args = tool_call.get("arguments", {})
            command = args.get("command", "")
            expected_outcome = args.get("expected_outcome", "")
            
            if not command:
                logger.debug(f"Parse fail: Empty command")
                return None
                
            logger.debug(f"Parse success: command='{command}', expected_outcome='{expected_outcome[:50]}...'")
            return {
                "command": command,
                "expected_outcome": expected_outcome
            }
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.debug(f"Parse fail: JSON decode error: {e}")
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
        
        # First pass: tokenize all alternatives
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
        
        # Get judge scores for all alternatives
        judge_scores = await self._score_actions_with_llm_judge(alternatives, messages, turn)
        
        # Store judge scores in alternatives for selection
        for i, alt in enumerate(alternatives):
            alt["judge_score"] = judge_scores[i] if i < len(judge_scores) else 0.0
        
        # Calculate length penalty
        token_lengths = [len(tokens) for tokens in tokens_list]
        length_mean = sum(token_lengths) / len(token_lengths) if token_lengths else 1.0
        
        # Combine length penalty, judge score, and format reward
        for i, tok_len in enumerate(token_lengths):
            # Length penalty component (encourages conciseness)
            length_penalty = math.tanh((length_mean - tok_len) / length_mean) / 2.0
            
            # Judge score component (LLM-as-judge ranking)
            judge_score = alternatives[i].get("judge_score", 0.0)
            
            # Format reward component (encourages proper structure)
            format_score = self._calculate_format_reward(alternatives[i]["response"])
            
            # Debug format scoring for first alternative
            if i == 0:
                logger.debug(f"Turn {turn} Alt {i} format score: {format_score:.4f}, judge score: {judge_score:.4f}")
            
            # Combine scores with proper weighting
            # Weights: 0.1 for format, 0.3 for length, 0.6 for judge score
            combined_score = (
                self.config.format_reward_weight * format_score +
                0.3 * length_penalty + 
                (0.7 - self.config.format_reward_weight) * judge_score
            )
            
            scores_list.append(combined_score)
        
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
        """Score actions using LLM-as-judge with consensus from multiple judges.
        
        Args:
            alternatives: List of alternative actions with parsed_action field
            messages: Context messages leading to this decision point
            turn: Current turn number
            
        Returns:
            List of scores for each alternative (0.0 to 1.0)
        """
        # Extract just the actions (no thinking blocks)
        actions = []
        for i, alt in enumerate(alternatives):
            action = alt.get("parsed_action", {}).get("action", "")
            if not action:
                # Fallback to extracting from response if parsing failed
                response = alt.get("response", "")
                if "<tool_call>" in response and "</tool_call>" in response:
                    start = response.find("<tool_call>") + len("<tool_call>")
                    end = response.find("</tool_call>")
                    action = response[start:end].strip()
                else:
                    action = "[Invalid action]"
            actions.append(action)
        
        # Create judge prompt
        context = messages[-1]["content"] if messages else ""  # Last observation
        judge_prompt = self._create_judge_prompt(context, actions)
        
        # Get rankings from multiple judges
        all_rankings = []
        for judge_idx in range(self.config.judge_group_size):
            ranking = await self._get_single_judge_ranking(judge_prompt, len(actions), judge_idx)
            if ranking:
                all_rankings.append(ranking)
                logger.debug(f"Judge {judge_idx} ranking: {ranking}")
        
        # Calculate consensus scores
        scores = self._calculate_consensus_scores(all_rankings, len(actions))
        
        logger.warning(f"Turn {turn}: Judge consensus scores: {[f'{s:.3f}' for s in scores]}")
        
        return scores
    
    def _create_judge_prompt(self, context: str, actions: List[str]) -> str:
        """Create prompt for the judge to rank actions."""
        prompt = f"""You are evaluating actions in a text-based game.

Current situation:
{context}

Possible actions to evaluate:
"""
        for i, action in enumerate(actions):
            prompt += f"\n{i+1}. {action}"
        
        prompt += """\n\nRank these actions from best to worst based on:
1. Likelihood of progressing toward the game objective
2. Safety (avoiding actions that might cause game failure)
3. Efficiency (direct progress vs unnecessary exploration)

Provide your ranking as a comma-separated list of numbers (e.g., "3,1,4,2" means action 3 is best, then 1, then 4, then 2).

Ranking:"""
        
        return prompt
    
    async def _get_single_judge_ranking(self, prompt: str, num_actions: int, judge_idx: int) -> Optional[List[int]]:
        """Get ranking from a single judge."""
        try:
            # Check if we're using Hermes-405B as judge (external API)
            if self.config.judge_model_name == "Hermes-4-405B":
                # Use Nous API directly for Hermes-405B
                import aiohttp
                import os
                
                api_key = os.getenv("NOUS_API_KEY") or os.getenv("HERMES_API_KEY")
                if not api_key:
                    logger.warning("No NOUS_API_KEY or HERMES_API_KEY found for Hermes-405B judge")
                    return None
                
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": "Hermes-4-405B",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 50,
                    "temperature": self.config.judge_temperature,
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "https://inference-api.nousresearch.com/v1/chat/completions",
                        headers=headers,
                        json=payload
                    ) as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            response = result["choices"][0]["message"]["content"]
                        else:
                            logger.warning(f"Hermes-405B API error: {resp.status}")
                            return None
            else:
                # Use regular SGLang servers for other models
                if self.config.use_same_model_for_judge:
                    server_idx = judge_idx % len(self.server_configs)
                else:
                    # Use different servers for diversity
                    server_idx = judge_idx % len(self.server_configs)
                
                response = await self.call_llm_api(
                    prompt,
                    max_tokens=50,
                    temperature=self.config.judge_temperature,
                    server_config_idx=server_idx,
                    model_override=self.config.judge_model_name if not self.config.use_same_model_for_judge else None,
                )
            
            # Parse ranking from response
            ranking_text = response.strip()
            ranking = self._parse_ranking(ranking_text, num_actions)
            
            return ranking
            
        except Exception as e:
            logger.warning(f"Judge {judge_idx} failed: {e}")
            return None
    
    def _parse_ranking(self, ranking_text: str, num_actions: int) -> Optional[List[int]]:
        """Parse ranking text into list of indices."""
        try:
            # Extract just numbers from the response
            import re
            numbers = re.findall(r'\d+', ranking_text)
            
            if not numbers:
                return None
            
            ranking = [int(n) - 1 for n in numbers]  # Convert to 0-indexed
            
            # Validate ranking
            if len(ranking) != num_actions:
                logger.debug(f"Invalid ranking length: expected {num_actions}, got {len(ranking)}")
                return None
            
            if set(ranking) != set(range(num_actions)):
                logger.debug(f"Invalid ranking indices: {ranking}")
                return None
            
            return ranking
            
        except Exception as e:
            logger.debug(f"Failed to parse ranking '{ranking_text}': {e}")
            return None
    
    def _calculate_consensus_scores(self, all_rankings: List[List[int]], num_actions: int) -> List[float]:
        """Calculate consensus scores from multiple judge rankings.
        
        Uses Borda count: each action gets points based on its position in each ranking.
        """
        if not all_rankings:
            # No valid rankings, return uniform scores
            return [0.5] * num_actions
        
        # Borda count scoring
        scores = [0.0] * num_actions
        
        for ranking in all_rankings:
            for position, action_idx in enumerate(ranking):
                # Higher score for better positions
                points = num_actions - position
                scores[action_idx] += points
        
        # Normalize scores to [0, 1]
        max_possible = num_actions * len(all_rankings)
        if max_possible > 0:
            scores = [s / max_possible for s in scores]
        
        return scores
    
    def _save_winning_episode(self, scored_data_groups: List[ScoredDataGroup], challenge_name: str, total_reward: float):
        """Save winning episode as JSON array of ScoredDataGroups to JSONL file."""
        import json
        import os
        
        # Use config output_path if provided, otherwise use hardcoded path
        data_path = self.config.output_path or "/home/maxpaperclips/atropos/data/textworld_genrm_hermes405b.jsonl"
        
        # Create directory if needed  
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        # Convert ScoredDataGroups to serializable format, excluding tokens and mask
        # Each episode is a list of ScoredDataGroups (one per step)
        episode_array = []
        for sdg in scored_data_groups:
            # ScoredDataGroup is a TypedDict, so treat as dict
            clean_sdg = dict(sdg).copy()
            
            # Remove tokens and masks at top level
            clean_sdg.pop('tokens', None)
            clean_sdg.pop('masks', None)  # Note: it's 'masks' not 'mask'
            clean_sdg.pop('mask', None)   # Just in case
            
            # Also clean the scored_data list inside if it exists
            if 'scored_data' in clean_sdg and isinstance(clean_sdg['scored_data'], list):
                cleaned_scored_data = []
                for item in clean_sdg['scored_data']:
                    clean_item = dict(item).copy() if item else {}
                    
                    # Remove tokens and mask from each scored data item
                    clean_item.pop('tokens', None)
                    clean_item.pop('masks', None)
                    clean_item.pop('mask', None)
                    cleaned_scored_data.append(clean_item)
                
                clean_sdg['scored_data'] = cleaned_scored_data
            
            episode_array.append(clean_sdg)
        
        # Write episode as single line JSON array
        with open(data_path, "a") as f:
            f.write(json.dumps(episode_array) + "\n")
        
        # Count total saved episodes
        try:
            with open(data_path, "r") as f:
                num_episodes = sum(1 for _ in f)
        except:
            num_episodes = 1
        
        logger.warning(f"Saved WINNING episode #{num_episodes}: {challenge_name}, reward={total_reward:.2f}, steps={len(scored_data_groups)}")
    
    def _transform_logprobs_for_confidence(self, logprobs) -> Optional[List[Dict[str, Any]]]:
        """Transform completion logprobs to format expected by confidence_score."""
        if not logprobs:
            return None
        
        logprobs_data = []
        
        # Handle OpenAI/SGLang completion Logprobs object
        # It has attributes: tokens, token_logprobs, top_logprobs
        if hasattr(logprobs, 'tokens') and hasattr(logprobs, 'token_logprobs'):
            tokens = logprobs.tokens or []
            token_logprobs_list = logprobs.token_logprobs or []
            top_logprobs_list = logprobs.top_logprobs or []
            
            for i in range(len(tokens)):
                token_data = {
                    "token": tokens[i] if i < len(tokens) else "",
                    "logprob": token_logprobs_list[i] if i < len(token_logprobs_list) else 0.0,
                    "top_logprobs": []
                }
                
                # Add top logprobs if available
                if i < len(top_logprobs_list) and top_logprobs_list[i]:
                    top_probs = top_logprobs_list[i]
                    if isinstance(top_probs, dict):
                        # Convert dict format to list format
                        for token, logprob in top_probs.items():
                            token_data["top_logprobs"].append({
                                "token": token,
                                "logprob": logprob
                            })
                
                logprobs_data.append(token_data)
            
            return logprobs_data if logprobs_data else None
        
        # Handle SGLang's ChatCompletionTokenLogprob format (for chat_completion compatibility)
        elif hasattr(logprobs, 'content'):
            for token_logprob in logprobs.content:
                token_data = {
                    "token": getattr(token_logprob, 'token', ''),
                    "logprob": getattr(token_logprob, 'logprob', 0.0),
                    "top_logprobs": []
                }
                
                # Add top logprobs if available
                if hasattr(token_logprob, 'top_logprobs') and token_logprob.top_logprobs:
                    for top in token_logprob.top_logprobs:
                        token_data["top_logprobs"].append({
                            "token": getattr(top, 'token', ''),
                            "logprob": getattr(top, 'logprob', 0.0)
                        })
                
                logprobs_data.append(token_data)
            
            return logprobs_data if logprobs_data else None
        
        return None
    
    def _select_best_alternative_idx(self, alternatives: List[Dict[str, Any]]) -> int:
        """Select the best alternative using judge scores."""
        if not alternatives:
            raise ValueError("No alternatives to select from")
        
        best_idx = 0
        best_score = float('-inf')
        judge_scores = []
        
        for i, alt in enumerate(alternatives):
            # Use judge score if available
            score = alt.get("judge_score", 0.0)
            judge_scores.append(score)
            
            if score > best_score:
                best_score = score
                best_idx = i
        
        logger.warning(f"Judge selection: chose alternative {best_idx} with score {best_score:.4f}")
        logger.warning(f"All judge scores: {[f'{s:.4f}' for s in judge_scores]}")
        
        return best_idx

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

    def _apply_credit_assignment(
        self,
        scored_data_groups: List[ScoredDataGroup],
        step_rewards: List[float],
        selected_indices: List[int],
        won: bool,
        lost: bool
    ) -> None:
        """Apply credit assignment for sparse rewards using discounted returns.
        
        Updates scores in scored_data_groups based on future rewards.
        """
        if not scored_data_groups or not step_rewards:
            return
            
        # Calculate final outcome reward
        final_outcome_reward = 0.0
        if won:
            final_outcome_reward = 1.0
        elif lost:
            final_outcome_reward = -1.0
            
        num_steps = len(step_rewards)
        if len(scored_data_groups) != num_steps or len(selected_indices) != num_steps:
            logger.warning(f"Mismatch in step counts for credit assignment")
            return
            
        # Calculate discounted returns backwards (Monte Carlo returns)
        discounted_return = final_outcome_reward
        for t in range(num_steps - 1, -1, -1):
            immediate_reward = step_rewards[t]
            
            # Update discounted return
            discounted_return = (
                immediate_reward + self.config.vrcli_discount_factor * discounted_return
            )
            
            sdg = scored_data_groups[t]
            chosen_idx = selected_indices[t]
            
            if 0 <= chosen_idx < len(sdg["scores"]):
                # Calculate future return (excluding immediate reward)
                future_return = self.config.vrcli_discount_factor * (
                    discounted_return - immediate_reward
                )
                
                # Apply credit assignment weight
                weighted_future_return = self.config.vrcli_weight * future_return
                
                # Update scores with future returns
                new_scores = list(sdg["scores"])
                new_scores[chosen_idx] += weighted_future_return
                
                # Also credit alternatives that produced the same action
                if "messages" in sdg and sdg["messages"] is not None:
                    chosen_messages = sdg["messages"][chosen_idx]
                    if chosen_messages and chosen_messages[-1]["role"] == "assistant":
                        chosen_response = chosen_messages[-1]["content"]
                        # Extract action from response using existing method
                        parsed_action = self._parse_response_format(chosen_response)
                        chosen_action = parsed_action.get("command", "invalid") if parsed_action else "invalid"
                        
                        if chosen_action and chosen_action != "invalid":
                            # Check other alternatives for same action
                            for alt_idx in range(len(sdg["messages"])):
                                if alt_idx != chosen_idx and alt_idx < len(new_scores):
                                    alt_messages = sdg["messages"][alt_idx]
                                    if alt_messages and alt_messages[-1]["role"] == "assistant":
                                        alt_response = alt_messages[-1]["content"]
                                        # Extract action from response using existing method
                                        parsed_alt = self._parse_response_format(alt_response)
                                        alt_action = parsed_alt.get("command", "invalid") if parsed_alt else "invalid"
                                        if alt_action == chosen_action:
                                            # This alternative would have led to the same outcome
                                            new_scores[alt_idx] += weighted_future_return
                                            logger.debug(
                                                f"Turn {t}: Alternative {alt_idx} also gets future return "
                                                f"for same action '{chosen_action}'"
                                            )
                
                sdg["scores"] = new_scores
                
                logger.debug(
                    f"Turn {t}: immediate_reward={immediate_reward:.3f}, "
                    f"future_return={weighted_future_return:.3f}, "
                    f"total_score={new_scores[chosen_idx]:.3f}"
                )

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
            score += self.config.format_thinking_reward  # Tool call gets same reward as thinking block
            
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

    def _count_tokens(self, messages: List[Message]) -> int:
        """Count tokens in a message list."""
        return len(self.tokenizer.apply_chat_template(messages, tokenize=True))
    
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
    def config_init(cls) -> Tuple[TextWorldEnvGenRMConfig, List[APIServerConfig]]:
        """Initialize default configuration."""
        env_config = TextWorldEnvGenRMConfig(
            tokenizer_name="NousResearch/Hermes-4-Qwen3-14B",
            group_size=16,
            use_wandb=True,
            wandb_name=cls.name,
            max_token_length=65536,
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
    TextWorldEnvGenRM.cli()