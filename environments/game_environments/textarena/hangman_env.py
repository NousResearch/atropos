import asyncio
import copy
import json
import logging
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import textarena as ta
import yaml
from typing_extensions import TypedDict

from atroposlib.envs.base import BaseEnv, BaseEnvConfig, OpenaiConfig, ScoredDataGroup
from atroposlib.envs.reward_fns import registry
from atroposlib.envs.reward_fns.combined_reward import CombinedReward
from atroposlib.type_definitions import Message
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer
from atroposlib.utils.tool_call_parser import (
    format_tool_call_for_hangman,
    parse_tool_call,
)

logger = logging.getLogger(__name__)


class HangmanEnvConfig(BaseEnvConfig):
    temperature: float = 0.7
    top_p: float = 0.9
    max_turns: int = 10
    thinking_active: bool = True
    single_step_only: bool = False
    inference_weight: float = 1.0
    wandb_name: str = "hangman"

    reward_functions: List[str] = ["format"]
    format_reward_weight: float = 0.3
    environment_reward_weight: float = 0.7
    format_thinking_weight: float = 0.5
    format_tool_weight: float = 0.5


class HangmanScoredDataGroup(ScoredDataGroup):
    seed: int
    tokens: Optional[List[List[int]]] = None
    masks: Optional[List[List[int]]] = None
    scores: Optional[List[float]] = None
    messages: Optional[List[List[Message]]] = None
    parsed_action: Optional[str] = None


class EpisodeMetrics(TypedDict):
    scores_per_step: List[float]
    total_score: float
    num_steps: int
    won: bool
    avg_token_length: float
    max_token_length: float


icl_messages = [
    {
        "role": "environment",
        "content": (
            "You are Player 0. You are playing Hangman.\n"
            "The objective of the game is to guess the word by providing one letter guesses "
            "or the entire word.\n\n"
            "Use the provided tools to either guess a letter or guess the full word based "
            "on the current game state. Only create a single tool call per response.\n\n"
            "  _______     \n"
            " |/      |    \n"
            " |            \n"
            " |            \n"
            " |            \n"
            " |            \n"
            "_|___         \n\n"
            "🎯 Word: _ _ _ _ _\n\n"
            "❤️ Tries left: 6"
        ),
    },
    {
        "role": "agent",
        "content": (
            "<think>\n"
            "I should guess the letter E because it's a very common letter.\n"
            "</think>\n"
            "<tool_call>\n"
            '{"arguments": {"letter": "E"}, "name": "guess_letter"}\n'
            "</tool_call>"
        ),
    },
    {
        "role": "environment",
        "content": (
            "  _______     \n"
            " |/      |    \n"
            " |            \n"
            " |            \n"
            " |            \n"
            " |            \n"
            "_|___         \n\n"
            "🎯 Word: _ _ _ E _\n\n"
            "❤️ Tries left: 6"
        ),
    },
    {
        "role": "agent",
        "content": (
            "<think>\n"
            "I should guess the letter I because it's another common letter.\n"
            "</think>\n"
            "<tool_call>\n"
            '{"arguments": {"letter": "I"}, "name": "guess_letter"}\n'
            "</tool_call>"
        ),
    },
    {
        "role": "environment",
        "content": (
            "Your guess of I is not in the word.\n"
            "  _______     \n"
            " |/      |    \n"
            " |      (X)   \n"
            " |            \n"
            " |            \n"
            " |            \n"
            "_|___         \n\n"
            "🎯 Word: _ _ _ E _\n\n"
            "❤️ Tries left: 5"
        ),
    },
    {
        "role": "agent",
        "content": (
            "<think>\n"
            "I should guess the letter A because it's another very common letter.\n"
            "</think>\n"
            "<tool_call>\n"
            '{"arguments": {"letter": "A"}, "name": "guess_letter"}\n'
            "</tool_call>"
        ),
    },
    {
        "role": "environment",
        "content": (
            "  _______     \n"
            " |/      |    \n"
            " |      (X)   \n"
            " |            \n"
            " |            \n"
            " |            \n"
            "_|___         \n\n"
            "🎯 Word: _ A _ E _\n\n"
            "❤️ Tries left: 5"
        ),
    },
    {
        "role": "agent",
        "content": (
            "<think>\n"
            "I should guess the letter T because it's another common letter.\n"
            "</think>\n"
            "<tool_call>\n"
            '{"arguments": {"letter": "T"}, "name": "guess_letter"}\n'
            "</tool_call>"
        ),
    },
    {
        "role": "environment",
        "content": (
            "  _______     \n"
            " |/      |    \n"
            " |      (X)   \n"
            " |            \n"
            " |            \n"
            " |            \n"
            "_|___         \n\n"
            "🎯 Word: _ A T E _\n\n"
            "❤️ Tries left: 5"
        ),
    },
    {
        "role": "agent",
        "content": (
            "<think>\n"
            "I should guess the word LATER because it's a common word that fits the current board state.\n"
            "</think>\n"
            "<tool_call>\n"
            '{"arguments": {"word": "LATER"}, "name": "guess_word"}\n'
            "</tool_call>"
        ),
    },
    {
        "role": "environment",
        "content": (
            "Congratulations! Player 0 completed the Hangman puzzle.\n"
            "  _______     \n"
            " |/      |    \n"
            " |      (X)   \n"
            " |            \n"
            " |            \n"
            " |            \n"
            "_|___         \n\n"
            "🎯 Word: L A T E R\n\n"
            "❤️ Tries left: 5"
        ),
    },
]


class EpisodeState:
    def __init__(self, seed: int):
        self.seed = seed
        self.env = ta.wrappers.LLMObservationWrapper(ta.make("Hangman-v0"))
        self.games_won = 0
        self.episodes_run = 0
        self.trajectory: List[HangmanScoredDataGroup] = []
        self.message_history: List[Message] = []
        self.step_rewards: List[float] = []
        self.episode_rewards: List[float] = []
        self.current_board_state: List[str] = []
        self.guessed_letters: List[str] = []
        self.tries_left: int = 6

        self.total_env_reward: float = 0.0
        self.total_format_reward: float = 0.0
        self.total_combined_reward: float = 0.0
        self.num_correct_actions: int = 0
        self.num_total_actions: int = 0


class HangmanOnlineEnv(BaseEnv):
    name = "hangman"

    def __init__(
        self,
        config: BaseEnvConfig,
        server_configs: List[OpenaiConfig],
        slurm=True,
        testing=False,
        debug_mode=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.debug_mode = debug_mode
        self.episodes: Dict[int, EpisodeState] = {}
        if self.debug_mode:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        system_prompt = (
            "You are a deep thinking AI agent. "
            "Use extremely long chains of thought to deeply consider the problem "
            "and deliberate via systematic reasoning processes. "
            "Help come to a correct solution prior to responding. "
            "Enclose your internal monologue inside <think></think> tags, "
            "then provide your solution or response to the problem."
        )

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "guess_letter",
                    "description": "Guess a single letter",
                    "parameters": {"letter": {"type": "string"}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "guess_word",
                    "description": "Guess the full word",
                    "parameters": {"word": {"type": "string"}},
                },
            },
        ]

        self.reward_function = self._initialize_reward_function()

        tools_json = json.dumps(self.tools)
        tool_calling_prompt = (
            "You are provided with function signatures within <tools> </tools> XML tags.\n"
            "You must call one of these functions to make your guess in the Hangman game.\n\n"
            "<tools>\n"
            f"{tools_json}\n"
            "</tools>\n\n"
            "Follow these instructions carefully:\n"
            "1. First, think step-by-step about the game state, previous guesses, and your strategy "
            "inside `<think></think>` tags.\n"
            "2. After your thinking, make ONE function call using the `<tool_call></tool_call>` tags.\n"
            "3. The content inside `<tool_call></tool_call>` MUST be ONLY a single, valid JSON object\n"
            "   conforming to the function signature.\n"
            "4. DO NOT include any other text, explanations, or XML tags inside `<tool_call></tool_call>`.\n\n"
            "Example of the required output format:\n"
            "<think>\n"
            "I need to guess a letter. The board is `_ _ _`. Common letters are E, T, A. I'll try E.\n"
            "</think>\n"
            "<tool_call>\n"
            '{"arguments": {"letter": "E"}, "name": "guess_letter"}\n'
            "</tool_call>\n\n"
            "Another example:\n"
            "<think>\n"
            "The board is `L A T E _`. The word looks like LATER. I should guess the word.\n"
            "</think>\n"
            "<tool_call>\n"
            '{"arguments": {"word": "LATER"}, "name": "guess_word"}\n'
            "</tool_call>\n\n"
            "Incorrect Example (JSON not isolated):\n"
            '<think>Thinking...</think><tool_call>Here is the JSON: \\n{"arguments": {"letter": "A"}, "name": '
            '"guess_letter"}</tool_call>\n\n'
            "Incorrect Example (Extra tags inside tool_call):\n"
            '<think>Thinking...</think><tool_call>\\n<arguments>{"letter": "A"}</arguments>\\n<n>'
            "guess_letter</n></tool_call>\n\n"
            "Strategy tips:\n"
            "- Start by guessing common letters (E, T, A, O, I, N) when the board is mostly empty\n"
            "- Track which letters you've already guessed to avoid repeating them\n"
            "- When you have enough information to make an educated guess, try guessing the complete word\n"
            "- For single letter guesses, ensure you only pass ONE letter\n"
            "- For word guesses, ensure the word matches the length of the puzzle\n"
            "- You can only choose a letter once - if you guess the same letter twice, the game will not accept it\n\n"
            "Don't try to guess the word if there's no clues available as to what the word is.\n"
            "Instead, try to guess common letters first to narrow down the possibilities.\n"
            "Use single letter guesses if the grid is empty, and try to reveal as many letters as possible.\n"
            "Once you have gathered enough information to guess the word, guess the word."
        )

        self.system_prompt = system_prompt + tool_calling_prompt
        self.icl_messages = [
            {"role": "system", "content": self.system_prompt}
        ] + icl_messages

    def _initialize_reward_function(self):
        if hasattr(self.config, "reward_functions") and self.config.reward_functions:
            reward_configs = []

            for reward_func in self.config.reward_functions:
                if isinstance(reward_func, str):
                    if reward_func == "format":
                        format_config = {
                            "type": "format",
                            "weight": self.config.format_reward_weight,
                            "params": {
                                "preferred_tags": ["think", "tool_call"],
                                "require_all_tags": True,
                            },
                        }
                        reward_configs.append(format_config)
                    elif reward_func == "tool_calling":
                        tool_calling_config = {
                            "type": "tool_calling",
                            "weight": self.config.format_reward_weight,
                            "params": {
                                "tools": self.tools,
                                "preferred_tags": ["tool_call"],
                                "check_arguments": True,
                            },
                        }
                        reward_configs.append(tool_calling_config)
                    else:
                        reward_configs.append(reward_func)
                else:
                    reward_configs.append(reward_func)

            if len(reward_configs) == 1:
                return registry.create(reward_configs[0])
            elif len(reward_configs) > 1:
                return CombinedReward(rewards=reward_configs, normalization="none")

        return None

    def _get_deterministic_word(self, seed: int, word_list: List[str]) -> str:
        """
        Deterministically selects a word from the provided word list based on the seed.
        Note: This is a hack to ensure the word is always the same for the given seed.
        It will be removed in the future after the environment is updated
        Args:
            seed: The seed for the random number generator.
            word_list: The list of words to choose from.

        Returns:
            The chosen word in uppercase.
        """
        if not word_list:
            logger.warning(
                "Provided word list is empty, returning default word 'AGENT'."
            )
            return "AGENT"
        try:
            # Use a Random instance seeded locally to avoid affecting global state
            seeded_rng = random.Random(seed)
            chosen_word = seeded_rng.choice(word_list).upper()
            # Filter out words with non-alphabetic characters if any exist in the list
            while not chosen_word.isalpha():
                logger.debug(
                    f"Word '{chosen_word}' contains non-alpha chars, re-sampling."
                )
                chosen_word = seeded_rng.choice(word_list).upper()
            logger.debug(f"Deterministic word for seed {seed}: {chosen_word}")
            return chosen_word
        except IndexError:
            logger.warning(
                f"Word list seems empty or invalid after filtering for seed {seed}. Returning 'AGENT'."
            )
            return "AGENT"
        except Exception as e:
            logger.error(
                f"Error generating deterministic word for seed {seed}: {e}. Returning 'AGENT'."
            )
            return "AGENT"

    def _get_or_create_episode(self, seed: int) -> EpisodeState:
        print("seed", seed)
        if seed not in self.episodes:
            self.episodes[seed] = EpisodeState(seed)
            self.episodes[seed].env.reset(seed=seed, num_players=1)
            env = self.episodes[seed].env
            chosen_word = self._get_deterministic_word(seed, env.word_list)

            # Manually set the chosen word and related state
            env.chosen_word = chosen_word
            print("chosen_word", chosen_word)
            env.game_board = list(chosen_word)
            env.state.game_state["board"] = ["_"] * len(chosen_word)
            env.guessed_letters = set()  # Ensure guessed letters are reset
            logger.info(f"Episode {seed}: Set deterministic word to '{chosen_word}'")

            self.episodes[seed].current_board_state = env.state.game_state["board"]
            self.episodes[seed].tries_left = env.state.game_state["tries_left"]

        # Always update current board state and tries left when retrieving episode
        else:
            self.episodes[seed].current_board_state = self.episodes[
                seed
            ].env.state.game_state["board"]
            self.episodes[seed].tries_left = self.episodes[seed].env.state.game_state[
                "tries_left"
            ]

        return self.episodes[seed]

    def _modify_observation(
        self, observation: str, episode: Optional[EpisodeState] = None
    ) -> str:
        instruction_pattern = (
            r"There are two ways you can answer\. You can provide one letter guesses "
            r"in the format of \[L\], or you can guess the entire word in the format "
            r"of \[LIGHT\]\."
        )
        modified = re.sub(
            instruction_pattern,
            "Use the provided tools to either guess a letter or guess the full word "
            "based on the current game state. Only create a single tool call per response.",
            observation,
        )

        modified = re.sub(
            r"^(C\d{2}(?:\s+C\d{2})*)\s*\n", "", modified, flags=re.MULTILINE
        )

        modified = re.sub(
            r"Board state:\s*\n(.*?)(?:\n\n|\n$|\Z)", "", modified, flags=re.DOTALL
        )
        modified = re.sub(
            r"Current Hangman Grid:\s*\n(.*?)(?:\n\n|\n$|\Z)",
            "",
            modified,
            flags=re.DOTALL,
        )

        if episode and episode.env:
            logger.debug(
                f"[MODIFY_OBS] Game state for renderer: {episode.env.state.game_state}"
            )
            ascii_art_board = episode.env.get_board_str()
            modified = modified + "\n" + ascii_art_board

        return modified

    async def setup(self):
        await asyncio.sleep(5)  # Add a small delay for server initialization

    def _parse_tool_call(self, response: str) -> str:
        tool_name, arguments, is_error = parse_tool_call(
            response, self.tools, ["tool_call"]
        )
        if is_error:
            return "-ERROR-"

        return format_tool_call_for_hangman(tool_name, arguments)

    def _strip_think_tags(self, content: str) -> str:
        return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)

    def _check_thinking_block(self, response: str) -> bool:
        return re.search(r"<think>.*?</think>", response, re.DOTALL) is not None

    def _score_response(
        self,
        pre_state: ta.State,
        post_state: ta.State,
        done: bool,
        info: Dict[str, Any],
        prevent_player_change: bool,
        response_text: str,
        parsed_action: str,
        episode_seed: int,
        update_episode_totals: bool = False,
    ) -> float:
        env_reward = 0.0
        format_reward = 0.0
        if self.debug_mode:
            logger.debug(
                f"[SCORE] Scoring action: {parsed_action} for seed {episode_seed}"
            )
            logger.debug(
                f"[SCORE] Done: {done}, Info: {info}, Prevent Change: {prevent_player_change}"
            )

        if parsed_action == "-ERROR-":
            env_reward -= 1.0
            if self.debug_mode:
                logger.debug("[SCORE] Penalty: Invalid tool call format (-1.0)")
        else:
            env_reward += 0.05

            if done:
                final_reward = post_state.rewards.get(0) if post_state.rewards else None
                if final_reward == 1:
                    env_reward += 6.0
                    if self.debug_mode:
                        logger.debug("[SCORE] Bonus: Game Won (+6.0)")
                elif final_reward == -1:
                    env_reward -= 1.5
                    if self.debug_mode:
                        logger.debug("[SCORE] Penalty: Game Lost (-1.5)")
                elif final_reward == 0:
                    env_reward -= 0.5
                    if self.debug_mode:
                        logger.debug("[SCORE] Penalty: Game Draw (-0.5)")
                elif "Congratulations" in info.get("reason", "").lower():
                    env_reward += 6.0
                    if self.debug_mode:
                        logger.debug("[SCORE] Bonus: Game Won (via info) (+6.0)")
                elif "lost" in info.get("reason", "").lower():
                    env_reward -= 1.5
                    if self.debug_mode:
                        logger.debug("[SCORE] Penalty: Game Lost (via info) (-1.5)")
                elif (
                    "draw" in info.get("reason", "").lower()
                    or "turn limit" in info.get("reason", "").lower()
                ):
                    env_reward -= 0.5
                    if self.debug_mode:
                        logger.debug("[SCORE] Penalty: Game Draw (via info) (-0.5)")

            else:
                if prevent_player_change:
                    env_reward -= 0.5
                    if self.debug_mode:
                        logger.debug(
                            "[SCORE] Penalty: Invalid non-terminal move (-0.5)"
                        )
                else:
                    pre_board = pre_state.game_state.get("board", [])
                    post_board = post_state.game_state.get("board", [])
                    pre_tries = pre_state.game_state.get("tries_left", 6)
                    post_tries = post_state.game_state.get("tries_left", 6)

                    if post_board != pre_board and post_tries == pre_tries:
                        newly_revealed = 0
                        guessed_letter = ""
                        match = re.search(r"\[([A-Za-z])\]", parsed_action)
                        if match:
                            guessed_letter = match.group(1).upper()
                            for i in range(len(post_board)):
                                if (
                                    post_board[i] == guessed_letter
                                    and pre_board[i] == "_"
                                ):
                                    newly_revealed += 1

                        if newly_revealed > 0:
                            total_letters = len(post_board)
                            remaining_underscores = post_board.count("_")
                            progress_factor = (
                                (total_letters - remaining_underscores) / total_letters
                                if total_letters > 0
                                else 0
                            )
                            letter_base_reward = newly_revealed * 1.0
                            progress_bonus = letter_base_reward * progress_factor * 0.5
                            env_reward += letter_base_reward + progress_bonus
                            if self.debug_mode:
                                logger.debug(
                                    f"[SCORE] Bonus: Correct letter '{guessed_letter}' ({newly_revealed}x). "
                                    f"Base: {letter_base_reward:.2f}, Prog: {progress_factor:.2f}, "
                                    f"Bonus: {progress_bonus:.2f}"
                                )
                        else:
                            env_reward -= 0.1
                            if self.debug_mode:
                                logger.debug(
                                    "[SCORE] Penalty: Board changed but no letter revealed? (-0.1)"
                                )

                    elif post_tries < pre_tries:
                        env_reward -= 0.2
                        if self.debug_mode:
                            logger.debug("[SCORE] Penalty: Incorrect letter (-0.2)")

        if self.reward_function:
            format_completions = [[{"role": "assistant", "content": response_text}]]
            try:
                format_rewards = self.reward_function(format_completions)
                if format_rewards and len(format_rewards) > 0:
                    format_reward = format_rewards[0]
                    if self.debug_mode:
                        logger.debug(f"[SCORE] Format reward: {format_reward:.2f}")
            except Exception as e:
                logger.error(f"Error calculating format reward: {e}")
                if self.debug_mode:
                    logger.debug(f"[SCORE] Error calculating format reward: {e}")

        response_tokens = len(self.tokenizer.encode(response_text))
        if self.config.max_token_length > 0:
            token_ratio = response_tokens / self.config.max_token_length
            if token_ratio > 0.1:
                base_env_reward_before_scale = env_reward
                if env_reward > 0:
                    efficiency_factor = 1.5 - (0.75 * token_ratio)
                    env_reward *= max(0.1, efficiency_factor)
                    if self.debug_mode:
                        logger.debug(
                            f"[SCORE] Scaled positive env reward from {base_env_reward_before_scale:.2f}"
                            f" to {env_reward:.2f} (Ratio: {token_ratio:.2f}, "
                            f"Factor: {efficiency_factor:.2f})"
                        )
                else:
                    penalty_factor = 1.0 + token_ratio
                    env_reward *= penalty_factor
                    if self.debug_mode:
                        logger.debug(
                            f"[SCORE] Scaled negative env reward from {base_env_reward_before_scale:.2f}"
                            f" to {env_reward:.2f} (Ratio: {token_ratio:.2f}, "
                            f"Factor: {penalty_factor:.2f})"
                        )
        else:
            if self.debug_mode:
                logger.debug(
                    "[SCORE] Skipping token length scaling (max_token_length=0)"
                )

        env_weight = getattr(self.config, "environment_reward_weight", 0.7)
        format_weight = getattr(self.config, "format_reward_weight", 0.3)
        combined_reward = (env_weight * env_reward) + (format_weight * format_reward)

        if update_episode_totals:
            try:
                episode = self._get_or_create_episode(episode_seed)
                episode.total_env_reward += env_reward
                episode.total_format_reward += format_reward
                episode.total_combined_reward += combined_reward
                episode.num_total_actions += 1
                episode.step_rewards.append(combined_reward)
                if parsed_action != "-ERROR-":
                    episode.num_correct_actions += 1
            except Exception as e:
                logger.error(f"Error updating episode totals: {e}")
                if self.debug_mode:
                    logger.debug(f"[SCORE] Error updating episode totals: {e}")

        if self.debug_mode:
            logger.debug(
                f"[SCORE] Final Env Reward: {env_reward:.4f}, "
                f"Format Reward: {format_reward:.4f}, Combined Reward: {combined_reward:.4f}"
            )
        return combined_reward

    async def _select_best_action(
        self,
        episode: EpisodeState,
        actions: List[str],
        messages_list: List[List[Dict[str, str]]],
    ) -> Tuple[str, List[float]]:
        best_action, best_score = None, float("-inf")
        scores = [0.0] * len(actions)
        token_lengths = [0] * len(actions)

        temp_env_for_replay = ta.wrappers.LLMObservationWrapper(ta.make("Hangman-v0"))
        temp_env_for_replay.reset(seed=episode.seed, num_players=1)

        for past_group in episode.trajectory:
            past_action = past_group["parsed_action"]
            if past_action and past_action != "-ERROR-":
                temp_env_for_replay.env.step(past_action)

        state_after_history = copy.deepcopy(temp_env_for_replay.state)
        # Store the chosen_word from the environment
        chosen_word = temp_env_for_replay.env.chosen_word
        # Also store the guessed_letters from the environment
        guessed_letters = temp_env_for_replay.env.guessed_letters
        del temp_env_for_replay

        for action_idx, action in enumerate(actions):
            if action == "-ERROR-":
                scores[action_idx] = -1.0
                continue

            pre_state = copy.deepcopy(state_after_history)
            sim_env = ta.make("Hangman-v0")
            sim_env.reset(num_players=1, seed=episode.seed)
            pre_state.game_state["board"] = state_after_history.game_state["board"]
            pre_state.game_state["tries_left"] = state_after_history.game_state[
                "tries_left"
            ]
            sim_env.state = pre_state
            # Set the chosen_word from our stored value, not from the state
            sim_env.chosen_word = chosen_word
            # Set the guessed_letters from our stored value
            sim_env.guessed_letters = guessed_letters
            # <<< ADDED: Ensure internal game_board used for win checks is consistent >>>
            sim_env.game_board = list(chosen_word)

            sim_env_wrapped = ta.wrappers.LLMObservationWrapper(sim_env)
            done, info = sim_env_wrapped.step(action)
            post_state = sim_env.state
            prevent_player_change = post_state.prevent_player_change

            response_text = messages_list[action_idx][-1]["content"]
            if response_text.startswith("<think>"):
                pass

            score = self._score_response(
                pre_state=pre_state,
                post_state=post_state,
                done=done,
                info=info,
                prevent_player_change=prevent_player_change,
                response_text=response_text,
                parsed_action=action,
                episode_seed=episode.seed,
                update_episode_totals=False,
            )
            scores[action_idx] = score

            token_lengths[action_idx] = len(self.tokenizer.encode(response_text))

            if score > best_score or (
                score == best_score
                and token_lengths[action_idx]
                < (
                    token_lengths[actions.index(best_action)]
                    if best_action is not None
                    else float("inf")
                )
            ):
                best_score = score
                best_action = action

        if best_action is None:
            best_action = next((a for a in actions if a != "-ERROR-"), actions[0])

        max_score = max(scores)
        for i, score in enumerate(scores):
            if score == max_score and i != scores.index(max_score):
                token_ratio = token_lengths[i] / self.config.max_token_length
                scores[i] -= 0.0001 * token_ratio

        return best_action, scores

    def _extract_last_paragraph(self, message: str) -> str:
        if not message:
            return "I need to make a guess."

        paragraphs = re.findall(r".*?(?:\n\n|\n$|\Z)", message, re.DOTALL)
        if not paragraphs:
            return "I need to make a guess."

        return paragraphs[-1].strip()

    def _extract_thinking_block(self, message: str) -> str:
        if not message:
            return "I need to make a guess."

        think_blocks = re.findall(r"<think>.*?</think>", message, re.DOTALL)
        if not think_blocks:
            return message

        return think_blocks[-1].strip()

    def _extract_after_thinking_block(self, message: str) -> str:
        if not message:
            return ""

        think_match = re.search(r"<think>.*?</think>", message, re.DOTALL)
        if not think_match:
            return message.strip()

        after_think = re.split(r"</think>", message, maxsplit=0)[-1]
        return after_think.strip()

    async def collect_trajectory(
        self, seed: int, interactive: bool = False
    ) -> List[ScoredDataGroup]:
        logger.warning(f"collect_trajectory: Starting with seed {seed}")
        episode = self._get_or_create_episode(seed)

        episode.message_history = [{"role": "system", "content": self.system_prompt}]
        episode.step_rewards = []
        episode.episode_rewards = []
        max_depth = self.config.max_turns
        episode.trajectory = []

        player_id, initial_observation_string = episode.env.get_observation()
        initial_raw_observation = (
            str(initial_observation_string)
            if not isinstance(initial_observation_string, str)
            else initial_observation_string
        )
        logger.warning(
            f"[INIT] Using Observation String: {initial_raw_observation[:100]}..."
        )

        initial_modified_observation = self._modify_observation(
            initial_raw_observation, episode
        )
        episode.current_board_state = episode.env.state.game_state["board"]
        episode.tries_left = episode.env.state.game_state["tries_left"]

        if self.debug_mode:
            logger.warning(f"[INIT] Starting trajectory collection with seed {seed}")
            logger.warning(f"[INIT] Max turns: {max_depth}")
            logger.warning("\n" + "◆" * 40)
            logger.warning("ENVIRONMENT OBSERVATION - Step 0:")
            logger.warning("-" * 80)
            logger.warning(initial_modified_observation)
            logger.warning(
                f"[STATE] Initial Board: {' '.join(episode.current_board_state)}"
            )
            logger.warning(f"[STATE] Initial Tries: {episode.tries_left}")
            logger.warning("◆" * 40)

        for i in range(max_depth):
            current_turn_modified_observation = ""
            if self.debug_mode:
                logger.warning(f"\n[STEP] Starting step {i+1}/{max_depth}")

            player_id, current_observation_string = episode.env.get_observation()
            current_raw_observation = (
                str(current_observation_string)
                if not isinstance(current_observation_string, str)
                else current_observation_string
            )
            logger.warning(
                f"[STEP {i+1}] Using Observation String: {current_raw_observation[:100]}..."
            )

            current_turn_modified_observation = self._modify_observation(
                current_raw_observation, episode
            )

            if self.debug_mode:
                logger.warning(
                    f"[STATE] Board state before step {i+1}: {' '.join(episode.current_board_state)}"
                )
                logger.warning(
                    f"[STATE] Tries left before step {i+1}: {episode.tries_left}"
                )

            current_prompt = (
                self.icl_messages
                + episode.message_history
                + [
                    {
                        "role": "environment",
                        "content": current_turn_modified_observation,
                    }
                ]
            )
            prompt = self.tokenizer.apply_chat_template(
                current_prompt + [{"role": "agent", "content": "<think>\n"}],
                tokenize=False,
            )

            step_info = f" - Step {i+1}"
            prompt_log_header = (
                "\n" + "▼" * 40 + f"\nMODEL PROMPT{step_info}:\n" + "-" * 80
            )
            prompt_log_footer = "\n" + "▼" * 40
            if len(prompt) > 2000:
                logger.warning(
                    f"{prompt_log_header}\n{prompt[:1000]}...[truncated]...{prompt[-1000:]}{prompt_log_footer}"
                )
            else:
                logger.warning(f"{prompt_log_header}\n{prompt}{prompt_log_footer}")

            completions = await self.server.completion(
                prompt=prompt,
                n=self.config.group_size,
                max_tokens=self.config.max_token_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )

            step_info = f" - Step {i+1}"
            completions_log_header = (
                "\n" + "▲" * 40 + f"\nMODEL COMPLETIONS{step_info}:\n" + "-" * 80
            )
            completions_log_footer = "\n" + "▲" * 40
            completions_text = ""
            for choice_idx, choice in enumerate(completions.choices):
                completions_text += f"\nCHOICE {choice_idx+1}:\n{choice.text}\n{'-'*40}"
            logger.warning(
                f"{completions_log_header}{completions_text}{completions_log_footer}"
            )

            if len(completions.choices) != self.config.group_size:
                if self.debug_mode:
                    logger.warning(
                        f"[ERROR] Did not receive {self.config.group_size} choices, cancelling episode"
                    )
                return []

            tokens, masks, messages_list, actions = [], [], [], []
            try:
                for idx, choice in enumerate(completions.choices):
                    response_text = (
                        choice.text if hasattr(choice, "text") else "Invalid response"
                    )
                    full_response_text_with_think = "<think>\n" + response_text
                    action = self._parse_tool_call(full_response_text_with_think)

                    if self.debug_mode:
                        step_info = f" - Step {i+1}"
                        logger.warning(f"\n→ PARSED ACTION{step_info}: {action}")

                    step_messages = [
                        {
                            "role": msg["role"],
                            "content": self._strip_think_tags(msg["content"]),
                        }
                        for msg in episode.message_history
                    ] + [
                        {
                            "role": "environment",
                            "content": current_turn_modified_observation,
                        }
                    ]
                    agent_msg = {
                        "role": "agent",
                        "content": full_response_text_with_think,
                    }
                    full_messages = step_messages + [agent_msg]

                    out_dict = tokenize_for_trainer(self.tokenizer, full_messages)
                    token_list = out_dict["tokens"]

                    if not isinstance(token_list, list):
                        if self.debug_mode:
                            logger.error("[ERROR] Invalid token list format")
                        return []
                    if not all(isinstance(t, int) for t in token_list):
                        if self.debug_mode:
                            logger.error("[ERROR] Invalid token type in list")
                        return []
                    tokens.append(token_list)
                    masks.append(out_dict["masks"])
                    messages_list.append(full_messages)
                    actions.append(action)
            except Exception as e:
                if self.debug_mode:
                    logger.exception(f"[ERROR] Error collecting trajectory: {e}")
                return []

            best_action, scores = await self._select_best_action(
                episode, actions, messages_list
            )

            if self.debug_mode:
                step_info = f" - Step {i+1}"
                logger.debug(f"\n★ ACTION{step_info}: {best_action}")
                best_idx = actions.index(best_action)
                logger.debug(
                    f"[SCORE] Selected best action (score: {scores[best_idx]:.4f})"
                )

            logger.info(f"collect_trajectory: Stepping best action: {best_action}")

            # Save pre-step state variables
            pre_board = copy.deepcopy(episode.current_board_state)
            pre_tries = episode.tries_left

            # Execute the step
            done, info = episode.env.step(best_action)

            # Get the post-step state
            best_action_idx = actions.index(best_action)
            response_text = messages_list[best_action_idx][-1]["content"]

            # Update current state variables
            episode.current_board_state = episode.env.state.game_state["board"]
            episode.tries_left = episode.env.state.game_state["tries_left"]

            # Manual score tracking instead of creating new State objects
            env_reward = 0.0
            format_reward = 0.0

            # Basic scoring: +1 for revealing letters, -0.2 for incorrect guesses
            if best_action != "-ERROR-":
                if done and "Congratulations" in info.get("reason", "").lower():
                    env_reward += 6.0
                    episode.games_won += 1
                elif done and "lost" in info.get("reason", "").lower():
                    env_reward -= 1.5
                else:
                    # Check if letters were revealed
                    newly_revealed = 0
                    for i in range(len(pre_board)):
                        if (
                            pre_board[i] == "_"
                            and episode.current_board_state[i] != "_"
                        ):
                            newly_revealed += 1

                    if newly_revealed > 0:
                        env_reward += newly_revealed * 1.0
                    elif pre_tries > episode.tries_left:
                        env_reward -= 0.2

                # Format reward (if configured)
                if self.reward_function:
                    format_completions = [
                        [{"role": "assistant", "content": response_text}]
                    ]
                    try:
                        format_rewards = self.reward_function(format_completions)
                        if format_rewards and len(format_rewards) > 0:
                            format_reward = format_rewards[0]
                    except Exception as e:
                        logger.error(f"Error calculating format reward: {e}")

                # Combined reward calculation
                env_weight = getattr(self.config, "environment_reward_weight", 0.7)
                format_weight = getattr(self.config, "format_reward_weight", 0.3)
                combined_reward = (env_weight * env_reward) + (
                    format_weight * format_reward
                )

                # Update episode totals
                episode.total_env_reward += env_reward
                episode.total_format_reward += format_reward
                episode.total_combined_reward += combined_reward
                episode.num_total_actions += 1
                episode.step_rewards.append(combined_reward)
                if best_action != "-ERROR-":
                    episode.num_correct_actions += 1

            player_id_after_step, actual_next_observation_string = (
                episode.env.get_observation()
            )
            actual_next_observation_raw_for_log = (
                str(actual_next_observation_string)
                if not isinstance(actual_next_observation_string, str)
                else actual_next_observation_string
            )
            logger.debug(
                f"[STEP {i+1} After] Using Observation String: {actual_next_observation_raw_for_log[:100]}..."
            )

            if self.debug_mode:
                step_info = f" - Step {i+1} (After Action: {best_action})"
                log_separator = "\n" + "◆" * 40
                logger.debug(log_separator)
                logger.debug(f"ENVIRONMENT OBSERVATION{step_info}:")
                logger.debug("-" * 80)
                if len(actual_next_observation_raw_for_log) > 500:
                    logger.debug(
                        f"{actual_next_observation_raw_for_log[:250]}...[truncated]..."
                        f"{actual_next_observation_raw_for_log[-250:]}"
                    )
                else:
                    logger.debug(actual_next_observation_raw_for_log)
                logger.debug("-" * 40)
                logger.debug(
                    f"[STATE] Updated board state: {' '.join(episode.current_board_state)}"
                )
                logger.debug(f"[STATE] Updated tries left: {episode.tries_left}")
                logger.debug(f"[STATE] Done flag: {done}")
                logger.debug(f"[STATE] Info: {info}")
                logger.debug(
                    f"[STATE] Prevent player change: {episode.env.state.prevent_player_change}"
                )
                logger.debug(log_separator)

            actual_next_observation_modified = self._modify_observation(
                actual_next_observation_raw_for_log, episode
            )

            try:
                episode.trajectory.append(
                    HangmanScoredDataGroup(
                        tokens=tokens,
                        masks=masks,
                        scores=scores,
                        messages=messages_list,
                        seed=seed,
                        parsed_action=best_action,
                    )
                )
            except Exception as e:
                if self.debug_mode:
                    logger.exception(f"[ERROR] Error storing trajectory step: {e}")

            best_action_idx = actions.index(best_action)

            try:
                best_full_response = completions.choices[best_action_idx].text

                thinking_block = self._extract_thinking_block(best_full_response)
                last_paragraph = self._extract_last_paragraph(thinking_block)
                after_thinking_block = self._extract_after_thinking_block(
                    best_full_response
                )

                if not last_paragraph.strip():
                    last_paragraph = "I need to make a guess."

                episode.message_history.extend(
                    [
                        {
                            "role": "environment",
                            "content": current_turn_modified_observation,
                        },
                        {
                            "role": "agent",
                            "content": (
                                f"<think>\n{last_paragraph}\n</think>\n"
                                f"{after_thinking_block}"
                            ),
                        },
                    ]
                )
            except Exception as history_error:
                if self.debug_mode:
                    logger.exception(
                        f"[ERROR] Error updating canonical history: {history_error}"
                    )
                episode.message_history.extend(
                    [
                        {
                            "role": "environment",
                            "content": current_turn_modified_observation,
                        },
                        {
                            "role": "agent",
                            "content": (
                                f"<think>\nI need to make a guess.\n</think>\n"
                                f"I'll make a guess: {best_action}"
                            ),
                        },
                    ]
                )

            if done:
                if self.debug_mode:
                    logger.debug(f"[DONE] Episode marked as done after step {i+1}")
                    logger.debug(f"[DONE] Reason: {info.get('reason', 'N/A')}")
                episode.episodes_run += 1
                if episode.env.state.rewards and episode.env.state.rewards.get(0) == 1:
                    episode.games_won += 1
                    if self.debug_mode:
                        logger.debug("[DONE] Outcome: Won")
                elif (
                    episode.env.state.rewards and episode.env.state.rewards.get(0) == -1
                ):
                    if self.debug_mode:
                        logger.debug("[DONE] Outcome: Lost")
                elif (
                    episode.env.state.rewards and episode.env.state.rewards.get(0) == 0
                ):
                    if self.debug_mode:
                        logger.debug("[DONE] Outcome: Draw")
                else:
                    if "Congratulations" in info.get("reason", "").lower():
                        episode.games_won += 1
                        if self.debug_mode:
                            logger.debug("[DONE] Outcome: Won (via info)")
                    elif "lost" in info.get("reason", "").lower():
                        if self.debug_mode:
                            logger.debug("[DONE] Outcome: Lost (via info)")
                    elif (
                        "draw" in info.get("reason", "").lower()
                        or "turn limit" in info.get("reason", "").lower()
                    ):
                        if self.debug_mode:
                            logger.debug("[DONE] Outcome: Draw (via info)")

                episode.message_history.append(
                    {
                        "role": "environment",
                        "content": actual_next_observation_modified,
                    }
                )
                break

            if len(episode.trajectory) >= self.config.max_turns:
                if self.debug_mode:
                    logger.debug(
                        f"[DONE] Episode reached max turns ({self.config.max_turns})"
                    )
                episode.episodes_run += 1
                break

        return episode.trajectory

    async def collect_trajectories(
        self, item: Tuple[int, int]
    ) -> Tuple[List[ScoredDataGroup], List[Message]]:
        seed, _ = item

        group_metrics: List[EpisodeMetrics] = []

        trajectory = await self.collect_trajectory(seed)

        if trajectory:
            scores_per_step = []
            token_lengths = []
            won = False

            for step in trajectory:
                if step["scores"]:
                    best_score = max(step["scores"])
                    scores_per_step.append(best_score)

                if step["tokens"]:
                    step_token_lengths = [len(tokens) for tokens in step["tokens"]]
                    token_lengths.extend(step_token_lengths)

                if step == trajectory[-1]:
                    for msg_list in step["messages"]:
                        if any(
                            "Congratulations" in msg.get("content", "")
                            for msg in msg_list
                        ):
                            won = True
                            break

            metrics = EpisodeMetrics(
                scores_per_step=scores_per_step,
                total_score=sum(scores_per_step),
                num_steps=len(scores_per_step),
                won=won,
                avg_token_length=(
                    sum(token_lengths) / len(token_lengths) if token_lengths else 0
                ),
                max_token_length=max(token_lengths) if token_lengths else 0,
            )
            group_metrics.append(metrics)

            if not hasattr(self, "recent_metrics"):
                self.recent_metrics = []
            self.recent_metrics.append(metrics)
            self.recent_metrics = self.recent_metrics[-100:]

        return trajectory, []

    async def score(
        self, rollout_group_data: List[ScoredDataGroup], interactive: bool = False
    ) -> List[Optional[ScoredDataGroup]]:
        logger.info(
            f"score: Scoring rollout_group_data with {len(rollout_group_data)} steps"
        )
        if not rollout_group_data:
            logger.warning("score: No valid rollout_group_data to score")
            return []

        seed = rollout_group_data[0]["seed"]

        canonical_actions = []
        for step_idx, group in enumerate(rollout_group_data):
            if not group["tokens"]:
                canonical_actions.append(None)
                continue
            best_score_idx = group["scores"].index(max(group["scores"]))
            content_prefix_len = len("<think>\n")
            best_response = group["messages"][best_score_idx][-1]["content"][
                content_prefix_len:
            ]
            action = self._parse_tool_call(best_response)
            canonical_actions.append(action)

        env_copy = ta.wrappers.LLMObservationWrapper(ta.make("Hangman-v0"))
        env_copy.reset(seed=seed, num_players=1)

        final_done = False
        final_info = {}
        steps_taken = 0
        for _, action in enumerate(canonical_actions):
            if action and action != "-ERROR-":
                steps_taken += 1
                final_done, final_info = env_copy.step(action)
                if final_done:
                    break

        final_bonus = 0.0
        if final_done:
            final_reward_val = (
                env_copy.state.rewards.get(0) if env_copy.state.rewards else None
            )
            if final_reward_val == 1:
                base_bonus = 7.0
                step_scale = (
                    max(0, (self.config.max_turns - steps_taken + 1))
                    / self.config.max_turns
                )
                final_bonus = base_bonus * step_scale
                if self.debug_mode:
                    logger.debug(
                        f"[SCORE][Bonus] Win detected. Steps: {steps_taken}, "
                        f"Scale: {step_scale:.2f}, Bonus: {final_bonus:.2f}"
                    )
            elif final_reward_val == -1:
                final_bonus = -1.5
                if self.debug_mode:
                    logger.debug(
                        f"[SCORE][Bonus] Loss detected. Bonus: {final_bonus:.2f}"
                    )
            elif final_reward_val == 0:
                final_bonus = -0.5
                if self.debug_mode:
                    logger.debug(
                        f"[SCORE][Bonus] Draw detected. Bonus: {final_bonus:.2f}"
                    )
            elif "Congratulations" in final_info.get("reason", "").lower():
                base_bonus = 7.0
                step_scale = (
                    max(0, (self.config.max_turns - steps_taken + 1))
                    / self.config.max_turns
                )
                final_bonus = base_bonus * step_scale
                if self.debug_mode:
                    logger.debug(
                        f"[SCORE][Bonus] Win detected (via info). Steps: {steps_taken}, "
                        f"Scale: {step_scale:.2f}, Bonus: {final_bonus:.2f}"
                    )
            elif "lost" in final_info.get("reason", "").lower():
                final_bonus = -1.5
                if self.debug_mode:
                    logger.debug(
                        f"[SCORE][Bonus] Loss detected (via info). Bonus: {final_bonus:.2f}"
                    )
            elif (
                "draw" in final_info.get("reason", "").lower()
                or "turn limit" in final_info.get("reason", "").lower()
            ):
                final_bonus = -0.5
                if self.debug_mode:
                    logger.debug(
                        f"[SCORE][Bonus] Draw detected (via info). Bonus: {final_bonus:.2f}"
                    )
        else:
            final_bonus = -0.5
            if self.debug_mode:
                logger.debug(
                    f"[SCORE][Bonus] Game incomplete after replay. Bonus: {final_bonus:.2f}"
                )

        scored_groups = []
        for step_idx, group in enumerate(rollout_group_data):
            if not group["tokens"]:
                scored_groups.append(None)
                continue
            scores = group["scores"].copy()
            best_score_idx = scores.index(max(scores))

            new_scores = [
                score + final_bonus if idx == best_score_idx else score
                for idx, score in enumerate(scores)
            ]

            token_lengths = []
            for msg_list in group["messages"]:
                response_text = ""
                if msg_list and isinstance(msg_list[-1], dict):
                    content_with_think = msg_list[-1].get("content", "")
                    if content_with_think.startswith("<think>"):
                        response_text = content_with_think
                    else:
                        response_text = content_with_think
                token_lengths.append(len(self.tokenizer.encode(response_text)))

            score_groups = {}
            for idx, score in enumerate(new_scores):
                if score not in score_groups:
                    score_groups[score] = []
                score_groups[score].append(idx)

            for score_val, indices in score_groups.items():
                if len(indices) > 1:
                    sorted_indices = sorted(indices, key=lambda i: token_lengths[i])

                    for rank, idx in enumerate(sorted_indices[1:], 1):
                        penalty = 0.0001 * rank
                        new_scores[idx] -= penalty

            scored_groups.append(
                ScoredDataGroup(
                    tokens=group["tokens"],
                    masks=group["masks"],
                    scores=new_scores,
                    messages=group["messages"],
                    parsed_action=group["parsed_action"],
                )
            )

        return scored_groups

    async def get_next_item(self, i: int = None) -> int:
        """Returns the next seed to be used for trajectory collection.

        Args:
            i: Optional index to use for seed generation in testing

        Returns:
            The seed to be used for trajectory collection
        """
        return i if i is not None else random.randint(0, 1000000)

    async def evaluate(self):
        eval_seeds = [random.randint(0, 1000000) for _ in range(10)]
        eval_tasks = [self.collect_trajectory(seed) for seed in eval_seeds]
        results = await asyncio.gather(*eval_tasks)
        wins_count = sum(
            1
            for result in results
            if isinstance(result, tuple)
            and any("Congratulations" in step["next_observation"] for step in result[0])
        )
        logger.info(f"Eval win rate: {wins_count / 10}")

    @classmethod
    def config_init(
        cls, config_name: Optional[str] = None
    ) -> Tuple[HangmanEnvConfig, List[OpenaiConfig]]:
        """Load settings from the local configs directory."""
        # Path to current directory's configs/config_name.yaml
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = config_name or "hangman_default.yaml"
        cfg_path = os.path.join(current_dir, "configs", config_file)

        try:
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    raw = yaml.safe_load(f) or {}
                logger.info(f"Loaded config from {cfg_path}")
            else:
                logger.warning(
                    f"Config file not found at {cfg_path}, using empty defaults"
                )
                raw = {}

            env_conf = HangmanEnvConfig(**raw)
            server_confs = []

            for sc in raw.get("server_configs", []):
                api_key = sc.get("api_key", os.getenv("OPENAI_API_KEY", "x"))
                base_url = sc.get(
                    "base_url", os.getenv("OPENAI_API_BASE", "http://localhost:9004/v1")
                )
                openai_config_args = {
                    "model_name": sc.get(
                        "model_name",
                        os.getenv(
                            "OPENAI_MODEL",
                            "NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                        ),
                    ),
                    "api_key": api_key,
                    "num_requests_for_eval": sc.get("num_requests_for_eval", 256),
                    "base_url": base_url,
                }

                server_confs.append(OpenaiConfig(**openai_config_args))

            if not server_confs:
                server_confs = [
                    OpenaiConfig(
                        model_name=os.getenv(
                            "OPENAI_MODEL",
                            "NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                        ),
                        base_url=os.getenv(
                            "OPENAI_API_BASE", "http://localhost:9004/v1"
                        ),
                        api_key=os.getenv("OPENAI_API_KEY", "x"),
                        num_requests_for_eval=256,
                    )
                ]

            return env_conf, server_confs

        except Exception as e:
            logger.error(f"Error loading config: {e}")
            # Fall back to empty configs
            return HangmanEnvConfig(), [
                OpenaiConfig(
                    model_name=os.getenv("OPENAI_MODEL", "gpt-4.1-nano"),
                    base_url=os.getenv("OPENAI_API_BASE"),
                    api_key=os.getenv("OPENAI_API_KEY", ""),
                    num_requests_for_eval=1,
                )
            ]


if __name__ == "__main__":
    HangmanOnlineEnv.cli()
