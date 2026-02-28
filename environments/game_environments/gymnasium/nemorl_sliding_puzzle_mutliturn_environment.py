import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import wandb
from pydantic import Field
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
    Item,
    ScoredDataGroup,
)
from atroposlib.type_definitions import ChatMessage
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

system_prompt = (
    "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the "
    "problem and deliberate with yourself via systematic reasoning processes to help come to a correct "
    "solution prior to answering. You should enclose your thoughts and internal monologue inside <think> "
    "</think> tags, and then provide your solution or response to the problem.\\\\n\\\\n"
    "You are playing a sliding puzzle game. The puzzle is a grid of numbers with one blank space. "
    "Your goal is to arrange the numbers in ascending order from left to right, top to bottom, "
    "with the blank space at the end (bottom-right for a 3x3 puzzle, it would be tile 8 if tiles are 0-8).\\\\n"
    "At each step, you will be shown the current board configuration and a list of possible actions.\\\\n"
    "An action consists of moving the blank tile. The available actions are: \\\\n"
    "0: Move blank tile LEFT\\\\n"
    "1: Move blank tile UP\\\\n"
    "2: Move blank tile RIGHT\\\\n"
    "3: Move blank tile DOWN\\\\n"
    "You must respond with ONLY the action number after your thinking process. For example:\\\\n"
    "<think>The blank is at (1,1). To move it left, the target position (1,0) must be valid. "
    "If it is, action 0 is a good candidate. Let me check the board state. "
    "Okay, moving left is possible and seems to progress towards the goal.</think>0"
)


class NemoRPSlidingPuzzleEnvConfig(BaseEnvConfig):
    board_size: int = Field(
        3, description="Size of the puzzle board (e.g., 3 for 3x3)."
    )
    max_steps_per_puzzle: int = Field(
        50, description="Maximum number of steps (LLM turns) allowed to solve a puzzle."
    )
    debug: bool = Field(
        False, description="Enable debug printing throughout the environment."
    )


class _SlidingPuzzleGame:
    """
    Adapted from NVIDIA NeMo-RL's SlidingPuzzle environment.
    Manages the game state and logic for a single sliding puzzle.
    """

    def __init__(
        self,
        board_size: int = 3,
        initial_state: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        debug_mode: bool = False,
    ):
        self.board_size = board_size
        self.num_tiles = self.board_size * self.board_size
        # Blank tile is represented by num_tiles - 1 (e.g., 8 for a 3x3 puzzle with tiles 0-8)
        self.blank_tile_value = self.num_tiles - 1
        self.solved_state = np.arange(self.num_tiles).reshape(
            (self.board_size, self.board_size)
        )
        self.debug_mode = debug_mode

        self._rng = np.random.RandomState(seed)

        if self.debug_mode:
            print(
                f"[DEBUG _SlidingPuzzleGame] Initializing with board_size={board_size}, seed={seed}"
            )
            print(f"[DEBUG _SlidingPuzzleGame] Solved state:\\n{self.solved_state}")

        if initial_state is not None:
            self.state = initial_state.reshape((self.board_size, self.board_size))
            if self.debug_mode:
                print(
                    f"[DEBUG _SlidingPuzzleGame] Using provided initial state:\\n{self.state}"
                )
        else:
            self.reset()

    def _get_blank_pos(self) -> Tuple[int, int]:
        pos = np.where(self.state == self.blank_tile_value)
        return pos[0][0], pos[1][0]

    def _is_solvable(self, board_config: np.ndarray) -> bool:
        # Flatten the board and remove the blank tile for inversion calculation
        flat_board = board_config.flatten()
        inversion_sequence = flat_board[flat_board != self.blank_tile_value]

        inversions = 0
        for i in range(len(inversion_sequence)):
            for j in range(i + 1, len(inversion_sequence)):
                if inversion_sequence[i] > inversion_sequence[j]:
                    inversions += 1

        solvable = False
        if self.board_size % 2 == 1:  # Odd-sized board
            solvable = inversions % 2 == 0
        else:  # Even-sized board
            blank_row_tuple = np.where(board_config == self.blank_tile_value)
            blank_row = blank_row_tuple[0][0]
            blank_row_from_bottom = self.board_size - blank_row
            if blank_row_from_bottom % 2 == 0:  # Blank is on an even row from bottom
                solvable = inversions % 2 == 1
            else:  # Blank is on an odd row from bottom
                solvable = inversions % 2 == 0

        if self.debug_mode:
            print(
                f"[DEBUG _SlidingPuzzleGame] Checking solvability for board:\\n{board_config}"
            )
            print(f"[DEBUG _SlidingPuzzleGame] Inversions: {inversions}")
            if self.board_size % 2 == 0:
                print(
                    "[DEBUG _SlidingPuzzleGame] Blank row: {}, "
                    "Blank row from bottom (1-indexed): {}".format(
                        blank_row, blank_row_from_bottom
                    )
                )
            print(f"[DEBUG _SlidingPuzzleGame] Is solvable: {solvable}")
        return solvable

    def reset(self) -> np.ndarray:
        if self.debug_mode:
            print("[DEBUG _SlidingPuzzleGame] Resetting puzzle.")
        while True:
            self.state = self._rng.permutation(self.num_tiles).reshape(
                (self.board_size, self.board_size)
            )
            if (
                self._is_solvable(self.state) and not self.is_solved()
            ):  # Ensure it\'s not already solved
                if self.debug_mode:
                    print(
                        "[DEBUG _SlidingPuzzleGame] Generated solvable and non-solved state:\\n"
                        f"{self.state}"
                    )
                break
            elif self.debug_mode:
                print(
                    "[DEBUG _SlidingPuzzleGame] Generated state (rejected, re-permuting):\\n"
                    f"{self.state}"
                )
        return self.state.copy()

    def get_possible_actions(self) -> List[int]:
        r, c = self._get_blank_pos()
        actions = []
        if c > 0:
            actions.append(0)  # Move blank LEFT (tile from right moves into blank)
        if r > 0:
            actions.append(1)  # Move blank UP (tile from below moves into blank)
        if c < self.board_size - 1:
            actions.append(2)  # Move blank RIGHT (tile from left moves into blank)
        if r < self.board_size - 1:
            actions.append(3)  # Move blank DOWN (tile from above moves into blank)
        if self.debug_mode:
            print(
                f"[DEBUG _SlidingPuzzleGame] Blank at ({r},{c}). Possible actions: {actions}"
            )
        return actions

    def step(self, action: int) -> Tuple[np.ndarray, bool, bool]:
        """
        Perform an action (move blank tile).
        Actions: 0:LEFT, 1:UP, 2:RIGHT, 3:DOWN
        Returns: (next_state, reward, done)
        Reward is 1.0 if solved, 0.0 otherwise. Done is True if solved.
        """
        if self.debug_mode:
            action_name = TO_ACTION_MAP.get(action, "Unknown Action")
            print(
                f"[DEBUG _SlidingPuzzleGame] Attempting action: {action} ({action_name})"
            )

        if action not in self.get_possible_actions():
            if self.debug_mode:
                print(
                    f"[DEBUG _SlidingPuzzleGame] Action {action} is invalid or not in possible actions."
                )
            # This case should ideally be prevented by LLM choosing from available actions
            return self.state.copy(), 0.0, self.is_solved()

        r, c = self._get_blank_pos()
        tile_to_move_r, tile_to_move_c = r, c

        if self.debug_mode:
            print(f"[DEBUG _SlidingPuzzleGame] Blank at ({r},{c}) before move.")

        if action == 0:  # Move blank LEFT => tile from RIGHT moves into blank
            tile_to_move_c += 1
        elif action == 1:  # Move blank UP => tile from DOWN moves into blank
            tile_to_move_r += 1
        elif action == 2:  # Move blank RIGHT => tile from LEFT moves into blank
            tile_to_move_c -= 1
        elif action == 3:  # Move blank DOWN => tile from UP moves into blank
            tile_to_move_r -= 1

        # Swap blank with the chosen tile
        self.state[r, c], self.state[tile_to_move_r, tile_to_move_c] = (
            self.state[tile_to_move_r, tile_to_move_c],
            self.state[r, c],
        )

        if self.debug_mode:
            print(
                f"[DEBUG _SlidingPuzzleGame] Moved tile from ({tile_to_move_r},{tile_to_move_c}) "
                f"to ({r},{c}). New blank at ({tile_to_move_r},{tile_to_move_c})"
            )
            print(
                f"[DEBUG _SlidingPuzzleGame] Board after action {action}:\\n{self.state}"
            )

        solved = self.is_solved()
        reward = 1.0 if solved else 0.0
        if self.debug_mode:
            print(
                f"[DEBUG _SlidingPuzzleGame] Action resulted in reward={reward}, solved={solved}"
            )
        return self.state.copy(), reward, solved

    def is_solved(self) -> bool:
        return np.array_equal(self.state, self.solved_state)

    def render(self) -> str:
        board_str = ""
        for i in range(self.board_size):
            row_str = []
            for j in range(self.board_size):
                tile = self.state[i, j]
                if tile == self.blank_tile_value:
                    row_str.append("B")  # Blank
                else:
                    row_str.append(str(tile))
            board_str += " ".join(row_str) + "\\\\n"
        # Replace last \\\\n with empty string
        if board_str.endswith("\\\\n"):
            board_str = board_str[:-2]
        return board_str


class NemoRPSlidingPuzzleEnv(BaseEnv):
    name = "nemorl_sliding_puzzle"
    env_config_cls = NemoRPSlidingPuzzleEnvConfig

    def __init__(
        self,
        config: NemoRPSlidingPuzzleEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: NemoRPSlidingPuzzleEnvConfig  # For type hinting
        self.percent_solved_buffer = list()  # Renamed from percent_correct for clarity
        self.steps_to_solve_buffer = list()
        self.eval_metrics = list()
        self.rollouts_for_wandb = []  # For logging conversations
        if self.config.debug:
            print(
                f"[DEBUG NemoRPSlidingPuzzleEnv] Initialized. debug_mode={self.config.debug}, "
                f"board_size={self.config.board_size}, max_steps={self.config.max_steps_per_puzzle}"
            )

    @classmethod
    def config_init(cls) -> Tuple[NemoRPSlidingPuzzleEnvConfig, List[APIServerConfig]]:
        env_config = NemoRPSlidingPuzzleEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",  # Example tokenizer
            group_size=8,  # Number of attempts per puzzle instance
            use_wandb=True,
            rollout_server_url="http://localhost:8000",  # Example
            total_steps=1000,
            batch_size=128,  # Number of ScoredDataItems to batch for training
            steps_per_eval=100,
            max_token_length=2048,  # Max length for the whole conversation
            wandb_name="nemorl_sliding_puzzle",
            board_size=3,
            max_steps_per_puzzle=30,  # Adjusted for a 3x3, might need tuning
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,  # Standard practice
            eval_limit_ratio=0.1,
            num_rollouts_to_keep=10,  # For wandb table logging
            num_rollouts_per_group_for_logging=1,
            debug=False,  # Default debug to False
        )
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",  # Example model
                base_url="http://localhost:9001/v1",  # Example API
                api_key="x",
                num_requests_for_eval=64,  # For evaluation
            ),
        ]
        return env_config, server_configs

    async def setup(self):
        self.iter = 0  # For generating unique puzzle seeds

    def save_checkpoint(self, step, data=None):
        if data is None:
            data = {}
        data["iter"] = self.iter
        super().save_checkpoint(step, data)

    async def get_next_item(self) -> Item:
        """
        Generates a new puzzle instance (or parameters to create one).
        """
        item = {
            "puzzle_seed": self.iter,
            "board_size": self.config.board_size,
            "max_steps": self.config.max_steps_per_puzzle,
        }
        self.iter += 1
        if self.config.debug:
            print(f"[DEBUG NemoRPSlidingPuzzleEnv] get_next_item: {item}")
        return item

    def _format_prompt_for_turn(
        self, game: _SlidingPuzzleGame, history: List[ChatMessage]
    ) -> List[ChatMessage]:
        """
        Creates the prompt for the LLM for the current turn.
        """
        current_board_render = game.render()
        possible_actions = game.get_possible_actions()
        actions_str = ", ".join(
            [f"{act} ({TO_ACTION_MAP.get(act, 'Unknown')})" for act in possible_actions]
        )

        user_content = (
            f"Current board state:\\\\n{current_board_render}\\\\n\\\\n"
            f"Available actions for the blank tile (B): {actions_str}\\\\n"
            "Recall: 0:LEFT, 1:UP, 2:RIGHT, 3:DOWN for the blank tile.\\\\n"
            "Provide your thinking process and then the action number."
        )

        if self.config.debug:
            print(
                "[DEBUG NemoRPSlidingPuzzleEnv] _format_prompt_for_turn - User content:\\n"
                f"{user_content}"
            )

        if not history:  # First turn
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
        else:  # Subsequent turns
            # We only append the new user message. The history is already there.
            return [{"role": "user", "content": user_content}]

    def _extract_action_with_thinking(
        self, response_text: str, possible_actions: List[int]
    ) -> Optional[int]:
        """
        Extracts the action number from the LLM's response.
        Enforces <think> </think> tags.
        """
        if self.config.debug:
            print(
                "[DEBUG NemoRPSlidingPuzzleEnv] _extract_action_with_thinking - Raw response:\\n"
                f"{response_text}"
            )

        # Check for multiple <think> tags
        think_open_tags = re.findall(r"<think>", response_text, re.IGNORECASE)
        if len(think_open_tags) > 1:
            return None
        if not think_open_tags:  # Must have one <think>
            return None

        think_close_tags = re.findall(r"</think>", response_text, re.IGNORECASE)
        if len(think_close_tags) != 1:  # Must have exactly one closing tag
            return None

        # Split the text into thinking and answer sections
        parts = re.split(r"</think>", response_text, flags=re.IGNORECASE, maxsplit=1)
        if len(parts) != 2:  # Should be two parts: thinking and action
            return None

        thinking_section, answer_section = parts
        thinking_section = thinking_section.strip()
        answer_section = answer_section.strip()

        if self.config.debug:
            print(
                f"[DEBUG NemoRPSlidingPuzzleEnv] Thinking part: '{thinking_section[:100]}...'"
            )
            print(f"[DEBUG NemoRPSlidingPuzzleEnv] Answer part: '{answer_section}'")

        if not thinking_section.lower().startswith("<think>"):  # Malformed thinking
            if self.config.debug:
                print(
                    "[DEBUG NemoRPSlidingPuzzleEnv] Malformed thinking section "
                    "(does not start with <think>). Returning None."
                )
            return None
        if "<think>" in answer_section.lower():  # No nested or subsequent think tags
            if self.config.debug:
                print(
                    "[DEBUG NemoRPSlidingPuzzleEnv] Found <think> in answer section. "
                    "Returning None."
                )
            return None

        # Try to find a number in the answer section
        match = re.search(r"(\\d+)", answer_section)
        if match:
            try:
                action = int(match.group(1))
                if action in possible_actions:
                    if self.config.debug:
                        print(
                            f"[DEBUG NemoRPSlidingPuzzleEnv] Extracted action: {action}"
                        )
                    return action
                elif self.config.debug:
                    print(
                        f"[DEBUG NemoRPSlidingPuzzleEnv] Extracted number {action} "
                        f"not in possible_actions {possible_actions}. Returning None."
                    )
            except ValueError:
                if self.config.debug:
                    print(
                        "[DEBUG NemoRPSlidingPuzzleEnv] Could not parse number from answer part. "
                        "Returning None."
                    )
                return None  # Not a valid integer
        elif self.config.debug:
            print(
                "[DEBUG NemoRPSlidingPuzzleEnv] No digit found in answer part. "
                "Returning None."
            )
        return None

    async def collect_trajectories(
        self, item: Item
    ) -> Tuple[Optional[ScoredDataGroup], List[Item]]:
        """
        For a given puzzle item, run `group_size` attempts (play-throughs) by the LLM.
        Each attempt is a sequence of turns up to `max_steps` or until solved/failed.
        """
        puzzle_seed = item["puzzle_seed"]
        board_size = item["board_size"]
        max_steps = item["max_steps"]

        playthrough_results = (
            []
        )  # Stores (messages_list, solved_status, num_steps_taken, finish_reason_str)

        if self.config.debug:
            print(
                f"[DEBUG NemoRPSlidingPuzzleEnv] collect_trajectories for item: {item}"
            )

        # We run self.config.group_size independent playthroughs for the SAME puzzle item (seed)
        for i_playthrough in range(self.config.group_size):
            if self.config.debug:
                print(
                    f"[DEBUG NemoRPSlidingPuzzleEnv] Starting playthrough "
                    f"{i_playthrough + 1}/{self.config.group_size} for seed {puzzle_seed}"
                )
            game = _SlidingPuzzleGame(
                board_size=board_size, seed=puzzle_seed, debug_mode=self.config.debug
            )  # Fresh game for each attempt with same seed
            current_messages: List[ChatMessage] = []

            solved_this_attempt = False
            steps_taken = 0
            finish_reason = "max_steps_reached"

            for step_num in range(max_steps):
                steps_taken = step_num + 1
                if self.config.debug:
                    print(
                        f"[DEBUG NemoRPSlidingPuzzleEnv] Playthrough {i_playthrough + 1}, "
                        f"Step {steps_taken}"
                    )

                prompt_messages_for_turn = self._format_prompt_for_turn(
                    game, current_messages
                )

                # Combine system (if first turn) with ongoing history for API call
                api_call_messages = []
                if not current_messages:  # First turn, add system prompt
                    api_call_messages.append(
                        {"role": "system", "content": system_prompt}
                    )
                api_call_messages.extend(
                    current_messages
                )  # Add previous turns user/assistant
                api_call_messages.append(
                    prompt_messages_for_turn[-1]
                )  # Add current user prompt

                # Check token length before API call
                # This is a simplified check; actual tokenization depends on specific model
                current_token_len = len(
                    self.tokenizer.apply_chat_template(
                        api_call_messages, add_generation_prompt=True
                    )
                )
                if (
                    current_token_len >= self.config.max_token_length - 50
                ):  # 50 as a buffer for response
                    finish_reason = "max_token_length_reached"
                    if self.config.debug:
                        print(
                            f"[DEBUG NemoRPSlidingPuzzleEnv] Max token length reached for "
                            f"playthrough {i_playthrough + 1}, step {steps_taken}. "
                            f"Current tokens: {current_token_len}"
                        )
                    break

                try:
                    if self.config.debug:
                        print(
                            f"[DEBUG NemoRPSlidingPuzzleEnv] API call messages "
                            f"(Playthrough {i_playthrough + 1}, Step {steps_taken}):\\n"
                            f"{api_call_messages}"
                        )
                    completion = await self.server.chat_completion(
                        messages=api_call_messages,
                        n=1,
                        max_tokens=150,  # Max tokens for one turn\'s response (thought + action)
                        temperature=0.7,  # Allow some exploration
                    )
                    llm_response_content = completion.choices[0].message.content
                    if self.config.debug:
                        print(
                            f"[DEBUG NemoRPSlidingPuzzleEnv] LLM response "
                            f"(Playthrough {i_playthrough + 1}, Step {steps_taken}):\\n"
                            f"{llm_response_content}"
                        )

                    # Add current user prompt and assistant response to history for this playthrough
                    current_messages.append(
                        prompt_messages_for_turn[-1]
                    )  # The user message that prompted this
                    current_messages.append(
                        {"role": "assistant", "content": llm_response_content}
                    )

                    action = self._extract_action_with_thinking(
                        llm_response_content, game.get_possible_actions()
                    )

                    if action is None:
                        finish_reason = "invalid_action_format"
                        if self.config.debug:
                            print(
                                f"[DEBUG NemoRPSlidingPuzzleEnv] Invalid action format for "
                                f"playthrough {i_playthrough + 1}, step {steps_taken}. "
                                "Ending playthrough."
                            )
                        break

                    if self.config.debug:
                        print(
                            f"[DEBUG NemoRPSlidingPuzzleEnv] Game pre-step board "
                            f"(Playthrough {i_playthrough + 1}, Step {steps_taken}):\\n{game.render()}"
                        )
                    _, _, solved = game.step(action)
                    if self.config.debug:
                        print(
                            f"[DEBUG NemoRPSlidingPuzzleEnv] Game post-step board "
                            f"(Playthrough {i_playthrough + 1}, Step {steps_taken}):\\n{game.render()}, "
                            f"Solved: {solved}"
                        )

                    if solved:
                        solved_this_attempt = True
                        finish_reason = "solved"
                        if self.config.debug:
                            print(
                                f"[DEBUG NemoRPSlidingPuzzleEnv] Puzzle SOLVED for "
                                f"playthrough {i_playthrough + 1} at step {steps_taken}!"
                            )
                        break
                except Exception as e:
                    # self.logger.error(f"Error during LLM call or game step: {e}") # TODO: Add logger
                    print(
                        f"[DEBUG NemoRPSlidingPuzzleEnv] Exception in playthrough "
                        f"{i_playthrough + 1}, step {steps_taken}: {e}"
                    )
                    finish_reason = "error_in_playthrough"
                    break  # End this attempt

            if self.config.debug:
                print(
                    f"[DEBUG NemoRPSlidingPuzzleEnv] Playthrough {i_playthrough + 1} ended. "
                    f"Solved: {solved_this_attempt}, Steps: {steps_taken}, Reason: {finish_reason}"
                )
            playthrough_results.append(
                (current_messages, solved_this_attempt, steps_taken, finish_reason)
            )

        # Now, score these playthroughs
        return await self.score(playthrough_results, item), []

    async def score(
        self,
        playthrough_results: List[Tuple[List[ChatMessage], bool, int, str]],
        original_item: Item,  # For logging
    ) -> Optional[ScoredDataGroup]:
        """
        Scores the collected playthroughs.
        Each playthrough is a full conversation.
        """
        if self.config.debug:
            print(
                f"[DEBUG NemoRPSlidingPuzzleEnv] score method called with "
                f"{len(playthrough_results)} playthrough_results."
            )

        scored_data_group = ScoredDataGroup(
            tokens=[], masks=[], scores=[], overrides=[], messages=[]
        )

        # Log one full rollout for wandb if conditions met
        # We\'ll pick the first one that was solved, or just the first one if none solved
        # This is for qualitative inspection
        logged_one_for_wandb = False

        for i_res, (messages, solved, steps, reason) in enumerate(playthrough_results):
            if (
                not messages
            ):  # Skip if a playthrough had no messages (e.g. immediate error)
                if self.config.debug:
                    print(
                        f"[DEBUG NemoRPSlidingPuzzleEnv] Skipping scoring for "
                        f"playthrough_result {i_res} due to no messages."
                    )
                continue

            current_score = 1.0 if solved else 0.0
            if self.config.debug:
                print(
                    f"[DEBUG NemoRPSlidingPuzzleEnv] Scoring playthrough_result {i_res}: "
                    f"Solved={solved}, Steps={steps}, Reason='{reason}', Initial Score={current_score}"
                )

            try:
                # Tokenize the entire conversation for this playthrough
                tokenization_output = tokenize_for_trainer(
                    self.tokenizer,
                    chat=messages,
                    finish_reason="solved" if solved else reason,  # Pass a reason
                    include_messages=True,  # Keep messages for potential length penalty later
                )
            except Exception as e:
                print(f"Tokenization failed for a playthrough: {e}")
                continue

            if self.config.debug:
                print(
                    f"[DEBUG NemoRPSlidingPuzzleEnv] Tokenization output for playthrough_result {i_res}: "
                    f"tokens_len={len(tokenization_output['tokens'])}, "
                    f"masks_len={len(tokenization_output['masks'])}"
                )

            # Basic filter: ensure there\'s some content to learn from
            if (
                len([m for m in tokenization_output["masks"] if m != -100]) < 5
            ):  # Min 5 learnable tokens
                if self.config.debug:
                    print(
                        f"[DEBUG NemoRPSlidingPuzzleEnv] Filtering playthrough_result {i_res} "
                        "due to insufficient learnable tokens."
                    )
                continue

            scored_data_group["tokens"].append(tokenization_output["tokens"])
            scored_data_group["masks"].append(tokenization_output["masks"])
            scored_data_group["scores"].append(current_score)
            scored_data_group["overrides"].append({})  # No specific overrides initially
            scored_data_group["messages"].append(
                tokenization_output["messages"]
            )  # Store for length penalty

            self.percent_solved_buffer.append(
                current_score
            )  # 1.0 if solved, 0.0 otherwise
            if solved:
                self.steps_to_solve_buffer.append(steps)

            if (
                not logged_one_for_wandb
                and self.config.num_rollouts_per_group_for_logging > 0
            ):
                if solved or not any(
                    res[1] for res in playthrough_results
                ):  # Log a solved one, or any if none solved
                    # wandb_item_info assignment removed here as it was unused (F841)
                    logged_one_for_wandb = True

        if not scored_data_group["tokens"]:  # No valid data to score
            if self.config.debug:
                print(
                    "[DEBUG NemoRPSlidingPuzzleEnv] No valid data to score after "
                    "processing all playthroughs. Returning None."
                )
            return None

        if self.config.debug:
            print(
                "[DEBUG NemoRPSlidingPuzzleEnv] Scores before length penalty: "
                f"{scored_data_group['scores']}"
            )

        # Length Penalty: If all attempts in this group scored 1.0 (all solved the puzzle)
        if all(s == 1.0 for s in scored_data_group["scores"]):
            if self.config.debug:
                print(
                    "[DEBUG NemoRPSlidingPuzzleEnv] All scores are 1.0. Applying length penalty."
                )
            token_lengths = [len(tokens) for tokens in scored_data_group["tokens"]]

            # Max token length from config (for the whole conversation)
            max_allowed_length = self.config.max_token_length
            # Threshold for penalty (e.g. 50% of max_token_length)
            length_threshold = max_allowed_length * 0.5

            new_scores = []
            for i, length in enumerate(token_lengths):
                if length <= length_threshold:
                    new_scores.append(1.0)  # No penalty
                else:
                    # Percentage of range used above threshold
                    percentage_of_range = (length - length_threshold) / (
                        max_allowed_length - length_threshold
                    )
                    percentage_of_range = min(
                        max(percentage_of_range, 0.0), 1.0
                    )  # Clamp to [0,1]

                    # Linear penalty: score = 1.0 - percentage_of_range
                    # This means score goes from 1.0 (at threshold) down to 0.0 (at max_allowed_length)
                    new_scores.append(1.0 - percentage_of_range)
            scored_data_group["scores"] = new_scores
            if self.config.debug:
                print(f"[DEBUG NemoRPSlidingPuzzleEnv] Token lengths: {token_lengths}")
                print(
                    "[DEBUG NemoRPSlidingPuzzleEnv] Scores after length penalty: "
                    f"{scored_data_group['scores']}"
                )

        # If all scores are identical after potential length penalty, no learning signal.
        if (
            len(set(scored_data_group["scores"])) <= 1
            and len(scored_data_group["scores"]) > 1
        ):
            if self.config.debug:
                print(
                    f"[DEBUG NemoRPSlidingPuzzleEnv] All scores are identical "
                    f"({scored_data_group['scores']}) after potential penalty. Returning None."
                )
            return None  # Avoid training on batches with no contrast

        # Add one representative rollout to wandb from this group
        if logged_one_for_wandb and len(scored_data_group["tokens"]) > 0:
            # Find the index we decided to log (e.g., first solved, or just first)
            log_idx = 0
            for i, (msgs, solved_flag, _, _) in enumerate(playthrough_results):
                if i < len(
                    scored_data_group["scores"]
                ):  # Ensure index is valid for scored_data_group
                    if solved_flag:
                        log_idx = i
                        break
            if log_idx >= len(scored_data_group["tokens"]):
                log_idx = 0  # Fallback

            if (
                len(scored_data_group["tokens"]) > 0
            ):  # Check again after potential filtering
                wandb_item_tuple = (
                    self.tokenizer.decode(scored_data_group["tokens"][log_idx]),
                    scored_data_group["scores"][log_idx],
                    (
                        f"Seed: {original_item['puzzle_seed']}, Solved: {playthrough_results[log_idx][1]}, "
                        f"Steps: {playthrough_results[log_idx][2]}"
                    ),
                    playthrough_results[log_idx][3],  # finish reason
                )
                # Create a temporary ScoredDataGroup for this single item for logging
                # The add_rollouts_for_wandb expects a ScoredDataGroup, not a list of them.
                # And its item argument structure varies.
                # Let\'s adapt to what mcqa_thinking_env.py does:
                # self.rollouts_for_wandb.append([(decoded_text, score, answer_letter, answer_string)])
                self.rollouts_for_wandb.append([wandb_item_tuple])
                if len(self.rollouts_for_wandb) > self.config.num_rollouts_to_keep:
                    self.rollouts_for_wandb.pop(0)

        return scored_data_group

    async def rollout_and_score_eval(self, test_item: Item) -> float:
        """
        Evaluate a single test item.
        Returns 1.0 if solved, 0.0 otherwise.
        """
        if self.config.debug:
            print(
                f"[DEBUG NemoRPSlidingPuzzleEnv] rollout_and_score_eval for item: {test_item}"
            )
        game = _SlidingPuzzleGame(
            board_size=test_item["board_size"],
            seed=test_item["puzzle_seed"],
            debug_mode=self.config.debug,
        )
        current_messages: List[ChatMessage] = []

        for i_step in range(test_item["max_steps"]):
            if self.config.debug:
                print(
                    f"[DEBUG NemoRPSlidingPuzzleEnv] Eval step {i_step + 1}/{test_item['max_steps']}"
                )
            prompt_messages_for_turn = self._format_prompt_for_turn(
                game, current_messages
            )
            api_call_messages = []
            if not current_messages:
                api_call_messages.append({"role": "system", "content": system_prompt})
            api_call_messages.extend(current_messages)
            api_call_messages.append(prompt_messages_for_turn[-1])

            current_token_len = len(
                self.tokenizer.apply_chat_template(
                    api_call_messages, add_generation_prompt=True
                )
            )
            if current_token_len >= self.config.max_token_length - 50:
                if self.config.debug:
                    print(
                        "[DEBUG NemoRPSlidingPuzzleEnv] Eval max token length reached at step "
                        f"{i_step + 1}."
                    )
                break

            try:
                if self.config.debug:
                    print(
                        "[DEBUG NemoRPSlidingPuzzleEnv] Eval API call messages "
                        f"(Step {i_step + 1}):\\n{api_call_messages}"
                    )
                completion = await self.server.chat_completion(
                    messages=api_call_messages,
                    n=1,
                    max_tokens=150,
                    temperature=0.0,  # Deterministic for eval
                    split="eval",
                )
                llm_response_content = completion.choices[0].message.content
                if self.config.debug:
                    print(
                        "[DEBUG NemoRPSlidingPuzzleEnv] Eval LLM response "
                        f"(Step {i_step + 1}):\\n{llm_response_content}"
                    )
                current_messages.append(prompt_messages_for_turn[-1])
                current_messages.append(
                    {"role": "assistant", "content": llm_response_content}
                )

                action = self._extract_action_with_thinking(
                    llm_response_content, game.get_possible_actions()
                )
                if action is None:
                    if self.config.debug:
                        print(
                            "[DEBUG NemoRPSlidingPuzzleEnv] Eval invalid action format at step "
                            f"{i_step + 1}. Returning 0.0."
                        )
                    return 0.0  # Invalid action format during eval

                _, _, solved = game.step(action)
                if solved:
                    if self.config.debug:
                        print(
                            "[DEBUG NemoRPSlidingPuzzleEnv] Eval puzzle SOLVED at step "
                            f"{i_step + 1}!"
                        )
                    return 1.0
            except Exception as e:
                if self.config.debug:
                    print(
                        f"[DEBUG NemoRPSlidingPuzzleEnv] Exception during eval step {i_step + 1}: "
                        f"{e}. Returning 0.0."
                    )
                return 0.0  # Error during eval
        if self.config.debug:
            print(
                "[DEBUG NemoRPSlidingPuzzleEnv] Eval not solved within max_steps. "
                "Returning 0.0."
            )
        return 0.0  # Not solved within max_steps

    async def evaluate(self, *args, **kwargs):
        if self.config.debug:
            print("[DEBUG NemoRPSlidingPuzzleEnv] evaluate method called.")
        eval_tasks = []
        # Create a small, fixed set of eval puzzle seeds or use a portion of iter sequence
        num_eval_puzzles = (
            self.config.server_configs[0].num_requests_for_eval
            if self.config.server_configs
            else 64
        )

        # Generate test items similar to get_next_item but for evaluation
        # To ensure consistent evaluation, use a fixed range of seeds or a dedicated test set if available.
        # For now, let\'s use a fixed range of seeds based on initial iter or a new counter.
        eval_start_seed = 1000000  # A large number to separate from training seeds
        for i in range(num_eval_puzzles):
            eval_item = {
                "puzzle_seed": eval_start_seed + i,
                "board_size": self.config.board_size,
                "max_steps": self.config.max_steps_per_puzzle,
            }
            eval_tasks.append(self.rollout_and_score_eval(eval_item))

        if not eval_tasks:
            return

        scores = await tqdm_asyncio.gather(*eval_tasks)
        if scores:
            avg_solved_rate = sum(scores) / len(scores)
            self.eval_metrics.append(("eval/percent_solved", avg_solved_rate))
            if self.config.debug:
                print(
                    f"[DEBUG NemoRPSlidingPuzzleEnv] Evaluation complete. Avg solved rate: "
                    f"{avg_solved_rate} over {len(scores)} puzzles."
                )
        elif self.config.debug:
            print(
                "[DEBUG NemoRPSlidingPuzzleEnv] Evaluation complete. No scores recorded."
            )
        # Could also log average steps for solved puzzles if tracked in eval

    async def create_rollout_table(self, wandb_metrics: Dict) -> Dict:
        """Creates a wandb.Table from stored rollouts."""
        if self.rollouts_for_wandb:
            # Columns: "full_conversation_text", "final_score", "puzzle_details", "finish_reason"
            table = wandb.Table(
                columns=["Conversation", "Score", "Puzzle Info", "Reason"]
            )
            for (
                rollout_group
            ) in self.rollouts_for_wandb:  # Each group has one entry here
                for (
                    item_tuple
                ) in rollout_group:  # item_tuple is (text, score, details, reason)
                    table.add_data(
                        item_tuple[0], item_tuple[1], item_tuple[2], item_tuple[3]
                    )
            wandb_metrics["train/example_rollouts"] = table
        self.rollouts_for_wandb = []  # Clear after logging
        return wandb_metrics

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}

        if self.percent_solved_buffer:
            wandb_metrics["train/percent_solved"] = sum(
                self.percent_solved_buffer
            ) / len(self.percent_solved_buffer)
            if self.config.debug:
                print(
                    "[DEBUG NemoRPSlidingPuzzleEnv] wandb_log: train/percent_solved = "
                    f"{wandb_metrics['train/percent_solved']}"
                )
            self.percent_solved_buffer = []

        if self.steps_to_solve_buffer:
            wandb_metrics["train/avg_steps_to_solve (if solved)"] = sum(
                self.steps_to_solve_buffer
            ) / len(self.steps_to_solve_buffer)
            if self.config.debug:
                print(
                    "[DEBUG NemoRPSlidingPuzzleEnv] wandb_log: train/avg_steps_to_solve = "
                    f"{wandb_metrics['train/avg_steps_to_solve (if solved)']}"
                )
            self.steps_to_solve_buffer = []

        for key, value in self.eval_metrics:
            wandb_metrics[key] = value
            if self.config.debug:
                print(
                    f"[DEBUG NemoRPSlidingPuzzleEnv] wandb_log: eval metric {key} = {value}"
                )
        self.eval_metrics = []

        # This will call create_rollout_table if it\'s defined in the parent or this class
        await super().wandb_log(wandb_metrics)


# Helper for rendering actions in prompts
TO_ACTION_MAP = {
    0: "Move Blank LEFT",
    1: "Move Blank UP",
    2: "Move Blank RIGHT",
    3: "Move Blank DOWN",
}


if __name__ == "__main__":
    NemoRPSlidingPuzzleEnv.cli()
