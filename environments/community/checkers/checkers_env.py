#!/usr/bin/env python3
"""
Checkers Environment: Trainer environment for training LLMs to play checkers

This environment implements a checkers game where an LLM learns to play against
a configurable opponent (random, basic AI, or another LLM). The environment
supports both "thinking" mode with <think> tags and standard mode.

Features:
- Standard 8x8 checkers board with American rules
- King promotion when reaching opposite end
- Mandatory jumps when available
- Multiple jumps in a single turn
- Tool-based action interface
- Configurable opponents (random, basic AI)
- Comprehensive game state tracking
"""

import json
import logging
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from atroposlib.envs.base import APIServerConfig, BaseEnv, BaseEnvConfig, ScoredDataItem
from atroposlib.type_definitions import Item, Message
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer
from atroposlib.utils.tool_call_parser import parse_tool_call

logger = logging.getLogger(__name__)


class PieceType(Enum):
    EMPTY = 0
    RED_PIECE = 1
    RED_KING = 2
    BLACK_PIECE = 3
    BLACK_KING = 4


class Player(Enum):
    RED = 1
    BLACK = 2


class OpponentType(Enum):
    RANDOM = "random"
    BASIC_AI = "basic_ai"


@dataclass
class Move:
    """Represents a checkers move with source and destination coordinates"""

    from_row: int
    from_col: int
    to_row: int
    to_col: int
    jumps: List[Tuple[int, int]] = None  # Captured pieces coordinates

    def __post_init__(self):
        if self.jumps is None:
            self.jumps = []

    def __str__(self):
        return f"({self.from_row},{self.from_col})->({self.to_row},{self.to_col})"


class CheckersBoard:
    """Represents the checkers game state and logic"""

    def __init__(self):
        self.board = [[PieceType.EMPTY for _ in range(8)] for _ in range(8)]
        self.current_player = Player.RED
        self.game_over = False
        self.winner = None
        self.move_count = 0
        self.setup_initial_board()

    def setup_initial_board(self):
        """Set up the initial checkers board position"""
        # Black pieces on top (rows 0-2)
        for row in range(3):
            for col in range(8):
                if (row + col) % 2 == 1:  # Dark squares only
                    self.board[row][col] = PieceType.BLACK_PIECE

        # Red pieces on bottom (rows 5-7)
        for row in range(5, 8):
            for col in range(8):
                if (row + col) % 2 == 1:  # Dark squares only
                    self.board[row][col] = PieceType.RED_PIECE

    def create_copy(self):
        """Create a deep copy of the board state"""
        new_board = CheckersBoard()
        new_board.board = [row[:] for row in self.board]
        new_board.current_player = self.current_player
        new_board.game_over = self.game_over
        new_board.winner = self.winner
        new_board.move_count = self.move_count
        return new_board

    def get_piece_at(self, row: int, col: int) -> PieceType:
        """Get the piece at the specified position"""
        if 0 <= row < 8 and 0 <= col < 8:
            return self.board[row][col]
        return PieceType.EMPTY

    def set_piece_at(self, row: int, col: int, piece: PieceType):
        """Set the piece at the specified position"""
        if 0 <= row < 8 and 0 <= col < 8:
            self.board[row][col] = piece

    def is_player_piece(self, piece: PieceType, player: Player) -> bool:
        """Check if a piece belongs to the specified player"""
        if player == Player.RED:
            return piece in [PieceType.RED_PIECE, PieceType.RED_KING]
        else:
            return piece in [PieceType.BLACK_PIECE, PieceType.BLACK_KING]

    def is_valid_position(self, row: int, col: int) -> bool:
        """Check if position is within board bounds"""
        return 0 <= row < 8 and 0 <= col < 8

    def get_possible_moves(self, player: Player) -> List[Move]:
        """Get all possible moves for the current player"""
        # First check for mandatory jumps
        jumps = self.get_jump_moves(player)
        if jumps:
            return jumps

        # If no jumps, return regular moves
        return self.get_regular_moves(player)

    def get_jump_moves(self, player: Player) -> List[Move]:
        """Get all possible jump moves for the player"""
        jumps = []
        for row in range(8):
            for col in range(8):
                piece = self.get_piece_at(row, col)
                if self.is_player_piece(piece, player):
                    jumps.extend(self.get_jumps_from_position(row, col, player, piece))
        return jumps

    def get_jumps_from_position(
        self, row: int, col: int, player: Player, piece: PieceType
    ) -> List[Move]:
        """Get all possible jump moves from a specific position (single jumps only)"""
        jumps = []
        directions = self.get_move_directions(piece)

        for dr, dc in directions:
            # Check for enemy piece to jump over
            enemy_row, enemy_col = row + dr, col + dc
            if not self.is_valid_position(enemy_row, enemy_col):
                continue

            enemy_piece = self.get_piece_at(enemy_row, enemy_col)
            if enemy_piece == PieceType.EMPTY or self.is_player_piece(
                enemy_piece, player
            ):
                continue

            # Check if landing square is empty
            land_row, land_col = enemy_row + dr, enemy_col + dc
            if not self.is_valid_position(land_row, land_col):
                continue

            if self.get_piece_at(land_row, land_col) != PieceType.EMPTY:
                continue

            # Create the jump move
            move = Move(row, col, land_row, land_col)
            move.jumps = [(enemy_row, enemy_col)]  # Set the captured piece
            jumps.append(move)

        return jumps

    def get_regular_moves(self, player: Player) -> List[Move]:
        """Get all regular (non-jump) moves for the player"""
        moves = []
        for row in range(8):
            for col in range(8):
                piece = self.get_piece_at(row, col)
                if self.is_player_piece(piece, player):
                    moves.extend(self.get_regular_moves_from_position(row, col, piece))
        return moves

    def get_regular_moves_from_position(
        self, row: int, col: int, piece: PieceType
    ) -> List[Move]:
        """Get regular moves from a specific position"""
        moves = []
        directions = self.get_move_directions(piece)

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if (
                self.is_valid_position(new_row, new_col)
                and self.get_piece_at(new_row, new_col) == PieceType.EMPTY
            ):
                moves.append(Move(row, col, new_row, new_col))

        return moves

    def get_move_directions(self, piece: PieceType) -> List[Tuple[int, int]]:
        """Get possible move directions for a piece type"""
        if piece == PieceType.RED_PIECE:
            return [(-1, -1), (-1, 1)]  # Move up (toward black side)
        elif piece == PieceType.BLACK_PIECE:
            return [(1, -1), (1, 1)]  # Move down (toward red side)
        else:  # Kings can move in all diagonal directions
            return [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    def apply_move(self, move: Move, validate: bool = True) -> bool:
        """Apply a move to the board"""
        # Validate the move first (unless we're in a copy operation)
        if validate and not self.is_valid_move(move):
            return False

        # Move the piece
        piece = self.get_piece_at(move.from_row, move.from_col)
        self.set_piece_at(move.from_row, move.from_col, PieceType.EMPTY)
        self.set_piece_at(move.to_row, move.to_col, piece)

        # Remove jumped pieces
        for jump_row, jump_col in move.jumps:
            self.set_piece_at(jump_row, jump_col, PieceType.EMPTY)

        # Check for king promotion
        if piece == PieceType.RED_PIECE and move.to_row == 0:
            self.set_piece_at(move.to_row, move.to_col, PieceType.RED_KING)
        elif piece == PieceType.BLACK_PIECE and move.to_row == 7:
            self.set_piece_at(move.to_row, move.to_col, PieceType.BLACK_KING)

        # Switch players
        self.current_player = (
            Player.BLACK if self.current_player == Player.RED else Player.RED
        )
        self.move_count += 1

        # Check for game over
        self.check_game_over()

        return True

    def is_valid_move(self, move: Move) -> bool:
        """Check if a move is valid"""
        possible_moves = self.get_possible_moves(self.current_player)
        for possible_move in possible_moves:
            if (
                move.from_row == possible_move.from_row
                and move.from_col == possible_move.from_col
                and move.to_row == possible_move.to_row
                and move.to_col == possible_move.to_col
            ):
                return True
        return False

    def check_game_over(self):
        """Check if the game is over and set winner"""
        red_pieces = black_pieces = 0
        red_can_move = black_can_move = False

        for row in range(8):
            for col in range(8):
                piece = self.get_piece_at(row, col)
                if self.is_player_piece(piece, Player.RED):
                    red_pieces += 1
                elif self.is_player_piece(piece, Player.BLACK):
                    black_pieces += 1

        # Check if players can move
        if red_pieces > 0:
            red_can_move = len(self.get_possible_moves(Player.RED)) > 0
        if black_pieces > 0:
            black_can_move = len(self.get_possible_moves(Player.BLACK)) > 0

        # Game over conditions
        if red_pieces == 0 or not red_can_move:
            self.game_over = True
            self.winner = Player.BLACK
        elif black_pieces == 0 or not black_can_move:
            self.game_over = True
            self.winner = Player.RED
        elif self.move_count >= 200:  # Draw after too many moves
            self.game_over = True
            self.winner = None

    def get_board_string(self) -> str:
        """Get a human-readable representation of the board"""
        symbols = {
            PieceType.EMPTY: ".",
            PieceType.RED_PIECE: "r",
            PieceType.RED_KING: "R",
            PieceType.BLACK_PIECE: "b",
            PieceType.BLACK_KING: "B",
        }

        result = "  0 1 2 3 4 5 6 7\n"
        for row in range(8):
            result += f"{row} "
            for col in range(8):
                if (row + col) % 2 == 0:
                    result += "  "  # Light squares (not used in checkers)
                else:
                    result += symbols[self.board[row][col]] + " "
            result += "\n"

        return result


class CheckersEnvConfig(BaseEnvConfig):
    """Configuration for the Checkers environment"""

    opponent_type: OpponentType = OpponentType.RANDOM
    max_episode_turns: int = 100
    eval_episodes: int = 50
    thinking_enabled: bool = True
    temperature: float = 0.7
    ai_plays_as: Player = Player.RED  # Which color the AI plays


class CheckersEnv(BaseEnv):
    name = "checkers"
    env_config_cls = CheckersEnvConfig

    def __init__(
        self,
        config: CheckersEnvConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = True,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: CheckersEnvConfig = config
        self.episode_outcomes_buffer: List[float] = []
        self.win_rate_buffer: List[float] = []
        self.move_count_buffer: List[int] = []
        self.invalid_move_buffer: List[float] = []
        self.eval_metrics_custom: List[Tuple[str, float]] = []

        # Define tools for the LLM to use
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "make_move",
                    "description": "Make a move in checkers by specifying source and destination coordinates",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "from_row": {
                                "type": "integer",
                                "description": "Source row (0-7)",
                                "minimum": 0,
                                "maximum": 7,
                            },
                            "from_col": {
                                "type": "integer",
                                "description": "Source column (0-7)",
                                "minimum": 0,
                                "maximum": 7,
                            },
                            "to_row": {
                                "type": "integer",
                                "description": "Destination row (0-7)",
                                "minimum": 0,
                                "maximum": 7,
                            },
                            "to_col": {
                                "type": "integer",
                                "description": "Destination column (0-7)",
                                "minimum": 0,
                                "maximum": 7,
                            },
                        },
                        "required": ["from_row", "from_col", "to_row", "to_col"],
                    },
                },
            }
        ]

        tools_json = json.dumps(self.tools)

        thinking_instruction = ""
        if self.config.thinking_enabled:
            thinking_instruction = (
                "You should enclose your thoughts and strategy analysis inside <think> </think> tags, "
                "considering the current board position, possible moves, and optimal strategy. "
            )

        self.system_prompt = (
            f"You are an AI agent playing checkers as the "
            f"{'RED' if config.ai_plays_as == Player.RED else 'BLACK'} player. "
            "You need to analyze the board position and make strategic moves to win the game.\n\n"
            "CHECKERS RULES:\n"
            "- Pieces move diagonally on dark squares only\n"
            "- Regular pieces can only move forward\n"
            "- Kings (promoted pieces) can move forward and backward\n"
            "- You must jump over opponent pieces when possible (mandatory jumps)\n"
            "- Multiple jumps in one turn are allowed and required\n"
            "- Pieces become kings when reaching the opposite end\n"
            "- Win by capturing all opponent pieces or blocking all moves\n\n"
            f"{thinking_instruction}"
            f"<tools>\n{tools_json}\n</tools>\n\n"
            "For your function call, return a JSON object with function name and arguments "
            "within <tool_call> </tool_call> tags with the following schema:\n"
            '<tool_call>\n{"arguments": {"from_row": 5, "from_col": 0, '
            '"to_row": 4, "to_col": 1}, "name": "make_move"}\n</tool_call>\n\n'
        )

        if self.config.thinking_enabled:
            self.system_prompt += (
                "Your full answer format should be:\n"
                "<think>\n[Your strategic analysis of the board position and move selection]\n</think>\n\n"
                '<tool_call>\n{"arguments": {"from_row": 5, "from_col": 0, '
                '"to_row": 4, "to_col": 1}, "name": "make_move"}\n</tool_call>'
            )
        else:
            self.system_prompt += (
                "Your answer format should be:\n"
                '<tool_call>\n{"arguments": {"from_row": 5, "from_col": 0, '
                '"to_row": 4, "to_col": 1}, "name": "make_move"}\n</tool_call>'
            )

    @classmethod
    def config_init(cls) -> Tuple[CheckersEnvConfig, List[APIServerConfig]]:
        env_config = CheckersEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=16,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            max_token_length=4096,
            wandb_name=cls.name,
            steps_per_eval=100,
            max_episode_turns=100,
            eval_episodes=50,
            opponent_type=OpponentType.RANDOM,
            thinking_enabled=True,
            ai_plays_as=Player.RED,
        )
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9001/v1",
                api_key="x",
                num_requests_for_eval=64,
            ),
        ]
        return env_config, server_configs

    def _parse_move_from_llm(self, llm_response: str) -> Optional[Move]:
        """Parse a move from the LLM's tool call response"""
        if not llm_response:
            logger.warning("Attempted to parse an empty LLM response.")
            return None

        parsed_name, parsed_args, is_error = parse_tool_call(
            llm_response, self.tools, ["tool_call"]
        )

        if is_error:
            logger.warning(
                f"Failed to parse tool call. Response: '{llm_response}'. Error: {parsed_name}"
            )
            return None

        if parsed_name != "make_move":
            logger.warning(
                f"Expected tool call name 'make_move', but got '{parsed_name}'"
            )
            return None

        try:
            from_row = int(parsed_args.get("from_row"))
            from_col = int(parsed_args.get("from_col"))
            to_row = int(parsed_args.get("to_row"))
            to_col = int(parsed_args.get("to_col"))

            return Move(from_row, from_col, to_row, to_col)
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid move arguments: {parsed_args}. Error: {e}")
            return None

    def _get_opponent_move(self, board: CheckersBoard) -> Optional[Move]:
        """Get a move from the configured opponent"""
        possible_moves = board.get_possible_moves(board.current_player)

        if not possible_moves:
            return None

        if self.config.opponent_type == OpponentType.RANDOM:
            return random.choice(possible_moves)
        elif self.config.opponent_type == OpponentType.BASIC_AI:
            return self._get_basic_ai_move(board, possible_moves)

        return None

    def _get_basic_ai_move(
        self, board: CheckersBoard, possible_moves: List[Move]
    ) -> Move:
        """Simple AI that prioritizes jumps and king moves"""
        # Prioritize moves with jumps (captures)
        jump_moves = [move for move in possible_moves if move.jumps]
        if jump_moves:
            # Prefer moves that capture more pieces
            return max(jump_moves, key=lambda m: len(m.jumps))

        # Prioritize king moves
        king_moves = []
        for move in possible_moves:
            piece = board.get_piece_at(move.from_row, move.from_col)
            if piece in [PieceType.RED_KING, PieceType.BLACK_KING]:
                king_moves.append(move)

        if king_moves:
            return random.choice(king_moves)

        # Otherwise, random move
        return random.choice(possible_moves)

    def _format_board_state(self, board: CheckersBoard) -> str:
        """Format the current board state for the LLM"""
        board_str = board.get_board_string()

        current_player_str = "RED" if board.current_player == Player.RED else "BLACK"
        ai_player_str = "RED" if self.config.ai_plays_as == Player.RED else "BLACK"

        possible_moves = board.get_possible_moves(board.current_player)
        moves_str = ""
        if possible_moves:
            moves_str = "\nPossible moves:\n"
            for i, move in enumerate(possible_moves[:10]):  # Limit to first 10 moves
                moves_str += f"  {move}\n"
            if len(possible_moves) > 10:
                moves_str += f"  ... and {len(possible_moves) - 10} more\n"

        return (
            f"Current board position:\n{board_str}\n"
            f"Current turn: {current_player_str} (You are {ai_player_str})\n"
            f"Move #{board.move_count + 1}\n"
            f"{moves_str}"
        )

    async def collect_trajectory(
        self, item: Item
    ) -> Tuple[Optional[ScoredDataItem], List[Item]]:
        """Collect a single game trajectory"""
        seed = item["seed"]
        random.seed(seed)

        board = CheckersBoard()
        messages: List[Message] = []

        # Determine if AI goes first
        if self.config.ai_plays_as != board.current_player:
            # Opponent goes first
            opponent_move = self._get_opponent_move(board)
            if opponent_move:
                board.apply_move(opponent_move)

        messages.append({"role": "system", "content": self.system_prompt})

        game_reward = 0.0
        move_count = 0
        valid_moves = 0
        invalid_moves = 0

        async with self.server.dedicated_server() as server:
            while not board.game_over and move_count < self.config.max_episode_turns:
                # Check if it's AI's turn
                if board.current_player != self.config.ai_plays_as:
                    # Opponent's turn
                    opponent_move = self._get_opponent_move(board)
                    if opponent_move is None:
                        break
                    board.apply_move(opponent_move)
                    continue

                # AI's turn
                board_state = self._format_board_state(board)
                messages.append({"role": "user", "content": board_state})

                # Check token limit
                if (
                    len(self.tokenizer.apply_chat_template(messages, tokenize=False))
                    > self.config.max_token_length - 200
                ):
                    logger.warning(
                        f"[Seed: {seed}] Max token length reached, ending game"
                    )
                    break

                try:
                    chat_completion = await server.chat_completion(
                        messages=messages,
                        n=1,
                        max_tokens=512,
                        temperature=self.config.temperature,
                    )

                    llm_response = chat_completion.choices[0].message.content or ""
                    messages.append({"role": "assistant", "content": llm_response})

                    # Parse move from response
                    move = self._parse_move_from_llm(llm_response)

                    if move is None:
                        invalid_moves += 1
                        game_reward -= 0.5  # Penalty for invalid move format
                        logger.warning(
                            f"[Seed: {seed}] Invalid move format: {llm_response}"
                        )
                        break

                    # Apply move
                    if board.apply_move(move):
                        valid_moves += 1
                        game_reward += 0.1  # Small reward for valid moves
                    else:
                        invalid_moves += 1
                        game_reward -= 0.5  # Penalty for invalid moves
                        logger.warning(f"[Seed: {seed}] Invalid move: {move}")
                        break

                    move_count += 1

                except Exception as e:
                    logger.error(f"[Seed: {seed}] Error during move generation: {e}")
                    break

        # Calculate final reward based on game outcome
        if board.game_over:
            if board.winner == self.config.ai_plays_as:
                game_reward += 2.0  # Win bonus
            elif board.winner is None:
                game_reward += 0.5  # Draw bonus
            else:
                game_reward -= 1.0  # Loss penalty
        else:
            game_reward -= 0.5  # Penalty for incomplete games

        # Add bonus for thinking (if enabled)
        if self.config.thinking_enabled:
            thinking_bonus = 0
            for message in messages:
                if message["role"] == "assistant" and "<think>" in message["content"]:
                    thinking_bonus += 0.1
            game_reward += min(thinking_bonus, 0.5)  # Cap thinking bonus

        # Track metrics
        self.episode_outcomes_buffer.append(game_reward)
        if board.game_over and board.winner == self.config.ai_plays_as:
            self.win_rate_buffer.append(1.0)
        else:
            self.win_rate_buffer.append(0.0)

        self.move_count_buffer.append(move_count)
        self.invalid_move_buffer.append(
            invalid_moves / max(1, valid_moves + invalid_moves)
        )

        # Prepare data for training
        full_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        tokens, masks = tokenize_for_trainer(
            full_text, self.tokenizer, self.config.max_token_length
        )

        return (
            ScoredDataItem(
                text=full_text,
                tokens=tokens,
                masks=masks,
                score=game_reward,
                extra_info={
                    "seed": seed,
                    "game_over": board.game_over,
                    "winner": board.winner.name if board.winner else "DRAW",
                    "move_count": move_count,
                    "valid_moves": valid_moves,
                    "invalid_moves": invalid_moves,
                },
            ),
            [],
        )

    async def get_next_item(self) -> Item:
        """Get the next item for training"""
        return {"seed": random.randint(0, 1000000)}

    async def setup(self):
        """Setup the environment"""
        logger.info("Checkers environment setup complete")

    async def evaluate(self, *args, **kwargs):
        """Run evaluation games"""
        logger.info("Starting checkers evaluation...")

        total_rewards = []
        win_count = 0
        draw_count = 0
        loss_count = 0
        avg_moves = []
        invalid_move_rates = []

        eval_items = [{"seed": i} for i in range(self.config.eval_episodes)]

        for item in eval_items:
            scored_item, _ = await self.collect_trajectory(item)
            if scored_item:
                total_rewards.append(scored_item.score)
                avg_moves.append(scored_item.extra_info["move_count"])

                winner = scored_item.extra_info["winner"]
                if winner == self.config.ai_plays_as.name:
                    win_count += 1
                elif winner == "DRAW":
                    draw_count += 1
                else:
                    loss_count += 1

                valid_moves = scored_item.extra_info["valid_moves"]
                invalid_moves = scored_item.extra_info["invalid_moves"]
                invalid_rate = invalid_moves / max(1, valid_moves + invalid_moves)
                invalid_move_rates.append(invalid_rate)

        # Calculate metrics
        avg_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0
        win_rate = win_count / self.config.eval_episodes
        draw_rate = draw_count / self.config.eval_episodes
        loss_rate = loss_count / self.config.eval_episodes
        avg_move_count = sum(avg_moves) / len(avg_moves) if avg_moves else 0
        avg_invalid_rate = (
            sum(invalid_move_rates) / len(invalid_move_rates)
            if invalid_move_rates
            else 0
        )

        self.eval_metrics_custom = [
            ("eval/avg_reward", avg_reward),
            ("eval/win_rate", win_rate),
            ("eval/draw_rate", draw_rate),
            ("eval/loss_rate", loss_rate),
            ("eval/avg_moves_per_game", avg_move_count),
            ("eval/avg_invalid_move_rate", avg_invalid_rate),
        ]

        logger.info(
            f"Evaluation complete: Win rate: {win_rate:.2%}, Avg reward: {avg_reward:.3f}"
        )

    async def wandb_log(self, wandb_metrics: Optional[Dict[str, float]] = None):
        """Log metrics to Weights & Biases"""
        if wandb_metrics is None:
            wandb_metrics = {}

        # Training metrics
        if self.episode_outcomes_buffer:
            wandb_metrics["train/avg_reward"] = sum(self.episode_outcomes_buffer) / len(
                self.episode_outcomes_buffer
            )

        if self.win_rate_buffer:
            wandb_metrics["train/win_rate"] = sum(self.win_rate_buffer) / len(
                self.win_rate_buffer
            )

        if self.move_count_buffer:
            wandb_metrics["train/avg_moves_per_game"] = sum(
                self.move_count_buffer
            ) / len(self.move_count_buffer)

        if self.invalid_move_buffer:
            wandb_metrics["train/invalid_move_rate"] = sum(
                self.invalid_move_buffer
            ) / len(self.invalid_move_buffer)

        # Evaluation metrics
        for metric_name, metric_value in self.eval_metrics_custom:
            wandb_metrics[metric_name] = metric_value

        self.eval_metrics_custom = []

        # Clear buffers periodically to prevent memory growth
        max_buffer_size = 1000
        if len(self.episode_outcomes_buffer) > max_buffer_size:
            self.episode_outcomes_buffer = self.episode_outcomes_buffer[
                -max_buffer_size // 2 :
            ]
        if len(self.win_rate_buffer) > max_buffer_size:
            self.win_rate_buffer = self.win_rate_buffer[-max_buffer_size // 2 :]
        if len(self.move_count_buffer) > max_buffer_size:
            self.move_count_buffer = self.move_count_buffer[-max_buffer_size // 2 :]
        if len(self.invalid_move_buffer) > max_buffer_size:
            self.invalid_move_buffer = self.invalid_move_buffer[-max_buffer_size // 2 :]

        await super().wandb_log(wandb_metrics)
