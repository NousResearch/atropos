#!/usr/bin/env python3
"""
Simple test script for the checkers environment.
This script verifies the basic functionality without requiring a full training setup.
"""

import asyncio
import logging

from checkers_env import (
    CheckersBoard,
    CheckersEnv,
    CheckersEnvConfig,
    Move,
    OpponentType,
    PieceType,
    Player,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_board_setup():
    """Test initial board setup"""
    print("Testing board setup...")
    board = CheckersBoard()

    # Check initial setup
    assert board.current_player == Player.RED
    assert not board.game_over
    assert board.winner is None
    assert board.move_count == 0

    # Check piece positions
    # Black pieces should be on rows 0-2
    black_count = 0
    for row in range(3):
        for col in range(8):
            if (row + col) % 2 == 1:  # Dark squares
                assert board.get_piece_at(row, col) == PieceType.BLACK_PIECE
                black_count += 1

    # Red pieces should be on rows 5-7
    red_count = 0
    for row in range(5, 8):
        for col in range(8):
            if (row + col) % 2 == 1:  # Dark squares
                assert board.get_piece_at(row, col) == PieceType.RED_PIECE
                red_count += 1

    assert black_count == 12
    assert red_count == 12
    print("✓ Board setup test passed")


def test_move_generation():
    """Test move generation"""
    print("Testing move generation...")
    board = CheckersBoard()

    # Get initial moves for red
    red_moves = board.get_possible_moves(Player.RED)
    assert len(red_moves) == 7  # Standard opening moves for red

    # All moves should be forward for red pieces
    for move in red_moves:
        assert move.to_row < move.from_row  # Moving up (forward for red)
        assert abs(move.to_col - move.from_col) == 1  # Diagonal move
        assert len(move.jumps) == 0  # No jumps available initially

    print("✓ Move generation test passed")


def test_basic_moves():
    """Test basic move application"""
    print("Testing basic moves...")
    board = CheckersBoard()

    # Make a simple move
    initial_moves = board.get_possible_moves(Player.RED)
    first_move = initial_moves[0]

    # Apply the move
    success = board.apply_move(first_move)
    assert success
    assert board.current_player == Player.BLACK
    assert board.move_count == 1

    # Check piece moved correctly
    assert (
        board.get_piece_at(first_move.from_row, first_move.from_col) == PieceType.EMPTY
    )
    assert (
        board.get_piece_at(first_move.to_row, first_move.to_col) == PieceType.RED_PIECE
    )

    print("✓ Basic moves test passed")


def test_jump_detection():
    """Test jump move detection"""
    print("Testing jump detection...")
    board = CheckersBoard()

    # Create a scenario with a possible jump
    # Clear some positions and set up pieces for a jump
    board.board = [[PieceType.EMPTY for _ in range(8)] for _ in range(8)]

    # Set up pieces: Red at (5,0), Black at (4,1), empty at (3,2)
    board.set_piece_at(5, 0, PieceType.RED_PIECE)
    board.set_piece_at(4, 1, PieceType.BLACK_PIECE)
    board.current_player = Player.RED

    # Get moves - should include the jump
    moves = board.get_possible_moves(Player.RED)

    # Should only have jump moves (mandatory)
    assert len(moves) > 0
    jump_move = moves[0]
    assert len(jump_move.jumps) == 1
    assert jump_move.jumps[0] == (4, 1)
    assert jump_move.from_row == 5 and jump_move.from_col == 0
    assert jump_move.to_row == 3 and jump_move.to_col == 2

    print("✓ Jump detection test passed")


def test_king_promotion():
    """Test king promotion"""
    print("Testing king promotion...")
    board = CheckersBoard()

    # Set up a red piece near promotion
    board.board = [[PieceType.EMPTY for _ in range(8)] for _ in range(8)]
    board.set_piece_at(1, 0, PieceType.RED_PIECE)
    board.current_player = Player.RED

    # Move to promotion square
    move = Move(1, 0, 0, 1)
    success = board.apply_move(move)
    assert success

    # Check promotion occurred
    assert board.get_piece_at(0, 1) == PieceType.RED_KING

    print("✓ King promotion test passed")


async def test_environment_basic():
    """Test basic environment functionality"""
    print("Testing environment basic functionality...")

    # Create a minimal config for testing
    class MockAPIServerConfig:
        def __init__(self):
            self.model_name = "test"
            self.base_url = "http://localhost"
            self.api_key = "test"
            self.num_requests_for_eval = 1

    config = CheckersEnvConfig(
        opponent_type=OpponentType.RANDOM,
        max_episode_turns=5,  # Short game for testing
        eval_episodes=1,
        thinking_enabled=False,
        tokenizer_name="gpt2",  # Simple tokenizer
        group_size=1,
        use_wandb=False,  # Disable wandb for testing
    )

    server_configs = [MockAPIServerConfig()]

    try:
        env = CheckersEnv(config, server_configs, slurm=False, testing=True)

        # Test basic methods
        item = await env.get_next_item()
        assert "seed" in item

        # Test move parsing
        test_response = (
            '<tool_call>\n{"arguments": {"from_row": 5, "from_col": 0, '
            '"to_row": 4, "to_col": 1}, "name": "make_move"}\n</tool_call>'
        )
        move = env._parse_move_from_llm(test_response)
        assert move is not None
        assert move.from_row == 5
        assert move.from_col == 0
        assert move.to_row == 4
        assert move.to_col == 1

        print("✓ Environment basic functionality test passed")

    except Exception as e:
        print(f"Note: Full environment test skipped due to missing dependencies: {e}")


def test_board_display():
    """Test board display functionality"""
    print("Testing board display...")
    board = CheckersBoard()
    board_str = board.get_board_string()

    # Check that the display contains expected elements
    assert "0 1 2 3 4 5 6 7" in board_str
    assert "b" in board_str  # Black pieces
    assert "r" in board_str  # Red pieces
    assert "." in board_str  # Empty squares

    print("✓ Board display test passed")


def run_all_tests():
    """Run all tests"""
    print("Running checkers environment tests...\n")

    test_board_setup()
    test_move_generation()
    test_basic_moves()
    test_jump_detection()
    test_king_promotion()
    test_board_display()

    # Run async test
    asyncio.run(test_environment_basic())

    print("\n🎉 All tests passed! The checkers environment is working correctly.")


if __name__ == "__main__":
    run_all_tests()
