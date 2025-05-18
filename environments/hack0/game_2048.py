import random
import numpy as np
from typing import Tuple, List, Optional, Dict, Any, Literal
import uuid

class Game2048:
    """
    A class that implements the 2048 game logic.
    """
    
    def __init__(self, size: int = 4):
        """
        Initialize a new 2048 game.
        
        Args:
            size: The size of the board (size x size grid)
        """
        self.id = str(uuid.uuid4())[:8]
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int32)
        self.score = 0
        self.moves = 0
        self.game_over = False
        
        # Add two initial tiles
        self._add_random_tile()
        self._add_random_tile()
    
    def reset(self) -> np.ndarray:
        """
        Reset the game to initial state.
        
        Returns:
            The game board after reset
        """
        self.board = np.zeros((self.size, self.size), dtype=np.int32)
        self.score = 0
        self.moves = 0
        self.game_over = False
        
        # Add two initial tiles
        self._add_random_tile()
        self._add_random_tile()
        
        return self.board.copy()
    
    def _add_random_tile(self) -> None:
        """
        Add a random tile (2 or 4) to the board.
        90% chance of adding a 2, 10% chance of adding a 4.
        """
        if self.game_over:
            return
            
        # Find all empty cells
        empty_cells = list(zip(*np.where(self.board == 0)))
        
        if empty_cells:
            # Choose a random empty cell
            row, col = random.choice(empty_cells)
            
            # Add a 2 (90% chance) or 4 (10% chance)
            self.board[row, col] = 2 if random.random() < 0.9 else 4
    
    def _move_left(self) -> bool:
        """
        Move all tiles to the left and merge if possible.
        
        Returns:
            True if the board changed, False otherwise
        """
        initial_board = self.board.copy()
        score_added = 0
        
        for row in range(self.size):
            # Merge tiles in this row
            merged_row = self._merge_sequence(self.board[row])
            self.board[row] = merged_row[0]
            score_added += merged_row[1]
            
        # Check if the board changed
        changed = not np.array_equal(initial_board, self.board)
        
        if changed:
            self.score += score_added
            
        return changed
    
    def _merge_sequence(self, sequence: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Merge a sequence of numbers according to 2048 rules.
        
        Args:
            sequence: A row or column from the board
            
        Returns:
            Tuple of (merged sequence, score added)
        """
        # Remove zeros
        sequence = sequence[sequence != 0]
        
        if len(sequence) <= 1:
            # Pad with zeros
            result = np.zeros(self.size, dtype=np.int32)
            result[:len(sequence)] = sequence
            return result, 0
        
        # Merge tiles
        merged = []
        score = 0
        i = 0
        
        while i < len(sequence):
            if i + 1 < len(sequence) and sequence[i] == sequence[i + 1]:
                # Merge equal tiles
                value = sequence[i] * 2
                merged.append(value)
                score += value
                i += 2
            else:
                # Keep single tile
                merged.append(sequence[i])
                i += 1
        
        # Pad with zeros
        result = np.zeros(self.size, dtype=np.int32)
        result[:len(merged)] = merged
        
        return result, score
    
    def move(self, direction: Literal['left', 'right', 'up', 'down']) -> Tuple[np.ndarray, int, bool]:
        """
        Make a move in the specified direction.
        
        Args:
            direction: One of 'left', 'right', 'up', 'down'
            
        Returns:
            Tuple of (board, score_added, changed)
        """
        if self.game_over:
            return self.board.copy(), 0, False
        
        initial_score = self.score
        changed = False
        
        if direction == 'left':
            changed = self._move_left()
        elif direction == 'right':
            # Reverse the board, move left, then reverse back
            self.board = np.fliplr(self.board)
            changed = self._move_left()
            self.board = np.fliplr(self.board)
        elif direction == 'up':
            # Transpose the board, move left, then transpose back
            self.board = self.board.T
            changed = self._move_left()
            self.board = self.board.T
        elif direction == 'down':
            # Transpose and reverse the board, move left, then transpose and reverse back
            self.board = np.fliplr(self.board.T)
            changed = self._move_left()
            self.board = np.fliplr(self.board).T
        
        score_added = self.score - initial_score
        
        if changed:
            self.moves += 1
            self._add_random_tile()
            
            # Check if game is over after this move
            self.game_over = not self._has_valid_moves()
        
        return self.board.copy(), score_added, changed
    
    def _has_valid_moves(self) -> bool:
        """
        Check if there are any valid moves left.
        
        Returns:
            True if there are valid moves left, False otherwise
        """
        # If there are empty cells, there are valid moves
        if np.any(self.board == 0):
            return True
        
        # Check for adjacent identical cells (horizontal)
        for row in range(self.size):
            for col in range(self.size - 1):
                if self.board[row, col] == self.board[row, col + 1]:
                    return True
        
        # Check for adjacent identical cells (vertical)
        for row in range(self.size - 1):
            for col in range(self.size):
                if self.board[row, col] == self.board[row + 1, col]:
                    return True
        
        # No valid moves
        return False
    
    def get_max_tile(self) -> int:
        """
        Get the value of the highest tile on the board.
        
        Returns:
            The maximum tile value
        """
        return np.max(self.board)
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the game.
        
        Returns:
            A dictionary containing the current state
        """
        return {
            'id': self.id,
            'board': self.board.copy(),
            'score': self.score,
            'moves': self.moves,
            'game_over': self.game_over,
            'max_tile': self.get_max_tile()
        }
    
    def render(self) -> str:
        """
        Render the game board as a string.
        
        Returns:
            A string representation of the board
        """
        # Convert zeros to None for cleaner display
        display_board = self.board.copy().astype(object)
        display_board[display_board == 0] = None
        
        # Find the width needed for the largest number
        cell_width = max(len(str(num)) for num in np.unique(self.board[self.board > 0])) if np.any(self.board > 0) else 1
        cell_width = max(cell_width, 1)  # Ensure at least 1 character width
        
        # Build the board string
        board_str = ""
        
        for row in display_board:
            row_str = "|"
            for cell in row:
                if cell is None:
                    row_str += "_".ljust(cell_width + 2) + "|"
                else:
                    row_str += f" {str(cell).ljust(cell_width)} |"
            board_str += row_str + "\n"
        
        return board_str 