"""
Word Hunt Solver for Atropos
"""

import random
import re
from typing import Dict, List, Optional, Set, Tuple

try:
    from .trie import Trie
except ImportError:
    from trie import Trie


class WordHuntSolver:
    """
    Solves a 4x4 Word Hunt game by finding all valid words on a given board.

    This solver uses a Trie data structure for efficient dictionary lookups and a
    recursive backtracking algorithm (Depth-First Search) to find words.
    """

    def __init__(self, dictionary_path: Optional[str] = None):
        """
        Initializes the solver, loading the dictionary into a Trie.

        Args:
            dictionary_path: The path to the dictionary file.
        """
        self.trie = self._load_dictionary(dictionary_path)

    def _load_dictionary(self, dictionary_path: Optional[str]) -> Trie:
        """Loads words from a file into the Trie, filtering by length."""
        trie = Trie()
        if not dictionary_path:
            print("⚠️ No dictionary path provided.")
            return trie
        try:
            with open(dictionary_path, "r") as f:
                for word in f:
                    clean_word = word.strip().upper()
                    if len(clean_word) >= 3:
                        trie.insert(clean_word)
            print(f"✅ Dictionary loaded from {dictionary_path}")
        except FileNotFoundError:
            print(f"❌ Dictionary file not found at {dictionary_path}.")
        return trie

    def generate_random_board(
        self, letter_frequencies: Dict[str, float], board_size: int = 4
    ) -> List[List[str]]:
        """
        Generates a random 4x4 board based on letter frequencies.

        Args:
            letter_frequencies: A dictionary mapping letters to their frequencies.
            board_size: The dimension of the square board (default is 4).

        Returns:
            A 4x4 list of lists representing the board.
        """
        letters = list(letter_frequencies.keys())
        weights = list(letter_frequencies.values())
        return [
            random.choices(letters, weights=weights, k=board_size)
            for _ in range(board_size)
        ]

    def solve_board(self, board: List[List[str]]) -> Set[str]:
        """Finds all valid words on the board using a Trie-based DFS."""
        found_words = set()
        board_size = len(board)
        for r in range(board_size):
            for c in range(board_size):
                self._solve_dfs(board, self.trie.root, r, c, "", set(), found_words)
        return found_words

    def _solve_dfs(self, board, node, r, c, path_str, visited, found_words):
        board_size = len(board)
        if not (0 <= r < board_size and 0 <= c < board_size) or (r, c) in visited:
            return

        char = board[r][c]
        if char not in node.children:
            return

        # Move to the next node in the trie
        node = node.children[char]

        # Update path and visited set
        path_str += char
        visited.add((r, c))

        # Check if the current path forms a valid word
        if node.is_end_of_word and len(path_str) >= 3:
            found_words.add(path_str)

        # Recurse on all 8 neighbors
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                self._solve_dfs(
                    board, node, r + dr, c + dc, path_str, visited, found_words
                )

        # Backtrack: remove the current cell from the visited set for other paths
        visited.remove((r, c))

    def score_word_hunt_response(
        self, response: str, board: List[List[str]], scoring_system: Dict[int, int]
    ) -> Tuple[float, Dict]:
        """
        Scores a model's response by finding all valid words on the board and checking
        the response against them. This is a more robust method than checking each
        word individually.
        """
        all_possible_words = self.solve_board(board)
        # Use regex to find all alphabetic words, making parsing more robust.
        submitted_words = {word.upper() for word in re.findall(r"[a-zA-Z]+", response)}

        valid_words = submitted_words.intersection(all_possible_words)
        invalid_words = submitted_words.difference(all_possible_words)

        total_score = 0
        for word in valid_words:
            word_len = len(word)
            if word_len in scoring_system:
                total_score += scoring_system[word_len]
            elif word_len >= 6:  # Official scoring for 6+ letter words
                total_score += 1400 + (400 * (word_len - 6))

        max_possible_score = 0
        for word in all_possible_words:
            word_len = len(word)
            if word_len in scoring_system:
                max_possible_score += scoring_system[word_len]
            elif word_len >= 6:  # Official scoring for 6+ letter words
                max_possible_score += 1400 + (400 * (word_len - 6))

        normalized_score = (
            (total_score / max_possible_score) if max_possible_score > 0 else 0.0
        )

        metadata = {
            "total_score": total_score,
            "valid_words": sorted(list(valid_words)),
            "invalid_words": sorted(list(invalid_words)),
            "num_valid_words": len(valid_words),
            "num_invalid_words": len(invalid_words),
        }

        return normalized_score, metadata
