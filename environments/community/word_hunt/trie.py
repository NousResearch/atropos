"""
Trie Data Structure for Word Hunt
"""


class TrieNode:
    """A node in the Trie structure."""

    def __init__(self):
        self.children = {}  # A dictionary of child nodes: {char: TrieNode}
        self.is_end_of_word = False


class Trie:
    """
    Trie data structure for efficient word and prefix lookups, optimized for the Word Hunt game.
    """

    def __init__(self):
        """Initializes the Trie with an empty root node."""
        self.root = TrieNode()

    def insert(self, word: str):
        """
        Inserts a word into the Trie. Assumes word is already uppercase.
        """
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def is_word(self, word: str) -> bool:
        """
        Searches for a complete word in the Trie. Assumes word is already uppercase.
        """
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

    def is_prefix(self, prefix: str) -> bool:
        """
        Checks if a string is a prefix of any word in the Trie. Assumes prefix is already uppercase.
        """
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
