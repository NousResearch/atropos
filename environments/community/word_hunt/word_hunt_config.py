"""
Custom configuration class for Word Hunt Environment
Defines all configurable parameters for board generation, scoring, and training
"""

from typing import Dict, List

from pydantic import Field

from atroposlib.envs.base import BaseEnvConfig


class WordHuntEnvConfig(BaseEnvConfig):
    """Configuration for Word Hunt Environment"""

    # Board Generation Parameters
    board_size: int = Field(default=4, description="Size of the word hunt board (4x4)")
    min_word_length: int = Field(default=3, description="Minimum word length to count")
    max_word_length: int = Field(default=16, description="Maximum word length possible")

    # Letter Distribution Parameters
    vowel_weight: float = Field(
        default=0.4, description="Probability of generating vowels vs consonants"
    )
    common_letter_bias: bool = Field(
        default=True, description="Bias towards more common English letters"
    )

    # Token and Response Parameters
    max_tokens_per_game: int = Field(
        default=100, description="Maximum tokens model can use per game"
    )
    use_official_scoring: bool = Field(
        default=True, description="Use official Word Hunt scoring rules"
    )
    normalize_scores: bool = Field(
        default=True, description="Normalize scores between 0 and 1"
    )

    # Dictionary and solver settings
    dictionary_path: str = Field(
        default="environments/community/word_hunt/Dictionary.txt",
        description="Path to the dictionary file for word validation",
    )
    validate_words: bool = Field(
        default=True, description="Validate words are in dictionary"
    )
    validate_board_paths: bool = Field(
        default=True, description="Validate words can be formed on board"
    )

    # Prompt settings
    prompt_style: str = Field(
        default="grid_visual",
        description="How to present board: 'grid_visual', 'text_description', 'both'",
    )
    include_instructions: bool = Field(
        default=True, description="Include game instructions in prompt"
    )
    include_scoring_info: bool = Field(
        default=True, description="Include scoring information in prompt"
    )

    # Tokenizer settings
    tokenizer_name: str = Field(
        default="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
        description="Name of the tokenizer to use for model generation",
    )

    # Training Parameters
    boards_per_epoch: int = Field(
        default=100, description="Number of boards to generate per training epoch"
    )
    shuffle_boards: bool = Field(
        default=True, description="Shuffle board order each epoch"
    )

    # Evaluation Parameters
    eval_board_count: int = Field(
        default=10, description="Number of boards for evaluation"
    )
    eval_metrics: List[str] = Field(
        default=["accuracy", "total_score", "word_count", "avg_word_length"],
        description="Metrics to track during evaluation",
    )

    # Advanced Parameters
    debug_mode: bool = Field(default=False, description="Enable debug logging")
    save_board_images: bool = Field(
        default=False, description="Save board visualizations for debugging"
    )

    # Override some BaseEnvConfig defaults for Word Hunt
    group_size: int = Field(
        default=16, description="Number of responses to generate for each board."
    )
    max_token_length: int = Field(
        default=1024 * 16,
        description="Max tokens for model generation (matching working environments)",
    )
    steps_per_eval: int = Field(
        default=25,
        description="Steps between evaluations (matching working environments)",
    )
    inference_weight: float = Field(
        default=1.0, description="Inference weight for training"
    )
    min_batch_allocation: float = Field(
        default=0.1,
        description="Minimum batch allocation (matching working environments)",
    )

    def get_scoring_system(self) -> Dict[int, int]:
        """Get the scoring system for word lengths"""
        if self.use_official_scoring:
            return {
                3: 100,  # 3-letter words
                4: 400,  # 4-letter words
                5: 800,  # 5-letter words
                # 6+ letter words: 1400 + (400 * (length - 6))
            }
        else:
            # Custom scoring: exponential growth
            return {length: 2 ** (length - 2) for length in range(3, 17)}

    def get_letter_frequencies(self) -> Dict[str, float]:
        """Get letter frequency distribution for board generation"""
        if self.common_letter_bias:
            return {
                "E": 12.0,
                "A": 8.2,
                "R": 6.7,
                "I": 6.3,
                "O": 6.1,
                "T": 5.9,
                "N": 5.7,
                "S": 5.3,
                "L": 4.0,
                "C": 3.8,
                "U": 3.0,
                "D": 2.8,
                "P": 2.7,
                "M": 2.4,
                "H": 2.3,
                "G": 2.0,
                "B": 1.5,
                "F": 1.4,
                "Y": 1.4,
                "W": 1.3,
                "K": 0.8,
                "V": 0.6,
                "X": 0.2,
                "Z": 0.1,
                "J": 0.1,
                "Q": 0.1,
            }
        else:
            # Uniform distribution
            return {chr(i): 1.0 for i in range(65, 91)}  # A-Z

    def get_prompt_template(self) -> str:
        """Get the prompt template for presenting boards to the model"""
        if self.prompt_style == "grid_visual":
            template = """You are playing Word Hunt! Find as many English words as possible on this 4x4 letter grid.

Rules:
- Words must be at least 3 letters long
- You can move in any direction (including diagonally)
- You cannot reuse the same letter in a single word
- Only real English words count

{scoring_info}

Here's your board:
{board_grid}

Find all the words you can! Return them as a space-separated list.
"""
        elif self.prompt_style == "text_description":
            template = """You are playing Word Hunt! Find English words from these letters arranged in a 4x4 grid.

Rules:
- Words must be at least 3 letters long
- You can move in any direction (including diagonally)
- You cannot reuse the same letter in a single word
- Only real English words count

{scoring_info}

Letters (reading left to right, top to bottom): {board_letters}

Find all the words you can! Return them as a space-separated list.
"""
        else:  # both
            template = """You are playing Word Hunt! Find as many English words as possible on this 4x4 letter grid.

Rules:
- Words must be at least 3 letters long
- You can move in any direction (including diagonally)
- You cannot reuse the same letter in a single word
- Only real English words count

{scoring_info}

Here's your board:
{board_grid}

Letters (reading left to right, top to bottom): {board_letters}

Find all the words you can! Return them as a space-separated list.
"""

        return template.strip()

    def get_scoring_info(self) -> str:
        """Get scoring information for the prompt"""
        if not self.include_scoring_info:
            return ""

        scoring = self.get_scoring_system()
        info = "Scoring: "
        info += ", ".join(
            [
                f"{length}-letter: {score}pts"
                for length, score in sorted(scoring.items())[:5]
            ]
        )  # Show first 5
        if len(scoring) > 5:
            info += f", 6+ letters: {scoring[6]}+ pts"

        return info


# For testing the config
if __name__ == "__main__":
    config = WordHuntEnvConfig()

    print("Word Hunt Environment Configuration:")
    print(f"Board size: {config.board_size}x{config.board_size}")
    print(f"Min word length: {config.min_word_length}")
    print(f"Max tokens per game: {config.max_tokens_per_game}")
    print(f"Prompt style: {config.prompt_style}")

    print("\nScoring system:")
    scoring = config.get_scoring_system()
    for length, score in sorted(scoring.items())[:8]:
        print(f"  {length}-letter words: {score} points")

    print("\nLetter frequencies (top 10):")
    frequencies = config.get_letter_frequencies()
    sorted_freq = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)[:10]
    for letter, freq in sorted_freq:
        print(f"  {letter}: {freq}")

    print("\nPrompt template:")
    print(config.get_prompt_template())
