"""
Word Hunt Environment for Atropos
Trains models to find English words on 4x4 letter grids
"""

import random
import uuid
from typing import List, Optional, Tuple

from atroposlib.envs.base import APIServerConfig, BaseEnv, ScoredDataGroup
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer


# Define a custom data group to include our reward metadata
class WordHuntScoredDataGroup(ScoredDataGroup):
    pass


# Handle imports for both direct execution and module import
try:
    from .word_hunt_config import WordHuntEnvConfig
    from .word_hunt_solver import WordHuntSolver
except ImportError:
    from word_hunt_config import WordHuntEnvConfig
    from word_hunt_solver import WordHuntSolver


class WordHuntEnv(BaseEnv):
    """Word Hunt Environment for training models to find words on 4x4 grids"""

    name = "word_hunt_environment"

    @classmethod
    def config_init(cls) -> Tuple[WordHuntEnvConfig, List[APIServerConfig]]:
        """Initializes the default configuration for the environment."""
        env_config = WordHuntEnvConfig()
        server_configs = [APIServerConfig()]
        return env_config, server_configs

    async def setup(self) -> None:
        """
        Initialize environment, load solver, set up state

        This method:
        1. Initializes the WordHuntSolver with dictionary
        2. Sets up board generation parameters
        3. Initializes training statistics
        4. Prepares prompt templates
        """
        # 1. Initialize WordHuntSolver with dictionary
        self.solver = WordHuntSolver(self.config.dictionary_path)
        print(
            f"✅ WordHuntSolver initialized with dictionary: {self.config.dictionary_path}"
        )

        # Initialize tokenizer for scoring
        try:
            from transformers import AutoTokenizer

            print(
                f"🔍 Debug: Initializing tokenizer with name: {self.config.tokenizer_name}"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Set chat template if not present (GPT-2 doesn't have one by default)
            if self.tokenizer.chat_template is None:
                self.tokenizer.chat_template = (
                    "{% for message in messages %}"
                    "{{ message['role'] + ': ' + message['content'] + '\\n' }}"
                    "{% endfor %}"
                )
                print("✅ Set default chat template for GPT-2 tokenizer")

            print("✅ Tokenizer initialized")
        except Exception as e:
            print(f"❌ Failed to initialize tokenizer: {e}")
            self.tokenizer = None

        # 2. Set up board generation parameters
        self.letter_frequencies = self.config.get_letter_frequencies()
        self.scoring_system = self.config.get_scoring_system()
        self.prompt_template = self.config.get_prompt_template()

        # 3. Initialize training statistics
        self.total_games = 0
        self.total_score = 0
        self.total_valid_words = 0
        self.total_invalid_words = 0
        self.current_board_index = 0
        self.boards_this_epoch = []
        self.current_item = None  # Store current item for scoring

        # 4. Set up random state for reproducible board generation
        random.seed(42)  # Fixed seed for reproducibility

        # 5. Generate initial batch of boards for this epoch
        await self._generate_epoch_boards()

        print("✅ Word Hunt Environment setup complete:")
        print(f"  - Board size: {self.config.board_size}x{self.config.board_size}")
        print(f"  - Boards per epoch: {self.config.boards_per_epoch}")
        print(f"  - Max tokens per game: {self.config.max_tokens_per_game}")
        print(f"  - Prompt style: {self.config.prompt_style}")
        if self.config.use_official_scoring:
            print("  - Scoring system: Official (3-5 letter scores + formula for 6+)")
        else:
            print(
                f"  - Scoring system: {len(self.scoring_system)} word lengths supported"
            )

    async def _generate_epoch_boards(self) -> None:
        """Generate boards for the current epoch"""
        self.boards_this_epoch = []

        for i in range(self.config.boards_per_epoch):
            board = self.solver.generate_random_board(self.letter_frequencies)
            board_id = f"board_{self.total_games + i}_{uuid.uuid4().hex[:8]}"

            self.boards_this_epoch.append(
                {
                    "board": board,
                    "board_id": board_id,
                    "max_tokens": self.config.max_tokens_per_game,
                }
            )

        if self.config.shuffle_boards:
            random.shuffle(self.boards_this_epoch)

        print(f"📋 Generated {len(self.boards_this_epoch)} boards for epoch")

    async def get_next_item(self) -> Optional[Tuple]:
        """Get the next board for the model to solve (following Atropos standard format).

        Returns:
            Tuple of (prompt_messages, board_data) following Atropos format, or None if epoch is complete
        """
        if self.current_board_index >= len(self.boards_this_epoch):
            # Epoch complete - generate new boards for next epoch
            await self._generate_epoch_boards()
            self.current_board_index = 0

        if self.current_board_index >= len(self.boards_this_epoch):
            return None  # No more boards available

        board_data = self.boards_this_epoch[self.current_board_index]
        self.current_board_index += 1

        # Format the board into a prompt
        board = board_data["board"]
        prompt_text = self._format_board_prompt(board)

        # Create prompt messages in Atropos standard format (frozenset tuples)
        prompt_messages = [frozenset({"role": "user", "content": prompt_text}.items())]

        # Return tuple following Atropos standard: (prompt_messages, board_data)
        return (tuple(prompt_messages), board_data)

    def _format_board_prompt(self, board: List[List[str]]) -> str:
        """Format the board into a prompt for the model."""
        prompt_parts = []

        # Add instructions if enabled
        if self.config.include_instructions:
            prompt_parts.append(
                "Find English words on this 4x4 letter grid to maximize your score. "
                "Longer words are worth more points. You must adhere to the following rules:"
            )
            prompt_parts.append(
                "Words must be AT LEAST 3 letters long and have to be formed by "
                "connecting adjacent letters on the board (including diagonally)."
            )
            prompt_parts.append(
                "The board does not wrap around; letters on opposite edges are not "
                "considered adjacent."
            )
            prompt_parts.append("The whole word must have an adjacent path through it")
            prompt_parts.append("Each letter can only be used once per word.")
            prompt_parts.append("The word must be a valid word in the English language")
            prompt_parts.append(
                "Making the same word in multiple ways does not count for extra points - "
                "each unique word only counts once."
            )
            prompt_parts.append(
                "Provide your answer as a comma-separated list, like this: "
                "WORD, ANOTHER, EXAMPLE"
            )
            prompt_parts.append("")

        # Add scoring info if enabled
        if self.config.include_scoring_info:
            scoring_info = self.config.get_scoring_info()
            prompt_parts.append(f"Scoring: {scoring_info}")
            prompt_parts.append("")

        # Add the board based on prompt style
        if self.config.prompt_style == "grid_visual":
            prompt_parts.append("Board:")
            for row in board:
                prompt_parts.append(" ".join(row))
        elif self.config.prompt_style == "text_description":
            letters = []
            for row in board:
                letters.extend(row)
            prompt_parts.append(f"Letters: {' '.join(letters)}")
        elif self.config.prompt_style == "both":
            prompt_parts.append("Board:")
            for row in board:
                prompt_parts.append(" ".join(row))
            letters = []
            for row in board:
                letters.extend(row)
            prompt_parts.append(f"Letters: {' '.join(letters)}")

        prompt_parts.append("")
        prompt_parts.append("Found words:")

        return "\n".join(prompt_parts)

    async def collect_trajectories(
        self, item
    ) -> Tuple[Optional[WordHuntScoredDataGroup], List]:
        """Collect trajectories with robust error handling and validation.

        Args:
            item: Tuple of (prompt_messages, board_data) following Atropos format

        Returns:
            Tuple of (scored_data, backlog):
            - scored_data: ScoredDataGroup with tokens, masks, and scores, or None if failed
            - backlog: Empty list (no follow-up items)
        """
        # Validate input structure
        if not isinstance(item, tuple) or len(item) != 2:
            print(
                f"❌ Invalid item format: expected tuple of (messages, data), got {type(item)}"
            )
            return None, []

        prompt_messages, board_data = item

        if not isinstance(prompt_messages, tuple) or not prompt_messages:
            print(
                f"❌ Invalid prompt_messages: expected non-empty tuple, got {type(prompt_messages)}"
            )
            return None, []

        if not isinstance(board_data, dict) or "board" not in board_data:
            print(
                f"❌ Invalid board_data: expected dict with 'board' key, got {type(board_data)}"
            )
            return None, []

        # Extract and validate messages
        try:
            messages = []
            for role_dict in prompt_messages:
                if not isinstance(role_dict, frozenset):
                    print(
                        f"❌ Invalid message format: expected frozenset, got {type(role_dict)}"
                    )
                    return None, []
                messages.append(dict(role_dict))

            if not messages:
                print("❌ No valid messages found")
                return None, []

        except Exception as e:
            print(f"❌ Failed to extract messages: {e}")
            return None, []

        # Store current item for scoring
        self.current_item = board_data

        # Debug: Print the messages structure
        print("🔍 Debug: Messages structure before chat template:")
        for i, msg in enumerate(messages):
            print(f"  Message {i}: {msg}")
        print()

        # Apply chat template with error handling
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )

            if not prompt or len(prompt.strip()) == 0:
                print("❌ Generated prompt is empty")
                return None, []

            # Debug: Print the actual prompt being sent to the model
            print("🔍 Debug: Actual prompt being sent to model:")
            print("=" * 50)
            print(prompt)
            print("=" * 50)

        except Exception as e:
            print(f"❌ Chat template application failed: {e}")
            return None, []

        # Get completions with timeout and validation
        try:
            print("🔍 Debug: About to call server.completion()")
            print(f"🔍 Debug: Prompt length: {len(prompt)} chars")
            print(f"🔍 Debug: Group size: {self.config.group_size}")

            # Rely on the server's built-in timeout and retry logic
            completions = await self.server.completion(
                prompt=prompt,
                n=self.config.group_size,
                max_tokens=self.config.max_tokens_per_game,
                temperature=0.8,
            )

            print("🔍 Debug: API call completed successfully")

            if not completions or not completions.choices:
                print("❌ No completions received from API")
                return None, []

            print(f"🔍 Debug: Got {len(completions.choices)} completions")

            # Debug: Print each completion response
            print("🔍 Debug: Model responses:")
            print("-" * 50)
            for i, choice in enumerate(completions.choices):
                print(f"  Response {i+1}:")
                print(f"  {choice.text.strip()}")
            print("-" * 50)

        except Exception as e:
            print(f"❌ Model call failed: {e}")
            import traceback

            traceback.print_exc()
            return None, []

        # Build trajectories efficiently
        try:
            # Pre-build base messages once
            base_messages = [dict(role_dict) for role_dict in prompt_messages]
            to_score = []

            for completion_choice in completions.choices:
                # Validate completion
                if not completion_choice or not completion_choice.text:
                    print("⚠️  Skipping invalid completion choice")
                    continue

                # Create trajectory efficiently
                trajectory_messages = base_messages + [
                    {"role": "assistant", "content": completion_choice.text.strip()}
                ]

                to_score.append((tuple(trajectory_messages), board_data))

            if not to_score:
                print("❌ No valid trajectories created")
                return None, []

        except Exception as e:
            print(f"❌ Failed to build trajectories: {e}")
            return None, []

        # Score trajectories with error handling
        try:
            scored_data = await self.score(to_score)

            if scored_data is None:
                print("❌ Scoring returned None")
                return None, []

            # Validate scored data structure
            required_keys = ["tokens", "masks", "scores"]
            if not all(key in scored_data for key in required_keys):
                print(f"❌ Scored data missing required keys: {required_keys}")
                return None, []

            if not scored_data["tokens"] or not scored_data["scores"]:
                print("❌ Scored data is empty")
                return None, []

            print(f"✅ Successfully scored {len(scored_data['scores'])} trajectories")
            return scored_data, []

        except Exception as e:
            print(f"❌ Scoring failed: {e}")
            import traceback

            traceback.print_exc()
            return None, []

    async def score(
        self, rollout_group_data: List
    ) -> Optional[WordHuntScoredDataGroup]:
        """Score the collected trajectories (following Atropos standard pattern).

        Args:
            rollout_group_data: List of tuples (trajectory_messages, board_data) from collect_trajectories

        Returns:
            ScoredDataGroup with tokens, masks, and scores
        """
        if not rollout_group_data or not self.current_item:
            return None

        board = self.current_item["board"]

        # Initialize our custom data group with empty lists for each key.
        scores = WordHuntScoredDataGroup()
        scores["tokens"] = []
        scores["masks"] = []
        scores["scores"] = []

        for trajectory_tuple in rollout_group_data:
            trajectory_messages, board_data = trajectory_tuple

            if not trajectory_messages or not isinstance(trajectory_messages, tuple):
                continue

            # Convert frozenset tuples back to dict format for processing
            trajectory_dicts = []
            for msg_frozenset in trajectory_messages:
                trajectory_dicts.append(dict(msg_frozenset))

            # Extract assistant response
            assistant_messages = [
                msg
                for msg in trajectory_dicts
                if isinstance(msg, dict) and msg.get("role") == "assistant"
            ]

            if not assistant_messages:
                continue

            response = assistant_messages[-1]["content"]

            # Score the response using our solver
            normalized_score, metadata = self.solver.score_word_hunt_response(
                response, board, self.scoring_system
            )

            # Update training statistics
            self.total_games += 1
            self.total_score += metadata["total_score"]
            self.total_valid_words += metadata["num_valid_words"]
            self.total_invalid_words += metadata["num_invalid_words"]

            # Tokenize the response (following Atropos standard)
            tokenized = tokenize_for_trainer(self.tokenizer, trajectory_dicts)
            tokens = tokenized["tokens"]
            masks = tokenized["masks"]

            scores["tokens"].append(tokens)
            scores["masks"].append(masks)
            scores["scores"].append(normalized_score)

        return scores if scores["tokens"] else None

    async def evaluate(self) -> None:
        """Run evaluation and log metrics.

        This method:
        1. Calculate and log training metrics
        2. Print summary statistics
        3. Reset statistics for next epoch
        """
        if self.total_games == 0:
            print("⚠️  No games played yet - skipping evaluation")
            return

        # Calculate metrics
        avg_score = self.total_score / self.total_games
        avg_valid_words = self.total_valid_words / self.total_games
        avg_invalid_words = self.total_invalid_words / self.total_games
        total_words = self.total_valid_words + self.total_invalid_words
        accuracy = self.total_valid_words / total_words if total_words > 0 else 0.0

        # Log to wandb
        metrics = {
            "eval/total_games": self.total_games,
            "eval/avg_score": avg_score,
            "eval/avg_valid_words": avg_valid_words,
            "eval/avg_invalid_words": avg_invalid_words,
            "eval/word_accuracy": accuracy,
            "eval/total_score": self.total_score,
            "eval/total_valid_words": self.total_valid_words,
            "eval/total_invalid_words": self.total_invalid_words,
        }

        await self.wandb_log(metrics)

        # Print summary
        print("\n📊 Word Hunt Evaluation Summary:")
        print(f"  Games played: {self.total_games}")
        print(f"  Average score: {avg_score:.1f}")
        print(f"  Average valid words: {avg_valid_words:.1f}")
        print(f"  Average invalid words: {avg_invalid_words:.1f}")
        print(f"  Word accuracy: {accuracy:.1%}")
        print(f"  Total score: {self.total_score}")
        print(f"  Total valid words: {self.total_valid_words}")
        print(f"  Total invalid words: {self.total_invalid_words}")

        # Reset statistics for next epoch
        self.total_games = 0
        self.total_score = 0
        self.total_valid_words = 0
        self.total_invalid_words = 0
        self.current_board_index = 0


if __name__ == "__main__":
    WordHuntEnv.cli()
