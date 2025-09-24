import random
import re
from typing import Dict, List, Optional, Tuple, Union

import chess
import chess.engine
import wandb
from datasets import load_dataset
from pydantic import Field
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    Item,
    ScoredDataGroup,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer


# === Config class ===
class ChessEnvConfig(BaseEnvConfig):
    # Whether to wrap reasoning in a "thinking" tag
    thinking_mode: bool = Field(
        default=True,
        description="If True, include the reasoning section wrapped in the chosen thinking tag.",
    )

    # Custom tag or marker for the reasoning section
    thinking_tag: Optional[str] = Field(
        default="think",
        description="Tag used to wrap the reasoning section. If None, omit reasoning entirely.",
    )

    # Optional prefix prompt injected before the main system prompt
    thinking_system_prompt: Optional[str] = Field(
        default=None,
        description="Custom system prompt to prepend for controlling thinking mode (/think, /no_think, etc.).",
    )


# === Helper to build system prompt dynamically ===
def build_system_prompt(config: ChessEnvConfig) -> str:
    """
    Build the system prompt dynamically for the chess rules assistant
    based on config values.
    """
    # dynamic tag for reasoning section
    tag = config.thinking_tag or "think"

    # prepend any extra system text if provided
    prefix = config.thinking_system_prompt or ""

    prompt = (
        prefix
        + "You are a chess rules assistant. You will be given a chess position in FEN format, "
        "and your task is to output all possible legal moves for the player whose turn it is. "
        "ALWAYS output exactly this format and nothing else:\n\n"
        "<moves>comma-separated UCI moves</moves>\n"
        f"<{tag}>explain your reasoning here (this may include internal chain-of-thought)</{tag}>[STOP]\n\n"
        "Rules:\n"
        "1) Only output legal moves for the current player; do NOT suggest illegal moves.\n"
        "2) Do NOT use <tool_call>, <function_call>, JSON, or any other tags — only <moves> and "
        f"<{tag}>.\n"
        "3) Close both tags before emitting [STOP].\n"
        "4) Use chess keywords (check, fork, skewer, promotion, castling) where applicable inside "
        f"<{tag}>.\n\n"
        "Example:\n"
        "<moves>e2e4,d2d3,g1f3</moves>\n"
        f"<{tag}>e2e4 opens the center; d2d3 supports the pawn structure; g1f3 develops the knight</{tag}>[STOP]\n"
    )
    return prompt


class ChessRulesEnv(BaseEnv):
    def __init__(
        self,
        config: BaseEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=False,
        testing=False,
    ):
        """
        Initialize the Chess Puzzle Solver environment.

        Args:
            config: Configuration for the base environment
            server_configs: List of server configurations for OpenAI API
            slurm: Whether to use Slurm for distributed training
            testing: Whether in testing mode
        """
        super().__init__(config, server_configs, slurm, testing)
        self.percent_correct_buffer = list()
        self.eval_metrics = list()

        self.system_prompt = build_system_prompt(config)

    @classmethod
    def config_init(cls) -> Tuple[ChessEnvConfig, List[APIServerConfig]]:
        env_config = ChessEnvConfig(
            tokenizer_name="Qwen/Qwen3-4B-Instruct-2507",
            group_size=4,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=12,
            steps_per_eval=100,
            max_token_length=2048,
            wandb_name="chess_puzzle_solver",
            thinking_mode=True,
            thinking_tag="analysis",  # default tag less likely to conflict
            thinking_system_prompt=None,
        )
        server_configs = [
            APIServerConfig(
                model_name="Qwen/Qwen3-4B-Instruct-2507",
                base_url="http://localhost:9000/v1",
                api_key="x",
                num_requests_for_eval=256,
            ),
        ]
        return env_config, server_configs

    async def setup(self):
        """
        Set up the environment by loading and preparing the dataset.
        """
        # Load the full dataset
        full_dataset = load_dataset(
            "codingmonster1234/chess_puzzles_dataset", split="train"
        )

        full_dataset = full_dataset.shuffle(seed=42)

        # Create train/test split on the fly (e.g., 95% train, 5% test)
        split_dataset = full_dataset.train_test_split(test_size=0.02, seed=42)

        # Keep the splits as is - no need to reformat
        self.train = split_dataset["train"]
        self.test = split_dataset["test"]

        # Print some dataset statistics
        print(
            f"Loaded dataset with {len(self.train)} training examples and {len(self.test)} test examples"
        )
        print(f"Example item format: {self.train[0]}")

        # Initialize iteration counter
        self.iter = 0

    def save_checkpoint(self, step, data=None):
        if data is None:
            data = {}
        data["iter"] = self.iter
        super().save_checkpoint(step, data)

    async def get_next_item(self):
        """
        Get the next training item from the dataset.

        Returns:
            A tuple containing prompt and expected answer
        """
        next_item = self.train[self.iter % len(self.train)]
        self.iter += 1

        self.curr_item = next_item

        # Extract question and options from the multiple choice item
        board = next_item["board"]
        tags = next_item["tags"].split()
        moves = next_item["moves"].split()
        # rating = next_item["rating"]
        white_kingside_castling_rights = next_item["white_kingside"]
        white_queenside_castling_rights = next_item["white_queenside"]
        black_kingside_castling_rights = next_item["black_kingside"]
        black_queenside_castling_rights = next_item["black_queenside"]
        turn = next_item["turn"]

        unicode_to_piece = {
            "♔": "K",
            "♕": "Q",
            "♖": "R",
            "♗": "B",
            "♘": "N",
            "♙": "P",
            "♚": "k",
            "♛": "q",
            "♜": "r",
            "♝": "b",
            "♞": "n",
            "♟": "p",
        }

        # Replace unicode pieces with letters
        board = "".join(unicode_to_piece.get(c, c) for c in board)

        # Determine the turn text
        turn_text = "white" if turn == "w" else "black"

        # Create castling rights strings for both white and black
        white_kingside = "can" if white_kingside_castling_rights else "can't"
        white_queenside = "can" if white_queenside_castling_rights else "can't"
        black_kingside = "can" if black_kingside_castling_rights else "can't"
        black_queenside = "can" if black_queenside_castling_rights else "can't"

        # Format the castling rights text
        castling_rights_text = (
            f"Castling Rights: "
            f"white {white_kingside} castle kingside, "
            f"white {white_queenside} castle queenside, "
            f"black {black_kingside} castle kingside, and "
            f"black {black_queenside} castle queenside"
        )

        # Combine all parts into the final question text with instructions
        question_text_with_instruction = (
            f"Prompt: What are all the legal moves in the position for the active player?\n"
            f"Turn: {turn_text}\n"
            f"{castling_rights_text}\n"
            f"Board:\n{board}"
        )

        # Create prompt tuple using frozensets as required
        prompt = []

        # Add system prompt as defined at the top of the script
        prompt.append(
            frozenset({"role": "system", "content": self.system_prompt}.items())
        )

        # Add user message with the question and instruction
        prompt.append(
            frozenset(
                {"role": "user", "content": question_text_with_instruction}.items()
            )
        )

        return (tuple(prompt), moves, tags)

    async def collect_trajectories(self, item) -> Tuple[ScoredDataGroup, List]:
        """
        Generate and collect model responses for scoring.

        Args:
                prompt: tuple(systemprompt, question_with_instruction)
            item: Input item containing prompt and expected answer: (prompt, moves, tags)

        Returns:
            Tuple of lists containing scored data groups and backlog
        """

        # Extract messages from the item
        messages = []
        for role_dict in item[0]:
            messages.append(dict(role_dict))
        # messages is: [{'role': 'system', 'content': system_prompt},
        # {'role': 'user', 'content': question_with_instruction}]

        # Apply chat template to convert messages to a single string
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        # Get completions from the model using completion() instead of chat_completion()
        completions = await self.server.completion(
            prompt=prompt,
            n=self.config.group_size,
            max_tokens=self.config.max_token_length,
            temperature=1.0,  # Using temperature to get diverse responses
        )

        to_score = list()

        for i, completion_choice in enumerate(completions.choices):
            # Create a copy of the prompt messages
            trajectory_messages = []
            for role_dict in item[0]:
                trajectory_messages.append(dict(role_dict))

            # Add the model's response
            trajectory_messages.append(
                {"role": "assistant", "content": completion_choice.text}
            )

            # Add to scoring queue with prompt messages,
            # model's answer+explanation, ground truth moves,
            # ground truth tags for explanation, and finish reason
            to_score.append(
                (
                    tuple(trajectory_messages),
                    item[1],  # moves
                    item[2],  # tags for explanation
                    completion_choice.finish_reason,  # Add the stop reason
                )
            )

        # Call score to get the scored data
        scored_data = await self.score(to_score)
        to_backlog = []

        return scored_data, to_backlog

    async def score(self, rollout_group_data: List) -> Optional[ScoredDataGroup]:
        """
        Args:
            list of tuples in this format: tn ->
            ((system prompt, question prompt, model's answer),
            ground truth moves, ground truth tags for explanation, and finish reason)

            rollout_group_data: [t1, t2, t3, ... ] ->
            List of generated responses with expected answers

        Returns:
            ScoredDataGroup with tokenized inputs and scores, or None if no valid scores
        """
        scores = ScoredDataGroup()
        scores["tokens"] = list()
        scores["masks"] = list()
        scores["scores"] = list()

        # Shuffle to avoid bias in selection
        random.shuffle(rollout_group_data)

        for item in rollout_group_data:
            # Extract the model's response
            model_response = item[0][-1]["content"]
            stop_reason = item[3]  # Get the stop reason

            # If the response was cut off due to length, give it a score of 0
            if stop_reason == "length":
                reward = 0
            else:
                # Extract the answer from the model's response

                text = model_response
                pattern = re.compile(r"</tool_call>(.*?)(\[STOP\])", flags=re.DOTALL)
                text = pattern.sub(r"<think>\1</think>\2", text)
                model_response = text

                model_moves, thinking_text = self._extract_model_moves(model_response)

                move_reward = 0.0
                if model_moves != 0:
                    move_reward += 0.1  # little reward for getting the correct format

                    board = chess.Board(self.curr_item["fen"])
                    correct_legal_moves = [move.uci() for move in board.legal_moves]

                    model_set = set(model_moves)
                    correct_set = set(correct_legal_moves)

                    # Count how many moves are correct
                    num_correct = len(model_set & correct_set)  # intersection

                    # Divide by total legal moves
                    accuracy = num_correct / len(correct_set)

                    move_reward = accuracy

                thinking_reward = 0.0
                if thinking_text != 0:
                    thinking_reward += (
                        0.1  # little reward for getting the correct format
                    )

                # weights are tunable
                reward = 0.95 * move_reward + 0.05 * thinking_reward

            # Tokenize the conversation for learning
            out_dict = tokenize_for_trainer(self.tokenizer, item[0])
            tokens = out_dict["tokens"]
            masks = out_dict["masks"]

            # Remove examples with insufficient context
            if len([1 for i in masks if i != -100]) < 10:
                continue

            scores["tokens"].append(tokens)
            scores["masks"].append(masks)
            scores["scores"].append(reward)

            # Break once we have enough examples
            if len(scores["tokens"]) >= self.config.group_size:
                break

        # Record success rate metrics for wandb logging
        for score in scores["scores"]:
            self.percent_correct_buffer.append(max(score, 0))

        # Return None if all scores are the same (no learning signal)
        if all(scores["scores"][0] == score for score in scores["scores"]):
            print("all scores are: ", scores["scores"][0])
            print("no learning signal")
            return None

        return scores

    def _extract_model_moves(self, text):
        """
        Extract sequence of moves inside </moves> tags and reasoning inside of </{thinking_tag}> tags.
        Only allows one valid answer format - multiple answer formats result in a score of 0.

        Args:
            text: Text containing the model's response

        Returns:
            List of moves, text for thinking section
        """
        # pull the dynamic tag
        tag = re.escape(self.config.thinking_tag)  # escape just in case

        # Regex patterns for open/close tags
        open_tag_pattern = rf"<{tag}>"
        close_tag_pattern = rf"</{tag}>"

        # --- Thinking tag checks ---
        think_tags = re.findall(open_tag_pattern, text, re.IGNORECASE)
        if len(think_tags) > 1 or len(think_tags) != 1:
            return 0, 0

        think_close_tags = re.findall(close_tag_pattern, text, re.IGNORECASE)
        if len(think_close_tags) != 1:
            return 0, 0  # Must have exactly one closing tag

        # --- Moves tag checks ---
        moves_tags = re.findall(r"<moves>", text, re.IGNORECASE)
        if len(moves_tags) > 1 or len(moves_tags) != 1:
            return 0, 0

        moves_close_tags = re.findall(r"</moves>", text, re.IGNORECASE)
        if len(moves_close_tags) != 1:
            return 0, 0

        # Check if dynamic <tag> comes immediately after </moves>
        tag_sequence_match = re.search(rf"</moves>\s*<{tag}>", text, re.IGNORECASE)
        if not tag_sequence_match:
            return 0, 0

        # --- Extract moves section ---
        moves_section = re.search(
            r"<moves>(.*?)</moves>", text, re.DOTALL | re.IGNORECASE
        )
        if moves_section:
            moves_text = moves_section.group(1).strip()
            if "," not in moves_text:
                return 0, 0
        else:
            return 0, 0

        # --- Extract thinking section ---
        thinking_section = re.search(
            rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE
        )
        if not thinking_section:
            return 0, 0

        # Validate thinking section actually contains the opening tag
        if f"<{self.config.thinking_tag}>" not in thinking_section.group():
            return 0, 0

        # Make sure no thinking tags appear inside moves section
        if f"<{self.config.thinking_tag}>" in moves_section.group():
            return 0, 0

        return [
            m.strip() for m in moves_section.group(1).split(",")
        ], thinking_section.group(1)

    async def rollout_and_score_eval(self, test_item):
        """
        Generate and score model responses for a single test item.

        Args:
            test_item: Test item from dataset (contains board, fen, moves, etc.)

        Returns:
            Score (float between 0 and 1 for accuracy)
        """
        # Extract and format the test item similar to get_next_item
        board = test_item["board"]
        turn = test_item["turn"]
        white_kingside_castling_rights = test_item["white_kingside"]
        white_queenside_castling_rights = test_item["white_queenside"]
        black_kingside_castling_rights = test_item["black_kingside"]
        black_queenside_castling_rights = test_item["black_queenside"]

        # Convert unicode pieces to letters
        unicode_to_piece = {
            "♔": "K",
            "♕": "Q",
            "♖": "R",
            "♗": "B",
            "♘": "N",
            "♙": "P",
            "♚": "k",
            "♛": "q",
            "♜": "r",
            "♝": "b",
            "♞": "n",
            "♟": "p",
        }
        board = "".join(unicode_to_piece.get(c, c) for c in board)

        # Format the question similar to get_next_item
        turn_text = "white" if turn == "w" else "black"

        white_kingside = "can" if white_kingside_castling_rights else "can't"
        white_queenside = "can" if white_queenside_castling_rights else "can't"
        black_kingside = "can" if black_kingside_castling_rights else "can't"
        black_queenside = "can" if black_queenside_castling_rights else "can't"

        castling_rights_text = (
            f"Castling Rights: "
            f"white {white_kingside} castle kingside, "
            f"white {white_queenside} castle queenside, "
            f"black {black_kingside} castle kingside, and "
            f"black {black_queenside} castle queenside"
        )

        question_text_with_instruction = (
            f"Prompt: What are all the legal moves in the position for the active player?\n"
            f"Turn: {turn_text}\n"
            f"{castling_rights_text}\n"
            f"Board:\n{board}"
        )

        # Construct messages for the model using system prompt
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question_text_with_instruction},
        ]

        # Apply chat template to convert messages to a single string
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        # Get model completion
        completion = await self.server.completion(
            prompt=prompt,
            n=1,
            max_tokens=1024,
            temperature=0.5,
            split="eval",
        )

        # Extract the model's response
        model_response = completion.choices[0].text

        # Apply the same text preprocessing as in score()
        text = model_response
        pattern = re.compile(r"</tool_call>(.*?)(\[STOP\])", flags=re.DOTALL)
        text = pattern.sub(r"<think>\1</think>\2", text)
        model_response = text

        # Extract moves and thinking from model response
        model_moves, thinking_text = self._extract_model_moves(model_response)

        move_reward = 0.0
        if model_moves != 0:
            move_reward += 0.1  # little reward for getting the correct format

            board = chess.Board(test_item["fen"])
            correct_legal_moves = [move.uci() for move in board.legal_moves]

            model_set = set(model_moves)
            correct_set = set(correct_legal_moves)

            # Count how many moves are correct
            num_correct = len(model_set & correct_set)  # intersection

            # Divide by total legal moves
            accuracy = num_correct / len(correct_set)
            move_reward = accuracy

        thinking_reward = 0.0
        if thinking_text != 0:
            thinking_reward += 0.1  # little reward for getting the correct format

        # weights are tunable
        reward = 0.95 * move_reward + 0.05 * thinking_reward
        return reward

    async def evaluate(self, *args, **kwargs):
        """
        Evaluate the model on test data.
        """
        eval_tasks = []
        for test_item in self.test:
            eval_tasks.append(self.rollout_and_score_eval(test_item))

        # Run evaluation with progress bar
        scores = await tqdm_asyncio.gather(*eval_tasks, desc="Evaluating")

        # Calculate average score
        avg_score = sum(scores) / len(scores) if scores else 0.0
        self.eval_metrics.append(("eval/percent_correct", avg_score))

        print(
            f"Evaluation completed: {avg_score:.3f} average score on {len(scores)} examples"
        )

    async def add_rollouts_for_wandb(
        self,
        scored_data: Union[ScoredDataGroup, List[ScoredDataGroup]],
        item: Item = None,
    ):
        """
        Add rollouts for wandb logging.

        Args:
            scored_data: ScoredDataGroup containing tokens, masks, and scores
            item: Current item containing (prompt, moves, tags)
        """
        # save rollout to trajectory
        num_keep = self.config.num_rollouts_per_group_for_logging
        if num_keep == -1:
            num_keep = self.config.group_size

        # Ensure we don't try to keep more rollouts than we have
        num_keep = min(num_keep, len(scored_data["tokens"]))

        rollout_data = []
        for i in range(num_keep):
            # Decode the tokenized conversation
            decoded_text = self.tokenizer.decode(scored_data["tokens"][i])
            score = scored_data["scores"][i]
            expected_moves = item[1] if item else "Unknown"  # ground truth moves
            expected_tags = item[2] if item else "Unknown"  # ground truth tags

            rollout_data.append((decoded_text, score, expected_moves, expected_tags))

        self.rollouts_for_wandb.append(rollout_data)

        # Keep only the most recent rollouts to avoid memory issues
        if len(self.rollouts_for_wandb) > self.config.num_rollouts_to_keep:
            self.rollouts_for_wandb.pop(0)

    async def create_rollout_table(self, wandb_metrics):
        """
        Create a wandb table from collected rollouts.

        Args:
            wandb_metrics: Dictionary to add the table to

        Returns:
            Updated wandb_metrics dictionary
        """
        if len(self.rollouts_for_wandb) > 0:
            table = wandb.Table(
                columns=["text", "score", "expected_moves", "expected_tags"]
            )
            for group in self.rollouts_for_wandb:
                for item in group:
                    # item is (decoded_text, score, expected_moves, expected_tags)
                    table.add_data(item[0], item[1], str(item[2]), str(item[3]))
            wandb_metrics["train/rollouts"] = table

        # Clear the rollouts buffer after creating the table
        self.rollouts_for_wandb = []
        return wandb_metrics

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """
        Log metrics to wandb, including training accuracy and evaluation results.

        Args:
            wandb_metrics: Optional dictionary of metrics to log
        """
        if wandb_metrics is None:
            wandb_metrics = {}

        # Calculate training percent_correct from buffer
        try:
            if self.percent_correct_buffer:
                wandb_metrics["train/percent_correct"] = sum(
                    self.percent_correct_buffer
                ) / len(self.percent_correct_buffer)
        except ZeroDivisionError:
            # Skip if buffer is empty
            pass

        # Clear the buffer after logging
        self.percent_correct_buffer = []

        # Add evaluation metrics
        for metric_name, metric_value in self.eval_metrics:
            wandb_metrics[metric_name] = metric_value

        # Clear eval metrics after logging
        self.eval_metrics = []

        # Call parent class wandb_log method
        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    ChessRulesEnv.cli()
