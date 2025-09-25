import math
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


def build_system_prompt(config: ChessEnvConfig) -> str:
    """
    Build the system prompt dynamically based on config values.
    """
    # tag for the reasoning section (avoid collisions)
    tag = config.thinking_tag or "analysis"

    # reasoning section
    if config.thinking_mode:
        reasoning_section = (
            f"<{tag}>explain your reasoning here "
            f"(this may include internal chain-of-thought)</{tag}>[STOP]\n\n"
        )
    else:
        reasoning_section = "[STOP]\n\n"

    # prepend a provider-specific marker if needed
    prefix = (
        (config.thinking_system_prompt + "\n\n")
        if config.thinking_system_prompt
        else ""
    )

    prompt = (
        prefix
        + "You are a chess-puzzle solver. ALWAYS output exactly this format and nothing else:\n\n"
        f"{reasoning_section}"
        "<moves>comma-separated UCI moves</moves>\n"
        "Rules:\n"
        "1) Do NOT use <tool_call>, <function_call>, JSON, or any other tags — only <moves> "
        f"and <{tag}>.\n"
        "2) Close both tags before emitting [STOP].\n"
        "3) Use chess keywords (fork, skewer, mate in 2, advanced pawn) where applicable "
        f"inside <{tag}>.\n\n"
        "Example:\n"
        f"<{tag}>e2e4 opens lines; e7e5 is a standard reply... add more chess reasoning</{tag}>\n"
        "<moves>e2e4,e7e5</moves>[STOP]\n"
    )

    return prompt


# === Environment class ===
class ChessPuzzlesEnv(BaseEnv):

    def __init__(
        self,
        config: ChessEnvConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = False,
        testing: bool = False,
    ):
        """
        Initialize the Chess Puzzle Solver environment.

        Args:
            config: Configuration for the chess environment.
            server_configs: List of server configurations for OpenAI API.
            slurm: Whether to use Slurm for distributed training.
            testing: Whether in testing mode.
        """
        super().__init__(config, server_configs, slurm, testing)
        self.percent_correct_buffer = []
        self.eval_metrics = []

        # build the dynamic system prompt once here
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
        ).shuffle(seed=42)

        # Create train/test split on the fly (e.g., 95% train, 5% test)
        split_dataset = full_dataset.train_test_split(test_size=0.02, seed=42)

        self.train = split_dataset["train"]
        self.test = split_dataset["test"]

        print(
            f"Loaded dataset with {len(self.train)} training examples and {len(self.test)} test examples"
        )
        print(f"Example item format: {self.train[0]}")

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
            f"Prompt: How to solve this puzzle?\n"
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

    def stockfish_reward(
        self,
        pred_moves,
        initial_fen,
        correct_moves,
        stockfish_path="stockfish_path",
    ):
        board = chess.Board(initial_fen)
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

        pred_board = board.copy()
        cumulative_reward = 0.0

        for step, move in enumerate(pred_moves, start=1):
            try:
                pred_board.push_uci(move)
            except ValueError:  # catch only expected errors
                break

            # Analyse the predicted board at this step
            info_pred = engine.analyse(pred_board, chess.engine.Limit(depth=12))
            pred_score = info_pred["score"].white().score(mate_score=10000)
            if pred_score is None:
                pred_score = 0

            # Analyse the correct board up to this step
            correct_board = board.copy()
            for correct_move in correct_moves[:step]:
                try:
                    correct_board.push_uci(correct_move)
                except ValueError:
                    break
            info_correct = engine.analyse(correct_board, chess.engine.Limit(depth=12))
            correct_score = info_correct["score"].white().score(mate_score=10000)
            if correct_score is None:
                correct_score = 0

            # Step-wise similarity: exponential reward for being close to Stockfish
            step_reward = math.exp(-abs(pred_score - correct_score) / 100.0)
            cumulative_reward += step_reward

        # Average across all predicted moves and clamp
        avg_reward = cumulative_reward / max(len(pred_moves), 1)
        engine.quit()

        return avg_reward

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

        ground_truth_tags = rollout_group_data[0][
            2
        ]  # ground truth tags for explanation

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
                model_moves, thinking_text = self._extract_model_moves(model_response)

                move_reward = 0.0
                if model_moves != 0:
                    move_reward += 0.1  # little reward for getting the correct format
                    move_reward = self.stockfish_reward(
                        model_moves,
                        self.curr_item["fen"],
                        self.curr_item["moves"].split(" "),
                    )

                thinking_reward = 0.0
                if thinking_text != 0:
                    thinking_reward += (
                        0.1  # little reward for getting the correct format
                    )
                    thinking_reward = sum(
                        1 for k in ground_truth_tags if k in thinking_text
                    ) / len(ground_truth_tags)

                # weights are tunable
                reward = 0.7 * move_reward + 0.3 * thinking_reward

            # Tokenize the conversation for learning
            out_dict = tokenize_for_trainer(self.tokenizer, item[0])
            tokens = out_dict["tokens"]
            masks = out_dict["masks"]

            # Remove examples with insufficient context
            if len([1 for i in masks if i != -100]) < 10:
                continue

            print("tokens:", len(tokens))

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
        Extract sequence of moves inside <moves> tags and reasoning inside <{thinking_tag}> tags.
        Ignores the order of the tags, but still enforces one <moves> and one <thinking_tag> section.

        Args:
            text: Text containing the model's response

        Returns:
            List of moves, text for thinking section, or (0, 0) if invalid
        """
        # Dynamic thinking tag
        tag = re.escape(self.config.thinking_tag)

        # --- Moves section ---
        moves_section = re.search(
            r"<moves>(.*?)</moves>", text, re.DOTALL | re.IGNORECASE
        )
        if not moves_section:
            return 0, 0
        moves_text = moves_section.group(1).strip()
        if "," not in moves_text:
            return 0, 0
        moves_list = [m.strip() for m in moves_text.split(",")]

        # --- Thinking section ---
        thinking_section = re.search(
            rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE
        )
        if not thinking_section:
            return 0, 0
        thinking_text = thinking_section.group(1).strip()

        # --- Validation ---
        # Must have exactly one of each tag
        if (
            len(re.findall(r"<moves>", text, re.IGNORECASE)) != 1
            or len(re.findall(r"</moves>", text, re.IGNORECASE)) != 1
        ):
            return 0, 0
        if (
            len(re.findall(rf"<{tag}>", text, re.IGNORECASE)) != 1
            or len(re.findall(rf"</{tag}>", text, re.IGNORECASE)) != 1
        ):
            return 0, 0

        # Make sure thinking tag does not appear inside moves section
        if re.search(rf"<{tag}>", moves_section.group(), re.IGNORECASE):
            return 0, 0

    return moves_list, thinking_text

    async def rollout_and_score_eval(self, test_item):
        """
        Generate and score model responses for a single test item.
        Args:
            test_item: Test item from dataset (similar format to training items)
        Returns:
            Score (reward value between 0 and 1)
        """
        # Convert test item to the same format as training items
        board = test_item["board"]
        tags = test_item["tags"].split()
        moves = test_item["moves"].split()
        white_kingside_castling_rights = test_item["white_kingside"]
        white_queenside_castling_rights = test_item["white_queenside"]
        black_kingside_castling_rights = test_item["black_kingside"]
        black_queenside_castling_rights = test_item["black_queenside"]
        turn = test_item["turn"]

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

        # Create castling rights strings
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

        # Create the question text
        question_text_with_instruction = (
            f"Prompt: How to solve this puzzle?\n"
            f"Turn: {turn_text}\n"
            f"{castling_rights_text}\n"
            f"Board:\n{board}"
        )

        # Construct messages for the model
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
            max_tokens=self.config.max_token_length,
            temperature=0.1,  # Lower temperature for evaluation
            split="eval",
        )

        # Extract the model's response
        model_response = completion.choices[0].text
        stop_reason = completion.choices[0].finish_reason

        # If the response was cut off due to length, give it a score of 0
        if stop_reason == "length":
            return 0.0

        # Extract moves and thinking from model response
        model_moves, thinking_text = self._extract_model_moves(model_response)

        # If extraction failed, return 0
        if model_moves == 0:
            return 0.0

        # Calculate reward using Stockfish evaluation
        move_reward = self.stockfish_reward(model_moves, test_item["fen"], moves)

        # Calculate thinking reward if thinking text was extracted
        thinking_reward = 0.0
        if thinking_text != 0:
            thinking_reward = 0.1  # Base reward for correct format
            thinking_reward += sum(1 for tag in tags if tag in thinking_text) / len(
                tags
            )

        # Combined reward (same weights as training)
        reward = 0.7 * move_reward + 0.3 * thinking_reward
        return reward

    async def evaluate(self, *args, **kwargs):
        """
        Evaluate the model on test data.
        """
        print(f"Starting evaluation on {len(self.test)} test examples...")

        eval_tasks = []
        # Limit evaluation to a reasonable number of examples to avoid timeouts
        eval_subset = self.test[: min(len(self.test), 100)]

        for test_item in eval_subset:
            eval_tasks.append(self.rollout_and_score_eval(test_item))

        # Run evaluation with progress bar
        scores = await tqdm_asyncio.gather(*eval_tasks, desc="Evaluating")

        # Calculate metrics
        avg_score = sum(scores) / len(scores) if scores else 0.0

        # Store evaluation metrics
        self.eval_metrics.append(("eval/avg_reward", avg_score))

        # Also track binary accuracy (reward > 0.5 as "correct")
        binary_accuracy = (
            sum(1 for score in scores if score > 0.5) / len(scores) if scores else 0.0
        )
        self.eval_metrics.append(("eval/binary_accuracy", binary_accuracy))

        print(
            f"Evaluation complete. Average reward: {avg_score:.3f}, Binary accuracy: {binary_accuracy:.3f}"
        )

    async def add_rollouts_for_wandb(
        self,
        scored_data: Union[ScoredDataGroup, List[ScoredDataGroup]],
        item: Item = None,
    ):
        """
        Add rollouts to the buffer for wandb logging.
        Args:
            scored_data: Scored data group containing tokens and scores
            item: Original item tuple (prompt, moves, tags)
        """
        if not hasattr(self.config, "num_rollouts_per_group_for_logging"):
            num_keep = min(self.config.group_size, 3)  # Default to 3 if not specified
        else:
            num_keep = self.config.num_rollouts_per_group_for_logging
            if num_keep == -1:
                num_keep = self.config.group_size

        # Ensure we don't exceed available data
        num_keep = min(num_keep, len(scored_data["tokens"]))

        rollout_group = []
        for i in range(num_keep):
            # Decode tokens to text
            decoded_text = self.tokenizer.decode(
                scored_data["tokens"][i], skip_special_tokens=True
            )
            score = scored_data["scores"][i]

            # Extract ground truth info from item
            if item is not None:
                ground_truth_moves = item[1]  # List of moves
                ground_truth_tags = item[2]  # List of tags

                # Convert moves list to string for display
                moves_str = (
                    ",".join(ground_truth_moves)
                    if isinstance(ground_truth_moves, list)
                    else str(ground_truth_moves)
                )
                tags_str = (
                    ",".join(ground_truth_tags)
                    if isinstance(ground_truth_tags, list)
                    else str(ground_truth_tags)
                )
            else:
                moves_str = "N/A"
                tags_str = "N/A"

            rollout_group.append((decoded_text, score, moves_str, tags_str))

        # Initialize rollouts buffer if it doesn't exist
        if not hasattr(self, "rollouts_for_wandb"):
            self.rollouts_for_wandb = []

        self.rollouts_for_wandb.append(rollout_group)

        # Maintain buffer size limit
        max_rollouts = getattr(self.config, "num_rollouts_to_keep", 10)
        if len(self.rollouts_for_wandb) > max_rollouts:
            self.rollouts_for_wandb.pop(0)

    async def create_rollout_table(self, wandb_metrics):
        """
        Create a wandb table from stored rollouts and add it to metrics.
        Args:
            wandb_metrics: Dictionary to add the table to
        Returns:
            Updated wandb_metrics dictionary
        """
        if hasattr(self, "rollouts_for_wandb") and len(self.rollouts_for_wandb) > 0:
            table = wandb.Table(
                columns=["text", "score", "ground_truth_moves", "ground_truth_tags"]
            )

            for group in self.rollouts_for_wandb:
                for item in group:
                    # item is (decoded_text, score, moves_str, tags_str)
                    table.add_data(item[0], item[1], item[2], item[3])

            wandb_metrics["train/rollouts"] = table

            # Clear the buffer after logging
            self.rollouts_for_wandb = []

        return wandb_metrics

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """
        Log metrics to wandb, including training and evaluation metrics.
        Args:
            wandb_metrics: Optional dictionary of additional metrics to log
        """
        if wandb_metrics is None:
            wandb_metrics = {}

        # Add training percent correct if available
        if len(self.percent_correct_buffer) > 0:
            wandb_metrics["train/percent_correct"] = sum(
                self.percent_correct_buffer
            ) / len(self.percent_correct_buffer)
            # Clear the buffer after logging
            self.percent_correct_buffer = []

        # Add evaluation metrics
        for metric_name, metric_value in self.eval_metrics:
            wandb_metrics[metric_name] = metric_value

        # Clear evaluation metrics after logging
        self.eval_metrics = []

        # Create rollout table if we have rollouts
        wandb_metrics = await self.create_rollout_table(wandb_metrics)

        # Call parent class wandb_log method
        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    ChessPuzzlesEnv.cli()
