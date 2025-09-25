import json
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
    Build the system prompt dynamically for the chess board visualizer
    based on config values.
    """
    # dynamic tag for reasoning section
    tag = config.thinking_tag or "think"

    # prepend any extra system text if provided
    prefix = (
        (config.thinking_system_prompt + "\n\n")
        if config.thinking_system_prompt
        else ""
    )

    prompt = (
        prefix
        + "You are a chess board visualizer. You will be given a list of moves in UCI notation,"
        "starting from the standard chess position. "
        "Play the moves internally and output the final board position in ASCII format.\n\n"
        "<board>Each square separated by commas; empty squares as '.',"
        "white pieces uppercase, black pieces lowercase, en passant squares as '*'. "
        "Must contain exactly 64 squares. Do NOT include any extra text or explanation inside <board>.</board>\n"
        f"<{tag}>Provide your step by step logic of each move being played out on the board."
        "Once you get your final position, explain what is happening in the final position.</{tag}>[STOP]\n\n"
        "Rules:\n"
        "1) Board is from White's perspective (top is Black, bottom is White).\n"
        "2) Pieces: K=White King, Q=White Queen, R=White Rook, "
        "B=White Bishop, N=White Knight, P=White Pawn; lowercase for Black.\n"
        "3) Empty squares as '.', en passant as '*'.\n"
        "4) Do NOT use <tool_call>, <function_call>, JSON, or any other tags — only <board> and "
        f"<{tag}>.\n"
        "5) Close both tags before [STOP].\n"
        "6) <board> must always have exactly 64 comma-separated characters.\n\n"
        "Example format (do NOT copy this board, just follow the format):\n"
        f"<{tag}>Provide your step by step logic of each move being played out on the board."
        "Once you get your final position, explain what is happening in the final position.</{tag}>[STOP]\n"
        "<board>r,.,.,q,.,r,.,k,p,p,.,.,b,.,"
        "p,p,.,.,n,p,.,n,.,.,.,.,.,N,p,b,.,.,.,.,P,.,.,.,.,.,N"
        ",.,.,.,B,.,.,.,P,P,.,.,B,P,P,P,R,.,.,Q,.,R,K,.</board>\n"
        
    )

    return prompt


class ChessBoardVisualizationEnv(BaseEnv):
    def __init__(
        self,
        config: BaseEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=False,
        testing=False,
    ):
        """
        Initialize the chess board visualization env.

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
        full_dataset = load_dataset("Thytu/ChessInstruct", split="train")

        full_dataset = full_dataset.filter(
            lambda example: example["KIND"] == "FIND_NEXT_BEST_MOVE"
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

    def fen_to_ascii_board(self, fen: str) -> str:
        """
        Convert a FEN string into a single-line ASCII board.
        Pieces: K,Q,R,B,N,P (uppercase for White, lowercase for Black)
        Empty squares: .
        En passant square: *
        Left-to-right, top-to-bottom (from White's perspective)
        """
        board = chess.Board(fen)

        result = []
        for rank in range(8, 0, -1):  # ranks 8 -> 1
            for file in range(8):  # files a -> h
                square = chess.square(file, rank - 1)
                piece = board.piece_at(square)
                if piece:
                    letter = piece.symbol()  # uppercase for White, lowercase for Black
                    result.append(letter)
                elif board.ep_square == square:
                    result.append("*")
                else:
                    result.append(".")

        return "".join(result)

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
        moves = json.loads(next_item["input"])["moves"]

        # Combine all parts into the final question text with instructions
        question_text_with_instruction = (
            f"Prompt: Internally playout all of these moves below and then "
            "output the final position in ASCII format and describe what is happening in the final position. "
            f"Moves: {','.join(str(m) for m in moves)}\n"
        )

        board = chess.Board()

        # Play each move
        for move_uci in moves:
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                board.push(move)
            else:
                raise ValueError(f"Illegal move: {move_uci}")

        final_ascii_string = self.fen_to_ascii_board(board.fen())

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

        return (tuple(prompt), final_ascii_string)

    async def collect_trajectories(self, item) -> Tuple[ScoredDataGroup, List]:
        """
        Generate and collect model responses for scoring.

        Args:
                prompt: tuple(systemprompt, question_with_instruction)
            item: Input item containing prompt and expected answer: (prompt, moves,)

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
                    item[1],  # final ascii board string
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
        ground_truth_ascii_strings = rollout_group_data[0][1]

        for item in rollout_group_data:
            # Extract the model's response
            model_response = item[0][-1]["content"]
            stop_reason = item[2]  # Get the stop reason

            # If the response was cut off due to length, give it a score of 0
            if stop_reason == "length":
                reward = 0
            else:
                # Extract the answer from the model's response
                model_board, thinking_text = self._extract_model_board(model_response)

                board_reward = 0.0
                if model_board != 0:
                    board_reward += 0.1  # little reward for getting the correct format
                    model_clean = model_board.replace(",", "")
                    truth_clean = ground_truth_ascii_strings.replace(",", "")

                    # Make both same length (ground truth is reference)
                    length = len(truth_clean)
                    model_clean = model_clean[:length]  # trim if model string is longer

                    matches = sum(
                        1 for mc, tc in zip(model_clean, truth_clean) if mc == tc
                    )
                    accuracy = matches / length if length > 0 else 0.0
                    board_reward = accuracy

                thinking_reward = 0.0
                if thinking_text != 0:
                    thinking_reward += (
                        0.1  # little reward for getting the correct format
                    )

                # weights are tunable
                reward = 0.95 * board_reward + 0.05 * thinking_reward

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

    def _extract_model_board(self, text):
        """
        Extract board state inside <board> tags and reasoning inside <{thinking_tag}> tags.

        Args:
            text: Text containing the model's response.

        Returns:
            Tuple[List[str], str]: (list of squares from board, analysis text)
        """
        tag = re.escape(self.config.thinking_tag)  # e.g. "analysis"

        # --- Extract board section ---
        board_match = re.search(
            r"<board>(.*?)</board>", text, re.DOTALL | re.IGNORECASE
        )
        if not board_match:
            return 0, 0
        board_text = board_match.group(1).strip()

        # --- Extract analysis/thinking section ---
        thinking_match = re.search(
            rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE
        )
        if not thinking_match:
            return 0, 0
        thinking_text = thinking_match.group(1).strip()

        # Extra: Make sure no thinking tags inside board
        if re.search(rf"<{tag}>", board_text, re.IGNORECASE):
            return 0, 0

        return board_text, thinking_text

    async def rollout_and_score_eval(self, test_item):
        """
        Generate and score model responses for a single test item.

        Args:
            test_item: Test item from dataset

        Returns:
            Score (1 for correct, 0 for incorrect)
        """

        # Extract question and options from the multiple choice item
        moves = json.loads(test_item["input"])["moves"]

        # Combine all parts into the final question text with instructions
        question_text_with_instruction = (
            f"Prompt: Internally playout all of these moves below and then "
            "output the final position in ASCII format and describe what is happening in the final position. "
            f"Moves: {','.join(str(m) for m in moves)}\n"
        )

        board = chess.Board()

        # Play each move
        for move_uci in moves:
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                board.push(move)
            else:
                raise ValueError(f"Illegal move: {move_uci}")

        final_ascii_string = self.fen_to_ascii_board(board.fen())

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

        # Extract board and thinking from model response
        model_board, thinking_text = self._extract_model_board(model_response)

        board_reward = 0.0
        if model_board != 0:
            board_reward += 0.1  # little reward for getting the correct format
            model_clean = model_board.replace(",", "")
            truth_clean = final_ascii_string.replace(",", "")

            # Make both same length (ground truth is reference)
            length = len(truth_clean)
            model_clean = model_clean[:length]  # trim if model string is longer

            matches = sum(1 for mc, tc in zip(model_clean, truth_clean) if mc == tc)
            accuracy = matches / length if length > 0 else 0.0
            board_reward = accuracy

        thinking_reward = 0.0
        if thinking_text != 0:
            thinking_reward += 0.1  # little reward for getting the correct format

        # weights are tunable
        reward = 0.95 * board_reward + 0.05 * thinking_reward
        return reward

    async def evaluate(self, *args, **kwargs):
        """
        Evaluate the model on test data.
        """
        eval_tasks = []
        for test_item in self.test:
            eval_tasks.append(self.rollout_and_score_eval(test_item))

        # Run evaluation
        scores = await tqdm_asyncio.gather(*eval_tasks)
        self.eval_metrics.append(("eval/percent_correct", sum(scores) / len(scores)))

    async def add_rollouts_for_wandb(
        self,
        scored_data: Union[ScoredDataGroup, List[ScoredDataGroup]],
        item: Item = None,
    ):
        # save rollout to trajectory
        num_keep = self.config.num_rollouts_per_group_for_logging
        if num_keep == -1:
            num_keep = self.config.group_size

        # Extract the ground truth answer from item
        ground_truth_answer = item[1] if item else "N/A"

        rollout_group = []
        for i in range(min(num_keep, len(scored_data["tokens"]))):
            # Decode the tokens to get the text
            decoded_text = self.tokenizer.decode(scored_data["tokens"][i])
            score = scored_data["scores"][i]

            # Extract just the assistant's response from the decoded text
            # This assumes the decoded text contains the full conversation
            lines = decoded_text.split("\n")
            assistant_response = ""
            for line in lines:
                if line.strip().startswith("assistant") or "<board>" in line:
                    assistant_response = line
                    break

            rollout_group.append(
                (
                    assistant_response if assistant_response else decoded_text,
                    score,
                    ground_truth_answer,
                    str(ground_truth_answer),  # string version of answer
                )
            )

        self.rollouts_for_wandb.append(rollout_group)

        # Keep only the most recent rollouts
        if len(self.rollouts_for_wandb) > self.config.num_rollouts_to_keep:
            self.rollouts_for_wandb.pop(0)

    async def create_rollout_table(self, wandb_metrics):
        if len(self.rollouts_for_wandb) > 0:
            table = wandb.Table(columns=["text", "score", "answer", "string_answer"])
            for group in self.rollouts_for_wandb:
                for item in group:
                    table.add_data(item[0], item[1], item[2], item[3])
            wandb_metrics["train/rollouts"] = table
        self.rollouts_for_wandb = []
        return wandb_metrics

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}

        # Try to calculate percent_correct, pass if there's a division by zero
        try:
            wandb_metrics["train/percent_correct"] = sum(
                self.percent_correct_buffer
            ) / len(self.percent_correct_buffer)
        except ZeroDivisionError:
            # Skip if buffer is empty
            pass

        self.percent_correct_buffer = list()
        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = list()

        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    ChessBoardVisualizationEnv.cli()
