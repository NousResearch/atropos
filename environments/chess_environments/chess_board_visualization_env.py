import json
import random
import re
from typing import Dict, List, Optional, Tuple, Union

import chess
import chess.engine
import wandb
from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    Item,
    ScoredDataGroup,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

system_prompt = (
    "You are a chess board visualizer. You will be given a list of moves in"
    "UCI notation starting from the standard chess position. "
    "Your task is to output the resulting board position after all the moves have been played. "
    "ALWAYS output exactly this format and nothing else:\n\n"
    "<board>ASCII board with each square separated by commas; empty squares as '.',"
    "white pieces as uppercase letters, black pieces as lowercase letters; en passant squares marked with '*'</board>\n"
    "<think>explain the current position and what is happening"
    "(include tactical or strategic observations, possible threats,"
    "and reasoning behind piece placement)</think>[STOP]\n\n"
    "Rules:\n"
    "1) Output the board from White's perspective (top is Black, bottom is White).\n"
    "2) Use the following letter notation: K=White King, Q=White Queen, R=White Rook, B=White Bishop,"
    "N=White Knight, P=White Pawn; lowercase for black pieces.\n"
    "3) Empty squares must be a dot '.', en passant target squares marked with '*'.\n"
    "4) Do NOT use <tool_call>, <function_call>, JSON, or any other tags — only <board> and <think>.\n"
    "5) Close both tags before emitting [STOP].\n\n"
    "Example:\n"
    "<board>r,n,b,q,k,b,n,r,p,p,p,p,.,.,.,.,.,.,.,.,.,.,.,.,P,P,P,P,R,N,B,Q,K,B,N,R</board>\n"
    "<think>All pawns are on their initial squares except for ... [describe moves]; pieces are developed according "
    "to the move sequence; White is ready to castle kingside, Black has yet to move</think>[STOP]\n"
)


class ChessBoardVisualizationEnv(BaseEnv):
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

    @classmethod
    def config_init(self) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
        env_config = BaseEnvConfig(
            tokenizer_name="Qwen/Qwen3-4B-Instruct-2507",
            group_size=4,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=12,
            steps_per_eval=100,
            max_token_length=2048,
            wandb_name="chess_puzzle_solver",
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
            f"Prompt: Internally playout all of these moves below and then"
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
        prompt.append(frozenset({"role": "system", "content": system_prompt}.items()))

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

    def toolcall_to_think(self, text: str) -> str:
        """
        Replace <tool_call>... (with or without closing tag) with <think>...</think>.
        If a [STOP] follows the tool_call content (and no closing tag exists), ensure
        the [STOP] ends up after the closing </think>.
        """
        pattern = re.compile(
            r"<tool_call\b[^>]*>(.*?)"  # capture content after opening tag
            r"(?:</tool_call>|(\s*\[STOP\])|$)",  # stop at closing tag, or capture trailing [STOP], or end
            flags=re.DOTALL | re.IGNORECASE,
        )

        def _repl(m):
            content = m.group(1).strip()
            stopper = m.group(2) or ""
            return f"<think>{content}</think>{stopper}"

        return pattern.sub(_repl, text)

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

                text = model_response
                pattern = re.compile(r"</tool_call>(.*?)(\[STOP\])", flags=re.DOTALL)
                text = pattern.sub(r"<think>\1</think>\2", text)
                model_response = text

                model_response = self.toolcall_to_think(model_response)

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
        Extract ASCII board inside </board> tags and reasoning inside of </think> tags
        Only allows one valid answer format - multiple answer formats result in a score of 0.

        Args:
            text: Text containing the model's response

        Returns:
            List of moves, text for thinking section
        """

        # Check for multiple <think> tags - score as 0 if found
        think_tags = re.findall(r"<think>", text, re.IGNORECASE)
        if len(think_tags) > 1:
            return 0, 0

        # Check if the think tag is properly opened - we need exactly one opening tag
        if len(think_tags) != 1:
            return 0, 0

        # Check for </think> closing tags
        think_close_tags = re.findall(r"</think>", text, re.IGNORECASE)
        if len(think_close_tags) != 1:
            return 0, 0  # Must have exactly one closing tag

        # Check for multiple <moves> tags - score as 0 if found
        board_tags = re.findall(r"<board>", text, re.IGNORECASE)
        if len(board_tags) > 1:
            return 0, 0

        # Check if the moves tag is properly opened - we need exactly one opening tag
        if len(board_tags) != 1:
            return 0, 0

        # Check for </moves> closing tags
        board_close_tags = re.findall(r"</board>", text, re.IGNORECASE)
        if len(board_close_tags) != 1:
            return 0, 0  # Must have exactly one closing tag

        # Check if <think> comes immediately after </moves>
        tag_sequence_match = re.search(r"</board>\s*<think>", text)
        if not tag_sequence_match:
            return 0, 0

        board_section = re.search(r"<board>(.*?)</board>", text, re.DOTALL)
        if board_section:
            board_text = board_section.group(1).strip()
            if "," in board_text:
                pass
            else:
                return 0, 0

        thinking_section = re.search(r"<think>(.*?)</think>", text, re.DOTALL)

        # Validate thinking section
        # Make sure thinking section actually contains the opening <think> tag
        if "<think>" not in thinking_section.group():
            return 0, 0  # Malformed thinking section

        # Check if there are any <think> tags in the answer section (after the first </think>)
        if "<think>" in board_section.group():
            return 0, 0

        return board_section.group(1), thinking_section.group(1)

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
            f"Prompt: Internally playout all of these moves below and then"
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
            {"role": "system", "content": system_prompt},
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

        text = model_response
        pattern = re.compile(r"</tool_call>(.*?)(\[STOP\])", flags=re.DOTALL)
        text = pattern.sub(r"<think>\1</think>\2", text)
        model_response = text

        model_response = self.toolcall_to_think(model_response)

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
        self.rollouts_for_wandb.append(
            [
                (
                    self.tokenizer.decode(scored_data["tokens"][i]),
                    scored_data["scores"][i],
                    item[1],
                    item[2],
                )
                for i in range(num_keep)
            ]
        )
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
