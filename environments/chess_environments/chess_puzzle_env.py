import math
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
    "You are a chess-puzzle solver. ALWAYS output exactly this format and nothing else:\n\n"
    "<moves>comma-separated UCI moves</moves>\n"
    "<think>explain your reasoning here (this may include internal chain-of-thought)</think>[STOP]\n\n"
    "Rules:\n"
    "1) Do NOT use <tool_call>, <function_call>, JSON, or any other tags — only <moves> and <think>.\n"
    "2) Close both tags before emitting [STOP].\n"
    "3) Use chess keywords (fork, skewer, mate in 2, advanced pawn) where applicable inside <think>.\n\n"
    "Example:\n"
    "<moves>e2e4,e7e5</moves>\n"
    "<think>e2e4 opens lines; e7e5 is a standard reply... add more chess reasoning</think>[STOP]\n"
)


class ChessPuzzlesEnv(BaseEnv):
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
            f"Prompt: How to solve this puzzle?\n"
            f"Turn: {turn_text}\n"
            f"{castling_rights_text}\n"
            f"Board:\n{board}"
        )

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

        # Apply a final non-linear transformation
        final_reward = math.tanh(avg_reward)  # emphasizes high values more
        return max(min(final_reward, 1.0), 0.0)

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

                text = model_response
                text = re.sub(
                    r"<tool_call.*?>(.*?)</tool_call>",
                    r"<think>\1</think>",
                    text,
                    flags=re.DOTALL | re.IGNORECASE,
                )
                # If <tool_call> is opened but not closed
                if "<tool_call" in text and "</tool_call>" not in text:
                    m = re.search(
                        r"<tool_call[^>]*>(.*?)(?:\[STOP\]|$)", text, flags=re.DOTALL
                    )
                    if m:
                        inner = m.group(1).strip()
                        text = re.sub(
                            r"<tool_call[^>]*>.*",
                            f"<think>{inner}</think>[STOP]",
                            text,
                            flags=re.DOTALL,
                        )
                model_response = text

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
        Extract sequence of moves inside </moves> tags and reasoning inside of </think> tags
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
        moves_tags = re.findall(r"<moves>", text, re.IGNORECASE)
        if len(moves_tags) > 1:
            return 0, 0

        # Check if the moves tag is properly opened - we need exactly one opening tag
        if len(moves_tags) != 1:
            return 0, 0

        # Check for </moves> closing tags
        moves_close_tags = re.findall(r"</moves>", text, re.IGNORECASE)
        if len(moves_close_tags) != 1:
            return 0, 0  # Must have exactly one closing tag

        # Check if <think> comes immediately after </moves>
        tag_sequence_match = re.search(r"</moves>\s*<think>", text)
        if not tag_sequence_match:
            return 0, 0

        moves_section = re.search(r"<moves>(.*?)</moves>", text, re.DOTALL)
        if moves_section:
            moves_text = moves_section.group(1).strip()
            if "," in moves_text:
                pass
            else:
                return 0, 0

        thinking_section = re.search(r"<think>(.*?)</think>", text, re.DOTALL)

        # Validate thinking section
        # Make sure thinking section actually contains the opening <think> tag
        if "<think>" not in thinking_section.group():
            return 0, 0  # Malformed thinking section

        # Check if there are any <think> tags in the answer section (after the first </think>)
        if "<think>" in moves_section.group():
            return 0, 0

        return [
            m.strip() for m in moves_section.group(1).split(",")
        ], thinking_section.group(1)

    async def rollout_and_score_eval(self, test_item):
        """
        Generate and score model responses for a single test item.

        Args:
            test_item: Test item from dataset

        Returns:
            Score (1 for correct, 0 for incorrect)
        """
        # Construct messages for the model using system prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": test_item["prompt"]},
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

        # Extract moves and thinking from model response
        model_moves, thinking_text = self._extract_model_moves(model_response)

        # Compare with ground truth
        correct_moves = test_item.get("moves", "").split()
        if model_moves == 0 or not correct_moves:
            return 0

        # Reward using Stockfish evaluation
        reward = self.stockfish_reward(model_moves, test_item["fen"], correct_moves)

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
    ChessPuzzlesEnv.cli()
