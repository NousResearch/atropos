import random
import re
from typing import Dict, List, Optional, Tuple, Union

from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio

import wandb
from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    Item,
    ScoredDataGroup,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

system_prompt = (
    "You are a deep-thinking chess AI trained to solve chess puzzles step by step.\n"
    "Your task is to analyze a chess position (given as an 8x8 unicode board), output the best sequence of moves, "
    "and explain your reasoning.\n\n"
    "The input will be in this format:\n"
    "Prompt: How to solve this chess puzzle?\n"
    "Turn: <white or black>\n"
    "Castling Rights: white can/can't castle kingside, white can/can't castle queenside, "
    "black can/can't castle kingside, and black can/can't castle queenside\n"
    "Board:\n"
    "8x8 unicode board separated by commas,"
    "where each piece is represented by its corresponding unicode characters,"
    "and an en-passant square is marked with !\n\n"
    "Note: The prompt may be worded differently, but if it implies that you should solve the chess puzzle, "
    "begin solving it using the format described below. "
    "For the output you must enclose your moves inside of <moves></moves> tags, each move should be separated "
    "by a comma and be written in UCI notation (eg. e2e4, d2d4). Underneath the moves you must give a detailed "
    "explanation of how you came to the sequence of moves inside of <think></think> tags. Your explanation must "
    "use keywords when applicable in chess such as advanced pawn, mate in 2, fork, skewer, etc. If you don’t use "
    "the correct keywords or don’t use keywords at all when they are applicable you will also be harshly penalized. "
    "You must follow this output format exactly or you will not receive credit for your answer.\n\n"
    "Important:\n"
    "- Pay special attention to your first move — future moves depend on it.\n"
    "- After each move, internally simulate the new board state accurately.\n"
    "- Make sure you consider chess rules and how the pieces move when coming up with your decision,"
    "you will be penalized harshly if you play an illegal move\n\n"
    "Example Input:\n"
    "Prompt: How to solve this puzzle?\n"
    "Turn: white\n"
    "Castling Rights: white can't castle kingside, white can't castle queenside,"
    "black can't castle kingside, and black can't castle queenside\n"
    "Board:\n"
    "., ., ., ., ., ., ., .,\n"
    "., ., ., ., ., ., ., ♖,\n"
    "., ., ., ., ., ♟, ., .,\n"
    "♟, ., ., ., ., ., ., .,\n"
    "., ., ., ., ., ., ., ♙,\n"
    "., ., ♟, ., ., ., ., .,\n"
    "., ., ., ♚, ., ., ♘, .,\n"
    "., ♔, ., ., ., ., ., .,\n\n"
    "Example Output:\n"
    "<moves> c3c2, b1a2, c2c1q, h7d7, d2e2 </moves>\n"
    "<think> Advancing the pawn down the board is crushing because it leads to the promotion of a"
    "queen because pushing the pawn comes with a check. </think>\n\n"
    "Rules:\n"
    "- Think carefully about the legality of each move — illegal moves will be harshly penalized.\n"
    "- Do not skip any lines.\n"
    "- Do not output anything beyond the specified format.\n"
    "- Stop once the puzzle is solved or the opponent is checkmated.\n"
    "- You will not receive credit for your answer if you don’t follow this format exactly.\n\n"
    "Be accurate, deliberate, and thoughtful. Your reasoning matters. Simulate carefully after every move."
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
        Initialize the MCQA (Multiple Choice Question Answering) environment.

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
            tokenizer_name="meta-llama/Llama-3.1-8B-Instruct",
            group_size=8,
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
                model_name="meta-llama/Llama-3.1-8B-Instruct",
                base_url="http://localhost:9001/v1",
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
            max_tokens=1024 * 15,
            temperature=1.0,  # Using temperature to get diverse responses
        )

        print(completions)

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

        # Get the expected answer letter
        ground_truth_moves = rollout_group_data[0][1].split()  # ground truth moves
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

                # Track metrics based on result
                if model_moves is None:
                    reward = 0  # Invalid format gets 0 reward
                elif model_moves == ground_truth_moves:
                    reward = 1  # Correct answer gets 1 reward
                else:
                    reward = 0  # Wrong answer gets 0 reward

                all_present = all(
                    keyword in thinking_text for keyword in ground_truth_tags
                )
                if all_present:
                    reward += 1

                reward /= 2

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
            return None

        # Check if the think tag is properly opened - we need exactly one opening tag
        if len(think_tags) != 1:
            return None

        # Check for </think> closing tags
        think_close_tags = re.findall(r"</think>", text, re.IGNORECASE)
        if len(think_close_tags) != 1:
            return None  # Must have exactly one closing tag

        # Check for multiple <moves> tags - score as 0 if found
        moves_tags = re.findall(r"<moves>", text, re.IGNORECASE)
        if len(moves_tags) > 1:
            return None

        # Check if the moves tag is properly opened - we need exactly one opening tag
        if len(moves_tags) != 1:
            return None

        # Check for </moves> closing tags
        moves_close_tags = re.findall(r"</moves>", text, re.IGNORECASE)
        if len(moves_close_tags) != 1:
            return None  # Must have exactly one closing tag

        # Check if <think> comes immediately after </moves>
        tag_sequence_match = re.search(r"</moves>\s*<think>", text)
        if not tag_sequence_match:
            return None

        moves_section = re.search(r"<moves>(.*?)</moves>", text, re.DOTALL)
        if moves_section:
            moves_text = moves_section.group(1).strip()
            if "," in moves_text:
                pass
            else:
                return None

        thinking_section = re.search(r"<think>(.*?)</think>", text, re.DOTALL)

        # Validate thinking section
        # Make sure thinking section actually contains the opening <think> tag
        if "<think>" not in thinking_section.lower():
            return None  # Malformed thinking section

        # Check if there are any <think> tags in the answer section (after the first </think>)
        if "<think>" in moves_section.lower():
            return None

        return moves_section.split(), thinking_section

    async def rollout_and_score_eval(self, test_item):
        """
        Generate and score model responses for a single test item.

        Args:
            test_item: Test item from dataset

        Returns:
            Score (1 for correct, 0 for incorrect)
        """
        # Extract question and options from the test item
        question_text = test_item["prompt"]
        correct_answer_index = test_item["answer"]
        expected_answer_letter = test_item["ground_truth"]
        options = test_item["options"]

        # Append the answer format instruction to the prompt
        question_text_with_instruction = f'{question_text}\n\nProvide your answer by saying "The best answer is: {{Answer}}"'  # noqa E501

        # Create messages for model
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
            max_tokens=1024 * 15,
            temperature=0.5,  # Lower for eval
            split="eval",
        )

        # Extract the model's response from the completion
        model_response = completion.choices[0].text

        # Extract the answer from the model's response
        model_answer = self._extract_mcqa_answer(
            model_response, options[correct_answer_index], expected_answer_letter
        )

        # Score 1 if the answers match, 0 otherwise
        score = 1 if model_answer and model_answer == expected_answer_letter else 0

        return score

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
