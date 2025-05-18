import math
import random
import string
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.type_definitions import Item
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

system_prompt = (
    "You are a deep thinking AI, you may use extremely long chains of thought "
    "to deeply consider the problem and deliberate with yourself via systematic "
    "reasoning processes to help come to a correct solution prior to answering. "
    "You should enclose your thoughts and internal monologue inside <think> </think> "
    "tags, and then provide your solution or response to the problem.\n\n"
)

system_prompt += """You are allocated a maximum of 2048 tokens, please strive to use less.

You will then provide your answer like this: \\boxed{your answer here}
It is important that you provide your answer in the correct format.
If you do not, you will not receive credit for your answer.
So please end your answer with \\boxed{your answer here}"""

DIFFICULTY_LEVELS = ["easy", "medium", "hard"]


def create_example_grid(
    difficulty_level: str,
    out_of_distribution: bool,
    seed: int,
    include_q_prompt: bool = True,
) -> Tuple[str, str]:
    """
    Create an example of based on the difficulty level and out of distribution status.

    If out of distribution, the resulting box will be using alphabetical characters.
    If in distribution, the resulting box will be using numerical characters.

    Ex: Easy, In Distribution:
    #####
    #1@ #
    #   #
    # 2 #
    #####
    Question: Where is 2 in relation to you?
    Answer: \\boxed{(0, 2)}
    Question: What is the Manhattan distance between you and 2?
    Answer: \\boxed{2}
    Question: What is the sum of the coordinates of you and 2?
    Answer: \\boxed{(0, 2)}
    Question: What is the product of the coordinates of you and 2?
    Answer: \\boxed{(0, 1)}
    Question: What is the (nearest integer) Euclidean distance between you and 2?
    Answer: \\boxed{2}

    Args:
        difficulty_level (str): The difficulty level of the question.
                                Easy - Box shape of 3x3 or 5x5
                                       Distractors - 1-2 distractors
                                Medium - Box shape of 5x5 or 7x7
                                       Distractors - 2-4 distractors
                                Hard - Box shape of 7x7 or 9x9
                                       Distractors - 4-8 distractors
        out_of_distribution (bool): Whether the question is out of distribution.
    """
    ex_string = """Ex: Easy, In Distribution:
#####
#1@ #
#   #
# 2 #
#####
Question: Where is 2 in relation to you?
Answer: \\boxed{(0, 2)}
Question: What is the Manhattan distance between you and 2?
Answer: \\boxed{2}
Question: What is the sum of the coordinates of you and 2?
Answer: \\boxed{(0, 2)}
Question: What is the product of the coordinates of you and 2?
Answer: \\boxed{(0, 1)}
Question: What is the (nearest integer) Euclidean distance between you and 2?
Answer: \\boxed{2}"""
    prng = random.Random(seed)
    box_size = 0
    num_distractors = 0
    if difficulty_level not in DIFFICULTY_LEVELS:
        raise ValueError(f"Invalid difficulty level: {difficulty_level}")
    if difficulty_level == "easy":
        box_size = prng.choice([3, 5])
        num_distractors = prng.randint(1, 2)
    elif difficulty_level == "medium":
        box_size = prng.choice([5, 7])
        num_distractors = prng.randint(2, 4)
    elif difficulty_level == "hard":
        box_size = prng.choice([7, 9])
        num_distractors = prng.randint(4, 8)

    if out_of_distribution:
        # Generate random characters for the box
        box_chars = prng.sample(string.ascii_letters, num_distractors)
    else:
        box_chars = prng.sample(string.digits, num_distractors)
    total_selections = []
    for i in range(box_size):
        total_selections.extend([(i, j) for j in range(box_size)])
    selected_locations = prng.sample(total_selections, num_distractors + 1)
    player_pos = selected_locations[0]
    distractors = selected_locations[1:]
    out_map = ""
    for i in range(box_size + 2):
        for j in range(box_size + 2):
            map_i = i - 1
            map_j = j - 1
            if (map_i < 0) or (map_i >= box_size) or (map_j < 0) or (map_j >= box_size):
                out_map += "#"
            elif (map_j, map_i) == player_pos:
                out_map += "@"
            elif (map_j, map_i) in distractors:
                out_map += box_chars[distractors.index((map_j, map_i))]
            else:
                out_map += " "
        out_map += "\n"
    potential_questions = [
        "What (x,y) coordinates are you at?",
        "What (x,y) coordinates is {x} at?",
        "Where is {x} in relation to you?",
        "What is the Manhattan distance between you and {x}?",
        "What is the (nearest integer) Euclidean distance between you and {x}?",
        "What is the sum of the coordinates of you and {x}? Answer in (x,y) format.",
        "What is the product of the coordinates of you and {x}? Answer in (x,y) format.",
    ]
    question_idx = prng.choice(list(range(len(potential_questions))))
    question = potential_questions[question_idx]
    if "{x}" in question:
        x_idx = prng.choice(list(range(len(box_chars))))
        question = question.replace("{x}", box_chars[x_idx])
    # Calculate the answer
    if question_idx == 0:
        answer = f"\\boxed{{{(player_pos[0], player_pos[1])}}}"
    elif question_idx == 1:
        answer = f"\\boxed{{{(distractors[x_idx])}}}"
    elif question_idx == 2:
        answer = f"\\boxed{{{(distractors[x_idx][0] - player_pos[0], distractors[x_idx][1] - player_pos[1])}}}"
    elif question_idx == 3:
        val = abs(player_pos[0] - distractors[x_idx][0]) + abs(
            player_pos[1] - distractors[x_idx][1]
        )
        answer = f"\\boxed{{{val}}})"
    elif question_idx == 4:
        val = round(
            math.sqrt(
                (player_pos[0] - distractors[x_idx][0]) ** 2
                + (player_pos[1] - distractors[x_idx][1]) ** 2
            )
        )
        answer = f"\\boxed{{{val}}})"
    elif question_idx == 5:
        answer = f"\\boxed{{{(player_pos[0] + distractors[x_idx][0], player_pos[1] + distractors[x_idx][1])}}})"
    elif question_idx == 6:
        answer = f"\\boxed{{{(player_pos[0] * distractors[x_idx][0], player_pos[1] * distractors[x_idx][1])}}}"
    if include_q_prompt:
        q_prompt = (
            ex_string
            + "\n\n"
            + "Below is a map, where you are marked with @, you will be asked a question about the map.\n"
            f"The coordinates are assumed to be 0 indexed, "
            f"with the top left corner is (0,0), the top right corner is ({box_size-1},0), "
            f"the bottom left corner is (0,{box_size-1}), and the bottom right corner is ({box_size-1},{box_size-1}).\n"
            "The # symbol represents a wall, and you should not consider them as taking a coordinate for your answer.\n"
            f"The map is {box_size}x{box_size} in size.\n"
            "When you have an answer, please return it in the format: \\boxed{your answer here}\n"
            "For coordinates, please return them in the format: \\boxed{(x,y)}\n"
            "For distance, please return the nearest integer in the format \\boxed{nearest integer}.\n"
            "For the sum and product, please return them in the format: \\boxed{(x,y)}\n\n"
        )
    return_q = ""
    if include_q_prompt:
        return_q = q_prompt
    return_q += (
        f"Your position is ({player_pos[0]},{player_pos[1]}).\n"
        "Map:\n"
        f"{out_map}"
        "Question:\n"
        f"{question}"
    )
    return return_q, answer


class NumberGridEnv(BaseEnv):

    name = "number_grid"

    def __init__(
        self,
        config: BaseEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.percent_correct_buffer = list()
        self.eval_metrics = list()
        self.easy_correct_buffer = list()
        self.medium_correct_buffer = list()
        self.hard_correct_buffer = list()
        self.total_in_buffers = 100

    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
        env_config = BaseEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-3B-Preview",
            group_size=8,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=12,
            steps_per_eval=100,
            max_token_length=256,
            wandb_name="number_grid",
        )
        server_configs = [
            APIServerConfig(
                model_name="gpt-4.1-mini",
                num_requests_for_eval=256,
            ),
        ]

        return env_config, server_configs

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
        if len(self.easy_correct_buffer) > 0:
            wandb_metrics["train/easy_correct"] = sum(self.easy_correct_buffer) / len(
                self.easy_correct_buffer
            )
        if len(self.medium_correct_buffer) > 0:
            wandb_metrics["train/medium_correct"] = sum(
                self.medium_correct_buffer
            ) / len(self.medium_correct_buffer)
        if len(self.hard_correct_buffer) > 0:
            wandb_metrics["train/hard_correct"] = sum(self.hard_correct_buffer) / len(
                self.hard_correct_buffer
            )
        self.percent_correct_buffer = list()
        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = list()
        # Call the parent method to handle the server metrics
        await super().wandb_log(wandb_metrics)

    async def setup(self):
        self.train = load_dataset("gsm8k", "main", split="train").shuffle(seed=42)
        test_data = load_dataset("gsm8k", "main", split="test").shuffle(seed=42)
        self.test = list()
        for item in test_data:
            self.test.append(
                {
                    "question": item["question"],
                    "gold_answer": item["answer"]
                    .split("#")[-1]
                    .strip()
                    .replace(",", ""),
                }
            )
        self.iter = 0

    def save_checkpoint(self, step, data=None):
        if data is None:
            data = {}
        data["iter"] = self.iter
        super().save_checkpoint(step, data)

    async def rollout_eval(self, seed, difficulty_level, out_of_distribution):
        question, answer = create_example_grid(
            difficulty_level, out_of_distribution, seed
        )
        resp = await self.server.chat_completion(
            messages=[
                {"role": "user", "content": question},
            ],
            n=1,
            max_tokens=self.config.max_token_length,
            temperature=0.0,
            split="eval",
        )
        resp_ans = resp.choices[0].message.content.split("\\boxed{")[-1].split("}")[0]
        gold_ans = answer.split("\\boxed{")[-1].split("}")[0]
        return {
            "correct": resp_ans.replace(" ", "") == gold_ans.replace(" ", ""),
            "difficulty_level": difficulty_level,
            "out_of_distribution": out_of_distribution,
        }

    async def evaluate(self, *args, **kwargs):
        tasks = []
        for i in range(80):
            if i >= 60:
                difficulty_level = "medium"
                out_of_distribution = True
            elif i >= 40:
                difficulty_level = "hard"
                out_of_distribution = False
            elif i >= 20:
                difficulty_level = "medium"
                out_of_distribution = False
            else:
                difficulty_level = "easy"
                out_of_distribution = False
            tasks.append(self.rollout_eval(i, difficulty_level, out_of_distribution))
        results = await tqdm_asyncio.gather(*tasks)
        eval_metrics = {
            "easy": {"correct": 0, "total": 0},
            "medium": {"correct": 0, "total": 0},
            "hard": {"correct": 0, "total": 0},
            "ood": {"correct": 0, "total": 0},
        }
        for result in results:
            if result["out_of_distribution"]:
                eval_metrics["ood"]["total"] += 1
                if result["correct"]:
                    eval_metrics["ood"]["correct"] += 1
            else:
                eval_metrics[result["difficulty_level"]]["total"] += 1
                if result["correct"]:
                    eval_metrics[result["difficulty_level"]]["correct"] += 1
        self.eval_metrics = [
            (key, val["correct"] / val["total"]) for key, val in eval_metrics.items()
        ]

    async def collect_trajectories(self, item) -> Tuple[ScoredDataGroup, list[Item]]:
        rng = random.Random(item)
        difficulty_level = rng.choice(DIFFICULTY_LEVELS)
        question, answer = create_example_grid(difficulty_level, False, item)
        resp = await self.server.chat_completion(
            messages=[
                {"role": "user", "content": question},
            ],
            n=self.config.group_size,
            max_tokens=self.config.max_token_length,
            temperature=1.0,
            top_p=0.95,
        )
        gold_ans = answer.split("\\boxed{")[-1].split("}")[0]
        to_postprocess = ScoredDataGroup()
        to_postprocess["tokens"] = []
        to_postprocess["messages"] = []
        to_postprocess["masks"] = []
        to_postprocess["scores"] = []
        to_backlog = list()
        for choice in resp.choices:
            try:
                resp_ans = choice.message.content.split("\\boxed{")[1].split("}")[0]
            except IndexError:
                resp_ans = ""
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": choice.message.content},
            ]
            to_postprocess["messages"].append(messages)
            postprocessed_messages = tokenize_for_trainer(self.tokenizer, messages)
            to_postprocess["tokens"].append(postprocessed_messages["tokens"])
            to_postprocess["masks"].append(postprocessed_messages["masks"])
            to_postprocess["scores"].append(
                resp_ans.replace(" ", "") == gold_ans.replace(" ", "")
            )
        if difficulty_level == "easy":
            self.easy_correct_buffer.append(
                sum(to_postprocess["scores"]) / len(to_postprocess["scores"])
            )
        elif difficulty_level == "medium":
            self.medium_correct_buffer.append(
                sum(to_postprocess["scores"]) / len(to_postprocess["scores"])
            )
        elif difficulty_level == "hard":
            self.hard_correct_buffer.append(
                sum(to_postprocess["scores"]) / len(to_postprocess["scores"])
            )
        return to_postprocess, to_backlog

    async def get_next_item(self):
        self.iter += 1
        return self.iter


if __name__ == "__main__":
    NumberGridEnv.cli()
