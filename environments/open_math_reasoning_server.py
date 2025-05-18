import random
from typing import Dict, List, Optional, Tuple, TypedDict, Union

from datasets import load_dataset
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.type_definitions import Item, number
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

system_prompt = (
    "You are a deep thinking AI specializing in advanced mathematics. "
    "You may use extremely long chains of thought "
    "to deeply consider mathematical problems and deliberate with yourself via systematic "
    "reasoning processes to help come to a correct solution prior to answering. "
    "You should enclose your thoughts and internal monologue inside <think> </think> "
    "tags, and then provide your solution or response to the problem.\n\n"
)

system_prompt += """You are allocated a maximum of 2048 tokens, please strive to use less.

You should use proper LaTeX notation when writing mathematical expressions and formulas.
For example, use \\frac{a}{b} for fractions, \\sqrt{x} for square roots, and ^ for exponents.

You will then provide your final answer like this: \\boxed{your answer here}
It is important that you provide your answer in the correct LaTeX format.
If you do not, you will not receive credit for your answer.
So please end your answer with \\boxed{your answer here}"""


class OpenMathReasoningRow(TypedDict):
    problem: str
    expected_answer: str
    problem_type: Optional[str]
    problem_source: Optional[str]
    generation_model: Optional[str]
    pass_rate_72b_tir: Optional[float]


class OpenMathReasoningEnv(BaseEnv):

    name = "open_math_reasoning"

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
        # Add tracking for wandb visualizations
        self.rollouts_for_wandb = []
        self.completion_lengths = []

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
            max_token_length=2048,
            wandb_name="try-nv",
        )
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-3B-Preview",
                base_url="http://localhost:9001/v1",
                api_key="x",
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

        self.percent_correct_buffer = list()
        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = list()
        # Call the parent method to handle the server metrics
        await super().wandb_log(wandb_metrics)

    async def setup(self):
        # Load OpenMathReasoning dataset - only 'cot' split is available
        full_dataset = load_dataset("nvidia/OpenMathReasoning", split="cot").shuffle(seed=42)
        
        # Create our own train/test split in 98:2 ratio
        dataset_size = len(full_dataset)
        test_size = int(dataset_size * 0.02)  # 2% for test
        train_size = dataset_size - test_size  # 98% for train
        
        # Split the dataset
        self.train = full_dataset.select(range(train_size))
        test_data = full_dataset.select(range(train_size, dataset_size))
        
        # Process test data
        self.test = list()
        for item in test_data:
            self.test.append(
                {
                    "problem": item["problem"],
                    "gold_answer": item["expected_answer"],
                    "problem_type": item.get("problem_type", None),
                    "problem_source": item.get("problem_source", None),
                }
            )
        self.iter = 0

    def save_checkpoint(self, step, data=None):
        if data is None:
            data = {}
        data["iter"] = self.iter
        super().save_checkpoint(step, data)

    async def rollout_and_score_eval(self, problem: str, answer: str) -> number:
        completion = await self.server.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": problem},
            ],
            n=1,
            max_tokens=self.config.max_token_length,
            temperature=0.0,
            split="eval",
        )
        gold_parsed = parse(
            answer,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        answer_parsed = parse(
            completion.choices[0].message.content.split("</think>")[-1],
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed="all",
                        units=True,
                    ),
                    # Ensures that boxed is tried first
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        score = 1 if verify(answer_parsed, gold_parsed) else 0
        return score

    async def evaluate(self, *args, **kwargs):
        eval_tasks = []
        for item in self.test:
            eval_tasks.append(
                self.rollout_and_score_eval(item["problem"], item["gold_answer"])
            )
        scores = await tqdm_asyncio.gather(*eval_tasks)
        self.eval_metrics.append(("eval/percent_correct", sum(scores) / len(scores)))

    async def collect_trajectories(
        self, item: OpenMathReasoningRow
    ) -> Tuple[ScoredDataGroup, list[Item]]:
        user_message = {"role": "user", "content": item["problem"]}
        gold_answer = item["expected_answer"]

        chat_completions = await self.server.chat_completion(
            messages=[{"role": "system", "content": system_prompt}, user_message],
            n=self.config.group_size,
            max_tokens=self.config.max_token_length,
        )
        to_score = list()
        to_backlog = list()
        for i, chat_completion in enumerate(chat_completions.choices):
            messages = (
                {"role": "system", "content": system_prompt},
                user_message,
                {"role": "assistant", "content": chat_completion.message.content},
            )
            to_score.append(
                {
                    "messages": messages,
                    "gold_answer": gold_answer,
                    "finish_reason": chat_completion.finish_reason,
                }
            )
        to_postprocess = await self.score(to_score)
        return to_postprocess, to_backlog

    async def score(
        self, rollout_group_data
    ) -> Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]:
        scores = ScoredDataGroup()
        scores["tokens"] = list()
        scores["masks"] = list()
        scores["scores"] = list()
        gold_parsed = parse(
            rollout_group_data[0]["gold_answer"],
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            random.shuffle(rollout_group_data)
            for item in rollout_group_data:
                answer_parsed = parse(
                    item["messages"][-1]["content"].split("</think>")[-1],
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                equations=True,
                                boxed="all",
                                units=True,
                            ),
                            # Ensures that boxed is tried first
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode="first_match",
                )
                # Reward 1 if the content is the same as the ground truth, 0 otherwise
                reward = verify(answer_parsed, gold_parsed)
                out_dict = tokenize_for_trainer(
                    self.tokenizer, item["messages"], item["finish_reason"]
                )
                tokens = out_dict["tokens"]
                masks = out_dict["masks"]
                # remove obviously bad examples
                if len([1 for i in masks if i != -100]) < 10:
                    continue
                scores["tokens"].append(tokens)
                scores["masks"].append(masks)
                scores["scores"].append(1.0 if reward else -1.0)
                if len(scores["tokens"]) >= self.config.group_size:
                    break
            for score in scores["scores"]:
                self.percent_correct_buffer.append(max(score, 0))
            # check if all the same
            if all([score == 1 for score in scores["scores"]]):
                # Do length penalty :)
                token_lengths = [len(token) for token in scores["tokens"]]
                if max(token_lengths) == 0:
                    # What? But don't want to crash a run so just in case...
                    return None

                # Get max allowed token length from config
                max_allowed_length = self.config.max_token_length
                # Set threshold at 50% of max_token_length - no penalty below this
                length_threshold = max_allowed_length * 0.5

                # Apply modified length penalty with threshold
                scores["scores"] = []
                for length in token_lengths:
                    if length <= length_threshold:
                        # No penalty for responses under threshold
                        scores["scores"].append(1.0)
                    else:
                        # Calculate how far we are between threshold and max as a percentage
                        percentage_of_range = (length - length_threshold) / (
                            max_allowed_length - length_threshold
                        )
                        # Cap at 1.0 in case length exceeds max_allowed_length
                        percentage_of_range = min(percentage_of_range, 1.0)
                        # Apply linear penalty scaling from 1.0 down to 0.0
                        scores["scores"].append(1.0 - percentage_of_range)
            if all([scores["scores"][0] == score for score in scores["scores"]]):
                return None  # If all the same, we return None
            return scores
        else:
            # If the gold solution is not parseable, we return None
            return None

    async def get_next_item(self) -> OpenMathReasoningRow:
        next_item = self.train[self.iter % len(self.train)]
        self.iter += 1
        return next_item


if __name__ == "__main__":
    OpenMathReasoningEnv.cli() 