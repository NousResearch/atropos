import logging
import random
import re
import time
from typing import Dict, List, Optional, Tuple, Union

from pydantic import Field
from regex_problems import PROBLEMS
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.type_definitions import Item

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are a deep thinking AI, you may use extremely long chains of thought "
    "to deeply consider the problem and deliberate with yourself via systematic "
    "reasoning processes to help come to a correct solution prior to answering. "
    "You should enclose your thoughts and internal monologue inside <think> </think> "
    "tags, and then provide your solution or response to the problem.\n\n"
    "You will be given a description of a pattern to match, along with examples of "
    "strings that should and should not match. Write a Python-compatible regular "
    "expression that matches the full string (the regex will be tested with re.fullmatch).\n\n"
    "Provide your answer inside <answer> </answer> tags, containing only the regex "
    "pattern with no delimiters, flags, or extra text. For example:\n"
    "<answer>^[a-z]+$</answer>"
)


def build_user_prompt(problem: dict) -> str:
    """Format a regex problem into a user prompt."""
    lines = [f"Description: {problem['description']}", ""]
    lines.append("Strings that SHOULD match:")
    for s in problem["positive"]:
        lines.append(f"  - {repr(s)}")
    lines.append("")
    lines.append("Strings that should NOT match:")
    for s in problem["negative"]:
        lines.append(f"  - {repr(s)}")
    return "\n".join(lines)


def extract_answer(text: str) -> Optional[str]:
    """Pull the regex pattern out of <answer>...</answer> tags."""
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def test_regex(pattern: str, positive: list, negative: list) -> dict:
    """
    Test a regex pattern against positive and negative examples.
    Returns a dict with pass counts and total score.
    """
    try:
        compiled = re.compile(pattern)
    except re.error:
        return {"score": 0.0, "valid": False, "pos_pass": 0, "neg_pass": 0}

    pos_pass = sum(1 for s in positive if compiled.fullmatch(s) is not None)
    neg_pass = sum(1 for s in negative if compiled.fullmatch(s) is None)

    total = len(positive) + len(negative)
    score = (pos_pass + neg_pass) / total if total > 0 else 0.0

    return {
        "score": score,
        "valid": True,
        "pos_pass": pos_pass,
        "neg_pass": neg_pass,
    }


class RegexEnvConfig(BaseEnvConfig):
    """Config for the regex generation environment."""

    difficulties: List[str] = Field(
        default=["easy", "medium", "hard"],
        description="Which difficulty levels to include",
    )
    score_threshold: float = Field(
        default=1.0,
        description="Minimum test pass rate to count as correct for eval metrics",
    )


class RegexEnv(BaseEnv):
    name = "regex_generation"
    env_config_cls = RegexEnvConfig

    def __init__(
        self,
        config: RegexEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.percent_correct_buffer = list()
        self.eval_metrics = list()

    @classmethod
    def config_init(cls) -> Tuple[RegexEnvConfig, List[APIServerConfig]]:
        env_config = RegexEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-3B-Preview",
            group_size=8,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=2000,
            batch_size=12,
            steps_per_eval=200,
            max_token_length=2048,
            wandb_name="regex_generation",
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

    async def setup(self):
        # Filter problems by configured difficulty levels
        all_problems = [
            p for p in PROBLEMS if p["difficulty"] in self.config.difficulties
        ]
        random.seed(42)
        random.shuffle(all_problems)

        # 80/20 train/test split
        split_idx = max(1, int(len(all_problems) * 0.8))
        self.train = all_problems[:split_idx]
        self.test = all_problems[split_idx:]

        if not self.test:
            # If too few problems, use last few from train as test
            self.test = self.train[-2:]

        self.iter = 0
        logger.info(
            f"Loaded {len(self.train)} train and {len(self.test)} test problems"
        )

    def save_checkpoint(self, step, data=None):
        if data is None:
            data = {}
        data["iter"] = self.iter
        super().save_checkpoint(step, data)

    async def get_next_item(self) -> Item:
        problem = self.train[self.iter % len(self.train)]
        self.iter += 1
        return problem

    async def collect_trajectories(
        self, item: dict
    ) -> Tuple[ScoredDataGroup, list[Item]]:
        user_content = build_user_prompt(item)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
            chat_completions = await managed.chat_completion(
                messages=messages,
                n=self.config.group_size,
                max_tokens=self.config.max_token_length,
                temperature=1.0,
            )
            state = managed.get_state()
            nodes = state["nodes"]

        to_score = []
        for i, choice in enumerate(chat_completions.choices):
            to_score.append(
                {
                    "response": choice.message.content,
                    "finish_reason": choice.finish_reason,
                    "tokens": nodes[i].tokens,
                    "masks": nodes[i].masked_tokens,
                    "logprobs": nodes[i].logprobs,
                    "positive": item["positive"],
                    "negative": item["negative"],
                }
            )

        scored = await self.score(to_score)
        return scored, []

    async def score(
        self, rollout_group_data: list
    ) -> Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]:
        scores = ScoredDataGroup()
        scores["tokens"] = []
        scores["masks"] = []
        scores["scores"] = []
        scores["inference_logprobs"] = []

        random.shuffle(rollout_group_data)

        for item in rollout_group_data:
            response = item["response"]

            # Skip truncated responses
            if item["finish_reason"] == "length":
                continue

            pattern = extract_answer(response)
            if pattern is None:
                reward = 0.0
            else:
                result = test_regex(pattern, item["positive"], item["negative"])
                reward = result["score"]

            tokens = item["tokens"]
            masks = item["masks"]
            logprobs = item["logprobs"]

            # Skip very short completions
            if len([t for t in masks if t != -100]) < 10:
                continue

            scores["tokens"].append(tokens)
            scores["masks"].append(masks)
            scores["inference_logprobs"].append(logprobs)
            scores["scores"].append(reward)

            if len(scores["tokens"]) >= self.config.group_size:
                break

        if not scores["tokens"]:
            return None

        for s in scores["scores"]:
            self.percent_correct_buffer.append(1.0 if s >= self.config.score_threshold else 0.0)

        # If all scores identical, no learning signal
        if len(set(scores["scores"])) == 1:
            return None

        return scores

    async def rollout_and_score_eval(self, problem: dict) -> dict:
        """Run a single eval rollout and score it."""
        user_content = build_user_prompt(problem)

        async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
            completion = await managed.chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                n=1,
                max_tokens=self.config.max_token_length,
                temperature=0.6,
            )
            response_content = completion.choices[0].message.content

        pattern = extract_answer(response_content)
        if pattern is None:
            test_result = {"score": 0.0, "valid": False, "pos_pass": 0, "neg_pass": 0}
        else:
            test_result = test_regex(pattern, problem["positive"], problem["negative"])

        return {
            "score": test_result["score"],
            "perfect": test_result["score"] == 1.0,
            "valid_regex": test_result.get("valid", False),
            "pattern": pattern,
            "sample": {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": response_content},
                ],
                "description": problem["description"],
                "difficulty": problem["difficulty"],
                "submitted_pattern": pattern,
                "score": test_result["score"],
                "correct": test_result["score"] == 1.0,
            },
        }

    async def evaluate(self, *args, **kwargs):
        start_time = time.time()

        eval_tasks = [self.rollout_and_score_eval(p) for p in self.test]
        results = await tqdm_asyncio.gather(*eval_tasks)

        scores = [r["score"] for r in results]
        samples = [r["sample"] for r in results]
        perfect_count = sum(1 for r in results if r["perfect"])
        valid_count = sum(1 for r in results if r["valid_regex"])

        avg_score = sum(scores) / len(scores) if scores else 0.0
        percent_perfect = perfect_count / len(results) if results else 0.0
        percent_valid = valid_count / len(results) if results else 0.0

        end_time = time.time()

        self.eval_metrics.append(("eval/avg_score", avg_score))
        self.eval_metrics.append(("eval/percent_perfect", percent_perfect))
        self.eval_metrics.append(("eval/percent_valid_regex", percent_valid))

        eval_metrics = {
            "eval/avg_score": avg_score,
            "eval/percent_perfect": percent_perfect,
            "eval/percent_valid_regex": percent_valid,
        }

        await self.evaluate_log(
            metrics=eval_metrics,
            samples=samples,
            start_time=start_time,
            end_time=end_time,
            generation_parameters={
                "temperature": 0.6,
                "max_tokens": self.config.max_token_length,
            },
        )

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}

        if self.percent_correct_buffer:
            wandb_metrics["train/percent_correct"] = sum(
                self.percent_correct_buffer
            ) / len(self.percent_correct_buffer)
        self.percent_correct_buffer = list()

        for key, value in self.eval_metrics:
            wandb_metrics[key] = value
        self.eval_metrics = list()

        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    RegexEnv.cli()
