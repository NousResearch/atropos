"""
Procedural multi-step arithmetic chains: start from an integer, apply add/sub/mul steps,
then answer the final value in \\boxed{}. Self-contained (no dataset download).
"""

import random
import time
from typing import List, Optional, Tuple, TypedDict, Union

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
    ServerBaseline,
)
from atroposlib.type_definitions import Item

system_prompt = (
    "You solve short arithmetic word problems. Think step by step if helpful, "
    "then give the final integer inside \\boxed{} with no extra text after it.\n\n"
)


class ArithmeticChainRow(TypedDict):
    question: str
    answer: str


def sample_chain(
    rng: random.Random, min_steps: int = 2, max_steps: int = 4
) -> ArithmeticChainRow:
    value = rng.randint(2, 24)
    parts = [f"You start with {value}."]
    num_steps = rng.randint(min_steps, max_steps)
    for _ in range(num_steps):
        choices = ["add", "mul"]
        if value > 2:
            choices.append("sub")
        op = rng.choice(choices)
        if op == "add":
            n = rng.randint(1, 18)
            value = value + n
            parts.append(f"Add {n}.")
        elif op == "sub":
            n = rng.randint(1, min(17, value - 1))
            value = value - n
            parts.append(f"Subtract {n}.")
        else:
            n = rng.randint(2, 9)
            value = value * n
            parts.append(f"Multiply by {n}.")
        if abs(value) > 900:
            break
    parts.append("What is the resulting integer? Answer with \\boxed{your_answer}.")
    question = " ".join(parts)
    return {"question": question, "answer": str(int(value))}


class ArithmeticChainEnv(BaseEnv):
    name = "arithmetic_chain"

    def __init__(
        self,
        config: BaseEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.percent_correct_buffer: list[float] = []
        self.eval_metrics: list[tuple[str, float]] = []
        self.train_rng = random.Random(42)
        self.eval_rng = random.Random(2025)

    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, ServerBaseline]:
        env_config = BaseEnvConfig(
            tokenizer_name="meta-llama/Llama-3.2-1B",
            group_size=8,
            use_wandb=False,
            rollout_server_url="http://localhost:8000",
            total_steps=500,
            batch_size=16,
            steps_per_eval=50,
            max_token_length=512,
            wandb_name="arithmetic_chain",
        )
        server_config = APIServerConfig(
            model_name="meta-llama/Llama-3.2-1B",
            base_url="http://localhost:8001/v1",
            api_key="x",
            num_requests_for_eval=128,
        )
        return env_config, server_config

    async def wandb_log(self, wandb_metrics: Optional[dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}
        if self.percent_correct_buffer:
            wandb_metrics["train/percent_correct"] = sum(
                self.percent_correct_buffer
            ) / len(self.percent_correct_buffer)
        self.percent_correct_buffer = []
        for key, val in self.eval_metrics:
            wandb_metrics[key] = val
        self.eval_metrics = []
        await super().wandb_log(wandb_metrics)

    async def setup(self):
        self.train = [sample_chain(self.train_rng) for _ in range(4096)]
        self.test = [sample_chain(self.eval_rng) for _ in range(64)]
        self.iter = 0

    def save_checkpoint(self, step, data=None):
        if data is None:
            data = {}
        data["iter"] = self.iter
        super().save_checkpoint(step, data)

    async def rollout_and_score_eval(self, question: str, answer: str) -> dict:
        async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
            completion = await managed.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                n=1,
                max_tokens=self.config.max_token_length,
                temperature=0.0,
                stop=(
                    [self.tokenizer.eos_token_id]
                    if self.tokenizer.eos_token_id is not None
                    else None
                ),
            )
            response_content = completion.choices[0].message.content

        gold_parsed = parse(
            "\\boxed{" + answer + "}",
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        answer_parsed = parse(
            response_content,
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
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        score = 1 if verify(answer_parsed, gold_parsed) else 0
        sample = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": response_content},
            ],
            "question": question,
            "gold_answer": answer,
            "score": int(score),
            "correct": bool(score),
            "finish_reason": completion.choices[0].finish_reason,
        }
        return {"score": score, "sample": sample}

    async def evaluate(self, *args, **kwargs):
        start_time = time.time()
        eval_tasks = [
            self.rollout_and_score_eval(item["question"], item["answer"])
            for item in self.test
        ]
        results = await tqdm_asyncio.gather(*eval_tasks)
        scores = [r["score"] for r in results]
        samples = [r["sample"] for r in results]
        percent_correct = sum(scores) / len(scores)
        end_time = time.time()
        self.eval_metrics.append(("eval/percent_correct", percent_correct))
        await self.evaluate_log(
            metrics={"eval/percent_correct": percent_correct},
            samples=samples,
            start_time=start_time,
            end_time=end_time,
            generation_parameters={
                "temperature": 0.0,
                "max_tokens": self.config.max_token_length,
            },
        )

    async def collect_trajectories(
        self, item: ArithmeticChainRow
    ) -> Tuple[ScoredDataGroup, list[Item]]:
        user_message = {"role": "user", "content": item["question"]}
        gold_answer = "\\boxed{" + item["answer"] + "}"
        stop = (
            [self.tokenizer.eos_token_id]
            if self.tokenizer.eos_token_id is not None
            else None
        )
        async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
            chat_completions = await managed.chat_completion(
                messages=[{"role": "system", "content": system_prompt}, user_message],
                n=self.config.group_size,
                max_tokens=self.config.max_token_length,
                temperature=1.0,
                stop=stop,
            )
            state = managed.get_state()
            nodes = state["nodes"]

        to_score = []
        to_backlog = []
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
                    "tokens": nodes[i].tokens,
                    "masks": nodes[i].masked_tokens,
                    "logprobs": nodes[i].logprobs,
                }
            )
        to_postprocess = await self.score(to_score)
        return to_postprocess, to_backlog

    async def score(
        self, rollout_group_data
    ) -> Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]:
        scores = ScoredDataGroup()
        scores["tokens"] = []
        scores["masks"] = []
        scores["scores"] = []
        scores["inference_logprobs"] = []
        gold_parsed = parse(
            rollout_group_data[0]["gold_answer"],
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            return None
        random.shuffle(rollout_group_data)
        for item in rollout_group_data:
            answer_parsed = parse(
                item["messages"][-1]["content"],
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
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            reward = verify(answer_parsed, gold_parsed)
            tokens = item["tokens"]
            masks = item["masks"]
            logprobs = item["logprobs"]
            if len([1 for m in masks if m != -100]) < 8:
                continue
            scores["tokens"].append(tokens)
            scores["masks"].append(masks)
            scores["inference_logprobs"].append(logprobs)
            scores["scores"].append(1.0 if reward else -1.0)
            if len(scores["tokens"]) >= self.config.group_size:
                break
        if not scores["scores"]:
            return None
        for s in scores["scores"]:
            self.percent_correct_buffer.append(max(s, 0))
        if all(s == 1 for s in scores["scores"]):
            token_lengths = [len(t) for t in scores["tokens"]]
            if not token_lengths:
                return None
            max_allowed = self.config.max_token_length
            threshold = max_allowed * 0.5
            scores["scores"] = []
            for length in token_lengths:
                if length <= threshold:
                    scores["scores"].append(1.0)
                else:
                    pct = (length - threshold) / (max_allowed - threshold)
                    pct = min(pct, 1.0)
                    scores["scores"].append(1.0 - pct)
        if len(scores["scores"]) >= 2 and all(
            scores["scores"][0] == s for s in scores["scores"]
        ):
            return None
        return scores

    async def get_next_item(self) -> ArithmeticChainRow:
        item = self.train[self.iter % len(self.train)]
        self.iter += 1
        return item


if __name__ == "__main__":
    ArithmeticChainEnv.cli()
