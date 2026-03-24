import asyncio
import os
import random
import re
import time
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset
from pydantic import Field

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
    Item,
    ScoredDataGroup,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

DEFAULT_THINKING_PROMPT = (
    "You are a deep thinking AI, you may use extremely long chains of thought to deeply "
    "consider the problem and deliberate with yourself via systematic reasoning processes "
    "to help come to a correct solution prior to answering. You should enclose your "
    "thoughts and internal monologue inside <think> </think> tags, and then provide your "
    "solution or response to the problem."
)

DEFAULT_SYSTEM_PROMPT = (
    "You are answering questions from a set of tasks that the base model historically gets "
    "wrong. Do not guess. If you do not know the answer with high confidence, abstain by "
    "responding with exactly <answer>I don't know</answer>. If you do know the answer, "
    "respond with your final answer inside <answer></answer> tags."
)

ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
THINK_CONTENT_AFTER_PATTERN = re.compile(r"</think>\s*(.*)", re.DOTALL | re.IGNORECASE)


class SocraticConfig(BaseEnvConfig):
    train_dataset: str = Field(
        default="",
        description="Hugging Face dataset name or local file path for training data.",
    )
    eval_dataset: Optional[str] = Field(
        default=None,
        description="Optional Hugging Face dataset name or local file path for eval data.",
    )
    train_split: str = Field(
        default="train",
        description="Split to use for training when loading from Hugging Face.",
    )
    eval_split: str = Field(
        default="train[:100]",
        description="Split to use for evaluation when loading from Hugging Face.",
    )
    question_field: str = Field(
        default="question",
        description="Field containing the prompt/question.",
    )
    answer_field: str = Field(
        default="answer",
        description="Field containing the golden/canonical answer to match against.",
    )
    custom_system_prompt: Optional[str] = Field(
        default=None,
        description="Optional system prompt override.",
    )
    rollout_temperature: float = Field(
        default=1.0,
        description="Temperature used for training rollouts.",
    )
    eval_temperature: float = Field(
        default=0.0,
        description="Temperature used for evaluation.",
    )
    train_max_tokens: int = Field(
        default=512,
        description="Maximum new tokens for training rollouts.",
    )
    eval_max_tokens: int = Field(
        default=512,
        description="Maximum new tokens for evaluation.",
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        description="Maximum retries for generation calls.",
    )
    retry_delay: float = Field(
        default=1.0,
        ge=0.0,
        description="Delay in seconds between retries.",
    )
    min_response_length: int = Field(
        default=0,
        ge=0,
        description="Minimum raw response length required to keep a completion.",
    )
    abstain_phrases: List[str] = Field(
        default_factory=lambda: [
            "i don't know",
            "i do not know",
            "unknown",
            "not sure",
            "i'm not sure",
            "i am not sure",
            "cannot determine",
            "can't determine",
            "insufficient information",
            "no answer",
        ],
        description="Phrases that count as abstention.",
    )
    reward_for_abstain: float = Field(
        default=1.0,
        description="Reward for abstaining instead of guessing.",
    )
    reward_for_correct: float = Field(
        default=0.25,
        description="Reward for a correct answer. Lower than abstain by default on this unknown-only dataset.",
    )
    penalty_for_wrong: float = Field(
        default=-1.0,
        description="Penalty for a wrong answer.",
    )


class SocraticEnv(BaseEnv):
    """
    Train an LLM to abstain on tasks that a base model historically misses.

    This environment expects a dataset with a question field and a golden-answer field.
    By default it rewards abstention more than correctness and heavily penalizes wrong
    answers. The defaults assume the dataset was pre-filtered to questions the base
    model does not reliably know.
    """

    name = "socratic"
    env_config_cls = SocraticConfig

    def __init__(
        self,
        config: SocraticConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: SocraticConfig = config
        self.iter = 0
        self.train = []
        self.test = []
        self.percent_correct_buffer: List[float] = []
        self.abstain_buffer = 0
        self.correct_buffer = 0
        self.wrong_buffer = 0

        self._normalized_abstain_phrases = {
            self._normalize_text(phrase) for phrase in self.config.abstain_phrases
        }

    @classmethod
    def config_init(cls) -> Tuple[SocraticConfig, List[APIServerConfig]]:
        env_config = SocraticConfig(
            tokenizer_name="NousResearch/Hermes-3-Llama-3.1-8B",
            group_size=8,
            use_wandb=True,
            max_num_workers_per_node=8,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=1024,
            steps_per_eval=25,
            wandb_name="socratic",
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.1,
            min_batch_allocation=0.1,
        )
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/Hermes-3-Llama-3.1-8B",
                base_url="http://localhost:9001/v1",
                api_key=os.getenv("OPENAI_API_KEY", "x"),
            )
        ]
        return env_config, server_configs

    def _normalize_text(self, text: Optional[str]) -> str:
        if text is None:
            return ""
        normalized = re.sub(r"\s+", " ", text).strip()
        normalized = normalized.strip(" \t\n\r\"'`.,;:!?()[]{}")
        return normalized.casefold()

    def _load_dataset(self, dataset_path: str, split: str) -> List[Dict]:
        if not dataset_path:
            raise ValueError(
                "train_dataset must be set to a Hugging Face dataset name or local file path."
            )

        if os.path.exists(dataset_path):
            if dataset_path.endswith((".jsonl", ".json")):
                dataset = load_dataset("json", data_files=dataset_path, split=split or "train")
            elif dataset_path.endswith(".csv"):
                dataset = load_dataset("csv", data_files=dataset_path, split=split or "train")
            elif dataset_path.endswith(".parquet"):
                dataset = load_dataset(
                    "parquet", data_files=dataset_path, split=split or "train"
                )
            else:
                dataset = load_dataset("json", data_files=dataset_path, split=split or "train")
        else:
            dataset = load_dataset(dataset_path, split=split)
        return dataset

    def _create_system_content(self) -> str:
        base_prompt = self.config.custom_system_prompt or DEFAULT_SYSTEM_PROMPT
        if self.config.thinking_mode:
            thinking_prompt = self.config.custom_thinking_prompt or DEFAULT_THINKING_PROMPT
            return f"{thinking_prompt}\n\n{base_prompt}"
        return base_prompt

    async def _chat_completion_with_retries(self, messages: List[Dict], **kwargs):
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                return await self.server.chat_completion(messages=messages, **kwargs)
            except Exception as exc:
                last_error = exc
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(self.config.retry_delay)
        raise last_error

    def _convert_messages_to_list(self, prompt_tuple: Tuple) -> List[Dict]:
        return [dict(role_dict) for role_dict in prompt_tuple]

    def _extract_scorable_response(self, response: str) -> str:
        if not response:
            return ""
        think_match = THINK_CONTENT_AFTER_PATTERN.search(response)
        if think_match:
            return think_match.group(1).strip()
        return response.strip()

    def _extract_final_answer(self, response: str) -> str:
        scorable_response = self._extract_scorable_response(response)
        match = ANSWER_TAG_PATTERN.search(scorable_response)
        if match:
            return match.group(1).strip()
        return scorable_response.strip()

    def _is_abstention(self, extracted_answer: str) -> bool:
        normalized = self._normalize_text(extracted_answer)
        if not normalized:
            return True

        for phrase in self._normalized_abstain_phrases:
            if normalized == phrase or normalized.startswith(f"{phrase} "):
                return True
        return False

    def _matches_golden_answer(self, extracted_answer: str, golden_answer: str) -> bool:
        normalized_answer = self._normalize_text(extracted_answer)
        normalized_golden = self._normalize_text(golden_answer)

        if not normalized_answer or not normalized_golden:
            return False

        return normalized_golden in normalized_answer

    def _score_answer(self, extracted_answer: str, golden_answer: str) -> Tuple[float, str]:
        if self._is_abstention(extracted_answer):
            return self.config.reward_for_abstain, "abstain"

        if self._matches_golden_answer(extracted_answer, golden_answer):
            return self.config.reward_for_correct, "correct"

        return self.config.penalty_for_wrong, "wrong"

    async def setup(self) -> None:
        self.train = self._load_dataset(self.config.train_dataset, self.config.train_split)
        self.train = self.train.shuffle(seed=42) if hasattr(self.train, "shuffle") else self.train

        eval_dataset = self.config.eval_dataset or self.config.train_dataset
        self.test = self._load_dataset(eval_dataset, self.config.eval_split)
        self.iter = 0

    async def get_next_item(self) -> Item:
        if len(self.train) == 0:
            await self.setup()

        example = self.train[self.iter % len(self.train)]
        self.iter += 1

        question = str(example[self.config.question_field]).strip()
        golden_answer = str(example[self.config.answer_field]).strip()

        prompt = tuple(
            [
                frozenset(
                    {"role": "system", "content": self._create_system_content()}.items()
                ),
                frozenset({"role": "user", "content": question}.items()),
            ]
        )
        return (prompt, golden_answer, question)

    async def collect_trajectories(self, item: Item) -> Tuple[Optional[ScoredDataGroup], List]:
        messages = self._convert_messages_to_list(item[0])

        try:
            completions = await self._chat_completion_with_retries(
                messages,
                    n=self.config.group_size,
                    max_tokens=self.config.train_max_tokens,
                    temperature=self.config.rollout_temperature,
                )
            if not completions.choices:
                return None, []

            to_score = []
            for choice in completions.choices:
                content = choice.message.content or ""
                if len(content.strip()) < self.config.min_response_length:
                    continue
                trajectory_messages = messages + [{"role": "assistant", "content": content}]
                to_score.append((tuple(trajectory_messages), item[1], item[2]))

            if to_score:
                scored_data = await self.score(to_score)
                return scored_data, []
        except Exception:
            return None, []

        return None, []

    async def score(self, rollout_group_data: List[Tuple]) -> Optional[ScoredDataGroup]:
        if not rollout_group_data:
            return None

        scores = ScoredDataGroup()
        scores["tokens"] = []
        scores["masks"] = []
        scores["scores"] = []

        random.shuffle(rollout_group_data)

        for trajectory_messages, golden_answer, _question in rollout_group_data:
            model_response = trajectory_messages[-1]["content"]
            extracted_answer = self._extract_final_answer(model_response)
            reward, label = self._score_answer(extracted_answer, golden_answer)

            out_dict = tokenize_for_trainer(self.tokenizer, trajectory_messages)
            tokens = out_dict["tokens"]
            masks = out_dict["masks"]

            if len([mask for mask in masks if mask != -100]) < 2:
                continue

            scores["tokens"].append(tokens)
            scores["masks"].append(masks)
            scores["scores"].append(reward)

            self.percent_correct_buffer.append(reward)
            if label == "abstain":
                self.abstain_buffer += 1
            elif label == "correct":
                self.correct_buffer += 1
            else:
                self.wrong_buffer += 1

            if len(scores["tokens"]) >= self.config.group_size:
                break

        if not scores["tokens"]:
            return None

        if self.config.ensure_scores_are_not_same and len(set(scores["scores"])) == 1:
            return None

        return scores

    async def _score_eval_item(self, eval_item: Dict) -> Dict:
        question = str(eval_item[self.config.question_field]).strip()
        golden_answer = str(eval_item[self.config.answer_field]).strip()
        messages = [
            {"role": "system", "content": self._create_system_content()},
            {"role": "user", "content": question},
        ]

        completion = await self._chat_completion_with_retries(
            messages,
            n=1,
            max_tokens=self.config.eval_max_tokens,
            temperature=self.config.eval_temperature,
            split="eval",
        )

        if not completion.choices:
            return {"score": None, "sample": None}

        raw_response = completion.choices[0].message.content or ""
        extracted_answer = self._extract_final_answer(raw_response)
        reward, label = self._score_answer(extracted_answer, golden_answer)

        sample = {
            "question": question,
            "golden_answer": golden_answer,
            "raw_response": raw_response,
            "extracted_answer": extracted_answer,
            "label": label,
            "score": reward,
            "finish_reason": completion.choices[0].finish_reason,
        }
        return {"score": reward, "label": label, "sample": sample}

    async def evaluate(self, *args, **kwargs) -> None:
        if len(self.test) == 0:
            await self.setup()

        start_time = time.time()
        eval_items = list(self.test)
        semaphore = asyncio.Semaphore(self.config.max_eval_workers)

        async def _bounded_score(item: Dict) -> Dict:
            async with semaphore:
                return await self._score_eval_item(item)

        tasks = [_bounded_score(item) for item in eval_items]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_results = [
            result
            for result in results
            if not isinstance(result, Exception)
            and result
            and result.get("score") is not None
        ]

        if not valid_results:
            return

        scores = [result["score"] for result in valid_results]
        labels = [result["label"] for result in valid_results]
        samples = [result["sample"] for result in valid_results]

        total = len(valid_results)
        metrics = {
            "eval/mean_score": sum(scores) / total,
            "eval/abstention_rate": labels.count("abstain") / total,
            "eval/correct_rate": labels.count("correct") / total,
            "eval/wrong_rate": labels.count("wrong") / total,
            "eval/total_samples": total,
        }

        await self.evaluate_log(
            metrics=metrics,
            samples=samples,
            start_time=start_time,
            end_time=time.time(),
            generation_parameters={
                "temperature": self.config.eval_temperature,
                "max_tokens": self.config.eval_max_tokens,
                "reward_for_abstain": self.config.reward_for_abstain,
                "reward_for_correct": self.config.reward_for_correct,
                "penalty_for_wrong": self.config.penalty_for_wrong,
            },
        )

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}

        total = self.abstain_buffer + self.correct_buffer + self.wrong_buffer
        if self.percent_correct_buffer:
            wandb_metrics["train/mean_score"] = sum(self.percent_correct_buffer) / len(
                self.percent_correct_buffer
            )
        if total > 0:
            wandb_metrics["train/abstention_rate"] = self.abstain_buffer / total
            wandb_metrics["train/correct_rate"] = self.correct_buffer / total
            wandb_metrics["train/wrong_rate"] = self.wrong_buffer / total

        self.percent_correct_buffer = []
        self.abstain_buffer = 0
        self.correct_buffer = 0
        self.wrong_buffer = 0

        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    SocraticEnv.cli()
