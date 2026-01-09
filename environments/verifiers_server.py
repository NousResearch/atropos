# To install a Verifiers/Prime environment:
# 1. uv tool install prime
# 2. prime login
# 3. prime env install will/wordle (or any owner/environment)
# Docs: https://docs.primeintellect.ai/tutorials-environments/install

import os
import time
from typing import List, Tuple

import verifiers as vf
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)


class VfEnvConfig(BaseEnvConfig):
    vf_env_name: str = ""
    env_args: dict = {}


class VerifiersEnv(BaseEnv):
    name = "verifiers"
    env_config_cls = VfEnvConfig  # type: ignore[assignment]

    def __init__(
        self,
        config: VfEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=False,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.eval_metrics = list()
        self.vf_env = vf.load_environment(config.vf_env_name, **config.env_args)
        self.rubric = self.vf_env.rubric

        self.parser = self.rubric.parser
        self.reward_funcs = self.rubric.funcs
        self.reward_weights = self.rubric.weights
        self.reward_scales = [
            weight / sum(self.reward_weights) for weight in self.reward_weights
        ]
        self.system_prompt = self.vf_env.system_prompt

    @classmethod
    def config_init(cls) -> Tuple[VfEnvConfig, List[APIServerConfig]]:
        env_config = VfEnvConfig(
            group_size=8,
            use_wandb=False,
            rollout_server_url="http://localhost:8000",
            total_steps=10,
            batch_size=4,
            steps_per_eval=1,
            max_token_length=2048,
            wandb_name="verifiers",
        )
        server_configs = [
            APIServerConfig(
                model_name="gpt-4.1-nano",
                base_url=None,
                api_key=os.getenv("OPENAI_API_KEY"),
                num_requests_for_eval=4,
            ),
        ]
        return env_config, server_configs

    async def setup(self):
        self.train = self.vf_env.get_dataset()
        test_data = self.vf_env.get_eval_dataset()
        self.test = test_data.select_columns(["question", "answer"]).to_list()
        self.iter = 0

    async def rollout_and_score_eval(
        self, question: str, answer: str, **kwargs
    ) -> dict:
        system_prompt = kwargs.get("system_prompt")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        completion = await self.server.chat_completion(
            messages=messages,
            n=1,
            max_tokens=self.config.max_token_length,
            temperature=0.0,
        )

        response_content = completion.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": response_content})

        answer_parsed = self.parser.parse_answer(completion=response_content)

        rewards = []
        for func in self.reward_funcs:
            reward = func(
                parser=self.parser,
                completion=messages,
                answer=answer,
            )
            rewards.append(reward)
        weighted_rewards = [r * self.reward_scales[j] for j, r in enumerate(rewards)]

        score = sum(weighted_rewards)

        sample = {
            "messages": messages,
            "question": question,
            "gold_answer": answer,
            "model_parsed": str(answer_parsed) if answer_parsed else None,
            "score": int(score),
            "correct": bool(score),
            "finish_reason": completion.choices[0].finish_reason,
        }

        return {"score": score, "sample": sample}

    async def evaluate(self, *args, **kwargs):
        start_time = time.time()

        eval_tasks = []
        for item in self.test:
            eval_tasks.append(
                self.rollout_and_score_eval(
                    item["question"], item["answer"], system_prompt=self.system_prompt
                )
            )
        results = await tqdm_asyncio.gather(*eval_tasks)

        scores = [result["score"] for result in results]
        samples = [result["sample"] for result in results]

        avg_total_score = sum(scores) / len(scores)

        end_time = time.time()

        self.eval_metrics.append(("eval/avg_total_score", avg_total_score))

        eval_metrics = {"eval/avg_total_score": avg_total_score}

        await self.evaluate_log(
            metrics=eval_metrics,
            samples=samples,
            start_time=start_time,
            end_time=end_time,
            generation_parameters={
                "temperature": 0.0,
                "max_tokens": self.config.max_token_length,
            },
        )

        return eval_metrics

    async def get_next_item(self):
        next_item = self.train[self.iter % len(self.train)]
        self.iter += 1
        return next_item

    async def collect_trajectories(self, item) -> Tuple[ScoredDataGroup, list]:
        question = item["question"]
        answer = item["answer"]

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]

        completions = await self.server.chat_completion(
            messages=messages,
            n=self.config.group_size,
            max_tokens=self.config.max_token_length,
            temperature=1.0,
        )

        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_len = len(prompt_tokens)

        scores: ScoredDataGroup = {
            "tokens": [],
            "masks": [],
            "scores": [],
            "inference_logprobs": [],
        }

        for choice in completions.choices:
            response = choice.message.content or ""

            # Tokenize full sequence (prompt + completion)
            full_text = prompt_text + response
            full_tokens = self.tokenizer.encode(full_text, add_special_tokens=False)

            # Create masks: -100 for prompt, actual tokens for completion
            masks = [-100] * prompt_len + full_tokens[prompt_len:]

            logprobs = [1.0] * prompt_len + [0.0] * (len(full_tokens) - prompt_len)

            # Score using reward funcs
            completion_messages = messages + [
                {"role": "assistant", "content": response}
            ]
            rewards = []
            for func in self.reward_funcs:
                reward = func(
                    parser=self.parser,
                    completion=completion_messages,
                    answer=answer,
                )
                rewards.append(reward)
            weighted_rewards = [
                r * self.reward_scales[j] for j, r in enumerate(rewards)
            ]
            score = sum(weighted_rewards)

            scores["tokens"].append(full_tokens)
            scores["masks"].append(masks)
            scores["inference_logprobs"].append(logprobs)
            scores["scores"].append(score)

        return scores, []


if __name__ == "__main__":
    VerifiersEnv.cli()
