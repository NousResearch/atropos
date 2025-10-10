import asyncio
import os
from typing import Dict, List, Optional, Tuple, TypedDict, Union

import verifiers as vf

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)


class VfEnvConfig:
    vf_env_name: str = "gsm8k"
    env_args: dict = {}


class VerifiersEnv(BaseEnv):
    def __init__(
        self,
        config: BaseEnvConfig,
        server_configs: List[APIServerConfig],
        vf_env_config: VfEnvConfig,
        slurm=False,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.vf_env = vf.load_environment(
            vf_env_config["vf_env_name"], **vf_env_config["env_args"]
        )
        self.rubric = self.vf_env.rubric

        self.parser = self.rubric.parser
        self.reward_funcs = self.rubric.get_reward_funcs()
        self.system_prompt = self.vf_env.system_prompt

    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
        env_config = BaseEnvConfig(
            group_size=8,
            use_wandb=False,
            rollout_server_url="http://localhost:8010",
            total_steps=10,
            batch_size=4,
            steps_per_eval=1,
            max_token_length=2048,
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
        self.test = list()
        for item in test_data:
            self.test.append(
                {
                    "question": item["question"],
                    "answer": item["answer"],
                }
            )
        self.iter = 0

    async def rollout_and_score_eval(
        self, question: str, answer: str, **kwargs
    ) -> dict:
        state = kwargs["state"] if "state" in kwargs else None
        info = kwargs["info"] if "info" in kwargs else None
        system_prompt = kwargs["system_prompt"] if "system_prompt" in kwargs else None
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

        response_content = completion.choices[0].message.content
        messages.append({"role": "assistant", "content": response_content})

        # PARSE HERE WITH VF PARSER
        answer_parsed = self.parser.parse_answer(completion=response_content)

        # USE REWARD FUNC HERE TO GET SCORE
        rewards = [
            await self.rubric.call_reward_func(
                func=func,
                prompt=question,
                completion=messages,
                answer=answer,
                info=info,
                state=state,
            )
            for func in self.reward_funcs
        ]

        score = sum(rewards)

        sample = {
            "messages": messages,
            "question": question,
            "gold_answer": answer,
            # "gold_parsed": str(gold_parsed) if gold_parsed else None,
            "model_parsed": str(answer_parsed) if answer_parsed else None,
            "score": int(score),
            "correct": bool(score),
            "finish_reason": completion.choices[0].finish_reason,
        }

        return {"score": score, "sample": sample}

    async def evaluate(self, *args, **kwargs):
        pass

    async def get_next_item(self):
        next_item = self.train[self.iter % len(self.train)]
        self.iter += 1
        return next_item


async def main():
    env_config, server_configs = VerifiersEnv.config_init()

    vf_env_config = {"vf_env_name": "wordle", "env_args": {"use_think": False}}

    env = VerifiersEnv(
        config=env_config,
        server_configs=server_configs,
        vf_env_config=vf_env_config,
    )

    await env.setup()

    item = await env.get_next_item()
    print(item)

    roll = await env.rollout_and_score_eval(
        question=item["question"],
        answer=item["answer"],
        system_prompt=env.system_prompt,
    )
    print(roll)


if __name__ == "__main__":
    # VerifiersEnv.cli()
    asyncio.run(main())
