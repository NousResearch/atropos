from typing import List, Optional

from pydantic import Field

from atroposlib.envs.base import BaseEnv, BaseEnvConfig, ScoredDataGroup
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer


class LinguisticPaleontologyConfig(BaseEnvConfig):
    system_prompt: Optional[str] = Field(
        "You are a linguistic paleontologist. Estimate the survival probability.",
        description="System prompt",
    )


class LinguisticPaleontologyEnv(BaseEnv):
    env_config_cls = LinguisticPaleontologyConfig

    async def get_next_item(self):
        prompt = "Estimate survival rate of archaic word 'thee' given current usage statistics."
        return (
            tuple([frozenset({"role": "user", "content": prompt}.items())]),
            None,
            None,
        )

    async def collect_trajectories(self, item):
        user_content = dict(item[0][0])["content"]
        messages = [{"role": "user", "content": user_content}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        completions = await self.server.completion(
            prompt=prompt, n=self.config.group_size
        )
        trajectories = []
        for completion in completions.choices:
            text = (
                completion.text
                if hasattr(completion, "text")
                else completion.message.content
            )
            msg_seq = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": text},
            ]
            trajectories.append(msg_seq)
        return trajectories, []

    async def score(self, rollout_group_data: List) -> Optional[ScoredDataGroup]:
        scored = ScoredDataGroup()
        scored["tokens"], scored["masks"], scored["scores"] = [], [], []
        for traj in rollout_group_data:
            content = traj[-1]["content"]
            try:
                val = float(content.strip())
                reward = 1.0 - abs(val - 0.01)
            except Exception:
                reward = 0.0
            out_dict = tokenize_for_trainer(self.tokenizer, traj)
            scored["tokens"].append(out_dict["tokens"])
            scored["masks"].append(out_dict["masks"])
            scored["scores"].append(reward)
        return scored
