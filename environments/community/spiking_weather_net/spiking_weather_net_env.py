from typing import List, Optional

from pydantic import Field

from atroposlib.envs.base import BaseEnv, BaseEnvConfig, ScoredDataGroup
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer


class SpikingWeatherConfig(BaseEnvConfig):
    system_prompt: Optional[str] = Field(
        "You are a spiking neural weather simulator. Output spike sequence (0s and 1s).",
        description="System prompt",
    )


class SpikingWeatherEnv(BaseEnv):
    env_config_cls = SpikingWeatherConfig

    async def get_next_item(self):
        prompt = "Convert sequence of signal inputs (0.2, 0.4, 0.5, 0.8) into spike responses."
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
            reward = 1.0 if "1" in content else 0.1
            out_dict = tokenize_for_trainer(self.tokenizer, traj)
            scored["tokens"].append(out_dict["tokens"])
            scored["masks"].append(out_dict["masks"])
            scored["scores"].append(reward)
        return scored
