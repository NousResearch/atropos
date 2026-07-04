from typing import List, Optional

from pydantic import Field

from atroposlib.envs.base import BaseEnv, BaseEnvConfig, ScoredDataGroup
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer


class AcousticProteinConfig(BaseEnvConfig):
    system_prompt: Optional[str] = Field(
        "Classify protein folding acoustic states. Respond only with 'FOLDED' or 'MISFOLDED'.",
        description="System prompt",
    )


class AcousticProteinEnv(BaseEnv):
    env_config_cls = AcousticProteinConfig

    async def get_next_item(self):
        prompt = "Determine folding state of acoustic signal with frequency signature centered at 440Hz."
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
            content = traj[-1]["content"].strip().upper()
            reward = 1.0 if "FOLDED" in content else 0.0
            out_dict = tokenize_for_trainer(self.tokenizer, traj)
            scored["tokens"].append(out_dict["tokens"])
            scored["masks"].append(out_dict["masks"])
            scored["scores"].append(reward)
        return scored
