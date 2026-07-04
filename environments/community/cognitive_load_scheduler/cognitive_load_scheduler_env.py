from typing import List, Optional

from pydantic import Field

from atroposlib.envs.base import BaseEnv, BaseEnvConfig, ScoredDataGroup
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer


class CognitiveSchedulerConfig(BaseEnvConfig):
    system_prompt: Optional[str] = Field(
        "You are a cognitive load CPU scheduler. Output optimal scheduler order.",
        description="System prompt",
    )


class CognitiveSchedulerEnv(BaseEnv):
    env_config_cls = CognitiveSchedulerConfig

    async def get_next_item(self):
        prompt = "Determine optimal execution order of Task A (load 8), Task B (load 4), and Task C (load 1) within limit 10."
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
            content = traj[-1]["content"].lower()
            reward = 1.0 if "task b" in content and "task c" in content else 0.1
            out_dict = tokenize_for_trainer(self.tokenizer, traj)
            scored["tokens"].append(out_dict["tokens"])
            scored["masks"].append(out_dict["masks"])
            scored["scores"].append(reward)
        return scored
