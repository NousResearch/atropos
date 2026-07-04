from typing import List, Optional
from pydantic import Field
from atroposlib.envs.base import BaseEnv, BaseEnvConfig, ScoredDataGroup
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

class TemporalKnowledgeDecayConfig(BaseEnvConfig):
    system_prompt: Optional[str] = Field("You are a dynamic knowledge decay predictor. Return survival float.", description="System prompt")

class TemporalKnowledgeDecayEnv(BaseEnv):
    env_config_cls = TemporalKnowledgeDecayConfig

    async def get_next_item(self):
        prompt = "Determine survival of spouse edge with half-life of 730 days at age of 365 days."
        return (tuple([frozenset({"role": "user", "content": prompt}.items())]), None, None)

    async def collect_trajectories(self, item):
        user_content = dict(item[0][0])["content"]
        messages = [{"role": "user", "content": user_content}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        completions = await self.server.completion(prompt=prompt, n=self.config.group_size)
        trajectories = []
        for completion in completions.choices:
            text = completion.text if hasattr(completion, "text") else completion.message.content
            msg_seq = [{"role": "user", "content": user_content}, {"role": "assistant", "content": text}]
            trajectories.append(msg_seq)
        return trajectories, []

    async def score(self, rollout_group_data: List) -> Optional[ScoredDataGroup]:
        scored = ScoredDataGroup()
        scored["tokens"], scored["masks"], scored["scores"] = [], [], []
        for traj in rollout_group_data:
            content = traj[-1]["content"]
            try:
                prob = float(content.strip())
                reward = 1.0 - abs(prob - 0.707)
            except Exception:
                reward = 0.0
            out_dict = tokenize_for_trainer(self.tokenizer, traj)
            scored["tokens"].append(out_dict["tokens"])
            scored["masks"].append(out_dict["masks"])
            scored["scores"].append(reward)
        return scored
