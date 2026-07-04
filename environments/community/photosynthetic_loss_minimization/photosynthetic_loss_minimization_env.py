from typing import List, Optional
from pydantic import Field
from atroposlib.envs.base import BaseEnv, BaseEnvConfig, ScoredDataGroup
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

class PhotosyntheticOptimizerConfig(BaseEnvConfig):
    system_prompt: Optional[str] = Field("You compute photosynthetic gradient steps. Return float step size.", description="System prompt")

class PhotosyntheticOptimizerEnv(BaseEnv):
    env_config_cls = PhotosyntheticOptimizerConfig
    async def get_next_item(self):
        prompt = "Optimize gradient step 0.5 given ATP level 8.0 using photosynthetic scaling."
        return (tuple([frozenset({"role": "user", "content": prompt}.items())]), None, None)
    async def collect_trajectories(self, item):
        user_content = dict(item[0][0])["content"]
        messages = [{"role": "user", "content": user_content}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        completions = await self.server.completion(prompt=prompt, n=self.config.group_size)
        trajectories = []
        for c in completions.choices:
            text = c.text if hasattr(c, "text") else c.message.content
            trajectories.append([{"role": "user", "content": user_content}, {"role": "assistant", "content": text}])
        return trajectories, []
    async def score(self, rollout_group_data: List) -> Optional[ScoredDataGroup]:
        scored = ScoredDataGroup(); scored["tokens"] = []; scored["masks"] = []; scored["scores"] = []
        for traj in rollout_group_data:
            try: val = float(traj[-1]["content"].strip()); reward = 1.0 - abs(val - 0.4)
            except: reward = 0.0
            out = tokenize_for_trainer(self.tokenizer, traj)
            scored["tokens"].append(out["tokens"]); scored["masks"].append(out["masks"]); scored["scores"].append(reward)
        return scored
