import json
import logging
import os
import random
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset

from atroposlib.envs.base import APIServerConfig, BaseEnv, BaseEnvConfig, ScoredDataItem
from atroposlib.type_definitions import Item, Message
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer


logger = logging.getLogger(__name__)


class ClineAgentEnvConfig(BaseEnvConfig):
    tokenizer_name: str = "NousResearch/Meta-Llama-3-8B"
    env_name: str = "cline_agent_env"
    dataset_name: str = "nebius/SWE-agent-trajectories"
    max_episode_turns: int = 1
    eval_episodes: int = 50
    scoring_function: str = "dataset_target"
    system_prompt: str = (
        "You are a senior software engineer helping to resolve a GitHub issue. "
        "Read the issue description carefully and propose a clear, concrete patch "
        "or explanation of how to resolve it."
    )


class ClineAgentEnv(BaseEnv):
    name = "cline_agent_env"
    env_config_cls = ClineAgentEnvConfig

    def __init__(
        self,
        config: ClineAgentEnvConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = False,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: ClineAgentEnvConfig = config
        self.dataset = None
        self.dataset_indices: List[int] = []
        self.dataset_position = 0
        self.episode_outcomes_buffer: List[float] = []
        self.eval_metrics_custom: List[Tuple[str, float]] = []

    @classmethod
    def config_init(cls) -> Tuple[ClineAgentEnvConfig, List[APIServerConfig]]:
        tokenizer_name = os.getenv("TOKENIZER_NAME", "gpt2")

        env_config = ClineAgentEnvConfig(
            tokenizer_name=tokenizer_name,
            group_size=4,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            max_token_length=4096,
            wandb_name=cls.name,
            steps_per_eval=100,
            max_episode_turns=1,
            eval_episodes=50,
        )
        server_configs = [
            APIServerConfig(
                model_name="anthropic_sonnet_like",
                base_url="http://localhost:9001/v1",
                api_key="x",
                num_requests_for_eval=128,
            ),
        ]
        return env_config, server_configs

    async def setup(self):
        if self.dataset is None:
            self.dataset = load_dataset(self.config.dataset_name, split="train")
            self.dataset_indices = list(range(len(self.dataset)))
            random.shuffle(self.dataset_indices)
            self.dataset_position = 0

    async def collect_trajectory(
        self, item: Item
    ) -> Tuple[Optional[ScoredDataItem], List[Item]]:
        issue_text: str = item["issue_text"]
        target: bool = item["target"]

        messages: List[Message] = [
            {"role": "system", "content": self.config.system_prompt, "reward": None},
            {"role": "user", "content": issue_text, "reward": None},
        ]

        chat_completion = await self.server.chat_completion(
            messages=messages,
            n=1,
            max_tokens=self.config.max_token_length,
        )
        assistant_content = chat_completion.choices[0].message.content

        messages.append(
            {"role": "assistant", "content": assistant_content, "reward": None}
        )

        if self.config.scoring_function == "dataset_target":
            reward = 1.0 if target else -1.0
        else:
            reward = 0.0

        self.episode_outcomes_buffer.append(reward)

        tokenized = tokenize_for_trainer(
            self.tokenizer,
            messages,
            include_messages=self.config.include_messages,
            train_on_all_assistant_turns=False,
        )

        scored_item: ScoredDataItem = {
            "tokens": tokenized["tokens"],
            "masks": tokenized["masks"],
            "scores": reward,
            "advantages": None,
            "ref_logprobs": None,
            "messages": messages if self.config.include_messages else None,
            "group_overrides": None,
            "overrides": None,
            "images": None,
        }
        return scored_item, []

    async def get_next_item(self) -> Item:
        if self.dataset is None:
            await self.setup()

        if not self.dataset_indices:
            raise RuntimeError("Dataset indices not initialized")

        index = self.dataset_indices[self.dataset_position % len(self.dataset_indices)]
        self.dataset_position += 1
        row = self.dataset[index]

        trajectory = row["trajectory"]

        issue_text = ""
        for entry in trajectory:
            if entry.get("role") == "user":
                issue_text = entry.get("text", "")
                if issue_text:
                    break

        if not issue_text:
            issue_text = trajectory[0].get("system_prompt", "")

        item: Item = {
            "instance_id": row["instance_id"],
            "model_name": row["model_name"],
            "target": bool(row["target"]),
            "issue_text": issue_text,
        }
        return item

    async def evaluate(self, *args, **kwargs):
        eval_outcomes: List[float] = []

        for _ in range(self.config.eval_episodes):
            item = await self.get_next_item()
            scored_item_tuple = await self.collect_trajectory(item)
            if scored_item_tuple and scored_item_tuple[0]:
                outcome = scored_item_tuple[0]["scores"]
                eval_outcomes.append(outcome)

        if not eval_outcomes:
            self.eval_metrics_custom = []
            return

        num_completed = len(eval_outcomes)
        avg_reward = sum(eval_outcomes) / num_completed if num_completed > 0 else 0.0
        success_rate = (
            sum(1 for r in eval_outcomes if r > 0) / num_completed
            if num_completed > 0
            else 0.0
        )

        self.eval_metrics_custom = [
            (f"{self.name}_eval/avg_reward", avg_reward),
            (f"{self.name}_eval/success_rate", success_rate),
            (f"{self.name}_eval/num_completed_episodes", num_completed),
        ]

    async def wandb_log(self, wandb_metrics: Optional[Dict[str, float]] = None):
        if wandb_metrics is None:
            wandb_metrics = {}

        if self.episode_outcomes_buffer:
            avg_training_reward = sum(self.episode_outcomes_buffer) / len(
                self.episode_outcomes_buffer
            )
            wandb_metrics[
                f"{self.name}_train/avg_episode_reward"
            ] = avg_training_reward
            wandb_metrics[
                f"{self.name}_train/num_episodes_in_batch"
            ] = len(self.episode_outcomes_buffer)

        self.episode_outcomes_buffer = []

        for key, value in self.eval_metrics_custom:
            wandb_metrics[key] = value
        self.eval_metrics_custom = []

        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    ClineAgentEnv.cli()

