"""
OpenReward Training Environment for Atropos

Unified environment for RL training (serve), SFT data generation (process),
and multi-turn interaction with OpenReward Standard (ORS) compliant environments.
Uses ManagedServerAdapter for OpenAI-compatible interaction with automatic token tracking.

Usage:
  # RL Training
  python environments/openreward_server.py serve \
      --env.or_env_name "kanishk/EndlessTerminals" \
      --openai.base_url http://localhost:9001/v1

  # SFT Data Generation
  python environments/openreward_server.py process \
      --env.or_env_name "kanishk/EndlessTerminals" \
      --env.data_path_to_save_groups or_sft.jsonl
"""

import asyncio
import json
import logging
import re
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import wandb
from pydantic import BaseModel, Field

# OpenReward is an optional dependency
try:
    from openreward import AsyncOpenReward
    from openreward.api.environments.types import Task
    from openreward.environments import ToolOutput

    ORWD_AVAILABLE = True
except ImportError:
    ORWD_AVAILABLE = False
    AsyncOpenReward = None
    ToolOutput = None

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.envs.server_handling.managed_server import ManagedServerAdapter

logger = logging.getLogger(__name__)


class OrwEnvConfig(BaseEnvConfig):
    or_env_name: str = Field(
        ..., description="OpenReward environment name (owner/name)"
    )
    max_steps: int = Field(default=16, description="Max turns per episode")
    reward_reduction: str = Field(
        default="sum", description="Aggregation: sum, mean, max, min"
    )
    split: str = Field(default="train", description="Dataset split to use")


def parse_tool_call(response_text: str) -> Optional[Dict[str, Any]]:
    """Parses tool calls from model response."""
    if not response_text:
        return None

    # 1. Try XML-style extraction
    xml_match = re.search(
        r"<tool_call>\s*(.*?)\s*</tool_call>", response_text, re.DOTALL
    )
    if xml_match:
        try:
            return json.loads(xml_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 2. Try raw JSON extraction
    json_match = re.search(r"(\{.*\})", response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    return None


class OpenRewardEnv(BaseEnv):
    name = "openreward"
    env_config_cls = OrwEnvConfig  # type: ignore[assignment]

    def __init__(
        self,
        config: OrwEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=False,
        testing=False,
    ):
        # Standardize tokenizer_name to avoid OS errors with remote model IDs (e.g. gemini)
        if config.tokenizer_name and (
            "gemini" in config.tokenizer_name.lower()
            or "models/" in config.tokenizer_name.lower()
        ):
            config.tokenizer_name = "gpt2"
        elif not config.tokenizer_name:
            config.tokenizer_name = "gpt2"

        # Also sanitize server_configs to prevent sglang_server from trying to load Gemini as a tokenizer
        for sc in server_configs:
            if (
                sc.tokenizer_name == "none"
                or sc.tokenizer_name == "default"
                or ("gemini" in sc.model_name.lower())
            ):
                sc.tokenizer_name = config.tokenizer_name

        super().__init__(config, server_configs, slurm, testing)

        # Ensure tokenizer has a chat template (required for rollout)
        if (
            not hasattr(self.tokenizer, "chat_template")
            or self.tokenizer.chat_template is None
        ):
            logger.info("Injecting default chat template into tokenizer")
            self.tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

        if not ORWD_AVAILABLE:
            raise ImportError("openreward library not found.")

        self.reward_buffer: List[float] = []
        self.num_turns_buffer: List[int] = []
        self.success_buffer: List[float] = []

        self._client: Optional[AsyncOpenReward] = None
        self.tasks: List[Dict[str, Any]] = []
        self.iter = 0

    @property
    def client(self) -> AsyncOpenReward:
        """Lazy initialization of the OpenReward client."""
        if self._client is None:
            import socket
            original_getaddrinfo = socket.getaddrinfo
            def patched_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
                if host and "openreward.ai" in host: return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("34.160.223.52", port))]
                return original_getaddrinfo(host, port, family, type, proto, flags)
            socket.getaddrinfo = patched_getaddrinfo
            self._client = AsyncOpenReward()
        return self._client

    @classmethod
    def config_init(cls) -> Tuple[OrwEnvConfig, List[APIServerConfig]]:
        env_config = OrwEnvConfig(
            or_env_name="kanishk/EndlessTerminals",
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=4,
            use_wandb=True,
            total_steps=1000,
            batch_size=4,
            max_token_length=8192,
            wandb_name="openreward_atropos",
        )
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9001/v1",
                api_key="x",
                server_type="sglang",
            ),
        ]
        return env_config, server_configs

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """SOTA W&B logging with metrics buffering."""
        if wandb_metrics is None:
            wandb_metrics = {}

        if self.reward_buffer:
            wandb_metrics["metrics/mean_reward"] = sum(self.reward_buffer) / len(
                self.reward_buffer
            )
            wandb_metrics["metrics/reward_std"] = (
                (
                    sum(
                        (r - wandb_metrics["metrics/mean_reward"]) ** 2
                        for r in self.reward_buffer
                    )
                    / len(self.reward_buffer)
                )
                ** 0.5
                if len(self.reward_buffer) > 1
                else 0.0
            )
            self.reward_buffer = []

        if self.num_turns_buffer:
            wandb_metrics["metrics/avg_num_turns"] = sum(self.num_turns_buffer) / len(
                self.num_turns_buffer
            )
            self.num_turns_buffer = []

        if self.success_buffer:
            wandb_metrics["metrics/success_rate"] = sum(self.success_buffer) / len(
                self.success_buffer
            )
            self.success_buffer = []

        await super().wandb_log(wandb_metrics)

    async def setup(self):
        """Dynamic task discovery and data setup."""
        logger.info("Initializing OpenReward session for: %s", self.config.or_env_name)

        # Dynamic discovery of task specifications
        specs = await self.client.environments.get(self.config.or_env_name).list_tasks(
            split=self.config.split
        )

        try:
            self.env_handle = self.client.environments.get(name=self.config.or_env_name)

            parts = self.config.or_env_name.split("/")
            ns = parts[0] if len(parts) > 1 else "matrix"
            sn = parts[1] if len(parts) > 1 else parts[0]

            # Dynamic Task Discovery
            real_tasks = await self.env_handle.get_task_range(
                split=self.config.split, start=0, stop=1
            )

            if not real_tasks:
                raise ValueError(f"No tasks found in {self.config.or_env_name}")

            self.injected_task = Task(
                server_name=sn,
                environment_name=real_tasks[0].environment_name,
                task_spec=real_tasks[0].task_spec,
                namespace=ns,
            )
            self.tasks = [self.injected_task]
            logger.info(f"Initialized with Task ID: {real_tasks[0].task_spec}")
        except Exception as e:
            logger.error("Failed to initialize OpenReward: %s", e)
            raise

    async def get_next_item(self):
        """Pick the next task."""
        if not self.tasks:
            raise RuntimeError("No tasks loaded")
        task = self.tasks[self.iter % len(self.tasks)]
        self.iter += 1
        return task

    async def evaluate(self) -> Dict[str, float]:
        """Unified evaluation loop compatible with BaseEnv CLI."""
        logger.info("Starting Unified Evaluation for: %s", self.config.or_env_name)

        # Determine number of evaluation items from total_steps or default
        num_items = self.config.total_steps if self.config.total_steps > 0 else 10
        total_reward = 0.0
        total_success = 0.0
        count = 0

        for i in range(num_items):
            logger.info("Evaluating rollout %d/%d...", i + 1, num_items)
            try:
                scored_data, _ = await self.collect_trajectories(None)
                if scored_data["scores"]:
                    avg_score = sum(scored_data["scores"]) / len(scored_data["scores"])
                    total_reward += avg_score
                    total_success += sum(
                        1 for s in scored_data["scores"] if s > 0
                    ) / len(scored_data["scores"])
                    count += 1
            except Exception as e:
                logger.error("Eval rollout %d failed: %s", i, e)

        if count == 0:
            return {"eval/avg_reward": 0.0, "eval/success_rate": 0.0}

        metrics = {
            "eval/avg_reward": total_reward / count,
            "eval/success_rate": total_success / count,
        }

        logger.info(
            "Eval Results: Avg Reward: %.4f, Success: %.2f%%",
            metrics["eval/avg_reward"],
            metrics["eval/success_rate"] * 100,
        )

        return metrics

    async def collect_trajectories(self, item) -> Tuple[ScoredDataGroup, list]:
        """Unified trajectory collection using ManagedServerAdapter."""
        # Ensure client is available in this thread/context
        if self._client is None:
            self._client = AsyncOpenReward()

        server_config = self.server.servers[0].config

        scored_data: ScoredDataGroup = {
            "tokens": [],
            "masks": [],
            "scores": [],
            "messages": [],
            "inference_logprobs": [],
        }

        async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
            # SOTA Adapter integration
            adapter = ManagedServerAdapter(
                managed_server=managed,
                base_url=server_config.base_url,
            )

            for i in range(self.config.group_size):
                try:
                    async with self.env_handle.session(
                        task=self.injected_task
                    ) as session:
                        prompt_blocks = await session.get_prompt()
                        prompt_text = "\n".join(
                            [b.text for b in prompt_blocks if hasattr(b, "text")]
                        )

                        messages = [{"role": "user", "content": prompt_text}]
                        done = False
                        steps = 0
                        total_reward = 0.0
                        success = 0.0

                        while not done and steps < self.config.max_steps:
                            # OpenAI-compatible call via Adapter
                            response = await adapter.chat.completions.create(
                                model=server_config.model_name,
                                messages=messages,
                                temperature=1.0,
                            )

                            assistant_msg = response.choices[0].message.content
                            messages.append(
                                {"role": "assistant", "content": assistant_msg}
                            )

                            action = parse_tool_call(assistant_msg)
                            if action:
                                try:
                                    tool_output = await session.call_tool(
                                        action["name"], 
                                        action.get("arguments", {})
                                    )

                                    reward = tool_output.reward or 0.0
                                    done = tool_output.finished
                                    total_reward += reward

                                    obs_text = "\n".join(
                                        [
                                            b.text
                                            for b in tool_output.blocks
                                            if hasattr(b, "text")
                                        ]
                                    )
                                    messages.append(
                                        {"role": "user", "content": obs_text}
                                    )

                                    if done and reward > 0:
                                        success = 1.0
                                except Exception as te:
                                    logger.warning("Tool error: %s", te)
                                    messages.append(
                                        {"role": "user", "content": f"Error: {str(te)}"}
                                    )
                            else:
                                done = True  # Terminate on malformed action for RL stability

                            steps += 1

                        # Record rollout results from ManagedServer
                        managed_state = managed.get_state()
                        node = managed_state["nodes"][-1]

                        scored_data["tokens"].append(node.tokens)
                        scored_data["masks"].append(node.masked_tokens)
                        scored_data["inference_logprobs"].append(node.logprobs)
                        scored_data["messages"].append(messages)

                        final_score = total_reward
                        if self.config.reward_reduction == "mean" and steps > 0:
                            final_score /= steps
                        scored_data["scores"].append(final_score)

                        # Metrics Buffers
                        self.reward_buffer.append(total_reward)
                        self.num_turns_buffer.append(steps)
                        self.success_buffer.append(success)

                    # Reset managed tracking for next rollout in group
                    managed.reset()
                except Exception as e:
                    logger.error("Rollout failed: %s", e)
                    continue

        return scored_data, []


if __name__ == "__main__":
    OpenRewardEnv.cli()
