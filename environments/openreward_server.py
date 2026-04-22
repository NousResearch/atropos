"""
OpenReward Training Environment for Atropos

Unified environment for RL training (serve), SFT data generation (process), 
and multi-turn interaction with OpenReward Standard (ORS) compliant environments.

Usage:
  # RL Training
  python environments/openreward_server.py serve \\
      --env.or_env_name "kanishk/EndlessTerminals" \\
      --openai.base_url http://localhost:9001/v1

  # SFT Data Generation
  python environments/openreward_server.py process \\
      --env.or_env_name "kanishk/EndlessTerminals" \\
      --env.data_path_to_save_groups or_sft.jsonl
"""

import json
import logging
import re
import asyncio
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import wandb
from pydantic import BaseModel, Field

# OpenReward is an optional dependency
try:
    from openreward import AsyncOpenReward
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

logger = logging.getLogger(__name__)


class OrwEnvConfig(BaseEnvConfig):
    or_env_name: str = Field(..., description="OpenReward environment name (owner/name)")
    max_steps: int = Field(default=16, description="Max turns per episode")
    reward_reduction: str = Field(default="sum", description="Aggregation: sum, mean, max, min")
    split: str = Field(default="train", description="Dataset split to use")


def parse_tool_call(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Parses tool calls from model response.
    Supports:
    1. XML-style: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    2. JSON-style: {"name": "...", "arguments": {...}} (if it's the whole string or wrapped)
    """
    if not response_text:
        return None

    # 1. Try XML-style extraction
    xml_match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", response_text, re.DOTALL)
    if xml_match:
        try:
            return json.loads(xml_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 2. Try raw JSON extraction (find first { and last })
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
        super().__init__(config, server_configs, slurm, testing)
        if not ORWD_AVAILABLE:
            raise ImportError(
                "openreward library not found. Install it with: pip install openreward"
            )

        # Metrics buffers
        self.reward_buffer: List[float] = []
        self.episode_length_buffer: List[int] = []
        self.success_buffer: List[float] = []
        self.metrics_buffer: Dict[str, List[float]] = defaultdict(list)

        self.client = AsyncOpenReward()
        self.tasks: List[Dict[str, Any]] = []
        self.iter = 0

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

    async def setup(self):
        """Initialize ORS connection and fetch tasks."""
        logger.info("Initializing OpenReward environment: %s", self.config.or_env_name)
        try:
            # list_tasks is a class method in ORS SDK, but we use the client
            # Actually, we can fetch tasks from the platform API
            env_api = self.client.environments(self.config.or_env_name)
            self.tasks = await env_api.list_tasks(split=self.config.split)
            logger.info("Fetched %d tasks from split '%s'", len(self.tasks), self.config.split)
        except Exception as e:
            logger.error("Failed to fetch tasks from OpenReward: %s", e)
            raise

    async def get_next_item(self):
        """Pick the next task from the list."""
        if not self.tasks:
            raise RuntimeError("No tasks loaded in OpenRewardEnv")
        
        task = self.tasks[self.iter % len(self.tasks)]
        self.iter += 1
        return task

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Log ORS metrics to wandb."""
        if wandb_metrics is None:
            wandb_metrics = {}

        if self.reward_buffer:
            wandb_metrics["metrics/avg_reward"] = sum(self.reward_buffer) / len(self.reward_buffer)
            wandb_metrics["metrics/reward_std"] = (
                (sum((r - wandb_metrics["metrics/avg_reward"])**2 for r in self.reward_buffer) / len(self.reward_buffer))**0.5
                if len(self.reward_buffer) > 1 else 0.0
            )
            self.reward_buffer = []

        if self.episode_length_buffer:
            wandb_metrics["metrics/avg_episode_length"] = sum(self.episode_length_buffer) / len(self.episode_length_buffer)
            self.episode_length_buffer = []

        if self.success_buffer:
            wandb_metrics["metrics/success_rate"] = sum(self.success_buffer) / len(self.success_buffer)
            self.success_buffer = []

        await super().wandb_log(wandb_metrics)

    async def evaluate(self) -> Dict[str, float]:
        """
        No-op implementation for BaseEnv compatibility.
        Use environments/eval_environments/openreward_eval.py for comprehensive evaluation.
        """
        logger.info("Evaluation should be run via environments/eval_environments/openreward_eval.py")
        return {}

    async def collect_trajectories(self, item) -> Tuple[ScoredDataGroup, list]:
        """
        Execute multi-turn rollout loop inside collect_trajectories.
        Maintains token integrity via ManagedServer.
        """
        scored_data: ScoredDataGroup = {
            "tokens": [],
            "masks": [],
            "scores": [],
            "messages": [],
            "inference_logprobs": [],
        }

        # We run group_size rollouts for the same task item
        async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
            for rollout_idx in range(self.config.group_size):
                try:
                    # 1. Create a new ORS session for this rollout
                    # item is the task JSONObject
                    session = await self.client.create_session(self.config.or_env_name, task_spec=item)
                    
                    # 2. Get the initial prompt from the environment
                    prompt_blocks = await session.get_prompt()
                    # Convert blocks to text (ORS uses Block structure)
                    # For now, we assume simple text blocks
                    prompt_text = "\n".join([b.text for b in prompt_blocks if hasattr(b, "text")])
                    
                    messages = [{"role": "user", "content": prompt_text}]
                    total_reward = 0.0
                    steps = 0
                    done = False
                    success = 0.0

                    # 3. Multi-turn interaction loop
                    while not done and steps < self.config.max_steps:
                        # Get model completion (this extends the current managed session)
                        # We use chat_completion to ensure ManagedServer handles template formatting
                        response = await managed.chat_completion(messages=messages)
                        model_text = response.choices[0].message.content
                        
                        # Add assistant message to history
                        messages.append({"role": "assistant", "content": model_text})

                        # Parse action
                        action = parse_tool_call(model_text)
                        
                        if action:
                            # Execute tool call in ORS
                            # ORS step returns (ToolOutput) which has reward, finished, and blocks
                            try:
                                tool_output: ToolOutput = await session.call_tool(action["name"], action.get("arguments", {}))
                                
                                reward = tool_output.reward or 0.0
                                done = tool_output.finished
                                total_reward += reward
                                
                                # Convert tool output blocks back to message
                                obs_text = "\n".join([b.text for b in tool_output.blocks if hasattr(b, "text")])
                                messages.append({"role": "tool", "content": obs_text, "tool_call_id": "ors_call"})
                                
                                if done and reward > 0:
                                    success = 1.0

                            except Exception as e:
                                logger.warning("Tool execution error: %s", e)
                                messages.append({"role": "tool", "content": f"Error: {str(e)}", "tool_call_id": "ors_call"})
                                # We don't terminate on tool error unless the env says so
                        else:
                            # Malformed or no tool call - could be a 'think' block only or random text
                            # We allow one more turn if max_steps not reached, but terminate if it keeps failing
                            messages.append({"role": "user", "content": "Error: No valid tool call found. Please use the required tool format."})
                            if steps > self.config.max_steps // 2: # Prevent deadlocks
                                done = True

                        steps += 1

                    # 4. Extract tracked tokens/logprobs from ManagedServer for this rollout
                    # In sequential mode, ManagedServer appends new nodes
                    state = managed.get_state()
                    nodes = state.get("nodes", [])
                    
                    if len(nodes) > rollout_idx:
                        node = nodes[rollout_idx]
                        scored_data["tokens"].append(node.tokens)
                        scored_data["masks"].append(node.masked_tokens)
                        scored_data["inference_logprobs"].append(node.logprobs)
                        scored_data["messages"].append(messages)
                        
                        # Apply reward reduction
                        final_score = total_reward
                        if self.config.reward_reduction == "mean" and steps > 0:
                            final_score /= steps
                        elif self.config.reward_reduction == "max":
                            # Max across turns? Or final step? ORS usually gives sparse final reward
                            pass 

                        scored_data["scores"].append(final_score)
                        
                        # Buffers for WandB
                        self.reward_buffer.append(total_reward)
                        self.episode_length_buffer.append(steps)
                        self.success_buffer.append(success)
                    else:
                        logger.error("Node mismatch: Rollout %d, but only %d nodes found", rollout_idx, len(nodes))

                    # 5. Cleanup session
                    await session.close()

                except Exception as e:
                    logger.error("Rollout %d failed: %s", rollout_idx, e)
                    logger.error(traceback.format_exc())
                    # Return empty to avoid crashing if possible, or continue
                    continue

        return scored_data, []


if __name__ == "__main__":
    OpenRewardEnv.cli()
