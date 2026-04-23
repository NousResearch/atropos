"""
OpenReward Evaluation Environment for Atropos
Uses ManagedServerAdapter for automatic token tracking and OpenAI compatibility.
"""

import argparse
import asyncio
# json removed
import logging
import time
from typing import Any, Dict

from openreward import AsyncOpenReward
from openreward.api.environments.types import Task

from atroposlib.envs.eval import EvalBase, evaluate_log
from atroposlib.envs.server_handling.managed_server import ManagedServerAdapter
from atroposlib.envs.server_handling.server_baseline import APIServerConfig
from atroposlib.envs.server_handling.server_manager import ServerManager

# Reuse the tool parser from the server script
from environments.openreward_server import parse_tool_call

logger = logging.getLogger(__name__)


class OpenRewardEval(EvalBase):
    """
    OpenReward Evaluation using EvalBase pattern.
    Matches Verifiers SOTA architecture.
    """

    def __init__(
        self,
        or_env_name: str,
        model_name: str,
        split: str = "train",
        max_steps: int = 16,
        temperature: float = 0.0,
        max_eval_items: int = -1,
        max_concurrent: int = 10,
        **kwargs,
    ):
        self.or_env_name = or_env_name
        self.model_name = model_name
        self.split = split
        self.max_steps = max_steps
        self.temperature = temperature
        self.max_eval_items = max_eval_items
        self.max_concurrent = max_concurrent
        super().__init__(**kwargs)

        self.client = AsyncOpenReward()

    def setup_data(self) -> list:
        """Dummy sync method to satisfy EvalBase __init__."""
        return []

    async def async_setup_data(self) -> list:
        """Fetch tasks and setup environment handles asynchronously."""
        self.env_handle = self.client.environments.get(name=self.or_env_name)

        parts = self.or_env_name.split("/")
        ns = parts[0] if len(parts) > 1 else "matrix"
        sn = parts[1] if len(parts) > 1 else parts[0]

        # Dynamic Task Discovery
        real_tasks = await self.env_handle.get_task_range(
            split=self.split, start=0, stop=1
        )

        if not real_tasks:
            raise ValueError(f"No tasks found in {self.or_env_name}")

        self.injected_task = Task(
            server_name=sn,
            environment_name=real_tasks[0].environment_name,
            task_spec=real_tasks[0].task_spec,
            namespace=ns,
        )

        logger.info(f"Initialized Evaluation with Task ID: {real_tasks[0].task_spec}")

        num_items = self.max_eval_items if self.max_eval_items > 0 else 50
        return list(range(num_items))

    async def run_item(
        self, server_manager: ServerManager, task_item: Any
    ) -> Dict[str, Any]:
        """Run a single evaluation item."""
        server_config = server_manager.servers[0].config

        async with server_manager.managed_server() as managed:
            # SOTA Adapter integration
            adapter = ManagedServerAdapter(
                managed_server=managed,
                base_url=server_config.base_url,
            )

            async with self.env_handle.session(task=self.injected_task) as session:
                prompt_blocks = await session.get_prompt()
                prompt_text = "\n".join(
                    [b.text for b in prompt_blocks if hasattr(b, "text")]
                )

                messages = [{"role": "user", "content": prompt_text}]
                total_reward = 0.0
                steps = 0
                done = False
                success = 0.0

                try:
                    while not done and steps < self.max_steps:
                        # OpenAI-compatible call
                        response = await adapter.chat.completions.create(
                            model=self.model_name,
                            messages=messages,
                            temperature=self.temperature,
                        )

                        assistant_msg = response.choices[0].message.content
                        messages.append({"role": "assistant", "content": assistant_msg})

                        action = parse_tool_call(assistant_msg)
                        if action:
                            tool_output = await session.call_tool(
                                action["name"], action.get("arguments", {})
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
                            messages.append({"role": "user", "content": obs_text})

                            if done and reward > 0:
                                success = 1.0
                        else:
                            done = True

                        steps += 1
                except Exception as e:
                    logger.error(f"Eval item failed: {e}")

                return {
                    "score": total_reward,
                    "success": success,
                    "steps": steps,
                    "messages": messages,
                }

    async def __call__(self, server_manager: ServerManager):
        """Execute batch evaluation and log results."""
        start_time = time.time()
        data = await self.async_setup_data()

        # Run items concurrently
        results = await asyncio.gather(
            *[self.run_item(server_manager, item) for item in data]
        )

        end_time = time.time()

        # Calculate metrics
        total = len(results)
        avg_reward = sum(r["score"] for r in results) / total if total > 0 else 0
        success_rate = sum(r["success"] for r in results) / total if total > 0 else 0
        avg_steps = sum(r["steps"] for r in results) / total if total > 0 else 0

        metrics = {
            "avg_reward": avg_reward,
            "success_rate": success_rate,
            "avg_steps": avg_steps,
        }

        # Build samples for logging
        samples = []
        for r in results:
            samples.append(
                {
                    "messages": r["messages"],
                    "score": r["score"],
                    "correct": r["success"] > 0,
                }
            )

        evaluate_log(
            metrics=metrics,
            eval_dir=getattr(self, "eval_dir", None),
            task_name=f"OpenRewardEval@{self.or_env_name}",
            model_name=self.model_name,
            start_time=start_time,
            end_time=end_time,
            samples=samples,
        )

        return metrics


async def main():
    parser = argparse.ArgumentParser(description="Evaluate OpenReward environments")
    parser.add_argument("--server-url", type=str, default="http://localhost:9001/v1")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--api-key", type=str, default="x")
    parser.add_argument("--or-env-name", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-eval-items", type=int, default=10)
    parser.add_argument("--eval-dir", type=str, default=None)

    args = parser.parse_args()

    server_manager = ServerManager(
        configs=[
            APIServerConfig(
                api_key=args.api_key,
                base_url=args.server_url,
                model_name=args.model_name,
                health_check=False,
            )
        ]
    )

    eval_env = OpenRewardEval(
        or_env_name=args.or_env_name,
        model_name=args.model_name,
        split=args.split,
        max_steps=args.max_steps,
        temperature=args.temperature,
        max_eval_items=args.max_eval_items,
        eval_dir=args.eval_dir,
    )

    await eval_env(server_manager)


if __name__ == "__main__":
    asyncio.run(main())
