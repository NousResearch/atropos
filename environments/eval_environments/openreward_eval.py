import argparse
import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Dict, Optional

# OpenReward SDK
try:
    import wandb
    from openreward import AsyncOpenReward
    from openreward.environments import ToolOutput

    ORWD_AVAILABLE = True
except ImportError:
    ORWD_AVAILABLE = False
    AsyncOpenReward = None
    ToolOutput = None
    wandb = None

from atroposlib.envs.base import APIServerConfig
from atroposlib.envs.eval import EvalBase, evaluate_log
from atroposlib.envs.server_handling.managed_server import ManagedServerAdapter
from atroposlib.envs.server_handling.server_manager import ServerManager
from environments.openreward_utils import get_openreward_system_prompt

logger = logging.getLogger(__name__)


def parse_tool_call(response_text: str) -> Optional[Dict[str, Any]]:
    """Parses tool calls from model response with leniency."""
    if not response_text:
        return None

    # 1. Try XML-style extraction (Lenient regex)
    xml_match = re.search(
        r"<tool_call>\s*(.*?)(\s*</tool_call>|$)", response_text, re.DOTALL
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

    # 3. Fallback for [X] pattern (Specific to GuessTheNumber)
    bracket_match = re.search(r"\[(\d+)\]", response_text)
    if bracket_match:
        return {
            "name": "guess_number",
            "arguments": {"number": int(bracket_match.group(1))},
        }

    return None


class OpenRewardEval(EvalBase):
    """
    Evaluation implementation for OpenReward Standard (ORS) environments.
    """

    def __init__(
        self,
        or_env_name: str,
        model_name: str,
        split: str = "train",
        max_steps: int = 16,
        temperature: float = 0.0,
        max_eval_items: int = 50,
        eval_dir: str = None,
    ):
        super().__init__(model_name=model_name, max_eval_items=max_eval_items)
        self.or_env_name = or_env_name
        self.split = split
        self.max_steps = max_steps
        self.temperature = temperature
        self.eval_dir = eval_dir
        self._client = None
        self.env_handle = None
        self.injected_task = None
        self.env_tools = []

    @property
    def client(self) -> AsyncOpenReward:
        if self._client is None:
            # Monkeypatch for DNS issue if needed
            import socket

            original_getaddrinfo = socket.getaddrinfo

            def patched_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
                if host and "openreward.ai" in host:
                    return [
                        (
                            socket.AF_INET,
                            socket.SOCK_STREAM,
                            6,
                            "",
                            ("34.160.223.52", port),
                        )
                    ]
                return original_getaddrinfo(host, port, family, type, proto, flags)

            socket.getaddrinfo = patched_getaddrinfo
            self._client = AsyncOpenReward()
        return self._client

    def setup_data(self) -> list:
        """Dummy implementation to satisfy EvalBase abstract class."""
        return []

    async def async_setup_data(self) -> list:
        """Fetch tasks and setup environment handles asynchronously."""
        self.env_handle = self.client.environments.get(name=self.or_env_name)

        # Dynamic Task Discovery
        real_tasks = await self.env_handle.get_task_range(
            split=self.split, start=0, stop=1
        )

        if not real_tasks:
            raise ValueError(f"No tasks found in {self.or_env_name}")

        self.injected_task = real_tasks[0]
        ts_final = getattr(real_tasks[0], "task_spec", None)
        if ts_final is None:
            ts_final = real_tasks[0].get("task_spec", "unknown")

        logger.info(f"Initialized Evaluation with Task ID: {ts_final}")

        # Fetch Tools
        try:
            self.env_tools = await self.env_handle.list_tools(format="openai")
            print(f"DEBUG EVAL: Fetched {len(self.env_tools)} tools from environment")
        except Exception as et:
            print(f"DEBUG EVAL: Failed to fetch tools: {et}")
            self.env_tools = []

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

                # Force Dynamic System Prompt for Tool Use
                system_instr = get_openreward_system_prompt(self.env_tools)
                messages = [
                    {"role": "system", "content": system_instr},
                    {"role": "user", "content": prompt_text},
                ]
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
                            args = action.get("arguments", {})
                            tool_output = await session.call_tool(action["name"], args)

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

        if wandb and wandb.run:
            wandb.log(metrics)

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

    # W&B Support
    if os.environ.get("WANDB_PROJECT") and wandb:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT"),
            name=os.environ.get("WANDB_NAME", f"openreward_eval_{args.or_env_name}"),
            config=vars(args),
        )

    try:
        # Run evaluation
        metrics = await eval_env(server_manager)
        print(f"Final Metrics: {metrics}")
    finally:
        if wandb and wandb.run:
            wandb.finish()


if __name__ == "__main__":
    asyncio.run(main())
