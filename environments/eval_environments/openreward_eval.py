"""
OpenReward Evaluation for Atropos

Evaluates models on OpenReward Standard (ORS) environments. 
Matches the UX and rigor of verifiers_eval.py.

Usage:
    python environments/eval_environments/openreward_eval.py \\
        --server-url http://localhost:9001/v1 \\
        --model-name "NousResearch/DeepHermes-3-Llama-3-8B-Preview" \\
        --or-env-name "kanishk/EndlessTerminals" \\
        --max-eval-items 50
"""

import argparse
import asyncio
import json
import logging
import os
import time
import traceback
from typing import Any, Dict, List, Optional

from atroposlib.envs.eval import EvalBase, evaluate_log
from atroposlib.envs.server_handling.server_baseline import APIServerConfig
from atroposlib.envs.server_handling.server_manager import ServerManager

# OpenReward is an optional dependency
try:
    from openreward import AsyncOpenReward
    from openreward.environments import ToolOutput
    ORWD_AVAILABLE = True
except ImportError:
    ORWD_AVAILABLE = False
    AsyncOpenReward = None
    ToolOutput = None

# Reuse tool call parsing from the main server script
# We define it here again to keep the eval script standalone
import re
def parse_tool_call(response_text: str) -> Optional[Dict[str, Any]]:
    if not response_text:
        return None
    xml_match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", response_text, re.DOTALL)
    if xml_match:
        try:
            return json.loads(xml_match.group(1).strip())
        except json.JSONDecodeError:
            pass
    json_match = re.search(r"(\{.*\})", response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass
    return None

logger = logging.getLogger(__name__)


class OpenRewardEval(EvalBase):
    """
    OpenReward Evaluation using EvalBase pattern.
    """

    def __init__(
        self,
        or_env_name: str,
        split: str = "test",
        max_steps: int = 16,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        max_eval_items: int = -1,
        **kwargs,
    ):
        self.or_env_name = or_env_name
        self.split = split
        self.max_steps = max_steps
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_eval_items = max_eval_items
        
        if not ORWD_AVAILABLE:
            raise ImportError("openreward library not found.")
        
        self.client = AsyncOpenReward()
        super().__init__(**kwargs)

    async def setup_data(self) -> list:
        """Fetch tasks from OpenReward."""
        logger.info("Fetching tasks for evaluation: %s (split: %s)", self.or_env_name, self.split)
        env_api = self.client.environments(self.or_env_name)
        tasks = await env_api.list_tasks(split=self.split)
        
        if self.max_eval_items > 0:
            tasks = tasks[:self.max_eval_items]
        
        return tasks

    async def run_item(self, server_manager: ServerManager, task_item: dict):
        """Run a single evaluation item (ORS session)."""
        server = server_manager.servers[0]
        model_name = server.config.model_name
        
        # We use a fresh managed server context per item for clean tracking
        async with server_manager.managed_server(tokenizer=None) as managed:
            session = await self.client.create_session(self.or_env_name, task_spec=task_item)
            
            prompt_blocks = await session.get_prompt()
            prompt_text = "\n".join([b.text for b in prompt_blocks if hasattr(b, "text")])
            
            messages = [{"role": "user", "content": prompt_text}]
            total_reward = 0.0
            steps = 0
            done = False
            success = 0.0
            
            try:
                while not done and steps < self.max_steps:
                    # Model call
                    response = await managed.chat_completion(
                        messages=messages,
                        model=model_name,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                    model_text = response.choices[0].message.content
                    messages.append({"role": "assistant", "content": model_text})
                    
                    # Tool call
                    action = parse_tool_call(model_text)
                    if action:
                        tool_output: ToolOutput = await session.call_tool(action["name"], action.get("arguments", {}))
                        
                        reward = tool_output.reward or 0.0
                        done = tool_output.finished
                        total_reward += reward
                        
                        obs_text = "\n".join([b.text for b in tool_output.blocks if hasattr(b, "text")])
                        messages.append({"role": "tool", "content": obs_text, "tool_call_id": "ors_eval"})
                        
                        if done and reward > 0:
                            success = 1.0
                    else:
                        messages.append({"role": "user", "content": "Error: No valid tool call found. Please use the required tool format."})
                        if steps > self.max_steps // 2:
                            done = True
                    
                    steps += 1
            except Exception as e:
                logger.error("Error during eval rollout: %s", e)
                logger.error(traceback.format_exc())
            finally:
                await session.close()
            
            return {
                "score": total_reward,
                "success": success,
                "steps": steps,
                "messages": messages,
            }

    async def __call__(self, server_manager: ServerManager):
        """Override __call__ to run parallel evaluation and log results."""
        start_time = time.time()
        
        # Load tasks
        tasks = await self.setup_data()
        total_tasks = len(tasks)
        logger.info("Starting evaluation on %d items...", total_tasks)
        
        # Standard EvalBase logic for running items in parallel
        # We'll use a semaphore to control concurrency
        semaphore = asyncio.Semaphore(getattr(self, "max_concurrent", 10))
        
        async def wrapped_run(task):
            async with semaphore:
                return await self.run_item(server_manager, task)
        
        results = await asyncio.gather(*[wrapped_run(t) for t in tasks])
        
        end_time = time.time()
        
        # Aggregate metrics
        total_reward = sum(r["score"] for r in results)
        total_success = sum(r["success"] for r in results)
        total_steps = sum(r["steps"] for r in results)
        
        metrics = {
            "avg_reward": total_reward / total_tasks if total_tasks > 0 else 0,
            "success_rate": total_success / total_tasks if total_tasks > 0 else 0,
            "avg_steps": total_steps / total_tasks if total_tasks > 0 else 0,
        }
        
        # Formatting for display
        print(f"\n{'=' * 60}")
        print(f"OpenReward Evaluation Results: {self.or_env_name}")
        print(f"{'=' * 60}")
        print(f"  Average Reward: {metrics['avg_reward']:.4f}")
        print(f"  Success Rate:   {metrics['success_rate']:.2%}")
        print(f"  Avg Steps:      {metrics['avg_steps']:.2f}")
        print(f"  Total Items:    {total_tasks}")
        print(f"  Time Elapsed:   {end_time - start_time:.1f}s")
        print(f"{'=' * 60}\n")
        
        # Log to WandB / JSON
        samples = []
        for i, res in enumerate(results):
            samples.append({
                "messages": res["messages"],
                "score": res["score"],
                "success": res["success"],
            })
        
        model_name = server_manager.servers[0].config.model_name
        evaluate_log(
            metrics=metrics,
            eval_dir=getattr(self, "eval_dir", None),
            task_name=f"ORW_{self.or_env_name.replace('/', '_')}",
            model_name=model_name,
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
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-eval-items", type=int, default=-1)
    parser.add_argument("--max-concurrent", type=int, default=10)
    parser.add_argument("--eval-dir", type=str, default=None)
    
    args = parser.parse_args()
    
    server_manager = ServerManager(configs=[
        APIServerConfig(
            api_key=args.api_key,
            base_url=args.server_url,
            model_name=args.model_name,
            health_check=False,
        )
    ])
    
    eval_env = OpenRewardEval(
        or_env_name=args.or_env_name,
        split=args.split,
        max_steps=args.max_steps,
        temperature=args.temperature,
        max_eval_items=args.max_eval_items,
        max_concurrent=args.max_concurrent,
        eval_dir=args.eval_dir,
    )
    
    await eval_env(server_manager)


if __name__ == "__main__":
    asyncio.run(main())
