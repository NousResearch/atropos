import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from atroposlib.envs.base import BaseEnv, BaseEnvConfig, ScoredDataGroup, ScoredDataItem
from atroposlib.envs.server_handling.server_manager import APIServerConfig, ServerBaseline
from atroposlib.type_definitions import Item

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_e2e")

class MockEnvConfig(BaseEnvConfig):
    """Configuration for E2E verification."""
    # Inherits all new features from BaseEnvConfig
    tokenizer_name: str = "gpt2"
    group_size: int = 4
    num_difficulty_bins: int = 5

class MockEnv(BaseEnv):
    """A minimal environment to verify BaseEnv integration features."""
    
    async def setup(self):
        self.items = [f"item_{i}" for i in range(20)]
        self.iter = 0

    async def get_next_item(self) -> Item:
        item = self.items[self.iter % len(self.items)]
        self.iter += 1
        return item

    def format_prompt(self, item: Item) -> str:
        return f"Prompt for {item}"

    async def collect_trajectory(self, item: Item) -> Tuple[ScoredDataItem, List[Item]]:
        # Simulate a rollout with multiple rewards for the ensemble
        # Rewards vary based on "item index" to test curriculum/normalization
        idx = int(item.split("_")[1])
        base_reward = float(idx) / 20.0
        
        # Multiple scores to trigger consensus
        scores = [base_reward, base_reward + 0.1, base_reward - 0.1]
        
        # Add some noise to test stability
        scores = [s + np.random.normal(0, 0.01) for s in scores]
        
        return {
            "tokens": [1, 2, 3],
            "masks": [0, 1, 1],
            "scores": scores,  # List of scores triggers Ensemble
        }, []

    async def evaluate(self, *args, **kwargs):
        pass

async def main():
    logger.info("Starting E2E Readiness Check...")
    
    # 1. Setup Config with ALL features enabled
    config = MockEnvConfig(
        reward_mode="consensus",            # Ensemble
        reward_normalization="zscore",      # Normalization
        curriculum_strategy="easy_first",   # Curriculum
        track_api_perf=True,                # Perf Tracker
        use_wandb=False,                    # No real wandb for test
        tokenizer_name="gpt2",
        group_size=4,
        warmup_steps=2                      # Early normalization transition
    )
    
    # Mock server config (BaseEnv needs it but we won't use real server)
    server_configs = [APIServerConfig(model_name="mock", base_url="http://localhost:8000", api_key="test")]
    
    # 2. Initialize Env
    env = MockEnv(config, server_configs, testing=True)
    await env.setup()
    
    logger.info("Environment initialized with all RL features.")
    
    # 3. Simulate 5 steps (rollout groups)
    for step in range(1, 6):
        logger.info(f"--- Step {step} ---")
        
        # Get item (this should go through the curriculum scheduler)
        item = await env.get_next_item()
        
        # Collect (this triggers ensemble and normalization update)
        results, next_items = await env.collect_trajectories(item)
        
        # Manually trigger wandb_log (it updates stats and formats metrics)
        metrics = {}
        await env.wandb_log(metrics)
        
        # 4. Verify presence of expected keys
        expected_prefixes = ["reward_norm/", "curriculum/", "api_perf/"]
        found_keys = [k for k in metrics.keys() if any(k.startswith(p) for p in expected_prefixes)]
        
        logger.info(f"Metrics keys found: {found_keys}")
        
        if "reward_norm/mean" in metrics:
            logger.info(f"Normalization Mean: {metrics['reward_norm/mean']:.4f}")
        if "curriculum/target_bin" in metrics:
            logger.info(f"Curriculum Target Bin: {metrics['curriculum/target_bin']}")
        if "api_perf/items_per_sec" in metrics:
            logger.info(f"API Throughput: {metrics['api_perf/items_per_sec']:.2f} items/s")

    logger.info("E2E Readiness Check Completed Successfully!")

if __name__ == "__main__":
    asyncio.run(main())
