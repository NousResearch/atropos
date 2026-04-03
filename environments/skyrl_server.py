"""
SkyRL Training Environment for Atropos

Unified environment for reasoning-heavy RL training (Project 11).
Integrates Berkeley SkyRL-gym with Atropos orchestration.
Supports Step-wise Process Rewards (PRM) and Zero-Copy SHM transport (Project 9).

Usage:
  python environments/skyrl_server.py serve \
      --env.skyrl_repo_id "NovaSky-AI/Sky-AIME-5K" \
      --openai.base_url http://localhost:9101/v1
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from atroposlib.envs.skyrl_adapter import SkyRLAdapter, SkyRLConfig
from atroposlib.envs.server_handling.server_baseline import APIServerConfig

logger = logging.getLogger(__name__)


class SkyRLServerEnv(SkyRLAdapter):
    """
    User-facing environment for SkyRL reasoning tasks.
    """
    
    @classmethod
    def config_init(cls) -> Tuple[SkyRLConfig, List[APIServerConfig]]:
        env_config = SkyRLConfig(
            tokenizer_name="Qwen/Qwen2.5-1.5B-Instruct",
            group_size=8,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=4,
            max_token_length=4096,
            wandb_name="skyrl-reasoning",
            enable_process_rewards=True,
        )
        server_configs = [
            APIServerConfig(
                model_name="Qwen/Qwen2.5-1.5B-Instruct",
                base_url="http://localhost:9001/v1",
                api_key="x",
                server_type="sglang",
            ),
        ]
        return env_config, server_configs

    async def setup(self):
        """
        Initialization logic for SkyRL benchmarks.
        """
        await super().setup()
        logger.info("SkyRL environment setup complete.")

    async def evaluate(self) -> Dict[str, float]:
        """
        Reasoning-specific evaluation logic.
        """
        logger.info("Running SkyRL Reasoning Evaluation...")
        return {"reasoning_acc": 0.0} # Placeholder for Phase 1


if __name__ == "__main__":
    SkyRLServerEnv.cli()
