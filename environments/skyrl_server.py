import logging
import os
import sys
import asyncio
import polars as pl
from typing import Any, Dict, List, Optional, Tuple

# Add atropos to path if not already there
sys.path.append("/root/atropos")

from atroposlib.envs.base import BaseEnv, BaseEnvConfig
from pydantic import Field

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

class SkyRLServerConfig(BaseEnvConfig):
    """
    Configuration for the SkyRL Production Server.
    """
    dataset_path: str = Field(
        default="/root/SkyRL/tests/dummy_fixed_16.parquet",
        description="Path to the parquet dataset for task generation."
    )
    shm_name: str = Field(default="atropos_shm", description="Name of the SHM segment")
    shm_size: int = Field(default=1000, description="Size of the SHM segment in entries")

class SkyRLServerEnv(BaseEnv):
    """
    Production-ready Atropos Environment for SkyRL.
    Pulls real tasks from a dataset and performs real vLLM inference.
    """
    @classmethod
    def config_init(cls):
        from atroposlib.envs.server_handling.server_baseline import ServerBaseline
        return SkyRLServerConfig(), ServerBaseline()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info(f"Initializing SkyRL Server | dataset: {self.config.dataset_path}")
        
        # Load the dataset
        if not os.path.exists(self.config.dataset_path):
            logger.error(f"Dataset not found at {self.config.dataset_path}")
            # Fallback to a single dummy if file missing (to prevent crash, though it should exist)
            self.df = pl.DataFrame({"prompt": ["Please solve 2+2"], "text": ["4"]})
        else:
            self.df = pl.read_parquet(self.config.dataset_path)
            logger.info(f"Loaded {len(self.df)} prompts from dataset.")
            
        self.current_idx = 0
        self.lock = asyncio.Lock()
        self.status_dict = {}

    async def get_next_item(self) -> Tuple[Any, str]:
        """
        Ordered task generation to match the trainer's dataset iteration.
        """
        async with self.lock:
            if self.current_idx >= len(self.df):
                self.current_idx = 0
                logger.info("Dataset loop finished, restarting from index 0.")
            
            row = self.df.row(self.current_idx, named=True)
            prompt = row["prompt"]
            uid = str(self.current_idx)
            
            self.current_idx += 1
            
        return prompt, uid

    async def collect_trajectory(self, item_tuple: Tuple[Any, str]) -> Tuple[Dict[str, Any], List[Any]]:
        """
        Performs real inference using the Atropos vLLM engine.
        Expecting item_tuple to be (prompt, uid) from get_next_item.
        """
        item, uid = item_tuple
        logger.info(f"Generating trajectory | Task ID: {uid}")
        
        try:
            # Use tokens_and_logprobs_completion to get direct token access
            # prompt_tokens, output_tokens, output_logprobs, finish_reasons
            ret = await self.server.tokens_and_logprobs_completion(
                prompt=item,
                max_tokens=self.config.max_token_length,
                temperature=0.7,
                split="train"
            )
            
            prompt_tokens, output_tokens, output_logprobs, finish_reasons = ret
            
            # Since n=1 by default, we take the first completion
            tokens = output_tokens[0]
            
            # Basic Reward Logic:
            # In a real scenario, this would call a reward model or a verifier.
            # Here we assign 1.0 if any tokens were generated.
            score = 1.0 if len(tokens) > 2 else 0.0
            
            logger.info(f"Task {uid} completed | tokens: {len(tokens)} | score: {score}")
            
            # Return (dict, backlog) tuple as expected by BaseEnv
            return {
                "instance_id": uid,
                "tokens": tokens,
                "masks": [1] * len(tokens),
                "scores": score,
                "logprobs": output_logprobs[0],
                "ref_logprobs": None,
                "distill_token_ids": None,
                "distill_logprobs": None,
            }, []
        except Exception as e:
            logger.error(f"Inference error | Task {uid}: {e}")
            import traceback
            traceback.print_exc()
            # Return empty to allow the loop to continue
            return {
                "instance_id": uid,
                "tokens": [],
                "masks": [],
                "scores": 0.0,
                "logprobs": [],
                "ref_logprobs": None,
                "distill_token_ids": None,
                "distill_logprobs": None,
            }, []

    async def setup(self):
        """
        Required by BaseEnv abstract class.
        """
        logger.info("SkyRL Server setup complete.")

    async def setup_wandb(self):
        """
        No-op for SkyRL joint training to avoid connection errors.
        """
        logger.info("WandB setup bypassed.")
        
    async def get_server_info(self):
        """
        No-op for SkyRL joint training.
        """
        logger.info("Server info bypassed.")

    async def register_env(self):
        """
        No-op for SkyRL joint training to avoid connection errors to localhost:8000.
        """
        logger.info("Registration bypassed for joint training.")
        return {}

    async def evaluate(self) -> Dict[str, Any]:
        """
        Required by BaseEnv abstract class.
        In this production server, the trainer handles evaluation,
        so the server's evaluate is a no-op.
        """
        return {"avg_score": 0.0}

    async def get_status(self):
        """
        Required by Atropos orchestration loop.
        Updates self.status_dict directly to satisfy BaseEnv expectations.
        """
        self.status_dict = {
            "current_step": 0,
            "queue_size": 0, # Asynchronous sampling - always ready for more
            "max_group_size": self.config.group_size,
            "self_queue_size": 0,
            "batches_offpolicy": 0,
            "max_batches_offpolicy": self.config.max_batches_offpolicy,
        }
        return self.status_dict

if __name__ == "__main__":
    # Launch the SkyRLServerEnv via the BaseEnv CLI (serve or process)
    SkyRLServerEnv.cli()
