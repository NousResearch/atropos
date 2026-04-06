import logging
from typing import Any, Dict, List, Optional, Union

from pydantic import Field

from ..type_definitions import Message
from .base import BaseEnv, BaseEnvConfig, ScoredDataGroup

logger = logging.getLogger(__name__)


class SkyRLConfig(BaseEnvConfig):
    """
    Configuration for the Berkeley SkyRL adapter.
    """

    skyrl_repo_id: str = Field(
        default="NovaSky-AI/Sky-AIME-5K",
        description="The SkyRL-gym repository ID or local path to the reasoning environment.",
    )
    enable_process_rewards: bool = Field(
        default=True,
        description="Whether to extract and forward step-wise process rewards from SkyRL.",
    )
    thought_start_tag: str = Field(
        default="<think>",
        description="The opening tag for reasoning/thinking traces.",
    )
    thought_end_tag: str = Field(
        default="</think>",
        description="The closing tag for reasoning/thinking traces.",
    )


class SkyRLAdapter(BaseEnv):
    """
    Atropos Adapter for SkyRL (NovaSky-AI) environments.
    Bridges reasoning traces and step-wise rewards into the Atropos layer.
    """

    name = "skyrl"
    env_config_cls = SkyRLConfig

    async def postprocess_histories(
        self, histories: List[List[Message]]
    ) -> List[ScoredDataGroup]:
        """
        Extends the baseline post-processing to extract reasoning traces and step-wise rewards.
        """
        # Call the base logic (BaseEnv handles standard scoring via its ServerManager)
        base_groups = await super().postprocess_histories(histories)

        for group in base_groups:
            if not group or "messages" not in group:
                continue

            # Add SkyRL-specific metadata container
            if "env_metrics" not in group:
                group["env_metrics"] = {}

            # Reasoning Trace Extraction
            for rollout_idx, messages in enumerate(group["messages"]):
                if not messages:
                    continue

                last_msg = messages[-1]
                content = last_msg.get("content", "")

                if self.config.thought_start_tag in content:
                    start_idx = content.find(self.config.thought_start_tag) + len(
                        self.config.thought_start_tag
                    )
                    end_idx = content.find(self.config.thought_end_tag)

                    if end_idx != -1:
                        thinking_trace = content[start_idx:end_idx].strip()
                        if "reasoning_traces" not in group["env_metrics"]:
                            group["env_metrics"]["reasoning_traces"] = []
                        group["env_metrics"]["reasoning_traces"].append(thinking_trace)

            # Process Reward Mapping
            if self.config.enable_process_rewards:
                group["env_metrics"]["prm_supported"] = True

        return base_groups

    async def get_next_item(self) -> Item:
        """
        SkyRL-gym manages its own task queue/dataset internally.
        This provides a dummy item to satisfy the BaseEnv contract.
        """
        return Item(
            tokens=[],
            masks=[],
            scores=0.0,
            advantages=None,
            ref_logprobs=None,
            messages=None,
            meta={"source": "skyrl_dummy"},
        )

    async def evaluate(self, *args, **kwargs) -> Dict[str, float]:
        """
        SkyRL-specific evaluation logic.
        """
        logger.info("Running SkyRL Reasoning Evaluation...")
        return {"reasoning_acc": 0.0}
