import logging
from typing import Any, Dict, List, Optional, Union

from pydantic import Field

from ..type_definitions import Message
from .base import BaseEnv, BaseEnvConfig, ScoredDataGroup
from .server_handling.managed_server import ManagedServerEnv, ManagedServerEnvConfig

logger = logging.getLogger(__name__)


class SkyRLConfig(ManagedServerEnvConfig):
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


class SkyRLAdapter(ManagedServerEnv):
    """
    Atropos Adapter for Berkeley's SkyRL (NovaSky-AI) environments.
    
    This adapter bridges the SkyRL-gym trajectory format (Thinking Traces + PRM)
    into the Atropos orchestration layer.
    """

    name = "skyrl"
    env_config_cls = SkyRLConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info(f"Initialized SkyRLAdapter with repo: {self.config.skyrl_repo_id}")

    async def postprocess_histories(
        self, histories: List[List[Message]]
    ) -> List[ScoredDataGroup]:
        """
        Extends the baseline post-processing to extract reasoning traces and step-wise rewards.
        """
        # Call the base managed_server logic to get initial scores
        base_groups = await super().postprocess_histories(histories)

        for group in base_groups:
            if not group or "messages" not in group:
                continue

            # Add SkyRL-specific metadata container
            if "env_metrics" not in group:
                group["env_metrics"] = {}

            # Phase 1: Reasoning Trace Extraction
            # Extract <think>...</think> blocks from the model's responses
            for rollout_idx, messages in enumerate(group["messages"]):
                if not messages:
                    continue

                # The last message is typically the model's response
                last_msg = messages[-1]
                content = last_msg.get("content", "")

                if self.config.thought_start_tag in content:
                    start_idx = content.find(self.config.thought_start_tag) + len(
                        self.config.thought_start_tag
                    )
                    end_idx = content.find(self.config.thought_end_tag)
                    
                    if end_idx != -1:
                        thinking_trace = content[start_idx:end_idx].strip()
                        # Inject into the group metadata for the trainer to consume
                        if "reasoning_traces" not in group["env_metrics"]:
                            group["env_metrics"]["reasoning_traces"] = []
                        group["env_metrics"]["reasoning_traces"].append(thinking_trace)

            # Phase 2: Process Reward Mapping
            # In Phase 1, we simulate step-wise rewards if the baseline environment
            # provides them in the 'overrides' or 'metadata' field.
            # (In Phase 2/3, we will stream these through the SHM Pinhole).
            if self.config.enable_process_rewards:
                # Place-holder for vectorized rewards
                # In Berkeley SkyRL, this maps to TrajectoryOutput.reward (List[float])
                group["env_metrics"]["prm_supported"] = True

        return base_groups

    def get_server_command(self) -> List[str]:
        """
        Command to launch the SkyRL-gym execution sidecar.
        """
        cmd = super().get_server_command()
        # Ensure we are passing the skyrl-specific flags to the sidecar
        cmd.extend(["--repo_id", self.config.skyrl_repo_id])
        return cmd
