from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings

from atroposlib.envs.base import BaseEnvConfig


class EVMEnvConfig(BaseEnvConfig, BaseSettings):
    """Configuration for the EVM Environment"""

    # Logging configuration
    debug_logging: bool = Field(
        default=False, description="Enable detailed debug logging"
    )
    suppress_base_env_logs: bool = Field(
        default=True,
        description="Suppress base environment INFO logs to reduce noise",
    )

    # Anvil configuration
    anvil_config_path: str = Field(
        "configs/token_transfers.yaml",
        description="Path to Anvil configuration YAML file",
    )
    max_steps: int = Field(1, description="Only one step per transaction episode")
    question_types: List[str] = Field(
        default=[
            "ETH transfer",
            "ERC-20 transfer using 18 decimal token",
            "ERC-20 transfer using a non-18 decimal token",
        ],
        description="Types of questions to generate for the agent",
    )

    # Question selection strategy configuration
    weak_performance_threshold: float = Field(
        default=0.9,
        description="Performance threshold below which question types are considered weak (0.0-1.0)",
    )
    weak_area_focus_ratio: float = Field(
        default=0.8,
        description="Probability of focusing on weak areas vs strong areas (0.0-1.0)",
    )

    # LLM generation configuration for dynamic questions
    question_generation_model: str = Field(
        default="gpt-4o-mini",
        description="Model to use for generating dynamic questions",
    )
    question_generation_temperature: float = Field(
        default=0.6,
        description="Temperature for question generation (0.0-2.0)",
    )
    question_generation_max_tokens: int = Field(
        default=256,
        description="Maximum tokens for question generation",
    )
    question_generation_n: int = Field(
        default=3,
        description="Number of responses to generate per question generation call",
    )

    class Config:
        env_file = "configs/token_transfers.yaml"
        env_file_encoding = "utf-8"
