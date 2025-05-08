from typing import Literal, Optional

from pydantic import BaseModel, Field


class ServerBaseline(BaseModel):
    """
    Baseline configuration for server information. If local, uses ports 9004-9007 for the servers,
    assuming a 1:1 split of GPUs.
    """

    timeout: int = Field(
        default=1200, description="Timeout for the request in seconds."
    )
    num_max_requests_at_once: int = Field(
        default=512,
        description="Maximum number of concurrent requests. You should divide this by the n kwarg.",
    )
    num_requests_for_eval: int = Field(
        default=64, description="Maximum number of concurrent requests for evaluation."
    )
    model_name: str = Field(
        default="default",
        description="The model name to use. Only works with sglang, please provide the model name.",
    )
    rolling_buffer_length: int = Field(
        default=1000, description="Length of the rolling buffer to store metrics."
    )


class APIServerConfig(ServerBaseline):
    """
    API server configuration.
    """

    api_key: Optional[str] = Field(default="", description="API key for the server.")
    base_url: Optional[str] = Field(default="", description="Base URL for the server.")
    server_type: Literal["openai"] = Field(
        default="openai", description="Type of server to use, openai or trl"
    )
