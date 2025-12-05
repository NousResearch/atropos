"""Utility functions for server configuration management.

This module provides shared utilities for resolving and validating server configurations
across different server implementations (VLLM, SGLang, OpenAI).
"""

from typing import Any, Dict, List, Union

from pydantic_cli import FailedExecutionException

from atroposlib.envs.constants import NAMESPACE_SEP, OPENAI_NAMESPACE
from atroposlib.envs.server_handling.server_baseline import APIServerConfig


def resolve_openai_configs(
    default_server_configs: Any,
    openai_config_dict: Dict[str, Any],
    yaml_config: Dict[str, Any],
    cli_passed_flags: Dict[str, Any],
    logger: Any,
) -> Union[APIServerConfig, List[APIServerConfig], Any]:
    """
    Helper to resolve the final server_configs, handling single, multiple servers, and overrides.

    Args:
        default_server_configs: Default server configuration(s) - can be a single config,
                                list of configs, or ServerBaseline
        openai_config_dict: Dictionary of OpenAI configuration settings
        yaml_config: Full YAML configuration dictionary
        cli_passed_flags: Dictionary of CLI flags passed by the user
        logger: Logger instance for logging messages

    Returns:
        Resolved server configuration(s) - can be a single APIServerConfig,
        list of APIServerConfig, or ServerBaseline

    Raises:
        FailedExecutionException: If configuration is invalid or CLI overrides are used
                                  with multi-server setup
    """
    from atroposlib.envs.server_handling.server_manager import ServerBaseline

    openai_full_prefix = f"{OPENAI_NAMESPACE}{NAMESPACE_SEP}"
    openai_yaml_config = yaml_config.get(OPENAI_NAMESPACE, None)
    openai_cli_config = {
        k: v for k, v in cli_passed_flags.items() if k.startswith(openai_full_prefix)
    }

    is_multi_server_yaml = (
        isinstance(openai_yaml_config, list) and len(openai_yaml_config) >= 2
    )
    is_multi_server_default = (
        (not is_multi_server_yaml)
        and isinstance(default_server_configs, list)
        and len(default_server_configs) >= 2
    )

    if (is_multi_server_yaml or is_multi_server_default) and openai_cli_config:
        raise FailedExecutionException(
            message=f"CLI overrides for OpenAI settings (--{openai_full_prefix}*) are not supported "
            f"when multiple servers are defined (either via YAML list under '{OPENAI_NAMESPACE}' "
            "or a default list with length >= 2).",
            exit_code=2,
        )

    if is_multi_server_yaml:
        logger.info(
            f"Using multi-server configuration defined in YAML under '{OPENAI_NAMESPACE}'."
        )
        try:
            server_configs = [APIServerConfig(**cfg) for cfg in openai_yaml_config]
        except Exception as e:
            raise FailedExecutionException(
                f"Error parsing multi-server OpenAI configuration from YAML under '{OPENAI_NAMESPACE}': {e}"
            ) from e
    elif isinstance(default_server_configs, ServerBaseline):
        logger.info("Using ServerBaseline configuration.")
        server_configs = default_server_configs
    elif is_multi_server_default:
        logger.info("Using default multi-server configuration (length >= 2).")
        server_configs = default_server_configs
    else:
        logger.info(
            "Using single OpenAI server configuration based on merged settings (default/YAML/CLI)."
        )
        try:
            final_openai_config = APIServerConfig(**openai_config_dict)
        except Exception as e:
            raise FailedExecutionException(
                f"Error creating final OpenAI configuration from merged settings: {e}\n"
                f"Merged Dict: {openai_config_dict}"
            ) from e

        if isinstance(default_server_configs, APIServerConfig):
            server_configs = final_openai_config
        elif isinstance(default_server_configs, list):
            server_configs = [final_openai_config]
        else:
            logger.warning(
                f"Unexpected type for default_server_configs: {type(default_server_configs)}. "
                f"Proceeding with single OpenAI server configuration based on merged settings."
            )
            server_configs = [final_openai_config]

    return server_configs
