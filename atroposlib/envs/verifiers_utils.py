from __future__ import annotations

from typing import Any, Union

from atroposlib.envs.base import APIServerConfig
from atroposlib.envs.server_handling.server_baseline import ServerBaseline
from atroposlib.type_definitions import Message


def normalize_vf_env_id(env_id: str) -> str:
    env_id = (env_id or "").strip()
    if not env_id:
        raise ValueError(
            "env.vf_env_name must be set to a Prime Env Hub id like 'owner/environment-name' "
            "(or an installed verifiers env id like 'environment-name')."
        )
    env_id = env_id.split("@", 1)[0].strip()
    if "/" in env_id:
        env_id = env_id.rsplit("/", 1)[-1].strip()
    if not env_id:
        raise ValueError(
            "env.vf_env_name must contain an environment name, e.g. 'owner/environment-name' or 'environment-name'."
        )
    return env_id


def reward_scales(weights: list[float]) -> list[float]:
    if not weights:
        return []
    total = sum(weights)
    if total <= 0:
        return [1.0 / len(weights) for _ in weights]
    return [w / total for w in weights]


def weighted_sum(rewards: list[float], scales: list[float]) -> float:
    if not rewards:
        return 0.0
    if not scales:
        return float(sum(rewards))
    return float(sum(r * s for r, s in zip(rewards, scales)))


def last_assistant_text(messages: list[dict[str, Any]] | None) -> str:
    if not messages:
        return ""
    for msg in reversed(messages):
        if msg.get("role") in ("assistant", "agent") and isinstance(
            msg.get("content"), str
        ):
            return msg["content"]
    return ""


def sanitize_messages(messages: list[dict[str, Any]]) -> list[Message]:
    sanitized: list[Message] = []
    for msg in messages:
        role = msg.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            continue
        sanitized.append({"role": role, "content": msg.get("content", "")})
    return sanitized


def infer_model_name(
    server_configs: Union[ServerBaseline, list[APIServerConfig], APIServerConfig],
) -> str:
    if isinstance(server_configs, list) and server_configs:
        return server_configs[0].model_name
    if isinstance(server_configs, APIServerConfig):
        return server_configs.model_name
    return getattr(server_configs, "model_name", "model")
