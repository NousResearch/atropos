"""
Teacher distillation environment layer.

This module adds teacher prompt-logprob fetching on top of BaseEnv without
modifying BaseEnv transport behavior.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import Field

from .base import BaseEnv, BaseEnvConfig, ScoredDataGroup
from .server_handling.server_baseline import APIServerConfig, ServerBaseline
from .server_handling.server_manager import ServerManager

logger = logging.getLogger(__name__)


class TeacherDistillationConfig(BaseEnvConfig):
    teacher_enabled: bool = Field(
        default=False,
        description="Whether to fetch teacher prompt logprobs for distillation.",
    )
    teacher_base_url: Optional[str] = Field(
        default=None,
        description="Teacher server base URL (OpenAI-compatible).",
    )
    teacher_model_name: Optional[str] = Field(
        default=None,
        description="Teacher model name used in teacher server requests.",
    )
    teacher_api_key: str = Field(
        default="",
        description="Teacher API key, if required by the teacher endpoint.",
    )
    teacher_server_type: str = Field(
        default="vllm",
        description="Teacher server type (e.g. vllm, sglang, trl, openai).",
    )
    teacher_tokenizer_name: str = Field(
        default="none",
        description=(
            "Tokenizer name for teacher server. If 'none', teacher_model_name is used."
        ),
    )
    teacher_top_k: int = Field(
        default=1,
        ge=1,
        description="Top-k prompt logprobs to fetch per token position.",
    )


class TeacherDistillationEnv(BaseEnv, ABC):
    """
    BaseEnv subclass that enriches scored groups with teacher distillation arrays.

    Distillation payload shape:
      - distill_token_ids: [sequence][position][k]
      - distill_logprobs: [sequence][position][k]
    """

    env_config_cls = TeacherDistillationConfig

    def __init__(
        self,
        config: TeacherDistillationConfig,
        server_configs: Union[ServerBaseline, List[APIServerConfig]],
        slurm: bool = False,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm=slurm, testing=testing)
        self.teacher_server: Optional[ServerManager] = None
        if config.teacher_enabled:
            if not config.teacher_base_url or not config.teacher_model_name:
                raise ValueError(
                    "teacher_enabled=True requires teacher_base_url and teacher_model_name."
                )
            teacher_cfg = APIServerConfig(
                server_type=config.teacher_server_type,  # type: ignore[arg-type]
                base_url=config.teacher_base_url,
                api_key=config.teacher_api_key,
                model_name=config.teacher_model_name,
                tokenizer_name=config.teacher_tokenizer_name,
                timeout=1200,
            )
            self.teacher_server = ServerManager(
                [teacher_cfg],
                slurm=False,
                testing=False,
            )
            logger.warning(
                "TeacherDistillationEnv: teacher server configured at %s "
                "(model=%s, top_k=%s)",
                config.teacher_base_url,
                config.teacher_model_name,
                config.teacher_top_k,
            )

    async def _fetch_teacher_for_sequence(
        self, token_ids: List[int], top_k: int
    ) -> Tuple[List[List[int]], List[List[float]]]:
        assert self.teacher_server is not None
        payload = await self.teacher_server.get_logprobs(
            input_ids=token_ids,
            top_k=top_k,
            max_tokens=1,
            split="train",
        )
        return payload["prompt_topk_token_ids"], payload["prompt_topk_logprobs"]

    async def _attach_teacher_distillation(
        self, group: ScoredDataGroup
    ) -> ScoredDataGroup:
        if not self.config.teacher_enabled or self.teacher_server is None:
            return group

        seqs = group.get("tokens", [])
        if not seqs:
            group["distill_token_ids"] = None
            group["distill_logprobs"] = None
            return group

        top_k = int(
            (group.get("group_overrides") or {}).get(
                "teacher_top_k", self.config.teacher_top_k
            )
        )
        top_k = max(1, top_k)

        tasks = [self._fetch_teacher_for_sequence(seq, top_k) for seq in seqs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        distill_token_ids: List[List[List[int]]] = []
        distill_logprobs: List[List[List[float]]] = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(
                    "Teacher logprob fetch failed for seq %s: %s. "
                    "Dropping distill payload for this group.",
                    idx,
                    result,
                )
                group["distill_token_ids"] = None
                group["distill_logprobs"] = None
                return group
            token_ids_k, logprobs_k = result
            if len(token_ids_k) != len(logprobs_k):
                logger.warning(
                    "Teacher prompt-topk length mismatch for seq %s (%s != %s). "
                    "Dropping distill payload for this group.",
                    idx,
                    len(token_ids_k),
                    len(logprobs_k),
                )
                group["distill_token_ids"] = None
                group["distill_logprobs"] = None
                return group
            distill_token_ids.append(token_ids_k)
            distill_logprobs.append(logprobs_k)

        group["distill_token_ids"] = distill_token_ids
        group["distill_logprobs"] = distill_logprobs
        return group

    async def handle_send_to_api(
        self,
        scored_data: Union[ScoredDataGroup, List[ScoredDataGroup]],
        item: Any = None,
        do_send_to_api: bool = True,
        abort_on_any_max_length_exceeded: bool = True,
    ):
        groups = scored_data if isinstance(scored_data, list) else [scored_data]
        enriched_groups: List[ScoredDataGroup] = []
        for group in groups:
            if group is None:
                continue
            enriched_groups.append(await self._attach_teacher_distillation(group))

        payload: Union[ScoredDataGroup, List[ScoredDataGroup]]
        if isinstance(scored_data, list):
            payload = enriched_groups
        else:
            payload = enriched_groups[0] if enriched_groups else scored_data

        return await super().handle_send_to_api(
            payload,
            item=item,
            do_send_to_api=do_send_to_api,
            abort_on_any_max_length_exceeded=abort_on_any_max_length_exceeded,
        )
