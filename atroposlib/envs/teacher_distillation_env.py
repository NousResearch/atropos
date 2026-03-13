"""
Teacher distillation environment layer.

This module adds teacher prompt-logprob fetching on top of BaseEnv without
modifying BaseEnv transport behavior.

This implementation supports same-tokenizer distillation only. The teacher and
student must share the same tokenizer vocabulary so the student's token IDs can
be forwarded directly to the teacher and the returned teacher top-k token IDs
can be looked up directly in the student's logits.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC
from typing import Any, List, Optional, Tuple, Union

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
    teacher_server: Optional[APIServerConfig] = Field(
        default=None,
        description="Teacher inference server configuration.",
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
      - distill_token_ids: [sequence][position][k]  (student vocab IDs)
      - distill_logprobs:  [sequence][position][k]
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
            if config.teacher_server is None:
                raise ValueError(
                    "teacher_enabled=True requires a teacher_server configuration."
                )
            teacher_cfg = config.teacher_server.model_copy(
                update={
                    "tokenizer_name": (
                        config.teacher_server.model_name
                        if config.teacher_server.tokenizer_name in ("", "none")
                        else config.teacher_server.tokenizer_name
                    ),
                    "timeout": 1200,
                }
            )
            self.teacher_server = ServerManager(
                [teacher_cfg],
                slurm=False,
                testing=False,
            )
            self._validate_teacher_tokenizer_compatibility(teacher_cfg.tokenizer_name)

    # ------------------------------------------------------------------
    # Core fetch
    # ------------------------------------------------------------------

    def _validate_teacher_tokenizer_compatibility(
        self, teacher_tokenizer_name: str
    ) -> None:
        student_tok_name = getattr(self.tokenizer, "name_or_path", None) or ""
        if student_tok_name == teacher_tokenizer_name:
            return

        try:
            from transformers import AutoTokenizer

            teacher_tokenizer = AutoTokenizer.from_pretrained(
                teacher_tokenizer_name, use_fast=True
            )
        except Exception as exc:
            raise ValueError(
                "Cross-tokenizer distillation is not supported in this PR, and the "
                f"teacher tokenizer for '{teacher_tokenizer_name}' could not be loaded to "
                f"verify compatibility: {exc}"
            ) from exc

        student_vocab = self.tokenizer.get_vocab()
        teacher_vocab = teacher_tokenizer.get_vocab()
        if student_vocab != teacher_vocab:
            raise ValueError(
                "Cross-tokenizer distillation is not supported in this PR. "
                f"Student tokenizer '{student_tok_name or type(self.tokenizer).__name__}' "
                f"and teacher tokenizer '{teacher_tokenizer_name}' do not match."
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

    # ------------------------------------------------------------------
    # Group enrichment
    # ------------------------------------------------------------------

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
