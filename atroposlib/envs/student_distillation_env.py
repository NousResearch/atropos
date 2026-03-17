"""
Student/self-distillation environment layer.

This module adds prompt-logprob fetching from the student rollout server itself
before the scored group is sent to the API.

By default, the student server scores the exact token IDs already present in the
group. Override-driven prompt/message scoring is supported only when the
resulting prompt tokenization matches the original token sequence exactly.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import Field

from .base import BaseEnv, BaseEnvConfig, ScoredDataGroup

logger = logging.getLogger(__name__)


class StudentDistillationConfig(BaseEnvConfig):
    student_distill_enabled: bool = Field(
        default=False,
        description="Whether to fetch prompt logprobs from the student server itself.",
    )
    student_top_k: int = Field(
        default=0,
        ge=-1,
        description=(
            "Number of extra prompt logprobs to fetch beyond the selected token. "
            "Use 0 for selected-token-only prompt logprobs and <= -1 to disable "
            "student distillation fetching."
        ),
    )


class StudentDistillationEnv(BaseEnv, ABC):
    """
    BaseEnv subclass that enriches scored groups with self-distillation arrays.

    Distillation payload shape:
      - distill_token_ids: [sequence][position][k]
      - distill_logprobs:  [sequence][position][k]
    """

    env_config_cls = StudentDistillationConfig

    def _get_student_logprob_overrides(
        self, group: ScoredDataGroup, seq_idx: int
    ) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}

        group_overrides = group.get("group_overrides") or {}
        group_kwargs = group_overrides.get(
            "student_logprob_kwargs",
            group_overrides.get("student_distill_kwargs"),
        )
        if isinstance(group_kwargs, dict):
            merged.update(group_kwargs)

        overrides = group.get("overrides") or []
        if seq_idx < len(overrides):
            seq_overrides = overrides[seq_idx] or {}
            seq_kwargs = seq_overrides.get(
                "student_logprob_kwargs",
                seq_overrides.get("student_distill_kwargs"),
            )
            if isinstance(seq_kwargs, dict):
                merged.update(seq_kwargs)

        return merged

    async def _fetch_student_for_sequence(
        self,
        token_ids: List[int],
        top_k: int,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[List[int]], List[List[float]]]:
        request_kwargs: Dict[str, Any] = {
            "input_ids": token_ids,
            "top_k": top_k,
            "max_tokens": 1,
            "split": "train",
        }
        if extra_kwargs:
            request_kwargs.update(extra_kwargs)
            if extra_kwargs.get("messages") is not None or extra_kwargs.get(
                "prompt"
            ) is not None:
                # Let message/prompt overrides drive tokenization instead of the
                # original input_ids when callers explicitly request it.
                request_kwargs.pop("input_ids", None)

        if request_kwargs.get("messages") is not None:
            async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
                payload = await managed.get_logprobs(**request_kwargs)
        else:
            payload = await self.server.get_logprobs(**request_kwargs)

        if payload.get("prompt_tokens") != token_ids:
            raise ValueError(
                "Student distillation request did not align to the original token "
                "sequence. Override-driven prompt/messages are only supported when "
                "they reproduce the exact same prompt tokens."
            )

        return payload["prompt_topk_token_ids"], payload["prompt_topk_logprobs"]

    async def _attach_student_distillation(
        self, group: ScoredDataGroup
    ) -> ScoredDataGroup:
        if not self.config.student_distill_enabled:
            return group

        seqs = group.get("tokens", [])
        if not seqs:
            group["distill_token_ids"] = None
            group["distill_logprobs"] = None
            return group

        group_overrides = group.get("group_overrides") or {}
        if group_overrides.get("skip_student_top_k", False):
            group["distill_token_ids"] = None
            group["distill_logprobs"] = None
            return group

        top_k = int(group_overrides.get("student_top_k", self.config.student_top_k))
        if top_k <= -1:
            group["distill_token_ids"] = None
            group["distill_logprobs"] = None
            return group

        tasks = [
            self._fetch_student_for_sequence(
                seq,
                top_k,
                self._get_student_logprob_overrides(group, idx),
            )
            for idx, seq in enumerate(seqs)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        distill_token_ids: List[List[List[int]]] = []
        distill_logprobs: List[List[List[float]]] = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(
                    "Student logprob fetch failed for seq %s: %s. "
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
                    "Student prompt-topk length mismatch for seq %s (%s != %s). "
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
            enriched_groups.append(await self._attach_student_distillation(group))

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
