"""
Teacher distillation environment layer.

This module adds teacher prompt-logprob fetching on top of BaseEnv without
modifying BaseEnv transport behavior.

Cross-tokenizer distillation
----------------------------
When student and teacher use the same tokenizer family (e.g. both Qwen3) the
student's raw token IDs can be forwarded directly to the teacher vLLM and the
returned top-k token IDs can be used as-is in the student logit lookup.

When tokenizers differ (e.g. Llama student, Qwen teacher) two problems arise:

  1. Token-ID aliasing: student token 42 = " the" in Llama, but 42 = "ท" in
     Qwen.  Sending student IDs to the teacher causes it to score garbage.

  2. Vocab-space mismatch: the teacher's top-k IDs live in the teacher's
     vocabulary.  The student logit lookup at those IDs would access random
     tokens in the student vocab.

This module fixes both problems automatically:

  • Re-tokenization  – student tokens are decoded to plain text and
    re-tokenized with the teacher tokenizer before being sent to the teacher
    server.  The teacher therefore always scores the correct text.

  • Character-level position alignment – after re-tokenisation the teacher
    has a different number of tokens than the student.  A character-offset
    map is built (requires a fast HuggingFace tokenizer) to project each
    teacher logprob position back onto the student token it overlaps with.

  • Vocabulary remapping – teacher top-k token IDs (teacher vocab) are
    decoded to text fragments and re-encoded with the student tokenizer so
    that the final distill_token_ids live in the student vocabulary and can
    be looked up directly in the student logit tensor.

Same-tokenizer fast path
------------------------
When teacher_tokenizer_name resolves to the same underlying vocabulary as the
student tokenizer the original fast path (no decode / re-tokenize / remap) is
taken automatically.
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
            "Tokenizer name for teacher server. If 'none', teacher_model_name is used. "
            "When this resolves to a different vocabulary than the student tokenizer, "
            "cross-tokenizer alignment is applied automatically."
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
        # Teacher tokenizer (only loaded when tokenizers may differ).
        self._teacher_tokenizer = None
        # True when student and teacher share the same vocabulary.
        self._same_tokenizer: bool = True
        # LRU-style cache: teacher_token_id -> student_token_id
        self._vocab_remap_cache: Dict[int, int] = {}

        if config.teacher_enabled:
            if not config.teacher_base_url or not config.teacher_model_name:
                raise ValueError(
                    "teacher_enabled=True requires teacher_base_url and teacher_model_name."
                )
            teacher_tok_name = (
                config.teacher_model_name
                if config.teacher_tokenizer_name in ("none", "")
                else config.teacher_tokenizer_name
            )
            teacher_cfg = APIServerConfig(
                server_type=config.teacher_server_type,  # type: ignore[arg-type]
                base_url=config.teacher_base_url,
                api_key=config.teacher_api_key,
                model_name=config.teacher_model_name,
                tokenizer_name=teacher_tok_name,
                timeout=1200,
            )
            self.teacher_server = ServerManager(
                [teacher_cfg],
                slurm=False,
                testing=False,
            )

            # Detect vocabulary mismatch.
            # Compare by name first; if names differ, load the teacher tokenizer
            # and do a vocab-size sanity check.  Same-family models (e.g. Qwen3-4B
            # and Qwen3-30B) share the same vocabulary, so even though the
            # name_or_path strings differ they should use the fast path.
            student_tok_name = getattr(self.tokenizer, "name_or_path", None) or ""
            if student_tok_name and teacher_tok_name and student_tok_name != teacher_tok_name:
                try:
                    from transformers import AutoTokenizer

                    loaded = AutoTokenizer.from_pretrained(
                        teacher_tok_name, use_fast=True
                    )
                    student_vocab_size = getattr(self.tokenizer, "vocab_size", None)
                    teacher_vocab_size = getattr(loaded, "vocab_size", None)
                    if (
                        student_vocab_size is not None
                        and teacher_vocab_size is not None
                        and student_vocab_size == teacher_vocab_size
                    ):
                        # Same vocab size — treat as same tokenizer (fast path).
                        # This covers same-family models (e.g. all Qwen3 variants).
                        self._same_tokenizer = True
                        logger.warning(
                            "TeacherDistillationEnv: names differ but vocab sizes match "
                            "(%d tokens). Using fast (same-tokenizer) path. "
                            "student=%s  teacher=%s",
                            student_vocab_size,
                            student_tok_name,
                            teacher_tok_name,
                        )
                    else:
                        self._teacher_tokenizer = loaded
                        self._same_tokenizer = False
                        logger.warning(
                            "TeacherDistillationEnv: cross-tokenizer mode active. "
                            "student=%s (%s tokens)  teacher=%s (%s tokens). "
                            "Token IDs will be decoded → re-tokenized → vocab-remapped.",
                            student_tok_name,
                            student_vocab_size,
                            teacher_tok_name,
                            teacher_vocab_size,
                        )
                except Exception as exc:
                    logger.warning(
                        "TeacherDistillationEnv: could not load teacher tokenizer '%s' "
                        "(%s). Falling back to same-tokenizer (fast) path — only safe if "
                        "student and teacher share the same vocabulary.",
                        teacher_tok_name,
                        exc,
                    )
                    self._same_tokenizer = True
            else:
                self._same_tokenizer = True

            logger.warning(
                "TeacherDistillationEnv: teacher server configured at %s "
                "(model=%s, top_k=%s, same_tokenizer=%s)",
                config.teacher_base_url,
                config.teacher_model_name,
                config.teacher_top_k,
                self._same_tokenizer,
            )

    # ------------------------------------------------------------------
    # Cross-tokenizer helpers
    # ------------------------------------------------------------------

    def _build_student_teacher_alignment(
        self,
        text: str,
        student_ids: List[int],
        teacher_ids: List[int],
    ) -> List[List[int]]:
        """
        For each student token position return the list of teacher token positions
        whose character spans overlap with the student token's character span.

        Requires fast (Rust-backed) HuggingFace tokenizers that support
        return_offsets_mapping.  Falls back to a proportional approximation
        if offset mapping is unavailable.
        """
        student_len = len(student_ids)
        teacher_len = len(teacher_ids)

        try:
            s_enc = self.tokenizer(
                text, return_offsets_mapping=True, add_special_tokens=False
            )
            t_enc = self._teacher_tokenizer(
                text, return_offsets_mapping=True, add_special_tokens=False
            )
            s_offsets: List[Tuple[int, int]] = s_enc["offset_mapping"][:student_len]
            t_offsets: List[Tuple[int, int]] = t_enc["offset_mapping"][:teacher_len]

            alignment: List[List[int]] = []
            for s_start, s_end in s_offsets:
                overlapping = [
                    t_idx
                    for t_idx, (t_start, t_end) in enumerate(t_offsets)
                    if t_start < s_end and t_end > s_start and s_end > s_start
                ]
                alignment.append(overlapping)
            return alignment

        except Exception as exc:
            logger.warning(
                "TeacherDistillationEnv: offset-mapping alignment failed (%s). "
                "Using proportional fallback.",
                exc,
            )
            ratio = teacher_len / max(student_len, 1)
            return [[int(i * ratio)] for i in range(student_len)]

    def _remap_teacher_token_to_student(self, teacher_token_id: int) -> int:
        """
        Convert a teacher vocabulary token ID to the best-matching student
        vocabulary token ID by decoding the teacher token to text then
        re-encoding with the student tokenizer.

        Results are cached to avoid repeated tokenizer calls.
        """
        if teacher_token_id in self._vocab_remap_cache:
            return self._vocab_remap_cache[teacher_token_id]

        try:
            text = self._teacher_tokenizer.decode(
                [teacher_token_id], clean_up_tokenization_spaces=False
            )
            student_ids = self.tokenizer.encode(text, add_special_tokens=False)
            # Use the first student token as the representative.
            sid = int(student_ids[0]) if student_ids else teacher_token_id
        except Exception:
            sid = teacher_token_id

        self._vocab_remap_cache[teacher_token_id] = sid
        return sid

    def _align_and_remap(
        self,
        student_ids: List[int],
        teacher_topk_ids: List[List[int]],
        teacher_topk_lps: List[List[float]],
        alignment: List[List[int]],
    ) -> Tuple[List[List[int]], List[List[float]]]:
        """
        Project teacher logprobs (teacher positions, teacher vocab) onto
        student positions in student vocab.

        For each student token position:
          1. Collect all teacher top-k entries from overlapping teacher positions.
          2. Remap each teacher token ID to the student vocab.
          3. Merge duplicates by keeping the maximum logprob.
          4. Return the top-k entries sorted by descending logprob.
        """
        k = max(1, len(teacher_topk_ids[0]) if teacher_topk_ids else 1)
        result_ids: List[List[int]] = []
        result_lps: List[List[float]] = []

        for s_idx in range(len(student_ids)):
            t_positions = alignment[s_idx] if s_idx < len(alignment) else []
            if not t_positions:
                result_ids.append([])
                result_lps.append([])
                continue

            # Merge all overlapping teacher positions, remap vocab.
            merged: Dict[int, float] = {}
            for t_idx in t_positions:
                if t_idx >= len(teacher_topk_ids):
                    continue
                for tid, tlp in zip(teacher_topk_ids[t_idx], teacher_topk_lps[t_idx]):
                    sid = self._remap_teacher_token_to_student(tid)
                    merged[sid] = max(merged.get(sid, -1e9), tlp)

            sorted_items = sorted(merged.items(), key=lambda x: -x[1])
            top = sorted_items[:k]
            result_ids.append([int(sid) for sid, _ in top])
            result_lps.append([float(lp) for _, lp in top])

        return result_ids, result_lps

    # ------------------------------------------------------------------
    # Core fetch
    # ------------------------------------------------------------------

    async def _fetch_teacher_for_sequence(
        self, token_ids: List[int], top_k: int
    ) -> Tuple[List[List[int]], List[List[float]]]:
        assert self.teacher_server is not None

        if self._same_tokenizer or self._teacher_tokenizer is None:
            # Fast path: same vocabulary — send student IDs directly.
            payload = await self.teacher_server.get_logprobs(
                input_ids=token_ids,
                top_k=top_k,
                max_tokens=1,
                split="train",
            )
            return payload["prompt_topk_token_ids"], payload["prompt_topk_logprobs"]

        # Cross-tokenizer path:
        #   1. Decode student tokens → plain text.
        #   2. Re-tokenize with teacher tokenizer → teacher IDs.
        #   3. Send teacher IDs to teacher vLLM.
        #   4. Align teacher positions → student positions.
        #   5. Remap teacher vocab IDs → student vocab IDs.
        text = self.tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)
        teacher_ids: List[int] = self._teacher_tokenizer.encode(
            text, add_special_tokens=False
        )

        payload = await self.teacher_server.get_logprobs(
            input_ids=teacher_ids,
            top_k=top_k,
            max_tokens=1,
            split="train",
        )
        teacher_topk_ids = payload["prompt_topk_token_ids"]
        teacher_topk_lps = payload["prompt_topk_logprobs"]

        alignment = self._build_student_teacher_alignment(text, token_ids, teacher_ids)
        return self._align_and_remap(token_ids, teacher_topk_ids, teacher_topk_lps, alignment)

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
