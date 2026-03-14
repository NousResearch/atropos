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
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
from pydantic import Field
from pydantic_cli import Cmd
from rich import print as rprint

from ..utils.cli import (
    extract_namespace,
    get_double_dash_flags,
    get_prefixed_pydantic_model,
    merge_dicts,
)
from .base import BaseEnv, BaseEnvConfig, ScoredDataGroup
from .constants import ENV_NAMESPACE, NAMESPACE_SEP, OPENAI_NAMESPACE
from .server_handling.openai_server import resolve_openai_configs
from .server_handling.server_baseline import APIServerConfig, ServerBaseline
from .server_handling.server_manager import ServerManager, ServerManagerConfig

logger = logging.getLogger(__name__)


class TeacherDistillationConfig(BaseEnvConfig):
    teacher_enabled: bool = Field(
        default=False,
        description="Whether to fetch teacher prompt logprobs for distillation.",
    )
    teacher_top_k: int = Field(
        default=0,
        ge=-1,
        description=(
            "Number of extra prompt logprobs to fetch beyond the selected token. "
            "Use 0 for selected-token-only prompt logprobs and <= -1 to disable "
            "teacher fetching."
        ),
    )


class TeacherDistillationEnv(BaseEnv, ABC):
    """
    BaseEnv subclass that enriches scored groups with teacher distillation arrays.

    Distillation payload shape:
      - distill_token_ids: [sequence][position][k]  (student vocab IDs)
      - distill_logprobs:  [sequence][position][k]
    """

    env_config_cls = TeacherDistillationConfig
    teacher_namespace = "teacher"

    @classmethod
    def teacher_config_init(
        cls,
    ) -> Optional[Union[ServerBaseline, List[APIServerConfig], APIServerConfig]]:
        return None

    @classmethod
    def _resolve_teacher_server_configs(
        cls,
        default_teacher_server_configs: Optional[
            Union[ServerBaseline, List[APIServerConfig], APIServerConfig]
        ],
        yaml_config: Dict[str, Any],
        cli_passed_flags: Dict[str, Any],
    ) -> Optional[Union[ServerBaseline, List[APIServerConfig]]]:
        teacher_full_prefix = f"{cls.teacher_namespace}{NAMESPACE_SEP}"
        teacher_cli_passed_args = extract_namespace(
            cli_passed_flags, teacher_full_prefix
        )
        yaml_teacher_config = yaml_config.get(cls.teacher_namespace, {})

        if (
            default_teacher_server_configs is None
            and not teacher_cli_passed_args
            and not yaml_teacher_config
        ):
            return None

        effective_teacher_server_configs = default_teacher_server_configs
        if effective_teacher_server_configs is None:
            effective_teacher_server_configs = APIServerConfig()
        elif isinstance(effective_teacher_server_configs, ServerBaseline) and (
            teacher_cli_passed_args or yaml_teacher_config
        ):
            effective_teacher_server_configs = APIServerConfig(
                **effective_teacher_server_configs.model_dump()
            )

        if (
            isinstance(effective_teacher_server_configs, list)
            and len(effective_teacher_server_configs) == 1
        ):
            default_teacher_config = effective_teacher_server_configs[0]
        else:
            default_teacher_config = effective_teacher_server_configs

        if isinstance(yaml_teacher_config, list) and len(yaml_teacher_config) == 1:
            yaml_teacher_config = yaml_teacher_config[0]

        if isinstance(default_teacher_config, APIServerConfig) and isinstance(
            yaml_teacher_config, dict
        ):
            teacher_config_dict = merge_dicts(
                default_teacher_config.model_dump(),
                yaml_teacher_config,
                teacher_cli_passed_args,
            )
        else:
            teacher_config_dict = {}

        teacher_yaml_wrapped = {OPENAI_NAMESPACE: yaml_teacher_config}
        teacher_cli_wrapped = {
            f"{OPENAI_NAMESPACE}{NAMESPACE_SEP}{key}": value
            for key, value in teacher_cli_passed_args.items()
        }
        return resolve_openai_configs(
            default_server_configs=effective_teacher_server_configs,
            openai_config_dict=teacher_config_dict,
            yaml_config=teacher_yaml_wrapped,
            cli_passed_flags=teacher_cli_wrapped,
            logger=logger,
        )

    @classmethod
    def get_cli_serve_config_cls(cls) -> type:
        default_env_config, default_server_configs = cls.config_init()
        default_teacher_server_configs = cls.teacher_config_init()

        env_full_prefix = f"{ENV_NAMESPACE}{NAMESPACE_SEP}"
        openai_full_prefix = f"{OPENAI_NAMESPACE}{NAMESPACE_SEP}"
        teacher_full_prefix = f"{cls.teacher_namespace}{NAMESPACE_SEP}"
        teacher_cli_base = get_prefixed_pydantic_model(
            APIServerConfig, teacher_full_prefix
        )

        class CliServeConfig(
            get_prefixed_pydantic_model(type(default_env_config), env_full_prefix),
            get_prefixed_pydantic_model(APIServerConfig, openai_full_prefix),
            teacher_cli_base,
            ServerManagerConfig,
            Cmd,
        ):
            config: str | None = Field(
                default=None,
                description="Path to .yaml config file. CLI args override this.",
            )

            def run(self) -> None:
                wandb_name_attr = f"{ENV_NAMESPACE}{NAMESPACE_SEP}wandb_name"
                if (
                    getattr(self, wandb_name_attr, None) is None
                    and cls.name is not None
                ):
                    setattr(self, wandb_name_attr, cls.name)

                if self.config is not None:
                    with open(self.config, "r") as f:
                        yaml_config = yaml.safe_load(f)
                    logger.info("Loaded config from %s", self.config)
                else:
                    yaml_config = {}

                cli_passed_flags = get_double_dash_flags()

                env_config_dict = merge_dicts(
                    default_env_config.model_dump(),
                    yaml_config.get(ENV_NAMESPACE, {}),
                    extract_namespace(cli_passed_flags, env_full_prefix),
                )

                oai_cli_passed_args = extract_namespace(
                    cli_passed_flags, openai_full_prefix
                )
                yaml_oai_config = yaml_config.get(OPENAI_NAMESPACE, {})

                effective_server_configs = default_server_configs
                if isinstance(effective_server_configs, ServerBaseline) and (
                    oai_cli_passed_args or yaml_oai_config
                ):
                    effective_server_configs = APIServerConfig(
                        **effective_server_configs.model_dump()
                    )

                if (
                    isinstance(effective_server_configs, list)
                    and len(effective_server_configs) == 1
                ):
                    default_openai_config_ = effective_server_configs[0]
                else:
                    default_openai_config_ = effective_server_configs

                if isinstance(yaml_oai_config, list) and len(yaml_oai_config) == 1:
                    yaml_oai_config = yaml_oai_config[0]

                if isinstance(default_openai_config_, APIServerConfig) and isinstance(
                    yaml_oai_config, dict
                ):
                    openai_config_dict = merge_dicts(
                        default_openai_config_.model_dump(),
                        yaml_oai_config,
                        oai_cli_passed_args,
                    )
                else:
                    openai_config_dict = {}

                server_manager_cli_passed_flags = {}
                if "slurm" in cli_passed_flags:
                    server_manager_cli_passed_flags["slurm"] = cli_passed_flags["slurm"]
                if "testing" in cli_passed_flags:
                    server_manager_cli_passed_flags["testing"] = cli_passed_flags[
                        "testing"
                    ]

                server_manager_yaml_dict = {}
                if "slurm" in yaml_config:
                    server_manager_yaml_dict["slurm"] = yaml_config["slurm"]
                if "testing" in yaml_config:
                    server_manager_yaml_dict["testing"] = yaml_config["testing"]

                server_manager_config_dict = merge_dicts(
                    ServerManagerConfig().model_dump(),
                    server_manager_yaml_dict,
                    server_manager_cli_passed_flags,
                )

                env_config = type(default_env_config)(**env_config_dict)
                server_manager_config = ServerManagerConfig(
                    **server_manager_config_dict
                )
                openai_configs = resolve_openai_configs(
                    default_server_configs=effective_server_configs,
                    openai_config_dict=openai_config_dict,
                    yaml_config=yaml_config,
                    cli_passed_flags=cli_passed_flags,
                    logger=logger,
                )
                teacher_configs = cls._resolve_teacher_server_configs(
                    default_teacher_server_configs=default_teacher_server_configs,
                    yaml_config=yaml_config,
                    cli_passed_flags=cli_passed_flags,
                )

                env_kwargs = {
                    "config": env_config,
                    "server_configs": openai_configs,
                    "slurm": server_manager_config.slurm,
                    "testing": server_manager_config.testing,
                }
                if teacher_configs is not None:
                    env_kwargs["teacher_server_configs"] = teacher_configs
                env = cls(**env_kwargs)
                rprint(env_config)
                rprint(openai_configs)
                if teacher_configs is not None:
                    rprint(teacher_configs)

                try:
                    loop = asyncio.get_running_loop()
                    task = loop.create_task(env.env_manager())
                    loop.run_until_complete(task)
                except RuntimeError:
                    asyncio.run(env.env_manager())

        return CliServeConfig

    def __init__(
        self,
        config: TeacherDistillationConfig,
        server_configs: Union[ServerBaseline, List[APIServerConfig]],
        teacher_server_configs: Optional[
            Union[ServerBaseline, APIServerConfig, List[APIServerConfig]]
        ] = None,
        slurm: bool = False,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm=slurm, testing=testing)
        self.teacher_server: Optional[ServerManager] = None

        if config.teacher_enabled:
            if teacher_server_configs is None:
                raise ValueError(
                    "teacher_enabled=True but no teacher server configuration was "
                    "provided. Pass teacher_server_configs=... when instantiating "
                    "the environment directly, or use the teacher-aware 'serve' CLI "
                    "path with --teacher.* flags. The generic BaseEnv 'process' and "
                    "'evaluate' commands do not currently wire teacher_server_configs."
                )
            if isinstance(teacher_server_configs, APIServerConfig):
                teacher_config_source = [teacher_server_configs]
            else:
                teacher_config_source = teacher_server_configs
            self.teacher_server = ServerManager(
                teacher_config_source,
                slurm=False,
                testing=False,
            )
            if isinstance(teacher_config_source, list):
                teacher_cfg = teacher_config_source[0]
            else:
                teacher_cfg = teacher_config_source

            teacher_tokenizer_name = (
                teacher_cfg.model_name
                if getattr(teacher_cfg, "tokenizer_name", "none") in ("", "none")
                else teacher_cfg.tokenizer_name
            )
            self._validate_teacher_tokenizer_compatibility(teacher_tokenizer_name)

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

        group_overrides = group.get("group_overrides") or {}
        if group_overrides.get("skip_teacher_top_k", False):
            group["distill_token_ids"] = None
            group["distill_logprobs"] = None
            return group

        top_k = int(group_overrides.get("teacher_top_k", self.config.teacher_top_k))
        if top_k <= -1:
            group["distill_token_ids"] = None
            group["distill_logprobs"] = None
            return group

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
