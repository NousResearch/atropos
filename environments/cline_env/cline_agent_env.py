import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import grpc
from datasets import load_dataset
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

from atroposlib.envs.base import APIServerConfig, BaseEnv, BaseEnvConfig, ScoredDataItem
from atroposlib.type_definitions import Item, Message
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer
from environments.cline_env.worker_manager import LocalWorkerManager, WorkerHandle


logger = logging.getLogger(__name__)

# Simple mapping from dataset language labels to environment profiles.
# For now we only support Rust via the ratatui bootstrap example.
LANGUAGE_PROFILE_MAP: Dict[str, str] = {
    "Rust": "rust_ratatui",
}


class ClineAgentEnvConfig(BaseEnvConfig):
    tokenizer_name: str = "NousResearch/Meta-Llama-3-8B"
    env_name: str = "cline_agent_env"
    dataset_name: str = "NousResearch/swe-agent-13k-2025-06-15" # "conversation" is json column, message idx 1 (role "human") should be task
    max_episode_turns: int = 1
    eval_episodes: int = 50
    scoring_function: str = "dataset_target"
    # Limit tasks to specific languages (by dataset `language` column).
    # If None, all languages are allowed.
    allowed_languages: Optional[List[str]] = None
    # Whether to route rollouts through a Cline worker (gRPC) instead of
    # directly calling the policy LLM. For now only Rust is supported.
    use_cline_worker: bool = False
    system_prompt: str = (
        "You are a senior software engineer helping to resolve a GitHub issue. "
        "Read the issue description carefully and propose a clear, concrete patch "
        "or explanation of how to resolve it."
    )


class ClineAgentEnv(BaseEnv):
    name = "cline_agent_env"
    env_config_cls = ClineAgentEnvConfig

    def __init__(
        self,
        config: ClineAgentEnvConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = False,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: ClineAgentEnvConfig = config
        self.dataset = None
        self.dataset_indices: List[int] = []
        self.dataset_position = 0
        self.episode_outcomes_buffer: List[float] = []
        self.eval_metrics_custom: List[Tuple[str, float]] = []

    @classmethod
    def config_init(cls) -> Tuple[ClineAgentEnvConfig, List[APIServerConfig]]:
        tokenizer_name = os.getenv("TOKENIZER_NAME", "gpt2")

        env_config = ClineAgentEnvConfig(
            tokenizer_name=tokenizer_name,
            group_size=4,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            max_token_length=4096,
            wandb_name=cls.name,
            steps_per_eval=100,
            max_episode_turns=1,
            eval_episodes=50,
            # Start by focusing on Rust, since we have a concrete
            # environment profile and bootstrap for it.
            allowed_languages=["Rust"],
        )
        server_configs = [
            APIServerConfig(
                model_name="anthropic_sonnet_like",
                base_url="http://localhost:9001/v1",
                api_key="x",
                num_requests_for_eval=128,
            ),
        ]
        return env_config, server_configs

    async def setup(self):
        if self.dataset is None:
            self.dataset = load_dataset(self.config.dataset_name, split="train")
            all_indices = list(range(len(self.dataset)))

            if self.config.allowed_languages:
                allowed = set(self.config.allowed_languages)
                filtered: List[int] = []
                for idx in all_indices:
                    row = self.dataset[idx]
                    lang = row.get("language", None)
                    if lang in allowed:
                        filtered.append(idx)
                if not filtered:
                    raise RuntimeError(
                        f"No dataset rows matched allowed_languages={self.config.allowed_languages}"
                    )
                self.dataset_indices = filtered
                logger.info(
                    "ClineAgentEnv: filtered dataset to %d/%d rows for languages %s",
                    len(self.dataset_indices),
                    len(all_indices),
                    sorted(allowed),
                )
            else:
                self.dataset_indices = all_indices

            random.shuffle(self.dataset_indices)
            self.dataset_position = 0

    async def collect_trajectory(
        self, item: Item
    ) -> Tuple[Optional[ScoredDataItem], List[Item]]:
        issue_text: str = item["issue_text"]
        target: bool = item["target"]
        language: str = item.get("language", "unknown")

        messages: List[Message] = [
            {"role": "system", "content": self.config.system_prompt, "reward": None},
            {"role": "user", "content": issue_text, "reward": None},
        ]

        assistant_content: str
        cline_task_id: Optional[str] = None
        if self.config.use_cline_worker:
            profile = LANGUAGE_PROFILE_MAP.get(language)
            if not profile:
                logger.warning(
                    "No Cline worker profile for language '%s'; falling back to policy LLM",
                    language,
                )
                chat_completion = await self.server.chat_completion(
                    messages=messages,
                    n=1,
                    max_tokens=self.config.max_token_length,
                )
                assistant_content = chat_completion.choices[0].message.content
            else:
                assistant_content, cline_task_id = self._run_cline_worker(profile, issue_text)
        else:
            chat_completion = await self.server.chat_completion(
                messages=messages,
                n=1,
                max_tokens=self.config.max_token_length,
            )
            assistant_content = chat_completion.choices[0].message.content

        messages.append(
            {"role": "assistant", "content": assistant_content, "reward": None}
        )

        if self.config.scoring_function == "dataset_target":
            reward = 1.0 if target else -1.0
        else:
            reward = 0.0

        self.episode_outcomes_buffer.append(reward)

        tokenized = tokenize_for_trainer(
            self.tokenizer,
            messages,
            include_messages=self.config.include_messages,
            train_on_all_assistant_turns=False,
        )

        overrides: Optional[Dict[str, object]] = None
        if cline_task_id is not None:
            overrides = {
                "cline_metadata": {
                    "task_id": cline_task_id,
                    "language": language,
                    "profile": LANGUAGE_PROFILE_MAP.get(language),
                }
            }

        scored_item: ScoredDataItem = {
            "tokens": tokenized["tokens"],
            "masks": tokenized["masks"],
            "scores": reward,
            "advantages": None,
            "ref_logprobs": None,
            "messages": messages if self.config.include_messages else None,
            "group_overrides": None,
            "overrides": overrides,
            "images": None,
        }
        return scored_item, []

    async def get_next_item(self) -> Item:
        if self.dataset is None:
            await self.setup()

        if not self.dataset_indices:
            raise RuntimeError("Dataset indices not initialized")

        index = self.dataset_indices[self.dataset_position % len(self.dataset_indices)]
        self.dataset_position += 1
        row = self.dataset[index]

        conversations = row["conversations"]

        issue_text = ""
        if isinstance(conversations, list) and len(conversations) > 1:
            second = conversations[1]
            if isinstance(second, dict) and second.get("from") in ("human", "user"):
                issue_text = second.get("value") or ""

        if not issue_text and isinstance(conversations, list) and conversations:
            first = conversations[0]
            if isinstance(first, dict):
                issue_text = first.get("value") or ""

        item: Item = {
            "instance_id": row.get("id", ""),
            "model_name": row.get("task_type", ""),
            "target": bool(row.get("target", False)),
            "issue_text": issue_text,
            "language": row.get("language", "unknown"),
            "dataset_index": index,
        }
        return item

    def _run_cline_worker(self, profile: str, issue_text: str) -> Tuple[str, Optional[str]]:
        """Start a local Cline worker for the given profile and trigger a newTask via gRPC.

        For now this is a minimal integration that:
        - Starts the worker (Rust/Ratatui profile only).
        - Configures Anthropic as the provider if API key is present.
        - Calls TaskService.newTask with the issue text.
        - Returns a simple assistant summary string.
        """
        manager = LocalWorkerManager()
        handle: WorkerHandle = manager.start_for_profile(profile)

        try:
            descriptor_candidates = [
                handle.cline_src_dir / "dist-standalone" / "proto" / "descriptor_set.pb",
                handle.cline_src_dir / "proto" / "descriptor_set.pb",
            ]
            descriptor_path = next((p for p in descriptor_candidates if p.exists()), None)
            if descriptor_path is None:
                raise FileNotFoundError(
                    f"descriptor_set.pb not found under {handle.cline_src_dir}"
                )

            descriptor_bytes = descriptor_path.read_bytes()
            descriptor_set = descriptor_pb2.FileDescriptorSet()
            descriptor_set.ParseFromString(descriptor_bytes)

            pool = descriptor_pool.DescriptorPool()
            for file_proto in descriptor_set.file:
                pool.Add(file_proto)

            factory = message_factory.MessageFactory(pool)

            def msg(name: str):
                return factory.GetPrototype(pool.FindMessageTypeByName(name))

            Metadata = msg("cline.Metadata")
            NewTaskRequest = msg("cline.NewTaskRequest")
            StringMsg = msg("cline.String")
            UpdateApiConfigurationPartialRequest = msg("cline.UpdateApiConfigurationPartialRequest")

            ApiProvider_enum = pool.FindEnumTypeByName("cline.ApiProvider")
            anthropic_value = ApiProvider_enum.values_by_name["ANTHROPIC"].number

            channel = grpc.insecure_channel(handle.protobus_address)

            def unary(method: str, request, response_cls):
                stub = channel.unary_unary(
                    method,
                    request_serializer=lambda m: m.SerializeToString(),
                    response_deserializer=lambda data: response_cls.FromString(data),
                )
                return stub(request)

            # Configure Anthropic provider if credentials exist.
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")
            anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
            if anthropic_key:
                cfg_req = UpdateApiConfigurationPartialRequest()
                cfg_req.metadata.CopyFrom(Metadata())
                api_cfg = cfg_req.api_configuration  # type: ignore[attr-defined]
                api_cfg.api_key = anthropic_key
                api_cfg.plan_mode_api_provider = anthropic_value
                api_cfg.act_mode_api_provider = anthropic_value
                api_cfg.plan_mode_api_model_id = anthropic_model
                api_cfg.act_mode_api_model_id = anthropic_model
                cfg_req.update_mask.paths.extend(
                    [
                        "apiKey",
                        "planModeApiProvider",
                        "actModeApiProvider",
                        "planModeApiModelId",
                        "actModeApiModelId",
                    ]
                )
                unary(
                    "/cline.ModelsService/updateApiConfigurationPartial",
                    cfg_req,
                    msg("cline.Empty"),
                )

            # Initialize UI and create a new task.
            unary(
                "/cline.UiService/initializeWebview",
                Metadata(),
                msg("cline.Empty"),
            )

            task_req = NewTaskRequest()
            task_req.metadata.CopyFrom(Metadata())
            task_req.text = issue_text
            resp = unary("/cline.TaskService/newTask", task_req, StringMsg)

            channel.close()

            task_id = resp.value
            summary = f"Cline created task {task_id} for issue: {issue_text[:200]}"
            return summary, task_id
        except Exception as exc:
            logger.exception(
                "Cline worker invocation failed, falling back to empty assistant: %s", exc
            )
            return "", None
        finally:
            manager.stop(handle)

    async def evaluate(self, *args, **kwargs):
        eval_outcomes: List[float] = []

        for _ in range(self.config.eval_episodes):
            item = await self.get_next_item()
            scored_item_tuple = await self.collect_trajectory(item)
            if scored_item_tuple and scored_item_tuple[0]:
                outcome = scored_item_tuple[0]["scores"]
                eval_outcomes.append(outcome)

        if not eval_outcomes:
            self.eval_metrics_custom = []
            return

        num_completed = len(eval_outcomes)
        avg_reward = sum(eval_outcomes) / num_completed if num_completed > 0 else 0.0
        success_rate = (
            sum(1 for r in eval_outcomes if r > 0) / num_completed
            if num_completed > 0
            else 0.0
        )

        self.eval_metrics_custom = [
            (f"{self.name}_eval/avg_reward", avg_reward),
            (f"{self.name}_eval/success_rate", success_rate),
            (f"{self.name}_eval/num_completed_episodes", num_completed),
        ]

    async def wandb_log(self, wandb_metrics: Optional[Dict[str, float]] = None):
        if wandb_metrics is None:
            wandb_metrics = {}

        if self.episode_outcomes_buffer:
            avg_training_reward = sum(self.episode_outcomes_buffer) / len(
                self.episode_outcomes_buffer
            )
            wandb_metrics[
                f"{self.name}_train/avg_episode_reward"
            ] = avg_training_reward
            wandb_metrics[
                f"{self.name}_train/num_episodes_in_batch"
            ] = len(self.episode_outcomes_buffer)

        self.episode_outcomes_buffer = []

        for key, value in self.eval_metrics_custom:
            wandb_metrics[key] = value
        self.eval_metrics_custom = []

        await super().wandb_log(wandb_metrics)

    def dump_trajectory(self, item: Item, scored: Optional[ScoredDataItem]) -> Dict[str, Any]:
        """Return a JSON-serializable row with the Cline trajectory in the `conversations` column.

        The output row mirrors the input dataset schema, but replaces `conversations`
        with a simplified conversation derived from the messages used in collect_trajectory,
        and attaches Cline metadata (if available) under `cline_metadata`.
        """
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded; call setup() first")

        dataset_index = item.get("dataset_index")
        if dataset_index is None:
            raise ValueError("Item missing dataset_index; ensure get_next_item was used")

        row = self.dataset[int(dataset_index)]
        out_row: Dict[str, Any] = dict(row)

        conversations: List[Dict[str, Any]] = []
        system_prompt = self.config.system_prompt
        if system_prompt:
            conversations.append({"from": "system", "value": system_prompt})

        conversations.append({"from": "human", "value": item["issue_text"]})

        assistant_text = ""
        if scored and scored.get("messages"):
            last_msg = scored["messages"][-1]
            if last_msg.get("role") == "assistant":
                assistant_text = str(last_msg.get("content") or "")
        out_row["conversations"] = conversations + (
            [{"from": "assistant", "value": assistant_text}] if assistant_text else []
        )

        overrides = scored.get("overrides") if scored else None
        if isinstance(overrides, dict) and "cline_metadata" in overrides:
            out_row["cline_metadata"] = overrides["cline_metadata"]

        out_row["score"] = float(scored["scores"]) if scored is not None else None

        return out_row


if __name__ == "__main__":
    ClineAgentEnv.cli()
