import json
import logging
import os
import random
import threading
import time
import contextlib
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import grpc
from datasets import load_dataset
from google.protobuf import descriptor_pb2, descriptor_pool, json_format, message_factory

from atroposlib.envs.base import APIServerConfig, BaseEnv, BaseEnvConfig, ScoredDataItem
from atroposlib.type_definitions import Item, Message
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer
from environments.cline_env.worker_manager import LocalWorkerManager, WorkerHandle
from environments.cline_env.profile_registry import get_profile_config, ProfileConfig, supported_languages


logger = logging.getLogger(__name__)

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
        tokenizer_name = os.getenv("TOKENIZER_NAME", "NousResearch/Meta-Llama-3-8B")

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
            include_messages=True,
            allowed_languages=list(supported_languages()),
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
        self, item: Item, skip_tokenization: bool = False
    ) -> Tuple[Optional[ScoredDataItem], List[Item]]:
        issue_text: str = item["issue_text"]
        target: bool = item["target"]
        language: str = item.get("language", "unknown")

        messages: List[Message] = [
            {"role": "system", "content": self.config.system_prompt, "reward": None},
            {"role": "user", "content": issue_text, "reward": None},
        ]

        assistant_content: str
        cline_metadata: Optional[Dict[str, Any]] = None
        profile_key: Optional[str] = None
        if self.config.use_cline_worker:
            try:
                profile_config = get_profile_config(language)
            except KeyError:
                profile_config = None
                logger.warning("No Cline worker profile for language '%s'; falling back to policy LLM", language)

            if profile_config:
                profile_key = profile_config.profile_key
                assistant_content, cline_metadata = self._run_cline_worker(
                    profile_config, item, issue_text
                )
                if not assistant_content and cline_metadata is None:
                    logger.warning(
                        "Cline worker for language '%s' returned no content; falling back to policy LLM",
                        language,
                    )
                    chat_completion = await self.server.chat_completion(
                        messages=messages,
                        n=1,
                        max_tokens=self.config.max_token_length,
                    )
                    assistant_content = chat_completion.choices[0].message.content
            else:
                chat_completion = await self.server.chat_completion(
                    messages=messages,
                    n=1,
                    max_tokens=self.config.max_token_length,
                )
                assistant_content = chat_completion.choices[0].message.content
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

        if skip_tokenization:
            tokens: List[int] = []
            masks: List[int] = []
        else:
            tokenized = tokenize_for_trainer(
                self.tokenizer,
                messages,
                include_messages=self.config.include_messages,
                train_on_all_assistant_turns=False,
            )
            tokens = tokenized["tokens"]
            masks = tokenized["masks"]

        overrides: Optional[Dict[str, object]] = None
        if cline_metadata is not None:
            cline_metadata.setdefault("language", language)
            if profile_key:
                cline_metadata.setdefault("profile_key", profile_key)
            overrides = {"cline_metadata": cline_metadata}

        scored_item: ScoredDataItem = {
            "tokens": tokens,
            "masks": masks,
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

        repo_name = row.get("repo_name") or row.get("repo") or ""
        repo_url = row.get("repo_url") or (f"https://github.com/{repo_name}" if repo_name else "")
        repo_branch = row.get("branch") or row.get("default_branch") or ""
        repo_commit = row.get("base_commit") or row.get("commit") or ""

        item: Item = {
            "instance_id": row.get("id", ""),
            "model_name": row.get("task_type", ""),
            "target": bool(row.get("target", False)),
            "issue_text": issue_text,
            "language": row.get("language", "unknown"),
            "dataset_index": index,
            "repo_name": repo_name,
            "repo_url": repo_url,
            "repo_branch": repo_branch,
            "repo_commit": repo_commit,
        }
        return item

    def _run_cline_worker(
        self,
        profile_config: ProfileConfig,
        item: Item,
        issue_text: str,
        max_ui_messages: int = 2048,
        stream_timeout_s: float = 300.0,
        idle_timeout_s: float = 10.0,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Start a local Cline worker for the given profile and capture the full UI trajectory.

        This method:
        - Starts the worker (Rust/Ratatui profile only).
        - Configures Anthropic as the provider if API key is present.
        - Subscribes to UiService.subscribeToPartialMessage to stream ClineMessage events.
        - Calls TaskService.newTask with the issue text.
        - Waits until the UI stream is idle or a timeout/max message limit is hit.
        - Returns:
            * assistant_content: concatenation of reasoning/text fields for all UI messages.
            * cline_metadata: dict containing task_id, profile, and the full ui_messages list.
        """
        repo_name = item.get("repo_name") or f"task-{item.get('instance_id', 'repo')}"
        repo_slug = repo_name.replace("/", "__")
        repo_url = item.get("repo_url") or ""
        workspace_root = Path(tempfile.mkdtemp(prefix=f"cline-{profile_config.profile_key}-"))
        repo_path = workspace_root / repo_slug
        task_env = {
            "TASK_LANGUAGE": item.get("language", "unknown"),
            "TASK_ID": str(item.get("instance_id")),
            "TASK_REPO_URL": repo_url,
            "TASK_REPO_NAME": repo_name,
            "TASK_REPO_BRANCH": item.get("repo_branch", ""),
            "TASK_REPO_REV": item.get("repo_commit", ""),
            "TASK_REPO_PATH": str(repo_path),
            "WORKSPACE_ROOT": str(workspace_root),
            "WORKSPACE_DIR": str(repo_path),
            "DEV_WORKSPACE_FOLDER": str(repo_path),
        }

        manager = LocalWorkerManager()
        handle: Optional[WorkerHandle] = None
        handle = manager.start_for_profile(profile_config.profile_key, task_env)

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
            message_cache: Dict[str, Any] = {}

            def get_message_class(full_name: str):
                if full_name not in message_cache:
                    desc = pool.FindMessageTypeByName(full_name)
                    get_proto = getattr(factory, "GetPrototype", None)
                    if callable(get_proto):
                        cls = get_proto(desc)
                    else:
                        cls = message_factory.GetMessageClass(desc)
                    message_cache[full_name] = cls
                return message_cache[full_name]

            def new_message(full_name: str):
                cls = get_message_class(full_name)
                return cls()

            def enum_value(full_name: str, name: str) -> int:
                enum_desc = pool.FindEnumTypeByName(full_name)
                try:
                    return enum_desc.values_by_name[name].number
                except KeyError as exc:
                    raise ValueError(f"Enum {full_name} has no value named {name}") from exc

            channel = grpc.insecure_channel(handle.protobus_address)
            grpc.channel_ready_future(channel).result(timeout=60.0)

            def unary_unary(method: str, request, response_type: str):
                stub = channel.unary_unary(
                    method,
                    request_serializer=lambda m: m.SerializeToString(),
                    response_deserializer=lambda data: get_message_class(response_type).FromString(
                        data
                    ),
                )
                return stub(request)

            def unary_stream(method: str, request, response_type: str):
                stub = channel.unary_stream(
                    method,
                    request_serializer=lambda m: m.SerializeToString(),
                    response_deserializer=lambda data: get_message_class(response_type).FromString(
                        data
                    ),
                )
                return stub(request)

            # Configure Anthropic provider if credentials exist.
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")
            anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
            if anthropic_key:
                cfg_req = new_message("cline.UpdateApiConfigurationPartialRequest")
                cfg_req.metadata.CopyFrom(new_message("cline.Metadata"))
                api_cfg = cfg_req.api_configuration  # type: ignore[attr-defined]
                api_cfg.api_key = anthropic_key
                anthropic_value = enum_value("cline.ApiProvider", "ANTHROPIC")
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
                unary_unary(
                    "/cline.ModelsService/updateApiConfigurationPartial",
                    cfg_req,
                    "cline.Empty",
                )

            # Initialize UI.
            unary_unary(
                "/cline.UiService/initializeWebview",
                new_message("cline.EmptyRequest"),
                "cline.Empty",
            )

            # Subscribe to partial message stream.
            ui_messages: List[Dict[str, Any]] = []
            stream_ready = threading.Event()
            stream_stop = threading.Event()
            stream_error: List[Optional[BaseException]] = [None]
            call_holder: Dict[str, Any] = {}

            def stream_consumer() -> None:
                request = new_message("cline.EmptyRequest")
                call = unary_stream(
                    "/cline.UiService/subscribeToPartialMessage",
                    request,
                    "cline.ClineMessage",
                )
                call_holder["call"] = call
                stream_ready.set()
                try:
                    for message in call:
                        msg_dict = json_format.MessageToDict(
                            message, preserving_proto_field_name=True
                        )
                        ui_messages.append(msg_dict)
                        if stream_stop.is_set() or len(ui_messages) >= max_ui_messages:
                            break
                except grpc.RpcError as exc:
                    if not stream_stop.is_set() and exc.code() != grpc.StatusCode.CANCELLED:
                        stream_error[0] = exc
                finally:
                    stream_stop.set()

            consumer_thread = threading.Thread(
                target=stream_consumer, name="cline-ui-stream", daemon=True
            )
            consumer_thread.start()

            if not stream_ready.wait(timeout=20.0):
                raise TimeoutError(
                    "Timed out waiting to subscribe to Cline UiService partial messages"
                )

            # Create a new task.
            metadata_msg = new_message("cline.Metadata")
            task_req = new_message("cline.NewTaskRequest")
            task_req.metadata.CopyFrom(metadata_msg)
            task_req.text = issue_text
            resp = unary_unary("/cline.TaskService/newTask", task_req, "cline.String")
            task_id = resp.value

            # Wait for the stream to be idle or timeout.
            deadline = time.time() + stream_timeout_s
            last_len = 0
            last_change = time.time()

            while time.time() < deadline:
                current_len = len(ui_messages)
                if current_len != last_len:
                    last_len = current_len
                    last_change = time.time()
                    if current_len >= max_ui_messages:
                        logger.warning(
                            "Reached max_ui_messages=%d for task %s; truncating UI stream",
                            max_ui_messages,
                            task_id,
                        )
                        break
                elif time.time() - last_change > idle_timeout_s:
                    break
                time.sleep(0.5)

            stream_stop.set()
            call = call_holder.get("call")
            if call is not None:
                with contextlib.suppress(Exception):
                    call.cancel()
            consumer_thread.join(timeout=5.0)

            if stream_error[0]:
                raise RuntimeError(f"Cline UI stream failed: {stream_error[0]}")

            channel.close()

            # Build assistant content as concatenation of reasoning/text fields.
            assistant_parts: List[str] = []
            for msg in ui_messages:
                reasoning = msg.get("reasoning") or ""
                text = msg.get("text") or ""
                parts = [p for p in (reasoning, text) if p]
                if parts:
                    assistant_parts.append("\n\n".join(parts))
            assistant_content = "\n\n---\n\n".join(assistant_parts)

            cline_metadata: Dict[str, Any] = {
                "task_id": task_id,
                "profile": profile_config.profile_key,
                "ui_messages": ui_messages,
                "repo_name": repo_name,
                "repo_url": repo_url,
                "workspace_root": str(workspace_root),
                "workspace_repo": str(repo_path),
            }
            return assistant_content, cline_metadata
        except Exception as exc:
            logger.exception(
                "Cline worker invocation failed, falling back to empty assistant: %s", exc
            )
            return "", None
        finally:
            if handle is not None:
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
        with a conversation containing:
          - system prompt
          - user issue text
          - full assistant trajectory reconstructed from Cline UI messages (if available)
        and attaches raw Cline metadata (including the full `ui_messages` list) under
        `cline_metadata`.
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

        overrides = scored.get("overrides") if scored else None
        cline_meta = overrides.get("cline_metadata") if isinstance(overrides, dict) else None

        # If we have Cline UI messages, reconstruct the full assistant trajectory from them.
        if isinstance(cline_meta, dict) and isinstance(
            cline_meta.get("ui_messages"), list
        ):
            ui_messages = cline_meta["ui_messages"]
            for msg in ui_messages:
                if not isinstance(msg, dict):
                    continue
                reasoning = msg.get("reasoning") or ""
                text = msg.get("text") or ""
                parts = [p for p in (reasoning, text) if p]
                if not parts:
                    continue
                conversations.append(
                    {"from": "assistant", "value": "\n\n".join(parts)}
                )
            out_row["cline_metadata"] = cline_meta
        else:
            # Fallback: use the assistant message stored in scored["messages"], if any.
            assistant_text = ""
            if scored and scored.get("messages"):
                last_msg = scored["messages"][-1]
                if last_msg.get("role") == "assistant":
                    assistant_text = str(last_msg.get("content") or "")
            if assistant_text:
                conversations.append({"from": "assistant", "value": assistant_text})

        out_row["conversations"] = conversations

        out_row["score"] = float(scored["scores"]) if scored is not None else None

        return out_row


if __name__ == "__main__":
    ClineAgentEnv.cli()
