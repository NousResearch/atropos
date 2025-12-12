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

import asyncio
import grpc
from datasets import load_dataset
from dotenv import load_dotenv

# Load .env from repo root for API keys and other settings
load_dotenv()
from google.protobuf import descriptor_pb2, descriptor_pool, json_format, message_factory

from atroposlib.envs.base import APIServerConfig, BaseEnv, BaseEnvConfig, ScoredDataItem
from atroposlib.type_definitions import Item, Message
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer
from environments.cline_env.profile_registry import get_profile_config, ProfileConfig, supported_languages
from environments.cline_env.scoring import (
    score_trajectory,
    extract_trajectory_summary,
    extract_files_modified,
)


logger = logging.getLogger(__name__)

class ClineAgentEnvConfig(BaseEnvConfig):
    tokenizer_name: str = "NousResearch/Meta-Llama-3-8B"
    env_name: str = "cline_agent_env"
    dataset_name: str = "NousResearch/swe-agent-13k-2025-06-15" # "conversation" is json column, message idx 1 (role "human") should be task
    max_episode_turns: int = 1
    eval_episodes: int = 50
    # Scoring function: "hybrid" (recommended), "dataset_target" (placeholder), "none"
    # hybrid = 0.3*syntax + 0.6*llm_judge + 0.1*complexity
    scoring_function: str = "hybrid"
    # Limit tasks to specific languages (by dataset `language` column).
    # If None, all languages are allowed.
    allowed_languages: Optional[List[str]] = None
    # Whether to route rollouts through a Cline worker (gRPC) instead of
    # directly calling the policy LLM. For now only Rust is supported.
    use_cline_worker: bool = False
    # Whether to use Modal-based workers instead of local Nomad workers.
    # Modal workers run in cloud containers and are recommended for scalable datagen.
    use_modal_worker: bool = False
    # Timeout for Modal worker tasks (seconds) - default 15 min for complex tasks
    modal_task_timeout_s: float = 900.0
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
        # todo: change this to ManagedServer and add proxy, currently just testing agent data capture
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

        assistant_content: Optional[str] = None
        cline_metadata: Optional[Dict[str, Any]] = None
        profile_key: Optional[str] = None
        
        # Choose worker type: Modal (cloud) > local Cline worker > direct LLM
        if self.config.use_modal_worker:
            # Use Modal-based cloud workers (recommended for scalable datagen)
            profile_key = language.lower()
            max_attempts = 3
            backoff_s = 1.0
            for attempt in range(1, max_attempts + 1):
                assistant_content, cline_metadata = await self._run_modal_cline_worker(
                    item, issue_text
                )
                if assistant_content or cline_metadata is not None:
                    break
                logger.warning(
                    "Modal worker attempt %d/%d for language '%s' (instance_id=%s) returned no content",
                    attempt,
                    max_attempts,
                    language,
                    item.get("instance_id"),
                )
                if attempt < max_attempts:
                    await asyncio.sleep(backoff_s)
                    backoff_s *= 2.0

            if not assistant_content and cline_metadata is None:
                logger.error(
                    "Modal worker failed after %d attempts for language '%s'; skipping episode instance_id=%s",
                    max_attempts,
                    language,
                    item.get("instance_id"),
                )
                return None, []
        elif self.config.use_cline_worker:
            # Use local Nomad-based Cline workers
            try:
                profile_config = get_profile_config(language)
            except KeyError as exc:
                logger.error(
                    "No Cline worker profile for language '%s'; skipping episode instance_id=%s (%s)",
                    language,
                    item.get("instance_id"),
                    exc,
                )
                return None, []

            profile_key = profile_config.profile_key
            max_attempts = 3
            backoff_s = 1.0
            for attempt in range(1, max_attempts + 1):
                assistant_content, cline_metadata = self._run_cline_worker(
                    profile_config, item, issue_text
                )
                if assistant_content or cline_metadata is not None:
                    break
                logger.warning(
                    "Cline worker attempt %d/%d for language '%s' (instance_id=%s) returned no content",
                    attempt,
                    max_attempts,
                    language,
                    item.get("instance_id"),
                )
                if attempt < max_attempts:
                    await asyncio.sleep(backoff_s)
                    backoff_s *= 2.0

            if not assistant_content and cline_metadata is None:
                logger.error(
                    "Cline worker failed after %d attempts for language '%s'; skipping episode instance_id=%s",
                    max_attempts,
                    language,
                    item.get("instance_id"),
                )
                return None, []
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

        # Calculate reward based on scoring function
        if self.config.scoring_function == "dataset_target":
            reward = 1.0 if target else -1.0
        elif self.config.scoring_function == "hybrid" and cline_metadata is not None:
            # Use hybrid scoring: execution + llm_judge + complexity
            try:
                workspace_path = Path(cline_metadata.get("workspace_root", ""))
                trajectory_summary = extract_trajectory_summary(cline_metadata)
                files_modified = extract_files_modified(cline_metadata)
                
                reward, score_meta = await score_trajectory(
                    issue_text=issue_text,
                    trajectory_summary=trajectory_summary,
                    workspace_path=workspace_path if workspace_path.exists() else None,
                    language=language,
                    cline_metadata=cline_metadata,
                    files_modified=files_modified,
                )
                # Store scoring metadata in cline_metadata for inspection
                cline_metadata["scoring"] = score_meta
                logger.info(
                    "Hybrid scoring for %s: %.3f (exec=%.2f, llm=%.2f, complexity=%.2f)",
                    item.get("instance_id"),
                    reward,
                    score_meta.get("component_scores", {}).get("execution", {}).get("score", 0),
                    score_meta.get("component_scores", {}).get("llm_judge", {}).get("score", 0),
                    score_meta.get("component_scores", {}).get("complexity", {}).get("score", 0),
                )
            except Exception as e:
                logger.warning("Hybrid scoring failed for %s: %s", item.get("instance_id"), e)
                reward = 0.0
        elif self.config.scoring_function == "none":
            reward = 0.0
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
        
        Note: This method requires the worker_manager module which is only used for local
        Nomad-based workers. For Modal cloud workers, use _run_modal_cline_worker instead.
        """
        # Lazy import - only needed for local Nomad workers
        from environments.cline_env.worker_manager import get_worker_manager, WorkerHandle
        
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

        # Use Nomad by default for worker management
        use_nomad = os.getenv("CLINE_USE_NOMAD", "true").lower() in ("true", "1", "yes")
        manager = get_worker_manager(use_nomad=use_nomad)
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

            # Subscribe to partial message stream for UI updates.
            ui_messages: List[Dict[str, Any]] = []
            stream_ready = threading.Event()
            stream_stop = threading.Event()
            stream_error: List[Optional[BaseException]] = [None]
            call_holder: Dict[str, Any] = {}

            # Also capture state updates which contain the full clineMessages array.
            state_snapshots: List[Dict[str, Any]] = []
            state_stream_ready = threading.Event()
            state_call_holder: Dict[str, Any] = {}

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

            def state_stream_consumer() -> None:
                """Subscribe to state updates to capture the full clineMessages array."""
                request = new_message("cline.EmptyRequest")
                try:
                    call = unary_stream(
                        "/cline.StateService/subscribeToState",
                        request,
                        "cline.State",
                    )
                    state_call_holder["call"] = call
                    state_stream_ready.set()
                    for state in call:
                        if stream_stop.is_set():
                            break
                        state_dict = json_format.MessageToDict(
                            state, preserving_proto_field_name=True
                        )
                        state_snapshots.append(state_dict)
                except grpc.RpcError as exc:
                    if not stream_stop.is_set() and exc.code() != grpc.StatusCode.CANCELLED:
                        logger.warning("State stream error: %s", exc)
                except Exception as exc:
                    logger.warning("State stream consumer error: %s", exc)

            consumer_thread = threading.Thread(
                target=stream_consumer, name="cline-ui-stream", daemon=True
            )
            consumer_thread.start()

            state_consumer_thread = threading.Thread(
                target=state_stream_consumer, name="cline-state-stream", daemon=True
            )
            state_consumer_thread.start()

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

            # Also cancel state stream.
            state_call = state_call_holder.get("call")
            if state_call is not None:
                with contextlib.suppress(Exception):
                    state_call.cancel()
            state_consumer_thread.join(timeout=2.0)

            if stream_error[0]:
                raise RuntimeError(f"Cline UI stream failed: {stream_error[0]}")

            channel.close()

            # Extract clineMessages from the last state snapshot (contains full trajectory).
            cline_messages: List[Dict[str, Any]] = []
            if state_snapshots:
                last_state = state_snapshots[-1]
                state_json_str = last_state.get("state_json") or last_state.get("stateJson", "")
                if state_json_str:
                    try:
                        state_data = json.loads(state_json_str)
                        cline_messages = state_data.get("clineMessages", [])
                        logger.info(
                            "Extracted %d clineMessages from state (vs %d partial UI messages)",
                            len(cline_messages),
                            len(ui_messages),
                        )
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning("Failed to parse state_json: %s", e)

            # Use clineMessages if available, otherwise fall back to ui_messages.
            # clineMessages contains the FULL trajectory with ASK and SAY messages.
            final_messages = cline_messages if cline_messages else ui_messages

            # Build assistant content as concatenation of reasoning/text fields.
            assistant_parts: List[str] = []
            for msg in final_messages:
                reasoning = msg.get("reasoning") or ""
                text = msg.get("text") or ""
                parts = [p for p in (reasoning, text) if p]
                if parts:
                    assistant_parts.append("\n\n".join(parts))
            assistant_content = "\n\n---\n\n".join(assistant_parts)

            cline_metadata: Dict[str, Any] = {
                "task_id": task_id,
                "profile": profile_config.profile_key,
                "ui_messages": final_messages,  # Use full messages from state if available.
                "ui_messages_partial_stream": ui_messages if cline_messages else None,
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

    def _filter_complete_ui_messages(self, ui_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter UI messages to remove partial/streaming updates and keep only complete messages.
        
        The subscribeToPartialMessage stream sends incremental token updates, each marked with
        `partial=True`. These are progressively longer versions of the same message. We want to:
        1. Group messages by their timestamp (ts) - partials share the same ts
        2. Keep only the last (most complete) message in each group
        3. Additionally filter out messages that are still marked partial if possible
        
        This converts hundreds of streaming token fragments into a small number of complete messages.
        """
        if not ui_messages:
            return []
        
        # Group by timestamp - partial messages share the same timestamp
        ts_groups: Dict[str, List[Dict[str, Any]]] = {}
        no_ts_messages: List[Dict[str, Any]] = []
        
        for msg in ui_messages:
            ts = msg.get("ts")
            if ts:
                ts_str = str(ts)
                if ts_str not in ts_groups:
                    ts_groups[ts_str] = []
                ts_groups[ts_str].append(msg)
            else:
                no_ts_messages.append(msg)
        
        # For each timestamp group, keep only the last message (most complete version)
        filtered: List[Dict[str, Any]] = []
        
        # Sort groups by timestamp to maintain order
        sorted_ts = sorted(ts_groups.keys(), key=lambda x: int(x) if x.isdigit() else 0)
        
        for ts_str in sorted_ts:
            group = ts_groups[ts_str]
            if group:
                # Take the last message in the group (most complete)
                last_msg = group[-1]
                # Only include if it has meaningful content
                if last_msg.get("text") or last_msg.get("reasoning"):
                    # Skip if still marked partial AND there's a non-partial version
                    # But if all are partial, we take the last one anyway
                    non_partial = [m for m in group if not m.get("partial")]
                    if non_partial:
                        filtered.append(non_partial[-1])
                    else:
                        # All partial - take the last (most complete) one
                        filtered.append(last_msg)
        
        # Add messages without timestamps
        for msg in no_ts_messages:
            if not msg.get("partial") and (msg.get("text") or msg.get("reasoning")):
                filtered.append(msg)
        
        return filtered

    async def _run_modal_cline_worker(
        self,
        item: Item,
        issue_text: str,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Run a Cline task on Modal cloud infrastructure.
        
        This method uses Modal-based workers to execute Cline tasks in isolated
        cloud containers. It's the recommended approach for scalable data generation.
        
        Note: Requires the Modal app "cline-workers" to be deployed first via:
            modal deploy environments/cline_env/modal_worker.py
        
        Args:
            item: The dataset item containing task information
            issue_text: The issue/task description to solve
            
        Returns:
            Tuple of (assistant_content, cline_metadata)
        """
        import modal
        
        language = item.get("language", "unknown")
        repo_url = item.get("repo_url") or None
        repo_branch = item.get("repo_branch") or None
        
        # Map language to Modal function name
        lang_lower = language.lower()
        function_name_map = {
            "python": "run_python_task",
            "rust": "run_rust_task",
            "typescript": "run_typescript_task",
            "javascript": "run_javascript_task",
            "go": "run_go_task",
            "c": "run_c_task",
            "cpp": "run_cpp_task",
            "c++": "run_cpp_task",
            "java": "run_java_task",
            "ruby": "run_ruby_task",
            "php": "run_php_task",
            "c#": "run_csharp_task",
            "csharp": "run_csharp_task",
            "swift": "run_swift_task",
            "kotlin": "run_kotlin_task",
            "scala": "run_scala_task",
            "haskell": "run_haskell_task",
            "elixir": "run_elixir_task",
            "dart": "run_dart_task",
            "lua": "run_lua_task",
            "shell": "run_shell_task",
            "bash": "run_shell_task",
        }
        
        func_name = function_name_map.get(lang_lower, "run_python_task")
        
        try:
            # Look up the deployed Modal function by name
            modal_func = modal.Function.from_name("cline-workers", func_name)
            
            # Call Modal function remotely
            result = modal_func.remote(
                issue_text=issue_text,
                repo_url=repo_url,
                repo_branch=repo_branch,
                task_timeout_s=self.config.modal_task_timeout_s,
            )
            
            # Extract results
            assistant_content = result.get("assistant_content", "")
            conversation_history = result.get("conversation_history", [])
            
            cline_metadata: Dict[str, Any] = {
                "task_id": result.get("task_id", ""),
                "profile": language.lower(),
                "ui_messages": conversation_history,  # Full conversation including tool calls
                "conversation_history": conversation_history,
                "repo_name": item.get("repo_name", ""),
                "repo_url": repo_url or "",
                "workspace_root": result.get("workspace_path", ""),
                "files_created": result.get("files_created", []),
                "execution_time_s": result.get("execution_time_s", 0.0),
                "success": result.get("success", False),
                "error": result.get("error"),
                "modal_worker": True,  # Flag to indicate this came from Modal
            }
            
            if result.get("error"):
                logger.warning(
                    "Modal worker error for %s (%s): %s",
                    item.get("instance_id"),
                    language,
                    result["error"],
                )
            
            return assistant_content, cline_metadata
            
        except Exception as exc:
            logger.exception(
                "Modal worker invocation failed for %s (%s): %s",
                item.get("instance_id"),
                language,
                exc,
            )
            return "", None

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
          - full assistant/user trajectory reconstructed from Cline messages
        and attaches raw Cline metadata (including the full `ui_messages` list) under
        `cline_metadata`.
        
        Supports two message formats:
        1. Old UI format: messages with `reasoning` and `text` keys
        2. Anthropic API format (from Modal): messages with `role` and `content` list
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

        # If we have Cline UI messages, reconstruct the full trajectory from them.
        if isinstance(cline_meta, dict) and isinstance(
            cline_meta.get("ui_messages"), list
        ):
            ui_messages = cline_meta["ui_messages"]
            
            # Detect format: Anthropic API format has 'role' and 'content' (list)
            # Old UI format has 'reasoning' and 'text' keys
            is_anthropic_format = (
                ui_messages and 
                isinstance(ui_messages[0], dict) and 
                "role" in ui_messages[0] and 
                isinstance(ui_messages[0].get("content"), list)
            )
            
            if is_anthropic_format:
                # Handle Anthropic API conversation history format
                conversations.extend(self._convert_anthropic_messages_to_conversations(ui_messages))
            else:
                # Handle old UI message format with reasoning/text keys
                filtered_messages = self._filter_complete_ui_messages(ui_messages)
                for msg in filtered_messages:
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

        if scored is not None:
            out_row["score"] = float(scored["scores"])
        else:
            out_row["score"] = None

        return out_row

    def _convert_anthropic_messages_to_conversations(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert Anthropic API conversation history to conversations format.
        
        The Anthropic API format has messages with:
        - role: "user" or "assistant"
        - content: list of {type: "text"|"thinking"|"tool_use"|"tool_result", ...}
        
        This converts them to the standard conversations format with:
        - from: "assistant" or "user" 
        - value: text content with XML-formatted tool calls/results
        """
        conversations: List[Dict[str, Any]] = []
        
        for msg in messages:
            if not isinstance(msg, dict):
                continue
                
            role = msg.get("role")
            content = msg.get("content", [])
            
            if not content:
                continue
            
            # Map role to from field
            from_field = "assistant" if role == "assistant" else "user"
            
            # Build value from content items
            parts: List[str] = []
            
            # Handle both list and string content formats
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    
                    item_type = item.get("type")
                    
                    if item_type == "text":
                        text = item.get("text", "")
                        if text:
                            parts.append(text)
                    elif item_type == "thinking":
                        thinking = item.get("thinking", "")
                        if thinking:
                            parts.append(f"<thinking>{thinking}</thinking>")
                    elif item_type == "tool_use":
                        # Format tool call as XML
                        tool_name = item.get("name", "")
                        tool_input = item.get("input", {})
                        tool_id = item.get("id", "")
                        tool_input_str = json.dumps(tool_input, indent=2) if isinstance(tool_input, dict) else str(tool_input)
                        parts.append(f"<tool_use id=\"{tool_id}\" name=\"{tool_name}\">\n{tool_input_str}\n</tool_use>")
                    elif item_type == "tool_result":
                        # Format tool result as XML
                        tool_use_id = item.get("tool_use_id", "")
                        result_content = item.get("content", "")
                        # Handle content that might be a list of text blocks
                        if isinstance(result_content, list):
                            result_parts = []
                            for rc in result_content:
                                if isinstance(rc, dict) and rc.get("type") == "text":
                                    result_parts.append(rc.get("text", ""))
                                elif isinstance(rc, str):
                                    result_parts.append(rc)
                            result_content = "\n".join(result_parts)
                        # Truncate very long tool results
                        if len(result_content) > 10000:
                            result_content = result_content[:10000] + "\n... [truncated]"
                        parts.append(f"<tool_result id=\"{tool_use_id}\">\n{result_content}\n</tool_result>")
            
            if parts:
                conversations.append({
                    "from": from_field,
                    "value": "\n\n".join(parts)
                })
        
        return conversations


if __name__ == "__main__":
    ClineAgentEnv.cli()
