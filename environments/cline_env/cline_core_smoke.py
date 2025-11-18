import contextlib
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import grpc
from dotenv import load_dotenv
from google.protobuf import descriptor_pb2, descriptor_pool, json_format, message_factory

from cline_core_launcher import ClineCoreConfig, ClineCoreProcess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProtoBusClient:
    """Minimal dynamic gRPC client built from descriptor_set.pb."""

    def __init__(self, descriptor_path: Path, address: str, ready_timeout: float = 30.0) -> None:
        if not descriptor_path.exists():
            raise FileNotFoundError(
                f"descriptor_set.pb not found at {descriptor_path}. "
                "Run `npm run compile-standalone` inside the vendored Cline repo first."
            )

        descriptor_bytes = descriptor_path.read_bytes()
        descriptor_set = descriptor_pb2.FileDescriptorSet()
        descriptor_set.ParseFromString(descriptor_bytes)

        pool = descriptor_pool.DescriptorPool()
        for file_proto in descriptor_set.file:
            pool.Add(file_proto)

        self._pool = pool
        self._factory = message_factory.MessageFactory(pool)
        self._message_cache: Dict[str, Any] = {}

        self._channel = grpc.insecure_channel(address)
        grpc.channel_ready_future(self._channel).result(timeout=ready_timeout)

    def enum_value(self, full_name: str, name: str) -> int:
        enum_descriptor = self._pool.FindEnumTypeByName(full_name)
        try:
            return enum_descriptor.values_by_name[name].number
        except KeyError as exc:
            raise ValueError(f"Enum {full_name} has no value named {name}") from exc

    def close(self) -> None:
        self._channel.close()

    def _get_message_class(self, full_name: str):
        if full_name not in self._message_cache:
            descriptor = self._pool.FindMessageTypeByName(full_name)
            self._message_cache[full_name] = self._factory.GetPrototype(descriptor)
        return self._message_cache[full_name]

    def new_message(self, full_name: str):
        cls = self._get_message_class(full_name)
        return cls()

    def unary_unary(self, method: str, request, response_type: str):
        stub = self._channel.unary_unary(
            method,
            request_serializer=lambda msg: msg.SerializeToString(),
            response_deserializer=lambda data: self._get_message_class(response_type).FromString(data),
        )
        return stub(request)

    def unary_stream(self, method: str, request, response_type: str):
        stub = self._channel.unary_stream(
            method,
            request_serializer=lambda msg: msg.SerializeToString(),
            response_deserializer=lambda data: self._get_message_class(response_type).FromString(data),
        )
        return stub(request)


def resolve_descriptor_path(cline_root: Path) -> Path:
    candidates = [
        cline_root / "dist-standalone/proto/descriptor_set.pb",
        cline_root / "proto/descriptor_set.pb",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate descriptor_set.pb. Checked:\n" + "\n".join(str(c) for c in candidates)
    )


def message_summary(message) -> str:
    data = json_format.MessageToDict(message, preserving_proto_field_name=True)
    text = data.get("text")
    say = data.get("say")
    ask = data.get("ask")
    msg_type = data.get("type")
    return f"type={msg_type} ask={ask} say={say} text={text!r}"


def wait_for_condition(predicate, timeout: float, interval: float = 0.5) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def configure_anthropic_provider(client: ProtoBusClient, api_key: str, model_id: str) -> None:
    logger.info("Updating API configuration to use Anthropic provider")
    request = client.new_message("cline.UpdateApiConfigurationPartialRequest")
    request.metadata.CopyFrom(client.new_message("cline.Metadata"))

    api_config = request.api_configuration
    api_config.api_key = api_key
    anthropic_enum = client.enum_value("cline.ApiProvider", "ANTHROPIC")
    api_config.plan_mode_api_provider = anthropic_enum
    api_config.act_mode_api_provider = anthropic_enum
    api_config.plan_mode_api_model_id = model_id
    api_config.act_mode_api_model_id = model_id

    # Update mask controls which fields are applied
    request.update_mask.paths.extend(
        [
            "apiKey",
            "planModeApiProvider",
            "actModeApiProvider",
            "planModeApiModelId",
            "actModeApiModelId",
        ]
    )

    client.unary_unary("/cline.ModelsService/updateApiConfigurationPartial", request, "cline.Empty")


def exercise_grpc_endpoints(cline_root: Path, protobus_port: int, anthropic_key: str, anthropic_model: str) -> None:
    descriptor_path = resolve_descriptor_path(cline_root)
    client = ProtoBusClient(descriptor_path, f"127.0.0.1:{protobus_port}")

    try:
        logger.info("Initializing Cline webview via UiService")
        client.unary_unary(
            "/cline.UiService/initializeWebview",
            client.new_message("cline.EmptyRequest"),
            "cline.Empty",
        )

        configure_anthropic_provider(client, anthropic_key, anthropic_model)

        logger.info("Subscribing to partial message stream")
        messages: List[Any] = []
        stream_ready = threading.Event()
        stream_stop = threading.Event()
        stream_error: List[Optional[BaseException]] = [None]
        call_holder: Dict[str, Any] = {}

        def stream_consumer() -> None:
            empty_request = client.new_message("cline.EmptyRequest")
            call = client.unary_stream(
                "/cline.UiService/subscribeToPartialMessage",
                empty_request,
                "cline.ClineMessage",
            )
            call_holder["call"] = call
            stream_ready.set()
            try:
                for message in call:
                    messages.append(message)
                    logger.info("Partial message #%d: %s", len(messages), message_summary(message))
                    if len(messages) >= 4 or stream_stop.is_set():
                        call.cancel()
                        break
            except grpc.RpcError as exc:
                if not stream_stop.is_set() and exc.code() != grpc.StatusCode.CANCELLED:
                    stream_error[0] = exc
            finally:
                stream_stop.set()

        consumer_thread = threading.Thread(target=stream_consumer, name="partial-messages", daemon=True)
        consumer_thread.start()

        if not stream_ready.wait(timeout=20.0):
            raise TimeoutError("Timed out waiting to subscribe to partial messages")

        metadata = client.new_message("cline.Metadata")
        new_task_request = client.new_message("cline.NewTaskRequest")
        new_task_request.metadata.CopyFrom(metadata)
        new_task_request.text = "List the files in the current workspace and summarize the main project."

        logger.info("Issuing TaskService.newTask request")
        response = client.unary_unary("/cline.TaskService/newTask", new_task_request, "cline.String")
        task_id = response.value
        logger.info("Task started with ID %s", task_id)

        if not wait_for_condition(lambda: len(messages) >= 3, timeout=120.0):
            raise TimeoutError("Did not receive enough partial messages from UiService")

        stream_stop.set()
        call = call_holder.get("call")
        if call is not None:
            with contextlib.suppress(Exception):
                call.cancel()
        if 'consumer_thread' in locals():
            consumer_thread.join(timeout=10.0)

        if stream_error[0]:
            raise RuntimeError(f"Partial message stream failed: {stream_error[0]}")
    finally:
        stream_stop.set()
        call = call_holder.get("call")
        if call is not None:
            with contextlib.suppress(Exception):
                call.cancel()
        if 'consumer_thread' in locals():
            consumer_thread.join(timeout=5.0)
        client.close()


def main() -> None:
    load_dotenv()
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is required for the gRPC smoke test. "
            "Set it in your environment or .env file before running this script."
        )
    anthropic_model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")

    this_dir = Path(__file__).resolve().parent
    cline_root = this_dir / "cline"
    if not cline_root.exists():
        raise FileNotFoundError(
            f"Expected Cline repo at {cline_root}. "
            "Ensure the Cline submodule is checked out at environments/cline_env/cline."
        )

    config = ClineCoreConfig(
        cline_root=cline_root,
        protobus_port=26040,
        hostbridge_port=26041,
        workspace_dir=None,
        use_coverage=False,
    )

    logger.info("Starting Cline core gRPC server for smoke test")
    proc = ClineCoreProcess(config)
    try:
        proc.start(timeout=60.0)
        logger.info("Cline core is listening on 127.0.0.1:%d", config.protobus_port)
        exercise_grpc_endpoints(cline_root, config.protobus_port, anthropic_key, anthropic_model)
        logger.info("Smoke test succeeded")
    except Exception as exc:
        logger.exception("Cline core smoke test failed: %s", exc)
        raise
    finally:
        logger.info("Stopping Cline core process")
        proc.stop()


if __name__ == "__main__":
    main()
