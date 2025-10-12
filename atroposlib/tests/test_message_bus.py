import asyncio
import socket
import sys
import time
import types
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
from fastapi.testclient import TestClient

try:
    import wandb  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    wandb = types.SimpleNamespace(
        Settings=lambda **kwargs: SimpleNamespace(**kwargs),
        Table=lambda *args, **kwargs: None,
        init=lambda **kwargs: None,
        log=lambda *args, **kwargs: None,
        finish=lambda *args, **kwargs: None,
    )
    sys.modules["wandb"] = wandb

from atroposlib.api import server
from atroposlib.envs.base import BaseEnv
from atroposlib.utils.message_bus import MessageBusClient


def _get_free_tcp_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class DummyTable:
    def __init__(self, columns: List[str]):
        self.columns = columns
        self.rows: List[Tuple[Any, ...]] = []

    def add_data(self, *values: Any) -> None:
        self.rows.append(values)


def test_api_message_bus_routes_metrics_to_wandb(monkeypatch):
    port = _get_free_tcp_port()
    endpoint = f"tcp://127.0.0.1:{port}"

    monkeypatch.setattr(server, "MESSAGE_BUS_ENABLED", True)
    monkeypatch.setattr(server, "MESSAGE_BUS_ENDPOINT", endpoint)

    init_calls: List[Dict[str, Any]] = []
    logged_calls: List[Tuple[Dict[str, Any], Any]] = []

    monkeypatch.setattr(server.wandb, "Table", DummyTable)
    monkeypatch.setattr(
        server.wandb,
        "init",
        lambda **kwargs: init_calls.append(kwargs) or object(),
    )
    monkeypatch.setattr(server.wandb, "log", lambda metrics, step=None: logged_calls.append((metrics, step)))
    monkeypatch.setattr(server.wandb, "finish", lambda *args, **kwargs: None)

    with TestClient(server.app) as client:
        response = client.post(
            "/register",
            json={
                "wandb_group": "test_group",
                "wandb_project": "test_project",
                "batch_size": 4,
                "max_token_len": 128,
                "checkpoint_dir": "/tmp",
                "save_checkpoint_interval": 10,
                "starting_step": 0,
                "num_steps": 100,
            },
        )
        assert response.status_code == 200
        assert init_calls, "wandb.init should be invoked during trainer registration"

        env_response = client.post(
            "/register-env",
            json={
                "max_token_length": 128,
                "desired_name": "env",
                "weight": 1.0,
                "group_size": 2,
                "min_batch_allocation": None,
            },
        )
        data = env_response.json()
        assert env_response.status_code == 200
        message_bus = data.get("message_bus")
        assert message_bus, "message bus details should be returned when enabled"
        assert message_bus["endpoint"] == endpoint

        bus_client = MessageBusClient(
            endpoint=message_bus["endpoint"],
            token=message_bus["token"],
        )
        payload = {
            "type": "metrics",
            "metrics": {"train/foo": 1.0},
            "server_metrics": {"server_metric": 2.0},
            "rollouts": [["hello world", 0.5]],
            "step": 5,
            "wandb_prepend": message_bus["wandb_prepend"],
        }
        asyncio.run(bus_client.send_json(payload))
        asyncio.run(bus_client.close())

        for _ in range(20):
            if logged_calls:
                break
            time.sleep(0.05)

        assert logged_calls, "Metrics sent over the message bus should reach wandb.log"
        metrics_logged, step_logged = logged_calls[-1]
        expected_prefix = message_bus["wandb_prepend"]
        assert metrics_logged[f"{expected_prefix}_train/foo"] == 1.0
        assert metrics_logged["server_metric"] == 2.0
        rollout_key = f"{expected_prefix}_train/rollouts"
        assert rollout_key in metrics_logged
        assert metrics_logged[rollout_key].rows[0] == ("hello world", 0.5)
        assert step_logged == 5

        client.get("/reset_data")


class DummyMessageBusClient:
    def __init__(self):
        self.messages: List[Dict[str, Any]] = []

    async def send_json(self, payload: Dict[str, Any]) -> None:
        self.messages.append(payload)


class DummyServer:
    async def wandb_metrics(self, metrics_dict: Optional[Dict[str, Any]], server_name: Optional[str]):
        metrics_dict = metrics_dict or {}
        metrics_dict[f"{server_name}_latency"] = np.float32(3.25)
        return metrics_dict


def _build_dummy_env(use_bus: bool, monkeypatch) -> Tuple[Any, DummyMessageBusClient]:
    dummy_client = DummyMessageBusClient() if use_bus else None

    server_manager = SimpleNamespace(servers=[DummyServer()])
    config = SimpleNamespace(
        use_wandb=True,
        num_rollouts_per_group_for_logging=1,
        group_size=2,
        num_rollouts_to_keep=8,
    )

    dummy = SimpleNamespace(
        server=server_manager,
        message_bus_client=dummy_client,
        config=config,
        completion_lengths=[1, 3],
        rollouts_for_wandb=[[("decoded text", 0.9)]],
        max_token_len=128,
        wandb_prepend="env_0",
        curr_step=7,
        env_id=42,
        message_bus_env_name="env",
        message_bus_wandb_prepend="env_0",
        task_duration=[0.5, 0.7],
        succeeded_task_duration=[0.3, 0.4],
        failed_task_duration=[],
        mainloop_timings=[0.1, 0.2],
        workers_added_list=[1, 2],
    )

    dummy._sanitize_for_json = BaseEnv._sanitize_for_json.__get__(dummy, BaseEnv)
    dummy.create_rollout_table = BaseEnv.create_rollout_table.__get__(dummy, BaseEnv)
    dummy.perf_stats = BaseEnv.perf_stats.__get__(dummy, BaseEnv)

    if not use_bus:
        monkeypatch.setattr("atroposlib.envs.base.wandb.Table", DummyTable)

    return dummy, dummy_client


def test_base_env_wandb_log_uses_message_bus(monkeypatch):
    dummy_env, dummy_client = _build_dummy_env(use_bus=True, monkeypatch=monkeypatch)

    def fail_log(*args, **kwargs):
        raise AssertionError("wandb.log should not be called when message bus is active")

    monkeypatch.setattr("atroposlib.envs.base.wandb.log", fail_log)

    asyncio.run(
        BaseEnv.wandb_log(
            dummy_env,
            {"custom_metric": np.float32(1.5)},
        )
    )

    assert dummy_client.messages, "Message bus client should receive payloads"
    payload = dummy_client.messages[0]
    assert payload["env_id"] == 42
    assert payload["metrics"]["custom_metric"] == pytest.approx(1.5)
    assert payload["server_metrics"]["server_0_latency"] == pytest.approx(3.25)
    assert payload["rollouts"][0][0][0] == "decoded text"
    assert dummy_env.rollouts_for_wandb == []
    assert dummy_env.completion_lengths == []


def test_base_env_wandb_log_falls_back_to_local_wandb(monkeypatch):
    dummy_env, _ = _build_dummy_env(use_bus=False, monkeypatch=monkeypatch)

    logged_calls: List[Tuple[Dict[str, Any], Any]] = []

    monkeypatch.setattr(
        "atroposlib.envs.base.wandb.log",
        lambda metrics, step=None: logged_calls.append((metrics, step)),
    )

    asyncio.run(
        BaseEnv.wandb_log(
            dummy_env,
            {"local_metric": 2.0},
        )
    )

    assert logged_calls, "wandb.log should be called when message bus is absent"
    metrics_logged, step_logged = logged_calls[0]
    assert metrics_logged["env_0_local_metric"] == 2.0
    rollout_table = metrics_logged["env_0_train/rollouts"]
    assert isinstance(rollout_table, DummyTable)
    assert rollout_table.rows[0] == ("decoded text", 0.9)
    assert step_logged == 7
