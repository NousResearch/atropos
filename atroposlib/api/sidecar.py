import argparse
import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import zmq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ZMQSidecar")

AGGREGATION_TIMEOUT = 60.0


class ZMQLogAggregator:
    """
    Sidecar service that aggregates metrics from multiple environment instances
    by (step, env_type) and routes aggregated data to the leader instance for
    each env_type. The leader is responsible for logging to wandb.
    """

    def __init__(self, port: int = 5555, context: Optional[zmq.Context] = None):
        self.port = port
        self.context = context or zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.running = False

        self.registered_envs: Dict[str, Set[str]] = defaultdict(set)
        self.pending_metrics: Dict[
            Tuple[int, str], Dict[str, List[Tuple[str, Any]]]
        ] = {}
        self.env_reported: Dict[Tuple[int, str], Set[str]] = defaultdict(set)
        self.pending_timestamps: Dict[Tuple[int, str], float] = {}

        # Leader info per env_type: {env_type: {"port": int, "socket": zmq.Socket}}
        self.leaders: Dict[str, Dict[str, Any]] = {}

    def start(self):
        if self.running:
            return

        try:
            self.socket.bind(f"tcp://*:{self.port}")
            logger.info(f"ZMQLogAggregator listening on port {self.port}")
        except zmq.ZMQError as e:
            logger.error(f"Failed to bind ZMQ socket on port {self.port}: {e}")
            raise

        self.running = True
        self._loop()

    def stop(self):
        self.running = False
        try:
            self.socket.close()
        except Exception:
            pass
        for leader_info in self.leaders.values():
            try:
                leader_info.get("socket", None).close()
            except Exception:
                pass

    def _connect_to_leader(self, env_type: str, port: int):
        """Create a ZMQ PUSH socket to send aggregated data to the leader."""
        if env_type in self.leaders:
            return

        try:
            socket = self.context.socket(zmq.PUSH)
            socket.setsockopt(zmq.SNDHWM, 10000)
            socket.setsockopt(zmq.LINGER, 1000)
            socket.connect(f"tcp://localhost:{port}")
            self.leaders[env_type] = {"port": port, "socket": socket}
            logger.info(f"Connected to leader for {env_type} on port {port}")
        except Exception as e:
            logger.error(f"Failed to connect to leader for {env_type}: {e}")

    def _handle_control_message(self, payload: Dict[str, Any]):
        msg_type = payload.get("_type")

        if msg_type == "env_register":
            env_type = payload.get("env_type")
            instance = payload.get("instance")
            is_leader = payload.get("is_leader", False)
            leader_receive_port = payload.get("leader_receive_port")

            if env_type and instance:
                self.registered_envs[env_type].add(instance)
                logger.info(f"Registered {instance} for {env_type}")

                if is_leader and leader_receive_port:
                    self._connect_to_leader(env_type, leader_receive_port)

        elif msg_type == "env_disconnect":
            env_type = payload.get("env_type")
            instance = payload.get("instance")
            was_leader = payload.get("was_leader", False)

            if env_type and instance:
                self.registered_envs[env_type].discard(instance)
                logger.info(f"Disconnected {instance} from {env_type}")
                self._check_pending_after_disconnect(env_type)

                if was_leader and env_type in self.leaders:
                    try:
                        self.leaders[env_type]["socket"].close()
                    except Exception:
                        pass
                    del self.leaders[env_type]
                    logger.info(f"Removed leader connection for {env_type}")

    def _check_pending_after_disconnect(self, env_type: str):
        keys_to_check = [k for k in self.pending_metrics if k[1] == env_type]
        for key in keys_to_check:
            if self._all_reported(key):
                self._aggregate_and_send(key)

    def _all_reported(self, key: Tuple[int, str]) -> bool:
        step, env_type = key
        expected = self.registered_envs.get(env_type, set())
        reported = self.env_reported.get(key, set())
        return bool(expected) and reported >= expected

    def _handle_log_payload(self, payload: Dict[str, Any]):
        step = payload.pop("_step", None)
        env_type = payload.pop("_env_type", None)
        instance = payload.pop("_instance", None)

        if env_type is None or instance is None:
            logger.warning("Received log without env_type or instance, dropping")
            return

        key = (step, env_type)

        if key not in self.pending_metrics:
            self.pending_metrics[key] = defaultdict(list)
            self.pending_timestamps[key] = time.time()

        for metric_name, value in payload.items():
            self.pending_metrics[key][metric_name].append((instance, value))

        self.env_reported[key].add(instance)

        if self._all_reported(key):
            self._aggregate_and_send(key)

    def _aggregate_and_send(self, key: Tuple[int, str]):
        """Aggregate metrics and send to the leader for this env_type."""
        step, env_type = key
        metrics = self.pending_metrics.pop(key, {})
        self.env_reported.pop(key, None)
        self.pending_timestamps.pop(key, None)

        if not metrics:
            return

        final_metrics = {}

        for metric_name, values in metrics.items():
            for instance, value in values:
                final_metrics[f"{env_type}/instances/{instance}/{metric_name}"] = value

            numeric_values = [v for _, v in values if isinstance(v, (int, float))]
            if numeric_values:
                final_metrics[f"{env_type}/aggregated/{metric_name}_mean"] = np.mean(
                    numeric_values
                )
                final_metrics[f"{env_type}/aggregated/{metric_name}_std"] = np.std(
                    numeric_values
                )
                final_metrics[f"{env_type}/aggregated/{metric_name}_min"] = np.min(
                    numeric_values
                )
                final_metrics[f"{env_type}/aggregated/{metric_name}_max"] = np.max(
                    numeric_values
                )

        if not final_metrics:
            return

        final_metrics["_step"] = step

        leader_info = self.leaders.get(env_type)
        if leader_info and leader_info.get("socket"):
            try:
                leader_info["socket"].send_pyobj(final_metrics, flags=zmq.NOBLOCK)
                logger.debug(
                    f"Sent aggregated metrics for {env_type} step {step} to leader"
                )
            except zmq.Again:
                logger.warning(
                    f"Leader buffer full for {env_type}, dropping aggregated data"
                )
            except Exception as e:
                logger.error(f"Failed to send to leader for {env_type}: {e}")
        else:
            logger.warning(
                f"No leader connected for {env_type}, dropping aggregated data"
            )

    def _check_timeouts(self):
        now = time.time()
        stale_keys = [
            k
            for k, ts in self.pending_timestamps.items()
            if now - ts > AGGREGATION_TIMEOUT
        ]
        for key in stale_keys:
            step, env_type = key
            logger.warning(f"Timeout for {env_type} step {step}, sending partial data")
            self._aggregate_and_send(key)

    def _loop(self):
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        logger.info("ZMQ Sidecar loop started")

        last_timeout_check = time.time()

        while self.running:
            try:
                socks = dict(poller.poll(1000))

                if self.socket in socks:
                    payload = self.socket.recv_pyobj()

                    if isinstance(payload, dict) and "_type" in payload:
                        self._handle_control_message(payload)
                    else:
                        self._handle_log_payload(payload)

                if time.time() - last_timeout_check > 10:
                    self._check_timeouts()
                    last_timeout_check = time.time()

            except Exception as e:
                logger.error(f"Error in ZMQLogAggregator loop: {e}")


def main():
    parser = argparse.ArgumentParser(description="Atropos ZMQ Logging Sidecar")
    parser.add_argument("--port", type=int, default=5555, help="Port to listen on")
    args = parser.parse_args()

    aggregator = ZMQLogAggregator(port=args.port)
    try:
        aggregator.start()
    except KeyboardInterrupt:
        logger.info("Stopping ZMQ Sidecar...")
        aggregator.stop()


if __name__ == "__main__":
    main()
