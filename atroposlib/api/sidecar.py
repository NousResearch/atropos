import argparse
import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import wandb
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
    by (step, env_type) and listens for log data over ZeroMQ and aggregates it
    into the centralized WandB run.

    """

    def __init__(self, port: int = 5555, context: Optional[zmq.Context] = None):
        self.port = port
        self.context = context or zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.running = False
        self.thread = None

        self.registered_envs: Dict[str, Set[str]] = defaultdict(set)
        self.pending_metrics: Dict[
            Tuple[int, str], Dict[str, List[Tuple[str, Any]]]
        ] = {}
        self.env_reported: Dict[Tuple[int, str], Set[str]] = defaultdict(set)
        self.pending_timestamps: Dict[Tuple[int, str], float] = {}

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

    def _handle_control_message(self, payload: Dict[str, Any]):
        msg_type = payload.get("_type")

        if msg_type == "init":
            config = payload.get("config", {})
            logger.info(f"Received INIT: {config.get('group', 'unknown')}")

            if wandb.run is not None:
                logger.info("Finishing existing WandB run")
                wandb.finish()

            try:
                wandb.init(**config)
                logger.info(f"WandB run initialized: {wandb.run.id}")
            except Exception as e:
                logger.error(f"Failed to initialize WandB: {e}")

        elif msg_type == "reset":
            logger.info("Received RESET")
            if wandb.run is not None:
                wandb.finish()

        elif msg_type == "env_register":
            env_type = payload.get("env_type")
            instance = payload.get("instance")
            if env_type and instance:
                self.registered_envs[env_type].add(instance)
                logger.info(f"Registered {instance} for {env_type}")

        elif msg_type == "env_disconnect":
            env_type = payload.get("env_type")
            instance = payload.get("instance")
            if env_type and instance:
                self.registered_envs[env_type].discard(instance)
                logger.info(f"Disconnected {instance} from {env_type}")
                self._check_pending_after_disconnect(env_type)

    def _check_pending_after_disconnect(self, env_type: str):
        keys_to_check = [k for k in self.pending_metrics if k[1] == env_type]
        for key in keys_to_check:
            if self._all_reported(key):
                self._aggregate_and_log(key)

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
            if wandb.run is not None:
                wandb.log(payload, step=step)
            return

        key = (step, env_type)

        if key not in self.pending_metrics:
            self.pending_metrics[key] = defaultdict(list)
            self.pending_timestamps[key] = time.time()

        for metric_name, value in payload.items():
            self.pending_metrics[key][metric_name].append((instance, value))

        self.env_reported[key].add(instance)

        if self._all_reported(key):
            self._aggregate_and_log(key)

    def _aggregate_and_log(self, key: Tuple[int, str]):
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
                # just some extra stats on top of the individual instance metrics
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

        if wandb.run is not None and final_metrics:
            wandb.log(final_metrics, step=step)
            logger.debug(f"Logged aggregated metrics for {env_type} step {step}")

    def _check_timeouts(self):
        now = time.time()
        stale_keys = [
            k
            for k, ts in self.pending_timestamps.items()
            if now - ts > AGGREGATION_TIMEOUT
        ]
        for key in stale_keys:
            step, env_type = key
            logger.warning(f"Timeout for {env_type} step {step}, logging partial data")
            self._aggregate_and_log(key)

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
