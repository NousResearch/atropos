import logging
import os
from typing import Any, Dict, Optional

import zmq

logger = logging.getLogger(__name__)


class ZMQLogger:
    """
    A client for the ZMQLogAggregator. Replaces local wandb.log calls
    by pushing data to the central API server.
    """

    def __init__(self, address: str, context: Optional[zmq.Context] = None):
        """
        Args:
            address: Full ZMQ address ("tcp://1.2.3.4:5555")
            context: Optional existing ZMQ context
        """
        self.context = context or zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.setsockopt(zmq.SNDHWM, 10000)
        self.socket.setsockopt(zmq.LINGER, 1000)

        logger.info(f"Connecting ZMQLogger to {address}")
        self.socket.connect(address)

    def log(
        self,
        data: Dict[str, Any],
        step: Optional[int] = None,
        env_type: Optional[str] = None,
        instance_name: Optional[str] = None,
        commit: Optional[bool] = None,
    ):
        """
        Send log data to the central server.

        Args:
            data: Dictionary of metrics to log
            step: Optional step number
            env_type: Environment type / name for aggregation (math)
            instance_name: Instance number (math_1)
            commit: Optional commit flag (wandb.log compatibility)
        """
        if step is not None:
            data["_step"] = step
        if env_type is not None:
            data["_env_type"] = env_type
        if instance_name is not None:
            data["_instance"] = instance_name

        try:
            self.socket.send_pyobj(data, flags=zmq.NOBLOCK)
        except zmq.Again:
            logger.warning("ZMQLogger buffer full, dropping log packet")
        except Exception as e:
            logger.error(f"Failed to send log data: {e}")

    def close(self):
        self.socket.close()


class ZMQLogReceiver:
    """
    A receiver for aggregated log data. Used by leader instances to receive
    aggregated metrics from the sidecar and log them to wandb.
    """

    def __init__(self, port: int, context: Optional[zmq.Context] = None):
        """
        Args:
            port: Port to bind to for receiving aggregated data
            context: Optional existing ZMQ context
        """
        self.port = port
        self.context = context or zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.setsockopt(zmq.RCVHWM, 10000)
        self.socket.bind(f"tcp://*:{port}")
        self.running = False
        logger.info(f"ZMQLogReceiver bound to port {port}")

    def recv_nowait(self) -> Optional[Dict[str, Any]]:
        """
        Non-blocking receive of aggregated data.

        Returns:
            Dictionary of aggregated metrics if available, None otherwise
        """
        try:
            return self.socket.recv_pyobj(flags=zmq.NOBLOCK)
        except zmq.Again:
            return None
        except Exception as e:
            logger.error(f"Failed to receive log data: {e}")
            return None

    def close(self):
        self.socket.close()


def setup_weave_for_worker(
    project_name: str, group_name: Optional[str] = None, run_id: Optional[str] = None
):
    """
    Configure environment variables so Weave uses the correct project,
    even if wandb.init() is not called locally.
    """
    if project_name:
        os.environ["WEAVE_PROJECT"] = project_name
        os.environ["WANDB_PROJECT"] = project_name

    if group_name:
        os.environ["WANDB_GROUP"] = group_name

    if run_id:
        # Weave often respects WANDB_RUN_ID to associate calls with a specific run
        os.environ["WANDB_RUN_ID"] = run_id
