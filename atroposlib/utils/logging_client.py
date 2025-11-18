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

        # 10 secs
        self.socket.setsockopt(zmq.SNDHWM, 10000)

        self.socket.setsockopt(zmq.LINGER, 1000)

        logger.info(f"Connecting ZMQLogger to {address}")
        self.socket.connect(address)

    def log(
        self,
        data: Dict[str, Any],
        step: Optional[int] = None,
        commit: Optional[bool] = None,
    ):
        """
        Send log data to the central server.

        Args:
            data: Dictionary of metrics to log
            step: Optional step number (wandb.log supports this)
            commit: Optional commit flag (wandb.log supports this)
        """
        if step is not None:
            data["_step"] = step

        try:
            # pyobj in case we do some other data stuff later
            self.socket.send_pyobj(data, flags=zmq.NOBLOCK)
        except zmq.Again:
            logger.warning("ZMQLogger buffer full, dropping log packet")
        except Exception as e:
            logger.error(f"Failed to send log data: {e}")

    def close(self):
        self.socket.close()


def setup_weave_for_worker(project_name: str):
    """
    Configure environment variables so Weave uses the correct project,
    even if wandb.init() is not called locally.
    """
    if project_name:
        os.environ["WEAVE_PROJECT"] = project_name
        os.environ["WANDB_PROJECT"] = project_name
