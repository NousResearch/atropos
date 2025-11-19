import argparse
import logging
import threading
from typing import Any, Dict, Optional

import wandb
import zmq

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ZMQSidecar")


class ZMQLogAggregator:
    """
    A sidecar service that listens for log data over ZeroMQ and aggregates it
    into the centralized WandB run.
    """

    def __init__(self, port: int = 5555, context: Optional[zmq.Context] = None):
        self.port = port
        self.context = context or zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.running = False
        self.thread = None

    def start(self):
        """Start the aggregator thread."""
        if self.running:
            return

        try:
            self.socket.bind(f"tcp://*:{self.port}")
            logger.info(f"ZMQLogAggregator listening on port {self.port}")
        except zmq.ZMQError as e:
            logger.error(f"Failed to bind ZMQ socket on port {self.port}: {e}")
            raise

        self.running = True
        # In process mode, we run directly, not in a thread
        self._loop()

    def stop(self):
        """Stop the aggregator."""
        self.running = False
        try:
            self.socket.close()
        except Exception:
            pass

    def _handle_control_message(self, payload: Dict[str, Any]):
        """Handle control messages for lifecycle management."""
        msg_type = payload.get("_type")

        if msg_type == "init":
            config = payload.get("config", {})
            logger.info(
                f"Received INIT command. Starting WandB run: {config.get('group', 'unknown')}"
            )

            # Make sure we finish any existing run
            if wandb.run is not None:
                logger.info("Finishing existing WandB run before starting new one")
                wandb.finish()

            try:
                wandb.init(**config)
                logger.info(f"WandB run initialized: {wandb.run.id}")
            except Exception as e:
                logger.error(f"Failed to initialize WandB: {e}")

        elif msg_type == "reset":
            logger.info("Received RESET command. Finishing WandB run.")
            if wandb.run is not None:
                wandb.finish()
            else:
                logger.info("No active WandB run to finish.")

    def _loop(self):
        """Main listening loop."""
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)

        logger.info("ZMQ Sidecar loop started")

        while self.running:
            try:
                # check if open
                socks = dict(poller.poll(1000))
                if self.socket in socks:
                    # pyobj in case of some other data stuff later
                    payload = self.socket.recv_pyobj()

                    # Check if it's a control message
                    if isinstance(payload, dict) and "_type" in payload:
                        self._handle_control_message(payload)
                        continue

                    # Otherwise treat as log payload
                    if wandb.run is not None:
                        wandb.log(payload)
                    else:
                        # Optional: accumulate logs buffer or just debug log
                        # For now, we just debug log to avoid memory leaks
                        pass
                        # logger.debug("Received log payload (wandb not active)")

            except Exception as e:
                logger.error(f"Error in ZMQLogAggregator loop: {e}")
                # Don't break on transient errors, but logging essential
                # if not self.running:
                #    break


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
