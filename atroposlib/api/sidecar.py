import threading
import zmq
import logging
import wandb
from typing import Optional

logger = logging.getLogger(__name__)

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
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the aggregator thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self.socket.close()

    def _loop(self):
        """Main listening loop."""
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        
        while self.running:
            try:
                # check if open
                socks = dict(poller.poll(1000))
                if self.socket in socks:
                    # pyobj in case of  some other data stuff later 
                    payload = self.socket.recv_pyobj()

                    if wandb.run is not None:
                       
                        wandb.log(payload)
                    else:
                        
                        logger.debug(f"Received log payload (wandb not active): {payload.keys() if isinstance(payload, dict) else 'unknown'}")
                        
            except Exception as e:
                logger.error(f"Error in ZMQLogAggregator loop: {e}")
                if not self.running:
                    break

