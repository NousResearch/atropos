"""
Run the Trajectory API server and the ZMQ Sidecar process.
"""

import argparse
import multiprocessing
import os
import signal
import sys
import time
from typing import Optional

import uvicorn


def run_sidecar(port: int):
    """
    Run the ZMQ sidecar process.
    """
    from atroposlib.api.sidecar import main as sidecar_main

    # Set the process title if possible
    try:
        import setproctitle

        setproctitle.setproctitle("atropos-zmq-sidecar")
    except ImportError:
        pass

    sys.argv = ["atropos-sidecar", "--port", str(port)]
    sidecar_main()


def main():
    """
    Run the API server.
    Args:
        host: The host to run the API server on.
        port: The port to run the API server on.
        reload: Whether to reload the API server on code changes.
        zmq_port: The port to run the ZMQ sidecar on.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--zmq-port", type=int, default=5555)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    # Set the ZMQ port environment variable for the API server to discover
    os.environ["ATROPOS_ZMQ_PORT"] = str(args.zmq_port)

    # Start the ZMQ sidecar process
    sidecar_process = multiprocessing.Process(
        target=run_sidecar, args=(args.zmq_port,), daemon=True
    )
    sidecar_process.start()
    print(f"Started ZMQ sidecar on port {args.zmq_port} (pid={sidecar_process.pid})")

    try:

        uvicorn.run(
            "atroposlib.api:app", host=args.host, port=args.port, reload=args.reload
        )
    except KeyboardInterrupt:
        print("Stopping API server...")
    finally:
        if sidecar_process.is_alive():
            print("Stopping ZMQ sidecar...")
            sidecar_process.terminate()
            sidecar_process.join(timeout=2)
            if sidecar_process.is_alive():
                sidecar_process.kill()


if __name__ == "__main__":

    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))
    main()
