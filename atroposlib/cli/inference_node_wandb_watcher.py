import argparse
import time
from urllib.parse import urlparse

import requests
import wandb

from atroposlib.utils.logging_client import ZMQLogger


def run(api_addr, tp, node_num):
    print(f"Starting up with {api_addr}, {tp}, {node_num}", flush=True)
    zmq_logger = None

    while True:
        try:
            data = requests.get(f"{api_addr}/wandb_info").json()
            wandb_group = data.get("group")
            wandb_project = data.get("project")
            zmq_port = data.get("zmq_port")
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            wandb_project = None
            wandb_group = None
            zmq_port = None
            print("Waiting for init...")

        if wandb_project is None:
            time.sleep(1)
        else:
            if zmq_port:
                try:
                    parsed = urlparse(api_addr)
                    host = parsed.hostname or "localhost"
                    zmq_addr = f"tcp://{host}:{zmq_port}"
                    zmq_logger = ZMQLogger(address=zmq_addr)
                    print(f"Connected to ZMQ Logger at {zmq_addr}")
                    break
                except Exception as e:
                    print(f"Failed to connect ZMQ: {e}")
                    # does our existing/old wandb setup if zmq isn't open

            wandb.init(
                project=wandb_project, group=wandb_group, name=f"inf_node_{node_num}"
            )
            break

    curr_step = 0
    health_statuses = {
        f"server/server_health_{node_num}_{i}": 0.0 for i in range(8 // tp)
    }
    while True:
        try:
            data = requests.get(f"{api_addr}/status").json()
            step = data["current_step"]
            if step > curr_step:
                if zmq_logger:
                    zmq_logger.log(health_statuses, step=step)
                else:
                    wandb.log(health_statuses, step=step)
                curr_step = step
        except Exception as e:
            print(f"Error fetching status: {e}")

        time.sleep(60)
        # Check on each server
        for i in range(8 // tp):
            try:
                health_status = requests.get(
                    f"http://localhost:{9000 + i}/health_generate"
                ).status_code
                health_statuses[f"server/server_health_{node_num}_{i}"] = (
                    1 if health_status == 200 else 0
                )
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                health_statuses[f"server/server_health_{node_num}_{i}"] = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_addr", type=str, required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--node_num", type=int, required=True)
    args = parser.parse_args()
    run(args.api_addr, args.tp, args.node_num)


if __name__ == "__main__":
    main()
