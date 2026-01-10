import contextlib
import socket
import subprocess
import sys
import time

import requests


def get_free_port(host: str = "127.0.0.1") -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def check_api_running(base_url: str) -> bool:
    try:
        data = requests.get(f"{base_url}/info", timeout=1)
        return data.status_code == 200
    except requests.exceptions.RequestException:
        return False


def launch_api_for_testing(
    *,
    host: str = "127.0.0.1",
    port: int | None = None,
    max_wait_for_api: int = 10,
) -> tuple[subprocess.Popen, str]:
    if port is None:
        port = get_free_port(host)

    base_url = f"http://{host}:{port}"

    api_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "atroposlib.cli.run_api",
            "--host",
            host,
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    counter = 0
    while not check_api_running(base_url):
        if api_proc.poll() is not None:
            raise RuntimeError(
                f"API server exited early with code {api_proc.returncode}"
            )
        time.sleep(1)
        counter += 1
        if counter > max_wait_for_api:
            api_proc.terminate()
            raise TimeoutError("API server did not start in time.")

    return api_proc, base_url
