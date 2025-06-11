import subprocess
import time

import requests


def check_api_running() -> bool:
    try:
        data = requests.get("http://localhost:8000/info")
        return data.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def launch_api_for_testing(max_wait_for_api: int = 10) -> subprocess.Popen:
    # Use subprocess instead of multiprocessing to avoid inheriting pytest args
    api_proc = subprocess.Popen(
        [
            "python",
            "-m",
            "atroposlib.cli.run_api",
            "--host",
            "localhost",
            "--port",
            "8000",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    counter = 0
    while not check_api_running():
        time.sleep(1)
        counter += 1
        if counter > max_wait_for_api:
            api_proc.terminate()
            raise TimeoutError("API server did not start in time.")
    print("API server started for testing.")
    return api_proc
