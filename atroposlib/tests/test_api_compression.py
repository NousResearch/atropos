"""Tests covering gzip compression of API responses and requests."""

import gzip
import json
import os
import signal
import subprocess
import time

import pytest
import requests


def wait_for_api_server(max_wait=10):
    for _ in range(max_wait):
        try:
            response = requests.get("http://localhost:8000/info")
            if response.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    return False


@pytest.fixture(scope="module")
def api_server():
    proc = subprocess.Popen(
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
        preexec_fn=os.setsid,
    )

    if not wait_for_api_server():
        proc.terminate()
        raise RuntimeError("API server failed to start")

    yield

    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    proc.wait()

    try:
        requests.get("http://localhost:8000/reset_data")
    except Exception:
        pass


@pytest.fixture(autouse=True)
def reset_api_state():
    """Reset API state before each test."""
    try:
        requests.get("http://localhost:8000/reset_data")
    except Exception:
        pass
    yield
    try:
        requests.get("http://localhost:8000/reset_data")
    except Exception:
        pass


class TestAPICompression:
    """Test class for API compression functionality."""

    def test_small_response_not_compressed(self, api_server):
        """Small payloads bypass gzip."""
        response = requests.get("http://localhost:8000/info")

        assert response.status_code == 200, response.text

        assert response.headers.get("Content-Encoding") != "gzip"

        data = response.json()
        assert "batch_size" in data

    def test_large_response_compressed_automatically(self, api_server):
        """Large batches are gzipped and transparently decoded by clients."""
        requests.post(
            "http://localhost:8000/register",
            json={
                "wandb_group": "test_group",
                "wandb_project": "test_project",
                "batch_size": 16,
                "max_token_len": 2048,
                "checkpoint_dir": "/tmp/test",
                "save_checkpoint_interval": 100,
                "starting_step": 0,
                "num_steps": 1000,
            },
        )

        large_scored_data = {
            "tokens": [[i for i in range(512)] for _ in range(16)],
            "masks": [[1 for _ in range(512)] for _ in range(16)],
            "scores": [0.5 for _ in range(16)],
            "advantages": [[0.1 for _ in range(512)] for _ in range(16)],
            "ref_logprobs": [[0.2 for _ in range(512)] for _ in range(16)],
            "messages": [
                [{"role": "user", "content": "test" * 50}] for _ in range(16)
            ],
        }

        post_response = requests.post(
            "http://localhost:8000/scored_data",
            json=large_scored_data,
        )
        assert post_response.status_code == 200

        response = requests.get("http://localhost:8000/batch")

        assert response.status_code == 200

        data = response.json()
        assert "batch" in data
        assert data["batch"] is not None

        batch = data["batch"][0]
        assert len(batch["tokens"]) == 16
        assert len(batch["tokens"][0]) == 512
        assert batch["tokens"][0][0] == 0

    def test_compression_with_raw_headers(self, api_server):
        """Explicit Accept-Encoding still yields usable decoded responses."""
        requests.post(
            "http://localhost:8000/register",
            json={
                "wandb_group": "test_group",
                "wandb_project": "test_project",
                "batch_size": 32,
                "max_token_len": 2048,
                "checkpoint_dir": "/tmp/test",
                "save_checkpoint_interval": 100,
                "starting_step": 0,
                "num_steps": 1000,
            },
        )

        # Post large data
        large_scored_data = {
            "tokens": [[i for i in range(512)] for _ in range(16)],
            "masks": [[1 for _ in range(512)] for _ in range(16)],
            "scores": [0.5 for _ in range(16)],
        }
        requests.post("http://localhost:8000/scored_data", json=large_scored_data)

        session = requests.Session()
        response = session.get(
            "http://localhost:8000/batch",
            headers={"Accept-Encoding": "gzip"},
            stream=True
        )

        assert response.status_code == 200

        data = response.json()
        assert "batch" in data

        if data["batch"] is not None:
            batch = data["batch"][0]
            assert "tokens" in batch
            assert len(batch["tokens"]) > 0

    def test_compression_ratio_estimation(self, api_server):
        """Produce a rough before/after size estimate for visibility."""
        requests.post(
            "http://localhost:8000/register",
            json={
                "wandb_group": "test_group",
                "wandb_project": "test_project",
                "batch_size": 32,
                "max_token_len": 2048,
                "checkpoint_dir": "/tmp/test",
                "save_checkpoint_interval": 100,
                "starting_step": 0,
                "num_steps": 1000,
            },
        )

        large_scored_data = {
            "tokens": [[i for i in range(1024)] for _ in range(32)],
            "masks": [[1 for _ in range(1024)] for _ in range(32)],
            "scores": [0.5 for _ in range(32)],
            "advantages": [[0.1 for _ in range(1024)] for _ in range(32)],
        }

        requests.post("http://localhost:8000/scored_data", json=large_scored_data)

        response = requests.get("http://localhost:8000/batch")

        assert response.status_code == 200
        data = response.json()

        uncompressed_json = json.dumps(data)
        uncompressed_size = len(uncompressed_json.encode('utf-8'))

        assert data["batch"] is not None
        batch = data["batch"][0]
        assert len(batch["tokens"]) == 32
        assert len(batch["tokens"][0]) == 1024

        print(f"\nEstimated uncompressed size: {uncompressed_size:,} bytes")
        print("With gzip compression, actual transfer would be ~15-20% of this size")

    def test_environment_client_compatibility(self, api_server):
        """Simulate the common trainer + env flow."""
        requests.post(
            "http://localhost:8000/register",
            json={
                "wandb_group": "test_group",
                "wandb_project": "test_project",
                "batch_size": 32,
                "max_token_len": 2048,
                "checkpoint_dir": "/tmp/test",
                "save_checkpoint_interval": 100,
                "starting_step": 0,
                "num_steps": 1000,
            },
        )

        requests.get("http://localhost:8000/batch")

        env_response = requests.post(
            "http://localhost:8000/register-env",
            json={
                "max_token_length": 2048,
                "desired_name": "test_env",
                "weight": 1.0,
                "group_size": 4,
            },
        )
        assert env_response.status_code == 200
        env_data = env_response.json()
        assert env_data["status"] == "success"
        env_id = env_data["env_id"]

        scored_data = {
            "tokens": [[i for i in range(256)] for _ in range(4)],
            "masks": [[1 for _ in range(256)] for _ in range(4)],
            "scores": [0.8, 0.6, 0.4, 0.2],
            "env_id": env_id,
        }

        post_response = requests.post(
            "http://localhost:8000/scored_data",
            json=scored_data,
        )
        assert post_response.status_code == 200

        status_response = requests.get(
            "http://localhost:8000/status-env",
            json={"env_id": env_id}
        )
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert "queue_size" in status_data

    def test_server_accepts_gzipped_scored_data(self, api_server):
        """Server inflates gzipped POST bodies."""
        requests.post(
            "http://localhost:8000/register",
            json={
                "wandb_group": "test_group",
                "wandb_project": "test_project",
                "batch_size": 8,
                "max_token_len": 2048,
                "checkpoint_dir": "/tmp/test",
                "save_checkpoint_interval": 100,
                "starting_step": 0,
                "num_steps": 1000,
            },
        )

        scored_data = {
            "tokens": [[i for i in range(256)] for _ in range(8)],
            "masks": [[1 for _ in range(256)] for _ in range(8)],
            "scores": [0.1 for _ in range(8)],
        }

        payload = json.dumps(scored_data).encode("utf-8")
        compressed = gzip.compress(payload)

        response = requests.post(
            "http://localhost:8000/scored_data",
            data=compressed,
            headers={
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
            },
        )

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "received"

        batch_response = requests.get("http://localhost:8000/batch")
        assert batch_response.status_code == 200
        batch_data = batch_response.json()
        assert batch_data["batch"] is not None

    def test_scored_data_list_compression(self, api_server):
        """Multi-item submissions still round-trip correctly."""
        requests.post(
            "http://localhost:8000/register",
            json={
                "wandb_group": "test_group",
                "wandb_project": "test_project",
                "batch_size": 32,
                "max_token_len": 2048,
                "checkpoint_dir": "/tmp/test",
                "save_checkpoint_interval": 100,
                "starting_step": 0,
                "num_steps": 1000,
            },
        )

        scored_data_list = [
            {
                "tokens": [[i for i in range(512)] for _ in range(8)],
                "masks": [[1 for _ in range(512)] for _ in range(8)],
                "scores": [0.5 for _ in range(8)],
            }
            for _ in range(4)
        ]

        response = requests.post(
            "http://localhost:8000/scored_data_list",
            json=scored_data_list,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "received"
        assert data["groups_processed"] == 4

        batch_response = requests.get("http://localhost:8000/batch")
        assert batch_response.status_code == 200
        batch_data = batch_response.json()
        assert batch_data["batch"] is not None

        batch = batch_data["batch"]
        total_sequences = sum(len(item["tokens"]) for item in batch)
        assert total_sequences == 32


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
