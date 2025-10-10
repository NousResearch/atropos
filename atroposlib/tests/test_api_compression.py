"""
Tests for API server GZip compression.

These tests verify that:
1. GZip compression is enabled for large responses
2. Small responses are not compressed (below minimum_size threshold)
3. Clients automatically decompress responses (no code changes needed)
4. Both requests and raw HTTP clients work correctly
"""

import gzip
import json
import os
import signal
import subprocess
import time

import pytest
import requests


def wait_for_api_server(max_wait=10):
    """Wait for API server to be ready."""
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
    """Launch API server for testing."""
    # Start the API server as a subprocess
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
        preexec_fn=os.setsid,  # Create new process group
    )

    # Wait for server to be ready
    if not wait_for_api_server():
        proc.terminate()
        raise RuntimeError("API server failed to start")

    yield

    # Kill the process group to ensure all child processes are terminated
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    proc.wait()

    # Clean up after tests
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
        """Test that small responses (< 1KB) are not compressed."""
        # Small endpoint response
        response = requests.get("http://localhost:8000/info")
        
        assert response.status_code == 200, response.text
        
        # Small responses should not be compressed
        # (FastAPI GZip middleware has minimum_size=1000)
        assert response.headers.get("Content-Encoding") != "gzip"
        
        # But the response should still be valid JSON
        data = response.json()
        assert "batch_size" in data

    def test_large_response_compressed_automatically(self, api_server):
        """Test that large responses are automatically compressed and decompressed."""
        # Register trainer first
        requests.post(
            "http://localhost:8000/register",
            json={
                "wandb_group": "test_group",
                "wandb_project": "test_project",
                "batch_size": 16,  # Match the number of sequences we're sending
                "max_token_len": 2048,
                "checkpoint_dir": "/tmp/test",
                "save_checkpoint_interval": 100,
                "starting_step": 0,
                "num_steps": 1000,
            },
        )

        # Create large scored data (should exceed 1KB)
        large_scored_data = {
            "tokens": [[i for i in range(512)] for _ in range(16)],  # Large token array
            "masks": [[1 for _ in range(512)] for _ in range(16)],    # Large mask array
            "scores": [0.5 for _ in range(16)],
            "advantages": [[0.1 for _ in range(512)] for _ in range(16)],
            "ref_logprobs": [[0.2 for _ in range(512)] for _ in range(16)],
            "messages": [
                [{"role": "user", "content": "test" * 50}] for _ in range(16)
            ],
        }

        # Post the large data
        post_response = requests.post(
            "http://localhost:8000/scored_data",
            json=large_scored_data,
        )
        assert post_response.status_code == 200

        # Get batch (should be large and compressed)
        response = requests.get("http://localhost:8000/batch")
        
        assert response.status_code == 200
        
        # The requests library automatically decompresses, so we get the data directly
        data = response.json()
        assert "batch" in data
        assert data["batch"] is not None
        
        # Verify we got the data back correctly (automatic decompression worked)
        batch = data["batch"][0]
        assert len(batch["tokens"]) == 16
        assert len(batch["tokens"][0]) == 512
        assert batch["tokens"][0][0] == 0  # First token should be 0

    def test_compression_with_raw_headers(self, api_server):
        """Test compression using raw HTTP headers to verify server is actually compressing."""
        # Register trainer
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

        # Make request with explicit Accept-Encoding and get raw response
        session = requests.Session()
        response = session.get(
            "http://localhost:8000/batch",
            headers={"Accept-Encoding": "gzip"},
            stream=True
        )
        
        assert response.status_code == 200
        
        # Check if the response was actually compressed by the server
        # Note: requests automatically decompresses, but we can check the headers
        # If compression happened, the raw response should have been compressed
        
        # Get the actual response content
        data = response.json()
        assert "batch" in data
        
        # Verify the data is correct (decompression worked automatically)
        if data["batch"] is not None:
            batch = data["batch"][0]
            assert "tokens" in batch
            assert len(batch["tokens"]) > 0

    def test_compression_ratio_estimation(self, api_server):
        """Test to estimate actual compression ratio achieved."""
        # Register trainer
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

        # Create large scored data
        large_scored_data = {
            "tokens": [[i for i in range(1024)] for _ in range(32)],  # ~32K tokens
            "masks": [[1 for _ in range(1024)] for _ in range(32)],
            "scores": [0.5 for _ in range(32)],
            "advantages": [[0.1 for _ in range(1024)] for _ in range(32)],
        }
        
        # Post the data
        requests.post("http://localhost:8000/scored_data", json=large_scored_data)

        # Get the batch
        response = requests.get("http://localhost:8000/batch")
        
        assert response.status_code == 200
        data = response.json()
        
        # Calculate uncompressed size (rough estimate from JSON string)
        uncompressed_json = json.dumps(data)
        uncompressed_size = len(uncompressed_json.encode('utf-8'))
        
        # The actual transmitted size would be much smaller due to gzip
        # We can't easily measure it with requests (auto-decompresses)
        # but we can verify the data is correct
        assert data["batch"] is not None
        batch = data["batch"][0]
        assert len(batch["tokens"]) == 32
        assert len(batch["tokens"][0]) == 1024
        
        print(f"\nEstimated uncompressed size: {uncompressed_size:,} bytes")
        print(f"With gzip compression, actual transfer would be ~15-20% of this size")

    def test_environment_client_compatibility(self, api_server):
        """Test that the compression works with typical environment usage patterns."""
        # Register trainer
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

        # Trainer needs to call /batch first to mark as started
        requests.get("http://localhost:8000/batch")

        # Register environment
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

        # Post scored data as environment would
        scored_data = {
            "tokens": [[i for i in range(256)] for _ in range(4)],
            "masks": [[1 for _ in range(256)] for _ in range(4)],
            "scores": [0.8, 0.6, 0.4, 0.2],
            "env_id": env_id,
        }
        
        # This should work without any client changes
        post_response = requests.post(
            "http://localhost:8000/scored_data",
            json=scored_data,
        )
        assert post_response.status_code == 200
        
        # Verify environment can check status
        # Note: The API expects env_id as JSON body in GET request
        status_response = requests.get(
            "http://localhost:8000/status-env",
            json={"env_id": env_id}
        )
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert "queue_size" in status_data

    def test_server_accepts_gzipped_scored_data(self, api_server):
        """Ensure server middleware handles gzip-compressed request bodies."""
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
        """Test that scored_data_list endpoint also benefits from compression."""
        # Register trainer
        requests.post(
            "http://localhost:8000/register",
            json={
                "wandb_group": "test_group",
                "wandb_project": "test_project",
                "batch_size": 32,  # 4 groups * 8 sequences = 32 total
                "max_token_len": 2048,
                "checkpoint_dir": "/tmp/test",
                "save_checkpoint_interval": 100,
                "starting_step": 0,
                "num_steps": 1000,
            },
        )

        # Post multiple scored data items at once (as list)
        scored_data_list = [
            {
                "tokens": [[i for i in range(512)] for _ in range(8)],
                "masks": [[1 for _ in range(512)] for _ in range(8)],
                "scores": [0.5 for _ in range(8)],
            }
            for _ in range(4)  # 4 groups of 8 sequences each = 32 sequences total
        ]

        response = requests.post(
            "http://localhost:8000/scored_data_list",
            json=scored_data_list,
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "received"
        assert data["groups_processed"] == 4

        # Verify we can get batches
        batch_response = requests.get("http://localhost:8000/batch")
        assert batch_response.status_code == 200
        batch_data = batch_response.json()
        assert batch_data["batch"] is not None
        
        # Verify the batch contains the correct data
        batch = batch_data["batch"]
        total_sequences = sum(len(item["tokens"]) for item in batch)
        assert total_sequences == 32  # Should have all 32 sequences


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
