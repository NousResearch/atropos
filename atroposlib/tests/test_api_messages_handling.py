"""
Tests for API server message handling, particularly for SFT (Supervised Fine-Tuning) scenarios.
"""

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
    requests.get("http://localhost:8000/reset_data")
    yield
    requests.get("http://localhost:8000/reset_data")


class TestAPIMessagesHandling:
    """Test class for API messages handling."""

    def test_register_trainer(self, api_server):
        """Test trainer registration."""
        response = requests.post(
            "http://localhost:8000/register",
            json={
                "wandb_group": "test_group",
                "wandb_project": "test_project",
                "batch_size": 32,
                "max_token_len": 2048,
                "checkpoint_dir": "/tmp/test_checkpoint",
                "save_checkpoint_interval": 100,
                "starting_step": 0,
                "num_steps": 1000,
            },
        )
        if response.status_code != 200:
            print(f"Error response: {response.text}")
        assert response.status_code == 200
        data = response.json()
        assert "uuid" in data
        assert isinstance(data["uuid"], int)

    def test_scored_data_with_messages(self, api_server):
        """Test posting scored data with messages field."""
        # First register
        register_response = requests.post(
            "http://localhost:8000/register",
            json={
                "wandb_group": "test",
                "wandb_project": "test",
                "batch_size": 2,
                "max_token_len": 512,
                "checkpoint_dir": "/tmp",
                "save_checkpoint_interval": 10,
                "starting_step": 0,
                "num_steps": 100,
            },
        )
        assert register_response.status_code == 200

        # Test with messages in OpenAI format
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
                "reward": None,
            },
            {
                "role": "user",
                "content": "What is the capital of France?",
                "reward": None,
            },
            {
                "role": "assistant",
                "content": "The capital of France is Paris.",
                "reward": None,
            },
        ]

        response = requests.post(
            "http://localhost:8000/scored_data",
            json={
                "tokens": [[1, 2, 3, 4, 5]],
                "masks": [[1, 1, 1, 1, 1]],
                "scores": [1.0],
                "messages": [messages],
                "advantages": [[0.5, 0.5, 0.5, 0.5, 0.5]],
                "ref_logprobs": [[-0.1, -0.2, -0.3, -0.4, -0.5]],
            },
        )
        if response.status_code != 200:
            print(f"Error response: {response.text}")
        assert response.status_code == 200
        assert response.json()["status"] == "received"

    def test_scored_data_list_with_messages(self, api_server):
        """Test posting a list of scored data with messages."""
        # First register
        register_response = requests.post(
            "http://localhost:8000/register",
            json={
                "wandb_group": "test",
                "wandb_project": "test",
                "batch_size": 4,
                "max_token_len": 512,
                "checkpoint_dir": "/tmp",
                "save_checkpoint_interval": 10,
                "starting_step": 0,
                "num_steps": 100,
            },
        )
        assert register_response.status_code == 200

        # Create multiple scored data items with messages
        scored_data_list = []
        for i in range(3):
            messages = [
                {"role": "user", "content": f"Question {i}", "reward": None},
                {"role": "assistant", "content": f"Answer {i}", "reward": None},
            ]
            scored_data_list.append(
                {
                    "tokens": [[i + 1, i + 2, i + 3]],
                    "masks": [[1, 1, 1]],
                    "scores": [float(i)],
                    "messages": [messages],
                }
            )

        response = requests.post(
            "http://localhost:8000/scored_data_list", json=scored_data_list
        )
        if response.status_code != 200:
            print(f"Error response: {response.text}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "received"
        assert data["groups_processed"] == 3

    def test_sft_style_messages(self, api_server):
        """Test SFT-style message handling with group overrides."""
        # Register first
        register_response = requests.post(
            "http://localhost:8000/register",
            json={
                "wandb_group": "sft_test",
                "wandb_project": "sft_test",
                "batch_size": 1,
                "max_token_len": 1024,
                "checkpoint_dir": "/tmp",
                "save_checkpoint_interval": 10,
                "starting_step": 0,
                "num_steps": 100,
            },
        )
        assert register_response.status_code == 200

        # SFT-style data with ShareGPT format messages
        sharegpt_messages = [
            {"role": "user", "content": "Explain quantum computing", "reward": None},
            {
                "role": "assistant",
                "content": "Quantum computing is a type of computing...",
                "reward": None,
            },
        ]

        response = requests.post(
            "http://localhost:8000/scored_data",
            json={
                "tokens": [[100, 101, 102, 103, 104, 105]],
                "masks": [[-100, -100, 102, 103, 104, 105]],  # Masked prefix
                "scores": [1.0],
                "messages": [sharegpt_messages],
                "advantages": [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
                "group_overrides": {"sft": True},
                "overrides": [{"sft": True}],
            },
        )
        if response.status_code != 200:
            print(f"Error response: {response.text}")
        assert response.status_code == 200
        assert response.json()["status"] == "received"

    def test_multimodal_messages_with_images(self, api_server):
        """Test messages with image data."""
        # Register first
        register_response = requests.post(
            "http://localhost:8000/register",
            json={
                "wandb_group": "multimodal_test",
                "wandb_project": "multimodal_test",
                "batch_size": 1,
                "max_token_len": 2048,
                "checkpoint_dir": "/tmp",
                "save_checkpoint_interval": 10,
                "starting_step": 0,
                "num_steps": 100,
            },
        )
        assert register_response.status_code == 200

        # Multimodal message with image reference
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,..."},
                    },
                ],
                "reward": None,
            },
            {
                "role": "assistant",
                "content": "I can see a cat in the image.",
                "reward": None,
            },
        ]

        response = requests.post(
            "http://localhost:8000/scored_data",
            json={
                "tokens": [[200, 201, 202, 203]],
                "masks": [[1, 1, 1, 1]],
                "scores": [0.9],
                "messages": [messages],
                "images": ["base64_encoded_image_data"],
            },
        )
        if response.status_code != 200:
            print(f"Error response: {response.text}")
        assert response.status_code == 200
        assert response.json()["status"] == "received"

    def test_batch_retrieval_with_messages(self, api_server):
        """Test retrieving batches that contain messages."""
        # Register with batch size 2
        register_response = requests.post(
            "http://localhost:8000/register",
            json={
                "wandb_group": "batch_test",
                "wandb_project": "batch_test",
                "batch_size": 2,
                "max_token_len": 512,
                "checkpoint_dir": "/tmp",
                "save_checkpoint_interval": 10,
                "starting_step": 0,
                "num_steps": 100,
            },
        )
        assert register_response.status_code == 200

        # Post two items to make a full batch
        for i in range(2):
            messages = [
                {"role": "user", "content": f"Test message {i}", "reward": None},
                {"role": "assistant", "content": f"Response {i}", "reward": None},
            ]
            response = requests.post(
                "http://localhost:8000/scored_data",
                json={
                    "tokens": [[i * 10 + j for j in range(5)]],
                    "masks": [[1] * 5],
                    "scores": [float(i)],
                    "messages": [messages],
                },
            )
            assert response.status_code == 200

        # Retrieve the batch
        batch_response = requests.get("http://localhost:8000/batch")
        assert batch_response.status_code == 200
        batch_data = batch_response.json()
        assert batch_data["batch"] is not None
        assert len(batch_data["batch"]) == 2

        # Verify messages are preserved in the batch
        for i, item in enumerate(batch_data["batch"]):
            assert "messages" in item
            assert item["messages"] is not None
            assert len(item["messages"]) == 1
            assert len(item["messages"][0]) == 2
            assert item["messages"][0][0]["role"] == "user"
            assert item["messages"][0][1]["role"] == "assistant"

    def test_latest_example_with_messages(self, api_server):
        """Test that latest example endpoint includes messages."""
        # Register first
        register_response = requests.post(
            "http://localhost:8000/register",
            json={
                "wandb_group": "latest_test",
                "wandb_project": "latest_test",
                "batch_size": 1,
                "max_token_len": 512,
                "checkpoint_dir": "/tmp",
                "save_checkpoint_interval": 10,
                "starting_step": 0,
                "num_steps": 100,
            },
        )
        assert register_response.status_code == 200

        # Post data with messages
        messages = [
            {
                "role": "system",
                "content": "You are a coding assistant.",
                "reward": None,
            },
            {"role": "user", "content": "Write a Python hello world.", "reward": None},
            {"role": "assistant", "content": "print('Hello, World!')", "reward": None},
        ]

        response = requests.post(
            "http://localhost:8000/scored_data",
            json={
                "tokens": [[42, 43, 44]],
                "masks": [[1, 1, 1]],
                "scores": [0.95],
                "messages": [messages],
            },
        )
        if response.status_code != 200:
            print(f"Error response: {response.text}")
        assert response.status_code == 200

        # Get latest example
        latest_response = requests.get("http://localhost:8000/latest_example")
        assert latest_response.status_code == 200
        latest_data = latest_response.json()

        assert "messages" in latest_data
        assert latest_data["messages"] == [messages]
        assert len(latest_data["messages"][0]) == 3

    def test_empty_messages_handling(self, api_server):
        """Test handling of empty or None messages."""
        # Register first
        register_response = requests.post(
            "http://localhost:8000/register",
            json={
                "wandb_group": "empty_test",
                "wandb_project": "empty_test",
                "batch_size": 1,
                "max_token_len": 512,
                "checkpoint_dir": "/tmp",
                "save_checkpoint_interval": 10,
                "starting_step": 0,
                "num_steps": 100,
            },
        )
        assert register_response.status_code == 200

        # Test with None messages (optional field)
        response = requests.post(
            "http://localhost:8000/scored_data",
            json={
                "tokens": [[1, 2, 3]],
                "masks": [[1, 1, 1]],
                "scores": [1.0],
                # messages field omitted
            },
        )
        if response.status_code != 200:
            print(f"Error response: {response.text}")
        assert response.status_code == 200
        assert response.json()["status"] == "received"

        # Test with empty messages list
        response = requests.post(
            "http://localhost:8000/scored_data",
            json={
                "tokens": [[4, 5, 6]],
                "masks": [[1, 1, 1]],
                "scores": [1.0],
                "messages": [],
            },
        )
        if response.status_code != 200:
            print(f"Error response: {response.text}")
        assert response.status_code == 200
        assert response.json()["status"] == "received"

    def test_complex_message_structures(self, api_server):
        """Test handling of complex message structures with tool calls."""
        # Register first
        register_response = requests.post(
            "http://localhost:8000/register",
            json={
                "wandb_group": "complex_test",
                "wandb_project": "complex_test",
                "batch_size": 1,
                "max_token_len": 2048,
                "checkpoint_dir": "/tmp",
                "save_checkpoint_interval": 10,
                "starting_step": 0,
                "num_steps": 100,
            },
        )
        assert register_response.status_code == 200

        # Complex messages with tool role
        messages = [
            {
                "role": "system",
                "content": "You have access to calculation tools.",
                "reward": None,
            },
            {"role": "user", "content": "What is 15 * 23?", "reward": None},
            {
                "role": "assistant",
                "content": "I'll calculate that for you.",
                "reward": None,
            },
            {"role": "tool", "content": "Result: 345", "reward": None},
            {"role": "assistant", "content": "15 * 23 = 345", "reward": None},
        ]

        response = requests.post(
            "http://localhost:8000/scored_data",
            json={
                "tokens": [[300, 301, 302, 303, 304]],
                "masks": [[1, 1, 1, 1, 1]],
                "scores": [0.85],
                "messages": [messages],
                "advantages": [[0.1, 0.2, 0.3, 0.4, 0.5]],
            },
        )
        if response.status_code != 200:
            print(f"Error response: {response.text}")
        assert response.status_code == 200
        assert response.json()["status"] == "received"

    def test_message_reward_field(self, api_server):
        """Test messages with reward field as defined in Message TypedDict."""
        # Register first
        register_response = requests.post(
            "http://localhost:8000/register",
            json={
                "wandb_group": "reward_test",
                "wandb_project": "reward_test",
                "batch_size": 1,
                "max_token_len": 512,
                "checkpoint_dir": "/tmp",
                "save_checkpoint_interval": 10,
                "starting_step": 0,
                "num_steps": 100,
            },
        )
        assert register_response.status_code == 200

        # Messages with reward field
        messages = [
            {"role": "user", "content": "Solve: 2+2", "reward": None},
            {"role": "assistant", "content": "2+2 = 4", "reward": 1.0},
        ]

        response = requests.post(
            "http://localhost:8000/scored_data",
            json={
                "tokens": [[400, 401, 402]],
                "masks": [[1, 1, 1]],
                "scores": [1.0],
                "messages": [messages],
            },
        )
        if response.status_code != 200:
            print(f"Error response: {response.text}")
        assert response.status_code == 200
        assert response.json()["status"] == "received"


class TestSFTIntegration:
    """Test SFT-specific integration scenarios."""

    def test_sft_completion_format(self, api_server):
        """Test SFT with completion format."""
        # Register
        register_response = requests.post(
            "http://localhost:8000/register",
            json={
                "wandb_group": "sft_completion",
                "wandb_project": "sft_completion",
                "batch_size": 1,
                "max_token_len": 512,
                "checkpoint_dir": "/tmp",
                "save_checkpoint_interval": 10,
                "starting_step": 0,
                "num_steps": 100,
            },
        )
        assert register_response.status_code == 200

        # Simple completion text - messages field is omitted for completion format
        response = requests.post(
            "http://localhost:8000/scored_data",
            json={
                "tokens": [[500, 501, 502, 503]],
                "masks": [[500, 501, 502, 503]],
                "scores": [1],
                "advantages": [[1, 1, 1, 1]],
                # messages field omitted - completion format doesn't use Message objects
                "group_overrides": {"sft": True},
                "overrides": [{"sft": True}],
            },
        )
        if response.status_code != 200:
            print(f"Error response: {response.text}")
        assert response.status_code == 200

    def test_sft_prefixed_completion(self, api_server):
        """Test SFT with prefixed completion format."""
        # Register
        register_response = requests.post(
            "http://localhost:8000/register",
            json={
                "wandb_group": "sft_prefixed",
                "wandb_project": "sft_prefixed",
                "batch_size": 1,
                "max_token_len": 512,
                "checkpoint_dir": "/tmp",
                "save_checkpoint_interval": 10,
                "starting_step": 0,
                "num_steps": 100,
            },
        )
        assert register_response.status_code == 200

        # Prefixed completion with masked prefix
        response = requests.post(
            "http://localhost:8000/scored_data",
            json={
                "tokens": [[600, 601, 602, 603, 604, 605]],
                "masks": [[-100, -100, -100, 603, 604, 605]],  # First 3 tokens masked
                "scores": [1],
                "advantages": [[1, 1, 1, 1, 1, 1]],
                # messages field omitted - prefixed completion format doesn't use Message objects
                "group_overrides": {"sft": True},
                "overrides": [{"sft": True}],
            },
        )
        if response.status_code != 200:
            print(f"Error response: {response.text}")
        assert response.status_code == 200

    def test_sft_batch_processing(self, api_server):
        """Test batch processing for SFT data."""
        # Register with larger batch size
        register_response = requests.post(
            "http://localhost:8000/register",
            json={
                "wandb_group": "sft_batch",
                "wandb_project": "sft_batch",
                "batch_size": 4,
                "max_token_len": 512,
                "checkpoint_dir": "/tmp",
                "save_checkpoint_interval": 10,
                "starting_step": 0,
                "num_steps": 100,
            },
        )
        assert register_response.status_code == 200

        # Send multiple SFT items
        for i in range(4):
            messages = [
                {"role": "user", "content": f"Question {i}", "reward": None},
                {"role": "assistant", "content": f"Answer {i}", "reward": None},
            ]
            response = requests.post(
                "http://localhost:8000/scored_data",
                json={
                    "tokens": [[700 + i, 701 + i, 702 + i]],
                    "masks": [[-100, 701 + i, 702 + i]],
                    "scores": [1],
                    "advantages": [[1, 1, 1, 1]],
                    "messages": [messages],
                    "group_overrides": {"sft": True},
                    "overrides": [{"sft": True}],
                },
            )
            assert response.status_code == 200

        # Verify queue size
        status_response = requests.get("http://localhost:8000/status")
        assert status_response.status_code == 200
        assert status_response.json()["queue_size"] == 4

        # Get batch
        batch_response = requests.get("http://localhost:8000/batch")
        assert batch_response.status_code == 200
        batch = batch_response.json()["batch"]
        assert len(batch) == 4

        # Verify all items have SFT overrides
        for item in batch:
            assert item.get("group_overrides", {}).get("sft") is True
            assert item.get("overrides", [{}])[0].get("sft") is True


class TestMessageRewardHandling:
    """Test different scenarios with reward field in messages."""

    def test_messages_without_reward_field(self, api_server):
        """Test messages without the reward field - should be accepted."""
        # Register first
        register_response = requests.post(
            "http://localhost:8000/register",
            json={
                "wandb_group": "no_reward_test",
                "wandb_project": "no_reward_test",
                "batch_size": 1,
                "max_token_len": 512,
                "checkpoint_dir": "/tmp",
                "save_checkpoint_interval": 10,
                "starting_step": 0,
                "num_steps": 100,
            },
        )
        assert register_response.status_code == 200

        # Messages without reward field should be accepted
        response = requests.post(
            "http://localhost:8000/scored_data",
            json={
                "tokens": [[800, 801, 802]],
                "masks": [[1, 1, 1]],
                "scores": [0.5],
                "messages": [
                    [
                        {
                            "role": "user",
                            "content": "Write a poem about the ocean",
                            # No reward field - this should be OK
                        },
                        {
                            "role": "assistant",
                            "content": "Waves crash upon the shore,\nEndless blue forevermore.",
                            # No reward field
                        },
                    ]
                ],
                "advantages": [[0.5, 0.3, 0.2]],
            },
        )
        # This should now succeed
        print(f"Status code without reward field: {response.status_code}")
        if response.status_code != 200:
            error_data = response.json()
            print(f"Error response: {error_data}")
        assert (
            response.status_code == 200
        ), "Messages without reward field should be accepted"

    def test_mixed_reward_presence(self, api_server):
        """Test messages with inconsistent reward field presence - should be accepted."""
        # Register first
        register_response = requests.post(
            "http://localhost:8000/register",
            json={
                "wandb_group": "mixed_reward_test",
                "wandb_project": "mixed_reward_test",
                "batch_size": 1,
                "max_token_len": 512,
                "checkpoint_dir": "/tmp",
                "save_checkpoint_interval": 10,
                "starting_step": 0,
                "num_steps": 100,
            },
        )
        assert register_response.status_code == 200

        # Mixed messages - some with reward, some without - should be OK
        response = requests.post(
            "http://localhost:8000/scored_data",
            json={
                "tokens": [[810, 811, 812]],
                "masks": [[1, 1, 1]],
                "scores": [0.7],
                "messages": [
                    [
                        {"role": "user", "content": "Hello"},  # No reward field
                        {
                            "role": "assistant",
                            "content": "Hi!",
                            "reward": 0.9,
                        },  # Has reward
                    ]
                ],
                "advantages": [[0.5, 0.3, 0.2]],
            },
        )

        print(f"Mixed reward presence status: {response.status_code}")
        if response.status_code != 200:
            print(f"Error: {response.json()}")
        assert response.status_code == 200, "Mixed reward presence should be accepted"

    def test_reward_none_vs_missing(self, api_server):
        """Test explicit None reward vs missing field."""
        # Register first
        register_response = requests.post(
            "http://localhost:8000/register",
            json={
                "wandb_group": "reward_none_test",
                "wandb_project": "reward_none_test",
                "batch_size": 1,
                "max_token_len": 512,
                "checkpoint_dir": "/tmp",
                "save_checkpoint_interval": 10,
                "starting_step": 0,
                "num_steps": 100,
            },
        )
        assert register_response.status_code == 200

        # Test 1: All messages have reward=None (should work)
        response1 = requests.post(
            "http://localhost:8000/scored_data",
            json={
                "tokens": [[820, 821, 822]],
                "masks": [[1, 1, 1]],
                "scores": [0.8],
                "messages": [
                    [
                        {"role": "user", "content": "Test with None", "reward": None},
                        {"role": "assistant", "content": "Response", "reward": None},
                    ]
                ],
                "advantages": [[0.5, 0.3, 0.2]],
            },
        )
        assert (
            response1.status_code == 200
        ), f"None reward should work: {response1.json()}"

        # Test 2: Messages without reward field should also work
        messages_missing_reward = []
        msg1 = {"role": "user", "content": "Test without reward"}
        msg2 = {"role": "assistant", "content": "Response without reward"}
        messages_missing_reward.append(msg1)
        messages_missing_reward.append(msg2)

        response2 = requests.post(
            "http://localhost:8000/scored_data",
            json={
                "tokens": [[830, 831, 832]],
                "masks": [[1, 1, 1]],
                "scores": [0.6],
                "messages": [messages_missing_reward],
                "advantages": [[0.5, 0.3, 0.2]],
            },
        )

        print(f"Missing reward field status: {response2.status_code}")
        if response2.status_code != 200:
            print(f"Error details: {response2.json()}")
        assert (
            response2.status_code == 200
        ), "Messages without reward field should be accepted"

    def test_extra_fields_in_messages(self, api_server):
        """Test messages with extra fields not defined in Message TypedDict."""
        # Register first
        register_response = requests.post(
            "http://localhost:8000/register",
            json={
                "wandb_group": "extra_fields_test",
                "wandb_project": "extra_fields_test",
                "batch_size": 1,
                "max_token_len": 512,
                "checkpoint_dir": "/tmp",
                "save_checkpoint_interval": 10,
                "starting_step": 0,
                "num_steps": 100,
            },
        )
        assert register_response.status_code == 200

        # Messages with extra fields that aren't in the TypedDict
        response = requests.post(
            "http://localhost:8000/scored_data",
            json={
                "tokens": [[840, 841, 842]],
                "masks": [[1, 1, 1]],
                "scores": [0.7],
                "messages": [
                    [
                        {
                            "role": "user",
                            "content": "Hello AI",
                            "reward": None,
                            "definitely_not_in_typeddict_kwarg": "surprise!",
                            "another_extra_field": 42,
                            "yet_another_field": {"nested": "data"},
                        },
                        {
                            "role": "assistant",
                            "content": "Hello human!",
                            "reward": 0.5,
                            "definitely_not_in_typeddict_kwarg": "another surprise!",
                            "random_metadata": ["list", "of", "things"],
                        },
                    ]
                ],
                "advantages": [[0.5, 0.3, 0.2]],
            },
        )

        print(f"Extra fields status: {response.status_code}")
        if response.status_code == 200:
            print("API accepts extra fields in messages!")
        else:
            error_data = response.json()
            print(f"Error with extra fields: {error_data}")
            # Check if it's complaining about the extra fields
            assert (
                "definitely_not_in_typeddict_kwarg" in str(error_data)
                or response.status_code == 422
            )

    def test_extra_fields_without_reward(self, api_server):
        """Test messages with extra fields but missing the reward field - should be accepted."""
        # Register first
        register_response = requests.post(
            "http://localhost:8000/register",
            json={
                "wandb_group": "extra_no_reward_test",
                "wandb_project": "extra_no_reward_test",
                "batch_size": 1,
                "max_token_len": 512,
                "checkpoint_dir": "/tmp",
                "save_checkpoint_interval": 10,
                "starting_step": 0,
                "num_steps": 100,
            },
        )
        assert register_response.status_code == 200

        # Messages with extra fields but NO reward field - should be OK
        response = requests.post(
            "http://localhost:8000/scored_data",
            json={
                "tokens": [[850, 851, 852]],
                "masks": [[1, 1, 1]],
                "scores": [0.8],
                "messages": [
                    [
                        {
                            "role": "user",
                            "content": "Test message",
                            # NO reward field!
                            "definitely_not_in_typeddict_kwarg": "I'm here but reward isn't!",
                            "extra_metadata": {"key": "value"},
                            "priority": 10,
                        },
                        {
                            "role": "assistant",
                            "content": "Response message",
                            # NO reward field!
                            "definitely_not_in_typeddict_kwarg": "Still no reward field",
                            "completion_tokens": 42,
                            "model": "gpt-4",
                        },
                    ]
                ],
                "advantages": [[0.5, 0.3, 0.2]],
            },
        )

        print(f"Extra fields without reward status: {response.status_code}")
        if response.status_code != 200:
            error_data = response.json()
            print(f"Error: {error_data}")
        assert (
            response.status_code == 200
        ), "Messages with extra fields but no reward should be accepted"


class TestWeirdMessageFormats:
    """Test edge cases with unusual data structures that users might try."""

    def test_messages_as_tuples(self, api_server):
        """Test sending messages as tuples instead of lists."""
        # Register first
        register_response = requests.post(
            "http://localhost:8000/register",
            json={
                "wandb_group": "tuple_test",
                "wandb_project": "tuple_test",
                "batch_size": 1,
                "max_token_len": 512,
                "checkpoint_dir": "/tmp",
                "save_checkpoint_interval": 10,
                "starting_step": 0,
                "num_steps": 100,
            },
        )
        assert register_response.status_code == 200

        # Try messages as tuple of tuples (JSON will convert to lists)
        response = requests.post(
            "http://localhost:8000/scored_data",
            json={
                "tokens": [[900, 901, 902]],
                "masks": [[1, 1, 1]],
                "scores": [0.7],
                # This will be converted to list by JSON serialization
                "messages": [
                    (
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi!"},
                    )
                ],
                "advantages": [[0.5, 0.3, 0.2]],
            },
        )

        print(f"Tuple messages status: {response.status_code}")
        if response.status_code != 200:
            print(f"Error: {response.json()}")
        # Should work since JSON converts tuples to lists
        assert response.status_code == 200

    def test_messages_with_nested_weird_types(self, api_server):
        """Test messages with nested unusual types."""
        # Register first
        register_response = requests.post(
            "http://localhost:8000/register",
            json={
                "wandb_group": "nested_weird_test",
                "wandb_project": "nested_weird_test",
                "batch_size": 1,
                "max_token_len": 512,
                "checkpoint_dir": "/tmp",
                "save_checkpoint_interval": 10,
                "starting_step": 0,
                "num_steps": 100,
            },
        )
        assert register_response.status_code == 200

        # Messages with weird nested content
        response = requests.post(
            "http://localhost:8000/scored_data",
            json={
                "tokens": [[910, 911, 912]],
                "masks": [[1, 1, 1]],
                "scores": [0.8],
                "messages": [
                    [
                        {
                            "role": "user",
                            "content": {
                                "text": "Complex content",
                                "metadata": {
                                    "nested": {"deeply": ["list", "of", "things"]}
                                },
                                "numbers": (1, 2, 3),  # Tuple becomes list in JSON
                            },
                        },
                        {
                            "role": "assistant",
                            "content": ["This", "is", "a", "list", "content"],
                            "reward": 0.5,
                        },
                    ]
                ],
                "advantages": [[0.5, 0.3, 0.2]],
            },
        )

        print(f"Nested weird types status: {response.status_code}")
        if response.status_code != 200:
            print(f"Error: {response.json()}")
        assert response.status_code == 200

    def test_messages_with_numeric_strings_as_content(self, api_server):
        """Test messages with numeric strings and edge case content."""
        # Register first
        register_response = requests.post(
            "http://localhost:8000/register",
            json={
                "wandb_group": "numeric_content_test",
                "wandb_project": "numeric_content_test",
                "batch_size": 1,
                "max_token_len": 512,
                "checkpoint_dir": "/tmp",
                "save_checkpoint_interval": 10,
                "starting_step": 0,
                "num_steps": 100,
            },
        )
        assert register_response.status_code == 200

        # Messages with numeric and edge case content
        response = requests.post(
            "http://localhost:8000/scored_data",
            json={
                "tokens": [[920, 921, 922]],
                "masks": [[1, 1, 1]],
                "scores": [0.6],
                "messages": [
                    [
                        {"role": "user", "content": "12345"},  # Numeric string
                        {"role": "assistant", "content": ""},  # Empty string
                        {"role": "user", "content": " "},  # Whitespace only
                        {"role": "assistant", "content": "0"},  # Zero as string
                    ]
                ],
                "advantages": [[0.5, 0.3, 0.2, 0.1]],
            },
        )

        print(f"Numeric content status: {response.status_code}")
        assert response.status_code == 200

    def test_messages_with_boolean_and_null_values(self, api_server):
        """Test messages with boolean and null values in various fields."""
        # Register first
        register_response = requests.post(
            "http://localhost:8000/register",
            json={
                "wandb_group": "bool_null_test",
                "wandb_project": "bool_null_test",
                "batch_size": 1,
                "max_token_len": 512,
                "checkpoint_dir": "/tmp",
                "save_checkpoint_interval": 10,
                "starting_step": 0,
                "num_steps": 100,
            },
        )
        assert register_response.status_code == 200

        # Try various edge cases
        response = requests.post(
            "http://localhost:8000/scored_data",
            json={
                "tokens": [[930, 931, 932]],
                "masks": [[1, 1, 1]],
                "scores": [0.5],
                "messages": [
                    [
                        {
                            "role": "user",
                            "content": "Normal message",
                            "reward": None,  # Explicit None
                            "extra_bool": True,
                            "extra_null": None,
                        },
                        {
                            "role": "assistant",
                            "content": "Response",
                            "reward": 0.0,  # Zero reward
                            "metadata": {"flag": False, "value": None},
                        },
                    ]
                ],
                "advantages": [[0.5, 0.3, 0.2]],
            },
        )

        print(f"Bool/null values status: {response.status_code}")
        assert response.status_code == 200

    def test_messages_with_very_large_content(self, api_server):
        """Test messages with very large content strings."""
        # Register first
        register_response = requests.post(
            "http://localhost:8000/register",
            json={
                "wandb_group": "large_content_test",
                "wandb_project": "large_content_test",
                "batch_size": 1,
                "max_token_len": 10000,
                "checkpoint_dir": "/tmp",
                "save_checkpoint_interval": 10,
                "starting_step": 0,
                "num_steps": 100,
            },
        )
        assert register_response.status_code == 200

        # Large content message
        large_content = "A" * 10000  # 10k character string
        response = requests.post(
            "http://localhost:8000/scored_data",
            json={
                "tokens": [[940, 941, 942]],
                "masks": [[1, 1, 1]],
                "scores": [0.7],
                "messages": [
                    [
                        {"role": "user", "content": large_content},
                        {"role": "assistant", "content": "Short response"},
                    ]
                ],
                "advantages": [[0.5, 0.3, 0.2]],
            },
        )

        print(f"Large content status: {response.status_code}")
        assert response.status_code == 200

    def test_messages_with_unicode_and_special_chars(self, api_server):
        """Test messages with unicode, emojis, and special characters."""
        # Register first
        register_response = requests.post(
            "http://localhost:8000/register",
            json={
                "wandb_group": "unicode_test",
                "wandb_project": "unicode_test",
                "batch_size": 1,
                "max_token_len": 512,
                "checkpoint_dir": "/tmp",
                "save_checkpoint_interval": 10,
                "starting_step": 0,
                "num_steps": 100,
            },
        )
        assert register_response.status_code == 200

        # Unicode and special character messages
        response = requests.post(
            "http://localhost:8000/scored_data",
            json={
                "tokens": [[950, 951, 952]],
                "masks": [[1, 1, 1]],
                "scores": [0.8],
                "messages": [
                    [
                        {
                            "role": "user",
                            "content": "Hello 👋 世界 🌍 مرحبا 🎉 Здравствуй",
                        },
                        {
                            "role": "assistant",
                            "content": "Special chars: \n\t\r \" ' \\ / 🚀",
                            "reward": 0.9,
                        },
                    ]
                ],
                "advantages": [[0.5, 0.3, 0.2]],
            },
        )

        print(f"Unicode/special chars status: {response.status_code}")
        assert response.status_code == 200

    def test_messages_with_custom_roles(self, api_server):
        """Test messages with custom role values like 'dog'."""
        # Register first
        register_response = requests.post(
            "http://localhost:8000/register",
            json={
                "wandb_group": "custom_role_test",
                "wandb_project": "custom_role_test",
                "batch_size": 1,
                "max_token_len": 512,
                "checkpoint_dir": "/tmp",
                "save_checkpoint_interval": 10,
                "starting_step": 0,
                "num_steps": 100,
            },
        )
        assert register_response.status_code == 200

        # Try custom/weird roles
        response = requests.post(
            "http://localhost:8000/scored_data",
            json={
                "tokens": [[960, 961, 962, 963, 964]],
                "masks": [[1, 1, 1, 1, 1]],
                "scores": [0.5],
                "messages": [
                    [
                        {"role": "dog", "content": "Woof woof!"},  # Custom role
                        {"role": "cat", "content": "Meow"},  # Another custom role
                        {"role": "narrator", "content": "The animals were talking"},
                        {"role": "USER", "content": "Wrong case but should work"},
                        {"role": "robot", "content": "Beep boop"},
                    ]
                ],
                "advantages": [[0.5, 0.4, 0.3, 0.2, 0.1]],
            },
        )

        print(f"Custom roles (dog, cat, etc) status: {response.status_code}")
        if response.status_code != 200:
            print(f"Error: {response.json()}")
        # Should accept custom roles
        assert response.status_code == 200

    def test_messages_missing_required_fields(self, api_server):
        """Test messages missing role or content fields."""
        # Register first
        register_response = requests.post(
            "http://localhost:8000/register",
            json={
                "wandb_group": "missing_fields_test",
                "wandb_project": "missing_fields_test",
                "batch_size": 1,
                "max_token_len": 512,
                "checkpoint_dir": "/tmp",
                "save_checkpoint_interval": 10,
                "starting_step": 0,
                "num_steps": 100,
            },
        )
        assert register_response.status_code == 200

        # Try missing role
        response1 = requests.post(
            "http://localhost:8000/scored_data",
            json={
                "tokens": [[970, 971, 972]],
                "masks": [[1, 1, 1]],
                "scores": [0.5],
                "messages": [
                    [
                        {"content": "Missing role field"},  # No role
                        {"role": "assistant", "content": "This one is OK"},
                    ]
                ],
                "advantages": [[0.5, 0.3, 0.2]],
            },
        )

        print(f"Missing role status: {response1.status_code}")
        if response1.status_code != 200:
            print(f"Error: {response1.json()}")
        assert response1.status_code == 422  # Should fail validation

        # Try missing content
        response2 = requests.post(
            "http://localhost:8000/scored_data",
            json={
                "tokens": [[980, 981, 982]],
                "masks": [[1, 1, 1]],
                "scores": [0.5],
                "messages": [
                    [
                        {"role": "user"},  # No content
                        {"role": "assistant", "content": "This one is OK"},
                    ]
                ],
                "advantages": [[0.5, 0.3, 0.2]],
            },
        )

        print(f"Missing content status: {response2.status_code}")
        if response2.status_code != 200:
            print(f"Error: {response2.json()}")
        assert response2.status_code == 422  # Should fail validation
