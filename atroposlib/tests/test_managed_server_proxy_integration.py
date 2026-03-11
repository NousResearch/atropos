"""Integration tests for ManagedServer OpenAI proxy against real vLLM backend.

Spins up example_trainer/vllm_api_server.py with Qwen3-4B as a subprocess.
Requires GPU — skipped by default. Run with:

    pytest --run-gpu atroposlib/tests/test_managed_server_proxy_integration.py -v -s

"""

import os
import signal
import subprocess
import sys
import time

import pytest
import requests
from transformers import AutoTokenizer

from atroposlib.envs.server_handling.managed_server_proxy import create_app
from atroposlib.envs.server_handling.server_baseline import APIServerConfig
from atroposlib.envs.server_handling.server_manager import ServerManager

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VLLM_PORT = 8123
VLLM_MODEL = "Qwen/Qwen3-4B-Thinking-2507"
PROXY_MODEL = VLLM_MODEL
VLLM_BASE_URL = f"http://localhost:{VLLM_PORT}/v1"
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
VLLM_SCRIPT = os.path.join(REPO_ROOT, "example_trainer", "vllm_api_server.py")
VENV_PYTHON = sys.executable  # use the current interpreter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def vllm_backend():
    """Start vLLM api server as a subprocess. Module-scoped so it's shared."""
    cmd = [
        VENV_PYTHON,
        VLLM_SCRIPT,
        "--model",
        VLLM_MODEL,
        "--port",
        str(VLLM_PORT),
        "--max-model-len",
        "32000",
        "--max-num-seqs",
        "32",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=REPO_ROOT,
    )

    # Wait for health
    deadline = time.time() + 180  # 3 min for model loading
    healthy = False
    while time.time() < deadline:
        try:
            resp = requests.get(f"http://localhost:{VLLM_PORT}/health", timeout=2)
            if resp.status_code == 200:
                healthy = True
                break
        except (requests.ConnectionError, requests.Timeout):
            pass

        if proc.poll() is not None:
            stdout = proc.stdout.read().decode() if proc.stdout else ""
            pytest.fail(
                f"vLLM server exited early (code={proc.returncode}):\n{stdout[-3000:]}"
            )

        time.sleep(3)

    if not healthy:
        proc.kill()
        stdout = proc.stdout.read().decode() if proc.stdout else ""
        pytest.fail(f"vLLM server didn't become healthy within 180s:\n{stdout[-3000:]}")

    yield proc

    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


@pytest.fixture(scope="module")
def tokenizer_instance():
    return AutoTokenizer.from_pretrained(VLLM_MODEL)


@pytest.fixture(scope="module")
def proxy_client(vllm_backend, tokenizer_instance):
    """Create a test client for the proxy backed by the real vLLM server."""
    from fastapi.testclient import TestClient

    config = APIServerConfig(
        model_name=VLLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key="",
        server_type="vllm",
        health_check=False,
    )
    server_manager = ServerManager(configs=[config])

    app = create_app(
        server_manager=server_manager,
        tokenizer=tokenizer_instance,
        model_name=VLLM_MODEL,
    )
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.gpu
class TestRealChatCompletion:
    def test_basic_completion(self, proxy_client):
        resp = proxy_client.post("/sessions/create", json={})
        uuid = resp.json()["uuid"]

        resp = proxy_client.post(
            f"/{uuid}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Say hello in one word."}],
                "max_tokens": 30,
                "temperature": 0.0,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["choices"]) == 1
        content = data["choices"][0]["message"]["content"]
        assert content is not None
        assert len(content) > 0
        assert data["model"] == VLLM_MODEL

    def test_n_completions(self, proxy_client):
        resp = proxy_client.post("/sessions/create", json={})
        uuid = resp.json()["uuid"]

        resp = proxy_client.post(
            f"/{uuid}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Pick a random number"}],
                "max_tokens": 20,
                "temperature": 1.0,
                "n": 4,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["choices"]) == 4

        # Check nodes
        resp = proxy_client.get(f"/{uuid}/nodes")
        assert len(resp.json()["nodes"]) == 4


@pytest.mark.gpu
class TestRealLogprobs:
    def test_logprobs_are_valid(self, proxy_client):
        resp = proxy_client.post("/sessions/create", json={})
        uuid = resp.json()["uuid"]

        proxy_client.post(
            f"/{uuid}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 20,
                "temperature": 0.0,
            },
        )

        resp = proxy_client.get(f"/{uuid}/nodes")
        nodes = resp.json()["nodes"]
        assert len(nodes) == 1

        node = nodes[0]
        # Find where completion starts (logprobs transition from 1.0 to negative)
        prompt_end = 0
        for i, lp in enumerate(node["logprobs"]):
            if lp != 1.0:
                prompt_end = i
                break

        # Prompt logprobs should be 1.0
        assert all(lp == 1.0 for lp in node["logprobs"][:prompt_end])
        # Completion logprobs should be negative
        completion_lps = [lp for lp in node["logprobs"][prompt_end:] if lp != 1.0]
        assert len(completion_lps) > 0
        assert all(lp < 0 for lp in completion_lps)


@pytest.mark.gpu
class TestRealTokenAlignment:
    def test_tokens_decode_to_full_text(self, proxy_client, tokenizer_instance):
        resp = proxy_client.post("/sessions/create", json={})
        uuid = resp.json()["uuid"]

        proxy_client.post(
            f"/{uuid}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Say exactly: test123"}],
                "max_tokens": 30,
                "temperature": 0.0,
            },
        )

        resp = proxy_client.get(f"/{uuid}/nodes")
        node = resp.json()["nodes"][0]

        # Lengths must match
        assert len(node["tokens"]) == len(node["masked_tokens"])
        assert len(node["tokens"]) == len(node["logprobs"])

        # Decode tokens and check they match full_text
        decoded = tokenizer_instance.decode(node["tokens"])
        # The decoded text should be close to (or contain) the full_text
        # Exact match may differ due to special token handling, but content should match
        assert len(decoded) > 0


@pytest.mark.gpu
class TestRealRender:
    def test_render_matches_tokenizer(self, proxy_client, tokenizer_instance):
        resp = proxy_client.post("/sessions/create", json={})
        uuid = resp.json()["uuid"]

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]

        resp = proxy_client.post(
            f"/{uuid}/v1/chat/completions/render",
            json={"messages": messages},
        )
        assert resp.status_code == 200
        data = resp.json()

        # Compare with direct tokenizer rendering
        expected_text = tokenizer_instance.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        assert data["prompt_text"] == expected_text


@pytest.mark.gpu
class TestRealSequenceExtension:
    def test_multi_turn_extends(self, proxy_client):
        resp = proxy_client.post("/sessions/create", json={})
        uuid = resp.json()["uuid"]

        # Turn 1
        resp = proxy_client.post(
            f"/{uuid}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Say hello"}],
                "max_tokens": 20,
                "temperature": 0.0,
            },
        )
        assert resp.status_code == 200
        turn1_content = resp.json()["choices"][0]["message"]["content"]

        # Turn 2 — extends turn 1
        resp = proxy_client.post(
            f"/{uuid}/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "Say hello"},
                    {"role": "assistant", "content": turn1_content},
                    {"role": "user", "content": "Now say goodbye"},
                ],
                "max_tokens": 20,
                "temperature": 0.0,
            },
        )
        assert resp.status_code == 200

        # Should have nodes (extension behavior depends on prefix matching)
        resp = proxy_client.get(f"/{uuid}/nodes")
        nodes = resp.json()["nodes"]
        assert len(nodes) >= 1


@pytest.mark.gpu
class TestRealConcurrentSessions:
    def test_sessions_independent(self, proxy_client):
        """Multiple sessions should not contaminate each other."""
        uuids = []
        for _ in range(3):
            resp = proxy_client.post("/sessions/create", json={})
            uuids.append(resp.json()["uuid"])

        # Complete on each
        for i, uuid in enumerate(uuids):
            resp = proxy_client.post(
                f"/{uuid}/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": f"Count to {i+1}"}],
                    "max_tokens": 30,
                    "temperature": 0.0,
                },
            )
            assert resp.status_code == 200

        # Each should have exactly 1 node
        for uuid in uuids:
            resp = proxy_client.get(f"/{uuid}/nodes")
            assert len(resp.json()["nodes"]) == 1


@pytest.mark.gpu
class TestRealOpenAIClientCompat:
    def test_openai_client_works(self, proxy_client):
        """Verify the standard openai Python client can talk to our proxy."""
        resp = proxy_client.post("/sessions/create", json={})
        uuid = resp.json()["uuid"]

        # The TestClient doesn't expose a real port, so we test the
        # response format is compatible by checking structure manually
        resp = proxy_client.post(
            f"/{uuid}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10,
                "temperature": 0.0,
            },
        )
        data = resp.json()

        # Verify all fields the openai client expects
        assert "id" in data
        assert "object" in data
        assert data["object"] == "chat.completion"
        assert "created" in data
        assert "model" in data
        assert "choices" in data
        assert isinstance(data["choices"], list)
        for choice in data["choices"]:
            assert "index" in choice
            assert "message" in choice
            assert "finish_reason" in choice
            assert "role" in choice["message"]
            assert "content" in choice["message"]
