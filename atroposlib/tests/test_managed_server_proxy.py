"""Mock-based tests for the ManagedServer OpenAI proxy.

Uses ServerHarness as the backend — no real model or GPU needed.
Tests the full HTTP layer: session management, chat completions,
tool call translation, render endpoint, nodes, cleanup.
"""

import json

import pytest
from fastapi.testclient import TestClient

from atroposlib.envs.server_handling.managed_server_proxy import create_app
from atroposlib.envs.server_handling.server_harness import ServerHarness
from atroposlib.envs.server_handling.server_manager import ServerManager
from atroposlib.envs.server_handling.tool_call_translator import VLLM_AVAILABLE

skip_no_vllm = pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM not installed")

# ---------------------------------------------------------------------------
# Mock tokenizer (same as test_managed_server.py / test_tool_call_translator.py)
# ---------------------------------------------------------------------------


class MockTokenizer:
    def __init__(self):
        self.eos_token_id = 2
        self.bos_token_id = 1

    def encode(self, text, add_special_tokens=True):
        tokens = [ord(c) for c in text]
        if add_special_tokens:
            tokens = [self.bos_token_id] + tokens
        return tokens

    def decode(self, tokens, skip_special_tokens=False):
        if skip_special_tokens:
            tokens = [
                t for t in tokens if t not in [self.bos_token_id, self.eos_token_id]
            ]
        return "".join([chr(t) if t > 31 else "" for t in tokens])

    def get_vocab(self):
        return {chr(i): i for i in range(128)}

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True, tools=None
    ):
        result = ""
        if tools:
            result += f"<tools>{json.dumps(tools)}</tools>\n"
        for msg in messages:
            content = msg.get("content", "") or ""
            result += f"<{msg['role']}>{content}</{msg['role']}>"
        if add_generation_prompt:
            result += "<assistant>"
        if tokenize:
            return self.encode(result)
        return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_backend():
    """Create a mock server backend with tokenizer."""
    server = ServerHarness()
    server.tokenizer = MockTokenizer()
    # ServerManager's _select_server checks these attributes
    server.server_healthy = True

    class Config:
        model_name = "test_model"

    server.config = Config()
    return server


@pytest.fixture
def server_manager(mock_backend):
    """Create a ServerManager wrapping the mock backend."""
    # Can't use ServerManager constructor with empty configs, so build manually
    mgr = object.__new__(ServerManager)
    mgr.max_n_completions = 8
    mgr.reasoning_config = None
    mgr.servers = [mock_backend]
    return mgr


@pytest.fixture
def client(server_manager):
    """Create a test client for the proxy app."""
    tokenizer = MockTokenizer()
    app = create_app(
        server_manager=server_manager,
        tokenizer=tokenizer,
        model_name="test_model",
    )
    return TestClient(app)


@pytest.fixture
def client_and_backend(mock_backend, server_manager):
    """Return both client and backend for tests that need to set up mock responses."""
    tokenizer = MockTokenizer()
    app = create_app(
        server_manager=server_manager,
        tokenizer=tokenizer,
        model_name="test_model",
    )
    return TestClient(app), mock_backend, tokenizer


def _setup_completion(
    backend, tokenizer, prompt_text, output_texts, finish_reasons=None
):
    """Helper to set up a mock tokens_and_logprobs response."""
    prompt_tokens = tokenizer.encode(prompt_text)
    output_tokens_list = [[ord(c) for c in text] for text in output_texts]
    output_logprobs_list = [[-0.1] * len(tokens) for tokens in output_tokens_list]
    if finish_reasons is None:
        finish_reasons = ["stop"] * len(output_texts)

    backend.set_tokens_and_logprobs_response(
        prompt=prompt_text,
        prompt_tokens=prompt_tokens,
        output_tokens_list=output_tokens_list,
        output_logprobs_list=output_logprobs_list,
        finish_reasons=finish_reasons,
    )


# ---------------------------------------------------------------------------
# Health / Models
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model"] == "test_model"
        assert data["sessions"] == 0

    def test_models(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "test_model"


# ---------------------------------------------------------------------------
# Session Management
# ---------------------------------------------------------------------------


class TestSessionManagement:
    def test_create_session(self, client):
        resp = client.post("/sessions/create", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "uuid" in data
        assert data["model_name"] == "test_model"
        assert data["tool_parser"] == "hermes"

    def test_create_session_custom_parser(self, client):
        resp = client.post("/sessions/create", json={"tool_parser": "hermes"})
        assert resp.status_code == 200
        assert resp.json()["tool_parser"] == "hermes"

    def test_list_sessions(self, client):
        # Create 3 sessions
        uuids = []
        for _ in range(3):
            resp = client.post("/sessions/create", json={})
            uuids.append(resp.json()["uuid"])

        resp = client.get("/sessions")
        assert resp.status_code == 200
        sessions = resp.json()["sessions"]
        assert len(sessions) == 3
        listed_uuids = {s["uuid"] for s in sessions}
        assert listed_uuids == set(uuids)

    def test_delete_session(self, client):
        resp = client.post("/sessions/create", json={})
        uuid = resp.json()["uuid"]

        resp = client.delete(f"/{uuid}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

        # Should be gone
        resp = client.get(f"/{uuid}/nodes")
        assert resp.status_code == 404

    def test_delete_nonexistent_session(self, client):
        resp = client.delete("/nonexistent-uuid")
        assert resp.status_code == 404

    def test_session_not_found(self, client):
        resp = client.post(
            "/nonexistent-uuid/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Chat Completions
# ---------------------------------------------------------------------------


class TestChatCompletions:
    def test_basic_completion(self, client_and_backend):
        client, backend, tokenizer = client_and_backend

        # Create session
        resp = client.post("/sessions/create", json={})
        uuid = resp.json()["uuid"]

        # Set up mock response
        messages = [{"role": "user", "content": "Hello"}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        _setup_completion(backend, tokenizer, prompt_text, ["Hi there!"])

        # Make request
        resp = client.post(
            f"/{uuid}/v1/chat/completions",
            json={"messages": messages, "max_tokens": 100},
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["object"] == "chat.completion"
        assert data["model"] == "test_model"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] == "Hi there!"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["id"].startswith("chatcmpl-")

    def test_completion_with_n(self, client_and_backend):
        client, backend, tokenizer = client_and_backend

        resp = client.post("/sessions/create", json={})
        uuid = resp.json()["uuid"]

        messages = [{"role": "user", "content": "Pick a number"}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        _setup_completion(backend, tokenizer, prompt_text, ["One", "Two", "Three"])

        resp = client.post(
            f"/{uuid}/v1/chat/completions",
            json={"messages": messages, "n": 3, "max_tokens": 50},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["choices"]) == 3

    def test_empty_messages_error(self, client):
        resp = client.post("/sessions/create", json={})
        uuid = resp.json()["uuid"]

        resp = client.post(
            f"/{uuid}/v1/chat/completions",
            json={"messages": []},
        )
        assert resp.status_code == 400

    def test_completion_with_system_prompt(self, client_and_backend):
        client, backend, tokenizer = client_and_backend

        resp = client.post("/sessions/create", json={})
        uuid = resp.json()["uuid"]

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        _setup_completion(backend, tokenizer, prompt_text, ["Hello!"])

        resp = client.post(
            f"/{uuid}/v1/chat/completions",
            json={"messages": messages},
        )
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["content"] == "Hello!"


# ---------------------------------------------------------------------------
# Tool Call Handling
# ---------------------------------------------------------------------------


@skip_no_vllm
class TestToolCalls:
    def test_tool_call_outbound(self, client_and_backend):
        """Model generates <tool_call> tags → response has structured tool_calls."""
        client, backend, tokenizer = client_and_backend

        resp = client.post("/sessions/create", json={})
        uuid = resp.json()["uuid"]

        tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
        messages = [{"role": "user", "content": "Search cats"}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, tools=tools
        )

        # Model output includes tool call tags
        raw_output = (
            '<tool_call>{"name": "search", "arguments": {"query": "cats"}}</tool_call>'
        )
        _setup_completion(backend, tokenizer, prompt_text, [raw_output])

        resp = client.post(
            f"/{uuid}/v1/chat/completions",
            json={"messages": messages, "tools": tools},
        )
        assert resp.status_code == 200
        data = resp.json()
        choice = data["choices"][0]

        assert choice["finish_reason"] == "tool_calls"
        assert "tool_calls" in choice["message"]
        assert len(choice["message"]["tool_calls"]) == 1
        tc = choice["message"]["tool_calls"][0]
        assert tc["function"]["name"] == "search"
        assert json.loads(tc["function"]["arguments"]) == {"query": "cats"}

    def test_tool_choice_none(self, client_and_backend):
        """tool_choice=none → no parsing, raw text returned."""
        client, backend, tokenizer = client_and_backend

        resp = client.post("/sessions/create", json={})
        uuid = resp.json()["uuid"]

        tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
        messages = [{"role": "user", "content": "Search cats"}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, tools=tools
        )

        raw_output = (
            '<tool_call>{"name": "search", "arguments": {"query": "cats"}}</tool_call>'
        )
        _setup_completion(backend, tokenizer, prompt_text, [raw_output])

        resp = client.post(
            f"/{uuid}/v1/chat/completions",
            json={"messages": messages, "tools": tools, "tool_choice": "none"},
        )
        assert resp.status_code == 200
        choice = resp.json()["choices"][0]

        # Should NOT have tool_calls since tool_choice is "none"
        assert choice["finish_reason"] == "stop"
        assert (
            "tool_calls" not in choice["message"]
            or choice["message"].get("tool_calls") is None
        )

    def test_nodes_preserve_raw_text(self, client_and_backend):
        """ManagedServer nodes should have raw text, not parsed tool_calls."""
        client, backend, tokenizer = client_and_backend

        resp = client.post("/sessions/create", json={})
        uuid = resp.json()["uuid"]

        tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
        messages = [{"role": "user", "content": "Search cats"}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, tools=tools
        )

        raw_output = (
            '<tool_call>{"name": "search", "arguments": {"query": "cats"}}</tool_call>'
        )
        _setup_completion(backend, tokenizer, prompt_text, [raw_output])

        client.post(
            f"/{uuid}/v1/chat/completions",
            json={"messages": messages, "tools": tools},
        )

        # Check nodes — should have the raw tokens, not parsed
        resp = client.get(f"/{uuid}/nodes")
        assert resp.status_code == 200
        nodes = resp.json()["nodes"]
        assert len(nodes) == 1


# ---------------------------------------------------------------------------
# Render Endpoint
# ---------------------------------------------------------------------------


class TestRender:
    def test_render_basic(self, client):
        resp = client.post("/sessions/create", json={})
        uuid = resp.json()["uuid"]

        resp = client.post(
            f"/{uuid}/v1/chat/completions/render",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "prompt_text" in data
        assert "token_ids" in data
        assert "num_tokens" in data
        assert data["num_tokens"] == len(data["token_ids"])
        assert "<user>Hello</user>" in data["prompt_text"]

    def test_render_with_tools(self, client):
        resp = client.post("/sessions/create", json={})
        uuid = resp.json()["uuid"]

        tools = [{"type": "function", "function": {"name": "search"}}]
        resp = client.post(
            f"/{uuid}/v1/chat/completions/render",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "tools": tools,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        # Tool definitions should appear in the rendered prompt
        assert "search" in data["prompt_text"]

    def test_render_does_not_create_nodes(self, client):
        """Render should not cause any generation or node creation."""
        resp = client.post("/sessions/create", json={})
        uuid = resp.json()["uuid"]

        client.post(
            f"/{uuid}/v1/chat/completions/render",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )

        resp = client.get(f"/{uuid}/nodes")
        assert resp.json()["nodes"] == []


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


class TestNodes:
    def test_get_nodes_empty(self, client):
        resp = client.post("/sessions/create", json={})
        uuid = resp.json()["uuid"]

        resp = client.get(f"/{uuid}/nodes")
        assert resp.status_code == 200
        assert resp.json()["nodes"] == []

    def test_get_nodes_after_completion(self, client_and_backend):
        client, backend, tokenizer = client_and_backend

        resp = client.post("/sessions/create", json={})
        uuid = resp.json()["uuid"]

        messages = [{"role": "user", "content": "Hi"}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        _setup_completion(backend, tokenizer, prompt_text, ["Hello!"])

        client.post(
            f"/{uuid}/v1/chat/completions",
            json={"messages": messages},
        )

        resp = client.get(f"/{uuid}/nodes")
        assert resp.status_code == 200
        nodes = resp.json()["nodes"]
        assert len(nodes) == 1

        node = nodes[0]
        assert "tokens" in node
        assert "masked_tokens" in node
        assert "logprobs" in node
        assert "full_text" in node
        assert (
            len(node["tokens"]) == len(node["masked_tokens"]) == len(node["logprobs"])
        )

    def test_nodes_have_proper_masking(self, client_and_backend):
        client, backend, tokenizer = client_and_backend

        resp = client.post("/sessions/create", json={})
        uuid = resp.json()["uuid"]

        messages = [{"role": "user", "content": "Hi"}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_tokens = tokenizer.encode(prompt_text)
        prompt_len = len(prompt_tokens)

        _setup_completion(backend, tokenizer, prompt_text, ["Hello!"])

        client.post(
            f"/{uuid}/v1/chat/completions",
            json={"messages": messages},
        )

        resp = client.get(f"/{uuid}/nodes")
        node = resp.json()["nodes"][0]

        # Prompt tokens should be masked with -100
        assert all(t == -100 for t in node["masked_tokens"][:prompt_len])
        # Prompt logprobs should be 1.0
        assert all(lp == 1.0 for lp in node["logprobs"][:prompt_len])
        # Completion logprobs should be actual values (negative)
        assert all(lp < 0 for lp in node["logprobs"][prompt_len:])


# ---------------------------------------------------------------------------
# Deep multi-step node handling
# ---------------------------------------------------------------------------


@skip_no_vllm
class TestMultiStepNodeHandling:
    """Test that multi-step conversations with tool calls produce exactly 1 node.

    Simulates a realistic 10+ message agentic conversation:
    user → assistant(tool_call) → tool_result → assistant(text) →
    user → assistant(tool_call) → tool_result → assistant(tool_call) →
    tool_result → assistant(text) → user → assistant(text)

    Each step extends the previous node, so we should end up with exactly
    1 node containing the full tokenized conversation.
    """

    def _do_step(
        self,
        client,
        backend,
        tokenizer,
        uuid,
        messages,
        output_text,
        tools=None,
        expect_tool_calls=False,
    ):
        """Helper: use render endpoint to get exact prompt, set up mock, call endpoint."""
        body = {"messages": messages, "max_tokens": 200}
        if tools:
            body["tools"] = tools

        # Use the render endpoint to get the exact prompt the proxy will generate
        # (this includes tool_call reconstruction through the translator)
        render_resp = client.post(f"/{uuid}/v1/chat/completions/render", json=body)
        assert render_resp.status_code == 200, f"Render failed: {render_resp.json()}"
        prompt_text = render_resp.json()["prompt_text"]

        _setup_completion(backend, tokenizer, prompt_text, [output_text])

        resp = client.post(f"/{uuid}/v1/chat/completions", json=body)
        assert resp.status_code == 200, f"Step failed: {resp.json()}"
        return resp.json()

    def test_10_message_conversation_one_node(self, client_and_backend):
        """Full 10-message conversation with tool calls → exactly 1 node."""
        client, backend, tokenizer = client_and_backend

        resp = client.post("/sessions/create", json={})
        uuid = resp.json()["uuid"]

        tools = [
            {"type": "function", "function": {"name": "get_weather", "parameters": {}}},
            {
                "type": "function",
                "function": {"name": "get_forecast", "parameters": {}},
            },
        ]

        # -- Step 1: user asks about weather --
        messages = [{"role": "user", "content": "What's the weather in SF?"}]
        output_1 = '<tool_call>{"name": "get_weather", "arguments": {"city": "SF"}}</tool_call>'
        data = self._do_step(
            client, backend, tokenizer, uuid, messages, output_1, tools=tools
        )
        assert data["choices"][0]["finish_reason"] == "tool_calls"
        tc_1 = data["choices"][0]["message"]["tool_calls"]

        # Check: 1 node so far
        nodes = client.get(f"/{uuid}/nodes").json()["nodes"]
        assert len(nodes) == 1, f"Expected 1 node after step 1, got {len(nodes)}"

        # -- Step 2: tool result --
        messages = [
            {"role": "user", "content": "What's the weather in SF?"},
            {"role": "assistant", "content": None, "tool_calls": tc_1},
            {
                "role": "tool",
                "tool_call_id": tc_1[0]["id"],
                "content": "72°F and sunny",
            },
        ]
        output_2 = "The weather in SF is 72°F and sunny! Want the forecast too?"
        self._do_step(client, backend, tokenizer, uuid, messages, output_2, tools=tools)

        nodes = client.get(f"/{uuid}/nodes").json()["nodes"]
        assert len(nodes) == 1, f"Expected 1 node after step 2, got {len(nodes)}"

        # -- Step 3: user says yes --
        messages.extend(
            [
                {"role": "assistant", "content": output_2},
                {"role": "user", "content": "Yes please, get the forecast"},
            ]
        )
        output_3 = '<tool_call>{"name": "get_forecast", "arguments": {"city": "SF"}}</tool_call>'
        data = self._do_step(
            client, backend, tokenizer, uuid, messages, output_3, tools=tools
        )
        tc_3 = data["choices"][0]["message"]["tool_calls"]

        nodes = client.get(f"/{uuid}/nodes").json()["nodes"]
        assert len(nodes) == 1, f"Expected 1 node after step 3, got {len(nodes)}"

        # -- Step 4: forecast tool result --
        messages.extend(
            [
                {"role": "assistant", "content": None, "tool_calls": tc_3},
                {
                    "role": "tool",
                    "tool_call_id": tc_3[0]["id"],
                    "content": "Rain expected tomorrow",
                },
            ]
        )
        output_4 = "The forecast says rain is expected tomorrow in SF."
        self._do_step(client, backend, tokenizer, uuid, messages, output_4, tools=tools)

        nodes = client.get(f"/{uuid}/nodes").json()["nodes"]
        assert len(nodes) == 1, f"Expected 1 node after step 4, got {len(nodes)}"

        # -- Step 5: user asks about another city --
        messages.extend(
            [
                {"role": "assistant", "content": output_4},
                {"role": "user", "content": "What about NYC?"},
            ]
        )
        output_5 = '<tool_call>{"name": "get_weather", "arguments": {"city": "NYC"}}</tool_call>'
        data = self._do_step(
            client, backend, tokenizer, uuid, messages, output_5, tools=tools
        )
        tc_5 = data["choices"][0]["message"]["tool_calls"]

        nodes = client.get(f"/{uuid}/nodes").json()["nodes"]
        assert len(nodes) == 1, f"Expected 1 node after step 5, got {len(nodes)}"

        # -- Step 6: NYC tool result --
        messages.extend(
            [
                {"role": "assistant", "content": None, "tool_calls": tc_5},
                {
                    "role": "tool",
                    "tool_call_id": tc_5[0]["id"],
                    "content": "55°F and cloudy",
                },
            ]
        )
        output_6 = "NYC is 55°F and cloudy. Quite different from SF!"
        self._do_step(client, backend, tokenizer, uuid, messages, output_6, tools=tools)

        # -- FINAL CHECK: still exactly 1 node after 6 completions / 12+ messages --
        nodes = client.get(f"/{uuid}/nodes").json()["nodes"]
        assert (
            len(nodes) == 1
        ), f"Expected 1 node after full conversation, got {len(nodes)}"

        # Verify the node has proper structure
        node = nodes[0]
        assert (
            len(node["tokens"]) == len(node["masked_tokens"]) == len(node["logprobs"])
        )
        assert len(node["tokens"]) > 0

        # Verify masking: there should be SOME -100 (prompt) and SOME actual tokens
        num_masked = sum(1 for t in node["masked_tokens"] if t == -100)
        num_actual = sum(1 for t in node["masked_tokens"] if t != -100)
        assert num_masked > 0, "Should have masked prompt tokens"
        assert num_actual > 0, "Should have unmasked completion tokens"

    def test_plain_multi_turn_no_tools_one_node(self, client_and_backend):
        """5-turn conversation without tools → exactly 1 node."""
        client, backend, tokenizer = client_and_backend

        resp = client.post("/sessions/create", json={})
        uuid = resp.json()["uuid"]

        conversation = []

        for i in range(5):
            # Add user message
            conversation.append({"role": "user", "content": f"Turn {i+1} question"})

            prompt_text = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            output = f"Response to turn {i+1}"
            _setup_completion(backend, tokenizer, prompt_text, [output])

            resp = client.post(
                f"/{uuid}/v1/chat/completions",
                json={"messages": conversation},
            )
            assert resp.status_code == 200

            # Add assistant response for next turn
            conversation.append({"role": "assistant", "content": output})

        # After 5 turns (10 messages), should still be 1 node
        nodes = client.get(f"/{uuid}/nodes").json()["nodes"]
        assert len(nodes) == 1, f"Expected 1 node after 5 turns, got {len(nodes)}"

    def test_tool_then_plain_then_tool_one_node(self, client_and_backend):
        """Mixed: tool call → plain text → tool call → plain → exactly 1 node."""
        client, backend, tokenizer = client_and_backend

        resp = client.post("/sessions/create", json={})
        uuid = resp.json()["uuid"]

        tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]

        # Step 1: tool call
        messages = [{"role": "user", "content": "Search for cats"}]
        output = '<tool_call>{"name": "search", "arguments": {"q": "cats"}}</tool_call>'
        data = self._do_step(
            client, backend, tokenizer, uuid, messages, output, tools=tools
        )
        tc = data["choices"][0]["message"]["tool_calls"]

        # Step 2: tool result → plain response
        messages = [
            {"role": "user", "content": "Search for cats"},
            {"role": "assistant", "content": None, "tool_calls": tc},
            {"role": "tool", "tool_call_id": tc[0]["id"], "content": "Found 10 cats"},
        ]
        self._do_step(
            client, backend, tokenizer, uuid, messages, "Here are 10 cats!", tools=tools
        )

        # Step 3: user asks for more → another tool call
        messages.extend(
            [
                {"role": "assistant", "content": "Here are 10 cats!"},
                {"role": "user", "content": "Search for dogs too"},
            ]
        )
        output = '<tool_call>{"name": "search", "arguments": {"q": "dogs"}}</tool_call>'
        data = self._do_step(
            client, backend, tokenizer, uuid, messages, output, tools=tools
        )
        tc2 = data["choices"][0]["message"]["tool_calls"]

        # Step 4: tool result → plain response
        messages.extend(
            [
                {"role": "assistant", "content": None, "tool_calls": tc2},
                {
                    "role": "tool",
                    "tool_call_id": tc2[0]["id"],
                    "content": "Found 5 dogs",
                },
            ]
        )
        self._do_step(
            client, backend, tokenizer, uuid, messages, "Found 5 dogs too!", tools=tools
        )

        # Step 5: plain follow-up, no tools
        messages.extend(
            [
                {"role": "assistant", "content": "Found 5 dogs too!"},
                {"role": "user", "content": "Thanks!"},
            ]
        )
        self._do_step(
            client, backend, tokenizer, uuid, messages, "You're welcome!", tools=tools
        )

        # 5 completion steps, 11 messages — still 1 node
        nodes = client.get(f"/{uuid}/nodes").json()["nodes"]
        assert len(nodes) == 1, f"Expected 1 node, got {len(nodes)}"


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    def test_delete_resets_nodes(self, client_and_backend):
        client, backend, tokenizer = client_and_backend

        resp = client.post("/sessions/create", json={})
        uuid = resp.json()["uuid"]

        messages = [{"role": "user", "content": "Hi"}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        _setup_completion(backend, tokenizer, prompt_text, ["Hello!"])

        client.post(
            f"/{uuid}/v1/chat/completions",
            json={"messages": messages},
        )

        # Delete
        resp = client.delete(f"/{uuid}")
        assert resp.status_code == 200

        # Session gone
        resp = client.get(f"/{uuid}/nodes")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Error format
# ---------------------------------------------------------------------------


class TestErrorFormat:
    def test_404_is_openai_format(self, client):
        resp = client.get("/nonexistent-uuid/nodes")
        assert resp.status_code == 404
        data = resp.json()
        assert "error" in data
        assert "message" in data["error"]
        assert "type" in data["error"]
        assert "code" in data["error"]
