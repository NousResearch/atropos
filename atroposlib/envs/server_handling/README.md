# Server Handling

This module provides server abstraction layers for different LLM inference backends.

## ManagedServer

For automatic token and logprob tracking, see the [ManagedServer Guide](MANAGED_SERVER.md).

> **Note:** OpenAI endpoints do not support token IDs/logprobs required for ManagedServer. Set `ATROPOS_ALLOW_DUMMY_MANAGED_SERVER=1` to use a placeholder implementation for testing/evaluation. See [OpenAI Endpoint Limitations](MANAGED_SERVER.md#openai-endpoint-limitations) for details.

### Normalized `get_logprobs` API

`ManagedServer` and supported server backends expose a normalized `get_logprobs(...)` interface so callers can consume a single schema:

- `prompt_tokens`
- `prompt_topk_token_ids`
- `prompt_topk_logprobs`

Backends are expected to return real prompt top-k arrays (`[pos][k]`) matching this schema.

## Tool Call Support

ManagedServer supports OpenAI-style tool calling via vLLM's tool parsers. Pass `tool_parser` at init:

```python
server_manager = ServerManager(
    configs=[APIServerConfig(...)],
    tool_parser="hermes",  # or llama3_json, mistral, deepseek_v3, qwen3_coder, etc.
)

async with server_manager.managed_server(tokenizer=tokenizer) as managed:
    result = await managed.chat_completion(
        messages=[{"role": "user", "content": "What's the weather?"}],
        tools=[{
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}
        }],
        tool_choice="auto",  # "auto", "none", "required"
    )

    # Structured tool_calls in response
    if result.choices[0].message.tool_calls:
        print(result.choices[0].message.tool_calls)

    # Nodes still have raw text with <tool_call> tags for training
    nodes = managed.get_state()["nodes"]
```

Requires `vllm` installed. Without it, tool parsing is disabled with a warning — everything else still works.

## OpenAI Proxy

Exposes ManagedServer as an OpenAI-compatible HTTP API for external tools (CLIs, GUIs, microservices).

### Standalone

```bash
python -m atroposlib.envs.server_handling.managed_server_proxy \
    --config servers.json --port 9100
```

`servers.json`:
```json
{
    "model_name": "Qwen/Qwen3-4B",
    "servers": [
        {"base_url": "http://gpu1:8000/v1", "server_type": "vllm"},
        {"base_url": "http://gpu2:8000/v1", "server_type": "vllm"}
    ]
}
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/sessions/create` | Create session. Optional `base_url` to pin to a server, `tool_parser` name. |
| POST | `/{uuid}/v1/chat/completions` | OpenAI chat completions (with tools support). |
| POST | `/{uuid}/v1/chat/completions/render` | Preview rendered prompt without generating. |
| GET | `/{uuid}/nodes` | Get tracked tokens/logprobs/masks for training. |
| DELETE | `/{uuid}` | Cleanup session. |
| GET | `/sessions` | List active sessions. |
| GET | `/servers` | List backend servers. |
| POST | `/setup` | Push server config (used by ServerManager). |
| GET | `/v1/models` | List models. |
| GET | `/health` | Health check. |

### Via ServerManager

```python
server_manager = ServerManager(
    configs=[APIServerConfig(...)],
    proxy_url="http://localhost:9100",  # auto-enables proxy mode
    tool_parser="hermes",
)

# managed_server() now routes through the proxy
async with server_manager.managed_server(tokenizer=tokenizer) as managed:
    result = await managed.chat_completion(messages=[...], tools=[...])
    url = managed.get_url()  # "http://localhost:9100/{uuid}/v1" — hand to external apps
    nodes = await managed.fetch_state()  # get tokens/logprobs
```

Or set `ATROPOS_PROXY_URL=http://localhost:9100` env var instead of passing `proxy_url`.

## Reasoning Model Support

The `ReasoningConfig` class enables support for reasoning/thinking models across different providers.

### Provider Differences

| Feature | OpenAI | OpenRouter / Others |
|---------|--------|---------------------|
| Format | `{"reasoning_effort": "high"}` | `{"reasoning": {"enabled": true, "effort": "high"}}` |
| Effort Levels | `none`, `minimal`, `low`, `medium`, `high`, `xhigh` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh` |
| Max Tokens | Not supported | `{"reasoning": {"max_tokens": 16000}}` |
| Temperature | Must be `1.0` | No restriction |
| Token Param | `max_completion_tokens` | `max_tokens` |

### Effort Level to Token Mapping

When providers don't support effort strings, effort levels map to approximate token budgets (based on 32k base):

| Effort | Tokens | Percentage |
|--------|--------|------------|
| none | 1,024 | Minimum |
| minimal | 3,200 | ~10% |
| low | 6,400 | ~20% |
| medium | 16,000 | ~50% |
| high | 25,600 | ~80% |
| xhigh | 30,400 | ~95% |

### Provider Token Limits

- **OpenRouter**: Caps Anthropic reasoning at 1,024-32,000 tokens ([docs](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens))
- **Native Anthropic**: Supports up to 128k extended thinking tokens

### Usage

Reasoning is only injected for **chat completions** (not completions or logprobs API).

```python
# Via environment config
config = BaseEnvConfig(
    thinking_mode=True,
    reasoning_effort="high",
    max_reasoning_tokens=16000,
)

# Direct ReasoningConfig
reasoning_config = ReasoningConfig(
    enabled=True,
    effort="high",
    max_tokens=16000,
)
```

### Bypassing Reasoning Injection

Pass `skip_reasoning=True` to any chat completion call:

```python
await server.chat_completion(messages=messages, skip_reasoning=True)
```

### Important Constraints

1. **OpenRouter**: Only accepts ONE of `effort` or `max_tokens`, not both. When both specified, effort takes priority.
2. **OpenAI**: All effort levels are passed through directly.
3. **Auto-enable**: Setting `effort` or `max_tokens` automatically enables reasoning mode.
