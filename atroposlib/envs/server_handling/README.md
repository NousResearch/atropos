# Server Handling

This module provides server abstraction layers for different LLM inference backends.

## ManagedServer

For automatic token and logprob tracking, see the [ManagedServer Guide](MANAGED_SERVER.md).

> **Note:** OpenAI endpoints do not support token IDs/logprobs required for ManagedServer. Set `ATROPOS_ALLOW_DUMMY_MANAGED_SERVER=1` to use a placeholder implementation for testing/evaluation. See [OpenAI Endpoint Limitations](MANAGED_SERVER.md#openai-endpoint-limitations) for details.

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

1. **OpenRouter**: Only accepts ONE of `effort` or `max_tokens`, not both. When both are specified, effort takes priority.
2. **OpenAI**: All effort levels are passed through directly.
3. **Auto-enable**: Setting `effort` or `max_tokens` automatically enables reasoning mode.
