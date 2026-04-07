# Browserbase Environments

This package contains the Atropos-native Browserbase integration.

Current status:

- `BrowserbaseEnv`: shared runtime layer for Browserbase-backed environments
- DOM mode: implemented with Stagehand and structured tool calls
- CUA mode: reserved but intentionally not enabled until Atropos has an explicit multimodal rollout-accounting path

Install browser dependencies with:

```bash
uv sync --extra browser
```

For training, use a backend with real token/logprob support and a configured ManagedServer tool parser.
