# Verifiers / Prime Env Hub Integration — Review + Repro Guide

This change integrates Prime Intellect’s Environment Hub environments into Atropos using the `verifiers` interface.

## What “connected to Env Hub” means here
- Prime Env Hub environments are installed locally via the `prime` CLI (`prime env install owner/environment-name@...`).
- At runtime, Atropos loads the installed environment via `verifiers.load_environment(...)`.
- The env process prints a startup line showing:
  - the hub id you configured (`owner/environment-name`)
  - the normalized verifiers id actually passed to `verifiers.load_environment` (`environment-name`)
  - the concrete Python type loaded (`env_type=...`)

## Code tour (small files on purpose)
- `atroposlib/envs/verifiers_env.py`: the Atropos `BaseEnv` implementation that:
  - loads the verifiers env
  - runs rollouts using an OpenAI-compatible client proxy backed by Atropos `ServerManager`
  - scores rollouts via the env’s rubric
  - converts results into Atropos `tokens/masks/scores/messages` and sends to the rollout API
- `atroposlib/envs/verifiers_openai_proxy.py`: the minimal async OpenAI-ish client used by verifiers envs.
- `atroposlib/envs/verifiers_utils.py`: small helper functions (id normalization, message sanitization, model name inference).
- `environments/verifiers_server.py`: tiny entrypoint to run the env as a microservice via `VerifiersEnv.cli()`.
- `environments/configs/verifiers.yaml`: example config.
- `environments/README.md`: quickstart + smoke test.

## Local testing (fast)
```bash
pytest -q atroposlib/tests/test_verifiers_env_adapter.py atroposlib/tests/test_verifiers_env_integration.py
```

If you use pre-commit in this repo:
```bash
pre-commit run --all-files
```

## Manual smoke test (proves Atropos ↔ verifiers wiring)
Terminal A:
```bash
run-api
```

Terminal B:
```bash
curl -sS http://localhost:8000/reset_data
curl -sS -X POST http://localhost:8000/register \
  -H 'Content-Type: application/json' \
  -d '{"wandb_group":"local","wandb_project":"local","batch_size":1,"max_token_len":4096,"checkpoint_dir":"","save_checkpoint_interval":-1,"starting_step":0,"num_steps":5}'
```

Terminal C (start the env):
```bash
python environments/verifiers_server.py serve --config environments/configs/verifiers.yaml
```

Back in Terminal B (once the env has sent a group):
```bash
curl -sS http://localhost:8000/latest_example
```

## Manual smoke test (proves Prime Env Hub install + load)
Prereqs:
```bash
uv tool install prime
prime login
prime env install owner/environment-name@latest --with pip
```

Then set `env.vf_env_name: "owner/environment-name"` in `environments/configs/verifiers.yaml` and repeat the “Manual smoke test” above.

To double-check you can load it locally:
```bash
prime env info owner/environment-name
python -c "import verifiers; verifiers.load_environment('environment-name')"
```

## Review checklist
- No secrets committed (API keys remain placeholders / env vars).
- `env.vf_env_name` supports `owner/environment-name@version` and normalizes to `environment-name`.
- Rollouts route through Atropos `ServerManager` (OpenAI-compatible proxy).
- Rubric scoring path works with modern verifiers envs (`score_group`/`score_rollout`), with fallback for older envs.
