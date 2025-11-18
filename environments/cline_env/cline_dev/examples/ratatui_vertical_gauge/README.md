# Ratatui Vertical Gauge Scenario

This example exercises the Rust environment profile by cloning [`ratatui/ratatui`](https://github.com/ratatui/ratatui) and staging a task that mirrors issue [#2148](https://github.com/ratatui/ratatui/issues/2148): *“Add support for vertical gauges.”* We use it as a local smoke test before wiring up Nomad-managed workers.

## Requirements

- Nix (Determinate Systems installer recommended) or Docker with the `cline-rust-env` image.
- Access to the Anthropic (or proxy) model endpoint used by Cline (configure via `.env`).
- Git + network connectivity to GitHub.

## Running Locally (Nix dev shell)

```bash
cd environments/cline_env/cline_dev/profiles/rust
nix develop  # drops into shell with rust + node
```

In another terminal (same repo root):

```bash
cd environments/cline_env/cline_dev/examples/ratatui_vertical_gauge
./bootstrap.sh /tmp/ratatui-workspace
```

`bootstrap.sh` will:

1. Clone `https://github.com/ratatui/ratatui` into `/tmp/ratatui-workspace/ratatui`.
2. Install Rust deps (cargo fetch + build).
3. Prepare a `cline-workspace.json` file containing metadata for our orchestrator.
4. Print the commands for launching the Cline core in that workspace.

## Launching Cline Core

After bootstrap completes, run:

```bash
cd environments/cline_env/cline
NODE_OPTIONS=--max-old-space-size=4096 \
  WORKSPACE_DIR=/tmp/ratatui-workspace/ratatui \
  DEV_WORKSPACE_FOLDER=/tmp/ratatui-workspace/ratatui \
  npm run test:standalone
```

Then, from host Python (or the existing `cline_core_smoke` script), connect to ProtoBus gRPC (`127.0.0.1:26040`), configure Anthropic, and call `TaskService.newTask` with a prompt derived from issue #2148.

## Next Steps

- Port this flow into a Nomad job spec using the `rust-env-image` container.
- Parameterize bootstrap so dataset rows drive repo URL + branch selection automatically.
- Add automated assertions (cargo tests, file diffs) so the example can run in CI.
