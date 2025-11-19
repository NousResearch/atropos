# Cline Dev Harness

This directory houses per-repo experiments for building, testing, and packaging Cline worker environments before they are promoted to Nomad-managed jobs. Each subdirectory documents a reproducible workflow (Nix profile + bootstrap scripts) for a specific language/toolchain or exemplar repo.

## Layout

```
cline_dev/
  profiles/
    rust/
      flake.nix          # rust-env toolchain definition (Nix)
      default.nix        # (optional) wrapper for dockerTools.buildLayer
      README.md          # profile-specific notes
  examples/
    ratatui_vertical_gauge/
      bootstrap.sh       # clones repo, installs deps, runs smoke tests
      README.md          # describes the scenario + how to run it locally
```

## Workflow

1. Pick a dataset row / repo.
2. Create an example under `examples/` with:
   - Reference to the dataset entry and target prompt.
   - Any repo-specific setup (patches, env vars, secrets).
   - Commands to build/run the Cline core inside a container.
3. If the repo requires a new toolchain, create a profile in `profiles/` and document build instructions.
4. Once the flow works locally (docker-compose or direnv + Nix shell), wire it into the Atropos `ClineAgentEnv` via the worker manager.

This structure keeps experimental scripts versioned while we iterate on the automation story.

## Progress

- ✅ `profiles/rust` flake + Docker image for Rust + Node toolchain.
- ✅ `examples/ratatui_vertical_gauge` bootstrap script (clones repo, runs cargo, emits metadata).
- ✅ `bootstrap_cline_worker.sh` for cloning NousResearch/cline, building `dist-standalone`, packaging `standalone.zip`, running repo bootstrap, and launching the standalone gRPC server.
- ✅ `cline_agent_local_server_smoke.py` orchestrates end-to-end: starts worker, configures Anthropic via ProtoBus, triggers `TaskService.newTask`, validates reasoning stream, and tears everything down.

## Remaining Work

- Nomad job template + worker manager in Atropos (submit, monitor, tear down per trajectory).
- Additional language profiles (python-env, node-env, etc.) and mapping dataset rows to profiles (`envs_required.csv`).
- Artifact capture (ui/api histories, diffs) and reward computation integrated into `collect_trajectory`.
- Parameterize smoke/orchestrator to drive arbitrary dataset rows instead of the single Ratatui example.
- CI or scripted tests to ensure profiles/examples remain reproducible.

### ManagedServer vs Cline Trajectories

- For now, Cline is treated as a **black-box environment**: it talks to the LLM internally and logs its own trajectories (ui/api histories, tool traces).
- We will consume those logs as **offline supervision data**, applying tokenization and masking ourselves during training/refresh, rather than trying to wrap Cline’s internal calls in Atropos’ `ManagedServer`.
- Possible future improvement (not currently planned):
  - Proxy Cline’s LLM/API calls through an Atropos-managed endpoint that uses `ManagedServer` under the hood to capture tokens/logprobs automatically.
  - This would add complexity (Cline → proxy → LLM) but could unify logging; for now it’s deferred in favor of reading Cline’s own logs directly.

## Worker Bootstrap Script

`bootstrap_cline_worker.sh` lives at the root of `cline_dev/` and encapsulates the logic for preparing a worker container:

- Clones (or updates) the NousResearch Cline fork (`CLINE_REPO_URL`, default `https://github.com/NousResearch/cline`) into `CLINE_SRC_DIR` (default `/tmp/nous-cline`).
- Runs `npm install` and `npm run compile-standalone` inside that repo.
- Optionally executes a repo-specific bootstrap script (set `TASK_BOOTSTRAP_SCRIPT`) to prepare `/workspace` or another path.
- Launches the standalone Cline gRPC server with `WORKSPACE_DIR`/`DEV_WORKSPACE_FOLDER` pointing at the prepared repo.

### Key Environment Variables

| Variable              | Default                        | Description                                                    |
|-----------------------|--------------------------------|----------------------------------------------------------------|
| `CLINE_SRC_DIR`       | `/tmp/nous-cline`              | Location where the Cline fork is cloned/built                  |
| `CLINE_REPO_URL`      | `https://github.com/NousResearch/cline` | Git URL for the forked Cline repo                     |
| `WORKSPACE_ROOT`      | `/workspace`                   | Root path for task workspace mounting                          |
| `TASK_BOOTSTRAP_SCRIPT` | *(unset)*                   | Executable script to prepare the repo (e.g., `examples/.../bootstrap.sh`) |
| `PROTOBUS_PORT`       | `26040`                        | Port Cline’s ProtoBus gRPC server listens on                   |
| `HOSTBRIDGE_PORT`     | `26041`                        | Port for HostBridge mock                                       |

### Example

```bash
WORKSPACE_ROOT=/tmp/ratatui-workspace \\
TASK_BOOTSTRAP_SCRIPT=$PWD/examples/ratatui_vertical_gauge/bootstrap.sh \\
CLINE_SRC_DIR=/tmp/nous-cline \\
PROTOBUS_PORT=36040 HOSTBRIDGE_PORT=36041 \\
./bootstrap_cline_worker.sh
```

Use this script as the entrypoint for local Docker containers or Nomad jobs; once it logs “Launching Cline core…”, the orchestrator can connect to `127.0.0.1:$PROTOBUS_PORT` and drive tasks via gRPC.
