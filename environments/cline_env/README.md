# Cline Environment for Atropos

This directory contains the Atropos environment integration for the Cline code agent. It defines how tasks are turned into sandboxed coding environments, how Cline is run inside those environments, and how trajectories are collected for GRPO-style training.

## High-Level Design

- Each **dataset row** (from `NousResearch/swe-agent-13k-2025-06-15`) defines a **task profile**:
  - `repo_name`: GitHub repo identifier (e.g. `phaserjs/phaser`).
  - `repo_url`: full Git URL (if present in the dataset).
  - `language`: primary development language (Python, JS/TS, etc.).
  - `conversations`: list of messages; index `1` with `from == "human"` is treated as the user task prompt.
- For every task profile, we run **`group_size` independent trajectories**:
  - In Atropos terms, `group_size` is the number of trajectories per task used for GRPO.
  - Each trajectory runs in its own **Cline worker container** with a sandboxed workspace.

## Per-Task Environment Profile

For a given task profile, the environment must:

### 1. Prepare the Workspace

- Clone the repo from `repo_url` (or construct it from `repo_name`) into a workspace path, e.g.:

  ```bash
  /workspace/<repo_name>
  ```

- Optionally apply dataset-specific modifications if needed (e.g. pre-existing patch state or partial changes).

### 2. Install Language Toolchain and Dependencies

- Install or activate a toolchain matching `language`, including:
  - Language runtime (Python, Node, Rust, Go, etc.).
  - Package manager (pip/conda, npm/pnpm/yarn, cargo, go, etc.).
- Run dependency installation commands, e.g.:
  - Python: `pip install -r requirements.txt` or `pip install -e .`.
  - Node/TS: `npm install` / `pnpm install` / `yarn install`.
  - Other ecosystems: project-specific build or install steps.
- Install any required CLI tools (compilers, test runners, linters) at pinned versions.

### 3. Install Cline Agent

Inside the worker container image:

- Ensure the container has Cline’s core repo and dependencies:
  - Node + npm/yarn.
  - Cline repo checked out and `npm install` run.
  - `npm run compile-standalone` executed to produce:
    - `dist-standalone/cline-core.js`
    - `dist-standalone/standalone.zip`
- Provide a Cline core gRPC launcher, for example:
  - `scripts/test-standalone-core-api-server.ts`, or
  - A slimmer “offline” variant that does not require remote telemetry / banners / cloud APIs.

### 4. Configure Cline for the Workspace

When starting Cline core inside the container:

- Set environment variables such as:

  ```bash
  WORKSPACE_DIR=/workspace/<repo_name>
  DEV_WORKSPACE_FOLDER=/workspace/<repo_name>
  PROTOBUS_PORT=<core_grpc_port>
  HOSTBRIDGE_PORT=<hostbridge_grpc_port>
  CLINE_DIR=/var/cline/<instance_id>
  ```

- Configure Cline’s API provider to point at the Atropos rollout server:
  - Use `StateService.UpdateSettings` over gRPC to set base URL and API key for the Anthropic / vLLM / sglang-compatible backend.
- Optionally enable auto-approval settings for:
  - File reads/writes.
  - Command execution.
  - Browser/MCP usage (if used in this environment).

## Execution Model per Task

For each task profile:

- Atropos spawns **`group_size` Cline worker containers**:
  - All configured with the same repo and initial workspace state.
  - Each worker runs a single Cline core instance bound to its own gRPC port.
- For each worker/trajectory:
  - Derive the task prompt from the dataset:
    - Use `conversations[1]` where `from == "human"` as the primary user prompt.
  - Send this prompt via gRPC using `TaskService.newTask`.
  - Let Cline run autonomously until completion or a cutoff condition.
  - Collect trajectory artifacts:
    - `api_conversation_history.json` (LLM-level messages).
    - `ui_messages.json` (Cline messages and tool traces).

Atropos then:

- Scores each trajectory (initially using dataset-based labels or a judge model).
- Converts each episode into a `ScoredDataItem`:
  - Tokens and masks via `tokenize_for_trainer`.
  - Scalar reward.
- Groups `group_size` items into a `ScoredDataGroup` for the trainer backend.

## Isolation and Scheduling

### Isolation Goals

- Each trajectory runs in a sandboxed container with:
  - No access to the host filesystem beyond its mounted workspace volume.
  - Restricted network egress (e.g. GitHub, model backend, and necessary package registries only).

### Cluster Scheduling

- Use a scheduler such as **Nomad** to manage Cline worker containers:
  - Define job specs with CPU, RAM, and disk limits.
  - No GPU requirement (Cline core + tooling is CPU-only).
  - Target CPU-rich worker nodes (e.g. DGX nodes’ CPU side with large RAM).
- Atropos environment manager:
  - Requests `group_size` workers per task.
  - Waits for each worker’s Cline gRPC endpoint to become healthy.
  - Runs trajectories via gRPC, then releases workers for reuse or teardown.

## Nix and Reproducible Builds (Optional but Recommended)

To make builds repeatable and portable:

- Use **Nix** to define per-repo or per-language dev environments:
  - Encode runtimes, compilers, and tools in Nix flakes or `devShell`s.
  - Use Nix-built derivations as bases for worker images so that:
    - Cloned repos see a stable toolchain.
    - System dependencies are pinned.
- Over time:
  - Move from a generic “fat” dev image to repo-specific Nix environments.
  - Reduce per-task setup time via Nix caching.

## Future Enhancements

- **Offline Cline core mode**
  - Disable banner fetching, remote config, telemetry, and any cloud APIs.
  - Ensure Cline never exits due to missing external services in the RL environment.

- **Richer reward functions**
  - Frontier LLM scoring of Cline’s output.
  - Code-execution-based scoring for tasks with test suites.
  - Combined signals (e.g. compilation success + test pass rate + style/quality).

- **Per-repo snapshots**
  - Build a base image per repo (repo cloned + deps installed).
  - Spawn `group_size` containers from that base via copy-on-write layers for fast startup.

These requirements should be kept in sync with changes to `ClineAgentEnv`, the Cline core gRPC launcher, and any Nomad/Nix tooling we introduce.

