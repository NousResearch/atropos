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

## Execution Environments: Detailed Plan

This section captures the concrete steps required to convert any dataset row into an isolated Cline rollout environment. It is intentionally prescriptive so each stage can be automated and tested independently.

### 1. Task Intake

1. Fetch the next row from `NousResearch/swe-agent-13k-2025-06-15` (or similar JSONL datasets).
2. Extract key columns:
   - `repo_name`, `repo_url`, `language`.
   - `conversation[1]` where `role == "human"` → task prompt.
   - Additional metadata (difficulty, tags, etc.) for logging/metrics.
3. Construct an Atropos `TaskProfile` object to track which rollout consumed which dataset row.

### 2. Environment Profiles

To keep builds reproducible we will curate a small library of Nix-based toolchain profiles:

| Profile ID   | Includes                                                     | Targets                              |
|--------------|--------------------------------------------------------------|--------------------------------------|
| `python-env` | CPython, pip/poetry/pdm, pytest tooling                      | Python web/backend repos             |
| `node-env`   | Node LTS, npm/pnpm/yarn, eslint/prettier                      | JS/TS frontends & services           |
| `rust-env`   | rustup + stable toolchain, cargo install, clippy, rustfmt     | Rust CLI/libs (e.g. `ratatui`)       |
| `go-env`     | Go toolchain, `gotest`, goreleaser basics                     | Go microservices/CLI                 |
| `c-env`      | GCC/Clang, make/cmake ninja, ctest, valgrind                  | C/C++ repos                          |

Each profile is defined as a Nix flake (`devShell`/`packages`) that can be built into a Docker/OCI image. Repo-specific overrides can be layered on top when necessary.

### 3. Container Preparation (per task)

1. **Select base profile** based on `language`; fall back to a generic `dev-base`.
2. Build container image via Nix (`nix build .#rust-env-container` etc.) that includes:
   - Toolchains, package managers, and the Cline CLI prerequisites (Node 22+, npm).
   - Helper scripts for cloning repos and bootstrapping Cline.
3. Layout inside container:
   - `/workspace` → task repo clone (writable).
   - `/cline` → vendored Cline repo + build artifacts (read-only base + writable overlay).
   - `/cache` → optional shared caches (cargo, pip, npm) with per-job isolation.
4. Bootstrap steps:
   - Clone `repo_url` into `/workspace/<repo_name>`.
   - Checkout dataset-specified branch/commit if provided.
   - Install project dependencies (cargo fetch/build, npm install, pip install, etc.).
   - Smoke-test tooling (`cargo test --lib`, `npm run lint`, etc.) and log results.
   - Copy the Atropos vendored Cline repo into `/cline`, run `npm install`, `npm run compile-standalone`, and cache `dist-standalone`.

### 4. Runtime Orchestration

We will create a `cline_env` orchestration module responsible for worker lifecycle:

1. **Nomad job template**:
   - Resources: ~4 CPU cores, 8–16 GB RAM, 20+ GB disk (no GPU).
   - Task driver: Docker; image is the per-language Nix build.
   - Entrypoint: `bootstrap_cline_worker.sh`.
   - Ports: expose ProtoBus + HostBridge (or use Nomad networking).
2. `bootstrap_cline_worker.sh` responsibilities:
   - Read environment variables (task ID, dataset row JSON, API credentials).
   - Ensure `/workspace/<repo>` exists (clone if necessary, apply dataset mutations).
   - Export `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`, or policy-server credentials.
   - Launch `node scripts/test-standalone-core-api-server.ts` with:
     - `WORKSPACE_DIR`/`DEV_WORKSPACE_FOLDER` pointing to repo.
     - `CLINE_DISABLE_BANNERS=true`, `CLINE_DISABLE_REMOTE_CONFIG=true`.
   - Wait for ProtoBus gRPC health check and signal readiness (e.g. via file/socket).
3. Atropos coordinator workflow:
   - For each dataset row, request `group_size` Nomad allocations.
   - When each worker is healthy:
     1. Use gRPC to call `UiService.initializeWebview`.
     2. Call `ModelsService.updateApiConfigurationPartial` to point Cline at our policy backend (Anthropic, vLLM, sglang, etc.).
     3. Call `TaskService.newTask` with dataset prompt and stream `UiService.subscribeToPartialMessage` for reasoning/tool use.
     4. Enforce time/toolcall limits; cancel via `TaskService.cancelTask` if necessary.
   - Collect artifacts (ui/api histories, repository diffs, metrics) per worker.
   - Signal workers to exit; Nomad tears down allocations and cleans ephemeral disks.

### 5. Local Development Harness (`cline_dev/`)

Before Nomad integration we will create a local test bench under `environments/cline_env/cline_dev/`:

1. Add per-profile `Dockerfile`s or Nix-based `dockerTools.buildLayer`.
2. Provide `compose.yml` (or tilt scripts) for quick ADR-style tests:
   - Example scenario: Rust task using `ratatui` repo and issue [#2148](https://github.com/ratatui/ratatui/issues/2148) “Add support for vertical gauges”.
   - Steps:
     1. Build `rust-env` image.
     2. Clone `ratatui` into `/workspace` and run `cargo test` to validate toolchain.
     3. Start Cline core inside the container using our launcher.
     4. From host orchestrator, run the gRPC smoke flow (configure Anthropic, call `newTask`, capture partial messages) to confirm end-to-end behaviour.
     5. Tear down container and inspect artifacts (logs, diffs).
3. Once this works manually, script it (`cline_dev/run_local_task.py`) to serve as a regression test.

### 6. Automation Components

Planned modules/files:

1. `environments/cline_env/cline_dev/`
   - `profiles/<lang>/flake.nix` or `Dockerfile`.
   - `bootstrap_cline_worker.sh`.
   - `compose.local.yml` for single-node testing.
2. `atroposlib/envs/cline_env/worker_manager.py`
   - Abstracts over Nomad jobs (submit, monitor, shutdown).
   - Handles local fallback when `use_nomad=False`.
3. `atroposlib/envs/cline_env/task_runner.py`
   - Integrates dataset intake, worker provisioning, gRPC driving, scoring, artifact upload.
4. `environments/cline_env/tests/` (future)
   - Pytests that mock the Nomad API and ensure we submit the expected job spec.

### 7. Next Actions

1. Implement the `rust-env` Nix profile + image (since `ratatui` is our first test case).
2. Create `cline_dev/examples/ratatui_vertical_gauge.md` describing the end-to-end flow.
3. Extend `ClineAgentEnv` to optionally call out to the worker manager (vs local gRPC).
4. Add W&B metrics for:
   - Environment prep time.
   - Container boot success/failure.
   - Trajectory outcomes, rewards, tokens.
5. After Rust flow works, add Python and Node profiles, then generalize the pipeline.

This plan keeps us focused on the Cline RL use case while establishing reusable building blocks (Nix profiles, Nomad job specs, gRPC automation) that we can later expose to other environments.
