# Cline Docker Images

Docker images for running the Cline AI coding agent with various language environments.

## Architecture

Uses the official `cline` CLI (npm package) for simplicity and maintainability:
- **No fork needed** - uses upstream Cline CLI
- **Auto-updates** - `npm install -g cline` pulls latest
- **Full trajectory logging** - saved to `~/.cline/data/tasks/`

```
Base Image (cline-base)
├── Node.js 22 + Cline CLI
├── Python 3.10
├── Common dev tools
└── Entrypoint for task execution
    │
    ├── cline-python (+ Python dev tools)
    ├── cline-rust (+ Rust/Cargo)
    ├── cline-node (+ Node dev tools)
    ├── cline-go (+ Go toolchain)
    ├── cline-java (+ JDK/Maven)
    └── ...more languages
```

## Quick Start

### Build Images

```bash
# Build base image
cd environments/cline_env/docker/base
docker build -t nousresearch/cline-base:latest .

# Build language-specific image
cd ../python
docker build -t nousresearch/cline-python:latest .
```

### Run a Task

```bash
# Simple task with Anthropic API
docker run --rm \
  -e "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY" \
  -e "CLINE_TASK=Create a file /workspace/hello.py that prints Hello World" \
  nousresearch/cline-python:latest

# With trajectory output
docker run --rm \
  -e "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY" \
  -e "CLINE_TASK=Create a Python script that calculates fibonacci" \
  -v $(pwd)/output:/output \
  nousresearch/cline-python:latest

# Interactive shell
docker run --rm -it \
  -e "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY" \
  nousresearch/cline-python:latest bash
```

### Use Custom Model (RL Training)

For pointing at your own vLLM server:

```bash
docker run --rm \
  -e "OPENAI_API_KEY=fake" \
  -e "OPENAI_BASE_URL=http://your-vllm-server:8000/v1" \
  -e "CLINE_MODEL=your-trained-model" \
  -e "CLINE_TASK=..." \
  nousresearch/cline-python:latest
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `OPENAI_API_KEY` | OpenAI/vLLM API key | - |
| `OPENAI_BASE_URL` | Custom OpenAI-compatible endpoint | - |
| `CLINE_MODEL` | Model ID to use | `claude-sonnet-4-5-20250929` |
| `CLINE_TASK` | Task prompt to execute | - |

## Trajectory Output

When a task runs, the full trajectory is saved:
- **Location**: `~/.cline/data/tasks/<task_id>/`
- **Files**:
  - `api_conversation_history.json` - Full API conversation (for training)
  - `ui_messages.json` - UI events and tool calls
  - `task_metadata.json` - Model usage, files touched

Mount `/output` to copy trajectories out:
```bash
docker run --rm -v $(pwd)/trajectories:/output ...
```

## Building All Images

```bash
./build_images.sh
```

## Available Images

| Image | Description |
|-------|-------------|
| `cline-base` | Base with Cline CLI + Python |
| `cline-python` | Python development |
| `cline-rust` | Rust + Cargo |
| `cline-node` | Node.js development |
| `cline-go` | Go toolchain |
| `cline-java` | JDK + Maven/Gradle |
| `cline-cpp` | C++ with GCC/Clang |
| `cline-c` | C development |
| `cline-shell` | Shell scripting |
