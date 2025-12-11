# Cline Environment

RL environment for training AI coding agents using the [Cline](https://github.com/cline/cline) AI coding assistant.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Atropos                                  │
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │ cline_agent_env │───▶│   Modal Worker  │                     │
│  │   (rollouts)    │    │  (task runner)  │                     │
│  └─────────────────┘    └────────┬────────┘                     │
│                                  │                               │
│                                  ▼                               │
│                         ┌───────────────────┐                   │
│                         │  Docker Container  │                   │
│                         │  ┌─────────────┐  │                   │
│                         │  │  Cline CLI  │  │                   │
│                         │  │  (npm pkg)  │  │                   │
│                         │  └─────────────┘  │                   │
│                         │  + Language Tools │                   │
│                         └───────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

### Cline CLI (`npm install -g cline`)
The official Cline CLI provides a simple interface for running coding tasks:
- `cline auth -p anthropic -k $API_KEY -m claude-sonnet-4-5-20250929` - Configure API
- `cline "task description" -y -o` - Run task in YOLO mode (auto-approve)
- Trajectories saved to `~/.cline/data/tasks/<task_id>/`

### Docker Images (`docker/`)
Language-specific containers built on a common base:
- **Base**: Node.js 22 + Cline CLI + Python 3.10
- **Language images**: Add compilers/runtimes (Rust, Go, Java, etc.)

### Modal Workers (`modal_worker.py`)
Serverless execution of Cline tasks on Modal:
- Runs Docker containers with language tooling
- Configures API credentials
- Collects trajectories for training

## Quick Start

### 1. Build Docker Images
```bash
cd environments/cline_env/docker

# Build base image
cd base && docker build -t nousresearch/cline-base:latest .

# Build language image (e.g., Python)
cd ../python && docker build -t nousresearch/cline-python:latest .
```

### 2. Run a Task Locally
```bash
docker run --rm \
  -e "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY" \
  -e "CLINE_TASK=Create a Python script that calculates fibonacci" \
  -v $(pwd)/output:/output \
  nousresearch/cline-python:latest
```

### 3. Run on Modal
```bash
# Install modal
pip install modal

# Deploy and run
modal run environments/cline_env/modal_worker.py
```

## File Structure

```
cline_env/
├── README.md                    # This file
├── cline_agent_env.py          # Main RL environment
├── modal_worker.py             # Modal serverless worker
├── scoring.py                  # Task success evaluation
├── profile_registry.py         # Language profile configuration
├── envs_required.csv           # Dataset task counts by language
├── dump_trajectories.py        # Extract training data
├── dump_trajectories_modal.py  # Modal trajectory extraction
├── launch_rejection_sampling.py # Rejection sampling runner
├── docker/                     # Docker images
│   ├── base/                   # Base image with Cline CLI
│   ├── python/                 # Python environment
│   ├── rust/                   # Rust environment
│   └── ...                     # Other languages
├── data/                       # Trajectory data files
└── cline/                      # Cline source (submodule, reference only)
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Anthropic API key | Required |
| `OPENAI_API_KEY` | For custom vLLM endpoints | - |
| `OPENAI_BASE_URL` | Custom model endpoint | - |
| `CLINE_MODEL` | Model to use | `claude-sonnet-4-5-20250929` |
| `CLINE_TASK` | Task prompt | - |

## Training Data Format

Trajectories are stored in `~/.cline/data/tasks/<task_id>/`:

- **`api_conversation_history.json`** - Full conversation with tool calls (for training)
- **`ui_messages.json`** - UI events and checkpoints
- **`task_metadata.json`** - Model usage, files modified

## Dataset Statistics

See `envs_required.csv` for task counts by language. Top languages:
- Python: 2,020 tasks
- TypeScript: 1,546 tasks
- Go: 1,143 tasks
- Rust: 1,042 tasks
- C++: 1,028 tasks

## Related Documentation

- [Docker Images README](docker/README.md)
- [Atropos Main README](../../README.md)
- [Cline CLI Docs](https://github.com/cline/cline)
