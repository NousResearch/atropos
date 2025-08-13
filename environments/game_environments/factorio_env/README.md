# Factorio Learning Environment for Atropos

This directory contains the Factorio Learning Environment (FLE) integration for the Atropos RL trainer.

## Setup

### 1. Clone with Submodules & Install Dependencies
```bash
# If you haven't cloned with submodules yet
git submodule update --init --recursive

# Install FLE in editable mode
cd environments/game_environments/factorio_env
pip install -e ./fle
pip install -r requirements.txt
```

### 2. Build Docker Image
```bash
# IMPORTANT: Must build the Docker image first!
cd fle/fle/cluster/docker

# On macOS (Apple Silicon), must force linux/amd64 platform for Rosetta
docker build -t factorio . --platform linux/amd64

# On Linux/Intel
docker build -t factorio .
```

### 3. Start Factorio Server
```bash
# Return to factorio_env directory
cd ../../../

# Start a single Factorio container
docker-compose up -d factorio_0

# Or use FLE's cluster management (after building image)
uv run fle cluster start
```

### 3. Start LLM Server (for agent)
```bash
# Using llama.cpp server or similar
llama-server -m <model_path> --port 8080
```

## Components

### Agents
- `llama_agent.py` - Task-based agent with self-planning capabilities
- `llama_agent_open_play.py` - Open-ended exploration agent

### Key Features
- **Self-Planning**: Agent uses `update_goals` to manage its own objectives
- **Task Support**: Works with FLE's throughput tasks (iron_ore, copper_plate, etc.)
- **Tool Integration**: Full access to FLE's tool system (nearest, place_entity, etc.)
- **Observation Parsing**: Tracks inventory, entities, research, and throughput

## Usage

### Run Task-Based Agent
```bash
python llama_agent.py
```
This will:
1. Load the iron_ore_throughput task
2. Prompt the agent to create goals
3. Execute the agent's plan
4. Track throughput progress

### Run Open Play Agent
```bash
python llama_agent_open_play.py
```
This provides open-ended exploration with custom goals.

## Environment Tasks

Available tasks include:
- `iron_ore_throughput` - Mine 16 iron ore/minute
- `copper_ore_throughput` - Mine copper ore
- `iron_plate_throughput` - Smelt iron plates
- `automation_science_pack_throughput` - Craft science packs
- And many more...

## Configuration

Add the following environment variables to the main Atropos `.env` file (in project root):
```bash
# LLM API Keys (if using API-based models)
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

# Factorio server configuration (optional)
FACTORIO_RCON_PASSWORD=factorio
FACTORIO_TCP_PORT=27000
FACTORIO_GAME_PORT=34197

# Docker platform override (for Apple Silicon Macs)
DOCKER_PLATFORM=linux/amd64
```

Note: The `.env` file should be in the root Atropos directory, not in the factorio_env folder.

## Docker Setup

The `docker-compose.yml` provides:
- Factorio server on port 34197 (game) and 27015 (RCON)
- Volume mounts for scenarios and mods
- Resource limits (1 CPU, 1GB RAM)

**Important Notes:**
- The Docker image MUST be built before running containers
- On macOS with Apple Silicon, use `--platform linux/amd64` to ensure Rosetta compatibility
- The FLE Dockerfile is located at `fle/fle/cluster/docker/Dockerfile`

## Integration with Atropos

TODO: Integration with Atropos training loop
- Environment wrapper for Atropos
- Reward shaping
- Curriculum learning
- Multi-agent support
