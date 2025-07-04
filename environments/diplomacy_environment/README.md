# Diplomacy Environment for Atropos

A reinforcement learning training environment for Diplomacy that integrates with Atropos using AI_Diplomacy's mature game engine and visualization tools.

## Overview

This environment allows you to:
- Train RL policies against strong LLM opponents (GPT-4, Claude, etc.)
- Visualize games with beautiful 3D animations
- Run mixed-agent games with humans, LLMs, and RL policies
- Leverage AI_Diplomacy's battle-tested game engine

## Architecture

```
Atropos RL Framework
        ↓
DiplomacyEnvNoThinking (Atropos Environment)
        ↓
AtroposClient (LLM Client Proxy)
        ↓
AI_Diplomacy Game Engine
        ↓
Mixed Agents (RL Policies, LLMs, Humans)
```

## Understanding the ServerManager System

### Key Concept: ServerManager is NOT Just for "Atropos Servers"

The **ServerManager** is a universal API gateway that works with ANY OpenAI-compatible API. This is a common misconception - despite the naming, it's designed to work with:

- **OpenAI** (GPT-4, GPT-3.5-turbo, etc.)
- **Anthropic Claude** (via OpenAI-compatible proxy)
- **Local vLLM servers**
- **Hugging Face Inference Endpoints**
- **Any server implementing the OpenAI API format**

### How ServerManager Works

When you create an Atropos environment, it automatically creates a ServerManager:

```python
# In BaseEnv.__init__:
self.server = ServerManager(
    server_configs,  # List of APIServerConfig objects
    slurm=slurm,
    testing=testing,
    server_class=self.server_cls
)
```

The ServerManager:
1. Takes a list of `APIServerConfig` objects
2. Checks the `server_type` field (defaults to "openai")
3. Creates appropriate server instances (`OpenAIServer` for OpenAI, `TrlVllmServer` for vLLM, etc.)
4. Provides unified `completion()` and `chat_completion()` methods
5. Handles load balancing across multiple servers

### APIServerConfig Examples

#### Using OpenAI Directly (Most Common)
```python
from atroposlib.envs.base import APIServerConfig

# This connects directly to OpenAI's API
server_configs = [
    APIServerConfig(
        model_name="gpt-4o-mini",  # Must be a valid OpenAI model
        base_url="https://api.openai.com/v1",  # OpenAI's endpoint
        api_key=os.getenv("OPENAI_API_KEY"),
        server_type="openai"  # This is the default
    )
]
```

#### Using a Local vLLM Server
```python
server_configs = [
    APIServerConfig(
        model_name="meta-llama/Llama-2-7b-hf",  # Whatever model you loaded
        base_url="http://localhost:8000/v1",  # vLLM implements OpenAI API
        api_key="dummy",  # vLLM doesn't need a real key
        server_type="openai"  # vLLM is OpenAI-compatible
    )
]
```

#### Using Multiple Servers (Load Balancing)
```python
server_configs = [
    # Primary server
    APIServerConfig(
        model_name="gpt-4",
        base_url="https://api.openai.com/v1",
        api_key=os.getenv("OPENAI_API_KEY")
    ),
    # Backup server
    APIServerConfig(
        model_name="gpt-3.5-turbo",
        base_url="https://api.openai.com/v1",
        api_key=os.getenv("OPENAI_API_KEY")
    )
]
# ServerManager will automatically load balance between them
```

### AtroposClient: The Adapter Pattern

The `AtroposClient` implements AI_Diplomacy's `BaseModelClient` interface and forwards requests to whatever server you configure:

```python
# AtroposClient forwards to ANY OpenAI-compatible endpoint
client = AtroposClient(
    model_name="my-model",  # This gets sent in the API request
    server_url="http://localhost:8000"  # Can be OpenAI, vLLM, etc.
)

# When AI_Diplomacy calls:
response = await client.generate_response(prompt)
# AtroposClient POSTs to: {server_url}/v1/completions with model=my-model
```

**CRITICAL**: The `model_name` in AtroposClient is what gets sent to the API, so it must be valid for your provider!

### Common Pitfalls and Solutions

#### Pitfall 1: "Model not found" errors
**Problem**: `The model 'atropos-training-policy' does not exist`

**Solution**: Use valid model names for your provider:
```python
# ❌ Wrong - OpenAI doesn't know this model
APIServerConfig(model_name="training-policy", ...)

# ✅ Correct - Valid OpenAI model
APIServerConfig(model_name="gpt-4o-mini", ...)
```

#### Pitfall 2: Thinking AtroposClient only works with "Atropos servers"
**Reality**: AtroposClient forwards to ANY server you configure. If you set `base_url="https://api.openai.com/v1"`, it talks directly to OpenAI.

#### Pitfall 3: Intercepting clients calling parent methods
**Problem**: In GRPO, calling `super().generate_response()` uses the wrong model name

**Solution**: Always use the environment's ServerManager:
```python
# ❌ Wrong
return await super().generate_response(prompt)  # Uses self.model_name

# ✅ Correct
completion = await self.env.server.completion(
    prompt=prompt,
    model=self.env.server_configs[0].model_name,  # Use real model
    ...
)
```

## Quick Start

### 1. Install Dependencies

```bash
# Initialize AI_Diplomacy submodule
git submodule update --init --recursive

# Install AI_Diplomacy
cd environments/diplomacy_environment/AI_Diplomacy
uv pip install -e .
cd ../../..

# Install Node.js dependencies for visualization (if using)
cd environments/diplomacy_environment/AI_Diplomacy/ai_animation
npm install
cd ../../../..
```

### 2. Run a Test Game

```bash
cd environments/diplomacy_environment

# Quick test (2 turns only)
uv run python diplomacy_local_server_no_thinking.py --quick

# Standard test (10 turns)
uv run python diplomacy_local_server_no_thinking.py

# Custom length
uv run python diplomacy_local_server_no_thinking.py --max-turns 20
```

This will:
- Start a mock Atropos policy server automatically
- Launch the Diplomacy web server on port 8432
- Run a game with France as the training agent vs 6 LLM opponents
- Save game logs for visualization

## Web UI and Visualization

AI_Diplomacy provides two powerful web interfaces:

### 1. AI Animation Interface (Three.js Visualization)

A beautiful 3D visualization for watching completed games with animations.

**Start the interface:**
```bash
cd environments/diplomacy_environment
./start_diplomacy_ui.sh
# Opens at http://localhost:5173
```

**Features:**
- **Animated Unit Movements**: Watch armies and fleets move across the board
- **Message Display**: See AI negotiations unfold in real-time
- **Phase Progression**: Step through game phases with summaries
- **Two-Power Conversations**: High-interest diplomatic moments highlighted
- **Victory Animations**: Celebration when a power wins
- **Debug Tools**: Province highlighting, next moment display

**Loading Games:**
1. Click "Load Game" in the interface
2. Navigate to a game JSON file (e.g., `./game_logs/game-xxxxx.json`)
3. Use Play button for automatic playback or Next/Previous for manual control

### 2. Diplomacy Game Server (Interactive Play)

The standard Diplomacy server for real-time games and monitoring.

**Start the server:**
```bash
cd environments/diplomacy_environment/AI_Diplomacy
uv run python -m diplomacy.server.run --port 8432
# Opens at http://localhost:8432
```

**Features:**
- Create and join games
- Watch ongoing games in real-time
- Submit orders manually
- View game state and history

## Visualization Features

### Game Replay Animation

The AI Animation interface provides rich visualization:

1. **Conversation Animation**:
   - Messages appear word-by-word with typing effect
   - Power-specific positioning on the map
   - Global broadcast messages in news banner

2. **Unit Movement Animation**:
   - Smooth transitions between provinces
   - Support moves shown with arrows
   - Failed moves indicated visually

3. **Phase Summaries**:
   - Text-to-speech narration (requires ElevenLabs API key)
   - Success/failure categorization
   - Strategic analysis display

4. **Agent State Visualization**:
   - Goals and objectives for each power
   - Relationship tracking (Enemy → Ally scale)
   - Private journal entries

### Debug Tools

Enable debug mode by setting `VITE_DEBUG_MODE=true`:

- **Province Highlighter**: Click provinces to highlight them
- **Next Moment Display**: See upcoming high-interest events
- **Phase Information**: Detailed phase parsing and timing

### Data Formats

**Game JSON Structure:**
```json
{
  "phases": [{
    "name": "S1901M",
    "messages": [...],
    "orders": {...},
    "state": {
      "units": {...},
      "centers": {...}
    }
  }],
  "powers": {
    "FRANCE": {
      "goals": ["Secure Belgium", "Alliance with England"],
      "relationships": {
        "ENGLAND": "Friendly",
        "GERMANY": "Unfriendly"
      },
      "journal": ["Planning defensive strategy..."]
    }
  }
}
```

**Moments JSON (High-Interest Events):**
```json
{
  "moments": [{
    "phase_name": "F1902M",
    "category": "Betrayal",
    "powers_involved": ["FRANCE", "ENGLAND"],
    "interest_score": 9.5
  }],
  "power_models": {
    "FRANCE": "gpt-4o",
    "ENGLAND": "claude-3"
  }
}
```

## Usage Examples

### Watch a Completed Game

```bash
# 1. Run a game
uv run python diplomacy_local_server_no_thinking.py --quick

# 2. Start visualization
./start_diplomacy_ui.sh

# 3. Load the game JSON from ./game_logs/ in the web interface
# 4. Click Play to watch the animated replay
```

### Monitor a Game in Progress

```bash
# 1. Start the game server (if not already running)
uv run python -m diplomacy.server.run --port 8432

# 2. Run your game (it will connect to the server)
uv run python diplomacy_local_server_no_thinking.py

# 3. Open http://localhost:8432 to watch in real-time
# 4. After completion, use AI Animation interface for replay
```

### Run with Custom Agents

```python
# In your test script
powers_config = {
    "FRANCE": PowerConfig(type="atropos", model="my-policy", is_training=True),
    "ENGLAND": PowerConfig(type="llm", model="gpt-4o", is_training=False),
    "GERMANY": PowerConfig(type="human", player_id="player1", is_training=False),
    # ... other powers
}
```

## Environment Configuration

Key configuration options in `DiplomacyEnvNoThinkingConfig`:

- `max_game_turns`: Maximum game length (default: 20)
- `game_deadline_seconds`: Time limit per phase (default: 300)
- `launch_web_server`: Auto-start game server (default: True)
- `powers_config`: Agent configuration for each power
- `save_game_logs`: Save games for visualization (default: True)

## Training Integration

### Standalone Training

```bash
# Generate training data
uv run python -m environments.diplomacy_environment.diplomacy_env_no_thinking process

# Serve environment for online RL
uv run python -m environments.diplomacy_environment.diplomacy_env_no_thinking serve
```

### Python API

```python
from environments.diplomacy_environment.diplomacy_env_no_thinking import (
    DiplomacyEnvNoThinking,
    DiplomacyEnvNoThinkingConfig,
    PowerConfig
)

# Configure environment
config = DiplomacyEnvNoThinkingConfig(
    powers_config={
        "FRANCE": PowerConfig(type="atropos", model="my-policy", is_training=True),
        # ... other powers
    }
)

# Create environment
env = DiplomacyEnvNoThinking(config, server_configs)

# Run trajectory collection
trajectory = await env.collect_trajectory(item)
```

## Troubleshooting

### Web UI Issues

**AI Animation won't start:**
- Ensure Node.js is installed
- Run `npm install` in the `ai_animation` directory
- Check for port conflicts on 5173

**Game won't load:**
- Verify game JSON exists in the path you're trying to load
- Check browser console for errors
- Ensure JSON format matches expected structure

**No animations playing:**
- Enable instant mode for faster testing: `VITE_INSTANT_MODE=true`
- Check that phases have orders to animate
- Verify unit positions in game state

### Server Issues

**Port already in use:**
```bash
# Find process using port 8432
lsof -i :8432
# Kill the process if needed
kill -9 <PID>
```

**Connection refused:**
- Ensure mock_atropos_server.py is running (or your real policy server)
- Check firewall settings
- Verify server URL configuration

**Game hangs:**
- Reduce `game_deadline_seconds` for faster testing
- Use `--quick` flag for 2-turn games
- Check API rate limits if using many LLM agents

### API Key Issues

**Missing API keys:**
Set environment variables:
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"  # If using Claude
export ELEVENLABS_API_KEY="your-key"  # For text-to-speech
```

## Advanced Features

### Custom Visualizations

The AI Animation interface supports custom styling:
- Edit `ai_animation/src/config.ts` for colors and timing
- Modify `ai_animation/src/map/mapConstants.ts` for province positions
- Add custom animations in `ai_animation/src/units/animate.ts`

### Analysis Tools

AI_Diplomacy includes analysis scripts:
- `analyze_game_moments_llm.py`: Find interesting diplomatic moments
- `analyze_lies_focused.py`: Detect deception in negotiations
- `analyze_game_results.py`: Statistical analysis of outcomes

### Performance Optimization

For faster training iterations:
- Use fewer LLM opponents (replace with simpler bots)
- Reduce `max_game_turns` to focus on early game
- Disable `save_game_logs` if visualization not needed
- Run multiple games in parallel with different seeds

## Contributing

When adding features:
1. Test with both web interfaces
2. Ensure game JSON format compatibility
3. Update visualization if adding new game elements
4. Document any new configuration options

## GRPO Training with LaTRo Rewards

The Diplomacy environment now supports GRPO (Group Relative Policy Optimization) with LaTRo (Latent Reasoning Optimization) rewards based on https://arxiv.org/html/2411.04282v2.

### Key Features

1. **Best-of-N Selection**: Sample multiple candidate responses and select the best using LaTRo scores
2. **Log Probability Scoring**: Uses model confidence (logprobs) to score responses
3. **Unified AtroposClient**: Single client handles both normal play and GRPO training modes

### Configuration

```python
from environments.diplomacy_environment.diplomacy_env_grpo import (
    DiplomacyEnvGRPO,
    DiplomacyEnvGRPOConfig
)

config = DiplomacyEnvGRPOConfig(
    # GRPO settings
    group_size=4,  # Sample 4 candidates per decision
    use_latro_rewards=True,  # Enable LaTRo scoring
    temperature=0.7,  # Sampling temperature

    # Training configuration
    training_power="FRANCE",  # Which power to train
    tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",

    # Opponent configuration (use strong models)
    opponent_models={
        "ENGLAND": "gpt-4o-mini",
        "GERMANY": "gpt-4o-mini",
        # ... other powers
    }
)
```

### Server Requirements

LaTRo rewards require an API that returns log probabilities:

```python
# OpenAI (native support)
server_configs = [
    APIServerConfig(
        model_name="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
]

# vLLM (full OpenAI compatibility)
server_configs = [
    APIServerConfig(
        model_name="meta-llama/Llama-3-8B",
        base_url="http://localhost:8000/v1",
        api_key="dummy",
    )
]

# llama.cpp (recent versions with PR #10783)
server_configs = [
    APIServerConfig(
        model_name="llama-3-8b",
        base_url="http://localhost:8080/v1",
        api_key="dummy",
    )
]
```

**Note**: Ollama's current OpenAI compatibility doesn't include logprobs. Use vLLM or llama.cpp for local models.

### Running GRPO Training

```bash
# Test the implementation
cd environments/diplomacy_environment
uv run python test_latro_rewards.py

# Generate training data with GRPO
uv run python -m environments.diplomacy_environment.diplomacy_env_grpo process \
    --config config.yaml

# Online training
uv run python -m environments.diplomacy_environment.diplomacy_env_grpo serve \
    --config config.yaml
```

### How It Works

1. **Intercept LLM Calls**: AtroposClient intercepts all generate_response calls from AI_Diplomacy
2. **Sample Candidates**: Generates N responses using the policy model
3. **Score with LaTRo**: Computes r(z) = Σ log p(z_i|z_<i) for each candidate
4. **Compute Advantages**: A_k = r(z_k) - mean(r(z_j))
5. **Select Best**: Choose response with highest advantage
6. **Store Trajectory**: Save all candidates with normalized scores for training

### Monitoring

Enable debug logging to see LaTRo scores:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This will show:
- Number of alternatives sampled
- Raw log probability scores
- Normalized advantages
- Selected response index

## License

This integration follows the licenses of both Atropos and AI_Diplomacy (AGPL-3.0).
