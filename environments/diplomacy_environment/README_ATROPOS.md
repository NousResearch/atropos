# Diplomacy Environment for Atropos

This is the Atropos-integrated Diplomacy training environment that supports mixed-agent games where your RL policy can train against strong LLM opponents.

## Features

- **Mixed Agent Games**: Train your policy against Claude, GPT-4, Gemini, and other strong LLMs
- **Flexible Configuration**: Choose which powers are controlled by training policies vs opponents
- **Web Interface**: Watch games in real-time at http://localhost:8432
- **Self-Play Support**: Can configure multiple powers to use the same training policy
- **Human Players**: Support for human opponents via the web interface
- **Comprehensive Scoring**: Based on supply centers, survival, and victory

## Quick Start

### 1. Ensure Prerequisites

```bash
# Initialize AI_Diplomacy submodule if needed
git submodule update --init --recursive

# Install AI_Diplomacy dependencies
cd environments/diplomacy_environment/AI_Diplomacy
uv pip install -e .
cd ../../..
```

### 2. Start Mock Policy Server (for testing)

```bash
cd environments/diplomacy_environment
uv run python mock_atropos_server.py
```

### 3. Run a Test Game

```bash
# Run a single game with France as training agent vs 6 LLM opponents
uv run python diplomacy_local_server_no_thinking.py
```

### 4. Watch the Game

Open http://localhost:8432 in your browser to watch the game progress in real-time.

## Configuration

### Power Configuration

Configure which agents control which powers in `powers_config`:

```python
powers_config = {
    # Training agent (contributes to RL training)
    "FRANCE": {
        "type": "atropos",
        "model": "training-policy",
        "is_training": True
    },

    # Strong LLM opponent
    "ENGLAND": {
        "type": "llm",
        "model": "claude-3-5-sonnet-20241022",
        "is_training": False
    },

    # Self-play (another instance of training policy)
    "GERMANY": {
        "type": "atropos",
        "model": "training-policy",
        "is_training": True
    },

    # Human player (via web interface)
    "ITALY": {
        "type": "human",
        "player_id": "player1",
        "is_training": False
    }
}
```

### Scoring System

- **Supply Center Score**: (final_centers - starting_centers) / 10
- **Survival Bonus**: 0.1 points per turn survived
- **Victory Bonus**: 10 points for winning the game

## Training Integration

### For Data Generation

```bash
# Generate training data with process command
uv run python -m environments.diplomacy_environment.diplomacy_env_no_thinking process \
    --env.max_game_turns=20 \
    --env.powers_config.FRANCE.is_training=true \
    --env.powers_config.GERMANY.is_training=true
```

### For Online Training

```bash
# Serve environment for online RL training
uv run python -m environments.diplomacy_environment.diplomacy_env_no_thinking serve \
    --env.rollout_server_url=http://localhost:8000
```

## Architecture

```
Atropos RL Trainer
       ↓
DiplomacyEnvNoThinking (collect_trajectory)
       ↓
AI_Diplomacy Game Engine (run_llm_game)
       ↓
Mixed Agents:
- AtroposClient → Policy Server (training agents)
- LLM Clients → OpenAI/Anthropic/Google APIs
- Human Interface → Web UI
```

## Environment Variables

Set these for LLM opponents:
- `OPENAI_API_KEY` - For GPT models
- `ANTHROPIC_API_KEY` - For Claude models
- `GOOGLE_API_KEY` - For Gemini models
- `ATROPOS_SERVER_URL` - URL of your policy server (default: http://localhost:8000)

## Game Logs

Games are saved to `./game_logs/` (or configured directory) as JSON files that can be:
- Loaded in the web UI for replay
- Analyzed for strategy insights
- Used for debugging

## Troubleshooting

### "Connection refused" to policy server
- Ensure mock_atropos_server.py is running
- Check ATROPOS_SERVER_URL environment variable

### LLM API errors
- Verify API keys are set correctly
- Check rate limits and quotas
- Consider using cheaper/faster models for testing

### Web interface not loading
- Check if port 8432 is already in use
- Ensure AI_Diplomacy is properly installed
- Look for errors in console output

## Advanced Configuration

### Different Opponent Sets for Evaluation

```python
eval_opponent_models = [
    "claude-3-5-sonnet-20241022",
    "gpt-4o",
    "gemini-2.0-flash-exp",
    "o3-mini"
]
```

### Adjust Game Parameters

```python
max_game_turns = 30  # Longer games
game_deadline_seconds = 600  # 10 minutes per phase
survival_bonus = 0.2  # Higher survival reward
```

### Enable Distributed Training

Configure multiple training powers across different nodes:
```python
# Node 1: France and England
# Node 2: Germany and Austria
# etc.
```

## Next Steps

1. Implement real policy server to replace mock
2. Add trajectory interception for proper message collection
3. Integrate with Atropos training loop
4. Add metrics for negotiation quality
5. Support for press/no-press variants
