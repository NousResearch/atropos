# 2048 Game Environment for Atropos

A reinforcement learning environment implementation of the classic 2048 sliding tile game for training language models through the Atropos framework.

## Overview

This environment provides a complete implementation of the 2048 game for reinforcement learning with LLMs. It includes:

- Core game logic with standard 2048 rules
- RL environment wrapper for agent interaction
- Atropos integration for distributed training
- Example agents and visualization tools

## Components

| File | Description |
|------|-------------|
| `game_2048.py` | Core game mechanics and logic |
| `env_2048.py` | RL environment wrapper for agent interaction |
| `atropos_2048.py` | Atropos environment implementation |
| `atropos_2048_train.py` | Training script with LLM integration |
| `run_2048.py` | Runner for testing agents |
| `data/` | HTML visualizations of game playouts |

## Usage Examples

### Running with Random Agent

Test the environment with a random agent:

```bash
python run_2048.py --agent random --episodes 5
```

### Running with Heuristic Agent

Test with the built-in heuristic agent:

```bash
python run_2048.py --agent heuristic --winning-value 128 --episodes 10
```

### Running with LLM Agent

Use an LLM to play the game (requires a running inference server):

```bash
python run_2048.py --agent llm --model Qwen/Qwen2.5-1.5B-Instruct
```

### Training with Atropos

1. Start the Atropos API server in a separate terminal:
   ```bash
   run-api
   ```

2. Start the 2048 environment:
   ```bash
   python atropos_2048.py serve --config ../configs/2048.yaml
   ```

3. Monitor training progress with Weights & Biases or the built-in visualizations.

## Game Logic

The 2048 game follows standard rules:

- 4x4 grid of tiles
- Swipe in one of four directions: up, down, left, right
- Tiles slide as far as possible in the chosen direction
- Tiles with the same value merge when they collide
- After each move, a new tile (2 or 4) appears in a random empty cell
- Game is won when a tile with value 2048 is created
- Game is lost when no more moves are possible

## Environment Interface

The environment provides a standard RL interface:

```python
# Initialize
env = Environment2048(winning_value=2048)

# Reset for new episode
observation, prompt = env.reset()

# Take action and get results
action = "<move>left</move>"
observation, prompt, done, reward, info = env.step(action)
```

## LLM Agent Format

LLM agents should return their moves in XML format:

```
<move>direction</move>
```

Where `direction` is one of: `left`, `right`, `up`, `down`

## Reward Design

The environment provides the following rewards:

- **Merging tiles**: Reward proportional to score gained from merges
- **Progress reward**: Small bonus based on highest tile achieved
- **Invalid move penalty**: -0.1 for moves that don't change the board
- **Win bonus**: +10.0 for reaching the winning tile
- **Loss penalty**: Scaled based on progress toward winning tile

## Curriculum Learning

The environment supports curriculum learning by adjusting the winning threshold:

```python
# Start with easier goals
env = Environment2048(winning_value=128)

# Gradually increase difficulty
env = Environment2048(winning_value=256)
env = Environment2048(winning_value=512)

# Full game
env = Environment2048(winning_value=2048)
```

## Configuration

When using with Atropos training, you can customize the environment through the `2048.yaml` configuration file:

```yaml
env:
  group_size: 16  # Number of parallel environments
  tokenizer_name: "Qwen/Qwen2.5-1.5B-Instruct"  # Model tokenizer
  use_wandb: true  # Enable Weights & Biases logging
  # Additional parameters...

openai:
  - model_name: "Qwen/Qwen2.5-1.5B-Instruct"  # Model to use
    base_url: "http://localhost:9001/v1"  # Inference server URL
    # Additional server parameters...
```

## Performance Tracking

The environment tracks and reports several metrics:

- **Win rate**: Percentage of games where winning tile was achieved
- **Average max tile**: Average highest tile value across games
- **Average score**: Average game score
- **Average moves**: Average number of moves per game

## Visualization

Game states can be visualized as text:

```
Game ID: abcd1234 | Score: 2048 | Moves: 150 | Max Tile: 512

|   2  |   4  |   2  |  256 |
|   4  |  16  |  32  |  128 |
|   8  |  64  |   4  |  ___ |
|  16  |  32  |   2  |   4  |
```

HTML visualizations are also available in the `data/` directory for more detailed inspection of game play.

## Integration with Existing Codebase

This environment extends the Atropos framework, demonstrating how to create custom environments for specific tasks or games. The implementation follows Atropos conventions:

- `BaseEnv` extension for environment logic
- `config_init` for configuration
- Trajectory collection and evaluation methods
- WandB integration for metrics tracking 