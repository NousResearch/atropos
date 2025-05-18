# 2048 Game Environment

This directory contains a standalone implementation of the 2048 game that can be used for reinforcement learning with language models.

## Overview

The implementation includes:

1. A core game logic class (`Game2048`)
2. An environment wrapper (`Environment2048`) for agent interaction
3. Sample agents (random and heuristic)
4. A runner script to test the environment

## Files

- `game_2048.py`: Core game logic and mechanics
- `env_2048.py`: Environment class for agent interaction
- `run_2048.py`: Runner script with sample agents
- `__init__.py`: Package exports

## Usage

To run a simple test with the heuristic agent:

```bash
python -m atropos.environments.hack0.run_2048 --agent heuristic --winning-value 128
```

Options:

- `--agent`: Agent type to use (`random` or `heuristic`)
- `--episodes`: Number of episodes to run
- `--winning-value`: Tile value needed to win (default: 2048)

## Environment Interface

The environment follows a standard reinforcement learning interface:

```python
# Initialize environment
env = Environment2048(winning_value=2048)

# Reset environment
observation, prompt = env.reset()

# Take a step
action = "<move>left</move>"  # Agent's action
observation, prompt, done, reward, info = env.step(action)
```

## Agent Interface

Agents should implement an `act` method that takes the current observation and prompt, and returns an action string:

```python
async def act(self, observation, prompt):
    # Process the current game state
    # Return a move in XML format
    return "<move>left</move>"
```

## Game State

The game state includes:

- `board`: 4x4 grid of tile values (0 for empty cells)
- `score`: Current score
- `moves`: Number of moves taken
- `game_over`: Whether the game is over
- `max_tile`: Highest tile value on the board

## Rewards

The environment provides rewards based on:

- Score gained from merging tiles
- Progress toward the winning tile value
- Penalties for invalid moves
- Bonus for winning 