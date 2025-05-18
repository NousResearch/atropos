# 2048 Game Environment Implementation

This document provides a structured overview of the 2048 game environment implementation created for reinforcement learning with language models.

## Architecture Overview

The implementation consists of four main components:

1. **Core Game Logic** (`Game2048`): Handles the game mechanics and state management
2. **Environment Wrapper** (`Environment2048`): Provides an RL-friendly interface for agents
3. **Sample Agents**: Implementations of random and heuristic-based players
4. **Runner Script**: Command-line interface for testing agents

## Component Details

### 1. Core Game Logic (`game_2048.py`)

**Class: `Game2048`**

Encapsulates the core 2048 game mechanics, including board representation, move logic, and scoring.

**Key Properties:**
- `id`: Unique game identifier
- `size`: Board dimensions (default 4×4)
- `board`: 2D NumPy array representing the game grid
- `score`: Current game score
- `moves`: Number of moves made
- `game_over`: Boolean indicating if the game is over

**Key Methods:**
- `__init__(size=4)`: Initialize a new game with optional board size
- `reset() -> np.ndarray`: Reset game to initial state, returns the board
- `move(direction) -> Tuple[np.ndarray, int, bool]`: Make a move in the specified direction, returns (board, score_added, changed)
- `get_state() -> Dict`: Get current game state as a dictionary
- `get_max_tile() -> int`: Get the highest tile value on the board
- `render() -> str`: Render the board as a formatted string

**Internal Methods:**
- `_add_random_tile()`: Add a new tile (2 or 4) to a random empty cell
- `_move_left() -> bool`: Apply left move logic, returns if board changed
- `_merge_sequence(sequence) -> Tuple[np.ndarray, int]`: Merge a sequence of numbers according to 2048 rules
- `_has_valid_moves() -> bool`: Check if any valid moves remain

### 2. Environment Wrapper (`env_2048.py`)

**Class: `Environment2048`**

Provides a reinforcement learning interface around the 2048 game, with observations, actions, rewards, and termination logic.

**Key Properties:**
- `game`: Instance of `Game2048`
- `winning_value`: Tile value required to win (default: 2048)
- `max_moves`: Maximum allowed moves (default: 1000)
- `system_message`: Instruction prompt for agents

**Key Methods:**
- `__init__(winning_value=2048, max_moves=1000)`: Initialize environment with configurable parameters
- `reset() -> Tuple[Dict, str]`: Reset the environment, returns (observation, prompt)
- `step(action) -> Tuple[Dict, str, bool, float, Dict]`: Process an agent action, returns (observation, prompt, done, reward, info)
- `render() -> str`: Generate a human-readable representation of the current state

**Internal Methods:**
- `_get_observation() -> Dict`: Create observation dictionary from game state
- `_get_prompt() -> str`: Generate text prompt for the agent
- `_parse_action(action) -> Optional[str]`: Parse direction from agent's response
- `_calculate_reward(...) -> float`: Calculate reward based on game state and action outcome

**Observation Dictionary:**
```python
{
    'id': str,               # Game identifier
    'board': List[List[int]], # 2D array representing the board
    'score': int,            # Current score
    'moves': int,            # Number of moves made
    'game_over': bool,       # Whether the game is over
    'max_tile': int,         # Highest tile value on the board
    'winning_value': int     # Target tile value for winning
}
```

**Reward Structure:**
- Successful moves: Proportional to score gained + small bonus for high tiles
- Invalid moves: -0.1
- Game won: +10.0
- Game lost: Based on progress toward winning value

### 3. Sample Agents (`run_2048.py`)

**Class: `RandomAgent`**

A simple agent that makes random valid moves.

**Key Methods:**
- `__init__()`: Initialize the agent
- `act(observation, prompt) -> str`: Generate a random action

**Class: `HeuristicAgent`**

A more sophisticated agent that uses a heuristic strategy to keep high values in a corner.

**Key Methods:**
- `__init__()`: Initialize the agent
- `act(observation, prompt) -> str`: Generate an action based on heuristic evaluation

**Heuristic Strategy Components:**
1. Position scoring: Encourages tiles to follow a monotonic pattern
2. Merge potential: Rewards board states with adjacent identical values
3. Score optimization: Considers the immediate score gain from a move

### 4. Runner Script (`run_2048.py`)

**Key Functions:**
- `run_episode(env, agent, max_steps=1000) -> Dict`: Run a single episode, returns statistics
- `main()`: Parse command-line arguments and run the specified agent

**Command-line Arguments:**
- `--agent`: Agent type to use (`random` or `heuristic`)
- `--episodes`: Number of episodes to run
- `--winning-value`: Tile value needed to win (default: 2048)

## Agent Interface

Agents should implement an asynchronous `act` method with the following signature:

```python
async def act(self, observation: Dict[str, Any], prompt: str) -> str
```

Where:
- `observation`: Dictionary containing the current game state
- `prompt`: String representation of the board
- Return value: Action string in XML format `<move>direction</move>`

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

## Agent Performance Comparison

We implemented and tested two agents:

### Random Agent:
- Steps to win: 91
- Total reward: 23.05
- Final score: 884

### Heuristic Agent:
- Steps to win: 74 
- Total reward: 21.91
- Final score: 812

The heuristic agent was more efficient, completing the game in fewer steps (74 vs 91), which demonstrates the effectiveness of the strategy to keep high-value tiles in a corner.

## Usage Instructions

To run the game with the heuristic agent:

```bash
python -m atropos.environments.hack0.run_2048 --agent heuristic --winning-value 128
```

To run with the random agent:

```bash
python -m atropos.environments.hack0.run_2048 --agent random --winning-value 128
``` 