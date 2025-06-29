# Checkers Environment

A training environment for teaching LLMs to play checkers (draughts) using the Atropos framework. This environment implements standard American checkers rules and provides configurable opponents for training and evaluation.

## Overview

This environment enables reinforcement learning from AI feedback (RLAIF) for the classic board game of checkers. The LLM learns to make strategic moves using tool-based actions while playing against configurable opponents.

### Features

- **Complete Checkers Implementation**: Full 8x8 board with American rules
- **Mandatory Jump Logic**: Proper implementation of forced captures and multiple jumps
- **King Promotion**: Pieces promote to kings when reaching the opposite end
- **Tool-Based Interface**: LLM uses structured function calls to make moves
- **Configurable Opponents**: Random or basic AI opponents for varied training
- **Thinking Mode**: Optional `<think>` tags for strategy analysis
- **Comprehensive Metrics**: Win rates, move validity, game length tracking
- **Weights & Biases Integration**: Full logging and visualization support

## Game Rules

The environment implements standard American checkers rules:

1. **Board**: 8x8 board with pieces on dark squares only
2. **Movement**: Pieces move diagonally forward; kings move in all diagonal directions
3. **Captures**: Jump over opponent pieces to capture them (mandatory when available)
4. **Multiple Jumps**: Multiple captures in one turn are required when possible
5. **Promotion**: Pieces become kings when reaching the opposite end of the board
6. **Victory**: Win by capturing all opponent pieces or blocking all moves

## Quick Start

### Basic Usage

```python
from environments.community.checkers.checkers_env import CheckersEnv

# Initialize with default configuration
config, server_configs = CheckersEnv.config_init()
env = CheckersEnv(config, server_configs)

# Setup and run training
await env.setup()
item = await env.get_next_item()
trajectory, _ = await env.collect_trajectory(item)
```

### Configuration Options

```python
from environments.community.checkers.checkers_env import CheckersEnvConfig, OpponentType, Player

config = CheckersEnvConfig(
    # Game settings
    opponent_type=OpponentType.BASIC_AI,  # or OpponentType.RANDOM
    ai_plays_as=Player.RED,               # or Player.BLACK
    max_episode_turns=100,

    # Training settings
    thinking_enabled=True,
    temperature=0.7,
    eval_episodes=50,
    format_gated_rewards=False,           # Set to True for format-gated rewards

    # Standard BaseEnvConfig options
    group_size=16,
    max_token_length=4096,
    use_wandb=True,
)
```

## Environment Architecture

### Core Classes

- **`CheckersBoard`**: Game state management and rule enforcement
- **`Move`**: Represents chess moves with capture tracking
- **`CheckersEnv`**: Main environment class extending `BaseEnv`
- **`CheckersEnvConfig`**: Configuration with checkers-specific options

### Action Interface

The LLM interacts with the game through a single tool:

```json
{
    "name": "make_move",
    "description": "Make a move in checkers by specifying source and destination coordinates",
    "parameters": {
        "from_row": {"type": "integer", "minimum": 0, "maximum": 7},
        "from_col": {"type": "integer", "minimum": 0, "maximum": 7},
        "to_row": {"type": "integer", "minimum": 0, "maximum": 7},
        "to_col": {"type": "integer", "minimum": 0, "maximum": 7}
    }
}
```

### Board Representation

The board is displayed to the LLM as an ASCII grid:

```
  0 1 2 3 4 5 6 7
0   b   b   b   b
1 b   b   b   b
2   b   b   b   b
3     .   .   .
4   .   .   .
5 r   r   r   r
6   r   r   r   r
7 r   r   r   r
```

Where:
- `b` = Black piece
- `B` = Black king
- `r` = Red piece
- `R` = Red king
- `.` = Empty dark square
- ` ` = Light square (unused)

## Training Process

### Reward Structure

**Default Behavior (`format_gated_rewards=False`):**
- **Valid moves**: +0.1 per move
- **Invalid moves**: -0.5 penalty
- **Game outcomes**:
  - Win: +2.0
  - Draw: +0.5
  - Loss: -1.0
- **Incomplete games**: -0.5 penalty
- **Thinking bonus**: +0.1 per response with `<think>` tags (capped at +0.5)

**Format-Gated Rewards (`format_gated_rewards=True`):**
- **Valid moves**: 0.0 (format is just a requirement)
- **Invalid moves**: -0.5 penalty
- **Game outcomes**: Same as above
- **Incomplete games**: -0.5 penalty
- **Thinking bonus**: Only given if no invalid moves occurred (+0.1 per `<think>` tag, capped at +0.5)

The format-gated approach prevents the model from gaming small format rewards and focuses training entirely on strategic play.

### Opponent Types

1. **Random Opponent** (`OpponentType.RANDOM`):
   - Selects moves uniformly at random from legal moves
   - Good for initial training and baseline evaluation

2. **Basic AI Opponent** (`OpponentType.BASIC_AI`):
   - Prioritizes captures and multiple jumps
   - Prefers king moves when no captures available
   - Provides moderate challenge for intermediate training

### Metrics Tracked

- **Training**: Average reward, win rate, moves per game, invalid move rate
- **Evaluation**: Win/draw/loss rates, average game length, move validity
- **Game State**: Board positions, move sequences, thinking content

## Example Output

### LLM Response Format

With thinking enabled:
```
<think>
Looking at the board, I have several pieces that can move. The opponent has a piece on (2,3) that I could potentially jump if I move my piece from (5,2) to (4,3), but that would put me in a vulnerable position.

I should look for safer advancing moves. My piece on (5,0) can move to (4,1) which advances toward the opponent's side while staying relatively safe.
</think>

<tool_call>
{"arguments": {"from_row": 5, "from_col": 0, "to_row": 4, "to_col": 1}, "name": "make_move"}
</tool_call>
```

## Advanced Features

### Multiple Jump Detection

The environment automatically detects and enforces multiple jump sequences:

```python
# If a piece can jump multiple opponents in one turn,
# the environment tracks all captured pieces
move = Move(2, 1, 6, 5, jumps=[(3, 2), (5, 4)])  # Captures two pieces
```

### King Promotion Logic

Pieces automatically promote to kings when reaching the opposite end:

```python
# Red piece reaching row 0 becomes a king
if piece == PieceType.RED_PIECE and move.to_row == 0:
    self.set_piece_at(move.to_row, move.to_col, PieceType.RED_KING)
```

### Game State Validation

Comprehensive move validation ensures legal gameplay:

- Position bounds checking
- Piece ownership verification
- Mandatory jump enforcement
- Multiple jump sequence detection

## Performance Considerations

- **Memory Management**: Automatic buffer clearing prevents memory growth
- **Token Efficiency**: Truncates game history when approaching token limits
- **Evaluation Speed**: Configurable episode counts for faster iteration

## Contributing

To extend this environment:

1. **New Opponent Types**: Add entries to `OpponentType` enum and implement in `_get_opponent_move()`
2. **Rule Variants**: Modify `CheckersBoard` class for international rules or other variants
3. **Enhanced AI**: Implement stronger opponents using minimax or neural networks
4. **UI Integration**: Add web interface following the patterns in `deepsacrifice_chess`

## Integration with Atropos

This environment follows Atropos best practices:

- **Standard Base Classes**: Extends `BaseEnv` and `BaseEnvConfig`
- **Tool Integration**: Uses `parse_tool_call` for reliable action parsing
- **Metric Tracking**: Implements comprehensive logging with WandB
- **Token Management**: Proper handling of context length limits
- **Async Design**: Full async/await support for scalable training

The checkers environment demonstrates how to implement complex game logic while maintaining compatibility with the Atropos training framework.
