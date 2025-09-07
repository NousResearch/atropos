# ReasoningGym Environment

A reinforcement learning environment for training language models on diverse reasoning tasks using the [reasoning-gym](https://github.com/reasoning-gym/reasoning-gym) library.

## Overview

The ReasoningGym environment provides access to 100+ reasoning tasks spanning mathematics, logic, programming, and more. It supports:

- **Diverse Task Types**: Arithmetic, algebra, logic puzzles, programming challenges, and more
- **Advanced Complexity Control**: Three modes for managing task difficulty (None, Random, Curriculum)
- **Adaptive Curriculum Learning**: Automatic difficulty adjustment based on model performance
- **Strict Answer Format Enforcement**: Models must use `<answer>` tags or receive 0 score
- **Dual-Format Scoring**: Tries both raw answers and tagged answers, using the higher score
- **Data Collection**: Optional rollout dumping for successful and failed attempts
- **Comprehensive Logging**: Detailed progress tracking and debugging information

## Features

### Task Diversity
- **102 tasks** from reasoning-gym with full complexity control coverage
- Automatic task discovery from the reasoning-gym registry
- Fallback to comprehensive task list if discovery fails
- Categories include: Arithmetic, Games, Logic, Algorithmic, Cognition, Algebra, Geometry, Code, Graph, ARC, GSM Symbolic, and more

### Complexity Control System

#### Three Complexity Modes

1. **None (Default)**: Uses reasoning-gym's default parameters for all tasks
2. **Random**: Randomizes complexity for each problem (0.0-1.0 scale)
3. **Curriculum**: Adaptive difficulty that adjusts based on model performance

#### Curriculum Learning Features
- **Per-task tracking**: Each task has independent complexity management
- **Target accuracy**: Maintains configurable target accuracy (default 70%)
- **Immediate adjustment**: Complexity updates after each group is scored
- **Stability detection**: Considers performance variance for robust adjustments
- **Fast-track adjustments**: Special handling for very high/low accuracy
- **Comprehensive monitoring**: Detailed curriculum statistics for wandb logging

#### Task Coverage
All 102 reasoning-gym tasks have complexity mappings with realistic parameter ranges:

**Arithmetic Tasks** (15+ tasks):
- `basic_arithmetic`, `leg_counting`, `decimal_arithmetic`, `complex_arithmetic`
- `fraction_simplification`, `bitwise_arithmetic`, `chain_sum`, `count_bits`
- `gcd`, `lcm`, `prime_factorization`, `power_function`, `products`
- `time_intervals`, `calendar_arithmetic`, `dice`, `number_format`

**Games** (15+ tasks):
- `n_queens`, `sudoku`, `mini_sudoku`, `futoshiki`, `tower_of_hanoi`
- `maze`, `sokoban`, `rush_hour`, `puzzle24`, `countdown`, `tsumego`
- `knight_swap`, `emoji_mystery`, `mahjong_puzzle`, `boxnet`

**Logic** (8+ tasks):
- `self_reference`, `propositional_logic`, `knights_knaves`, `syllogism`
- `circuit_logic`, `zebra_puzzles`, `aiw`

**Algorithmic** (30+ tasks):
- `graph_color`, `shortest_path`, `largest_island`, `course_schedule`
- `string_manipulation`, `palindrome_generation`, `word_ladder`
- `binary_matrix`, `spiral_matrix`, `number_sorting`, and many more

**And all other categories**: Cognition, Algebra, Geometry, Code, Graph, ARC, GSM Symbolic, Induction

### Scoring System
- **Binary Tasks**: 0.0 or 1.0 (most tasks)
- **Partial Credit**: Some tasks like GSM Symbolic give 0.01 for wrong but valid numbers
- **Continuous Scoring**: Word Ladder, Sentence Reordering use percentage-based scoring
- **Length Penalty**: Applied to overly long responses when all are correct

### Data Collection
- **Successful Rollouts**: Save groups with scores above configurable threshold
- **Failed Rollouts**: Save completely failed groups (all 0 scores) for debugging
- **Progress Tracking**: Shows buffer progress toward save thresholds
- **JSONL Format**: Easy to process saved data

## Configuration

### Key Parameters

```python
class ReasoningGymEnvConfig(BaseEnvConfig):
    # Data collection
    dump_rollouts: bool = False  # Save successful rollouts
    dump_failed_rollouts: bool = False  # Save failed rollouts for debugging
    rollout_save_score_threshold: float = 0.7  # Minimum score to save group

    # Complexity control
    complexity_mode: Optional[Literal["curriculum", "random"]] = None
    curriculum_target_accuracy: float = 0.7  # Target accuracy for curriculum mode

    # Evaluation
    num_eval_samples_per_task: int = 5  # Samples per task for evaluation
    eval_seed: int = 123  # Fixed seed for reproducible evaluation

    # Logging and debugging
    debug_logging: bool = False  # Enable verbose logging
    suppress_base_env_logs: bool = True  # Hide base environment logs
    seed: int = 42  # Random seed for reproducibility
```

### Example Configurations

#### Basic Training (Default Complexity)
```python
env_config = ReasoningGymEnvConfig(
    tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
    group_size=16,
    max_token_length=1024 * 16,
    complexity_mode=None,  # Use default parameters
    dump_rollouts=True,
)
```

#### Random Complexity Training
```python
env_config = ReasoningGymEnvConfig(
    tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
    group_size=16,
    max_token_length=1024 * 16,
    complexity_mode="random",  # Randomize difficulty
    dump_rollouts=True,
    debug_logging=True,
)
```

#### Curriculum Learning
```python
env_config = ReasoningGymEnvConfig(
    tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
    group_size=16,
    max_token_length=1024 * 16,
    complexity_mode="curriculum",  # Adaptive difficulty
    curriculum_target_accuracy=0.7,  # Maintain 70% accuracy
    dump_rollouts=True,
    debug_logging=True,
)
```

## Setup

### Prerequisites

1. **reasoning-gym submodule**: Clone the reasoning-gym repository as a submodule:
   ```bash
   cd atropos/environments/reasoning_gym_environment/
   git submodule add https://github.com/reasoning-gym/reasoning-gym.git reasoning-gym
   ```

2. **Dependencies**: Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

### Directory Structure
```
reasoning_gym_environment/
├── reasoning_gym_environment.py  # Main environment code
├── reasoning-gym/                # Git submodule
├── data_dumps/                   # Generated rollout data (created automatically)
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## Usage

### Basic Training

```python
from atropos.environments.reasoning_gym_environment import ReasoningGymEnv

# Initialize environment
env_config, server_configs = ReasoningGymEnv.config_init()
env = ReasoningGymEnv(env_config, server_configs)

# Setup and run
await env.setup()
# Training loop handled by atropos framework
```

### Command Line

```bash
python reasoning_gym_environment.py
```

### Monitoring Curriculum Learning

When using curriculum mode, the environment logs detailed statistics:

```python
# Get curriculum statistics
stats = env.get_curriculum_stats()
print(f"Total tasks tracked: {stats['total_tasks_tracked']}")
print(f"Tasks with adjustments: {stats['tasks_with_adjustments']}")
print(f"Average complexity: {stats['avg_complexity']:.2f}")
```

Curriculum metrics are automatically logged to wandb:
- `curriculum/total_tasks_tracked`
- `curriculum/tasks_with_adjustments`
- `curriculum/avg_complexity`
- `curriculum/avg_recent_accuracy`

## Complexity Control Details

### Parameter Mappings

Each task has carefully crafted complexity parameter mappings based on examination of reasoning-gym source code:

#### Example: Basic Arithmetic
```python
"basic_arithmetic": {
    "min_terms": int(2 + complexity_level * 4),  # 2-6 terms
    "max_terms": int(2 + complexity_level * 4),
    "min_digits": int(1 + complexity_level * 3),  # 1-4 digits
    "max_digits": int(1 + complexity_level * 3),
    "allow_parentheses": complexity_level > 0.3,
    "allow_negation": complexity_level > 0.5,
}
```

#### Example: N-Queens
```python
"n_queens": {
    "n": int(4 + complexity_level * 8),  # 4-12 board size
    "min_remove": int(1 + complexity_level * 6),  # 1-7 pieces removed
    "max_remove": int(1 + complexity_level * 6),
}
```

### Curriculum Algorithm

The curriculum system uses the following logic:

1. **Initialization**: All tasks start at 30% complexity
2. **Tracking**: Each task maintains independent performance history (last 10 groups)
3. **Adjustment Trigger**: Requires ≥3 groups before making adjustments
4. **Target Accuracy**: Default 70%, configurable
5. **Adjustment Logic**:
   - If accuracy > target + 5%: Increase complexity by 5%
   - If accuracy < target - 5%: Decrease complexity by 5%
   - Special fast-track for very high (>90%) or very low (<30%) accuracy
6. **Stability**: Considers performance variance to avoid erratic adjustments

### Complexity Ranges

All parameter ranges are based on actual reasoning-gym defaults with reasonable variations:

- **Integer parameters**: Properly converted with `int()`
- **Float parameters**: Only used where appropriate (e.g., edge probabilities)
- **Boolean parameters**: Threshold-based activation
- **Reasonable bounds**: No extreme values that would break tasks

## System Prompt

The environment uses a structured reasoning prompt that encourages models to:

1. Use `<think>` tags for internal reasoning
2. Provide final answers in `<answer>` tags
3. Follow strict format requirements

Example model response:
```
<think>
This is a math problem. Let me work through it step by step.
2 + 3 = 5
</think>

Looking at this problem, I need to add 2 and 3.

<answer>5</answer>
```

## Data Output

### Successful Rollouts
Saved to `data_dumps/reasoning_gym_environment_rollouts_{uuid}_{batch}.jsonl`:

```json
{
  "item_id": "gsm_symbolic",
  "rollouts": [
    {
      "conversation": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "What is 2 + 3?"},
        {"role": "assistant", "content": "<think>2 + 3 = 5</think>\n<answer>5</answer>"}
      ],
      "score": 1.0
    }
  ]
}
```

### Failed Rollouts
Saved to `data_dumps/reasoning_gym_environment_FAILED_rollouts_{uuid}_{batch}.jsonl` with same format but all scores are 0.0.

## Logging

The environment provides comprehensive logging:

### Standard Logging
- **Setup**: Task discovery and initialization
- **Training**: Group scores, task selection, progress tracking
- **Data Dumping**: Save progress and file creation
- **Format Violations**: When models don't follow answer tag requirements

### Curriculum Logging
- **Complexity Adjustments**: Real-time difficulty changes per task
- **Performance Tracking**: Accuracy trends and stability metrics
- **Target Achievement**: When tasks reach optimal difficulty zones

### Debug Mode
Enable with `debug_logging=True` for detailed information:
- Answer extraction attempts
- Scoring method comparisons
- Format violation details
- Task selection patterns
- Complexity parameter usage

## Task Examples

### Mathematics
- **GSM Symbolic**: Grade school math with symbolic reasoning
- **Basic Arithmetic**: Addition, subtraction, multiplication, division with configurable complexity
- **Algebra**: Linear equations and polynomial manipulation

### Logic
- **Sudoku**: Classic number placement puzzles with variable difficulty
- **Propositional Logic**: Boolean reasoning tasks with adjustable clause counts
- **Knights and Knaves**: Logic puzzles with configurable people and statements

### Programming
- **ARC**: Abstract reasoning corpus visual patterns
- **Code Generation**: Simple programming challenges
- **Algorithm Design**: Sorting, searching, and optimization with scalable complexity

### Games
- **N-Queens**: Chess queen placement with variable board sizes
- **Tower of Hanoi**: Disk movement puzzles with adjustable disk counts
- **Rush Hour**: Traffic jam puzzles with configurable car counts

## Troubleshooting

### Common Issues

1. **No tasks discovered**: Ensure reasoning-gym submodule is properly initialized
2. **Import errors**: Check that requirements.txt dependencies are installed
3. **No rollouts saved**: Verify `dump_rollouts=True` and scores exceed threshold
4. **Format violations**: Models not using `<answer>` tags receive 0 scores
5. **Curriculum not adjusting**: Ensure tasks get enough groups (≥3) for adjustments

### Debug Mode

Enable debug logging for detailed information:
```python
env_config.debug_logging = True
```

This shows:
- Answer extraction attempts
- Scoring method comparisons
- Format violation details
- Task selection patterns
- Complexity parameter mappings
- Curriculum adjustment decisions

### Curriculum Monitoring

Monitor curriculum effectiveness:
```python
# Check curriculum statistics
stats = env.get_curriculum_stats()
for task, details in stats['task_details'].items():
    if details['adjustable']:
        print(f"{task}: complexity={details['complexity']:.2f}, "
              f"accuracy={details['recent_accuracy']:.2f}")
```

## Performance Considerations

### Complexity Modes
- **None**: Fastest, no overhead
- **Random**: Minimal overhead, good for exploration
- **Curriculum**: Slight overhead for tracking, optimal for learning

### Memory Usage
- Curriculum mode stores performance history (last 10 groups per task)
- Typical memory overhead: <1MB for all 102 tasks

### Convergence
- Curriculum typically converges to target accuracy within 50-100 groups per task
- Fast-track adjustments help with extreme performance cases
- Stability detection prevents oscillation around target

## Advanced Usage

### Custom Complexity Mappings

To add complexity control for new tasks:

```python
def _get_complexity_params_for_task(self, task_name: str, complexity_level: float):
    # Add your custom task mapping
    if task_name == "my_custom_task":
        return {
            "difficulty": int(1 + complexity_level * 9),  # 1-10
            "size": int(5 + complexity_level * 15),       # 5-20
        }
    # ... existing mappings
```

### Curriculum Customization

Adjust curriculum parameters:

```python
# More aggressive curriculum
env_config.curriculum_target_accuracy = 0.8  # Higher target
# In _adjust_task_complexity, modify:
adjustment_threshold = 0.03  # Smaller threshold for more frequent adjustments
complexity_step = 0.1        # Larger steps for faster adaptation
```

### Integration with External Systems

The environment supports integration with external curriculum systems:

```python
# Override complexity for specific tasks
env.task_complexity_levels["basic_arithmetic"] = 0.8  # Set to 80% complexity
env.task_complexity_levels["n_queens"] = 0.3          # Set to 30% complexity
```
