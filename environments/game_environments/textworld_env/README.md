# TextWorld RLPR Environment

## Overview

The TextWorld RLPR (Reinforcement Learning with Perplexity Reward) environment is an advanced implementation that uses **VR-CLI (Verifiable Rewards via Completion Likelihood Improvement)** to train language models on text-based adventure games. Unlike the minimal implementation which generates entire episodes per alternative, this system generates alternatives step-by-step, selects the best action using entropy-based confidence scoring, and trains the model's internal world model through perplexity improvement rewards.

## Key Innovations

### 1. Step-by-Step Action Generation
Instead of rolling out complete episodes for each alternative:
- Generate `group_size` alternatives for the **next step only**
- Select the best alternative using entropy/confidence scoring
- Execute the selected action in the environment
- Generate rewards for both selected and unselected alternatives
- Continue until episode completion

### 2. VR-CLI Perplexity Rewards
The system implements the VR-CLI paper's approach to reward prediction accuracy:
- Agent predicts expected outcome for each action
- After execution, measure perplexity improvement: `PPL(actual|context+action+prediction) vs PPL(actual|context)`
- Higher prediction accuracy (lower perplexity) = higher reward
- Directly trains the LLM's world model and planning abilities

### 3. Entropy-Based Action Selection
Uses logit entropy as a proxy for model confidence:
- Calculate entropy and varentropy from token logprobs
- Select action with highest confidence score
- Provides more principled selection than random sampling

### 4. FAISS Memory System
Implements vector-based episodic memory:
- Store memory summaries as embeddings in FAISS index
- Retrieve relevant memories via semantic similarity
- Enables long-term learning across episodes

### 5. Sliding Window Context Management
Maintains manageable context length:
- Strip verbose thinking blocks from history
- Keep memory blocks for structure learning
- Token-aware sliding window with most recent turns prioritized

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TextWorld RLPR Environment                  │
├─────────────────────────────────────────────────────────────────┤
│  1. Generate group_size alternatives for current step          │
│  2. Calculate entropy confidence scores for each               │
│  3. Select best alternative (highest confidence)               │
│  4. Execute selected action in TextWorld environment           │
│  5. Score alternatives:                                         │
│     • Selected + same-action alternatives: VR-CLI rewards      │
│     • Other alternatives: entropy confidence scores            │
│  6. Create ScoredDataGroup for this step                       │
│  7. Update agent memory with selected action outcome           │
│  8. Repeat until episode completion                            │
│  9. Apply credit assignment with discounted returns            │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### TextWorldEnvConfig

Key configuration parameters:

```python
class TextWorldEnvConfig(BaseEnvConfig):
    # Basic environment settings
    group_size: int = 16                    # Alternatives per step
    max_steps: int = 20                     # Max turns per episode
    max_trajectory_tokens: int = 32768      # Token budget per episode
    
    # VR-CLI perplexity reward settings
    vrcli_weight: float = 0.3               # Weight for VR-CLI in final score
    vrcli_discount_factor: float = 0.99     # Credit assignment discount
    
    # Format reward settings (encourage structured responses)
    format_reward_enabled: bool = True
    format_reward_weight: float = 0.1
    format_memory_reward: float = 0.05      # Reward for <memory> blocks
    format_thinking_reward: float = 0.05    # Reward for <think> blocks
    
    # Token length management
    token_length_penalty_enabled: bool = True
    token_length_baseline: int = 500        # Neutral response length
    token_length_penalty_scale: float = 0.0002
```

### AtroposAgent

The agent manages conversation history and memory:

- **Structured Response Format**: Enforces `<think>` → `<memory>` → `<tool_call>` structure
- **Memory Management**: Extracts memory blocks and stores in FAISS
- **Context Management**: Sliding window with token-aware history pruning
- **Action Generation**: Produces multiple alternatives with logprob tracking

### Memory System

FAISS-based episodic memory using sentence transformers:

```python
class AtroposMemoryManager:
    # Uses sentence-transformers/all-MiniLM-L6-v2 for embeddings
    # Stores memories in FAISS IndexFlatL2
    # Retrieves top-k relevant memories for each observation
```

**Memory Flow:**
1. Agent generates `<memory>` block in response
2. Memory extracted and validated after action selection
3. Embedded using SentenceTransformer and stored in FAISS
4. Retrieved via semantic similarity for future observations

## Reward System

### Multi-Component Scoring

Each alternative receives a composite score:

```python
total_score = (
    vrcli_score * vrcli_weight +           # Perplexity improvement
    format_score * format_weight +         # Structure adherence
    token_length_adjustment +              # Length penalty/bonus
    base_confidence_score                  # Entropy-based confidence
)
```

### VR-CLI Perplexity Calculation

For selected actions and same-action alternatives:

1. **Base Perplexity**: `PPL(actual_outcome | previous_context)`
2. **Prediction Perplexity**: `PPL(actual_outcome | previous_context + action + prediction)`
3. **Improvement**: `[1 - PPL_pred/PPL_base] × 100`
4. **Discrete Rewards**:
   - `improvement ≥ 5%`: reward = 1.0
   - `1% ≤ improvement < 5%`: reward = 0.5
   - `improvement < 1%`: reward = 0.0

### Format Rewards

Encourage structured thinking:
- **Memory Block**: +0.05 for valid `<memory>` content
- **Think Block**: +0.05 for valid `<think>` content
- **Structure Penalty**: -0.5× for wrong order or extra blocks
- **JSON Validation**: Proper `<tool_call>` structure required

### Credit Assignment

Monte Carlo returns with same-action credit propagation:

```python
# Backward pass through episode
for step in reversed(episode):
    future_return = discount_factor * (total_return - immediate_reward)
    
    # Update selected alternative
    selected_score += future_return
    
    # Credit alternatives that would have chosen same action
    for alt in alternatives:
        if alt.action == selected_action:
            alt.score += future_return
```

## Response Format

The agent must follow this exact structure:

```xml
<think>
Long chain of thought considering the current situation, objectives, 
and likely outcomes of potential actions. This should be extremely 
detailed and thorough.
</think>

<memory>
Concise summary building on previous memories (if shown), noting the 
outcome of the last action, current game state, inventory, location, 
and progress toward objectives.
</memory>

<tool_call>
{"name": "execute_command", "arguments": {"command": "go north", "expected_outcome": "I expect to move north to a new room, possibly the kitchen based on the house layout I've observed."}}
</tool_call>
```

### Key Requirements:
- **Exactly one** of each block type, in order
- **Think blocks** are stripped from history to save tokens
- **Memory blocks** are preserved for structure learning
- **Expected outcome** is crucial for VR-CLI scoring

## Memory Integration

### Memory Retrieval
When generating responses, relevant memories are prepended:

```
Relevant Memories:
- Found kitchen has stove and ingredients in previous exploration
- Objective requires cooking something, need to gather materials
- Successfully used similar commands in past episodes

[Current observation]
You are in a living room...
```

### Memory Generation
Memories are extracted from the `<memory>` block of selected actions:

```python
# From agent response
memory_content = extract_memory_block(response_text)
if validate_memory_content(memory_content):
    await memory_manager.add_memory(memory_content)
```

## Configuration Examples

### Development/Testing
```python
config = TextWorldEnvRLPRConfig(
    group_size=4,              # Faster generation
    max_steps=5,               # Short episodes  
    total_steps=1,             # Single episode
    max_num_workers=1,         # Single worker
    use_wandb=False,           # No logging
)
```

### Production Training
```python
config = TextWorldEnvRLPRConfig(
    group_size=16,             # Full alternative space
    max_steps=20,              # Reasonable episode length
    total_steps=500,           # Many episodes for training
    max_num_workers=16,        # Parallel processing
    vrcli_weight=0.3,          # Balanced VR-CLI contribution
    use_wandb=True,            # Full metrics tracking
)
```

## Running the Environment

### Local Testing
```bash
cd /home/maxpaperclips/atropos/environments/game_environments/textworld_env
python textworld_rlpr_local_server.py
```

### SLURM Training
```bash
cd /home/maxpaperclips/simple-trainer
sbatch --export=ALL,CONFIG_FILE=/path/to/config.toml,MODEL_NAME=NousResearch/Hermes-4-Qwen3-14B-1-e3,PYTHON_SCRIPT=/home/maxpaperclips/atropos/environments/game_environments/textworld_env/textworld_env_rlpr.py,WANDB_API_KEY=$WANDB_API_KEY online_singlenode_textworld.slurm
```

## Performance Characteristics

### Memory Usage
- **FAISS Index**: ~384 bytes per memory + overhead
- **Context Window**: Managed via sliding window (20K tokens default)
- **Episode State**: Minimal storage, only current episode data

### Computational Cost
- **Perplexity Calculation**: 2 forward passes per VR-CLI evaluation
- **Entropy Scoring**: Requires logprobs from inference server
- **Memory Retrieval**: O(log n) FAISS search per step

### Scalability
- **Episodes**: Parallel processing across workers
- **Alternatives**: Generated concurrently via async API calls
- **Memory**: FAISS scales to millions of memories efficiently

## Key Differences from Minimal Implementation

| Aspect | Minimal Env | RLPR Env |
|--------|-------------|----------|
| **Episode Generation** | Full episodes per alternative | Step-by-step with action selection |
| **Reward Signal** | Basic TextWorld rewards | VR-CLI perplexity improvement |
| **Action Selection** | Random/top-k sampling | Entropy-based confidence scoring |
| **Memory System** | None | FAISS vector memory with retrieval |
| **Context Management** | Basic history | Sliding window with think-block stripping |
| **Credit Assignment** | Simple episode rewards | Monte Carlo with same-action propagation |
| **Response Structure** | Free-form | Enforced think/memory/tool_call format |

## Monitoring and Debugging

### Key Metrics (WandB)
- `episode_completion_rate`: Fraction of episodes successfully completed
- `average_episode_length`: Mean number of turns per episode
- `vrcli_score_distribution`: Distribution of perplexity improvement scores
- `memory_retrieval_accuracy`: Relevance of retrieved memories
- `entropy_confidence_correlation`: Relationship between entropy and success

### Log Analysis
```bash
# Monitor episode progress
tail -f /home/maxpaperclips/simple-trainer/logs/JOB_ID/env_server.log | grep "Episode.*Turn.*completed"

# Check VR-CLI scoring
grep "VR-CLI.*perplexity" /home/maxpaperclips/simple-trainer/logs/JOB_ID/env_server.log

# Memory system status
grep "memory.*active\|FAISS" /home/maxpaperclips/simple-trainer/logs/JOB_ID/env_server.log
```

### Common Issues
1. **Memory Parser Import Error**: Missing `memory_parser.py` utility
2. **FAISS Not Available**: Install with `pip install faiss-cpu sentence-transformers`
3. **Context Overflow**: Reduce `max_history_tokens` or `group_size`
4. **Perplexity Calculation Failure**: Check inference server logprobs support

## Future Enhancements

### Algorithmic Improvements
- **Adaptive Group Size**: Scale alternatives based on action complexity
- **Curiosity Rewards**: Additional reward for surprising outcomes
- **Temperature Annealing**: Reduce exploration as training progresses
- **Hierarchical Memory**: Multi-level memory with different time scales

### Performance Optimizations
- **Batch Perplexity**: Calculate multiple perplexities in single forward pass
- **Memory Clustering**: IVF-FAISS for sub-linear memory search
- **Async Parallelization**: Concurrent alternative generation and scoring
- **Caching**: Cache perplexity scores for repeated evaluations

### Training Improvements
- **Curriculum Learning**: Start with simple tasks, increase complexity
- **Importance Sampling**: Replay high-value experiences more frequently
- **Multi-Task Training**: Train on multiple TextWorld challenges simultaneously
- **Regularization**: Prevent overfitting to specific game patterns

This implementation represents a significant advancement over the minimal environment, providing much richer training signals and more sophisticated agent capabilities while maintaining computational efficiency through careful optimization of context management and reward calculation.