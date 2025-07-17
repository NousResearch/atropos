# TextWorld Environment for Atropos

A trainer environment for Microsoft TextWorld that integrates with AtroposAgent for reinforcement learning on multi-step reasoning and planning tasks.

## Features

- **TextWorld Integration**: Generates and manages TextWorld games for episodic training
- **Rejection Sampling**: Generates multiple action alternatives per state
- **Hybrid Reward System**: Combines VR-CLI, LaTRo, and environment rewards
- **Inline Memory Generation**: Agent generates memories as part of response
- **Game Registry**: Mix of generated games (70%) and pre-built challenges (30%)
- **Sparse Reward Handling**: Dense learning signals even with sparse environment rewards

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables in `.env` (Optional: if you want to generate data using `process`):
   ```bash
   OPENAI_API_KEY=your_api_key_here
   ```

3. Run a test episode:
   ```bash
   python textworld_local_server.py
   ```

## Configuration

Key configuration options in `TextWorldEnvConfig`:

- `challenge_name`: TextWorld challenge type (default: "tw-simple")
- `max_steps`: Maximum steps per episode (default: 50)
- `group_size`: Number of action alternatives (default: 2)
- `enable_memory`: Enable RAG-based memory system
- `enable_policy_thinking_summarization`: Summarize thinking blocks for efficiency

## Reward System: VR-CLI vs LaTRo

The environment uses a hybrid reward system combining three complementary signals to provide dense learning in sparse reward environments:

### VR-CLI (Verifiable Rewards via Completion Likelihood Improvement)

**What it measures**: How much does the agent's prediction improve our ability to predict what actually happens?

**How it works**:
1. Agent predicts: "Using the key will unlock the door"
2. Actual outcome: "The door unlocks with a click"
3. VR-CLI calculates:
   - Base perplexity: P(outcome | state)
   - Conditioned perplexity: P(outcome | state + prediction)
   - Reward: (base_ppl - cond_ppl) / base_ppl

**Key insight**: Good predictions should make outcomes more predictable (lower perplexity).

**Advantages**:
- Rewards accurate world modeling
- Works even when actions fail (partial credit for understanding)
- Provides learning signal before seeing final game outcome
- Helps agent learn causal relationships

**Disadvantages**:
- Requires computing perplexity twice
- Can reward accurate but useless predictions
- Still needs actual outcomes from environment

### LaTRo (Cross-Entropy Based Action Scoring)

**What it measures**: How confident is the model in its own action given its reasoning?

**How it works**:
1. Agent reasons: "I need to unlock the door with the key"
2. Agent selects action: "unlock door with key"
3. LaTRo calculates: log P(action | state + reasoning)

**Key insight**: Well-reasoned actions should have high probability under the model's own distribution.

**Advantages**:
- Immediate reward (no need to wait for outcomes)
- Model serves as its own reward function
- Rewards coherent reasoning-to-action flow
- Computationally efficient (single forward pass)

**Disadvantages**:
- Can lead to overconfidence
- No external verification
- May reward self-consistent but wrong behaviors

### Why Both Are Needed

VR-CLI and LaTRo are complementary:

- **VR-CLI** verifies the agent's world model against reality
- **LaTRo** ensures the agent's actions follow from its reasoning
- **Environment rewards** keep both grounded in actual task success

The combination provides:
1. **Dense learning signal**: Rewards at every step, not just task completion
2. **Multi-faceted feedback**: Action quality, prediction accuracy, and task progress
3. **Balanced optimization**: Prevents reward hacking through any single metric

### Reward Combination

The final reward for each action is:
```
reward = α * latro_score + β * vrcli_score + γ * env_reward
```

Current weights:
- α (LaTRo weight) = 0.3
- β (VR-CLI weight) = 0.3
- γ (Environment weight) = 0.4

This weighting ensures the agent:
- Learns to reason coherently (LaTRo)
- Builds accurate world models (VR-CLI)
- Stays focused on task completion (Environment)

## Agent Architecture

The TextWorld agent (AtroposAgent) operates with a structured response format:

```xml
<think>
[Long reasoning chains about the current state, objectives, and planning]
</think>
<memory>
[Concise summary of key information to remember for future turns]
</memory>
<tool_call>
{"name": "execute_command", "arguments": {"command": "go north", "expected_outcome": "..."}}
</tool_call>
```

### Components:
- **Thinking blocks**: Allow unrestricted reasoning and planning
- **Memory blocks**: Persist important information across turns
- **Tool calls**: Structured action format with outcome predictions

## Memory System

The inline memory system provides:
- **Agent-generated summaries**: Model creates memories as part of response
- **Vector storage**: Memories stored in FAISS index with sentence embeddings
- **Top-k retrieval**: Most relevant memories retrieved each turn
- **Continuity**: New memories build upon retrieved memories

## Game Types

The environment uses a registry system for game diversity:

### Generated Games (70%)
- **Quest**: Fetch, delivery, rescue missions
- **Puzzle**: Lock sequences, weight puzzles, light puzzles
- **Navigation**: Mazes, labyrinths, dungeons
- **Mixed**: Combinations of above mechanics

### Pre-built Challenges (30%)
- **tw-simple**: Basic single-room puzzles
- **tw-cooking**: Recipe following tasks
- **tw-coin_collector**: Exploration and collection
- **tw-treasure_hunter**: Multi-step treasure finding

## Data Flow

1. **Game Selection**: Registry selects game (generated or challenge)
2. **Multi-alternative Generation**: Agent generates N alternatives per state
3. **Triple Scoring**: 
   - LaTRo scores action confidence (immediate)
   - Best action executed in environment
   - VR-CLI scores prediction accuracy (after outcome)
   - Environment provides task reward
4. **Memory Update**: Agent's inline memory stored if present
5. **Episode Tracking**: Each turn saved with episode_id and turn_number

## Configuration

Key parameters in `config_process.yaml`:

```yaml
# Game settings
max_steps: 50  # Maximum turns per episode
use_registry: true  # Use game registry for diversity
registry_generation_ratio: 0.7  # 70% generated, 30% challenges

# Reward weights
vrcli_weight: 0.3  # Prediction accuracy weight
latro_weight: 0.3  # Action confidence weight
# Environment weight: 1 - vrcli_weight - latro_weight = 0.4

# Memory settings
memory_top_k: 3  # Number of memories to retrieve
enable_memory: true  # Use inline memory system

# Generation settings
group_size: 16  # Alternatives per state (rejection sampling)
temperature: 0.7  # Action generation temperature
```

## Analysis Tools

- **`analyze_episodes.py`**: Reconstruct and analyze episodes from generated data
  - Episode-by-episode analysis
  - VR-CLI score distributions
  - Memory coherence checking
  - Turn-by-turn breakdowns

- **`filter_dataset.py`**: Filter generated data by various criteria
  - Score thresholds
  - Episode length
  - Game types

## Implementation Notes

### VR-CLI Implementation
- Currently uses continuous scoring: `(base_ppl - pred_ppl) / base_ppl`
- Paper uses discrete levels (0, 0.5, 0.9, 1.0) based on % improvement
- TODO: Update to match paper's threshold-based approach

### LaTRo Implementation
- Calculates `log P(action | state + reasoning)`
- No KL penalty (handled by main RL trainer)
- Provides immediate dense reward signal

### Memory System
- Inline generation during agent response
- Falls back to LLM summarization if no inline memory
- Retrieves top-k memories by cosine similarity

## Future Directions

1. **VR-CLI Improvements**:
   - Implement discrete reward levels from paper
   - Test different perplexity temperatures
   - Validate against paper's results

2. **Memory Enhancements**:
   - Cross-episode memory transfer
   - Memory importance weighting
   - Forgetting mechanisms

3. **Game Extensions**:
   - Larger, more complex games
   - Multi-agent scenarios
   - Real-time constraints

4. **Reward Research**:
   - Experiment with different weight combinations
   - Test purely self-supervised variants
   - Compare with traditional reward modeling

## Citations

- **VR-CLI**: "Verifiable Rewards via Completion Likelihood Improvement" (https://arxiv.org/html/2503.22828v1)
- **LaTRo**: "LaTRO: Improving Reasoning in LLMs through Latent Representation Optimization" (https://arxiv.org/html/2411.04282v2)
- **TextWorld**: "TextWorld: A Learning Environment for Text-based Games" (https://arxiv.org/abs/1806.11532)

## Files

- `textworld_env.py`: Main environment implementation with VR-CLI/LaTRo scoring
- `textworld_local_server.py`: Local testing without distributed training
- `textworld_registry.py`: Game selection and generation system
- `generation_utils.py`: TextWorld game generation utilities
- `generators/`: Procedural game generators (quest, puzzle, navigation, mixed)
- `agents/atropos_agent.py`: Agent with thinking, memory, and tool calling
- `utils/memory_parser.py`: XML parsing for inline memory extraction
- `analyze_episodes.py`: Episode reconstruction and analysis
- `filter_dataset.py`: Data filtering and format conversion
- `config_process.yaml`: Configuration for data generation
- `run_datagen_with_sglang.slurm`: SLURM script for distributed generation
