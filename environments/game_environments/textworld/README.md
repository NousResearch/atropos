# TextWorld Environment for Atropos

A trainer environment for Microsoft TextWorld that integrates with AtroposAgent and AtroposRM for reinforcement learning from human feedback (RLHF).

## Features

- **TextWorld Integration**: Generates and manages TextWorld games for episodic training
- **Best-of-N Policy Training**: Uses AtroposAgent to generate multiple action alternatives
- **Reward Model Evaluation**: Uses AtroposRM to score actions and select the best alternative
- **Memory System**: Optional RAG-based memory for long-horizon planning
- **Thinking Block Summarization**: Efficient LLM-based summarization with episode-level caching
- **Monte Carlo Returns**: Calculates discounted returns for chosen actions

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

## Architecture

- `TextWorldEnv`: Main environment class extending BaseEnv
- `AtroposAgent`: Policy agent that generates action alternatives
- `AtroposRM`: Reward model that evaluates actions
- `TextWorldMemoryManager`: Optional memory system using FAISS
- `TextWorldEpisodeState`: Per-episode state management

## Memory System

The optional memory system provides:
- Semantic embedding of game observations and actions
- FAISS-based similarity search for relevant memories
- Contextual memory retrieval for decision making

Memory is particularly useful for long-horizon games like Zork where important information may exceed the context window.

## Data Flow

1. Generate TextWorld game and initial observation
2. Agent generates multiple action alternatives
3. RM evaluates each alternative and assigns Q-values
4. Select best action and execute in environment
5. Record experience and update memory (if enabled)
6. Calculate Monte Carlo returns for training data
7. Process thinking blocks with summarization caching

## Files

- `textworld_env.py`: Main environment implementation
- `textworld_local_server.py`: Test runner and example usage
- `generation_utils.py`: TextWorld game generation utilities
- `agents/`: Agent implementations and memory management
