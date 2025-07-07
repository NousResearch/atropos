# CLAUDE.md - Atropos Project Guide

This document contains project-specific instructions and learnings for Claude assistants working on the Atropos RL training framework.

## Current Work: Testing Diplomacy Environment with SGLang (July 7, 2025)

### Status
We were setting up and testing the Diplomacy RL training environment with the NousResearch/DeepHermes-3-Llama-3-8B-Preview model using SGLang server.

### Completed
1. Updated `/home/maxpaperclips/sglang/run_server.sh`:
   - Changed model to `NousResearch/DeepHermes-3-Llama-3-8B-Preview`
   - Set port to `8000` (matching diplomacy test expectations)
   - Fixed Python path to use `/home/maxpaperclips/sglang/.venv/bin/python`

2. Created `/home/maxpaperclips/atropos/environments/diplomacy_environment/test_sglang_setup.py`:
   - Test script specifically for Diplomacy environment with SGLang
   - Configured to use `http://localhost:8000/v1` endpoint
   - Uses DeepHermes model for all Diplomacy powers

### Issue Encountered
SGLang server failed to start due to permission error:
```
PermissionError: [Errno 13] Permission denied: '/tmp/08f57d08a2eb8581873d780e0b93be175dfdd43949c43f516e06a5c02f60a254NousResearch-DeepHermes-3-Llama-3-8B-Preview.lock'
```
The lock file is owned by user `hjcpuro` and prevents model download.

### Next Steps After Manual Fix
Once the SGLang server is running:
1. Run the test script:
   ```bash
   cd /home/maxpaperclips/atropos/environments/diplomacy_environment
   uv run python test_sglang_setup.py
   ```

2. The test will:
   - Initialize Diplomacy environment
   - Test connection to SGLang server
   - Run a quick game test with the DeepHermes model

### Notes
- The Diplomacy environment expects the inference server on port 8000
- All test scripts use `uv run python` (not plain python)
- The SGLang server needs the virtual environment at `/home/maxpaperclips/sglang/.venv`

## IMPORTANT: Always Use UV Run
- **NEVER use `python` directly** - always use `uv run python`
- This project uses UV for dependency management
- Example: `uv run python diplomacy_env.py`

## Understanding ServerManager and API Integration

### Key Insight: ServerManager is NOT Just for "Atropos Servers"

The ServerManager is a universal API gateway that works with ANY OpenAI-compatible API:
- **OpenAI**: Direct integration with GPT-4, GPT-3.5, etc.
- **Anthropic**: Via OpenAI-compatible proxy
- **Local vLLM**: Any local server implementing OpenAI API
- **Hugging Face**: Inference endpoints with OpenAI format

### How It Works

```python
# Example 1: Using OpenAI directly (most common)
server_configs = [
    APIServerConfig(
        model_name="gpt-4o-mini",  # Must be valid OpenAI model
        base_url="https://api.openai.com/v1",  # OpenAI's API
        api_key=os.getenv("OPENAI_API_KEY"),
        server_type="openai"  # Default
    )
]

# Example 2: Using local vLLM server
server_configs = [
    APIServerConfig(
        model_name="llama-2-70b",  # Whatever model you loaded
        base_url="http://localhost:8000/v1",  # vLLM OpenAI-compatible
        api_key="dummy"  # vLLM doesn't need real key
    )
]
```

### Common Pitfalls When Using ServerManager

1. **Model Name Errors**: The `model_name` in APIServerConfig must be valid for your provider
   - ❌ `model_name="training-policy"` → OpenAI doesn't know this
   - ✅ `model_name="gpt-4o-mini"` → Valid OpenAI model

2. **Intercepting Clients**: When implementing GRPO or similar patterns
   - ❌ `await super().generate_response()` → Uses wrong model name
   - ✅ `await self.env.server.completion(model=self.env.server_configs[0].model_name)`

3. **Mock Server Confusion**: You don't need mock servers for real API calls!
   - Just configure ServerManager with the actual API endpoint
   - It handles OpenAI, Anthropic (via proxy), vLLM, etc.

## Current Work: Diplomacy Training Environment (December 2024)

### Summary
We're building a multi-agent Diplomacy training environment that integrates AI_Diplomacy with Atropos. The environment supports both rejection sampling for data generation and online reinforcement learning.

### Key Architecture Decisions

1. **Integration Approach**: Hybrid model using AI_Diplomacy as a git submodule with subprocess + websocket communication
2. **Multi-Agent Design**: Each power has isolated state with structured XML output format
3. **Memory System**: FAISS-based episodic memory with inline generation during responses
4. **Scoring**: Composite system with VR-CLI (30%), game outcomes (50%), negotiation quality (20%)

### Files Created
- `environments/diplomacy_environment/` - Main environment directory
  - `diplomacy_env.py` - Core environment class
  - `diplomacy_agent.py` - Per-power agent implementation
  - `diplomacy_game_manager.py` - AI_Diplomacy interface
  - `diplomacy_scoring.py` - Composite scoring system
  - `memory_manager.py` - FAISS-based memory
  - `diplomacy_registry.py` - Scenario variations
  - `TODO.md` - Comprehensive 500+ line planning document

### AI_Diplomacy Integration Complete ✅

The AI_Diplomacy submodule has been added and installed in our UV environment with all dependencies:
- Core: `diplomacy` package (game engine)
- AI dependencies: `openai`, `anthropic`, `google-generativeai`, `json-repair`, `together`, `json5`
- All tests passing - ready for integration!

#### Architecture Overview
- **Core Game Engine**: `diplomacy/` - DATC-compliant game implementation
- **AI Agent System**: `ai_diplomacy/` - Stateful LLM-powered agents with memory
- **Visualization**: `ai_animation/` - Modern Three.js interface for game playback
- **Analysis Tools**: Scripts for betrayal detection, lie analysis, strategic moments

#### Key Components for Integration
1. **DiplomacyAgent** (`ai_diplomacy/agent.py`):
   - Maintains goals, relationships, and private diary
   - Robust JSON parsing for LLM responses
   - Memory consolidation system

2. **Game Orchestration** (`lm_game.py`):
   - Manages agent lifecycle and phases
   - Async LLM coordination
   - Phase summaries and relationship tracking

3. **Strategic Analysis** (`ai_diplomacy/possible_order_context.py`):
   - BFS pathfinding for threats/opportunities
   - XML context generation for orders

#### Running the UI
```bash
# Start the AI animation interface (for visualizing AI games)
cd environments/diplomacy_environment
./start_diplomacy_ui.sh
# Opens at http://localhost:5173
# Load game JSON files to see animated playback

# Or start the Diplomacy server (for interactive play)
uv run python -m diplomacy.server.run --port 8432
# Server runs at http://localhost:8432

# Test the installation
uv run python test_ai_diplomacy.py
```

### Next Implementation Steps

1. **Adapt AI_Diplomacy Agent Interface**
   - Map our `DiplomacyAgent` to their `ai_diplomacy.agent.DiplomacyAgent`
   - Integrate their prompt templates and memory system
   - Use their proven XML parsing for orders

2. **Leverage Existing Components**
   - Use `lm_game.py` patterns for episode management
   - Adapt their phase summary system for our scoring
   - Integrate betrayal/lie detection for negotiation scoring

3. **Create Integration Layer**
   - Wrapper to translate between Atropos episodes and AI_Diplomacy games
   - Adapter for their LLM client system to use our server configs
   - Bridge their memory system with our FAISS implementation

### Key Technical Patterns

#### Agent Output Format
```xml
<think>Strategic reasoning here...</think>
<memory>Key facts to remember</memory>
<negotiation>
  <message to="FRANCE" type="proposal">Specific agreement proposal</message>
</negotiation>
```

#### Episode State Management
- Each episode has isolated `DiplomacyEpisodeState`
- Agents maintain per-power state and memories
- Clean separation between episodes for parallel processing

#### Scoring Credit Assignment
- Uses Monte Carlo returns with γ=0.99
- Propagates future returns to unselected alternatives with same action
- Handles sparse rewards through VR-CLI guidance

### Testing Commands
```bash
# Test environment setup
cd environments/diplomacy_environment
uv run python diplomacy_env.py --help

# Once integration complete:
# Data generation
uv run python -m environments.diplomacy_environment.diplomacy_env process --config config.yaml

# Online training
uv run python -m environments.diplomacy_environment.diplomacy_env serve --config config.yaml
```

### Common Issues & Solutions

1. **Submodule Missing**: Run `git submodule update --init --recursive`
2. **FAISS Import Error**: Install with `uv pip install faiss-cpu`
3. **Sentence Transformers**: Install with `uv pip install sentence-transformers`

### Previous Work Completed

#### Process Command Improvements (Cherry-picked from textworld-env-vrcli)
- Added parallel processing support for bulk data generation (10-50x speedup)
- Added `strip_tokens_and_masks` option to reduce file sizes
- Added `do_send_to_api` field to control standalone vs API mode
- Fixed pytest-multiple-versions-actions submodule issue

### Architecture Notes

The Diplomacy environment follows patterns from the TextWorld environment:
- Registry system for scenario variety
- Episode-based state isolation
- VR-CLI scoring for prediction quality
- Inline memory generation in XML blocks

Key differences:
- Multi-agent coordination required
- Negotiation phase before orders
- Trust and relationship tracking
- Longer episodes (multiple game years)

### Research References
See `environments/diplomacy_environment/TODO.md` for:
- Detailed architecture diagrams
- AI_Diplomacy analysis
- Implementation phases
- Code examples
- Performance considerations

## LaTRo (Latent Reasoning Optimization) Implementation

### Summary
Implemented LaTRo rewards for GRPO best-of-N selection as described in https://arxiv.org/html/2411.04282v2. This uses log probabilities from the model to score candidate responses during training.

### Key Implementation Details

1. **AtroposClient Enhanced**: The unified client now handles both normal and GRPO modes
   - In GRPO mode: Performs best-of-N sampling with LaTRo scoring
   - Supports both chat_completion and completion APIs
   - Extracts logprobs from API responses for reward calculation

2. **LaTRo Reward Function**:
   ```python
   def _compute_latro_reward(self, logprobs_sequence: List[float]) -> float:
       """Compute LaTRo reward as sum of log probabilities."""
       if not logprobs_sequence:
           return -10.0  # Default for missing logprobs
       return sum(logprobs_sequence)
   ```

3. **Advantage Computation**: Following the paper's approach
   - Raw scores: r(z_k(i)) = sum of logprobs for response k
   - Advantages: A_k(i) = r(z_k(i)) - mean(r(z_j(i)))
   - Normalized to [0, 1] for training

### Configuration
```python
# In DiplomacyEnvGRPOConfig
use_latro_rewards: bool = True  # Enable LaTRo rewards
latro_beta: float = 0.05  # KL penalty (not currently used)
```

### API Requirements
- **OpenAI API**: Supports logprobs natively
- **Ollama**: Limited support (no logprobs in current version)
- **llama.cpp**: Recent versions support logprobs (PR #10783)
- **vLLM**: Full OpenAI compatibility including logprobs

### Testing
```bash
# Test LaTRo implementation
cd environments/diplomacy_environment
uv run python test_latro_simple.py  # Test logprobs extraction
uv run python test_latro_rewards.py  # Test full GRPO with LaTRo
```

### Known Limitations
1. Ollama's OpenAI compatibility doesn't include logprobs yet
2. Some models/servers may not return logprobs even when requested
3. Fallback to heuristic scoring when logprobs unavailable
