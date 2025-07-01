# CLAUDE.md - Atropos Project Guide

This document contains project-specific instructions and learnings for Claude assistants working on the Atropos RL training framework.

## IMPORTANT: Always Use UV Run
- **NEVER use `python` directly** - always use `uv run python`
- This project uses UV for dependency management
- Example: `uv run python diplomacy_env.py`

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