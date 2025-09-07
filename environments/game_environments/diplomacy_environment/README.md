# Minimal Diplomacy Environment

A simplified Diplomacy RL training environment for Atropos that integrates with AI_Diplomacy.

## Overview

This minimal implementation provides:
- Basic game integration via AI_Diplomacy submodule
- Parallel rollouts with configurable group_size
- LLM request interception through AtroposClient proxy
- Simple supply center based scoring
- No complex features (no GRPO, memory systems, or advanced scoring)

## Architecture

```
Atropos Policy Server
        ↓
AtroposClientMinimal (proxy)
        ↓
AI_Diplomacy Game Engine
        ↓
Game Execution
```

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
cd AI_Diplomacy
pip install -e .
```

2. Start your Atropos policy server on port 8000

3. Run the environment:
```bash
python diplomacy_env_minimal.py serve
```

## Configuration

Key settings in `DiplomacyEnvMinimalConfig`:
- `max_game_turns`: Number of game turns (default: 10)
- `training_power`: Which power the RL agent controls (default: "FRANCE")
- `group_size`: Number of parallel games per trajectory (default: 4)

## How It Works

1. **Parallel Rollouts**: Each training step runs `group_size` games with the same initial seed
2. **LLM Interception**: AtroposClientMinimal intercepts all LLM calls from AI_Diplomacy
3. **Trajectory Collection**: Game interactions are collected and scored
4. **Best Selection**: The highest scoring trajectory is returned for training
