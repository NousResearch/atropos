# VR-CLI Integration for Diplomacy Environment

## Overview

This document describes the VR-CLI (Verifiable Rewards via Completion Likelihood Improvement) integration for the Diplomacy environment. VR-CLI provides an additional reward signal based on how well agents predict the outcomes of their actions.

## Components

### 1. VRCLIScorer (`vrcli_scorer.py`)
- Calculates perplexity of outcomes with and without predictions
- Measures prediction quality as perplexity improvement
- Maps improvements to discrete reward levels (0.0, 0.5, 0.9, 1.0)

### 2. DiplomacyVRCLIIntegration (`diplomacy_vrcli_integration.py`)
- Manages prediction storage and outcome tracking
- Extracts actual outcomes from game results
- Combines VR-CLI scores with base rewards (e.g., LaTRo)

### 3. Integration Points

#### AtroposClient (`atropos_client.py`)
- Modified to extract predictions from tool_call responses
- Stores predictions in `predictions_history` for later scoring
- Maintains compatibility with AI_Diplomacy

#### DiplomacyEnvGRPO (`diplomacy_env_grpo.py`)
- Initializes VR-CLI components when `use_vrcli=True`
- Extracts and stores outcomes after each game phase
- Applies VR-CLI scores to trajectory data before credit assignment

## How It Works

1. **During Game Play**:
   - Agent makes predictions about outcomes in tool_call format
   - AtroposClient extracts and stores these predictions
   - Game proceeds normally with AI_Diplomacy

2. **After Each Phase**:
   - Extract actual outcomes (messages sent, board changes, trust changes)
   - Store outcomes for comparison with predictions

3. **Scoring**:
   - Calculate base perplexity: P(outcome | game_state)
   - Calculate conditioned perplexity: P(outcome | game_state + prediction)
   - Score = improvement in perplexity (lower is better)

4. **Credit Assignment**:
   - Combine VR-CLI scores with LaTRo rewards
   - Apply combined scores to trajectory data
   - Use for GRPO training updates

## Configuration

In your training config:

```python
class DiplomacyEnvGRPOConfig:
    use_vrcli: bool = True  # Enable VR-CLI scoring
    vrcli_weight: float = 0.3  # Weight for VR-CLI in final reward
```

## Testing

Run the test script to verify integration:

```bash
cd /home/maxpaperclips/atropos/environments/diplomacy_environment
uv run python test_vrcli_integration.py
```

## Benefits

1. **Better World Models**: Agents learn to predict consequences
2. **Reduced Hallucination**: Rewards accurate predictions
3. **Complement to LaTRo**: Combines action confidence with prediction quality
4. **Sparse Reward Guidance**: Provides signal even before game ends