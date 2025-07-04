# Diplomacy Environment TODO

This document contains comprehensive research, architecture plans, and implementation details for the Diplomacy training environment in Atropos. It serves as the primary reference for development to avoid repeating research.

## Table of Contents
1. [Research Summary](#research-summary)
2. [Architecture Overview](#architecture-overview)
3. [Core Components Design](#core-components-design)
4. [Implementation Plan](#implementation-plan)
5. [Technical Details](#technical-details)
6. [Testing Strategy](#testing-strategy)
7. [Performance Considerations](#performance-considerations)
8. [Future Enhancements](#future-enhancements)

## GRPO Intercepting Client Architecture

### Key Concept: Event-Driven vs Step-Based Environments

Unlike typical RL environments (gym-style) where we control the step-by-step flow, Diplomacy is **event-driven**:
- AI_Diplomacy drives the game flow
- It calls LLM clients whenever it needs decisions
- We can't easily "step" through the environment

### Solution: Intercepting Client Architecture

We intercept LLM calls to implement GRPO's best-of-N selection:

```
┌─────────────────────────────────────────────────────────────────┐
│                        DiplomacyEnv                              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              collect_trajectories(item)                   │    │
│  │  1. Creates InterceptingAtroposClient for training agent │    │
│  │  2. Creates standard LLM clients for opponents           │    │
│  │  3. Starts lm_game.main() with these clients            │    │
│  │  4. Collects ScoredDataGroups from intercepting client  │    │
│  │  5. Applies credit assignment to trajectory             │    │
│  └─────────────────┬───────────────────────────────────────┘    │
└─────────────────────┼────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              InterceptingAtroposClient                           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │          generate_response(prompt) called by AI_Diplomacy│    │
│  │  1. Samples N responses from policy (via ServerManager)  │    │
│  │  2. Scores each response                                │    │
│  │  3. Creates ScoredDataGroup with all alternatives       │    │
│  │  4. Selects best response to return                     │    │
│  │  5. Accumulates trajectory for later retrieval          │    │
│  └─────────────────┬───────────────────────────────────────┘    │
└─────────────────────┼────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ServerManager                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Routes to configured API (OpenAI, vLLM, etc.)           │    │
│  │  CRITICAL: Uses correct model name (e.g., gpt-4o-mini)   │    │
│  └─────────────────┬───────────────────────────────────────┘    │
└─────────────────────┼────────────────────────────────────────────┘
```

### Implementation Details

1. **Unified AtroposClient with GRPO Mode**
   - Single client handles both normal and GRPO training modes
   - In GRPO mode: Performs best-of-N selection and data collection
   - No separate InterceptingAtroposClient needed

2. **Best-of-N Selection with LaTRo Rewards**
   ```python
   async def generate_response(self, prompt: str, ...):
       if not self.is_training or not self.env:
           # Normal mode: just forward to API
           return await self._normal_generate_response(prompt, ...)

       # GRPO mode with LaTRo rewards
       # Sample N responses with logprobs
       responses, all_logprobs = await self._sample_n_responses(prompt, n=4, temperature)

       # Score each response using LaTRo
       raw_scores = []
       for i, response in enumerate(responses):
           logprobs = all_logprobs[i]
           score = self._compute_latro_reward(logprobs)  # r(z) = Σ log p(z_i)
           raw_scores.append(score)

       # Compute advantages: A_k = r(z_k) - mean(r(z_j))
       mean_score = np.mean(raw_scores)
       advantages = [s - mean_score for s in raw_scores]

       # Normalize to [0, 1] for training
       scores = normalize_advantages(advantages)

       # Create ScoredDataGroup
       group = ScoredDataGroup(tokens=..., scores=scores, ...)
       self.trajectory_data.append(group)

       # Return best response
       return responses[np.argmax(scores)]
   ```

3. **Critical: Use ServerManager for API Calls**
   ```python
   # ✅ CORRECT - Uses ServerManager's configured model
   await self.env.server.chat_completion(
       messages=messages,
       model=self.env.server_configs[0].model_name,  # e.g., "gpt-4o-mini"
       logprobs=True,  # Enable for LaTRo
       top_logprobs=5,
       ...
   )

   # Fallback to completion API if chat fails
   prompt = self.env.tokenizer.apply_chat_template(messages, ...)
   await self.env.server.completion(
       prompt=prompt,
       model=self.env.server_configs[0].model_name,
       logprobs=5,  # Request top 5 logprobs
       ...
   )
   ```

4. **LaTRo Implementation Status**
   - ✅ Implemented in AtroposClient
   - ✅ Supports both chat_completion and completion APIs
   - ✅ Falls back to heuristic scoring when logprobs unavailable
   - ✅ Configuration via `use_latro_rewards` flag
   - ⚠️ Requires API with logprobs support (OpenAI, vLLM, llama.cpp)
   - ❌ Ollama doesn't support logprobs yet

4. **Credit Assignment After Game**
   - Game completes with final score
   - Work backwards through trajectory
   - Apply Monte Carlo returns with discounting

### Common Pitfalls

1. **Model Name Confusion**
   - AI_Diplomacy uses "intercepting-france" as model name
   - This is NOT a valid OpenAI model
   - Must use ServerManager with correct model configuration

2. **Parent Method Calls**
   - AtroposClient.generate_response() uses self.model_name
   - This sends invalid model names to the API
   - Always implement your own method using ServerManager

3. **Assuming Mock Servers**
   - No mocking needed!
   - ServerManager + APIServerConfig handle real API calls
   - Just configure with OpenAI URL and valid model names

## Research Summary

### TextWorld Environment Analysis (from textworld-env-vrcli branch)

#### Key Architectural Patterns Identified:

1. **Episode-Based State Management**
   - Clean separation between episodes with `TextWorldEpisodeState`
   - Agent instances are isolated per episode
   - State replay capability for candidate evaluation

2. **VR-CLI Scoring System**
   ```python
   # Verifiable Rewards via Completion Likelihood Improvement
   improvement = (1 - PPL(y|x,a)/PPL(y|x)) × 100

   # Discrete reward levels:
   - 0.0: improvement < 0.05 (negligible)
   - 0.5: 0.05 ≤ improvement < 1 (small)
   - 0.9: 1 ≤ improvement < 2 (moderate)
   - 1.0: improvement ≥ 2 (significant)
   ```
   - Encourages accurate world model development
   - Rewards good predictions of action outcomes

3. **Memory Management**
   - Inline memory generation in XML blocks
   - FAISS vector database with sentence embeddings
   - Episode-isolated memory managers
   - Fallback LLM summarization

4. **Registry System**
   - 70% procedurally generated, 30% pre-built challenges
   - Difficulty scaling (easy, medium, hard, expert)
   - Automatic file cleanup
   - LRU caching for performance

5. **Structured Agent Output**
   ```xml
   <think>Reasoning about the situation</think>
   <memory>Key facts to remember</memory>
   <tool_call>{"name": "action", "arguments": {...}}</tool_call>
   ```

6. **Credit Assignment**
   - Monte Carlo returns with discounting (γ = 0.99)
   - Future returns propagated to alternatives with same action
   - Handles sparse reward environments effectively

### AI_Diplomacy Project Analysis

#### Technologies & Architecture:
- **Core**: Python-based game engine
- **AI Integration**: Supports OpenAI, Anthropic, Google, OpenRouter LLMs
- **Frontend**: Vue/React for web visualization
- **Infrastructure**: Docker support, websocket communication
- **Multi-Agent**: 7 simultaneous powers with different LLMs

#### Key Features:
1. **Stateful AI Agents**
   - Dynamic goal management
   - Relationship tracking (Enemy/Unfriendly/Neutral/Friendly/Ally)
   - Private diary entries
   - Yearly memory consolidation

2. **Game Phases**
   - Negotiation rounds
   - Order planning
   - Move execution
   - Result processing

3. **Analysis Tools**
   - Lie detection mechanisms
   - Betrayal analysis
   - Strategic pathfinding (BFS)
   - Post-game visualization

4. **Communication**
   - Multi-round message exchanges
   - Private and broadcast messages
   - Context construction for decisions

## Architecture Overview

### Integration Strategy: Intercepting Client Architecture

#### Core Design Principle
The Diplomacy game engine (via `lm_game.py`) drives the game flow and makes LLM calls at specific decision points. Rather than trying to control the stepping process externally, we intercept these LLM calls to implement GRPO's best-of-N selection and trajectory collection.

#### Architecture Diagram
```
┌─────────────────────────────────────────────────────────────────┐
│                        DiplomacyEnv                              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              collect_trajectories(item)                   │    │
│  │  1. Creates InterceptingAtroposClient for training agent │    │
│  │  2. Creates standard LLM clients for opponents           │    │
│  │  3. Starts lm_game.main() with these clients            │    │
│  │  4. Collects ScoredDataGroups from intercepting client  │    │
│  │  5. Applies credit assignment to trajectory             │    │
│  └─────────────────┬───────────────────────────────────────┘    │
│                    │                                             │
│  ┌─────────────────▼───────────────────────────────────────┐    │
│  │           InterceptingAtroposClient                      │    │
│  │  - Inherits from AtroposClient                          │    │
│  │  - Intercepts each LLM call from Diplomacy             │    │
│  │  - Implements best-of-N selection per decision         │    │
│  │  ┌─────────────────────────────────────────────────┐   │    │
│  │  │         generate_response(prompt)                │   │    │
│  │  │  1. Sample N responses from policy server       │   │    │
│  │  │  2. Score each response                        │   │    │
│  │  │  3. Create ScoredDataGroup                     │   │    │
│  │  │  4. Select best response                       │   │    │
│  │  │  5. Return to Diplomacy                        │   │    │
│  │  └─────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AI_Diplomacy (lm_game.py)                    │
│  - Runs game loop                                               │
│  - Calls LLM clients for decisions                              │
│  - Manages game state                                           │
│  - Unaware of GRPO training                                     │
└─────────────────────────────────────────────────────────────────┘
```

#### Key Benefits
1. **Minimal Invasiveness**: AI_Diplomacy code remains unchanged
2. **Clean Separation**: Training logic isolated from game logic
3. **Natural Integration**: Uses existing LLM client interface
4. **Flexible**: Easy to switch between training and evaluation modes
5. **Debuggable**: Clear data flow and decision points

### Implementation Details

#### InterceptingAtroposClient

```python
class InterceptingAtroposClient(AtroposClient):
    """
    Extends AtroposClient to intercept LLM calls and implement GRPO best-of-N selection.

    Key responsibilities:
    1. Sample multiple responses for each prompt
    2. Score responses (initially random, later using per-step rewards)
    3. Create ScoredDataGroup for each decision
    4. Accumulate trajectory data
    5. Return only the selected response to Diplomacy
    """

    def __init__(self, model_name: str, server_url: str, env: 'DiplomacyEnv'):
        super().__init__(model_name, server_url)
        self.env = env  # Reference to parent environment
        self.trajectory_data: List[ScoredDataGroup] = []
        self.canonical_history: List[Dict] = []  # Selected responses only

    async def generate_response(self, prompt: str, temperature: float = 0.0) -> str:
        """
        Intercept point for GRPO best-of-N selection.

        Flow:
        1. Build messages from canonical history + current prompt
        2. Sample N responses from policy server
        3. Score each response
        4. Create ScoredDataGroup with all alternatives
        5. Select best response
        6. Update canonical history
        7. Return selected response
        """
```

#### ScoredDataGroup Structure

For each decision point, we create:
```python
{
    "tokens": [alt1_tokens, alt2_tokens, ..., altN_tokens],  # Full history + alternative
    "masks": [alt1_masks, alt2_masks, ..., altN_masks],
    "scores": [score1, score2, ..., scoreN],  # Per-step scores (random initially)
    "messages": [alt1_msgs, alt2_msgs, ..., altN_msgs],  # Optional, for debugging
    "group_overrides": {"power": "FRANCE", "phase": "S1901M", "decision_type": "orders"}
}
```

#### Credit Assignment Strategy

After game completion:
1. Calculate final game score (e.g., supply center differential)
2. For each ScoredDataGroup in trajectory:
   - Apply discounted return: `score = step_score + γ * future_return`
   - Update scores for both selected and unselected alternatives that took same action
3. This ensures proper GRPO advantage estimation

#### Canonical History Management

Critical for GRPO correctness:
- Each alternative in a group shares the same prefix (canonical history)
- Only the selected response is added to canonical history
- This ensures all alternatives at step t+1 build on the same history from step t

### Legacy Hybrid Model Design (For Reference)

```
┌─────────────────────────────────────────────────────────────┐
│                     Atropos Environment                       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              DiplomacyEnv (BaseEnv)                  │    │
│  │  ┌─────────────────┐  ┌──────────────────────┐     │    │
│  │  │ Episode Manager │  │  Agent Coordinator   │     │    │
│  │  └─────────────────┘  └──────────────────────┘     │    │
│  │  ┌─────────────────┐  ┌──────────────────────┐     │    │
│  │  │ Scoring System  │  │  Memory Management   │     │    │
│  │  └─────────────────┘  └──────────────────────┘     │    │
│  └─────────────────────────────────────────────────────┘    │
│                              ▲                               │
│                              │ WebSocket/API                 │
│                              ▼                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │          AI_Diplomacy Game Server (Subprocess)       │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │    │
│  │  │ Game Engine │  │ Rule Engine │  │ Map State  │  │    │
│  │  └─────────────┘  └─────────────┘  └────────────┘  │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
environments/
  diplomacy_environment/
    __init__.py
    diplomacy_env.py              # Main environment class
    diplomacy_agent.py            # Agent wrapper for LLM interaction
    diplomacy_memory.py           # Memory and relationship tracking
    diplomacy_scoring.py          # VR-CLI and reward calculation
    diplomacy_registry.py         # Scenario management
    negotiation_protocol.py       # Communication structure
    game_wrapper.py              # Interface to AI_Diplomacy
    analysis_tools.py            # Game analysis utilities

    config/
      config_process.yaml        # Data generation configuration
      config_train.yaml          # RL training configuration

    scenarios/
      classic_1901.yaml          # Standard game start
      gunboat.yaml               # No negotiation variant
      alliance_heavy.yaml        # Emphasis on cooperation
      betrayal_scenario.yaml     # High conflict setup

    prompts/
      system_prompts.py          # Power-specific personalities
      negotiation_prompts.py     # Communication templates

    utils/
      state_parser.py            # Game state parsing
      move_validator.py          # Action validation
      trust_calculator.py        # Trust metric computation

    tests/
      test_integration.py
      test_scoring.py
      test_memory.py

    AI_Diplomacy/                # Git submodule
```

## Core Components Design

### 1. DiplomacyEnv Class

```python
from typing import Dict, List, Optional, Tuple, Any
from atroposlib.envs.base import BaseEnv, BaseEnvConfig
from pydantic import Field
import asyncio
import websockets
import json

class DiplomacyEnvConfig(BaseEnvConfig):
    """Configuration for Diplomacy environment."""

    # Game configuration
    game_variant: str = Field(default="classic", description="Game variant to play")
    scenario: str = Field(default="classic_1901", description="Starting scenario")
    max_game_length: int = Field(default=20, description="Maximum game years")

    # Agent configuration
    powers_to_control: List[str] = Field(
        default=["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"],
        description="Which powers to control with AI"
    )
    negotiation_rounds: int = Field(default=3, description="Negotiation rounds per turn")
    message_max_length: int = Field(default=500, description="Max length per message")

    # Scoring configuration
    vrcli_weight: float = Field(default=0.3, description="Weight for VR-CLI reward")
    game_score_weight: float = Field(default=0.4, description="Weight for game outcome")
    diplomatic_score_weight: float = Field(default=0.3, description="Weight for diplomacy")

    # Memory configuration
    memory_top_k: int = Field(default=10, description="Number of memories to retrieve")
    relationship_decay: float = Field(default=0.95, description="Trust decay factor")

    # Game server configuration
    game_server_port: int = Field(default=8432, description="Port for game server")
    game_server_timeout: int = Field(default=300, description="Timeout in seconds")

    # Training configuration
    use_self_play: bool = Field(default=True, description="Use self-play training")
    include_human_players: bool = Field(default=False, description="Allow human players")

class DiplomacyEnv(BaseEnv):
    """Multi-agent Diplomacy environment."""

    name = "diplomacy"
    env_config_cls = DiplomacyEnvConfig

    def __init__(self, config: DiplomacyEnvConfig, server_configs, **kwargs):
        super().__init__(config, server_configs, **kwargs)
        self.game_wrapper = None
        self.episode_states = {}  # episode_id -> DiplomacyEpisodeState
        self.scenario_registry = DiplomacyScenarioRegistry()

    async def setup(self):
        """Initialize game server and connections."""
        self.game_wrapper = DiplomacyGameWrapper(self.config)
        await self.game_wrapper.start_server()

    async def get_next_item(self) -> Dict[str, Any]:
        """Get next game configuration for training."""
        scenario = self.scenario_registry.get_scenario(self.config.scenario)
        return {
            "episode_id": str(uuid.uuid4()),
            "scenario": scenario,
            "powers": self.config.powers_to_control,
        }

    async def collect_trajectory(self, item: Dict[str, Any]) -> Tuple[Optional[Dict], List]:
        """Run one complete game trajectory."""
        episode_state = await self.initialize_episode(item)

        # Play through the game
        while not episode_state.is_game_over():
            # Negotiation phase
            await self.run_negotiation_phase(episode_state)

            # Order generation phase
            await self.run_order_phase(episode_state)

            # Execute moves and update state
            await self.execute_turn(episode_state)

        # Calculate final scores
        scored_data = await self.calculate_episode_scores(episode_state)
        return scored_data, []
```

### 2. Episode State Management

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import numpy as np

@dataclass
class PowerState:
    """State for a single power."""
    power: str
    agent: Optional[DiplomacyAgent]
    supply_centers: Set[str] = field(default_factory=set)
    units: List[Dict] = field(default_factory=list)

    # Relationship tracking
    relationships: Dict[str, float] = field(default_factory=dict)  # -1 to 1
    promises_made: List[Dict] = field(default_factory=list)
    promises_received: List[Dict] = field(default_factory=list)

    # Memory
    negotiation_history: List[Dict] = field(default_factory=list)
    strategic_goals: List[str] = field(default_factory=list)

@dataclass
class DiplomacyEpisodeState:
    """Complete state for a Diplomacy episode."""
    episode_id: str
    scenario: Dict[str, Any]

    # Game state
    current_phase: str = "S1901M"  # Spring 1901 Movement
    board_state: Optional[Dict] = None
    power_states: Dict[str, PowerState] = field(default_factory=dict)

    # History
    move_history: List[Dict] = field(default_factory=list)
    negotiation_rounds: List[List[Dict]] = field(default_factory=list)

    # Scoring components
    vrcli_evaluator: Optional[Any] = None
    trust_matrix: np.ndarray = field(default_factory=lambda: np.ones((7, 7)))

    def is_game_over(self) -> bool:
        """Check if game has ended."""
        # Victory condition: 18+ supply centers
        for power_state in self.power_states.values():
            if len(power_state.supply_centers) >= 18:
                return True

        # Draw conditions
        year = int(self.current_phase[1:5])
        if year >= 1920:  # Time limit
            return True

        # Check for elimination/stalemate
        active_powers = [p for p in self.power_states.values() if len(p.units) > 0]
        if len(active_powers) <= 1:
            return True

        return False
```

### 3. Agent Architecture

```python
class DiplomacyAgent:
    """Agent wrapper for a single power in Diplomacy."""

    def __init__(self, power: str, server_client, config: DiplomacyEnvConfig):
        self.power = power
        self.server_client = server_client
        self.config = config

        # Components
        self.memory_manager = DiplomacyMemoryManager(power, config)
        self.relationship_tracker = RelationshipTracker(power)
        self.strategy_planner = StrategyPlanner(power)

        # Prompts
        self.system_prompt = self._get_system_prompt()

    async def negotiate(self,
                       game_state: Dict,
                       other_powers: List[str],
                       round_num: int) -> List[Dict[str, str]]:
        """Generate negotiation messages for one round."""

        # Retrieve relevant memories
        context = self._build_negotiation_context(game_state, other_powers)
        relevant_memories = self.memory_manager.retrieve_memories(context)

        # Build prompt
        prompt = self._build_negotiation_prompt(
            game_state, other_powers, round_num, relevant_memories
        )

        # Generate response
        response = await self._generate_response(prompt)

        # Parse messages and update memory
        messages = self._parse_negotiation_response(response)
        self.memory_manager.add_negotiation_memory(round_num, messages)

        return messages

    async def generate_orders(self,
                            game_state: Dict,
                            negotiation_history: List[Dict]) -> Dict[str, str]:
        """Generate movement orders based on game state and negotiations."""

        # Analyze negotiation outcomes
        negotiation_analysis = self.relationship_tracker.analyze_negotiations(
            negotiation_history
        )

        # Build strategic context
        context = self._build_order_context(game_state, negotiation_analysis)

        # Generate orders with reasoning
        prompt = self._build_order_prompt(context)
        response = await self._generate_response(prompt)

        # Parse and validate orders
        orders = self._parse_order_response(response)
        validated_orders = self._validate_orders(orders, game_state)

        return validated_orders

    def _build_negotiation_prompt(self, game_state, other_powers, round_num, memories):
        """Build prompt for negotiation phase."""
        return f"""
You are playing as {self.power} in a game of Diplomacy. Current phase: {game_state['phase']}.

Your current position:
- Supply Centers: {game_state['supply_centers'][self.power]}
- Units: {game_state['units'][self.power]}

Relevant memories from past interactions:
{self._format_memories(memories)}

Current relationships:
{self._format_relationships()}

This is negotiation round {round_num} of {self.config.negotiation_rounds}.

Generate your negotiation messages following this format:

<diplomacy_negotiation>
    <thinking>
        Analyze the current situation, your goals, and how to approach each power.
        Consider past interactions and current board position.
    </thinking>

    <memory>
        Key facts about current situation and relationships to remember.
    </memory>

    <messages>
        <message to="FRANCE" type="proposal">
            <content>Let's coordinate against Germany. I can support your move to Munich if you help me in the Balkans.</content>
            <commitment_level>medium</commitment_level>
            <truthfulness>high</truthfulness>
        </message>

        <message to="TURKEY" type="information">
            <content>Russia seems to be building fleets in the north. They might not be focused on you.</content>
            <commitment_level>none</commitment_level>
            <truthfulness>medium</truthfulness>
        </message>
    </messages>
</diplomacy_negotiation>
"""
```

### 4. Memory and Relationship System

```python
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from collections import defaultdict
from typing import Dict, List, Tuple

class DiplomacyMemoryManager:
    """Manages memories and retrieval for a Diplomacy agent."""

    def __init__(self, power: str, config: DiplomacyEnvConfig):
        self.power = power
        self.config = config
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        # Memory storage
        self.memories = []
        self.memory_embeddings = []
        self.index = None

        # Specialized memory types
        self.promises_made = defaultdict(list)
        self.promises_received = defaultdict(list)
        self.betrayals = []
        self.successful_cooperations = []

    def add_negotiation_memory(self, turn: str, messages: List[Dict]):
        """Store negotiation round in memory."""
        for msg in messages:
            memory = {
                "turn": turn,
                "type": "negotiation",
                "to": msg["to"],
                "content": msg["content"],
                "commitment": msg.get("commitment_level", "none"),
                "truthfulness": msg.get("truthfulness", "unknown")
            }

            # Track promises
            if memory["commitment"] in ["high", "medium"]:
                self.promises_made[msg["to"]].append(memory)

            self._add_memory(memory)

    def add_outcome_memory(self, turn: str, orders: Dict, results: Dict):
        """Store turn outcomes and evaluate promises."""
        # Check promise fulfillment
        for power, promise_list in self.promises_made.items():
            for promise in promise_list:
                if promise["turn"] == turn:
                    fulfilled = self._check_promise_fulfillment(promise, orders, results)
                    memory = {
                        "turn": turn,
                        "type": "promise_outcome",
                        "promise": promise,
                        "fulfilled": fulfilled
                    }
                    self._add_memory(memory)

                    if not fulfilled:
                        self.betrayals.append(memory)

    def retrieve_memories(self, context: str, k: int = None) -> List[Dict]:
        """Retrieve relevant memories for current context."""
        if k is None:
            k = self.config.memory_top_k

        if not self.memories:
            return []

        # Encode context
        context_embedding = self.encoder.encode([context])[0]

        # Search similar memories
        distances, indices = self.index.search(
            context_embedding.reshape(1, -1), k
        )

        return [self.memories[idx] for idx in indices[0]]

    def _add_memory(self, memory: Dict):
        """Add memory to storage with embedding."""
        # Create text representation
        text = f"{memory['type']} {memory.get('turn', '')} {memory.get('content', '')}"

        # Generate embedding
        embedding = self.encoder.encode([text])[0]

        # Store
        self.memories.append(memory)
        self.memory_embeddings.append(embedding)

        # Update index
        self._rebuild_index()

    def _rebuild_index(self):
        """Rebuild FAISS index with current embeddings."""
        if not self.memory_embeddings:
            return

        embeddings = np.array(self.memory_embeddings)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

class RelationshipTracker:
    """Tracks relationships and trust between powers."""

    def __init__(self, power: str):
        self.power = power
        self.trust_scores = defaultdict(float)  # -1 to 1
        self.interaction_history = defaultdict(list)

    def update_trust(self, other_power: str, event: str, impact: float):
        """Update trust score based on event."""
        old_trust = self.trust_scores[other_power]

        # Apply update with momentum
        self.trust_scores[other_power] = (
            0.8 * old_trust + 0.2 * np.clip(old_trust + impact, -1, 1)
        )

        # Record event
        self.interaction_history[other_power].append({
            "event": event,
            "impact": impact,
            "new_trust": self.trust_scores[other_power]
        })

    def get_relationship_summary(self) -> Dict[str, str]:
        """Get current relationship status with all powers."""
        relationships = {}
        for power, trust in self.trust_scores.items():
            if trust > 0.6:
                relationships[power] = "ally"
            elif trust > 0.2:
                relationships[power] = "friendly"
            elif trust > -0.2:
                relationships[power] = "neutral"
            elif trust > -0.6:
                relationships[power] = "unfriendly"
            else:
                relationships[power] = "enemy"
        return relationships
```

### 5. VR-CLI Scoring for Diplomacy

```python
class DiplomacyVRCLI:
    """VR-CLI scoring adapted for Diplomacy."""

    def __init__(self, server_client, config):
        self.server_client = server_client
        self.config = config

    async def score_negotiation_prediction(self,
                                         power: str,
                                         predicted_responses: Dict[str, str],
                                         actual_responses: Dict[str, str]) -> float:
        """Score how well agent predicted negotiation outcomes."""
        scores = []

        for other_power, predicted in predicted_responses.items():
            if other_power not in actual_responses:
                continue

            # Calculate perplexity improvement
            score = await self._calculate_vrcli_score(
                context=f"Negotiation from {other_power} to {power}",
                predicted=predicted,
                actual=actual_responses[other_power]
            )
            scores.append(score)

        return np.mean(scores) if scores else 0.0

    async def score_move_prediction(self,
                                  power: str,
                                  predicted_moves: Dict[str, List[str]],
                                  actual_moves: Dict[str, List[str]]) -> float:
        """Score how well agent predicted other powers' moves."""
        scores = []

        for other_power, predicted in predicted_moves.items():
            if other_power == power or other_power not in actual_moves:
                continue

            # Compare move sets
            score = await self._calculate_move_accuracy(
                predicted, actual_moves[other_power]
            )
            scores.append(score)

        return np.mean(scores) if scores else 0.0

    async def _calculate_vrcli_score(self, context: str, predicted: str, actual: str) -> float:
        """Calculate VR-CLI score for a prediction."""
        # Get perplexities
        ppl_with_prediction = await self._get_perplexity(context, predicted, actual)
        ppl_without = await self._get_perplexity(context, "", actual)

        # Calculate improvement percentage
        improvement = (1 - ppl_with_prediction / ppl_without) * 100

        # Map to discrete rewards
        if improvement < 0.05:
            return 0.0
        elif improvement < 1:
            return 0.5
        elif improvement < 2:
            return 0.9
        else:
            return 1.0
```

### 6. Game Wrapper Interface

```python
import subprocess
import asyncio
import websockets
import json
from typing import Dict, List, Optional

class DiplomacyGameWrapper:
    """Wrapper for AI_Diplomacy game server."""

    def __init__(self, config: DiplomacyEnvConfig):
        self.config = config
        self.process = None
        self.websocket = None
        self.game_id = None

    async def start_server(self):
        """Launch AI_Diplomacy server process."""
        self.process = subprocess.Popen([
            "python", "-m", "diplomacy.server.run",
            "--port", str(self.config.game_server_port),
            "--no-browser"
        ], cwd="environments/diplomacy_environment/AI_Diplomacy")

        # Wait for server to start
        await asyncio.sleep(5)

        # Connect websocket
        await self.connect()

    async def connect(self):
        """Establish websocket connection to game server."""
        uri = f"ws://localhost:{self.config.game_server_port}/ws"
        self.websocket = await websockets.connect(uri)

    async def create_game(self, scenario: Dict) -> str:
        """Create a new game with given scenario."""
        message = {
            "type": "create_game",
            "scenario": scenario,
            "variant": self.config.game_variant,
            "powers": self.config.powers_to_control
        }

        await self.websocket.send(json.dumps(message))
        response = json.loads(await self.websocket.recv())

        self.game_id = response["game_id"]
        return self.game_id

    async def submit_orders(self, power: str, orders: Dict[str, str]):
        """Submit orders for a power."""
        message = {
            "type": "submit_orders",
            "game_id": self.game_id,
            "power": power,
            "orders": orders
        }

        await self.websocket.send(json.dumps(message))
        response = json.loads(await self.websocket.recv())

        if not response["success"]:
            raise ValueError(f"Order submission failed: {response['error']}")

    async def process_turn(self) -> Dict:
        """Process current turn and get results."""
        message = {
            "type": "process_turn",
            "game_id": self.game_id
        }

        await self.websocket.send(json.dumps(message))
        response = json.loads(await self.websocket.recv())

        return response["results"]

    async def get_game_state(self) -> Dict:
        """Get current game state."""
        message = {
            "type": "get_state",
            "game_id": self.game_id
        }

        await self.websocket.send(json.dumps(message))
        response = json.loads(await self.websocket.recv())

        return response["state"]

    async def cleanup(self):
        """Clean up server process and connections."""
        if self.websocket:
            await self.websocket.close()

        if self.process:
            self.process.terminate()
            self.process.wait()
```

### 7. Scenario Registry

```python
class DiplomacyScenarioRegistry:
    """Registry for Diplomacy game scenarios."""

    def __init__(self):
        self.scenarios = {
            "classic_1901": self._classic_1901(),
            "gunboat": self._gunboat_diplomacy(),
            "alliance_heavy": self._alliance_focused(),
            "betrayal_training": self._betrayal_scenario(),
            "endgame_practice": self._endgame_scenario(),
        }

    def get_scenario(self, name: str) -> Dict:
        """Get scenario configuration by name."""
        if name == "random":
            return random.choice(list(self.scenarios.values()))
        return self.scenarios.get(name, self.scenarios["classic_1901"])

    def _classic_1901(self) -> Dict:
        """Standard 1901 game start."""
        return {
            "name": "Classic 1901",
            "description": "Standard Diplomacy opening",
            "starting_year": 1901,
            "starting_phase": "S1901M",
            "variant": "standard",
            "power_configs": {
                power: {"personality": "balanced"}
                for power in ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY",
                            "ITALY", "RUSSIA", "TURKEY"]
            }
        }

    def _gunboat_diplomacy(self) -> Dict:
        """No negotiation variant."""
        return {
            "name": "Gunboat Diplomacy",
            "description": "No communication between powers",
            "starting_year": 1901,
            "starting_phase": "S1901M",
            "variant": "gunboat",
            "negotiation_allowed": False,
            "power_configs": {
                power: {"personality": "aggressive"}
                for power in ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY",
                            "ITALY", "RUSSIA", "TURKEY"]
            }
        }

    def _alliance_focused(self) -> Dict:
        """Scenario encouraging long-term alliances."""
        return {
            "name": "Alliance Builder",
            "description": "Rewards stable alliances",
            "starting_year": 1901,
            "starting_phase": "S1901M",
            "variant": "standard",
            "scoring_modifiers": {
                "alliance_bonus": 2.0,
                "betrayal_penalty": 2.0
            },
            "power_configs": {
                power: {"personality": "cooperative"}
                for power in ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY",
                            "ITALY", "RUSSIA", "TURKEY"]
            }
        }
```

## Implementation Plan

### Phase 1: Intercepting Client Architecture (Current Focus)
- [x] Design intercepting client architecture
- [x] Document architecture in TODO.md
- [ ] Create InterceptingAtroposClient class
- [ ] Implement response sampling from policy server
- [ ] Add per-step scoring (random initially)
- [ ] Create ScoredDataGroup construction
- [ ] Implement canonical history management
- [ ] Add trajectory accumulation

### Phase 2: Environment Integration
- [ ] Create new DiplomacyEnv with proper collect_trajectories
- [ ] Integrate InterceptingAtroposClient with lm_game
- [ ] Implement mixed agent support (1 training, 6 fixed)
- [ ] Add basic credit assignment
- [ ] Test single episode collection
- [ ] Verify ScoredDataGroup correctness

### Phase 3: GRPO Correctness
- [ ] Ensure proper group structure (shared prefixes)
- [ ] Implement advantage calculation
- [ ] Add Monte Carlo returns with discounting
- [ ] Handle unselected alternatives with same action
- [ ] Test trajectory token limits
- [ ] Verify GRPO training compatibility

### Phase 4: Scoring and Rewards
- [ ] Design per-step reward function
- [ ] Implement negotiation quality scoring
- [ ] Add order validity checking
- [ ] Create diplomatic success metrics
- [ ] Integrate final game outcome
- [ ] Test reward distribution

### Phase 5: Production Features
- [ ] Add parallel game execution
- [ ] Implement proper evaluation mode
- [ ] Create visualization/debugging tools
- [ ] Add configuration flexibility
- [ ] Performance optimization
- [ ] Documentation and examples

### Legacy Phases (For Future Reference)
- Memory and Relationships (FAISS retrieval, trust tracking)
- Advanced diplomacy features (promise tracking, betrayal detection)
- Tournament and analysis tools
- Human player integration

## Technical Details

### Communication Protocol

#### Negotiation Message Format
```xml
<negotiation_message>
    <metadata>
        <from>ENGLAND</from>
        <to>FRANCE</to>
        <turn>S1901M</turn>
        <round>2</round>
        <type>proposal</type>
    </metadata>
    <content>
        I propose we form a Western Alliance against Germany.
        I'll move F LON-NTH and A LVP-EDI if you move A PAR-BUR.
    </content>
    <commitments>
        <commitment>
            <action>F LON-NTH</action>
            <condition>A PAR-BUR</condition>
            <strength>high</strength>
        </commitment>
    </commitments>
    <analysis>
        <truthfulness>0.8</truthfulness>
        <strategic_value>0.7</strategic_value>
    </analysis>
</negotiation_message>
```

#### Order Format
```xml
<orders>
    <metadata>
        <power>ENGLAND</power>
        <turn>S1901M</turn>
        <phase>movement</phase>
    </metadata>
    <order_list>
        <order unit="F LON" action="NTH" />
        <order unit="F EDI" action="NWG" />
        <order unit="A LVP" action="YOR" />
    </order_list>
    <expectations>
        <expected_support from="FRANCE" unit="F BRE" target="MAO" />
        <expected_hold from="GERMANY" unit="F KIE" />
    </expectations>
</orders>
```

### State Representation

```python
# Board state representation
board_state = {
    "phase": "S1901M",
    "year": 1901,
    "season": "spring",
    "type": "movement",

    "units": {
        "ENGLAND": [
            {"type": "F", "location": "LON"},
            {"type": "F", "location": "EDI"},
            {"type": "A", "location": "LVP"}
        ],
        # ... other powers
    },

    "supply_centers": {
        "ENGLAND": ["LON", "EDI", "LVP"],
        # ... other powers
    },

    "ownership": {
        "LON": "ENGLAND",
        "EDI": "ENGLAND",
        # ... all provinces
    }
}
```

### Scoring Components

1. **Game Score** (40% weight)
   - Supply center count
   - Survival bonus
   - Victory achievement
   - Position improvement

2. **VR-CLI Score** (30% weight)
   - Negotiation prediction accuracy
   - Move prediction accuracy
   - Alliance formation prediction
   - Betrayal timing prediction

3. **Diplomatic Score** (30% weight)
   - Successful negotiations
   - Promise keeping rate
   - Alliance stability
   - Deception effectiveness

### Performance Optimizations

1. **Parallel Game Execution**
   ```python
   async def run_parallel_games(self, num_games: int):
       tasks = []
       for _ in range(num_games):
           task = asyncio.create_task(self.run_single_game())
           tasks.append(task)

       results = await asyncio.gather(*tasks)
       return results
   ```

2. **Batched LLM Calls**
   - Group all power negotiations in single batch
   - Parallel order generation for all powers
   - Efficient memory retrieval with batch encoding

3. **State Caching**
   - Cache board evaluations
   - Reuse memory embeddings
   - Share encoder models across agents

## Testing Strategy

### Unit Tests
- [ ] Test game wrapper communication
- [ ] Test state parsing and validation
- [ ] Test memory storage and retrieval
- [ ] Test scoring calculations
- [ ] Test order validation

### Integration Tests
- [ ] Test full game execution
- [ ] Test multi-agent coordination
- [ ] Test scenario loading
- [ ] Test save/load functionality
- [ ] Test error recovery

### Performance Tests
- [ ] Measure game execution speed
- [ ] Test parallel game scaling
- [ ] Profile memory usage
- [ ] Benchmark LLM call efficiency
- [ ] Test with 100+ concurrent games

## Performance Considerations

### Bottlenecks
1. **LLM API Calls**
   - 7 powers × 3 negotiation rounds × 2 calls = 42 calls per turn
   - Solution: Batching, caching, parallel execution

2. **Game State Computation**
   - Complex move resolution
   - Solution: Efficient state representation, caching

3. **Memory Search**
   - Growing memory over long games
   - Solution: Efficient indexing, memory pruning

### Optimization Strategies
1. **Reduce API Calls**
   - Batch multiple powers in single call
   - Cache repeated contexts
   - Use smaller models for simple decisions

2. **Efficient State Management**
   - Incremental state updates
   - Compressed state representation
   - Fast state cloning for evaluation

3. **Smart Memory Management**
   - Limit memory size with importance sampling
   - Hierarchical memory (recent + important)
   - Periodic memory consolidation

## Future Enhancements

### Advanced Features
1. **Meta-Learning**
   - Learn optimal negotiation strategies
   - Adapt to opponent play styles
   - Transfer learning across scenarios

2. **Advanced Analysis**
   - Real-time strategy evaluation
   - Counterfactual reasoning
   - Post-game replay analysis

3. **Tournament Support**
   - ELO rating system
   - Matchmaking
   - League play

4. **Human Interface**
   - Web-based game viewer
   - Interactive negotiation interface
   - Real-time game commentary

### Research Directions
1. **Emergent Communication**
   - Study how agents develop communication protocols
   - Analyze deception patterns
   - Measure trust dynamics

2. **Multi-Agent RL**
   - Test different RL algorithms
   - Explore curriculum learning
   - Study equilibrium strategies

3. **LLM Capabilities**
   - Compare different model sizes
   - Test specialized fine-tuning
   - Explore few-shot learning

## Appendix: Key Code Snippets

### Running a Complete Game
```python
async def run_game(env: DiplomacyEnv):
    # Initialize
    item = await env.get_next_item()
    episode_state = await env.initialize_episode(item)

    # Game loop
    while not episode_state.is_game_over():
        # Negotiation
        for round_num in range(env.config.negotiation_rounds):
            messages = await env.run_negotiation_round(episode_state, round_num)
            episode_state.negotiation_rounds[-1].append(messages)

        # Orders
        all_orders = {}
        for power in env.config.powers_to_control:
            orders = await episode_state.power_states[power].agent.generate_orders(
                episode_state.board_state,
                episode_state.negotiation_rounds[-1]
            )
            all_orders[power] = orders

        # Execute
        results = await env.game_wrapper.process_turn()
        episode_state.update(results)

    # Score
    return await env.calculate_final_scores(episode_state)
```

### Memory Retrieval Example
```python
# During negotiation
context = f"Negotiating with {other_power} in {current_phase}"
memories = agent.memory_manager.retrieve_memories(context, k=5)

# Format for prompt
memory_text = "\n".join([
    f"- {m['turn']}: {m['content']} (trust: {m.get('trust_impact', 0)})"
    for m in memories
])
```

### Trust Update Example
```python
# After turn resolution
for power, orders in turn_results.items():
    # Check promises
    for promise in episode_state.power_states[power].promises_made:
        if promise['turn'] == current_turn:
            fulfilled = check_promise_fulfillment(promise, orders)

            # Update trust
            impact = 0.3 if fulfilled else -0.5
            episode_state.trust_matrix[promise['from']][promise['to']] += impact
```

## References

1. **TextWorld Environment**: Key patterns for episode management, VR-CLI scoring, memory systems
2. **AI_Diplomacy Project**: Game engine, multi-agent architecture, negotiation mechanics
3. **Diplomacy Research Papers**:
   - "No Press Diplomacy: Modeling Multi-Agent Gameplay" (Meta, 2021)
   - "Human-Level Play in Diplomacy" (Meta, 2022)
   - "Mastering the Game of No-Press Diplomacy" (DeepMind, 2020)

## Contact & Questions

For questions about implementation details or design decisions, refer to:
- TextWorld implementation: `environments/game_environments/textworld/`
- Base environment docs: `atroposlib/envs/base.py`
- Training examples: `example_trainer/`

This document will be updated as implementation progresses.
