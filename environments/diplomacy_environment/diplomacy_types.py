"""
Type definitions for the Diplomacy environment.

This module contains all the custom types and data structures used
throughout the Diplomacy environment implementation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field


class DiplomacyPhase(Enum):
    """Game phases in Diplomacy."""

    SPRING_NEGOTIATION = "spring_negotiation"
    SPRING_ORDERS = "spring_orders"
    SPRING_RETREATS = "spring_retreats"
    FALL_NEGOTIATION = "fall_negotiation"
    FALL_ORDERS = "fall_orders"
    FALL_RETREATS = "fall_retreats"
    FALL_BUILDS = "fall_builds"


class PowerRelationship(Enum):
    """Relationship status between powers."""

    ENEMY = "enemy"
    UNFRIENDLY = "unfriendly"
    NEUTRAL = "neutral"
    FRIENDLY = "friendly"
    ALLY = "ally"


class OrderType(Enum):
    """Types of orders in Diplomacy."""

    HOLD = "hold"
    MOVE = "move"
    SUPPORT = "support"
    CONVOY = "convoy"
    BUILD = "build"
    DISBAND = "disband"
    RETREAT = "retreat"


@dataclass
class Territory:
    """Represents a territory on the board."""

    name: str
    abbreviation: str
    is_supply_center: bool
    is_land: bool
    is_sea: bool
    adjacent: List[str] = field(default_factory=list)
    owner: Optional[str] = None


@dataclass
class Unit:
    """Represents a military unit."""

    unit_id: str
    power: str
    unit_type: str  # "army" or "fleet"
    location: str
    can_retreat_to: List[str] = field(default_factory=list)


@dataclass
class Order:
    """Represents an order for a unit."""

    unit_id: str
    order_type: OrderType
    source: str
    target: Optional[str] = None
    target_unit_id: Optional[str] = None
    convoy_path: Optional[List[str]] = None

    def to_string(self) -> str:
        """Convert order to standard Diplomacy notation."""
        if self.order_type == OrderType.HOLD:
            return f"{self.source} H"
        elif self.order_type == OrderType.MOVE:
            return f"{self.source} - {self.target}"
        elif self.order_type == OrderType.SUPPORT:
            if self.target_unit_id:
                return f"{self.source} S {self.target_unit_id}"
            else:
                return f"{self.source} S {self.target}"
        elif self.order_type == OrderType.CONVOY:
            return f"{self.source} C {self.target}"
        else:
            return str(self.order_type)


@dataclass
class NegotiationMessage:
    """Represents a message between powers."""

    message_id: str
    from_power: str
    to_powers: List[str]  # Can be multiple for broadcasts
    content: str
    phase: DiplomacyPhase
    turn: int
    timestamp: datetime
    message_type: str = "negotiation"  # negotiation, commitment, proposal


@dataclass
class GameState:
    """Current state of a Diplomacy game."""

    game_id: str
    variant: str
    year: int
    phase: DiplomacyPhase

    # Board state
    territories: Dict[str, Territory]
    units: Dict[str, Unit]
    supply_centers: Dict[str, int]  # Power -> number of centers

    # Current orders and results
    current_orders: Dict[str, List[Order]]
    order_results: Optional[Dict[str, str]] = None

    # Negotiation state
    messages_this_phase: List[NegotiationMessage] = field(default_factory=list)

    # History
    previous_turns: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PowerState:
    """State information for a single power."""

    power: str
    supply_centers: List[str]
    units: List[Unit]

    # Relationships and beliefs
    relationships: Dict[str, PowerRelationship]
    trust_scores: Dict[str, float]  # 0.0 to 1.0

    # Goals and strategy
    short_term_goals: List[str]
    long_term_goals: List[str]
    current_strategy: str

    # Memory
    diary_entries: List[str]
    important_events: List[Dict[str, Any]]


class DiplomacyState(BaseModel):
    """Complete state for RL training."""

    game_state: Dict[str, Any]  # Serialized GameState
    power_states: Dict[str, Dict[str, Any]]  # Serialized PowerStates
    current_power: str
    legal_actions: List[str]

    # Context for decision making
    recent_messages: List[Dict[str, Any]]
    recent_orders: List[Dict[str, Any]]

    # Metadata
    turn_number: int
    episode_id: str


class DiplomacyAction(BaseModel):
    """Action taken by an agent."""

    action_type: str  # "negotiate", "order", "build", "retreat"
    power: str

    # For negotiations
    messages: Optional[List[Dict[str, Any]]] = None

    # For orders
    orders: Optional[List[Dict[str, Any]]] = None

    # Reasoning
    reasoning: Optional[str] = None
    confidence: Optional[float] = None


class DiplomacyMemory(BaseModel):
    """Memory entry for episodic recall."""

    memory_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    episode_id: str
    power: str
    turn: int
    phase: str

    # Content
    summary: str
    details: Dict[str, Any]

    # Importance scoring
    importance: float = Field(default=0.5)

    # Embedding (will be computed)
    embedding: Optional[List[float]] = None


@dataclass
class DiplomacyEpisodeState:
    """Complete state for an episode."""

    episode_id: str
    scenario: Dict[str, Any]
    power_assignment: Dict[str, str]

    # Current state
    game_state: Optional[GameState] = None
    power_states: Dict[str, PowerState] = field(default_factory=dict)

    # Agents
    agents: Dict[str, Any] = field(default_factory=dict)  # Power -> Agent

    # Memory systems
    memories: Dict[str, List[DiplomacyMemory]] = field(default_factory=dict)

    # History
    turn_history: List[Dict[str, Any]] = field(default_factory=list)
    negotiation_history: List[NegotiationMessage] = field(default_factory=list)

    # Scoring
    current_scores: Dict[str, float] = field(default_factory=dict)
    final_outcome: Optional[Dict[str, Any]] = None


# Type aliases for clarity
OrderSet = List[Order]
MessageSet = List[NegotiationMessage]
PowerName = str
TerritoryName = str
