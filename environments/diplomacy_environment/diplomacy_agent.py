"""
Diplomacy Agent Implementation

This module implements the agent architecture for Diplomacy powers,
including negotiation, order generation, and memory management.
"""

import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

from .diplomacy_types import (
    DiplomacyAction,
    DiplomacyMemory,
    DiplomacyPhase,
    DiplomacyEpisodeState,
    GameState,
    NegotiationMessage,
    Order,
    OrderType,
    PowerRelationship,
    PowerState,
)
from .diplomacy_prompts import DiplomacyPrompts
from .memory_manager import DiplomacyMemoryManager

logger = logging.getLogger(__name__)


class DiplomacyAgentConfig(BaseModel):
    """Configuration for Diplomacy agents."""
    
    # Agent identity
    power: Optional[str] = Field(
        default=None,
        description="Which power this agent controls",
    )
    
    is_baseline: bool = Field(
        default=False,
        description="Whether this is a baseline agent (not trained)",
    )
    
    # Generation parameters
    temperature: float = Field(
        default=0.7,
        description="Temperature for action generation",
    )
    
    top_p: float = Field(
        default=0.9,
        description="Top-p for action generation",
    )
    
    max_thinking_tokens: int = Field(
        default=4096,
        description="Maximum tokens for thinking/reasoning",
    )
    
    max_message_tokens: int = Field(
        default=512,
        description="Maximum tokens per negotiation message",
    )
    
    # Strategy parameters
    personality_traits: Dict[str, float] = Field(
        default_factory=lambda: {
            "aggressive": 0.5,
            "trustworthy": 0.5,
            "cooperative": 0.5,
            "deceptive": 0.5,
        },
        description="Personality traits affecting behavior",
    )
    
    initial_strategy: str = Field(
        default="balanced",
        description="Initial strategic approach",
    )
    
    # Memory parameters
    use_memory: bool = Field(
        default=True,
        description="Whether to use episodic memory",
    )
    
    memory_top_k: int = Field(
        default=5,
        description="Number of memories to retrieve",
    )


class DiplomacyAgent:
    """
    Agent for a Diplomacy power.
    
    Handles:
    - Negotiation message generation
    - Order planning and generation
    - Memory management
    - Relationship tracking
    """
    
    def __init__(
        self,
        config: DiplomacyAgentConfig,
        episode_state: DiplomacyEpisodeState,
    ):
        self.config = config
        self.episode_state = episode_state
        self.power = config.power
        
        # Initialize memory manager if enabled
        if config.use_memory:
            self.memory_manager = DiplomacyMemoryManager(
                power=self.power,
                episode_id=episode_state.episode_id,
            )
        else:
            self.memory_manager = None
        
        # Initialize power state
        self.power_state = PowerState(
            power=self.power,
            supply_centers=[],
            units=[],
            relationships=self._initialize_relationships(),
            trust_scores=self._initialize_trust_scores(),
            short_term_goals=[],
            long_term_goals=[],
            current_strategy=config.initial_strategy,
            diary_entries=[],
            important_events=[],
        )
        
        # Cache for current phase planning
        self.current_phase_plan = None
        self.negotiation_commitments = {}
    
    def _initialize_relationships(self) -> Dict[str, PowerRelationship]:
        """Initialize neutral relationships with all powers."""
        all_powers = ["ENGLAND", "FRANCE", "GERMANY", "ITALY", 
                      "AUSTRIA", "RUSSIA", "TURKEY"]
        relationships = {}
        for power in all_powers:
            if power != self.power:
                relationships[power] = PowerRelationship.NEUTRAL
        return relationships
    
    def _initialize_trust_scores(self) -> Dict[str, float]:
        """Initialize neutral trust scores."""
        all_powers = ["ENGLAND", "FRANCE", "GERMANY", "ITALY", 
                      "AUSTRIA", "RUSSIA", "TURKEY"]
        trust_scores = {}
        for power in all_powers:
            if power != self.power:
                trust_scores[power] = 0.5
        return trust_scores
    
    async def generate_negotiation_alternatives(
        self,
        num_alternatives: int,
        **generation_kwargs,
    ) -> List[DiplomacyAction]:
        """Generate alternative negotiation strategies."""
        game_state = self.episode_state.game_state
        
        # Build context for negotiation
        context = await self._build_negotiation_context(game_state)
        
        # Generate alternatives
        alternatives = []
        for i in range(num_alternatives):
            # Get system prompt
            system_prompt = DiplomacyPrompts.get_negotiation_system_prompt(
                power=self.power,
                phase=game_state.phase,
                personality=self.config.personality_traits,
            )
            
            # Build user prompt with context
            user_prompt = self._build_negotiation_user_prompt(context)
            
            # Generate negotiation strategy
            response = await self._generate_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                **generation_kwargs,
            )
            
            # Parse response into action
            action = self._parse_negotiation_response(response)
            alternatives.append(action)
        
        return alternatives
    
    async def generate_order_alternatives(
        self,
        num_alternatives: int,
        **generation_kwargs,
    ) -> List[DiplomacyAction]:
        """Generate alternative order sets."""
        game_state = self.episode_state.game_state
        
        # Build context for orders
        context = await self._build_order_context(game_state)
        
        # Generate alternatives
        alternatives = []
        for i in range(num_alternatives):
            # Get system prompt
            system_prompt = DiplomacyPrompts.get_order_system_prompt(
                power=self.power,
                phase=game_state.phase,
                personality=self.config.personality_traits,
            )
            
            # Build user prompt with context
            user_prompt = self._build_order_user_prompt(context)
            
            # Generate orders
            response = await self._generate_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                **generation_kwargs,
            )
            
            # Parse response into action
            action = self._parse_order_response(response)
            alternatives.append(action)
        
        return alternatives
    
    async def _build_negotiation_context(
        self, game_state: GameState
    ) -> Dict[str, Any]:
        """Build context for negotiation decisions."""
        context = {
            "game_state": self._serialize_game_state(game_state),
            "power_state": self._serialize_power_state(),
            "recent_messages": self._get_recent_messages(),
            "relationships": dict(self.power_state.relationships),
            "trust_scores": dict(self.power_state.trust_scores),
            "commitments": dict(self.negotiation_commitments),
        }
        
        # Add relevant memories
        if self.memory_manager:
            query = f"Negotiation strategy for {self.power} in {game_state.phase}"
            memories = await self.memory_manager.retrieve_memories(
                query=query,
                top_k=self.config.memory_top_k,
            )
            context["relevant_memories"] = [m.dict() for m in memories]
        
        return context
    
    async def _build_order_context(
        self, game_state: GameState
    ) -> Dict[str, Any]:
        """Build context for order decisions."""
        context = {
            "game_state": self._serialize_game_state(game_state),
            "power_state": self._serialize_power_state(),
            "legal_moves": self._get_legal_moves(game_state),
            "negotiation_outcomes": self._get_negotiation_outcomes(),
            "commitments": dict(self.negotiation_commitments),
        }
        
        # Add relevant memories
        if self.memory_manager:
            query = f"Order strategy for {self.power} in {game_state.phase}"
            memories = await self.memory_manager.retrieve_memories(
                query=query,
                top_k=self.config.memory_top_k,
            )
            context["relevant_memories"] = [m.dict() for m in memories]
        
        return context
    
    def _serialize_game_state(self, game_state: GameState) -> Dict[str, Any]:
        """Serialize game state for LLM context."""
        return {
            "year": game_state.year,
            "phase": game_state.phase.value,
            "supply_centers": dict(game_state.supply_centers),
            "my_units": [
                {
                    "id": u.unit_id,
                    "type": u.unit_type,
                    "location": u.location,
                }
                for u in game_state.units.values()
                if u.power == self.power
            ],
            "other_units": [
                {
                    "id": u.unit_id,
                    "power": u.power,
                    "type": u.unit_type,
                    "location": u.location,
                }
                for u in game_state.units.values()
                if u.power != self.power
            ],
        }
    
    def _serialize_power_state(self) -> Dict[str, Any]:
        """Serialize power state for LLM context."""
        return {
            "power": self.power,
            "supply_centers": self.power_state.supply_centers,
            "num_units": len(self.power_state.units),
            "current_strategy": self.power_state.current_strategy,
            "short_term_goals": self.power_state.short_term_goals,
            "long_term_goals": self.power_state.long_term_goals,
        }
    
    def _get_recent_messages(self) -> List[Dict[str, Any]]:
        """Get recent negotiation messages."""
        messages = self.episode_state.game_state.messages_this_phase
        return [
            {
                "from": msg.from_power,
                "to": msg.to_powers,
                "content": msg.content,
                "type": msg.message_type,
            }
            for msg in messages[-10:]  # Last 10 messages
        ]
    
    def _get_legal_moves(self, game_state: GameState) -> Dict[str, List[str]]:
        """Get legal moves for our units."""
        # TODO: Implement legal move generation
        # This would interface with the Diplomacy engine
        return {}
    
    def _get_negotiation_outcomes(self) -> Dict[str, Any]:
        """Summarize negotiation outcomes."""
        return {
            "agreements_made": len(self.negotiation_commitments),
            "powers_negotiated_with": list(self.negotiation_commitments.keys()),
        }
    
    async def _generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs,
    ) -> str:
        """Generate LLM response."""
        # TODO: Implement actual LLM call
        # This would use the server_client from episode_state
        return "<think>Thinking about strategy...</think><response>Generated response</response>"
    
    def _parse_negotiation_response(self, response: str) -> DiplomacyAction:
        """Parse LLM response into negotiation action."""
        # TODO: Implement XML parsing for negotiation format
        # Expected format:
        # <think>...</think>
        # <negotiation>
        #   <message to="FRANCE">Let's work together...</message>
        #   <message to="GERMANY">I'm concerned about...</message>
        # </negotiation>
        
        return DiplomacyAction(
            action_type="negotiate",
            power=self.power,
            messages=[],
            reasoning=response,
        )
    
    def _parse_order_response(self, response: str) -> DiplomacyAction:
        """Parse LLM response into order action."""
        # TODO: Implement XML parsing for order format
        # Expected format:
        # <think>...</think>
        # <orders>
        #   <order unit="F LON" type="move" target="NTH"/>
        #   <order unit="A LVP" type="move" target="EDI"/>
        # </orders>
        
        return DiplomacyAction(
            action_type="order",
            power=self.power,
            orders=[],
            reasoning=response,
        )
    
    def update_state_after_turn(
        self,
        game_state: GameState,
        turn_results: Dict[str, Any],
    ) -> None:
        """Update agent state after a turn."""
        # Update power state
        self.power_state.supply_centers = [
            t for t, owner in game_state.territories.items()
            if owner == self.power and game_state.territories[t].is_supply_center
        ]
        
        self.power_state.units = [
            u for u in game_state.units.values()
            if u.power == self.power
        ]
        
        # Update relationships based on actions
        self._update_relationships(turn_results)
        
        # Create memory of this turn
        if self.memory_manager:
            memory = DiplomacyMemory(
                episode_id=self.episode_state.episode_id,
                power=self.power,
                turn=game_state.year,
                phase=game_state.phase.value,
                summary=self._summarize_turn(turn_results),
                details=turn_results,
                importance=self._calculate_turn_importance(turn_results),
            )
            self.memory_manager.add_memory(memory)
    
    def _update_relationships(self, turn_results: Dict[str, Any]) -> None:
        """Update relationships based on turn outcomes."""
        # TODO: Implement relationship updates based on:
        # - Support received/given
        # - Attacks
        # - Negotiation outcomes
        pass
    
    def _summarize_turn(self, turn_results: Dict[str, Any]) -> str:
        """Create a summary of the turn for memory."""
        # TODO: Implement turn summarization
        return f"Turn {turn_results.get('year', '?')} completed"
    
    def _calculate_turn_importance(self, turn_results: Dict[str, Any]) -> float:
        """Calculate importance score for turn memory."""
        # TODO: Implement importance calculation based on:
        # - Territory changes
        # - Unit losses/gains
        # - Relationship changes
        return 0.5