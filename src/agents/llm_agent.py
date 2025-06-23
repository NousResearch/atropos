"""
LLM-driven Agent implementation for Society Simulation
"""

import asyncio
import logging
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np
import random
from mesa import Agent

from src.economics.market_system import ResourceType, TradeOrderType
from src.economics.banking_system import AccountType, LoanType, TransactionType
from src.flame_gpu.flame_gpu_simulation import AgentType, CulturalGroup
# Attempt to import WorldObject, if it's not directly available,
# the agent will rely on objects having a .description attribute.
# from src.simulation.society_simulator import WorldObject # This might cause circular dependency if not careful

logger = logging.getLogger(__name__)

class AgentState(Enum):
    """Agent behavioral states"""
    IDLE = "idle"
    MOVING = "moving"
    SOCIALIZING = "socializing"
    WORKING = "working"
    CREATING = "creating"
    THINKING = "thinking"

@dataclass
class Memory:
    """Agent memory item"""
    content: str
    timestamp: float
    importance: float = 0.5
    tags: List[str] = field(default_factory=list)
    agent_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        # Use asdict for shallow conversion, then handle specific fields if needed (like enums)
        # For Memory, all fields are currently primitive or list of primitives, so asdict is fine.
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        # Ensure all fields required by __init__ are present or provide defaults
        # The dataclass __init__ will handle this if data keys match field names.
        # For `tags`, ensure it's a list.
        data["tags"] = list(data.get("tags", []))
        return cls(**data)

@dataclass
class Position:
    """3D position with utilities"""
    x: float
    y: float
    z: float = 0.0
    
    def distance_to(self, other: "Position") -> float:
        """Calculate Euclidean distance to another position"""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
    
    def move_towards(self, target: "Position", speed: float) -> "Position":
        """Move towards target position with given speed"""
        direction = np.array([target.x - self.x, target.y - self.y, target.z - self.z])
        distance = np.linalg.norm(direction)
        
        if distance <= speed:
            return Position(target.x, target.y, target.z)
        
        direction = direction / distance * speed
        return Position(self.x + direction[0], self.y + direction[1], self.z + direction[2])

    def to_dict(self) -> Dict[str, Any]:
        return {"x": self.x, "y": self.y, "z": self.z}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Position":
        return cls(x=data.get("x", 0.0), y=data.get("y", 0.0), z=data.get("z", 0.0))

class LLMAgent(Agent):
    """
    LLM-driven agent for society simulation
    
    Inherits from Mesa Agent for compatibility
    """
    
    def __init__(
        self,
        model,
        unique_id: str,
        llm_coordinator,
        config,
        position: Optional[Position] = None,
        persona: Optional[str] = None,
        initial_age: float = 25.0, 
        initial_health: float = 1.0,
        initial_employed_status: int = 0 # 0 for unemployed, 1 for employed
    ):
        super().__init__(model)
        
        # Core attributes
        self.unique_id = unique_id
        self.config = config
        self.llm_coordinator = llm_coordinator
        
        # Generate persona and determine agent_type from it
        # _generate_random_persona will now return a tuple (persona_str, agent_type_enum)
        generated_persona, determined_agent_type = self._generate_random_persona()
        self.persona = persona or generated_persona
        self.agent_type: AgentType = determined_agent_type
        
        # Spatial attributes
        self.position = position or Position(
            random.uniform(0, config.simulation.world_size[0]),
            random.uniform(0, config.simulation.world_size[1]),
            0.0
        )
        self.target_position: Optional[Position] = None
        self.movement_speed = config.agents.movement_speed
        
        # Behavioral attributes
        self.state = AgentState.IDLE
        self.energy = 1.0
        self.happiness = 0.5
        self.social_connections: Dict[str, float] = {}  # agent_id -> connection strength
        self.social_reputation: float = 0.5 # Default from FlameGPU initial state

        # New core attributes
        self.age: float = initial_age 
        self.health: float = max(0.0, min(1.0, initial_health)) # Clamp health 0-1
        self.employed: int = initial_employed_status # 0 or 1

        # Family & Cultural attributes from FlameGPU
        self.family_id: Optional[int] = None
        self.cultural_group_id: Optional[int] = None # Matches FlameGPU CulturalGroup enum
        self.cultural_affinities: Dict[str, float] = { # Matches FlameGPU cultural_affinity_... names
            "harmonists": 0.2,
            "builders": 0.2,
            "guardians": 0.2,
            "scholars": 0.2,
            "wanderers": 0.2
        }

        # Economic attributes from FlameGPU
        self.credit_score: float = 700.0 # Default from FlameGPU initial state
        self.total_debt: float = 0.0
        self.monthly_income: float = 0.0
        
        # Memory system
        self.memories: List[Memory] = []
        self.memory_size = config.agents.memory_size
        
        # Resources and inventory
        self.resources: Dict[str, int] = {"food": 10, "materials": 5, "tools": 1, "currency": 500, "energy_item": 5}
        self.inventory: List[str] = []
        
        # LLM interaction
        self.last_llm_call = 0.0
        self.llm_cache: Dict[str, str] = {}
        self.conversation_history: List[Dict[str, str]] = []
        self.last_market_research_result: Optional[str] = None # For recent research
        self.last_banking_statement: Optional[str] = None # For recent banking statement
        
        # Metrics
        self.step_count = 0
        self.social_interactions = 0
        self.objects_created = 0
        
        logger.debug(f"Created agent {self.unique_id} (Type: {self.agent_type.name}, Age: {self.age:.1f}, Health: {self.health:.2f}, Employed: {self.employed}) persona: {self.persona[:50]}...")
    
    def _generate_random_persona(self) -> Tuple[str, AgentType]:
        traits = [
            "curious", "creative", "analytical", "social", "introverted", 
            "ambitious", "peaceful", "adventurous", "cautious", "optimistic"
        ]
        # Professions mapped to AgentType enum names (ensure these are valid AgentType names)
        professions_map = {
            "artist": AgentType.UNEMPLOYED, # Placeholder, map to a relevant type or extend AgentType
            "scientist": AgentType.SCHOLAR,
            "trader": AgentType.TRADER,
            "builder": AgentType.CRAFTSMAN, # Assuming CRAFTSMAN is like a builder
            "explorer": AgentType.UNEMPLOYED, # Placeholder
            "teacher": AgentType.SCHOLAR,
            "inventor": AgentType.CRAFTSMAN, # Or SCHOLAR
            "farmer": AgentType.FARMER,
            "craftsperson": AgentType.CRAFTSMAN,
            "storyteller": AgentType.SCHOLAR, # Or UNEMPLOYED
            "leader": AgentType.LEADER
        }
        profession_names = list(professions_map.keys())
        
        trait = random.choice(traits)
        chosen_profession_name = random.choice(profession_names)
        determined_agent_type = professions_map.get(chosen_profession_name, AgentType.UNEMPLOYED)
        
        persona_str = f"I am a {trait} {chosen_profession_name} who enjoys learning and creating new things."
        return persona_str, determined_agent_type
    
    async def step(self):
        """Execute one simulation step"""
        self.step_count += 1
        
        try:
            # Update internal state
            await self._update_state()
            
            # Decide on action
            action = await self._decide_action()
            
            # Execute action
            await self._execute_action(action)
            
            # Update energy and happiness
            self._update_resources()
            
            # Clean up old memories
            await self._manage_memory()
            
        except Exception as e:
            logger.error(f"Error in agent {self.unique_id} step: {e}", exc_info=True)
    
    async def _update_state(self):
        """Update agent's internal state based on environment"""
        # Get nearby agents
        nearby_agents = self._get_nearby_agents()
        
        # Update social connections
        for agent in nearby_agents:
            if agent.unique_id not in self.social_connections:
                self.social_connections[agent.unique_id] = 0.1
            else:
                # Strengthen connection through proximity
                self.social_connections[agent.unique_id] = min(
                    1.0, self.social_connections[agent.unique_id] + 0.01
                )
        
        # Update happiness based on social connections
        social_factor = min(1.0, len(self.social_connections) * 0.1)
        self.happiness = 0.3 * self.happiness + 0.7 * social_factor
    
    async def _decide_action(self) -> Dict[str, Any]:
        """Decide what action to take this step"""
        situation_env_summary = self._get_situation_summary()
        
        # Create prompt for LLM, now awaiting it
        prompt = await self._create_decision_prompt(situation_env_summary)
        
        try:
            response = await self.llm_coordinator.get_response(
                agent_id=self.unique_id,
                prompt=prompt,
                max_tokens=self.config.llm.max_tokens,
                temperature=getattr(self.config.llm, 'temperature', 0.7)
            )
            action = self._parse_llm_response(response)
        except Exception as e:
            logger.warning(f"LLM decision failed for {self.unique_id}: {e}", exc_info=True)
            action = self._fallback_decision()
        return action
    
    async def _create_decision_prompt(self, situation_summary: str) -> str:
        """Create a prompt for the LLM to decide on action (now async)"""
        cultural_group_name = "Unknown"
        if self.cultural_group_id is not None:
            try: cultural_group_name = CulturalGroup(self.cultural_group_id).name
            except ValueError: logger.warning(f"Agent {self.unique_id} invalid cultural_group_id: {self.cultural_group_id}."); cultural_group_name = f"Group ID {self.cultural_group_id}"
            except Exception as e: logger.error(f"Error resolving cultural group name for {self.unique_id}: {e}"); cultural_group_name = f"Group ID {self.cultural_group_id} (Error)"
        
        market_passive_summary = "Market info unavailable."; banking_passive_summary = "Banking info unavailable."; family_passive_summary = "Family info unavailable."
        if hasattr(self.model, 'market_system'):
            try:
                market_info_strings = []
                for res_type_enum in [ResourceType.FOOD, ResourceType.TOOLS, ResourceType.MATERIALS]:
                    market_info = self.model.market_system.get_resource_market_summary(res_type_enum)
                    if market_info: market_info_strings.append(str(market_info))
                if market_info_strings: market_passive_summary = f"Market Info: {'; '.join(market_info_strings)}."
            except Exception as e: logger.debug(f"Could not fetch passive market summary for {self.unique_id}: {e}")
        if hasattr(self.model, 'banking_system') and hasattr(self.model.banking_system, 'get_concise_account_summary_for_llm'):
            try:
                banking_summary = self.model.banking_system.get_concise_account_summary_for_llm(self.unique_id)
                if banking_summary: banking_passive_summary = banking_summary
            except Exception as e: logger.debug(f"Could not fetch passive banking summary for {self.unique_id}: {e}")
        if hasattr(self.model, 'family_system') and hasattr(self.model.family_system, 'get_concise_family_summary_for_llm'):
            try:
                family_summary = self.model.family_system.get_concise_family_summary_for_llm(self.unique_id)
                if family_summary: family_passive_summary = family_summary
            except Exception as e: logger.debug(f"Could not fetch passive family summary for {self.unique_id}: {e}")

        # _format_recent_memories is async, so await it here
        formatted_recent_memories = await self._format_recent_memories()

        prompt_context = f"""You are {self.persona}

Current Situation Overview:
{situation_summary} 

Your Current State:
  Energy: {self.energy:.2f}, Happiness: {self.happiness:.2f}, Health: {self.health:.2f}
  Age: {self.age:.1f} simulation years/cycles
  Employment Status: {'Employed' if self.employed == 1 else 'Unemployed'}
  Status: {self.state.value}
  Position: ({self.position.x:.1f}, {self.position.y:.1f}, {self.position.z:.1f})

Social & Cultural Context:
  Social Reputation: {self.social_reputation:.2f}
  Family ID (if any): {self.family_id if self.family_id is not None else 'N/A'}
  Cultural Group: {cultural_group_name}
  Cultural Affinities: {', '.join([f'{k}: {v:.2f}' for k, v in self.cultural_affinities.items()])}
  {family_passive_summary}

Economic Context:
Resources: {self.resources}
  Credit Score: {self.credit_score:.0f}
  Total Debt: {self.total_debt:.2f}
  Monthly Income: {self.monthly_income:.2f}
  {market_passive_summary}
  {banking_passive_summary}

Recent Memories:
{formatted_recent_memories}
"""
        active_info_section = ""
        if self.last_market_research_result: active_info_section += f"\nRecent Market Research Result:\n{self.last_market_research_result}\n"; self.last_market_research_result = None
        if self.last_banking_statement: active_info_section += f"\nRecent Banking Statement:\n{self.last_banking_statement}\n"; self.last_banking_statement = None
        
        prompt_actions = """\nWhat do you want to do next? Choose from:
- move_to <x> <y> (e.g., "move_to 10.5 23.0")
- talk_to <agent_id> (e.g., "talk_to agent_3")
- create_object <description> (e.g., "create_object a small wooden bird statue")
- gather_resources (e.g., "gather_resources")
- rest (e.g., "rest")
- family_interact_spouse <description> (e.g., "family_interact_spouse discuss buying a new table")
- family_interact_child <child_id> <description> (e.g., "family_interact_child child_7 help with homework")
- family_manage_household <description> (e.g., "family_manage_household plan weekend activities")
- market_trade <buy/sell> <resource> <qty> <price> (e.g., "market_trade buy food 5 2.5")
- market_research <resource | all> (e.g., "market_research food" or "market_research all")
- banking_action <deposit/withdraw/apply_loan/pay_loan> <amount_or_loan_id> [amount_if_pay_loan_or_details_if_apply] (e.g., "banking_action deposit 100", "banking_action pay_loan loan_id_123 50")
- get_banking_statement (get your detailed bank account and loan status)

Respond with just the action and parameters, like "move_to 10 15" or "talk_to agent_123".
"""
        return prompt_context + active_info_section + prompt_actions
    
    def _get_situation_summary(self) -> str:
        """Get a summary of the immediate environment (position, nearby entities)."""
        nearby_agents = self._get_nearby_agents()
        nearby_world_objects = self._get_nearby_objects()
        
        summary_parts = [] 
        summary_parts.append(f"You are at position ({self.position.x:.1f}, {self.position.y:.1f}, {self.position.z:.1f}).")
        
        if nearby_agents:
            agent_names = [a.unique_id for a in nearby_agents[:3]]
            summary_parts.append(f"Nearby agents: {', '.join(agent_names)}.")
        else:
            summary_parts.append("No agents are nearby.") # More explicit if none
        
        if nearby_world_objects:
            object_descriptions = [obj.description for obj in nearby_world_objects[:3] if hasattr(obj, 'description')]
            if object_descriptions:
                summary_parts.append(f"Nearby objects include: {', '.join(object_descriptions)}.")
            else:
                summary_parts.append("You see no specific objects nearby that can be described.")
        else:
            summary_parts.append("You see no specific objects nearby.")
        
        return " ".join(summary_parts)
    
    def _get_nearby_agents(self) -> List["LLMAgent"]:
        """Get agents within social radius"""
        nearby = []
        social_r = getattr(self.config.agents, 'social_radius', 10.0)
        for agent in self.model.schedule.agents:
            if agent.unique_id != self.unique_id and self.position.distance_to(agent.position) <= social_r:
                    nearby.append(agent)
        return nearby
    
    def _get_nearby_objects(self) -> List[Any]: # Changed return type to List[Any] to avoid import issue for now, expecting objects with .description
        """Get objects within interaction radius"""
        if hasattr(self.model, 'get_objects_near') and self.position:
            interaction_r = getattr(self.config.agents, 'interaction_radius', 5.0) 
            return self.model.get_objects_near(self.position, interaction_r)
        return []
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into action dictionary with enhanced validation."""
        parts = response.strip().split()
        action_type = parts[0].lower() if parts else "rest"
        action = {"type": action_type, "params": {}}
        valid_action = True
        error_msg = ""

        try:
            if action_type == "move_to":
                if len(parts) == 3:
                    action["params"]["x"] = float(parts[1])
                    action["params"]["y"] = float(parts[2])
                else: valid_action = False; error_msg = "move_to requires 2 float parameters (x y)."
            elif action_type == "talk_to":
                if len(parts) == 2 and parts[1].startswith("agent_"):
                    action["params"]["target_id"] = parts[1]
                else: valid_action = False; error_msg = "talk_to requires valid agent_id (e.g., agent_X)."
            elif action_type == "create_object":
                if len(parts) > 1:
                    action["params"]["description"] = " ".join(parts[1:])
                else: valid_action = False; error_msg = "create_object requires a description."
            elif action_type == "gather_resources" or action_type == "rest" or action_type == "get_banking_statement":
                if len(parts) != 1: valid_action = False; error_msg = f"{action_type} requires no parameters."
                # No params to validate beyond command length
            elif action_type == "family_interact_spouse" or action_type == "family_manage_household":
                if len(parts) > 1:
                    action["params"]["description"] = " ".join(parts[1:])
                else: valid_action = False; error_msg = f"{action_type} requires a description."
            elif action_type == "family_interact_child":
                if len(parts) == 3 and parts[1].startswith("agent_"): # Basic child_id format check
                    action["params"]["child_id"] = parts[1]
                    action["params"]["description"] = parts[2]
                else: valid_action = False; error_msg = "family_interact_child requires child_id (e.g. agent_X) and description."
            elif action_type == "market_trade":
                if len(parts) == 5:
                    order_type_str = parts[1].lower()
                    if order_type_str not in [ot.value for ot in TradeOrderType]: valid_action = False; error_msg = f"Invalid order_type for market_trade: {order_type_str}. Must be 'buy' or 'sell'."
                    else: action["params"]["order_type"] = order_type_str
                    
                    resource_str = parts[2].lower()
                    try: ResourceType(resource_str); action["params"]["resource"] = resource_str
                    except ValueError: valid_action = False; error_msg = f"Invalid resource_name for market_trade: {resource_str}."
                    
                    quantity = float(parts[3]); price = float(parts[4])
                    if quantity <= 0 or price <= 0: valid_action = False; error_msg = "Quantity and price for market_trade must be positive."
                    else: action["params"]["quantity"] = quantity; action["params"]["price"] = price
                else: valid_action = False; error_msg = "market_trade requires 4 parameters: <buy/sell> <resource> <qty> <price>."
            elif action_type == "market_research":
                if len(parts) == 2:
                    target_str = parts[1].lower()
                    if target_str == "all" or any(rt.value == target_str for rt in ResourceType):
                        action["params"]["target"] = target_str
                    else: valid_action = False; error_msg = f"Invalid target for market_research: {target_str}. Must be 'all' or a valid resource name."
                else: valid_action = False; error_msg = "market_research requires 1 parameter: <resource_name | all>."
            elif action_type == "banking_action":
                if len(parts) >= 3:
                    sub_action = parts[1].lower()
                    action["params"]["sub_action"] = sub_action
                    if sub_action == "pay_loan":
                        if len(parts) == 4:
                            action["params"]["loan_id"] = parts[2]
                            action["params"]["amount"] = float(parts[3])
                            if float(parts[3]) <= 0: valid_action = False; error_msg = "Payment amount for pay_loan must be positive."
                        else: valid_action = False; error_msg = "banking_action pay_loan requires <loan_id> <amount>."
                    elif sub_action in ["deposit", "withdraw"]:
                        if len(parts) == 3:
                            action["params"]["amount"] = float(parts[2])
                            if float(parts[2]) <= 0: valid_action = False; error_msg = f"Amount for {sub_action} must be positive."
                        else: valid_action = False; error_msg = f"banking_action {sub_action} requires <amount>."
                    elif sub_action == "apply_loan":
                        action["params"]["amount"] = float(parts[2])
                        if float(parts[2]) <= 0: valid_action = False; error_msg = "Loan amount for apply_loan must be positive."
                        if len(parts) > 3: action["params"]["loan_details"] = " ".join(parts[3:])
                        # else no details, which is fine for apply_loan
                    else: valid_action = False; error_msg = f"Invalid sub_action for banking_action: {sub_action}."
                else: valid_action = False; error_msg = "banking_action requires at least <sub_action> <param1>."
            else: # Unrecognized action type
                valid_action = False; error_msg = f"Unrecognized action type: {action_type}."

        except ValueError as e: # Catch float conversion errors etc.
            valid_action = False
            error_msg = f"Invalid parameter format for {action_type}. Expected numbers where appropriate. Details: {e}"
        
        if not valid_action:
            logger.warning(f"Agent {self.unique_id} LLM response invalid: '{response}'. Reason: {error_msg}. Defaulting to rest.")
            self._add_memory(f"Tried action '{response}', but it was invalid ({error_msg}). Resting instead.", importance=0.6)
            return {"type": "rest", "params": {}} # Default to rest
            
        return action
    
    def _fallback_decision(self) -> Dict[str, Any]:
        """Simple rule-based decision when LLM fails"""
        if self.energy < 0.3:
            return {"type": "rest"}
        
        if random.random() < 0.3:
            # Random movement
            x = random.uniform(0, self.config.simulation.world_size[0])
            y = random.uniform(0, self.config.simulation.world_size[1])
            return {"type": "move", "target": Position(x, y)}
        
        nearby_agents = self._get_nearby_agents()
        if nearby_agents and random.random() < 0.5:
            target = random.choice(nearby_agents)
            return {"type": "talk", "target_id": target.unique_id}
        
        return {"type": "rest"}
    
    async def _execute_action(self, action: Dict[str, Any]):
        """Execute the decided action and log it as a simulation event."""
        action_type = action.get("type", "rest")
        params = action.get("params", {})
        
        # Log the chosen action as a simulation event
        if hasattr(self.model, 'database_handler') and self.model.database_handler:
            event_details = {"action_type": action_type, "parameters": params}
            event_description = f"Agent {self.unique_id} chose action: {action_type}"
            if params:
                event_description += f" with params: {str(params)[:100]}" # Truncate long params
            try:
                await self.model.database_handler.save_simulation_event(
                    event_type="AGENT_ACTION_CHOSEN",
                    step=self.model.current_step,
                    agent_id_primary=self.unique_id,
                    details=event_details,
                    description=event_description
                )
            except Exception as e_event_log:
                logger.error(f"Agent {self.unique_id}: Failed to log AGENT_ACTION_CHOSEN event to DB: {e_event_log}", exc_info=True)

        self.state = AgentState.THINKING # Generic state during action execution
        
        if action_type == "move_to":
            target_pos = Position(params["x"], params["y"], self.position.z)
            await self._execute_move(target_pos)
        elif action_type == "talk_to":
            await self._execute_talk(params["target_id"])
        elif action_type == "create_object":
            await self._execute_create(params["description"])
        elif action_type == "gather_resources":
            await self._execute_gather()
        elif action_type == "rest":
            await self._execute_rest()
        elif action_type == "family_interact_spouse":
            await self._execute_family_interaction_spouse(params.get("description"))
        elif action_type == "family_interact_child":
            await self._execute_family_interaction_child(params.get("child_id"), params.get("description"))
        elif action_type == "family_manage_household":
            await self._execute_family_interaction_household(params.get("description"))
        elif action_type == "market_trade":
            await self._execute_market_trade(
                params.get("order_type"), 
                params.get("resource"), 
                params.get("quantity"), 
                params.get("price")
            )
        elif action_type == "market_research":
            await self._execute_market_research(params.get("target"))
        elif action_type == "banking_action":
            await self._execute_banking_action(
                params.get("sub_action"), 
                params.get("amount"), 
                params.get("loan_details"),
                params.get("loan_id")
            )
        elif action_type == "get_banking_statement":
            await self._execute_get_banking_statement()
        else:
            logger.warning(f"Agent {self.unique_id} attempted unknown action type in _execute_action: {action_type}")
            await self._execute_rest() # Default to rest for unknown actions
            
        self.state = AgentState.IDLE # Reset to idle after action, or specific state
    
    async def _execute_move(self, target: Position):
        """Execute movement action"""
        self.target_position = target
        self.position = self.position.move_towards(target, self.movement_speed)
        self.energy -= 0.01  # Movement costs energy
        
        await self._add_memory(f"Moved towards ({target.x:.1f}, {target.y:.1f})", importance=0.1)
    
    async def _execute_talk(self, target_id: str):
        """Execute social interaction"""
        # Find target agent
        target_agent = next((a for a in self.model.schedule.agents if a.unique_id == target_id), None)
        
        if target_agent:
            distance = self.position.distance_to(target_agent.position)
            if distance <= getattr(self.config.agents, 'social_radius', 10.0):
                # Successful interaction
                self.social_interactions += 1
                self.happiness += 0.05
                
                # Generate conversation
                conversation = await self._generate_conversation(target_agent)
                await self._add_memory(f"Talked with {target_id}: {conversation[:50]}...", importance=0.6)
                
                # Update relationship
                if target_id in self.social_connections:
                    self.social_connections[target_id] += 0.1
                else:
                    self.social_connections[target_id] = 0.2
    
    async def _execute_create(self, description: str):
        """Execute object creation with 3D asset generation"""
        self.objects_created += 1
        self.energy -= 0.1 # Cost of creation
        self.happiness += 0.1 # Joy of creation
        
        logger.info(f"Agent {self.unique_id} attempting to create: {description}")

        if not hasattr(self.model, 'asset_manager') or not self.model.asset_manager:
            await self._add_memory(f"Attempted to create: {description} (asset system unavailable)", importance=0.5)
            logger.warning(f"AssetManager not found on model for agent {self.unique_id}")
            return

        try:
            # Determine complexity based on description
            complexity = "simple"
            if any(word in description.lower() for word in ["complex", "detailed", "intricate"]):
                complexity = "complex"
            
            # Call AssetManager to generate the asset
            created_asset = await self.model.asset_manager.create_asset_for_agent(
                agent_id=self.unique_id,
                description=description,
                complexity=complexity
            )
            
            if created_asset:
                self.inventory.append(created_asset.asset_id) # Add asset ID to inventory
                await self._add_memory(f"Created 3D asset: {description} (ID: {created_asset.asset_id}) at {created_asset.file_path}", importance=0.9)
                logger.info(f"Agent {self.unique_id} successfully generated asset: {created_asset.asset_id} - {description}")

                # Add the created asset to the world via the simulation model
                if hasattr(self.model, 'add_world_object'):
                    world_object = self.model.add_world_object(
                        asset=created_asset, 
                        position=self.position, # Pass the Position object directly
                        creator_id=self.unique_id
                    )
                    logger.info(f"Agent {self.unique_id} placed object {world_object.obj_id} in the world at {(self.position.x, self.position.y, self.position.z)}")
                else:
                    logger.warning(f"Model does not have add_world_object method. Cannot place {created_asset.description} in world.")
            else:
                await self._add_memory(f"Attempted to create: {description} (generation failed)", importance=0.5)
                logger.warning(f"Asset generation failed for {self.unique_id} with description: {description}")
                
        except Exception as e: # This except block corresponds to the try block above
            await self._add_memory(f"Failed to create: {description} due to error: {str(e)[:100]}", importance=0.4)
            logger.error(f"Error during _execute_create for agent {self.unique_id}: {e}", exc_info=True)
    
    async def _execute_gather(self):
        """Execute resource gathering"""
        # Simple resource gain
        resource_type = random.choice(["food", "materials"])
        amount = random.randint(1, 3)
        self.resources[resource_type] += amount
        self.energy -= 0.05
        
        await self._add_memory(f"Gathered {amount} {resource_type}", importance=0.3)
    
    async def _execute_rest(self):
        """Execute rest action"""
        self.energy = min(1.0, self.energy + 0.2)
        self.state = AgentState.IDLE
        
        await self._add_memory("Rested and recovered energy", importance=0.2)
    
    async def _execute_family_interaction_spouse(self, description: str):
        """Executes a family interaction with the agent's spouse."""
        family_member_data = self.model.family_system.family_members.get(self.unique_id)
        if not family_member_data:
            await self._add_memory(f"Tried spouse interaction: {description} (not in family)", importance=0.4)
            return

        spouse_id = None
        for related_agent_id, relationship_type in family_member_data.relationships.items():
            if relationship_type == RelationshipType.SPOUSE:
                spouse_id = related_agent_id
                break

        if not spouse_id:
            await self._add_memory(f"Tried spouse interaction: {description} (not married)", importance=0.4)
            logger.info(f"Agent {self.unique_id} attempted to interact with spouse but is not married.")
            return

        family_id_str = family_member_data.married_family_id or family_member_data.birth_family_id
        if not family_id_str:
            await self._add_memory(f"Tried spouse interaction with {spouse_id}: {description} (no family ID)", importance=0.4)
            return

        logger.info(f"Agent {self.unique_id} performing interaction with spouse {spouse_id} (Family {family_id_str}): {description}")
        # Call the new specific method in FamilySystem
        self.model.family_system.process_interaction_with_spouse(
            agent_id=self.unique_id,
            spouse_id=spouse_id, # Pass spouse_id for context if needed by the method
            family_id=family_id_str,
            description=description, # Original description from LLM
            current_step=self.model.current_step
        )
        await self._add_memory(f"Interacted with spouse {spouse_id}: {description}", importance=0.7)
        self.energy -= 0.05
        self.happiness += 0.07 # Spouse interaction might be more impactful

    async def _execute_family_interaction_child(self, child_id: str, description: str):
        """Executes a family interaction with a specific child."""
        family_member_data = self.model.family_system.family_members.get(self.unique_id)
        if not family_member_data:
            await self._add_memory(f"Tried child interaction {child_id}: {description} (not in family)", importance=0.4)
            return

        is_child = False
        if child_id in family_member_data.relationships and family_member_data.relationships[child_id] == RelationshipType.CHILD:
            is_child = True
        
        if not is_child:
            await self._add_memory(f"Tried child interaction with {child_id} (not my child): {description}", importance=0.4)
            logger.info(f"Agent {self.unique_id} attempted to interact with {child_id}, but they are not listed as a child.")
            return

        # Determine the relevant family_id for interaction with a child
        # This could be birth_family_id or married_family_id if the child is part of that household unit
        family_id_str = family_member_data.birth_family_id 
        # A more robust way to find the family_id context for the child might be needed,
        # e.g., checking if the child belongs to the agent's married family.
        # For now, using birth_family_id as a primary context.
        if family_member_data.married_family_id and child_id in self.model.family_system.families.get(family_member_data.married_family_id, Family(family_id="", family_name="", family_type=FamilyType.NUCLEAR)).members:
             family_id_str = family_member_data.married_family_id

        if not family_id_str:
            await self._add_memory(f"Tried child interaction with {child_id}: {description} (no family ID context)", importance=0.4)
            return

        logger.info(f"Agent {self.unique_id} performing interaction with child {child_id} (Family {family_id_str}): {description}")
        # Call the new specific method in FamilySystem
        self.model.family_system.process_interaction_with_child(
            agent_id=self.unique_id,
            child_id=child_id,
            family_id=family_id_str,
            description=description,
            current_step=self.model.current_step
        )
        await self._add_memory(f"Interacted with child {child_id}: {description}", importance=0.65)
        self.energy -= 0.04
        self.happiness += 0.06

    async def _execute_family_interaction_household(self, description: str):
        """Executes a general household management task for the family."""
        family_member_data = self.model.family_system.family_members.get(self.unique_id)
        if not family_member_data:
            await self._add_memory(f"Tried household task: {description} (not in family)", importance=0.4)
            return

        family_id_str = family_member_data.married_family_id or family_member_data.birth_family_id
        if not family_id_str:
            await self._add_memory(f"Tried household task: {description} (no family ID)", importance=0.4)
            return

        logger.info(f"Agent {self.unique_id} performing household management task for family {family_id_str}: {description}")
        # Call the new specific method in FamilySystem
        self.model.family_system.process_household_management_task(
            agent_id=self.unique_id,
            family_id=family_id_str,
            description=description,
            current_step=self.model.current_step
        )
        await self._add_memory(f"Managed household task for family {family_id_str}: {description}", importance=0.5)
        self.energy -= 0.06 # Household tasks can be tiring
        # Happiness change might depend on task, could be neutral or slightly positive if successful

    async def _execute_market_trade(self, order_type_str: str, resource_str: str, quantity: float, price: float):
        """Executes a market trade by submitting an order to the MarketSystem."""
        try:
            order_type = TradeOrderType(order_type_str.lower())
            resource_type = ResourceType(resource_str.lower())
        except ValueError as e:
            logger.warning(f"Agent {self.unique_id} provided invalid trade parameters: {order_type_str}, {resource_str}. Error: {e}")
            await self._add_memory(f"Failed market trade due to invalid parameters: {order_type_str} {resource_str}", importance=0.5)
            return

        logger.info(f"Agent {self.unique_id} attempting market trade: {order_type.value} {quantity} {resource_type.value} at {price}")
        
        order_id = self.model.market_system.submit_order(
            agent_id=self.unique_id,
            resource_type=resource_type,
            order_type=order_type,
            quantity=quantity,
            price_per_unit=price
        )
        
        if order_id:
            await self._add_memory(f"Submitted market order {order_id}: {order_type.value} {quantity} {resource_type.value} at {price}", importance=0.7)
            # TODO: Agent needs to check order status later and update resources upon successful trade
            # For now, we assume immediate resource change for simplicity IF it's a buy order and agent has money, or sell order and agent has resource
            # This is a major simplification and needs a proper order fulfillment and resource update mechanism.
            # if order_type == TradeOrderType.BUY and self.resources.get("currency", 0) >= quantity * price:
            #     self.resources["currency"] = self.resources.get("currency", 0) - (quantity * price)
            #     self.resources[resource_type.value] = self.resources.get(resource_type.value, 0) + quantity
            #     logger.info(f"Agent {self.unique_id} provisionally bought {quantity} {resource_type.value}")
            # elif order_type == TradeOrderType.SELL and self.resources.get(resource_type.value, 0) >= quantity:
            #     self.resources[resource_type.value] = self.resources.get(resource_type.value, 0) - quantity
            #     self.resources["currency"] = self.resources.get("currency", 0) + (quantity * price)
            #     logger.info(f"Agent {self.unique_id} provisionally sold {quantity} {resource_type.value}")
            # else:
            #     logger.warning(f"Agent {self.unique_id} order {order_id} submitted but could not be provisionally fulfilled due to resource/currency limits.")
            logger.info(f"Agent {self.unique_id} submitted order {order_id} to market.")
        else:
            await self._add_memory(f"Failed to submit market order: {order_type.value} {quantity} {resource_type.value}", importance=0.5)
            logger.warning(f"Agent {self.unique_id} failed to submit market order.")

        self.energy -= 0.05

    async def _execute_banking_action(self, sub_action_str: str, amount: Optional[float], 
                                      loan_details_str: Optional[str] = None, 
                                      loan_id_str: Optional[str] = None):
        """Executes a banking action by calling the BankingSystem."""
        agent_id = self.unique_id
        # Ensure model and necessary systems are available
        if not hasattr(self.model, 'banking_system') or not self.model.banking_system:
            logger.warning(f"BankingSystem not found on model for agent {agent_id}.")
            await self._add_memory("Tried banking action, but system is unavailable.", importance=0.3)
            return
        banking_system = self.model.banking_system
        current_sim_step = self.model.current_step # Get current simulation step

        primary_account_id = None
        if agent_id in banking_system.agent_accounts and banking_system.agent_accounts[agent_id]:
            primary_account_id = banking_system.agent_accounts[agent_id][0]
        
        # Auto-create account for deposit/withdraw if none exists and it's not an apply_loan/pay_loan action
        if sub_action_str in ["deposit", "withdraw"] and not primary_account_id:
            logger.info(f"Agent {agent_id} has no account for {sub_action_str}. Attempting to create one.")
            # Pass current_sim_step to create_account as it might log initial deposit
            new_account = await banking_system.create_account(
                agent_id=agent_id, 
                account_type=AccountType.CHECKING, 
                initial_deposit=0.0, 
                current_step=current_sim_step
            )
            if new_account:
                primary_account_id = new_account.account_id
                await self._add_memory(f"Auto-opened checking account {primary_account_id} for {sub_action_str}.", importance=0.6)
                logger.info(f"Agent {agent_id} auto-created account {primary_account_id}.")
            else:
                logger.warning(f"Failed to auto-create bank account for {agent_id}. Banking action '{sub_action_str}' may fail or be skipped.")
                await self._add_memory(f"Banking action ({sub_action_str}) failed: could not ensure bank account.", importance=0.5)
                self.energy -= 0.02; return

        try:
            if sub_action_str == "deposit":
                if amount is None or amount <= 0: logger.warning(f"Deposit for {agent_id} invalid amount: {amount}."); await self._add_memory("Tried deposit with invalid amount.",0.4); return
                if not primary_account_id: logger.warning(f"{agent_id} no account for deposit."); await self._add_memory("Tried deposit, no account.",0.4); return
                if self.resources.get("currency",0) >= amount: 
                    success = await banking_system.process_transaction(
                        account_id=primary_account_id,transaction_type=TransactionType.DEPOSIT,amount=amount,
                        description="Agent deposit", current_step=current_sim_step
                    )
                    if success: self.resources["currency"]-=amount; await self._add_memory(f"Deposited {amount} to {primary_account_id}.",0.7); logger.info(f"{agent_id} deposited {amount}.")
                    else: await self._add_memory(f"Failed deposit {amount} to {primary_account_id}.",0.5); logger.warning(f"{agent_id} failed deposit {amount}.")
                else: await self._add_memory(f"Tried deposit {amount}, insufficient cash.",0.5); logger.warning(f"{agent_id} insufficient cash for deposit.")
            
            elif sub_action_str == "withdraw":
                if amount is None or amount <= 0: logger.warning(f"Withdraw for {agent_id} invalid amount: {amount}."); await self._add_memory("Tried withdraw with invalid amount.",0.4); return
                if not primary_account_id: logger.warning(f"{agent_id} no account for withdraw."); await self._add_memory("Tried withdraw, no account.",0.4); return
                account=banking_system.accounts.get(primary_account_id)
                if account and account.balance>=amount:
                    success = await banking_system.process_transaction(
                        account_id=primary_account_id,transaction_type=TransactionType.WITHDRAWAL,amount=amount,
                        description="Agent withdrawal", current_step=current_sim_step
                    )
                    if success: self.resources["currency"]=self.resources.get("currency",0)+amount; await self._add_memory(f"Withdrew {amount} from {primary_account_id}.",0.7); logger.info(f"{agent_id} withdrew {amount}.")
                    else: await self._add_memory(f"Failed withdraw {amount} from {primary_account_id}.",0.5); logger.warning(f"{agent_id} failed withdraw {amount}.")
                else: await self._add_memory(f"Tried withdraw {amount} from {primary_account_id}, insufficient balance.",0.5); logger.warning(f"{agent_id} insufficient balance for withdraw.")
            
            elif sub_action_str == "apply_loan":
                if amount is None or amount <= 0: logger.warning(f"Apply_loan for {agent_id} invalid amount: {amount}."); await self._add_memory("Tried apply_loan with invalid amount.",0.4); return
                loan_purpose=loan_details_str or "personal expenses"; loan_type_enum=LoanType.PERSONAL
                if "business" in loan_purpose.lower(): loan_type_enum=LoanType.BUSINESS
                elif "education" in loan_purpose.lower(): loan_type_enum=LoanType.EDUCATION
                elif "house" in loan_purpose.lower(): loan_type_enum=LoanType.MORTGAGE
                # Pass current_sim_step to apply_for_loan
                loan_app=await banking_system.apply_for_loan(agent_id,loan_type_enum,amount,loan_purpose,36, current_step=current_sim_step)
                if loan_app: await self._add_memory(f"Applied for {loan_type_enum.value} loan of {amount} for '{loan_purpose}'. ID: {loan_app.loan_id}",0.8); logger.info(f"{agent_id} applied for loan {loan_app.loan_id}.")
                else: await self._add_memory(f"Failed to apply for loan of {amount} for '{loan_purpose}'.",0.6); logger.warning(f"{agent_id} failed to apply for loan.")
            
            elif sub_action_str == "pay_loan":
                if not loan_id_str or amount is None or amount <= 0: logger.warning(f"Pay_loan for {agent_id} invalid params: {loan_id_str}, {amount}."); await self._add_memory("Tried pay_loan with invalid/missing params.",0.4); return
                logger.info(f"{agent_id} attempting to pay {amount} for loan {loan_id_str}.")
                # Pass current_sim_step to process_loan_payment
                success = await banking_system.process_loan_payment(loan_id_str,amount, current_step=current_sim_step)
                if success: await self._add_memory(f"Paid {amount} for loan {loan_id_str}.",0.85); logger.info(f"{agent_id} paid {amount} for loan {loan_id_str}.")
                else: await self._add_memory(f"Failed payment of {amount} for loan {loan_id_str}. Check funds/status.",0.6); logger.warning(f"{agent_id} failed to pay {amount} for loan {loan_id_str}.")
            
            else: logger.warning(f"{agent_id} unknown banking sub-action: {sub_action_str}"); await self._add_memory(f"Unknown banking action: {sub_action_str}",0.3)
        
        except Exception as e: logger.error(f"Error banking action for {agent_id}: {sub_action_str}. Error: {e}",exc_info=True); await self._add_memory(f"Error banking action '{sub_action_str}'.",0.5)
        
        self.energy-=0.02 # General cost for banking actions

    async def _execute_market_research(self, target: str):
        """Executes market research by querying the MarketSystem."""
        logger.info(f"Agent {self.unique_id} performing market research for: {target}")
        market_info_result = "No market information obtained."
        self.last_market_research_result = None # Clear previous result first

        if not hasattr(self.model, 'market_system'):
            logger.warning(f"MarketSystem not found on model for agent {self.unique_id}. Cannot perform market research.")
            await self._add_memory(f"Attempted market research for {target}, but market system is unavailable.", importance=0.3)
            return

        try:
            if target == "all":
                if hasattr(self.model.market_system, 'get_general_market_overview'):
                    market_info_result = self.model.market_system.get_general_market_overview()
                    if not market_info_result: market_info_result = "General market overview is currently empty or unavailable."
                else:
                    market_info_result = "Function to get general market overview is not available."
            else: # Specific resource
                if hasattr(self.model.market_system, 'get_detailed_resource_market_info'):
                    try:
                        resource_enum = ResourceType(target.lower())
                        market_info_result = self.model.market_system.get_detailed_resource_market_info(resource_enum)
                        if not market_info_result: market_info_result = f"Detailed market info for {target} is currently empty or unavailable."
                    except ValueError:
                        market_info_result = f"Unknown resource '{target}' for market research."
                else:
                    market_info_result = f"Function to get detailed market info for {target} is not available."
            
            await self._add_memory(f"Market research for '{target}': {market_info_result[:200]}...", importance=0.6)
            self.last_market_research_result = market_info_result # Store for next prompt
            logger.info(f"Agent {self.unique_id} market research result for '{target}': {market_info_result}")

        except Exception as e:
            logger.error(f"Error during market research for agent {self.unique_id} (target: {target}): {e}", exc_info=True)
            await self._add_memory(f"Error performing market research for '{target}'.", importance=0.4)
            self.last_market_research_result = f"Error during market research for {target}."
        
        self.energy -= 0.01 # Small energy cost for research

    async def _execute_get_banking_statement(self):
        """Retrieves and records the agent's detailed banking statement."""
        logger.info(f"Agent {self.unique_id} requesting banking statement.")
        self.last_banking_statement = None # Clear previous statement
        statement_str = "Banking statement unavailable or no accounts found."

        if not hasattr(self.model, 'banking_system') or not hasattr(self.model.banking_system, 'get_agent_financial_summary'):
            logger.warning(f"BankingSystem or get_agent_financial_summary method not found for agent {self.unique_id}.")
            await self._add_memory("Tried to get banking statement, but system/method is unavailable.", importance=0.3)
            self.last_banking_statement = "System error: Banking statement function unavailable."
            self.energy -= 0.01 # Minimal cost for failed attempt
            return

        try:
            summary_dict = self.model.banking_system.get_agent_financial_summary(self.unique_id)
            if summary_dict and "error" not in summary_dict:
                parts = [f"Financial Summary for {self.unique_id}:"]
                parts.append(f"  Total Balance: {summary_dict.get('total_balance', 0.0):.2f}")
                parts.append(f"  Total Debt: {summary_dict.get('total_debt', 0.0):.2f}")
                parts.append(f"  Net Worth: {summary_dict.get('net_worth', 0.0):.2f}")
                parts.append(f"  Credit Score: {summary_dict.get('credit_score', 0.0):.0f}")
                
                account_details = summary_dict.get("account_details", [])
                if account_details:
                    parts.append(f"  Accounts ({len(account_details)}):")
                    for acc in account_details[:2]: # Show details for up to 2 accounts for brevity in prompt
                        parts.append(f"    - ID: {acc.get('account_id')}, Type: {acc.get('type')}, Balance: {acc.get('balance',0):.2f}")
                else:
                    parts.append("  No bank accounts found.")
                
                loan_details = summary_dict.get("loan_details", [])
                active_loans = [loan for loan in loan_details if loan.get("status") == LoanStatus.ACTIVE.value]
                if active_loans:
                    parts.append(f"  Active Loans ({len(active_loans)}):")
                    for loan in active_loans[:2]: # Show details for up to 2 loans
                        parts.append(f"    - ID: {loan.get('loan_id')}, Type: {loan.get('type')}, Remaining: {loan.get('remaining_balance',0):.2f}, Payment: {loan.get('monthly_payment',0):.2f}")
                else:
                    parts.append("  No active loans found.")
                statement_str = "\n".join(parts)
            elif summary_dict and "error" in summary_dict:
                statement_str = f"Banking statement error: {summary_dict['error']}"
            
            await self._add_memory(f"Retrieved banking statement: {statement_str[:250]}...", importance=0.7)
            self.last_banking_statement = statement_str
            logger.info(f"Agent {self.unique_id} banking statement: {statement_str}")

        except Exception as e:
            logger.error(f"Error during get_banking_statement for agent {self.unique_id}: {e}", exc_info=True)
            await self._add_memory("Error retrieving banking statement.", importance=0.4)
            self.last_banking_statement = "Error retrieving banking statement."
        
        self.energy -= 0.01 # Small energy cost for fetching statement
    
    async def _generate_conversation(self, other_agent: "LLMAgent") -> str:
        """Generate conversation with another agent"""
        try:
            prompt = f"""
You ({self.persona}) are talking with {other_agent.unique_id} ({other_agent.persona}).

Your recent memories: {await self._format_recent_memories()}
Their recent activities: {await other_agent._format_recent_memories()}

Generate a brief conversation (1-2 lines) between you two.
"""
            
            response = await self.llm_coordinator.get_response(
                agent_id=self.unique_id,
                prompt=prompt,
                max_tokens=100
            )
            
            return response.strip()
            
        except Exception as e:
            logger.warning(f"Conversation generation failed: {e}")
            return f"Had a pleasant chat about daily activities."
    
    async def _add_memory(self, content: str, importance: float = 0.5, tags: List[str] = None):
        """Add a memory to the agent's memory system and save to DB."""
        memory = Memory(
            agent_id=self.unique_id, # Store agent_id within memory for easier DB association
            content=content,
            timestamp=self.model.current_step, # Use simulation step as primary timestamp
            importance=importance,
            tags=tags or []
        )
        self.memories.append(memory)
        
        # Save to database
        if hasattr(self.model, 'database_handler') and self.model.database_handler:
            try: await self.model.database_handler.save_agent_memory(
                agent_id=self.unique_id, 
                memory_data=memory.to_dict(), 
                step=self.model.current_step
            )
            except Exception as e: logger.error(f"Agent {self.unique_id}: Failed to save memory to DB: {e}", exc_info=True)
        
        # Keep memory size manageable (in-memory list)
        if len(self.memories) > self.memory_size:
            self.memories.sort(key=lambda m: m.importance, reverse=True)
            self.memories = self.memories[:self.memory_size]
    
    async def _format_recent_memories(self, count: int = 3) -> str:
        """Format recent memories for LLM context"""
        recent = sorted(self.memories, key=lambda m: m.timestamp, reverse=True)[:count]
        if not recent:
            return "No recent memories."
        
        return "; ".join([m.content for m in recent])
    
    def _update_resources(self):
        """Update energy and resource decay"""
        # Gradual energy decay
        self.energy = max(0.0, self.energy - 0.005)
        
        # Resource consumption
        if self.step_count % 20 == 0:  # Every 20 steps
            self.resources["food"] = max(0, self.resources["food"] - 1)
            if self.resources["food"] == 0:
                self.energy = max(0.0, self.energy - 0.1)  # Hunger effect
    
    async def _manage_memory(self):
        """Clean up old or unimportant memories"""
        if len(self.memories) > self.memory_size * 1.2:
            # Aggressive cleanup
            self.memories = sorted(self.memories, key=lambda m: m.importance, reverse=True)
            self.memories = self.memories[:self.memory_size]
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status for monitoring"""
        return {
            "id": self.unique_id,
            "position": (self.position.x, self.position.y, self.position.z),
            "state": self.state.value,
            "energy": self.energy,
            "happiness": self.happiness,
            "resources": self.resources,
            "social_connections": len(self.social_connections),
            "memories": len(self.memories),
            "step_count": self.step_count,
            "social_interactions": self.social_interactions,
            "objects_created": self.objects_created,
        } 

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the agent's state to a dictionary."""
        # Convert Position object to dict
        position_dict = {"x": self.position.x, "y": self.position.y, "z": self.position.z}
        
        # Convert Memory objects to dicts
        memories_list_of_dicts = []
        for mem in self.memories:
            memories_list_of_dicts.append(mem.to_dict())

        return {
            "unique_id": self.unique_id,
            "persona": self.persona,
            "agent_type": self.agent_type.value, # Store enum value
            "position": position_dict,
            "target_position": {"x": self.target_position.x, "y": self.target_position.y, "z": self.target_position.z} if self.target_position else None,
            "movement_speed": self.movement_speed,
            "state": self.state.value, # Store enum value
            "energy": self.energy,
            "happiness": self.happiness,
            "age": self.age,
            "health": self.health,
            "employed": self.employed,
            "social_connections": dict(self.social_connections), # Ensure it's a plain dict
            "social_reputation": self.social_reputation,
            "family_id": self.family_id,
            "cultural_group_id": self.cultural_group_id,
            "cultural_affinities": dict(self.cultural_affinities),
            "credit_score": self.credit_score,
            "total_debt": self.total_debt,
            "monthly_income": self.monthly_income,
            "memories": memories_list_of_dicts,
            "memory_size": self.memory_size,
            "resources": dict(self.resources),
            "inventory": list(self.inventory),
            "last_llm_call": self.last_llm_call,
            # self.llm_cache is not serialized as it can be rebuilt or might be too large/transient
            "conversation_history": list(self.conversation_history),
            "last_market_research_result": self.last_market_research_result, # This is transient, but might be useful if saving mid-decision
            "last_banking_statement": self.last_banking_statement, # Also transient
            "step_count": self.step_count,
            "social_interactions": self.social_interactions,
            "objects_created": self.objects_created
            # Excluded: self.model, self.config, self.llm_coordinator (passed during reconstruction)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], model: "mesa.Model", llm_coordinator: "LLMCoordinator", config: "Config") -> "LLMAgent":
        """Deserialize an agent's state from a dictionary."""
        
        # Reconstruct Position objects
        pos_data = data.get("position", {})
        position = Position(x=pos_data.get("x", 0.0), y=pos_data.get("y", 0.0), z=pos_data.get("z", 0.0))
        
        target_pos_data = data.get("target_position")
        target_position = None
        if target_pos_data:
            target_position = Position(x=target_pos_data.get("x", 0.0), y=target_pos_data.get("y", 0.0), z=target_pos_data.get("z", 0.0))

        # Get unique_id, persona, and agent_type for __init__
        unique_id = data["unique_id"]
        persona = data.get("persona") # Persona might be generated if not in data, but it should be.
        
        # Get initial values for new attributes for the constructor call, if they exist in data
        initial_age = data.get("age", 25.0)
        initial_health = data.get("health", 1.0)
        initial_employed_status = data.get("employed", 0)

        # Create agent instance. __init__ will handle agent_type based on persona if not passed directly,
        # but since we serialized it, we can pass it. If agent_type is not in data, init will derive it.
        # For deserialization, we prefer to use the stored agent_type.
        agent = cls(
            model=model,
            unique_id=unique_id,
            llm_coordinator=llm_coordinator,
            config=config,
            position=position,
            persona=persona, # Pass serialized persona
            initial_age=initial_age,
            initial_health=initial_health,
            initial_employed_status=initial_employed_status
        )

        # Override agent_type if it was in the serialized data (it should be)
        if "agent_type" in data:
            try: agent.agent_type = AgentType(data["agent_type"])
            except ValueError: logger.warning(f"Invalid agent_type '{data['agent_type']}' in serialized data for {unique_id}. Using default from persona.")

        # Set other attributes
        agent.target_position = target_position
        agent.movement_speed = data.get("movement_speed", config.agents.movement_speed) # Fallback to config default
        try: agent.state = AgentState(data.get("state", AgentState.IDLE.value)) # Fallback to IDLE
        except ValueError: logger.warning(f"Invalid state '{data.get('state')}' for {unique_id}. Defaulting to IDLE."); agent.state = AgentState.IDLE
        
        agent.energy = data.get("energy", 1.0)
        agent.happiness = data.get("happiness", 0.5)
        agent.social_connections = dict(data.get("social_connections", {}))
        agent.social_reputation = data.get("social_reputation", 0.5)
        agent.family_id = data.get("family_id")
        agent.cultural_group_id = data.get("cultural_group_id")
        agent.cultural_affinities = dict(data.get("cultural_affinities", {}))
        agent.credit_score = data.get("credit_score", 700.0)
        agent.total_debt = data.get("total_debt", 0.0)
        agent.monthly_income = data.get("monthly_income", 0.0)
        
        # Reconstruct Memories
        agent.memories = []
        memories_data = data.get("memories", [])
        for mem_data in memories_data:
            agent.memories.append(Memory.from_dict(mem_data))
        agent.memory_size = data.get("memory_size", config.agents.memory_size)
        
        agent.resources = dict(data.get("resources", {"food": 10, "materials": 5, "tools": 1, "currency": 500, "energy_item": 5}))
        agent.inventory = list(data.get("inventory", []))
        agent.last_llm_call = data.get("last_llm_call", 0.0)
        agent.conversation_history = list(data.get("conversation_history", []))
        agent.last_market_research_result = data.get("last_market_research_result")
        agent.last_banking_statement = data.get("last_banking_statement")
        agent.step_count = data.get("step_count", 0)
        agent.social_interactions = data.get("social_interactions", 0)
        agent.objects_created = data.get("objects_created", 0)

        return agent

    # Need to forward declare types for type hinting if they are used in method signatures
    # For example, if LLMCoordinator and Config are classes defined elsewhere:
    # class LLMCoordinator: pass 
    # class Config: pass
    # class mesa: class Model: pass # Mocking Mesa for type hint only 