"""
Intelligent Agent with LLM-driven decision making
Enhanced version of SocietyAgent with real AI behavior
"""

import asyncio
import random
import time
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from collections import deque

from llm_integration import LLMManager, LLMRequest, create_agent_prompt
from society_demo import AgentType, CulturalGroup, AgentState, Position, Memory

@dataclass
class Goal:
    """Agent goal with priority and deadline"""
    goal_id: str
    description: str
    priority: float  # 0.0 to 1.0
    target_value: Optional[float] = None
    deadline_steps: Optional[int] = None
    created_step: int = 0
    completed: bool = False

@dataclass
class Relationship:
    """Relationship between agents"""
    agent_id: str
    relationship_type: str  # friend, family, rival, neutral, romantic
    strength: float  # -1.0 to 1.0 (negative = dislike, positive = like)
    trust: float    # 0.0 to 1.0
    history: List[str] = None
    
    def __post_init__(self):
        if self.history is None:
            self.history = []

class MemoryType(Enum):
    INTERACTION = "interaction"
    ACHIEVEMENT = "achievement"
    EVENT = "event"
    OBSERVATION = "observation"
    DECISION = "decision"

@dataclass
class EnhancedMemory:
    """Enhanced memory with categorization and emotional impact"""
    content: str
    memory_type: MemoryType
    timestamp: float
    importance: float = 0.5
    emotional_impact: float = 0.0  # -1.0 to 1.0
    associated_agents: List[str] = None
    location: Optional[Position] = None
    
    def __post_init__(self):
        if self.associated_agents is None:
            self.associated_agents = []

class IntelligentAgent:
    """LLM-driven intelligent agent with memory, goals, and relationships"""
    
    def __init__(self, agent_id: str, world_size: Tuple[float, float] = (100, 100), 
                 llm_manager: Optional[LLMManager] = None):
        # Basic attributes from original agent
        self.agent_id = agent_id
        self.position = Position(
            random.uniform(0, world_size[0]),
            random.uniform(0, world_size[1]),
            0.0
        )
        
        # Core attributes
        self.agent_type = random.choice(list(AgentType))
        self.cultural_group = random.choice(list(CulturalGroup))
        self.state = AgentState.IDLE
        
        # Status
        self.energy = random.uniform(0.7, 1.0)
        self.happiness = random.uniform(0.4, 0.8)
        self.health = random.uniform(0.8, 1.0)
        self.age = random.uniform(18, 65)
        
        # Enhanced personality with more depth
        self.personality_traits = {
            "extroversion": random.random(),
            "conscientiousness": random.random(),
            "openness": random.random(),
            "agreeableness": random.random(),
            "neuroticism": random.random(),
            "ambition": random.random(),
            "curiosity": random.random(),
            "empathy": random.random()
        }
        
        # Social attributes
        self.social_reputation = 0.5
        self.family_id = None
        self.relationships: Dict[str, Relationship] = {}
        
        # Economic
        self.resources = {
            "food": random.randint(10, 50),
            "currency": random.randint(100, 1000),
            "materials": random.randint(5, 25),
            "tools": random.randint(1, 5)
        }
        self.employed = random.choice([True, False])
        
        # Enhanced memory system
        self.short_term_memory = deque(maxlen=10)  # Recent events
        self.long_term_memory: List[EnhancedMemory] = []  # Important memories
        self.working_memory = deque(maxlen=5)  # Current context
        
        # Goals and planning
        self.goals: List[Goal] = []
        self.current_plan: List[str] = []
        self.last_decision_reason = ""
        
        # LLM integration
        self.llm_manager = llm_manager
        self.last_llm_call = 0
        self.llm_cooldown = 3  # Seconds between LLM calls
        
        # Movement
        self.target_position = None
        self.movement_speed = 2.0
        
        # Generate initial personality and goals
        self._initialize_personality()
        self._create_initial_goals()
        
    def _initialize_personality(self):
        """Create personality-based traits and preferences"""
        # Create personality description for LLM context
        traits = []
        if self.personality_traits["extroversion"] > 0.7:
            traits.append("outgoing and social")
        elif self.personality_traits["extroversion"] < 0.3:
            traits.append("introverted and thoughtful")
            
        if self.personality_traits["conscientiousness"] > 0.7:
            traits.append("organized and responsible")
        elif self.personality_traits["conscientiousness"] < 0.3:
            traits.append("spontaneous and flexible")
            
        if self.personality_traits["openness"] > 0.7:
            traits.append("creative and curious")
            
        if self.personality_traits["agreeableness"] > 0.7:
            traits.append("cooperative and trusting")
        elif self.personality_traits["agreeableness"] < 0.3:
            traits.append("competitive and skeptical")
            
        self.personality_description = f"I am {', '.join(traits) if traits else 'balanced in my approach to life'}."
        
    def _create_initial_goals(self):
        """Create personality-based initial goals"""
        goal_templates = {
            "economic": ["Accumulate 1000 currency", "Acquire valuable tools", "Establish trade relationships"],
            "social": ["Make 5 close friends", "Gain social recognition", "Help others in the community"],
            "personal": ["Improve my skills", "Maintain good health", "Find happiness and fulfillment"],
            "cultural": ["Spread my cultural values", "Learn from other cultures", "Preserve traditions"]
        }
        
        # Choose goals based on personality
        num_goals = random.randint(2, 4)
        selected_goals = []
        
        for i in range(num_goals):
            # Weight goal types by personality
            if self.personality_traits["ambition"] > 0.6 and random.random() < 0.4:
                category = "economic"
            elif self.personality_traits["extroversion"] > 0.6 and random.random() < 0.4:
                category = "social"
            elif self.personality_traits["openness"] > 0.6 and random.random() < 0.3:
                category = "cultural"
            else:
                category = "personal"
                
            template = random.choice(goal_templates[category])
            goal = Goal(
                goal_id=f"{self.agent_id}_goal_{i}",
                description=template,
                priority=random.uniform(0.3, 0.9),
                deadline_steps=random.randint(100, 500) if random.random() < 0.7 else None,
                created_step=0
            )
            selected_goals.append(goal)
            
        self.goals = selected_goals
    
    async def step(self, world):
        """Enhanced agent step with LLM decision making"""
        current_time = time.time()
        
        # Update working memory with current context
        self._update_working_memory(world)
        
        # Make decision using LLM or fallback
        if (self.llm_manager and 
            current_time - self.last_llm_call > self.llm_cooldown and 
            random.random() < 0.3):  # Don't call LLM every step
            decision = await self._make_llm_decision(world)
            self.last_llm_call = current_time
        else:
            decision = self._make_rule_based_decision(world)
        
        # Execute decision
        await self._execute_action(decision, world)
        
        # Update goals and memories
        self._update_goals()
        self._update_state()
        
    async def _make_llm_decision(self, world) -> str:
        """Use LLM to make intelligent decision"""
        context = self._gather_enhanced_context(world)
        prompt = self._create_decision_prompt(context)
        
        request = LLMRequest(
            agent_id=self.agent_id,
            prompt=prompt,
            context=context,
            max_tokens=200,
            temperature=0.7 + (self.personality_traits["openness"] - 0.5) * 0.4  # Personality affects creativity
        )
        
        try:
            response = await self.llm_manager.get_response(request)
            
            if response.success:
                # Parse LLM response for action and reasoning
                action, reasoning = self._parse_llm_response(response.response)
                self.last_decision_reason = reasoning
                
                # Store decision in memory
                self._add_memory(
                    f"Decided to {action}. Reasoning: {reasoning}",
                    MemoryType.DECISION,
                    importance=0.6,
                    emotional_impact=0.1
                )
                
                return action
            else:
                return self._make_rule_based_decision(world)
                
        except Exception as e:
            print(f"LLM decision failed for {self.agent_id}: {e}")
            return self._make_rule_based_decision(world)
    
    def _create_decision_prompt(self, context: Dict[str, Any]) -> str:
        """Create detailed prompt for LLM decision making"""
        # Recent memories for context
        recent_memories = list(self.short_term_memory)[-3:] if self.short_term_memory else []
        memory_text = "\n".join([f"- {mem.content}" for mem in recent_memories]) if recent_memories else "- No recent events"
        
        # Active goals
        active_goals = [g for g in self.goals if not g.completed][:3]
        goals_text = "\n".join([f"- {g.description} (priority: {g.priority:.1f})" for g in active_goals]) if active_goals else "- No active goals"
        
        # Nearby agents with relationships
        nearby_text = ""
        for agent_info in context.get("nearby_agents", []):
            agent = agent_info["agent"]
            agent_id = agent.agent_id
            relationship = self.relationships.get(agent_id)
            rel_desc = f" (relationship: {relationship.relationship_type}, trust: {relationship.trust:.1f})" if relationship else " (stranger)"
            nearby_text += f"- {agent.agent_type.value}{rel_desc} at distance {agent_info['distance']:.1f}\n"
        
        prompt = f"""You are {self.agent_id}, a {self.agent_type.value} in a virtual society.

PERSONALITY: {self.personality_description}

CURRENT STATUS:
- Energy: {self.energy:.2f}/1.0 ({'tired' if self.energy < 0.4 else 'energetic'})
- Happiness: {self.happiness:.2f}/1.0 ({'unhappy' if self.happiness < 0.4 else 'content'})
- Health: {self.health:.2f}/1.0
- Age: {self.age:.1f} years
- Position: ({self.position.x:.1f}, {self.position.y:.1f})

RESOURCES:
- Food: {self.resources['food']}
- Currency: {self.resources['currency']}
- Materials: {self.resources['materials']}
- Tools: {self.resources['tools']}

RECENT MEMORIES:
{memory_text}

CURRENT GOALS:
{goals_text}

NEARBY AGENTS:
{nearby_text if nearby_text else "- No one nearby"}

RELATIONSHIPS: {len(self.relationships)} established relationships

Based on your personality, current situation, goals, and relationships, what would you like to do next?

Choose ONE action and explain your reasoning in 1-2 sentences:

ACTIONS:
- work: Focus on your profession to earn resources and feel productive
- socialize: Interact with nearby agents to build relationships
- trade: Exchange resources with others for mutual benefit
- rest: Recover energy and take care of your health
- move: Explore the world or move toward a specific goal
- help: Assist another agent with their needs
- learn: Observe and gain knowledge about your environment

Format your response as:
ACTION: [chosen action]
REASONING: [1-2 sentences explaining why based on your personality and situation]"""

        return prompt
    
    def _parse_llm_response(self, response: str) -> Tuple[str, str]:
        """Parse LLM response to extract action and reasoning"""
        lines = response.strip().split('\n')
        action = "work"  # Default
        reasoning = "I need to stay productive."  # Default
        
        for line in lines:
            line = line.strip()
            if line.startswith("ACTION:"):
                action_text = line.replace("ACTION:", "").strip().lower()
                # Extract first word as action
                action = action_text.split()[0] if action_text else "work"
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()
        
        # Validate action
        valid_actions = ["work", "socialize", "trade", "rest", "move", "help", "learn"]
        if action not in valid_actions:
            action = "work"
            
        return action, reasoning
    
    def _make_rule_based_decision(self, world) -> str:
        """Fallback rule-based decision making"""
        context = self._gather_enhanced_context(world)
        
        # Priority-based decision making
        if self.energy < 0.3:
            return "rest"
        elif self.resources["food"] < 5:
            return "work"  # Focus on basic needs
        elif len(context["nearby_agents"]) > 0 and self.personality_traits["extroversion"] > 0.6:
            return "socialize"
        elif self.resources["currency"] > 300 and random.random() < 0.3:
            return "trade"
        elif random.random() < 0.2:
            return "move"
        else:
            return "work"
    
    def _gather_enhanced_context(self, world) -> Dict[str, Any]:
        """Gather comprehensive context for decision making"""
        nearby_agents = []
        for other in world.agents:
            if other.agent_id != self.agent_id:
                distance = self.position.distance_to(other.position)
                if distance < 20.0:  # Expanded awareness radius
                    nearby_agents.append({
                        "agent": other,
                        "distance": distance,
                        "type": other.agent_type.value,
                        "cultural_group": other.cultural_group.value,
                        "relationship": self.relationships.get(other.agent_id)
                    })
        
        return {
            "agent_type": self.agent_type.value,
            "energy": self.energy,
            "happiness": self.happiness,
            "resources": self.resources.copy(),
            "nearby_agents": nearby_agents,
            "personality": self.personality_traits.copy(),
            "goals": [g for g in self.goals if not g.completed],
            "relationships": len(self.relationships),
            "recent_memories": list(self.short_term_memory)[-3:] if self.short_term_memory else []
        }
    
    async def _execute_action(self, action: str, world):
        """Execute the chosen action with enhanced behaviors"""
        if action == "socialize":
            await self._enhanced_socialize(world)
        elif action == "trade":
            await self._enhanced_trade(world)
        elif action == "work":
            await self._enhanced_work()
        elif action == "rest":
            await self._enhanced_rest()
        elif action == "move":
            await self._enhanced_move(world)
        elif action == "help":
            await self._help_others(world)
        elif action == "learn":
            await self._learn_and_observe(world)
        else:
            await self._enhanced_work()  # Default
        
        self.state = AgentState(action) if action in [s.value for s in AgentState] else AgentState.WORKING
    
    async def _enhanced_socialize(self, world):
        """Enhanced socialization with relationship building"""
        nearby = [agent for agent in world.agents 
                 if agent.agent_id != self.agent_id and 
                 self.position.distance_to(agent.position) < 15.0]
        
        if nearby:
            # Choose interaction target based on existing relationships and personality
            target = self._choose_interaction_target(nearby)
            
            if target:
                await self._interact_with_agent(target)
                
        self.energy -= 0.01
        self.happiness += 0.05 * self.personality_traits["extroversion"]
    
    def _choose_interaction_target(self, nearby_agents) -> Optional[Any]:
        """Choose who to interact with based on relationships and personality"""
        if not nearby_agents:
            return None
            
        # Prefer agents we have positive relationships with
        friends = [agent for agent in nearby_agents 
                  if agent.agent_id in self.relationships and 
                  self.relationships[agent.agent_id].strength > 0.3]
        
        if friends and random.random() < 0.7:
            return random.choice(friends)
        
        # Otherwise, choose based on cultural compatibility
        same_culture = [agent for agent in nearby_agents 
                       if agent.cultural_group == self.cultural_group]
        
        if same_culture and random.random() < 0.5:
            return random.choice(same_culture)
        
        # Random interaction
        return random.choice(nearby_agents)
    
    async def _interact_with_agent(self, other):
        """Deep interaction with relationship dynamics"""
        # Create or update relationship
        if other.agent_id not in self.relationships:
            self._create_new_relationship(other)
        
        relationship = self.relationships[other.agent_id]
        
        # Interaction outcome based on personality compatibility
        compatibility = self._calculate_compatibility(other)
        interaction_success = random.random() < (0.5 + compatibility * 0.3)
        
        if interaction_success:
            # Positive interaction
            relationship.strength = min(1.0, relationship.strength + 0.1)
            relationship.trust = min(1.0, relationship.trust + 0.05)
            
            self._add_memory(
                f"Had a great conversation with {other.agent_id}",
                MemoryType.INTERACTION,
                importance=0.6,
                emotional_impact=0.3,
                associated_agents=[other.agent_id]
            )
            
            self.happiness += 0.1
            
        else:
            # Negative interaction
            relationship.strength = max(-1.0, relationship.strength - 0.05)
            relationship.trust = max(0.0, relationship.trust - 0.02)
            
            self._add_memory(
                f"Had an awkward interaction with {other.agent_id}",
                MemoryType.INTERACTION,
                importance=0.4,
                emotional_impact=-0.2,
                associated_agents=[other.agent_id]
            )
            
        # Update other agent's relationship too (if they're also intelligent)
        if hasattr(other, 'relationships'):
            if self.agent_id not in other.relationships:
                other._create_new_relationship(self)
            other.relationships[self.agent_id].strength += (0.1 if interaction_success else -0.05)
            other.relationships[self.agent_id].trust += (0.05 if interaction_success else -0.02)
    
    def _create_new_relationship(self, other):
        """Create new relationship based on first impression"""
        compatibility = self._calculate_compatibility(other)
        
        # Initial relationship type
        if compatibility > 0.7:
            rel_type = "friend"
            initial_strength = 0.4
        elif compatibility < 0.3:
            rel_type = "acquaintance"
            initial_strength = 0.1
        else:
            rel_type = "neutral"
            initial_strength = 0.2
            
        self.relationships[other.agent_id] = Relationship(
            agent_id=other.agent_id,
            relationship_type=rel_type,
            strength=initial_strength,
            trust=0.3,
            history=[f"First met {other.agent_id}"]
        )
    
    def _calculate_compatibility(self, other) -> float:
        """Calculate personality compatibility with another agent"""
        if not hasattr(other, 'personality_traits'):
            return 0.5  # Neutral for non-intelligent agents
            
        # Compare personality traits
        compatibility = 0.0
        
        # Extroversion compatibility
        ext_diff = abs(self.personality_traits["extroversion"] - other.personality_traits["extroversion"])
        compatibility += (1.0 - ext_diff) * 0.3
        
        # Agreeableness compatibility
        agree_avg = (self.personality_traits["agreeableness"] + other.personality_traits["agreeableness"]) / 2
        compatibility += agree_avg * 0.4
        
        # Cultural similarity
        if self.cultural_group == other.cultural_group:
            compatibility += 0.3
        
        return max(0.0, min(1.0, compatibility))
    
    async def _enhanced_trade(self, world):
        """Enhanced trading with relationship consideration"""
        nearby_traders = [agent for agent in world.agents 
                         if agent.agent_id != self.agent_id and 
                         self.position.distance_to(agent.position) < 20.0 and
                         agent.resources["currency"] > 50]
        
        if nearby_traders and self.resources["materials"] > 5:
            # Prefer trading with friends
            friends = [agent for agent in nearby_traders 
                      if agent.agent_id in self.relationships and 
                      self.relationships[agent.agent_id].strength > 0.0]
            
            target = random.choice(friends) if friends else random.choice(nearby_traders)
            
            # Trade with relationship bonus
            relationship = self.relationships.get(target.agent_id)
            trust_bonus = relationship.trust if relationship else 0.3
            
            trade_amount = min(5, self.resources["materials"])
            base_price = trade_amount * 10
            final_price = int(base_price * (0.8 + trust_bonus * 0.4))  # Friends get better prices
            
            if target.resources["currency"] >= final_price:
                # Execute trade
                self.resources["materials"] -= trade_amount
                self.resources["currency"] += final_price
                target.resources["materials"] += trade_amount
                target.resources["currency"] -= final_price
                
                # Update relationship
                if relationship:
                    relationship.trust = min(1.0, relationship.trust + 0.1)
                
                self._add_memory(
                    f"Successfully traded {trade_amount} materials for {final_price} currency with {target.agent_id}",
                    MemoryType.INTERACTION,
                    importance=0.5,
                    emotional_impact=0.2,
                    associated_agents=[target.agent_id]
                )
        
        self.energy -= 0.02
    
    async def _enhanced_work(self):
        """Enhanced work with goal consideration"""
        # Work output modified by personality and goals
        productivity = self.personality_traits["conscientiousness"] * self.energy
        
        if self.agent_type == AgentType.FARMER:
            food_produced = int(random.randint(3, 8) * productivity)
            self.resources["food"] += food_produced
            
        elif self.agent_type == AgentType.CRAFTSMAN:
            if self.resources["materials"] > 2:
                self.resources["materials"] -= 2
                tools_made = int(1 * productivity)
                currency_earned = int(15 * productivity)
                self.resources["tools"] += tools_made
                self.resources["currency"] += currency_earned
                
        elif self.agent_type == AgentType.TRADER:
            currency_earned = int(random.randint(5, 20) * productivity)
            self.resources["currency"] += currency_earned
            
        else:
            currency_earned = int(random.randint(3, 10) * productivity)
            self.resources["currency"] += currency_earned
        
        self.energy -= 0.05
        self.happiness += 0.02 * self.personality_traits["conscientiousness"]
        
        # Work contributes to goals
        self._progress_goals("work")
    
    async def _enhanced_rest(self):
        """Enhanced rest with personality effects"""
        energy_recovery = 0.15 * (1.0 + self.personality_traits["conscientiousness"] * 0.2)
        self.energy = min(1.0, self.energy + energy_recovery)
        self.resources["food"] -= 1
        
        # Introverts enjoy rest more
        happiness_bonus = 0.05 * (1.0 - self.personality_traits["extroversion"])
        self.happiness += happiness_bonus
    
    async def _enhanced_move(self, world):
        """Enhanced movement with goal-directed behavior"""
        # Sometimes move toward goals or interesting agents
        if self.goals and random.random() < 0.3:
            # Move toward a goal-related area (simplified)
            self.target_position = Position(
                random.uniform(0, world.world_size[0]),
                random.uniform(0, world.world_size[1]),
                0.0
            )
        elif self.target_position is None or self.position.distance_to(self.target_position) < 2.0:
            self.target_position = Position(
                random.uniform(0, world.world_size[0]),
                random.uniform(0, world.world_size[1]),
                0.0
            )
        
        self.position = self.position.move_towards(self.target_position, self.movement_speed)
        self.energy -= 0.02
    
    async def _help_others(self, world):
        """Help nearby agents based on empathy"""
        if self.personality_traits["empathy"] < 0.3:
            return await self._enhanced_work()  # Not helpful, work instead
            
        nearby_agents = [agent for agent in world.agents 
                        if agent.agent_id != self.agent_id and 
                        self.position.distance_to(agent.position) < 15.0 and
                        hasattr(agent, 'energy') and agent.energy < 0.4]
        
        if nearby_agents and self.resources["food"] > 10:
            target = random.choice(nearby_agents)
            
            # Give food to help
            help_amount = min(5, self.resources["food"] - 5)
            self.resources["food"] -= help_amount
            target.resources["food"] += help_amount
            
            # Build relationship
            if target.agent_id not in self.relationships:
                self._create_new_relationship(target)
            
            self.relationships[target.agent_id].strength += 0.2
            self.relationships[target.agent_id].trust += 0.1
            
            self._add_memory(
                f"Helped {target.agent_id} by giving them {help_amount} food",
                MemoryType.INTERACTION,
                importance=0.7,
                emotional_impact=0.4,
                associated_agents=[target.agent_id]
            )
            
            self.happiness += 0.1 * self.personality_traits["empathy"]
        
        self.energy -= 0.01
    
    async def _learn_and_observe(self, world):
        """Learn from environment and other agents"""
        # Observe successful agents
        successful_agents = [agent for agent in world.agents 
                           if agent.agent_id != self.agent_id and 
                           hasattr(agent, 'resources') and
                           agent.resources.get("currency", 0) > self.resources["currency"] * 1.5]
        
        if successful_agents:
            target = random.choice(successful_agents)
            observation = f"Observed {target.agent_id} ({target.agent_type.value}) who seems very successful"
            
            self._add_memory(
                observation,
                MemoryType.OBSERVATION,
                importance=0.4,
                emotional_impact=0.1
            )
            
            # Learn from their behavior (simplified)
            if target.agent_type == AgentType.TRADER and random.random() < 0.1:
                self.resources["currency"] += 5  # Small learning bonus
        
        self.energy -= 0.01
        self.happiness += 0.02 * self.personality_traits["openness"]
    
    def _update_working_memory(self, world):
        """Update working memory with current important context"""
        # Add current situation to working memory
        situation = f"Step {world.step_count}: Energy {self.energy:.2f}, at ({self.position.x:.1f}, {self.position.y:.1f})"
        
        if len(self.working_memory) == 0 or self.working_memory[-1].content != situation:
            self.working_memory.append(EnhancedMemory(
                content=situation,
                memory_type=MemoryType.OBSERVATION,
                timestamp=time.time(),
                importance=0.2
            ))
    
    def _add_memory(self, content: str, memory_type: MemoryType, 
                   importance: float = 0.5, emotional_impact: float = 0.0,
                   associated_agents: List[str] = None):
        """Add enhanced memory with categorization"""
        memory = EnhancedMemory(
            content=content,
            memory_type=memory_type,
            timestamp=time.time(),
            importance=importance,
            emotional_impact=emotional_impact,
            associated_agents=associated_agents or [],
            location=Position(self.position.x, self.position.y, self.position.z)
        )
        
        # Add to short-term memory
        self.short_term_memory.append(memory)
        
        # Important memories go to long-term storage
        if importance > 0.6:
            self.long_term_memory.append(memory)
            
            # Keep long-term memory manageable
            if len(self.long_term_memory) > 50:
                # Remove least important memories
                self.long_term_memory.sort(key=lambda m: m.importance, reverse=True)
                self.long_term_memory = self.long_term_memory[:40]
    
    def _update_goals(self):
        """Update and check goal progress"""
        for goal in self.goals:
            if goal.completed:
                continue
                
            # Simple goal completion logic
            if "currency" in goal.description.lower():
                target_amount = 1000  # Extract from description in real implementation
                if self.resources["currency"] >= target_amount:
                    goal.completed = True
                    self._add_memory(
                        f"Achieved goal: {goal.description}",
                        MemoryType.ACHIEVEMENT,
                        importance=0.8,
                        emotional_impact=0.5
                    )
                    self.happiness += 0.2
            
            elif "friends" in goal.description.lower():
                friend_count = len([r for r in self.relationships.values() if r.strength > 0.5])
                if friend_count >= 5:  # Extract from description in real implementation
                    goal.completed = True
                    self._add_memory(
                        f"Achieved goal: {goal.description}",
                        MemoryType.ACHIEVEMENT,
                        importance=0.8,
                        emotional_impact=0.5
                    )
                    self.happiness += 0.2
    
    def _progress_goals(self, action: str):
        """Progress goals based on actions taken"""
        for goal in self.goals:
            if goal.completed:
                continue
                
            if action == "work" and ("currency" in goal.description.lower() or "skills" in goal.description.lower()):
                self.happiness += 0.02  # Satisfaction from progress
            elif action == "socialize" and "friends" in goal.description.lower():
                self.happiness += 0.02
    
    def _update_state(self):
        """Update agent state each step"""
        # Age slowly
        self.age += 0.001
        
        # Consume food
        if self.resources["food"] > 0:
            self.resources["food"] -= 0.2
        else:
            self.energy -= 0.05  # Starving
            self.happiness -= 0.02
        
        # Health effects
        if self.energy < 0.2:
            self.health -= 0.01
        elif self.energy > 0.8:
            self.health = min(1.0, self.health + 0.001)
        
        # Happiness effects from relationships
        friend_count = len([r for r in self.relationships.values() if r.strength > 0.5])
        if friend_count > 3:
            self.happiness += 0.01
        elif friend_count == 0:
            self.happiness -= 0.005
        
        # Personality-based happiness changes
        if self.personality_traits["neuroticism"] > 0.7:
            self.happiness -= 0.005  # Neurotic agents lose happiness faster
        
        # Natural happiness decay
        self.happiness = max(0.1, self.happiness - 0.005)
        
        # Goal satisfaction affects happiness
        completed_goals = len([g for g in self.goals if g.completed])
        total_goals = len(self.goals)
        if total_goals > 0:
            goal_satisfaction = completed_goals / total_goals
            if goal_satisfaction > 0.5:
                self.happiness += 0.01
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status"""
        return {
            "agent_id": self.agent_id,
            "type": self.agent_type.value,
            "cultural_group": self.cultural_group.value,
            "position": asdict(self.position),
            "state": self.state.value,
            "energy": round(self.energy, 2),
            "happiness": round(self.happiness, 2),
            "health": round(self.health, 2),
            "age": round(self.age, 1),
            "resources": self.resources.copy(),
            "personality": {k: round(v, 2) for k, v in self.personality_traits.items()},
            "social": {
                "relationships": len(self.relationships),
                "friends": len([r for r in self.relationships.values() if r.strength > 0.5]),
                "reputation": round(self.social_reputation, 2)
            },
            "memory": {
                "short_term": len(self.short_term_memory),
                "long_term": len(self.long_term_memory),
                "working": len(self.working_memory)
            },
            "goals": {
                "total": len(self.goals),
                "completed": len([g for g in self.goals if g.completed]),
                "active": [g.description for g in self.goals if not g.completed][:3]
            },
            "last_decision": self.last_decision_reason[:100] if self.last_decision_reason else ""
        }

# Usage example for testing
async def test_intelligent_agent():
    """Test the intelligent agent system"""
    print("🧠 Testing Intelligent Agent System")
    
    from llm_integration import LLMManager, LLMProvider
    
    # Create LLM manager
    llm_manager = LLMManager(LLMProvider.MOCK)
    
    # Create intelligent agent
    agent = IntelligentAgent("test_agent_001", (100, 100), llm_manager)
    
    print(f"Created agent: {agent.agent_id}")
    print(f"Personality: {agent.personality_description}")
    print(f"Goals: {[g.description for g in agent.goals]}")
    
    # Create a mock world
    class MockWorld:
        def __init__(self):
            self.agents = [agent]
            self.step_count = 0
            self.world_size = (100, 100)
    
    world = MockWorld()
    
    # Run a few steps
    for i in range(5):
        world.step_count = i
        await agent.step(world)
        print(f"Step {i}: State={agent.state.value}, Energy={agent.energy:.2f}, Decision='{agent.last_decision_reason[:50]}...'")
    
    # Show final status
    status = agent.get_status()
    print(f"\nFinal Status:")
    print(f"  Resources: {status['resources']}")
    print(f"  Goals: {status['goals']['active']}")
    print(f"  Memory: {status['memory']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_intelligent_agent())