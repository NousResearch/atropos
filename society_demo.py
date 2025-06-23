#!/usr/bin/env python3
"""
Society Simulator Demo - Core Functionality Working
Tests the actual society simulation concepts without problematic dependencies
"""

import random
import time
import json
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from collections import defaultdict

class AgentType(Enum):
    FARMER = "farmer"
    CRAFTSMAN = "craftsman"
    TRADER = "trader"
    SCHOLAR = "scholar"
    LEADER = "leader"
    UNEMPLOYED = "unemployed"

class CulturalGroup(Enum):
    HARMONISTS = "harmonists"
    BUILDERS = "builders"
    GUARDIANS = "guardians"
    SCHOLARS = "scholars"
    WANDERERS = "wanderers"

class AgentState(Enum):
    IDLE = "idle"
    MOVING = "moving"
    SOCIALIZING = "socializing"
    WORKING = "working"
    TRADING = "trading"

@dataclass
class Position:
    x: float
    y: float
    z: float = 0.0
    
    def distance_to(self, other: "Position") -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
    
    def move_towards(self, target: "Position", speed: float) -> "Position":
        distance = self.distance_to(target)
        if distance <= speed:
            return Position(target.x, target.y, target.z)
        
        dx = target.x - self.x
        dy = target.y - self.y
        dz = target.z - self.z
        factor = speed / distance
        
        return Position(
            self.x + dx * factor,
            self.y + dy * factor,
            self.z + dz * factor
        )

@dataclass
class Memory:
    content: str
    timestamp: float
    importance: float = 0.5
    
class SocietyAgent:
    """Agent with LLM-like decision making (simplified)"""
    
    def __init__(self, agent_id: str, world_size: Tuple[float, float] = (100, 100)):
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
        
        # Social
        self.social_connections = {}  # agent_id -> strength
        self.family_id = None
        self.social_reputation = 0.5
        
        # Economic
        self.resources = {
            "food": random.randint(10, 50),
            "currency": random.randint(100, 1000),
            "materials": random.randint(5, 25),
            "tools": random.randint(1, 5)
        }
        self.employed = random.choice([True, False])
        
        # Memory and personality
        self.memories = []
        self.personality_traits = {
            "extroversion": random.random(),
            "conscientiousness": random.random(),
            "openness": random.random(),
            "agreeableness": random.random()
        }
        
        # Movement
        self.target_position = None
        self.movement_speed = 2.0
        
    def step(self, world):
        """Main agent step function"""
        # Make a decision
        decision = self._make_decision(world)
        
        # Execute the decision
        self._execute_action(decision, world)
        
        # Update state
        self._update_state()
        
    def _make_decision(self, world) -> str:
        """Simplified decision making (would be LLM in full version)"""
        context = self._gather_context(world)
        
        # Rule-based decision making based on agent type and context
        if self.energy < 0.3:
            return "rest"
        elif self.resources["food"] < 5:
            return "gather_food"
        elif len(context["nearby_agents"]) > 0 and self.personality_traits["extroversion"] > 0.7:
            return "socialize"
        elif self.agent_type == AgentType.TRADER and self.resources["currency"] > 200:
            return "trade"
        elif self.agent_type == AgentType.FARMER:
            return "work_farm"
        elif self.agent_type == AgentType.CRAFTSMAN:
            return "work_craft"
        elif random.random() < 0.3:
            return "move"
        else:
            return "work"
    
    def _gather_context(self, world) -> Dict[str, Any]:
        """Gather information about surroundings"""
        nearby_agents = []
        for other in world.agents:
            if other.agent_id != self.agent_id:
                distance = self.position.distance_to(other.position)
                if distance < 15.0:  # Interaction radius
                    nearby_agents.append({
                        "agent": other,
                        "distance": distance,
                        "type": other.agent_type.value,
                        "cultural_group": other.cultural_group.value
                    })
        
        return {
            "nearby_agents": nearby_agents,
            "my_resources": self.resources.copy(),
            "my_energy": self.energy,
            "my_happiness": self.happiness
        }
    
    def _execute_action(self, action: str, world):
        """Execute the chosen action"""
        if action == "move":
            self._move_randomly(world)
        elif action == "socialize":
            self._socialize(world)
        elif action == "trade":
            self._attempt_trade(world)
        elif action == "work" or action.startswith("work_"):
            self._work()
        elif action == "gather_food":
            self._gather_food()
        elif action == "rest":
            self._rest()
    
    def _move_randomly(self, world):
        """Move to random location"""
        if self.target_position is None or self.position.distance_to(self.target_position) < 2.0:
            self.target_position = Position(
                random.uniform(0, world.world_size[0]),
                random.uniform(0, world.world_size[1]),
                0.0
            )
        
        self.position = self.position.move_towards(self.target_position, self.movement_speed)
        self.energy -= 0.02
        self.state = AgentState.MOVING
    
    def _socialize(self, world):
        """Interact with nearby agents"""
        nearby = [agent for agent in world.agents 
                 if agent.agent_id != self.agent_id and 
                 self.position.distance_to(agent.position) < 15.0]
        
        if nearby:
            other = random.choice(nearby)
            self._interact_with(other)
        
        self.state = AgentState.SOCIALIZING
        self.energy -= 0.01
        self.happiness += 0.03
    
    def _interact_with(self, other):
        """Social interaction with another agent"""
        # Build or strengthen connection
        if other.agent_id not in self.social_connections:
            self.social_connections[other.agent_id] = 0.1
            other.social_connections[self.agent_id] = 0.1
        else:
            self.social_connections[other.agent_id] = min(1.0,
                self.social_connections[other.agent_id] + 0.05)
            other.social_connections[self.agent_id] = min(1.0,
                other.social_connections[self.agent_id] + 0.05)
        
        # Cultural influence
        if self.cultural_group != other.cultural_group and random.random() < 0.1:
            # Slight chance to be influenced by other culture
            if random.random() < 0.3:
                self.cultural_group = other.cultural_group
    
    def _attempt_trade(self, world):
        """Try to trade with nearby agents"""
        nearby_traders = [agent for agent in world.agents 
                         if agent.agent_id != self.agent_id and 
                         self.position.distance_to(agent.position) < 20.0 and
                         agent.resources["currency"] > 50]
        
        if nearby_traders and self.resources["materials"] > 5:
            other = random.choice(nearby_traders)
            # Simple trade: materials for currency
            trade_amount = min(5, self.resources["materials"])
            price = trade_amount * 10
            
            if other.resources["currency"] >= price:
                self.resources["materials"] -= trade_amount
                self.resources["currency"] += price
                other.resources["materials"] += trade_amount
                other.resources["currency"] -= price
                
                # Record trade memory
                self._add_memory(f"Traded {trade_amount} materials for {price} currency with {other.agent_id}")
                other._add_memory(f"Bought {trade_amount} materials for {price} currency from {self.agent_id}")
        
        self.state = AgentState.TRADING
        self.energy -= 0.02
    
    def _work(self):
        """Work to produce resources"""
        if self.agent_type == AgentType.FARMER:
            self.resources["food"] += random.randint(3, 8)
        elif self.agent_type == AgentType.CRAFTSMAN:
            if self.resources["materials"] > 2:
                self.resources["materials"] -= 2
                self.resources["tools"] += 1
                self.resources["currency"] += 15
        elif self.agent_type == AgentType.TRADER:
            self.resources["currency"] += random.randint(5, 20)
        else:
            self.resources["currency"] += random.randint(3, 10)
        
        self.state = AgentState.WORKING
        self.energy -= 0.05
        self.happiness += 0.02
    
    def _gather_food(self):
        """Gather food from environment"""
        self.resources["food"] += random.randint(5, 15)
        self.energy -= 0.03
    
    def _rest(self):
        """Rest to recover energy"""
        self.energy = min(1.0, self.energy + 0.15)
        self.resources["food"] -= 1
        self.state = AgentState.IDLE
    
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
        
        # Happiness decay
        self.happiness = max(0.1, self.happiness - 0.005)
        
        # Social effects
        if len(self.social_connections) > 5:
            self.happiness += 0.01
        
        # Economic pressure
        if self.resources["currency"] < 50:
            self.happiness -= 0.01
    
    def _add_memory(self, content: str):
        """Add a memory"""
        memory = Memory(content, time.time(), random.uniform(0.3, 0.9))
        self.memories.append(memory)
        
        # Keep only recent memories (simple version)
        if len(self.memories) > 20:
            self.memories = sorted(self.memories, key=lambda m: m.importance, reverse=True)[:15]
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status for monitoring"""
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
            "social_connections": len(self.social_connections),
            "memories": len(self.memories)
        }

class SocietyWorld:
    """World container for the society simulation"""
    
    def __init__(self, num_agents: int = 50, world_size: Tuple[float, float] = (100, 100)):
        self.world_size = world_size
        self.agents = []
        self.step_count = 0
        
        # Create agents
        for i in range(num_agents):
            agent = SocietyAgent(f"agent_{i}", world_size)
            self.agents.append(agent)
        
        # World state
        self.total_resources = {
            "food": sum(agent.resources["food"] for agent in self.agents),
            "currency": sum(agent.resources["currency"] for agent in self.agents),
            "materials": sum(agent.resources["materials"] for agent in self.agents)
        }
        
        print(f"Created world with {len(self.agents)} agents")
    
    def step(self):
        """Run one simulation step"""
        self.step_count += 1
        
        # Shuffle agents for random order
        agents_shuffled = self.agents.copy()
        random.shuffle(agents_shuffled)
        
        # Update all agents
        for agent in agents_shuffled:
            agent.step(self)
        
        # Update world state
        self._update_world_state()
    
    def _update_world_state(self):
        """Update global world state"""
        self.total_resources = {
            "food": sum(agent.resources["food"] for agent in self.agents),
            "currency": sum(agent.resources["currency"] for agent in self.agents),
            "materials": sum(agent.resources["materials"] for agent in self.agents),
            "tools": sum(agent.resources["tools"] for agent in self.agents)
        }
        
        # Population events (simplified)
        if self.step_count % 100 == 0:
            self._population_event()
    
    def _population_event(self):
        """Handle population-level events"""
        event_type = random.choice(["disaster", "good_harvest", "trade_boom", "cultural_festival"])
        
        if event_type == "disaster":
            # Reduce resources
            for agent in self.agents:
                agent.resources["food"] = max(0, agent.resources["food"] - random.randint(5, 15))
                agent.happiness -= 0.1
        elif event_type == "good_harvest":
            # Increase food
            for agent in self.agents:
                if agent.agent_type == AgentType.FARMER:
                    agent.resources["food"] += random.randint(10, 25)
        elif event_type == "trade_boom":
            # Increase currency for traders
            for agent in self.agents:
                if agent.agent_type == AgentType.TRADER:
                    agent.resources["currency"] += random.randint(50, 150)
        elif event_type == "cultural_festival":
            # Increase happiness
            for agent in self.agents:
                agent.happiness = min(1.0, agent.happiness + 0.15)
        
        print(f"   🌟 World Event: {event_type}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get world statistics"""
        # Agent type distribution
        type_counts = defaultdict(int)
        cultural_counts = defaultdict(int)
        state_counts = defaultdict(int)
        
        total_energy = 0
        total_happiness = 0
        total_connections = 0
        
        for agent in self.agents:
            type_counts[agent.agent_type.value] += 1
            cultural_counts[agent.cultural_group.value] += 1
            state_counts[agent.state.value] += 1
            total_energy += agent.energy
            total_happiness += agent.happiness
            total_connections += len(agent.social_connections)
        
        return {
            "step": self.step_count,
            "agents": len(self.agents),
            "avg_energy": total_energy / len(self.agents),
            "avg_happiness": total_happiness / len(self.agents),
            "total_connections": total_connections,
            "total_resources": self.total_resources,
            "agent_types": dict(type_counts),
            "cultural_groups": dict(cultural_counts),
            "agent_states": dict(state_counts)
        }

def run_society_simulation(num_agents: int = 50, steps: int = 200):
    """Run the society simulation"""
    print(f"🏘️  Society Simulation")
    print(f"   Agents: {num_agents}")
    print(f"   Steps: {steps}")
    print("=" * 50)
    
    world = SocietyWorld(num_agents)
    
    start_time = time.time()
    
    for step in range(steps):
        world.step()
        
        # Print statistics every 40 steps
        if step % 40 == 0:
            stats = world.get_statistics()
            print(f"Step {step:3d}: "
                  f"Energy: {stats['avg_energy']:.2f}, "
                  f"Happiness: {stats['avg_happiness']:.2f}, "
                  f"Connections: {stats['total_connections']}, "
                  f"Currency: {stats['total_resources']['currency']}")
    
    elapsed = time.time() - start_time
    sps = steps / elapsed
    
    print(f"\nCompleted in {elapsed:.2f}s ({sps:.1f} SPS)")
    
    # Final analysis
    final_stats = world.get_statistics()
    print(f"\n📊 Final Results:")
    print(f"   Average Energy: {final_stats['avg_energy']:.2f}")
    print(f"   Average Happiness: {final_stats['avg_happiness']:.2f}")
    print(f"   Social Connections: {final_stats['total_connections']}")
    print(f"   Total Resources: {final_stats['total_resources']}")
    
    print(f"\n👥 Agent Types:")
    for agent_type, count in final_stats['agent_types'].items():
        print(f"   {agent_type}: {count}")
    
    print(f"\n🌍 Cultural Groups:")
    for group, count in final_stats['cultural_groups'].items():
        print(f"   {group}: {count}")
    
    return world

def main():
    """Main function"""
    print("🤖 Society Simulator - Core Functionality Demo")
    print("=" * 60)
    
    # Test different scales
    for num_agents in [25, 50, 100]:
        print(f"\n🔬 Testing {num_agents} agents:")
        world = run_society_simulation(num_agents, 200)
        
        # Analyze emergent properties
        final_stats = world.get_statistics()
        social_density = final_stats['total_connections'] / num_agents
        cultural_diversity = len(final_stats['cultural_groups'])
        
        print(f"   Social Density: {social_density:.1f} connections per agent")
        print(f"   Cultural Diversity: {cultural_diversity} active groups")

if __name__ == "__main__":
    main()