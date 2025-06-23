#!/usr/bin/env python3
"""
Simple Society Simulator Demo - No External Dependencies
A minimal working version to test core concepts
"""

import random
import time
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from enum import Enum

class AgentState(Enum):
    IDLE = "idle"
    MOVING = "moving"
    SOCIALIZING = "socializing"
    WORKING = "working"

@dataclass
class Position:
    x: float
    y: float
    z: float = 0.0
    
    def distance_to(self, other: "Position") -> float:
        return ((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)**0.5
    
    def move_towards(self, target: "Position", speed: float) -> "Position":
        distance = self.distance_to(target)
        if distance <= speed:
            return Position(target.x, target.y, target.z)
        
        dx = target.x - self.x
        dy = target.y - self.y
        dz = target.z - self.z
        
        # Normalize and scale by speed
        factor = speed / distance
        return Position(
            self.x + dx * factor,
            self.y + dy * factor, 
            self.z + dz * factor
        )

class SimpleAgent:
    """Simple agent without LLM dependencies"""
    
    def __init__(self, agent_id: str, position: Position = None):
        self.agent_id = agent_id
        self.position = position or Position(
            random.uniform(0, 100),
            random.uniform(0, 100),
            0.0
        )
        self.state = AgentState.IDLE
        self.energy = 1.0
        self.happiness = 0.5
        self.age = random.uniform(18, 65)
        self.social_connections = {}
        self.resources = {
            "food": random.randint(10, 50),
            "currency": random.randint(100, 500)
        }
        
    def step(self):
        """Simple agent behavior"""
        # Random movement
        if self.state == AgentState.IDLE and random.random() < 0.3:
            self.state = AgentState.MOVING
            self.target = Position(
                random.uniform(0, 100),
                random.uniform(0, 100),
                0.0
            )
        
        if self.state == AgentState.MOVING:
            self.position = self.position.move_towards(self.target, 2.0)
            if self.position.distance_to(self.target) < 1.0:
                self.state = AgentState.IDLE
        
        # Age and energy
        self.age += 0.001  # Age slowly
        self.energy = max(0.1, self.energy - 0.01 + random.uniform(0, 0.02))
        
        # Resource consumption
        self.resources["food"] = max(0, self.resources["food"] - 0.1)
        
    def interact_with(self, other: "SimpleAgent"):
        """Simple social interaction"""
        if self.agent_id not in other.social_connections:
            other.social_connections[self.agent_id] = 0.1
            self.social_connections[other.agent_id] = 0.1
        else:
            # Strengthen connection
            other.social_connections[self.agent_id] = min(1.0, 
                other.social_connections[self.agent_id] + 0.1)
            self.social_connections[other.agent_id] = min(1.0,
                self.social_connections[other.agent_id] + 0.1)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "position": asdict(self.position),
            "state": self.state.value,
            "energy": self.energy,
            "happiness": self.happiness,
            "age": self.age,
            "resources": self.resources,
            "social_connections": len(self.social_connections)
        }

class SimpleSocietySimulator:
    """Minimal society simulator"""
    
    def __init__(self, num_agents: int = 20):
        self.agents = []
        self.step_count = 0
        self.world_size = (100, 100)
        
        # Create agents
        for i in range(num_agents):
            agent = SimpleAgent(f"agent_{i}")
            self.agents.append(agent)
        
        print(f"Created {len(self.agents)} agents")
    
    def step(self):
        """Run one simulation step"""
        self.step_count += 1
        
        # Update all agents
        for agent in self.agents:
            agent.step()
        
        # Social interactions (agents within 10 units)
        for i, agent_a in enumerate(self.agents):
            for agent_b in self.agents[i+1:]:
                distance = agent_a.position.distance_to(agent_b.position)
                if distance < 10.0 and random.random() < 0.1:
                    agent_a.interact_with(agent_b)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get simulation statistics"""
        total_energy = sum(agent.energy for agent in self.agents)
        total_connections = sum(len(agent.social_connections) for agent in self.agents)
        avg_age = sum(agent.age for agent in self.agents) / len(self.agents)
        
        return {
            "step": self.step_count,
            "agents": len(self.agents),
            "avg_energy": total_energy / len(self.agents),
            "total_connections": total_connections,
            "avg_age": avg_age,
            "avg_food": sum(agent.resources["food"] for agent in self.agents) / len(self.agents)
        }
    
    def run(self, steps: int = 100):
        """Run simulation"""
        print(f"Running simulation for {steps} steps...")
        start_time = time.time()
        
        for step in range(steps):
            self.step()
            
            # Print stats every 20 steps
            if step % 20 == 0:
                stats = self.get_stats()
                print(f"Step {step:3d}: "
                      f"Energy: {stats['avg_energy']:.2f}, "
                      f"Connections: {stats['total_connections']:3d}, "
                      f"Age: {stats['avg_age']:.1f}")
        
        elapsed = time.time() - start_time
        sps = steps / elapsed
        
        print(f"\nCompleted {steps} steps in {elapsed:.2f}s ({sps:.1f} SPS)")
        
        # Final stats
        final_stats = self.get_stats()
        print(f"\nFinal Statistics:")
        print(f"  Agents: {final_stats['agents']}")
        print(f"  Average Energy: {final_stats['avg_energy']:.2f}")
        print(f"  Social Connections: {final_stats['total_connections']}")
        print(f"  Average Age: {final_stats['avg_age']:.1f}")
        print(f"  Average Food: {final_stats['avg_food']:.1f}")
        
        return final_stats

def main():
    """Main demo function"""
    print("🤖 Simple Society Simulator Demo")
    print("=" * 40)
    
    # Test different population sizes
    for num_agents in [10, 25, 50]:
        print(f"\n📊 Testing with {num_agents} agents:")
        simulator = SimpleSocietySimulator(num_agents)
        stats = simulator.run(100)
        
        # Calculate social connectivity
        connectivity = stats['total_connections'] / stats['agents']
        print(f"  Social Connectivity: {connectivity:.1f} connections per agent")

if __name__ == "__main__":
    main()