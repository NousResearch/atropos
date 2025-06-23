#!/usr/bin/env python3
"""
Working Society Simulator Demo
Uses actual Mesa framework but removes problematic dependencies
"""

import random
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import mesa
import numpy as np
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.space import ContinuousSpace


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
        return np.sqrt(
            (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2
        )


class WorkingAgent(Agent):
    """Mesa-compatible agent without complex dependencies"""

    def __init__(self, unique_id: str, model, position: Position = None):
        super().__init__(unique_id, model)
        self.position = position or Position(
            random.uniform(0, model.space.width),
            random.uniform(0, model.space.height),
            0.0,
        )
        self.state = AgentState.IDLE
        self.energy = 1.0
        self.happiness = random.uniform(0.3, 0.7)
        self.age = random.uniform(18, 65)
        self.social_connections = {}
        self.resources = {
            "food": random.randint(10, 50),
            "currency": random.randint(100, 500),
            "tools": random.randint(1, 5),
        }
        self.personality = random.choice(
            ["extrovert", "introvert", "analytical", "creative"]
        )

    def step(self):
        """Agent behavior step"""
        # Simple decision making without LLM
        decision = self._make_decision()
        self._execute_action(decision)

        # Update agent state
        self._update_state()

    def _make_decision(self) -> str:
        """Simple rule-based decision making"""
        nearby_agents = self.model.space.get_neighbors(
            (self.position.x, self.position.y), radius=10, include_center=False
        )

        # Decision logic based on personality and context
        if self.energy < 0.3:
            return "rest"
        elif (
            len(nearby_agents) > 0
            and self.personality == "extrovert"
            and random.random() < 0.4
        ):
            return "socialize"
        elif self.resources["food"] < 5:
            return "gather_food"
        elif random.random() < 0.2:
            return "move"
        else:
            return "work"

    def _execute_action(self, action: str):
        """Execute the chosen action"""
        if action == "move":
            self._move_randomly()
        elif action == "socialize":
            self._socialize()
        elif action == "work":
            self._work()
        elif action == "gather_food":
            self._gather_food()
        elif action == "rest":
            self._rest()

    def _move_randomly(self):
        """Move to a random nearby location"""
        new_x = max(
            0, min(self.model.space.width, self.position.x + random.uniform(-5, 5))
        )
        new_y = max(
            0, min(self.model.space.height, self.position.y + random.uniform(-5, 5))
        )
        self.position = Position(new_x, new_y, self.position.z)
        self.model.space.move_agent(self, (new_x, new_y))
        self.energy -= 0.02

    def _socialize(self):
        """Interact with nearby agents"""
        nearby_agents = self.model.space.get_neighbors(
            (self.position.x, self.position.y), radius=10, include_center=False
        )

        for other in nearby_agents:
            if hasattr(other, "social_connections"):
                # Build social connections
                if other.unique_id not in self.social_connections:
                    self.social_connections[other.unique_id] = 0.1
                    other.social_connections[self.unique_id] = 0.1
                else:
                    self.social_connections[other.unique_id] = min(
                        1.0, self.social_connections[other.unique_id] + 0.05
                    )

                # Trade occasionally
                if random.random() < 0.1:
                    self._trade_with(other)

        self.happiness += 0.05
        self.energy -= 0.01

    def _work(self):
        """Work to produce resources"""
        if self.personality == "analytical":
            # Produce tools
            self.resources["tools"] += 1
            self.resources["currency"] += 5
        else:
            # Produce food or currency
            self.resources["food"] += 2
            self.resources["currency"] += 3

        self.energy -= 0.05

    def _gather_food(self):
        """Gather food from environment"""
        self.resources["food"] += random.randint(3, 8)
        self.energy -= 0.03

    def _rest(self):
        """Rest to recover energy"""
        self.energy = min(1.0, self.energy + 0.1)
        self.resources["food"] -= 0.5

    def _trade_with(self, other):
        """Simple trading mechanism"""
        if self.resources["currency"] > 10 and other.resources["food"] > 5:
            # Buy food
            trade_amount = min(5, other.resources["food"])
            cost = trade_amount * 2

            if self.resources["currency"] >= cost:
                self.resources["food"] += trade_amount
                self.resources["currency"] -= cost
                other.resources["food"] -= trade_amount
                other.resources["currency"] += cost

    def _update_state(self):
        """Update agent state"""
        # Age slowly
        self.age += 0.001

        # Consume food
        if self.resources["food"] > 0:
            self.resources["food"] -= 0.1
        else:
            self.energy -= 0.02  # Starving

        # Happiness decay
        self.happiness = max(0.1, self.happiness - 0.01)


class SocietyModel(Model):
    """Mesa model for society simulation"""

    def __init__(self, num_agents: int = 50, width: int = 100, height: int = 100):
        super().__init__()

        self.num_agents = num_agents
        self.space = ContinuousSpace(width, height, torus=True)
        # Create a simple scheduler since mesa.time doesn't exist in newer versions
        self.agent_list = []

        # Create agents
        for i in range(num_agents):
            agent = WorkingAgent(f"agent_{i}", self)
            self.agent_list.append(agent)
            self.space.place_agent(agent, (agent.position.x, agent.position.y))

        # Data collection
        self.datacollector = DataCollector(
            model_reporters={
                "Total_Energy": lambda m: sum(agent.energy for agent in m.agent_list),
                "Average_Happiness": lambda m: sum(
                    agent.happiness for agent in m.agent_list
                )
                / len(m.agent_list),
                "Total_Connections": lambda m: sum(
                    len(agent.social_connections) for agent in m.agent_list
                ),
                "Average_Age": lambda m: sum(agent.age for agent in m.agent_list)
                / len(m.agent_list),
                "Total_Currency": lambda m: sum(
                    agent.resources["currency"] for agent in m.agent_list
                ),
            }
        )

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        """Run one model step"""
        # Step all agents
        for agent in self.agent_list:
            agent.step()
        self.datacollector.collect(self)


def run_simulation(num_agents: int = 50, steps: int = 200):
    """Run a society simulation"""
    print("🏘️  Running Society Simulation")
    print(f"   Agents: {num_agents}")
    print(f"   Steps: {steps}")
    print("=" * 50)

    model = SocietyModel(num_agents=num_agents)

    start_time = time.time()

    for step in range(steps):
        model.step()

        if step % 40 == 0:
            data = model.datacollector.get_model_vars_dataframe().iloc[-1]
            print(
                f"Step {step:3d}: "
                f"Energy: {data['Total_Energy']:.1f}, "
                f"Happiness: {data['Average_Happiness']:.2f}, "
                f"Connections: {int(data['Total_Connections'])}, "
                f"Currency: {int(data['Total_Currency'])}"
            )

    elapsed = time.time() - start_time
    sps = steps / elapsed

    print(f"\nCompleted in {elapsed:.2f}s ({sps:.1f} SPS)")

    # Final analysis
    final_data = model.datacollector.get_model_vars_dataframe().iloc[-1]
    print("\n📊 Final Results:")
    print(f"   Average Energy: {final_data['Total_Energy']/num_agents:.2f}")
    print(f"   Average Happiness: {final_data['Average_Happiness']:.2f}")
    print(f"   Social Connections: {int(final_data['Total_Connections'])}")
    print(f"   Average Age: {final_data['Average_Age']:.1f}")
    print(f"   Total Wealth: {int(final_data['Total_Currency'])}")

    # Agent analysis
    personalities = {}
    for agent in model.agent_list:
        personalities[agent.personality] = personalities.get(agent.personality, 0) + 1

    print("\n👥 Agent Personalities:")
    for personality, count in personalities.items():
        print(f"   {personality}: {count}")

    return model


def main():
    """Main function"""
    print("🤖 Working Society Simulator with Mesa")
    print("Using real Mesa framework and realistic agent behaviors")
    print("=" * 60)

    # Test different scales
    for num_agents in [25, 50, 100]:
        print(f"\n🔬 Test {num_agents} agents:")
        model = run_simulation(num_agents, 150)

        # Calculate social density
        total_connections = sum(
            len(agent.social_connections) for agent in model.agent_list
        )
        social_density = total_connections / num_agents
        print(f"   Social Density: {social_density:.1f} connections per agent")


if __name__ == "__main__":
    main()
