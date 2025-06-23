"""
Intelligent World - Society simulation with LLM-driven agents
Enhanced version of SocietyWorld using IntelligentAgent
"""

import asyncio
import random
import time
from collections import defaultdict
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from intelligent_agent import Goal, IntelligentAgent, MemoryType, Relationship
from llm_integration import LLMManager, LLMProvider
from society_demo import AgentType, CulturalGroup


class IntelligentWorld:
    """Enhanced world with intelligent agents and emergent behaviors"""

    def __init__(
        self,
        num_agents: int = 50,
        world_size: Tuple[float, float] = (100, 100),
        llm_provider: LLMProvider = LLMProvider.NONE,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.world_size = world_size
        self.agents: List[IntelligentAgent] = []
        self.step_count = 0

        # LLM integration
        self.llm_manager = (
            LLMManager(llm_provider, api_key, model)
            if llm_provider != LLMProvider.NONE
            else None
        )

        # World state tracking
        self.total_resources = {}
        self.social_network = {}  # Track all relationships
        self.family_trees = {}  # Track family relationships
        self.cultural_influence = defaultdict(int)  # Track cultural dominance

        # Events and dynamics
        self.recent_events = []
        self.world_mood = 0.5  # Overall societal mood
        self.economic_pressure = 0.3  # Economic stress level

        # Create intelligent agents
        self._create_intelligent_agents(num_agents)

        print(f"Created intelligent world with {len(self.agents)} agents")
        if self.llm_manager:
            print(
                f"LLM integration: {llm_provider.value} with model {model or 'default'}"
            )
        else:
            print("Using rule-based intelligence (no LLM)")

    def _create_intelligent_agents(self, num_agents: int):
        """Create intelligent agents with diverse personalities"""
        for i in range(num_agents):
            agent = IntelligentAgent(f"agent_{i}", self.world_size, self.llm_manager)
            self.agents.append(agent)

        # Create some initial relationships for social dynamics
        self._create_initial_relationships()

        # Create some families
        self._create_initial_families()

    def _create_initial_relationships(self):
        """Create some initial relationships between agents"""
        num_initial_relationships = min(len(self.agents) // 3, 20)

        for _ in range(num_initial_relationships):
            agent1 = random.choice(self.agents)
            agent2 = random.choice(self.agents)

            if (
                agent1.agent_id != agent2.agent_id
                and agent2.agent_id not in agent1.relationships
            ):
                # Create mutual relationship
                compatibility = agent1._calculate_compatibility(agent2)

                if compatibility > 0.4:  # Only create if somewhat compatible
                    rel_type = "friend" if compatibility > 0.7 else "acquaintance"
                    strength = 0.2 + compatibility * 0.3

                    agent1.relationships[agent2.agent_id] = Relationship(
                        agent_id=agent2.agent_id,
                        relationship_type=rel_type,
                        strength=strength,
                        trust=0.3 + compatibility * 0.2,
                    )

                    agent2.relationships[agent1.agent_id] = Relationship(
                        agent_id=agent1.agent_id,
                        relationship_type=rel_type,
                        strength=strength,
                        trust=0.3 + compatibility * 0.2,
                    )

    def _create_initial_families(self):
        """Create some initial family units"""
        available_agents = [a for a in self.agents if a.family_id is None]
        family_id = 1

        while len(available_agents) >= 2:
            # Create family of 2-4 agents
            family_size = min(random.randint(2, 4), len(available_agents))
            family_members = random.sample(available_agents, family_size)

            # Assign family ID
            for member in family_members:
                member.family_id = family_id
                available_agents.remove(member)

            # Create family relationships
            for i, member1 in enumerate(family_members):
                for member2 in family_members[i + 1 :]:
                    # Family members have strong positive relationships
                    rel_type = "family"
                    strength = random.uniform(0.6, 0.9)
                    trust = random.uniform(0.7, 1.0)

                    member1.relationships[member2.agent_id] = Relationship(
                        agent_id=member2.agent_id,
                        relationship_type=rel_type,
                        strength=strength,
                        trust=trust,
                    )

                    member2.relationships[member1.agent_id] = Relationship(
                        agent_id=member1.agent_id,
                        relationship_type=rel_type,
                        strength=strength,
                        trust=trust,
                    )

            family_id += 1

            # Don't create too many families
            if family_id > len(self.agents) // 3:
                break

    async def step(self):
        """Run one simulation step with intelligent agents"""
        self.step_count += 1

        # Shuffle agents for random order
        agents_shuffled = self.agents.copy()
        random.shuffle(agents_shuffled)

        # Step all agents asynchronously in batches to manage LLM load
        batch_size = 10  # Process 10 agents at a time for LLM efficiency
        for i in range(0, len(agents_shuffled), batch_size):
            batch = agents_shuffled[i : i + batch_size]
            await asyncio.gather(*[agent.step(self) for agent in batch])

        # Update world state
        self._update_world_dynamics()

        # Periodic events
        if self.step_count % 50 == 0:
            await self._world_event()

    def _update_world_dynamics(self):
        """Update global world dynamics"""
        # Update resource totals
        self.total_resources = {
            "food": sum(agent.resources["food"] for agent in self.agents),
            "currency": sum(agent.resources["currency"] for agent in self.agents),
            "materials": sum(agent.resources["materials"] for agent in self.agents),
            "tools": sum(agent.resources["tools"] for agent in self.agents),
        }

        # Update world mood based on agent happiness
        avg_happiness = sum(agent.happiness for agent in self.agents) / len(self.agents)
        self.world_mood = 0.7 * self.world_mood + 0.3 * avg_happiness

        # Update economic pressure
        avg_currency = self.total_resources["currency"] / len(self.agents)
        if avg_currency < 200:
            self.economic_pressure = min(1.0, self.economic_pressure + 0.1)
        elif avg_currency > 800:
            self.economic_pressure = max(0.0, self.economic_pressure - 0.1)

        # Update cultural influence
        self.cultural_influence.clear()
        for agent in self.agents:
            self.cultural_influence[agent.cultural_group.value] += 1

        # Social network analysis
        self._analyze_social_network()

    def _analyze_social_network(self):
        """Analyze the social network structure"""
        # Build network representation
        self.social_network = {}
        for agent in self.agents:
            connections = []
            for rel_id, relationship in agent.relationships.items():
                if relationship.strength > 0.3:  # Only count meaningful relationships
                    connections.append(
                        {
                            "agent_id": rel_id,
                            "strength": relationship.strength,
                            "type": relationship.relationship_type,
                        }
                    )
            self.social_network[agent.agent_id] = connections

    async def _world_event(self):
        """Generate world events that affect all agents"""
        event_types = [
            ("cultural_festival", 0.3, "positive"),
            ("economic_boom", 0.2, "positive"),
            ("resource_discovery", 0.15, "positive"),
            ("natural_disaster", 0.2, "negative"),
            ("economic_crisis", 0.1, "negative"),
            ("cultural_conflict", 0.05, "negative"),
        ]

        event_type, probability, valence = random.choice(event_types)

        if random.random() < probability:
            await self._execute_world_event(event_type, valence)

    async def _execute_world_event(self, event_type: str, valence: str):
        """Execute a world event affecting all agents"""
        self.recent_events.append(
            {"type": event_type, "step": self.step_count, "valence": valence}
        )

        # Keep only recent events
        if len(self.recent_events) > 10:
            self.recent_events = self.recent_events[-10:]

        print(f"   🌟 World Event: {event_type}")

        if event_type == "cultural_festival":
            # Boost happiness and cultural connections
            for agent in self.agents:
                agent.happiness += random.uniform(0.1, 0.3)
                agent._add_memory(
                    f"Attended the grand cultural festival celebrating {agent.cultural_group.value} traditions",
                    MemoryType.EVENT,
                    importance=0.6,
                    emotional_impact=0.3,
                )

        elif event_type == "economic_boom":
            # Increase currency for all agents
            for agent in self.agents:
                bonus = random.randint(50, 200)
                agent.resources["currency"] += bonus
                agent._add_memory(
                    f"Benefited from the economic boom, earning an extra {bonus} currency",
                    MemoryType.EVENT,
                    importance=0.5,
                    emotional_impact=0.2,
                )

        elif event_type == "resource_discovery":
            # Increase materials and tools
            for agent in self.agents:
                agent.resources["materials"] += random.randint(5, 15)
                if random.random() < 0.3:
                    agent.resources["tools"] += 1

        elif event_type == "natural_disaster":
            # Reduce resources and happiness
            for agent in self.agents:
                agent.resources["food"] = max(
                    0, agent.resources["food"] - random.randint(5, 20)
                )
                agent.happiness -= random.uniform(0.1, 0.2)
                agent._add_memory(
                    "Survived a devastating natural disaster that affected the whole community",
                    MemoryType.EVENT,
                    importance=0.8,
                    emotional_impact=-0.4,
                )

        elif event_type == "economic_crisis":
            # Reduce currency
            for agent in self.agents:
                loss = int(agent.resources["currency"] * random.uniform(0.1, 0.3))
                agent.resources["currency"] = max(0, agent.resources["currency"] - loss)
                agent._add_memory(
                    f"Lost {loss} currency due to the economic crisis",
                    MemoryType.EVENT,
                    importance=0.7,
                    emotional_impact=-0.3,
                )

        elif event_type == "cultural_conflict":
            # Create tension between cultural groups
            groups = list(CulturalGroup)
            group1, group2 = random.sample(groups, 2)

            for agent in self.agents:
                if agent.cultural_group in [group1, group2]:
                    agent.happiness -= random.uniform(0.05, 0.15)

                    # Create negative relationships with other group
                    for other in self.agents:
                        if (
                            other.cultural_group != agent.cultural_group
                            and other.cultural_group in [group1, group2]
                            and other.agent_id in agent.relationships
                        ):
                            agent.relationships[other.agent_id].strength -= 0.2
                            agent.relationships[other.agent_id].trust -= 0.1

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive world statistics"""
        # Basic statistics
        basic_stats = {
            "step": self.step_count,
            "agents": len(self.agents),
            "world_mood": round(self.world_mood, 3),
            "economic_pressure": round(self.economic_pressure, 3),
        }

        # Agent statistics
        avg_energy = sum(agent.energy for agent in self.agents) / len(self.agents)
        avg_happiness = sum(agent.happiness for agent in self.agents) / len(self.agents)
        avg_health = sum(agent.health for agent in self.agents) / len(self.agents)
        avg_age = sum(agent.age for agent in self.agents) / len(self.agents)

        # Relationship statistics
        total_relationships = sum(len(agent.relationships) for agent in self.agents)
        total_friendships = sum(
            len([r for r in agent.relationships.values() if r.strength > 0.5])
            for agent in self.agents
        )
        family_count = len(
            set(agent.family_id for agent in self.agents if agent.family_id is not None)
        )

        # Goal statistics
        total_goals = sum(len(agent.goals) for agent in self.agents)
        completed_goals = sum(
            len([g for g in agent.goals if g.completed]) for agent in self.agents
        )

        # Memory statistics
        total_memories = sum(len(agent.long_term_memory) for agent in self.agents)
        avg_memories = total_memories / len(self.agents) if self.agents else 0

        # Agent type distribution
        type_counts = defaultdict(int)
        for agent in self.agents:
            type_counts[agent.agent_type.value] += 1

        # Cultural distribution
        cultural_counts = dict(self.cultural_influence)

        # Advanced statistics
        if self.llm_manager:
            llm_stats = self.llm_manager.get_stats()
        else:
            llm_stats = {"provider": "none", "total_requests": 0, "cache_rate": "0%"}

        return {
            **basic_stats,
            "averages": {
                "energy": round(avg_energy, 3),
                "happiness": round(avg_happiness, 3),
                "health": round(avg_health, 3),
                "age": round(avg_age, 1),
            },
            "social": {
                "total_relationships": total_relationships,
                "friendships": total_friendships,
                "families": family_count,
                "social_density": round(total_relationships / len(self.agents), 2),
            },
            "goals": {
                "total": total_goals,
                "completed": completed_goals,
                "completion_rate": round(
                    completed_goals / max(1, total_goals) * 100, 1
                ),
            },
            "memory": {
                "total_memories": total_memories,
                "avg_per_agent": round(avg_memories, 1),
            },
            "total_resources": self.total_resources,
            "agent_types": dict(type_counts),
            "cultural_groups": cultural_counts,
            "recent_events": self.recent_events[-3:],  # Last 3 events
            "llm_stats": llm_stats,
        }

    def get_agent_insights(self) -> Dict[str, Any]:
        """Get insights about agent behaviors and patterns"""
        insights = {
            "most_social": [],
            "most_successful": [],
            "most_goal_oriented": [],
            "relationship_networks": [],
            "personality_clusters": defaultdict(list),
        }

        # Most social agents
        social_scores = [
            (agent.agent_id, len(agent.relationships)) for agent in self.agents
        ]
        social_scores.sort(key=lambda x: x[1], reverse=True)
        insights["most_social"] = social_scores[:5]

        # Most successful (by currency)
        wealth_scores = [
            (agent.agent_id, agent.resources["currency"]) for agent in self.agents
        ]
        wealth_scores.sort(key=lambda x: x[1], reverse=True)
        insights["most_successful"] = wealth_scores[:5]

        # Most goal-oriented
        goal_scores = [
            (agent.agent_id, len([g for g in agent.goals if g.completed]))
            for agent in self.agents
        ]
        goal_scores.sort(key=lambda x: x[1], reverse=True)
        insights["most_goal_oriented"] = goal_scores[:5]

        # Personality clustering (simplified)
        for agent in self.agents:
            if agent.personality_traits["extroversion"] > 0.7:
                insights["personality_clusters"]["extroverts"].append(agent.agent_id)
            if agent.personality_traits["conscientiousness"] > 0.7:
                insights["personality_clusters"]["conscientious"].append(agent.agent_id)
            if agent.personality_traits["openness"] > 0.7:
                insights["personality_clusters"]["creative"].append(agent.agent_id)
            if agent.personality_traits["agreeableness"] > 0.7:
                insights["personality_clusters"]["agreeable"].append(agent.agent_id)

        return insights


async def run_intelligent_simulation(
    num_agents: int = 50,
    steps: int = 200,
    llm_provider: LLMProvider = LLMProvider.NONE,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
):
    """Run intelligent society simulation"""
    print("🧠 Intelligent Society Simulation")
    print(f"   Agents: {num_agents}")
    print(f"   Steps: {steps}")
    print(f"   Intelligence: {llm_provider.value}")
    print("=" * 50)

    world = IntelligentWorld(
        num_agents, llm_provider=llm_provider, api_key=api_key, model=model
    )

    start_time = time.time()

    for step in range(steps):
        await world.step()

        # Print statistics every 40 steps
        if step % 40 == 0:
            stats = world.get_statistics()
            print(
                f"Step {step:3d}: "
                f"Mood: {stats['world_mood']:.2f}, "
                f"Happiness: {stats['averages']['happiness']:.2f}, "
                f"Relationships: {stats['social']['total_relationships']}, "
                f"Goals: {stats['goals']['completed']}/{stats['goals']['total']}"
            )

    elapsed = time.time() - start_time
    sps = steps / elapsed

    print(f"\nCompleted in {elapsed:.2f}s ({sps:.1f} SPS)")

    # Final analysis
    final_stats = world.get_statistics()
    insights = world.get_agent_insights()

    print("\n🧠 Intelligent Simulation Results:")
    print(f"   World Mood: {final_stats['world_mood']:.2f}")
    print(f"   Average Happiness: {final_stats['averages']['happiness']:.2f}")
    print(
        f"   Social Density: {final_stats['social']['social_density']} relationships per agent"
    )
    print(f"   Goal Completion: {final_stats['goals']['completion_rate']}%")
    print(f"   Total Memories: {final_stats['memory']['total_memories']}")
    print(f"   Families Formed: {final_stats['social']['families']}")

    if final_stats["llm_stats"]["total_requests"] > 0:
        print(f"   LLM Requests: {final_stats['llm_stats']['total_requests']}")
        print(f"   Cache Efficiency: {final_stats['llm_stats']['cache_rate']}")

    print("\n👥 Agent Insights:")
    print(f"   Most Social: {insights['most_social'][:3]}")
    print(f"   Wealthiest: {insights['most_successful'][:3]}")
    print(f"   Goal Achievers: {insights['most_goal_oriented'][:3]}")

    for personality, agents in insights["personality_clusters"].items():
        if agents:
            print(f"   {personality.title()}: {len(agents)} agents")

    return world


if __name__ == "__main__":
    import asyncio

    # Test intelligent simulation
    asyncio.run(run_intelligent_simulation(25, 150, LLMProvider.MOCK))
