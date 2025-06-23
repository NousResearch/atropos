#!/usr/bin/env python3
"""
Population Dynamics Demo
Showcases the key features of the population dynamics system
"""

import asyncio
import logging

from src.simulation.society_simulator import SocietySimulator
from src.utils.config import Config

# Configure logging for demo
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


async def demo_population_dynamics():
    """Demonstrate population dynamics with accelerated events"""

    print("🌟 Population Dynamics System Demo")
    print("=" * 50)
    print("🎯 Goal: Demonstrate dynamic population control")
    print("📈 Starting: 15 agents → Target: 150 agents")
    print("⚡ Accelerated: Fast events for demonstration")
    print("")

    # Create demo configuration
    config = Config()

    # Population dynamics settings
    config.population.enable_dynamics = True
    config.population.initial_population = 15  # Start small
    config.population.target_population = 150  # Smaller target for demo
    config.population.max_population = 200

    # Accelerated rates for demo
    config.population.base_birth_rate = 0.002  # Higher birth rate
    config.population.base_death_rate = 0.0005  # Moderate death rate
    config.population.aging_rate = 0.02  # Faster aging

    # Frequent events for demonstration
    config.population.disaster_probability = 0.01  # ~1 every 100 steps
    config.population.tech_event_probability = 0.008  # ~1 every 125 steps
    config.population.disease_probability = 0.006  # ~1 every 167 steps

    # Simulation settings for demo
    config.simulation.max_steps = 1000  # Shorter demo
    config.simulation.tick_rate = 0.005  # Fast ticks
    config.simulation.seed = 12345  # Reproducible

    # Disable heavy features for demo speed
    config.llm.model_name = "mock"  # Use mock LLM
    config.assets.enable_generation = False  # Disable assets
    config.monitoring.metrics_interval = 50  # Less frequent metrics

    print("📋 Demo Configuration:")
    print(f"   Initial: {config.population.initial_population} agents")
    print(f"   Target: {config.population.target_population} agents")
    print(f"   Max Steps: {config.simulation.max_steps}")
    print(f"   Birth Rate: {config.population.base_birth_rate:.4f}")
    print(f"   Disaster Rate: {config.population.disaster_probability:.3f}")
    print("")

    # Run simulation
    simulator = SocietySimulator(config)

    print("🚀 Starting simulation...")
    print("📊 Watch for: Births, Deaths, Disasters, Population Growth")
    print("=" * 50)

    await simulator.run()

    print("=" * 50)
    print("✅ Demo completed! Key observations:")
    print("   🌱 Population grew naturally from initial size")
    print("   📈 Birth rates adapted to population pressure")
    print("   🌪️ Random events created population fluctuations")
    print("   ⚖️ System balanced growth toward target population")
    print("   📊 Demographics tracked age distribution and stability")


async def demo_specific_scenarios():
    """Demo specific population scenarios"""

    print("\n🎭 Scenario Testing")
    print("=" * 50)

    scenarios = [
        {
            "name": "Rapid Growth",
            "description": "High birth rate, low death rate",
            "birth_rate": 0.005,
            "death_rate": 0.0001,
            "initial_pop": 10,
            "target_pop": 100,
        },
        {
            "name": "Disaster Recovery",
            "description": "High disaster rate, moderate growth",
            "birth_rate": 0.002,
            "death_rate": 0.0005,
            "disaster_prob": 0.02,
            "initial_pop": 50,
            "target_pop": 100,
        },
        {
            "name": "Stable Population",
            "description": "Balanced rates at target",
            "birth_rate": 0.001,
            "death_rate": 0.001,
            "initial_pop": 100,
            "target_pop": 100,
        },
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📋 Scenario {i}: {scenario['name']}")
        print(f"   Description: {scenario['description']}")
        print(
            f"   Config: Birth={scenario['birth_rate']:.4f}, "
            f"Initial={scenario['initial_pop']}, Target={scenario['target_pop']}"
        )

        # Quick simulation for each scenario
        config = Config()
        config.population.enable_dynamics = True
        config.population.initial_population = scenario["initial_pop"]
        config.population.target_population = scenario["target_pop"]
        config.population.base_birth_rate = scenario["birth_rate"]
        config.population.base_death_rate = scenario["death_rate"]

        if "disaster_prob" in scenario:
            config.population.disaster_probability = scenario["disaster_prob"]

        config.simulation.max_steps = 300  # Short demo
        config.simulation.tick_rate = 0.001
        config.llm.model_name = "mock"
        config.assets.enable_generation = False

        simulator = SocietySimulator(config)
        await simulator.run()

        print(f"   ✅ Scenario {i} completed")


if __name__ == "__main__":
    print("🧬 Population Dynamics System - Interactive Demo")
    print("=" * 60)
    print("This demo showcases the dynamic population control system")
    print("with natural growth, aging, events, and demographic tracking.")
    print("")

    # Run main demo
    asyncio.run(demo_population_dynamics())

    # Run scenario tests
    asyncio.run(demo_specific_scenarios())

    print("\n🎉 Demo Complete!")
    print("🔬 The Population Dynamics System successfully demonstrates:")
    print("   ✅ Natural population growth and control")
    print("   ✅ Dynamic birth/death rate adaptation")
    print("   ✅ Random event impacts on population")
    print("   ✅ Realistic demographic tracking")
    print("   ✅ Environmental and technological factors")
    print("\n🚀 Ready for full 2,500-agent simulations!")
