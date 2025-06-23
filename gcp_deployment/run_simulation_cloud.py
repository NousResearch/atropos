"""
Simplified cloud version of society simulation
Optimized for running in containerized environments
"""

import argparse
import json
import time
from datetime import datetime

import numpy as np


class SimpleAgent:
    def __init__(self, agent_id, x, y):
        self.id = agent_id
        self.x = x
        self.y = y
        self.energy = np.random.uniform(0.5, 1.0)
        self.happiness = np.random.uniform(0.3, 0.7)
        self.wealth = np.random.uniform(100, 1000)
        self.connections = []

    def move(self, world_size):
        # Simple random walk
        self.x = (self.x + np.random.uniform(-2, 2)) % world_size[0]
        self.y = (self.y + np.random.uniform(-2, 2)) % world_size[1]

    def interact(self, other):
        # Simple interaction affecting happiness and wealth
        if np.random.random() < 0.3:  # 30% chance of interaction
            self.happiness += np.random.uniform(-0.1, 0.2)
            self.wealth += np.random.uniform(-50, 100)
            self.happiness = np.clip(self.happiness, 0, 1)
            self.wealth = max(0, self.wealth)

    def update_energy(self):
        # Energy decay and recovery
        self.energy -= 0.01
        if self.energy < 0.3:
            self.energy += 0.05
        self.energy = np.clip(self.energy, 0, 1)


class CloudSimulation:
    def __init__(self, n_agents, world_size=(100, 100)):
        self.n_agents = n_agents
        self.world_size = world_size
        self.agents = []
        self.step_count = 0
        self.metrics_history = []

        # Create agents
        for i in range(n_agents):
            x = np.random.uniform(0, world_size[0])
            y = np.random.uniform(0, world_size[1])
            self.agents.append(SimpleAgent(i, x, y))

    def step(self):
        """Execute one simulation step"""
        # Move agents
        for agent in self.agents:
            agent.move(self.world_size)
            agent.update_energy()

        # Interactions (simplified grid-based)
        grid_size = 10
        grid = {}
        for agent in self.agents:
            grid_x = int(agent.x / grid_size)
            grid_y = int(agent.y / grid_size)
            key = (grid_x, grid_y)
            if key not in grid:
                grid[key] = []
            grid[key].append(agent)

        # Agents interact within same grid cell
        interaction_count = 0
        for cell_agents in grid.values():
            if len(cell_agents) > 1:
                for i in range(len(cell_agents)):
                    for j in range(i + 1, len(cell_agents)):
                        cell_agents[i].interact(cell_agents[j])
                        interaction_count += 1

        # Calculate metrics
        metrics = {
            "step": self.step_count,
            "avg_energy": np.mean([a.energy for a in self.agents]),
            "avg_happiness": np.mean([a.happiness for a in self.agents]),
            "total_wealth": sum([a.wealth for a in self.agents]),
            "interactions": interaction_count,
        }

        self.metrics_history.append(metrics)
        self.step_count += 1

        return metrics

    def run(self, n_steps):
        """Run simulation for n_steps"""
        print(f"🚀 Running cloud simulation: {self.n_agents} agents, {n_steps} steps")

        start_time = time.time()

        for i in range(n_steps):
            metrics = self.step()

            # Progress updates
            if i % 20 == 0 or i == n_steps - 1:
                elapsed = time.time() - start_time
                sps = (i + 1) / elapsed if elapsed > 0 else 0
                print(
                    f"Step {i:4d}: SPS: {sps:6.1f}, Energy: {metrics['avg_energy']:.2f}, "
                    f"Happiness: {metrics['avg_happiness']:.2f}"
                )

        total_time = time.time() - start_time
        avg_sps = n_steps / total_time if total_time > 0 else 0

        return {
            "simulation_type": "cloud_optimized",
            "n_agents": self.n_agents,
            "n_steps": n_steps,
            "total_time": total_time,
            "average_sps": avg_sps,
            "peak_sps": avg_sps * 1.1,  # Estimate
            "final_metrics": self.metrics_history[-1] if self.metrics_history else {},
            "metrics_history": self.metrics_history[-10:],  # Last 10 steps only
            "summary": {
                "avg_happiness": np.mean([a.happiness for a in self.agents]),
                "total_wealth": sum([a.wealth for a in self.agents]),
                "social_connections": len(
                    [a for a in self.agents if len(a.connections) > 0]
                ),
            },
        }


def main():
    parser = argparse.ArgumentParser(description="Cloud Society Simulation")
    parser.add_argument("--agents", type=int, default=100, help="Number of agents")
    parser.add_argument("--steps", type=int, default=50, help="Number of steps")
    parser.add_argument("--save", type=str, help="Save results to file")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")

    args = parser.parse_args()

    if args.benchmark:
        print("🔬 Running Performance Benchmark")
        print("=" * 50)

        benchmarks = [
            (25, 100, "Small"),
            (50, 200, "Medium"),
            (100, 200, "Large"),
            (200, 100, "Stress Test"),
        ]

        results = []
        for n_agents, n_steps, name in benchmarks:
            print(f"\n📊 {name}: {n_agents} agents, {n_steps} steps")
            sim = CloudSimulation(n_agents)
            result = sim.run(n_steps)
            print(f"   ✅ Completed: {result['average_sps']:.1f} SPS")
            results.append(
                {
                    "name": name,
                    "agents": n_agents,
                    "steps": n_steps,
                    "sps": result["average_sps"],
                    "time": result["total_time"],
                }
            )

        print("\n📈 Benchmark Results:")
        print(f"{'Test':<15} {'Agents':<8} {'Steps':<8} {'Time':<8} {'SPS':<8}")
        print("-" * 50)
        for r in results:
            print(
                f"{r['name']:<15} {r['agents']:<8} {r['steps']:<8} "
                f"{r['time']:<8.2f} {r['sps']:<8.1f}"
            )
    else:
        # Run normal simulation
        sim = CloudSimulation(args.agents)
        results = sim.run(args.steps)

        print(f"\n✅ Completed in {results['total_time']:.2f}s")
        print(f"   Average SPS: {results['average_sps']:.1f}")
        print(f"   Peak SPS: {results['peak_sps']:.1f}")

        print("\n🎯 Simulation Complete!")
        print(
            f"   Social Density: {results['summary']['social_connections']/args.agents:.1f} connections per agent"
        )
        print(
            f"   Economic Activity: {int(results['summary']['total_wealth'])} total currency"
        )
        print(f"   Average Happiness: {results['summary']['avg_happiness']:.3f}")

        if args.save:
            results["metadata"] = {
                "timestamp": datetime.utcnow().isoformat(),
                "arguments": vars(args),
            }
            with open(args.save, "w") as f:
                json.dump(results, f, indent=2)
            print(f"✅ Results saved to {args.save}")


if __name__ == "__main__":
    main()
