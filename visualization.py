"""
Basic 2D Visualization for Society Simulator
Real-time agent positions, states, and statistics
"""

import time
from collections import deque
from typing import Any, Dict, List, Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


class SocietyVisualizer:
    """Real-time visualization of society simulation"""

    def __init__(self, world_size: tuple = (100, 100), update_interval: int = 100):
        self.world_size = world_size
        self.update_interval = update_interval

        # Setup matplotlib
        plt.ion()  # Interactive mode
        self.fig, (self.ax_main, self.ax_stats) = plt.subplots(1, 2, figsize=(15, 7))

        # Main plot setup
        self.ax_main.set_xlim(0, world_size[0])
        self.ax_main.set_ylim(0, world_size[1])
        self.ax_main.set_title("Society Simulation - Agent Positions")
        self.ax_main.set_xlabel("X Position")
        self.ax_main.set_ylabel("Y Position")
        self.ax_main.grid(True, alpha=0.3)

        # Statistics plot setup
        self.ax_stats.set_title("Live Statistics")

        # Color maps for different agent attributes
        self.agent_type_colors = {
            "farmer": "green",
            "craftsman": "brown",
            "trader": "gold",
            "scholar": "blue",
            "leader": "red",
            "unemployed": "gray",
        }

        self.cultural_group_colors = {
            "harmonists": "lightblue",
            "builders": "orange",
            "guardians": "purple",
            "scholars": "darkblue",
            "wanderers": "lime",
        }

        self.state_markers = {
            "idle": "o",
            "moving": "^",
            "socializing": "s",
            "working": "D",
            "trading": "*",
        }

        # Statistics tracking
        self.stats_history = {
            "steps": deque(maxlen=100),
            "avg_energy": deque(maxlen=100),
            "avg_happiness": deque(maxlen=100),
            "total_connections": deque(maxlen=100),
            "total_currency": deque(maxlen=100),
        }

        # Legends
        self._create_legends()

    def _create_legends(self):
        """Create legends for agent types and states"""
        # Agent type legend
        type_legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=8,
                label=agent_type,
            )
            for agent_type, color in self.agent_type_colors.items()
        ]

        self.ax_main.legend(
            handles=type_legend_elements,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.0),
            title="Agent Types",
        )

    def update(self, world, step_count: int):
        """Update visualization with current world state"""
        # Clear previous plots
        self.ax_main.clear()
        self.ax_stats.clear()

        # Reset main plot
        self.ax_main.set_xlim(0, self.world_size[0])
        self.ax_main.set_ylim(0, self.world_size[1])
        self.ax_main.set_title(f"Society Simulation - Step {step_count}")
        self.ax_main.set_xlabel("X Position")
        self.ax_main.set_ylabel("Y Position")
        self.ax_main.grid(True, alpha=0.3)

        # Plot agents
        self._plot_agents(world)

        # Plot statistics
        self._plot_statistics(world, step_count)

        # Update display
        plt.tight_layout()
        plt.pause(0.01)

    def _plot_agents(self, world):
        """Plot agent positions with colors and markers"""
        # Group agents by type for efficient plotting
        agent_groups = {}

        for agent in world.agents:
            agent_type = agent.agent_type.value
            if agent_type not in agent_groups:
                agent_groups[agent_type] = {
                    "x": [],
                    "y": [],
                    "states": [],
                    "energy": [],
                    "happiness": [],
                }

            agent_groups[agent_type]["x"].append(agent.position.x)
            agent_groups[agent_type]["y"].append(agent.position.y)
            agent_groups[agent_type]["states"].append(agent.state.value)
            agent_groups[agent_type]["energy"].append(agent.energy)
            agent_groups[agent_type]["happiness"].append(agent.happiness)

        # Plot each agent type
        for agent_type, data in agent_groups.items():
            color = self.agent_type_colors.get(agent_type, "black")

            # Plot with size based on energy and alpha based on happiness
            sizes = [max(20, e * 100) for e in data["energy"]]  # Size based on energy
            alphas = [
                max(0.3, h) for h in data["happiness"]
            ]  # Transparency based on happiness

            scatter = self.ax_main.scatter(
                data["x"],
                data["y"],
                c=color,
                s=sizes,
                alpha=0.7,
                label=f"{agent_type} ({len(data['x'])})",
                edgecolors="black",
                linewidth=0.5,
            )

        # Plot social connections as lines
        self._plot_social_connections(world)

        # Recreate legend
        self.ax_main.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0))

    def _plot_social_connections(self, world):
        """Plot social connections between agents"""
        # Sample a subset of connections to avoid clutter
        connection_count = 0
        max_connections = 50  # Limit to avoid visual clutter

        for agent in world.agents:
            if connection_count >= max_connections:
                break

            for connected_id, strength in agent.social_connections.items():
                if strength > 0.5:  # Only show strong connections
                    # Find the connected agent
                    connected_agent = next(
                        (a for a in world.agents if a.agent_id == connected_id), None
                    )
                    if connected_agent:
                        # Draw line with alpha based on connection strength
                        self.ax_main.plot(
                            [agent.position.x, connected_agent.position.x],
                            [agent.position.y, connected_agent.position.y],
                            "gray",
                            alpha=strength * 0.5,
                            linewidth=0.5,
                        )
                        connection_count += 1

    def _plot_statistics(self, world, step_count: int):
        """Plot real-time statistics"""
        stats = world.get_statistics()

        # Update history
        self.stats_history["steps"].append(step_count)
        self.stats_history["avg_energy"].append(stats["avg_energy"])
        self.stats_history["avg_happiness"].append(stats["avg_happiness"])
        self.stats_history["total_connections"].append(stats["total_connections"])
        self.stats_history["total_currency"].append(
            stats["total_resources"]["currency"]
        )

        # Plot statistics over time
        steps = list(self.stats_history["steps"])

        if len(steps) > 1:
            # Create subplots for different metrics
            self.ax_stats.clear()

            # Plot multiple metrics
            ax2 = self.ax_stats.twinx()

            # Energy and happiness (0-1 scale)
            self.ax_stats.plot(
                steps,
                list(self.stats_history["avg_energy"]),
                "b-",
                label="Avg Energy",
                alpha=0.8,
            )
            self.ax_stats.plot(
                steps,
                list(self.stats_history["avg_happiness"]),
                "g-",
                label="Avg Happiness",
                alpha=0.8,
            )

            # Social connections (different scale)
            ax2.plot(
                steps,
                list(self.stats_history["total_connections"]),
                "r-",
                label="Social Connections",
                alpha=0.8,
            )

            self.ax_stats.set_xlabel("Simulation Step")
            self.ax_stats.set_ylabel("Energy / Happiness", color="b")
            ax2.set_ylabel("Social Connections", color="r")

            self.ax_stats.legend(loc="upper left")
            ax2.legend(loc="upper right")

            self.ax_stats.set_title("Live Statistics")
            self.ax_stats.grid(True, alpha=0.3)

        # Add text statistics
        stats_text = """Step: {step_count}
Agents: {len(world.agents)}
Avg Energy: {stats['avg_energy']:.2f}
Avg Happiness: {stats['avg_happiness']:.2f}
Social Connections: {stats['total_connections']}
Total Currency: {stats['total_resources']['currency']:,}

Agent Types:
"""
        for agent_type, count in stats["agent_types"].items():
            stats_text += f"  {agent_type}: {count}\n"

        self.ax_stats.text(
            0.02,
            0.02,
            stats_text,
            transform=self.ax_stats.transAxes,
            verticalalignment="bottom",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    def close(self):
        """Close the visualization"""
        plt.close(self.fig)


def run_simulation_with_visualization(num_agents: int = 50, steps: int = 500):
    """Run simulation with real-time visualization"""
    from society_demo import SocietyWorld

    print("🎬 Starting Visualized Simulation")
    print(f"   Agents: {num_agents}")
    print(f"   Steps: {steps}")
    print("   Close the plot window to stop simulation")

    # Create world and visualizer
    world = SocietyWorld(num_agents)
    viz = SocietyVisualizer(world.world_size)

    try:
        # Initial display
        viz.update(world, 0)

        # Run simulation with visualization
        for step in range(steps):
            world.step()

            # Update visualization every 5 steps for performance
            if step % 5 == 0:
                viz.update(world, step)

                # Check if window was closed
                if not plt.get_fignums():
                    print("🛑 Visualization window closed. Stopping simulation.")
                    break

        # Keep final visualization open
        if plt.get_fignums():
            print("\n✅ Simulation complete! Close the plot window to exit.")
            plt.show(block=True)

    except KeyboardInterrupt:
        print("\n🛑 Simulation interrupted by user")
    finally:
        viz.close()

    return world


if __name__ == "__main__":
    # Test visualization
    world = run_simulation_with_visualization(25, 200)
