"""
Performance Optimizer for Society Simulator
Implements parallel processing and performance optimizations for large-scale simulations
"""

import asyncio
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil

from intelligent_agent import IntelligentAgent
from intelligent_world import IntelligentWorld
from llm_integration import LLMManager, LLMProvider
from society_demo import (
    AgentState,
    AgentType,
    CulturalGroup,
    SocietyAgent,
    SocietyWorld,
)


@dataclass
class PerformanceMetrics:
    """Performance tracking metrics"""

    agents_processed: int = 0
    total_time: float = 0.0
    step_times: List[float] = None
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    agents_per_second: float = 0.0

    def __post_init__(self):
        if self.step_times is None:
            self.step_times = []


class ParallelSocietyWorld:
    """Optimized society world with parallel processing"""

    def __init__(
        self,
        num_agents: int = 100,
        world_size: Tuple[float, float] = (100, 100),
        use_intelligent_agents: bool = False,
        llm_provider: LLMProvider = LLMProvider.NONE,
        num_workers: Optional[int] = None,
    ):
        self.world_size = world_size
        self.step_count = 0
        self.use_intelligent_agents = use_intelligent_agents
        self.llm_provider = llm_provider

        # Performance optimization settings
        self.num_workers = num_workers or min(8, mp.cpu_count())
        self.batch_size = max(10, num_agents // self.num_workers)

        # Performance tracking
        self.metrics = PerformanceMetrics()

        # Create agents
        self.agents = []
        self._create_optimized_agents(num_agents)

        # Agent spatial indexing for fast neighbor queries
        self.spatial_grid = {}
        self.grid_size = 10.0  # Grid cell size for spatial partitioning

        # Pre-allocated arrays for performance
        self._agent_positions = np.zeros((num_agents, 2))
        self._agent_energies = np.zeros(num_agents)
        self._agent_happiness = np.zeros(num_agents)

        # World state caching
        self._cached_statistics = {}
        self._stats_cache_step = -1

        print("🚀 Optimized world created:")
        print(f"   Agents: {len(self.agents)}")
        print(f"   Workers: {self.num_workers}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Intelligence: {'LLM' if use_intelligent_agents else 'Rule-based'}")

    def _create_optimized_agents(self, num_agents: int):
        """Create agents with optimized initialization"""
        if self.use_intelligent_agents:
            llm_manager = (
                LLMManager(self.llm_provider)
                if self.llm_provider != LLMProvider.NONE
                else None
            )
            for i in range(num_agents):
                agent = IntelligentAgent(f"agent_{i}", self.world_size, llm_manager)
                self.agents.append(agent)
        else:
            for i in range(num_agents):
                agent = SocietyAgent(f"agent_{i}", self.world_size)
                self.agents.append(agent)

    def _update_spatial_index(self):
        """Update spatial grid for fast neighbor queries"""
        self.spatial_grid.clear()

        for agent in self.agents:
            grid_x = int(agent.position.x // self.grid_size)
            grid_y = int(agent.position.y // self.grid_size)
            grid_key = (grid_x, grid_y)

            if grid_key not in self.spatial_grid:
                self.spatial_grid[grid_key] = []
            self.spatial_grid[grid_key].append(agent)

    def get_nearby_agents(self, agent, radius: float = 20.0) -> List:
        """Fast neighbor query using spatial indexing"""
        grid_x = int(agent.position.x // self.grid_size)
        grid_y = int(agent.position.y // self.grid_size)

        nearby = []
        search_radius = int(np.ceil(radius / self.grid_size))

        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                grid_key = (grid_x + dx, grid_y + dy)
                if grid_key in self.spatial_grid:
                    for other in self.spatial_grid[grid_key]:
                        if other.agent_id != agent.agent_id:
                            distance = agent.position.distance_to(other.position)
                            if distance <= radius:
                                nearby.append(other)

        return nearby

    def _update_agent_arrays(self):
        """Update pre-allocated arrays for vectorized operations"""
        for i, agent in enumerate(self.agents):
            self._agent_positions[i] = [agent.position.x, agent.position.y]
            self._agent_energies[i] = agent.energy
            self._agent_happiness[i] = agent.happiness

    async def step_parallel(self):
        """Parallel agent stepping with optimizations"""
        start_time = time.time()
        self.step_count += 1

        # Update spatial indexing
        self._update_spatial_index()

        # Update agent arrays for vectorized operations
        self._update_agent_arrays()

        if self.use_intelligent_agents:
            # Async batch processing for intelligent agents
            await self._step_intelligent_agents_parallel()
        else:
            # Thread-based parallel processing for rule-based agents
            await self._step_basic_agents_parallel()

        # Update performance metrics
        step_time = time.time() - start_time
        self.metrics.step_times.append(step_time)
        self.metrics.total_time += step_time
        self.metrics.agents_processed += len(self.agents)

        # Memory and CPU tracking
        process = psutil.Process()
        self.metrics.memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        self.metrics.cpu_usage = process.cpu_percent()

        # Clear statistics cache
        self._stats_cache_step = -1

    async def _step_intelligent_agents_parallel(self):
        """Parallel processing for intelligent agents"""
        # Process in optimized batches
        batch_size = min(self.batch_size, 15)  # Smaller batches for LLM agents

        for i in range(0, len(self.agents), batch_size):
            batch = self.agents[i : i + batch_size]

            # Process batch asynchronously
            tasks = []
            for agent in batch:
                # Add nearby agents to agent context for faster access
                agent._cached_nearby = self.get_nearby_agents(agent)
                tasks.append(agent.step(self))

            await asyncio.gather(*tasks, return_exceptions=True)

    async def _step_basic_agents_parallel(self):
        """Parallel processing for basic agents using threads"""

        def step_agent_batch(agent_batch):
            """Step a batch of agents in a thread"""
            for agent in agent_batch:
                # Use cached nearby agents
                agent._cached_nearby = self.get_nearby_agents(agent)
                agent.step(self)

        # Create batches
        batches = []
        for i in range(0, len(self.agents), self.batch_size):
            batches.append(self.agents[i : i + self.batch_size])

        # Process batches in parallel using threads
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(executor, step_agent_batch, batch)
                for batch in batches
            ]
            await asyncio.gather(*tasks)

    def get_statistics(self) -> Dict[str, Any]:
        """Get optimized statistics with caching"""
        if self._stats_cache_step == self.step_count:
            return self._cached_statistics

        # Vectorized statistics calculation
        total_resources = {
            "food": float(np.sum([agent.resources["food"] for agent in self.agents])),
            "currency": float(
                np.sum([agent.resources["currency"] for agent in self.agents])
            ),
            "materials": float(
                np.sum([agent.resources["materials"] for agent in self.agents])
            ),
            "tools": float(np.sum([agent.resources["tools"] for agent in self.agents])),
        }

        # Fast agent type counting
        agent_types = {}
        cultural_groups = {}

        for agent in self.agents:
            agent_type = agent.agent_type.value
            cultural_group = agent.cultural_group.value

            agent_types[agent_type] = agent_types.get(agent_type, 0) + 1
            cultural_groups[cultural_group] = cultural_groups.get(cultural_group, 0) + 1

        # Connection counting (optimized)
        total_connections = 0
        if hasattr(self.agents[0], "relationships"):
            # Intelligent agents
            for agent in self.agents:
                total_connections += len(agent.relationships)
        else:
            # Basic agents
            for agent in self.agents:
                total_connections += len(getattr(agent, "social_connections", []))

        stats = {
            "step": self.step_count,
            "agents": len(self.agents),
            "total_resources": total_resources,
            "total_connections": total_connections,
            "agent_types": agent_types,
            "cultural_groups": cultural_groups,
            "averages": {
                "energy": float(np.mean(self._agent_energies)),
                "happiness": float(np.mean(self._agent_happiness)),
            },
            "performance": {
                "agents_per_second": len(self.agents)
                / (self.metrics.step_times[-1] if self.metrics.step_times else 1),
                "memory_mb": self.metrics.memory_usage,
                "cpu_percent": self.metrics.cpu_usage,
                "workers": self.num_workers,
            },
        }

        # Cache results
        self._cached_statistics = stats
        self._stats_cache_step = self.step_count

        return stats

    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report"""
        if not self.metrics.step_times:
            return {"error": "No performance data available"}

        step_times = np.array(self.metrics.step_times)

        return {
            "total_steps": len(step_times),
            "total_agents": len(self.agents),
            "total_time": self.metrics.total_time,
            "average_sps": len(self.agents) / np.mean(step_times),
            "peak_sps": len(self.agents) / np.min(step_times),
            "step_time_stats": {
                "mean": float(np.mean(step_times)),
                "min": float(np.min(step_times)),
                "max": float(np.max(step_times)),
                "std": float(np.std(step_times)),
            },
            "memory_usage_mb": self.metrics.memory_usage,
            "cpu_usage_percent": self.metrics.cpu_usage,
            "optimization_settings": {
                "workers": self.num_workers,
                "batch_size": self.batch_size,
                "spatial_grid_size": self.grid_size,
                "use_intelligent_agents": self.use_intelligent_agents,
            },
        }


async def run_optimized_simulation(
    num_agents: int = 100,
    steps: int = 200,
    use_intelligent_agents: bool = False,
    llm_provider: LLMProvider = LLMProvider.NONE,
    num_workers: Optional[int] = None,
    show_progress: bool = True,
) -> ParallelSocietyWorld:
    """Run optimized parallel simulation"""

    print("🚀 Optimized Society Simulation")
    print(f"   Agents: {num_agents}")
    print(f"   Steps: {steps}")
    print(f"   Intelligence: {'LLM' if use_intelligent_agents else 'Rule-based'}")
    print(f"   Workers: {num_workers or 'auto'}")
    print("=" * 50)

    world = ParallelSocietyWorld(
        num_agents=num_agents,
        use_intelligent_agents=use_intelligent_agents,
        llm_provider=llm_provider,
        num_workers=num_workers,
    )

    start_time = time.time()

    for step in range(steps):
        await world.step_parallel()

        if show_progress and step % (steps // 5) == 0:
            stats = world.get_statistics()
            perf = stats["performance"]
            print(
                f"Step {step:3d}: "
                f"SPS: {perf['agents_per_second']:.1f}, "
                f"Energy: {stats['averages']['energy']:.2f}, "
                f"Memory: {perf['memory_mb']:.1f}MB"
            )

    elapsed = time.time() - start_time
    avg_sps = (num_agents * steps) / elapsed

    print(f"\n✅ Completed in {elapsed:.2f}s")
    print(f"   Average SPS: {avg_sps:.1f}")
    print(f"   Peak SPS: {world.get_performance_report()['peak_sps']:.1f}")

    return world


# Performance testing utilities
async def benchmark_optimization():
    """Benchmark different optimization settings"""
    print("🔬 Optimization Benchmark")
    print("=" * 60)

    test_configs = [
        (100, False, 1, "100 agents, 1 worker"),
        (100, False, 2, "100 agents, 2 workers"),
        (100, False, 4, "100 agents, 4 workers"),
        (200, False, 4, "200 agents, 4 workers"),
        (500, False, 8, "500 agents, 8 workers"),
    ]

    results = []

    for agents, intelligent, workers, label in test_configs:
        print(f"\n📊 {label}")

        world = await run_optimized_simulation(
            num_agents=agents,
            steps=50,
            use_intelligent_agents=intelligent,
            num_workers=workers,
            show_progress=False,
        )

        perf = world.get_performance_report()
        results.append(
            {
                "config": label,
                "agents": agents,
                "workers": workers,
                "avg_sps": perf["average_sps"],
                "peak_sps": perf["peak_sps"],
                "memory_mb": perf["memory_usage_mb"],
            }
        )

        print(f"   Average SPS: {perf['average_sps']:.1f}")
        print(f"   Peak SPS: {perf['peak_sps']:.1f}")
        print(f"   Memory: {perf['memory_usage_mb']:.1f}MB")

    print("\n📈 Optimization Results:")
    print(
        f"{'Config':<25} {'Agents':<7} {'Workers':<8} {'Avg SPS':<8} {'Peak SPS':<9} {'Memory':<8}"
    )
    print("-" * 75)
    for r in results:
        print(
            f"{r['config']:<25} {r['agents']:<7} {r['workers']:<8} {r['avg_sps']:<8.1f} {r['peak_sps']:<9.1f} {r['memory_mb']:<8.1f}"
        )


if __name__ == "__main__":
    # Test optimized simulation
    asyncio.run(benchmark_optimization())
