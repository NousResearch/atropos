#!/usr/bin/env python3
"""
Society Simulator - Main Entry Point
Clean, simple interface for running society simulations
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Import intelligent simulation
from intelligent_world import IntelligentWorld, run_intelligent_simulation
from llm_integration import LLMProvider

# Import performance optimization
from performance_optimizer import run_optimized_simulation

# Import our working simulation
from society_demo import SocietyWorld, run_society_simulation


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Society Simulator - Multi-agent LLM-driven society simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_simulation.py --agents 50 --steps 500
  python run_simulation.py --agents 100 --llm openai --visualize
  python run_simulation.py --config experiment.json --save results.json
  python run_simulation.py --load previous.json --continue 1000
        """,
    )

    # Basic simulation parameters
    parser.add_argument(
        "--agents", type=int, default=50, help="Number of agents (default: 50)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Number of simulation steps (default: 200)",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        nargs=2,
        default=[100, 100],
        metavar=("WIDTH", "HEIGHT"),
        help="World dimensions (default: 100 100)",
    )

    # LLM configuration
    parser.add_argument(
        "--llm",
        choices=["none", "openai", "anthropic", "mock"],
        default="none",
        help="LLM provider for agent intelligence (default: none)",
    )
    parser.add_argument(
        "--model", type=str, help="Specific model to use (e.g., gpt-3.5-turbo)"
    )
    parser.add_argument("--api-key", type=str, help="API key for LLM provider")

    # Visualization and output
    parser.add_argument(
        "--visualize", action="store_true", help="Show real-time visualization"
    )
    parser.add_argument("--save", type=str, help="Save results to file")
    parser.add_argument("--load", type=str, help="Load simulation from file")
    parser.add_argument(
        "--continue",
        dest="continue_steps",
        type=int,
        help="Continue loaded simulation for N steps",
    )

    # Configuration
    parser.add_argument("--config", type=str, help="Load configuration from JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")

    # Performance and testing
    parser.add_argument(
        "--benchmark", action="store_true", help="Run performance benchmark"
    )
    parser.add_argument(
        "--test", action="store_true", help="Run quick functionality test"
    )
    parser.add_argument(
        "--optimized", action="store_true", help="Use optimized parallel processing"
    )
    parser.add_argument(
        "--workers", type=int, help="Number of worker processes for parallel execution"
    )

    return parser


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: Configuration file "{config_path}' not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {e}")
        sys.exit(1)


def save_results(world, filepath: str, args: argparse.Namespace):
    """Save simulation results to file"""
    # Handle both intelligent and basic world types
    if hasattr(world, "get_agent_insights"):
        # Intelligent world
        results = {
            "metadata": {
                "agents": len(world.agents),
                "steps": world.step_count,
                "world_size": world.world_size,
                "timestamp": time.time(),
                "arguments": vars(args),
                "simulation_type": "intelligent",
            },
            "statistics": world.get_statistics(),
            "insights": world.get_agent_insights(),
            "agents": [agent.get_status() for agent in world.agents],
        }
    else:
        # Basic world
        results = {
            "metadata": {
                "agents": len(world.agents),
                "steps": world.step_count,
                "world_size": world.world_size,
                "timestamp": time.time(),
                "arguments": vars(args),
                "simulation_type": "basic",
            },
            "statistics": world.get_statistics(),
            "agents": [agent.get_status() for agent in world.agents],
        }

    try:
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✅ Results saved to {filepath}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")


def load_simulation(filepath: str) -> Optional[Dict[str, Any]]:
    """Load simulation from file"""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: Simulation file "{filepath}' not found")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in simulation file: {e}")
        return None


def run_benchmark():
    """Run performance benchmark"""
    print("🔬 Running Performance Benchmark")
    print("=" * 50)

    test_configs = [
        (25, 100, "Small"),
        (50, 200, "Medium"),
        (100, 200, "Large"),
        (200, 100, "Stress Test"),
    ]

    results = []

    for agents, steps, label in test_configs:
        print(f"\n📊 {label}: {agents} agents, {steps} steps")

        start_time = time.time()
        world = SocietyWorld(agents)

        for step in range(steps):
            world.step()
            if step % 50 == 0 and step > 0:
                elapsed = time.time() - start_time
                sps = step / elapsed
                print(f"   Step {step}: {sps:.1f} SPS")

        elapsed = time.time() - start_time
        sps = steps / elapsed

        results.append(
            {
                "agents": agents,
                "steps": steps,
                "time": elapsed,
                "sps": sps,
                "label": label,
            }
        )

        print(f"   ✅ Completed: {sps:.1f} SPS")

    print("\n📈 Benchmark Results:")
    print(f"{'Test':<12} {'Agents':<7} {'Steps':<6} {'Time':<8} {'SPS':<8}")
    print("-" * 50)
    for r in results:
        print(
            f"{r['label']:<12} {r['agents']:<7} {r['steps']:<6} {r['time']:<8.2f} {r['sps']:<8.1f}"
        )


def run_test():
    """Run quick functionality test"""
    print("🧪 Running Functionality Test")
    print("=" * 40)

    # Test 1: Basic simulation
    print("Test 1: Basic simulation (10 agents, 50 steps)")
    world = SocietyWorld(10)
    for i in range(50):
        world.step()
    stats = world.get_statistics()
    print(f"✅ Basic simulation: {stats['total_connections']} connections formed")

    # Test 2: Agent behaviors
    print("\nTest 2: Agent behavior verification")
    behaviors = set()
    for agent in world.agents:
        behaviors.add(agent.state.value)
    print(f"✅ Agent behaviors: {', '.join(behaviors)}")

    # Test 3: Economic activity
    print("\nTest 3: Economic system")
    total_currency = sum(agent.resources["currency"] for agent in world.agents)
    total_trades = sum(1 for agent in world.agents if len(agent.memories) > 0)
    print(
        f"✅ Economic activity: {total_currency} total currency, {total_trades} active traders"
    )

    # Test 4: Cultural dynamics
    print("\nTest 4: Cultural dynamics")
    cultural_groups = set(agent.cultural_group.value for agent in world.agents)
    print(f"✅ Cultural diversity: {len(cultural_groups)} active groups")

    print("\n🎉 All tests passed!")


def main():
    """Main function"""
    parser = create_parser()
    args = parser.parse_args()

    # Handle special commands
    if args.benchmark:
        run_benchmark()
        return

    if args.test:
        run_test()
        return

    # Load configuration if specified
    config = {}
    if args.config:
        config = load_config(args.config)

        # Override with command line arguments (only if not explicitly set)
        if "simulation" in config:
            sim_config = config["simulation"]
            if args.agents == 50:  # Default value
                args.agents = sim_config.get("agents", args.agents)
            if args.steps == 200:  # Default value
                args.steps = sim_config.get("steps", args.steps)
            if args.world_size == [100, 100]:  # Default value
                args.world_size = sim_config.get("world_size", args.world_size)

    # Set output level
    if args.quiet:
        # Minimal output mode - could redirect stdout
        pass
    elif args.verbose:
        print("🔧 Configuration:")
        print(f"   Agents: {args.agents}")
        print(f"   Steps: {args.steps}")
        print(f"   World Size: {args.world_size}")
        print(f"   LLM: {args.llm}")
        if args.visualize:
            print("   Visualization: Enabled")

    # Handle loading existing simulation
    if args.load:
        sim_data = load_simulation(args.load)
        if not sim_data:
            return

        print(
            f"📁 Loaded simulation: {sim_data['metadata']['agents']} agents, {sim_data['metadata']['steps']} steps"
        )

        if args.continue_steps:
            print(f"🔄 Continuing for {args.continue_steps} additional steps...")
            # TODO: Implement continuation logic
            print("⚠️  Continuation not yet implemented - running new simulation")

    # Check for LLM configuration
    llm_provider = LLMProvider.NONE
    if args.llm != "none":
        if args.llm == "openai":
            llm_provider = LLMProvider.OPENAI
        elif args.llm == "anthropic":
            llm_provider = LLMProvider.ANTHROPIC
        elif args.llm == "mock":
            llm_provider = LLMProvider.MOCK

        if args.llm in ["openai", "anthropic"] and not args.api_key:
            print(f"⚠️  Warning: {args.llm} selected but no API key provided")
            print("   Set --api-key or use environment variable")
            print("   Falling back to rule-based agents")
            llm_provider = LLMProvider.NONE

    # Run the simulation
    print("🚀 Starting Society Simulation")
    print(f"   Agents: {args.agents}")
    print(f"   Steps: {args.steps}")
    print(f"   Intelligence: {llm_provider.value}")

    if args.visualize:
        print("📊 Visualization mode enabled")
        # TODO: Implement visualization
        print("⚠️  Visualization not yet implemented")

    # Choose simulation type based on configuration
    if args.optimized:
        print("⚡ Using Optimized Parallel Processing")
        use_intelligent = llm_provider != LLMProvider.NONE
        if use_intelligent:
            print("🧠 + Intelligent Agents with LLM integration")
        else:
            print("🤖 + Rule-based Agents")

        world = asyncio.run(
            run_optimized_simulation(
                num_agents=args.agents,
                steps=args.steps,
                use_intelligent_agents=use_intelligent,
                llm_provider=llm_provider,
                num_workers=args.workers,
                show_progress=not args.quiet,
            )
        )
    elif llm_provider != LLMProvider.NONE:
        print("🧠 Using Intelligent Agents with LLM integration")
        # Run intelligent simulation
        world = asyncio.run(
            run_intelligent_simulation(
                args.agents, args.steps, llm_provider, args.api_key, args.model
            )
        )
    else:
        print("🤖 Using Rule-based Agents")
        # Run basic simulation
        world = run_society_simulation(args.agents, args.steps)

    # Save results if requested
    if args.save:
        save_results(world, args.save, args)

    # Summary
    final_stats = world.get_statistics()
    print("\n🎯 Simulation Complete!")

    if hasattr(world, "get_agent_insights"):
        # Intelligent world summary
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
    elif hasattr(world, "get_performance_report"):
        # Optimized world summary
        perf = world.get_performance_report()
        social_connections = final_stats.get("total_connections", 0)
        print(
            f"   Social Density: {social_connections / args.agents:.1f} connections per agent"
        )
        print(
            f"   Economic Activity: {final_stats['total_resources']['currency']:.0f} total currency"
        )
        print(
            f"   Cultural Diversity: {len(final_stats['cultural_groups'])} active groups"
        )
        print(f"   Peak Performance: {perf['peak_sps']:.1f} SPS")
        print(f"   Memory Usage: {perf['memory_usage_mb']:.1f}MB")
        print(f"   Workers: {perf['optimization_settings']['workers']}")
    else:
        # Basic world summary
        social_connections = final_stats.get("total_connections", 0)
        print(
            f"   Social Density: {social_connections / args.agents:.1f} connections per agent"
        )
        print(
            f"   Economic Activity: {final_stats['total_resources']['currency']} total currency"
        )
        print(
            f"   Cultural Diversity: {len(final_stats['cultural_groups'])} active groups"
        )


if __name__ == "__main__":
    main()
