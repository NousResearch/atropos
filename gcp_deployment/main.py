#!/usr/bin/env python3
"""
Main entry point for 2,500-Agent LLM Society Simulation
"""

import asyncio
import logging
import os
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.monitoring.metrics import MetricsCollector
from src.simulation.society_simulator import SocietySimulator
from src.utils.config import Config

app = typer.Typer(name="llm-society", help="2,500-Agent LLM Society Simulation")
console = Console()


def setup_logging(debug: bool = False):
    """Set up rich logging"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@app.command()
def run(
    agents: int = typer.Option(
        50, "--agents", "-a", help="Number of agents to simulate"
    ),
    steps: int = typer.Option(1000, "--steps", "-s", help="Number of simulation steps"),
    model: str = typer.Option("gemini-pro", "--model", "-m", help="LLM model to use"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug logging"),
    config_file: Optional[str] = typer.Option(
        None, "--config", "-c", help="Configuration file path"
    ),
    output_dir: str = typer.Option(
        "./results", "--output", "-o", help="Output directory for results"
    ),
):
    """Run the LLM society simulation"""
    setup_logging(debug)
    logger = logging.getLogger(__name__)

    console.print(
        "🚀 [bold green]Starting 2,500-Agent LLM Society Simulation[/bold green]"
    )
    console.print(f"📊 Agents: {agents}")
    console.print(f"⏱️  Steps: {steps}")
    console.print(f"🤖 Model: {model}")

    try:
        # Load configuration
        config = Config.load(config_file) if config_file else Config.default()
        config.agents.count = agents
        config.simulation.max_steps = steps
        config.llm.model_name = model
        config.output.directory = output_dir

        # Create simulator
        simulator = SocietySimulator(config)

        # Run simulation
        asyncio.run(simulator.run())

        console.print("✅ [bold green]Simulation completed successfully![/bold green]")

    except KeyboardInterrupt:
        console.print("⚠️  [yellow]Simulation interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"❌ [bold red]Simulation failed: {e}[/bold red]")
        if debug:
            console.print_exception()
        sys.exit(1)


@app.command()
def demo(
    scenario: str = typer.Option(
        "basic", "--scenario", "-s", help="Demo scenario to run"
    ),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug logging"),
):
    """Run a demo scenario"""
    setup_logging(debug)
    console.print(f"🎭 [bold blue]Running demo scenario: {scenario}[/bold blue]")

    from src.simulation.demo_scenarios import run_demo_scenario

    asyncio.run(run_demo_scenario(scenario))


@app.command()
def benchmark(
    agents: int = typer.Option(
        100, "--agents", "-a", help="Number of agents for benchmark"
    ),
    duration: int = typer.Option(
        60, "--duration", "-d", help="Benchmark duration in seconds"
    ),
):
    """Run performance benchmarks"""
    console.print("⚡ [bold yellow]Running performance benchmark[/bold yellow]")
    console.print(f"📊 Agents: {agents}")
    console.print(f"⏱️  Duration: {duration}s")

    from src.monitoring.benchmarks import run_benchmark

    asyncio.run(run_benchmark(agents, duration))


@app.command()
def install_deps():
    """Install additional dependencies"""
    console.print("📦 [bold blue]Installing additional dependencies...[/bold blue]")
    os.system("./install_dev_dependencies.sh")


@app.command()
def validate():
    """Validate installation and configuration"""
    console.print("🔍 [bold blue]Validating installation...[/bold blue]")

    try:
        # Test imports
        import mesa
        import torch
        import transformers

        console.print("✅ Core dependencies imported successfully")

        # Test mesa-frames
        try:
            import mesa_frames

            console.print("✅ Mesa-frames available")
        except ImportError:
            console.print("⚠️  Mesa-frames not available, will use fallback")

        # Test GPU availability
        if torch.cuda.is_available():
            console.print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        else:
            console.print("⚠️  CUDA not available, using CPU")

        console.print("🎉 [bold green]Installation validation complete![/bold green]")

    except Exception as e:
        console.print(f"❌ [bold red]Validation failed: {e}[/bold red]")


def main():
    """Main entry point"""
    app()


if __name__ == "__main__":
    main()
