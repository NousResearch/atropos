#!/usr/bin/env python3
"""
Enhanced AMIEN Startup Script
Initializes and runs the enhanced research pipeline with AI Scientist and FunSearch
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from enhanced_research_orchestrator import EnhancedResearchOrchestrator

# Configuration
CONFIG = {
    "project_id": "spatial-research-pipeline",  # Update with your GCP project ID
    "use_ai_scientist_v2": True,  # Use AI Scientist v2 (agentic tree search)
    "daily_experiments": 100,
    "papers_per_week": 50,
    "functions_per_week": 50,
    "max_parallel_discoveries": 10,
    "discovery_cycle_hours": 4,  # Run discovery cycle every 4 hours
    "log_level": "INFO",
}


async def main():
    """Main startup function"""

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, CONFIG["log_level"]),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("enhanced_amien.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting Enhanced AMIEN Discovery Pipeline")

    try:
        # Initialize orchestrator
        orchestrator = EnhancedResearchOrchestrator(
            project_id=CONFIG["project_id"], config=CONFIG
        )

        # Setup discovery pipeline
        logger.info("📦 Setting up discovery pipeline...")
        await orchestrator.setup_discovery_pipeline()
        logger.info("✅ Discovery pipeline setup complete")

        # Print startup banner
        print_startup_banner(CONFIG)

        # Run initial discovery cycle
        logger.info("🔬 Running initial discovery cycle...")
        await orchestrator.run_discovery_cycle()
        logger.info("✅ Initial discovery cycle complete")

        # Start continuous discovery loop
        logger.info("🔄 Starting continuous discovery loop...")
        await run_continuous_discovery(orchestrator, CONFIG)

    except KeyboardInterrupt:
        logger.info("👋 Shutting down Enhanced AMIEN...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)


async def run_continuous_discovery(
    orchestrator: EnhancedResearchOrchestrator, config: dict
):
    """Run continuous discovery cycles"""

    logger = logging.getLogger(__name__)
    cycle_interval = config["discovery_cycle_hours"] * 3600  # Convert to seconds

    cycle_count = 0

    while True:
        try:
            cycle_count += 1
            cycle_start = datetime.now()

            logger.info(f"🔬 Starting discovery cycle #{cycle_count}")

            # Run discovery cycle
            await orchestrator.run_discovery_cycle()

            # Get and log statistics
            stats = await orchestrator.get_discovery_statistics()
            log_discovery_statistics(stats, cycle_count)

            # Calculate next cycle time
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            sleep_time = max(0, cycle_interval - cycle_duration)

            if sleep_time > 0:
                logger.info(
                    f"😴 Sleeping for {sleep_time/3600:.1f} hours until next cycle..."
                )
                await asyncio.sleep(sleep_time)
            else:
                logger.warning(
                    f"⚠️  Cycle took longer than interval ({cycle_duration/3600:.1f}h)"
                )

        except Exception as e:
            logger.error(f"❌ Discovery cycle #{cycle_count} failed: {e}")
            # Sleep for 30 minutes before retrying
            await asyncio.sleep(1800)


def print_startup_banner(config: dict):
    """Print startup banner with configuration"""

    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                        🧠 ENHANCED AMIEN DISCOVERY PIPELINE                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  🤖 AI Scientist Integration:                                                ║
║     • Version: {'v2 (Agentic Tree Search)' if config['use_ai_scientist_v2'] else 'v1 (Template-based)'}                                    ║
║     • Target: {config['papers_per_week']} papers/week                                              ║
║                                                                              ║
║  🔬 FunSearch Integration:                                                   ║
║     • Mathematical function discovery                                        ║
║     • Target: {config['functions_per_week']} functions/week                                        ║
║                                                                              ║
║  🌐 Cross-Domain Inspiration:                                               ║
║     • Fireflies, Casino Psychology, Nature Patterns                         ║
║     • Cognitive Science insights                                             ║
║                                                                              ║
║  ⚙️  Configuration:                                                          ║
║     • Project ID: {config['project_id']:<50} ║
║     • Daily Experiments: {config['daily_experiments']:<44} ║
║     • Discovery Cycle: {config['discovery_cycle_hours']} hours                                              ║
║     • Max Parallel: {config['max_parallel_discoveries']:<47} ║
║                                                                              ║
║  🎯 Expected Outcomes:                                                       ║
║     • 100+ research papers per month                                         ║
║     • 100+ novel algorithms per month                                        ║
║     • Autonomous scientific breakthroughs                                    ║
║     • Cross-domain innovation                                                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

🚀 Enhanced AMIEN is now running...
📊 Monitor progress in enhanced_amien.log
🌐 GCS buckets: {config['project_id']}-research-papers, {config['project_id']}-discovered-functions

"""
    print(banner)


def log_discovery_statistics(stats: dict, cycle_count: int):
    """Log discovery statistics"""

    logger = logging.getLogger(__name__)

    if not stats:
        logger.warning("⚠️  No statistics available")
        return

    ai_scientist_stats = stats.get("ai_scientist", {})
    funsearch_stats = stats.get("funsearch", {})
    active_stats = stats.get("active_discoveries", {})
    rates = stats.get("discovery_rates", {})

    logger.info(
        """
📊 Discovery Statistics (Cycle #{cycle_count}):

🤖 AI Scientist:
   • Total Papers: {ai_scientist_stats.get('total_papers', 0)}
   • Total Experiments: {ai_scientist_stats.get('total_experiments', 0)}
   • Storage: {ai_scientist_stats.get('total_size_mb', 0)} MB

🔬 FunSearch:
   • Total Functions: {funsearch_stats.get('total_functions', 0)}
   • Storage: {funsearch_stats.get('total_size_kb', 0)} KB

🔄 Active Discoveries:
   • AI Scientist: {active_stats.get('active_ai_scientist', 0)}
   • FunSearch: {active_stats.get('active_funsearch', 0)}
   • Cross-Domain: {active_stats.get('active_cross_domain', 0)}

📈 Discovery Rates:
   • Papers/day: {rates.get('papers_per_day', 0):.1f}
   • Functions/day: {rates.get('functions_per_day', 0):.1f}
   • Cross-domain insights/day: {rates.get('cross_domain_insights_per_day', 0):.1f}
   • Total discoveries/day: {rates.get('total_discoveries_per_day', 0):.1f}
"""
    )


def check_prerequisites():
    """Check system prerequisites"""

    logger = logging.getLogger(__name__)

    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("❌ Python 3.8+ required")
        sys.exit(1)

    # Check required environment variables
    required_env_vars = [
        "GOOGLE_APPLICATION_CREDENTIALS",
        "PADRES_API_URL",
        "PADRES_API_KEY",
    ]

    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        logger.error(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        logger.info("💡 Set up your environment variables in .env file or export them")
        sys.exit(1)

    # Check GCP project ID
    if not CONFIG["project_id"] or CONFIG["project_id"] == "spatial-research-pipeline":
        logger.warning("⚠️  Update CONFIG['project_id'] with your actual GCP project ID")

    logger.info("✅ Prerequisites check passed")


def setup_directories():
    """Setup required directories"""

    directories = ["logs", "experiments", "papers", "functions", "data"]

    for directory in directories:
        Path(directory).mkdir(exist_ok=True)


if __name__ == "__main__":
    # Check prerequisites
    check_prerequisites()

    # Setup directories
    setup_directories()

    # Run main function
    asyncio.run(main())
