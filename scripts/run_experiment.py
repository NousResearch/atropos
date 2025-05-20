#!/usr/bin/env python3
"""
A simple script to run experiments using the MCP framework
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import our components
from backend_services.mcp_manager import MCPManager
from backend_services.experiment_orchestrator import ExperimentOrchestrator

async def run_experiment(experiment_id, config_path=None):
    """Run a single experiment by ID"""
    
    # Load config
    config_path = config_path or os.path.join(project_root, "config", "pipeline_config.json")
    with open(config_path, 'r') as f:
        pipeline_config = json.load(f)
    
    print(f"Loaded pipeline config from {config_path}")
    print(f"Running experiment: {experiment_id}")
    
    # Initialize MCP Manager and ExperimentOrchestrator
    mcp_manager = MCPManager(config=pipeline_config)
    
    experiment_orchestrator = ExperimentOrchestrator(
        mcp_manager=mcp_manager,
        config=pipeline_config.get("experiment_orchestrator_config", {})
    )
    
    # Fetch and run the experiment
    try:
        config = await experiment_orchestrator.fetch_experiment_config(experiment_id)
        print(f"Fetched experiment config: {config}")
        await experiment_orchestrator.run_single_experiment(config)
        print(f"Experiment {experiment_id} completed successfully.")
    except Exception as e:
        print(f"Error running experiment {experiment_id}: {e}")
        raise

async def run_experiment_batch(batch_name, experiment_ids, config_path=None):
    """Run a batch of experiments"""
    
    # Load config
    config_path = config_path or os.path.join(project_root, "config", "pipeline_config.json")
    with open(config_path, 'r') as f:
        pipeline_config = json.load(f)
    
    print(f"Loaded pipeline config from {config_path}")
    print(f"Running experiment batch '{batch_name}' with experiments: {experiment_ids}")
    
    # Initialize MCP Manager and ExperimentOrchestrator
    mcp_manager = MCPManager(config=pipeline_config)
    
    experiment_orchestrator = ExperimentOrchestrator(
        mcp_manager=mcp_manager,
        config=pipeline_config.get("experiment_orchestrator_config", {})
    )
    
    # Run the batch
    batch_config = {
        "name": batch_name,
        "experiment_ids": experiment_ids
    }
    
    try:
        await experiment_orchestrator.run_experiment_batch(batch_config)
        print(f"Experiment batch '{batch_name}' completed successfully.")
    except Exception as e:
        print(f"Error running experiment batch '{batch_name}': {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Run AI research experiments")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Single experiment command
    experiment_parser = subparsers.add_parser("experiment", help="Run a single experiment")
    experiment_parser.add_argument("experiment_id", help="ID of the experiment to run")
    experiment_parser.add_argument("--config", help="Path to the pipeline config file")
    
    # Batch experiment command
    batch_parser = subparsers.add_parser("batch", help="Run a batch of experiments")
    batch_parser.add_argument("batch_name", help="Name for this experiment batch")
    batch_parser.add_argument("experiment_ids", nargs="+", help="IDs of experiments to run in this batch")
    batch_parser.add_argument("--config", help="Path to the pipeline config file")
    
    args = parser.parse_args()
    
    if args.command == "experiment":
        asyncio.run(run_experiment(args.experiment_id, args.config))
    elif args.command == "batch":
        asyncio.run(run_experiment_batch(args.batch_name, args.experiment_ids, args.config))
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 