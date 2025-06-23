#!/usr/bin/env python3
"""
A script to run a complete example experiment workflow with mock components.
This is ideal for testing the full pipeline without real GCP credentials.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from backend_services.experiment_orchestrator import ExperimentOrchestrator

# Import our components
from backend_services.mcp_manager import MCPManager
from research_automation.analysis_engine import AnalysisEngine
from research_automation.report_generator import ReportGenerator
from research_automation.research_assistant import ResearchAssistant


async def run_full_experiment(
    experiment_id="exp_spatial_reasoning_001", config_path=None
):
    """
    Run a full experiment workflow including analysis and reporting.

    This uses mock components to simulate a complete research workflow.

    Args:
        experiment_id (str): ID of the experiment to run
        config_path (str): Path to the pipeline configuration file
    """
    # Load configuration
    config_path = config_path or os.path.join(
        project_root, "config", "pipeline_config.json"
    )
    with open(config_path, "r") as f:
        pipeline_config = json.load(f)

    print(f"Loaded pipeline config from {config_path}")
    print(f"Mock mode enabled: {pipeline_config.get('mock_mode', False)}")

    # Initialize MCP Manager - the core of our system
    mcp_manager = MCPManager(config=pipeline_config)

    # Initialize Experiment Orchestrator with MCP Manager
    experiment_orchestrator = ExperimentOrchestrator(
        mcp_manager=mcp_manager,
        config=pipeline_config["experiment_orchestrator_config"],
    )

    # Initialize Research Assistant
    research_assistant = ResearchAssistant(
        mcp_manager=mcp_manager, config=pipeline_config["research_assistant_config"]
    )

    # Initialize Analysis Engine
    analysis_engine = AnalysisEngine(
        mcp_manager=mcp_manager, config=pipeline_config["analysis_engine_config"]
    )

    # Initialize Report Generator
    report_generator = ReportGenerator(
        mcp_manager=mcp_manager,
        config=pipeline_config.get("report_generator_config", {}),
    )

    print("\n1. RUNNING EXPERIMENT")
    print("=====================")
    print(f"Running experiment: {experiment_id}")

    try:
        # Step 1: Run the experiment
        experiment_config = await experiment_orchestrator.fetch_experiment_config(
            experiment_id
        )
        experiment_result = await experiment_orchestrator.run_single_experiment(
            experiment_config
        )
        print(f"Experiment result: {json.dumps(experiment_result, indent=2)}")

        # Step 2: Analyze experimental results
        print("\n2. ANALYZING RESULTS")
        print("===================")
        analysis_result = await research_assistant.analyze_experimental_results(
            [experiment_id]
        )
        print(f"Analysis result: {json.dumps(analysis_result, indent=2)}")

        # Step 3: Perform deep analysis
        print("\n3. PERFORMING DEEP ANALYSIS")
        print("==========================")
        deep_analysis_config = {
            "name": f"Performance Analysis for {experiment_id}",
            "query": "SELECT * FROM trial_results WHERE experiment_id = "{experiment_id}'",
        }
        deep_analysis_result = await analysis_engine.perform_deep_analysis(
            deep_analysis_config
        )
        print(f"Deep analysis result: {json.dumps(deep_analysis_result, indent=2)}")

        # Step 4: Generate report
        print("\n4. GENERATING REPORT")
        print("===================")

        # Construct data_sources for the report generator based on previous step outputs
        report_data_sources = []
        if (
            analysis_result
            and analysis_result.get("status") == "success"
            and analysis_result.get("storage_info", {}).get("document_id")
        ):
            ra_output_doc_id = analysis_result["storage_info"]["document_id"]
            ra_collection_id = pipeline_config["research_assistant_config"].get(
                "firestore_insights_collection", "research_assistant_outputs"
            )
            report_data_sources.append(
                {
                    "name": "research_assistant_insights",
                    "type": "firestore",  # Helps ReportGenerator select default MCP if not specified
                    # "mcp_server_name": "firestore_main", # Can be explicit if needed
                    "tool_name": "get_document",
                    "parameters": {
                        "collection_id": ra_collection_id,
                        "document_id": ra_output_doc_id,
                    },
                }
            )
        else:
            print(
                f"Warning: Could not get Research Assistant output document ID from: {analysis_result.get('storage_info')}"
            )

        if (
            deep_analysis_result
            and deep_analysis_result.get("status") == "success"
            and deep_analysis_result.get("storage_info", {}).get("analysis_id_stored")
        ):
            ae_analysis_id = deep_analysis_result["storage_info"]["analysis_id_stored"]
            ae_config = pipeline_config["analysis_engine_config"]
            ae_table_id = ae_config.get(
                "bigquery_analysis_results_table", "analysis_outputs"
            )
            # Assuming project_id and dataset_id are available in ae_config or ReportGenerator will use its defaults
            # For constructing the query, it's better to have them explicitly.
            bq_project = ae_config.get(
                "bigquery_project_id", pipeline_config.get("gcp_project_id")
            )
            bq_dataset = ae_config.get(
                "bigquery_dataset_id",
                pipeline_config.get("bigquery_config", {}).get("dataset_id"),
            )

            if bq_project and bq_dataset:
                report_data_sources.append(
                    {
                        "name": "analysis_engine_summary",
                        "type": "bigquery",
                        "tool_name": "run_query",
                        "parameters": {
                            "query": f"SELECT * FROM `{bq_project}.{bq_dataset}.{ae_table_id}` WHERE analysis_id = '{ae_analysis_id}' LIMIT 1"
                        },
                    }
                )
            else:
                print(
                    "Warning: Missing BigQuery project/dataset details for Analysis Engine source. Cannot query AE results."
                )
        else:
            print(
                f"Warning: Could not get Analysis Engine stored analysis ID from: {deep_analysis_result.get('storage_info')}"
            )

        report_config = {
            "name": f"Research Report for {experiment_id}",
            "data_sources": report_data_sources,
            "template_id": "comprehensive_summary_v1",  # Example template ID for ReportGenerator's LLM formatter
            "use_llm_formatter": True,
            "output_gdoc": True,
            "send_notifications": True,
            "slack_channel_id": "#example-experiment-reports",  # Example channel
        }

        report_result = await report_generator.generate_report(report_config)
        print(f"Report generation result: {json.dumps(report_result, indent=2)}")

        print("\nFULL WORKFLOW COMPLETED SUCCESSFULLY")
        return {
            "experiment_result": experiment_result,
            "analysis_result": analysis_result,
            "deep_analysis_result": deep_analysis_result,
            "report_result": report_result,
        }

    except Exception as e:
        print(f"Error in experiment workflow: {e}")
        return {"error": str(e)}
    finally:
        # Clean up MCP client resources if any
        print("\nCleaning up MCPManager client resources...")
        for server_instance in mcp_manager.mcp_servers.values():
            if hasattr(server_instance, "close_client") and asyncio.iscoroutinefunction(
                server_instance.close_client
            ):
                try:
                    await server_instance.close_client()
                except Exception as ex_close:
                    print(
                        f"Error closing client for server {server_instance.server_name}: {ex_close}"
                    )
            elif hasattr(server_instance, "close_client"):  # if not async
                try:
                    server_instance.close_client()
                except Exception as ex_close:
                    print(
                        f"Error closing client for server {server_instance.server_name}: {ex_close}"
                    )
        print("MCPManager client resource cleanup finished.")


def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(
        description="Run a complete example experiment workflow"
    )
    parser.add_argument(
        "--experiment-id",
        default="exp_spatial_reasoning_001",
        help="Experiment ID to run (default: exp_spatial_reasoning_001)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config file (default: config/pipeline_config.json)",
    )
    args = parser.parse_args()

    # Run the full experiment workflow
    result = asyncio.run(run_full_experiment(args.experiment_id, args.config))

    if "error" in result:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
