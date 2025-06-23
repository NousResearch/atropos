# backend_services/celery_tasks.py

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List

# Ensure project root is in sys.path for consistent imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from backend_services.celery_app import celery_app  # Import the Celery app instance
from backend_services.experiment_orchestrator import ExperimentOrchestrator
from backend_services.mcp_manager import MCPManager

# Import other components as needed for future tasks
# from research_automation.research_assistant import ResearchAssistant
# from research_automation.analysis_engine import AnalysisEngine
# from research_automation.report_generator import ReportGenerator

# _get_pipeline_config is removed as tasks will now receive the config dictionary directly.
# DEFAULT_CONFIG_PATH = str(project_root / "config" / "pipeline_config.json")
# def _get_pipeline_config(config_path_str: str = None) -> dict:
#     """Loads the pipeline configuration file."""
#     config_path = Path(config_path_str or DEFAULT_CONFIG_PATH)
#     if not config_path.exists():
#         alt_config_path = project_root / (config_path_str or "config/pipeline_config.json")
#         if alt_config_path.exists():
#             config_path = alt_config_path
#         else:
#             raise FileNotFoundError(f"Pipeline configuration not found at {config_path} or {alt_config_path}")
#
#     with open(config_path, 'r') as f:
#         return json.load(f)


def _initialize_orchestrator(pipeline_config: dict) -> ExperimentOrchestrator:
    """Initializes MCPManager and ExperimentOrchestrator."""
    mcp_manager = MCPManager(config=pipeline_config)
    # It's important that ExperimentOrchestrator can handle its config possibly being None
    # or having a default if not found in pipeline_config
    orchestrator_config = pipeline_config.get("experiment_orchestrator_config", {})
    experiment_orchestrator = ExperimentOrchestrator(
        mcp_manager=mcp_manager, config=orchestrator_config
    )
    return experiment_orchestrator


async def _close_mcp_manager_clients(mcp_manager: MCPManager):
    """Helper to close MCP server clients."""
    if mcp_manager:
        for server_instance in mcp_manager.mcp_servers.values():
            if hasattr(server_instance, "close_client") and asyncio.iscoroutinefunction(
                server_instance.close_client
            ):
                try:
                    await server_instance.close_client()
                    print(
                        f"Closed client for MCP server {server_instance.server_name} in Celery task."
                    )
                except Exception as e_close:
                    print(
                        f"Error closing client for server {server_instance.server_name} in Celery task: {e_close}"
                    )
            elif hasattr(server_instance, "close_client"):
                try:
                    server_instance.close_client()
                    print(
                        f"Closed client for MCP server {server_instance.server_name} (sync) in Celery task."
                    )
                except Exception as e_close:
                    print(
                        f"Error closing client for server {server_instance.server_name} (sync) in Celery task: {e_close}"
                    )


@celery_app.task(name="pipeline_tasks.run_experiment")
def run_experiment_task(experiment_id: str, pipeline_config_dict: dict) -> dict:
    """Celery task to run a single experiment by ID."""
    print(
        f"Celery task: run_experiment_task started for experiment_id: {experiment_id}"
    )
    loop = asyncio.get_event_loop()
    orchestrator = None
    mcp_manager = None  # To access for cleanup
    try:
        # pipeline_config = _get_pipeline_config(config_path_str) # No longer using path
        orchestrator = _initialize_orchestrator(pipeline_config_dict)
        mcp_manager = orchestrator.mcp_manager  # Get manager from orchestrator

        experiment_config_dict_fetched = loop.run_until_complete(
            orchestrator.fetch_experiment_config(experiment_id)
        )
        if not experiment_config_dict_fetched or experiment_config_dict_fetched.get(
            "error"
        ):
            error_msg = f"Failed to fetch experiment config for {experiment_id}: {experiment_config_dict_fetched.get('error', 'Not found')}"
            print(error_msg)
            return {
                "status": "error",
                "experiment_id": experiment_id,
                "error": error_msg,
            }

        result = loop.run_until_complete(
            orchestrator.run_single_experiment(experiment_config_dict_fetched)
        )
        print(
            f"Celery task: run_experiment_task completed for experiment_id: {experiment_id}. Result: {json.dumps(result, indent=2)}"
        )
        return result
    except FileNotFoundError as fnf:
        print(f"Error in run_experiment_task: {fnf}")
        return {"status": "error", "experiment_id": experiment_id, "error": str(fnf)}
    except Exception as e:
        print(f"Error in run_experiment_task for {experiment_id}: {e}")
        # Consider adding traceback here for better debugging
        return {"status": "error", "experiment_id": experiment_id, "error": str(e)}
    finally:
        if mcp_manager:  # mcp_manager is set if orchestrator was initialized
            loop.run_until_complete(_close_mcp_manager_clients(mcp_manager))
        print(
            f"Celery task: run_experiment_task finished for experiment_id: {experiment_id}"
        )


@celery_app.task(name="pipeline_tasks.run_experiment_batch")
def run_experiment_batch_task(
    batch_config_dict: dict, pipeline_config_dict: dict
) -> dict:
    """
    Celery task to run a batch of experiments.
    batch_config_dict should contain 'name' and 'experiment_ids'.
    """
    batch_name = batch_config_dict.get("name", "Unnamed Batch")
    experiment_ids = batch_config_dict.get("experiment_ids", [])
    print(
        "Celery task: run_experiment_batch_task started for batch: "{batch_name}' with IDs: {experiment_ids}"
    )
    loop = asyncio.get_event_loop()
    orchestrator = None
    mcp_manager = None
    try:
        if not experiment_ids:
            return {
                "status": "error",
                "batch_name": batch_name,
                "error": "No experiment_ids provided in batch_config.",
            }

        # pipeline_config = _get_pipeline_config(config_path_str) # No longer using path
        orchestrator = _initialize_orchestrator(pipeline_config_dict)
        mcp_manager = orchestrator.mcp_manager

        # The ExperimentOrchestrator.run_experiment_batch expects a config object
        # that itself contains the list of experiment_ids.
        # The input `batch_config_dict` should already be in the correct format.
        result = loop.run_until_complete(
            orchestrator.run_experiment_batch(batch_config_dict)
        )
        print(
            "Celery task: run_experiment_batch_task completed for batch "{batch_name}'. Result: {json.dumps(result, indent=2)}"
        )
        return result
    except FileNotFoundError as fnf:
        print(f"Error in run_experiment_batch_task: {fnf}")
        return {"status": "error", "batch_name": batch_name, "error": str(fnf)}
    except Exception as e:
        print("Error in run_experiment_batch_task for batch "{batch_name}': {e}")
        return {"status": "error", "batch_name": batch_name, "error": str(e)}
    finally:
        if mcp_manager:
            loop.run_until_complete(_close_mcp_manager_clients(mcp_manager))
        print(
            "Celery task: run_experiment_batch_task finished for batch: "{batch_name}'"
        )


# Example of how to add a ResearchAssistant task (conceptual for now)
# @celery_app.task(name="pipeline_tasks.analyze_results")
# def analyze_results_task(experiment_ids: list, config_path_str: str = None) -> dict:
#     print(f"Celery task: analyze_results_task started for experiment_ids: {experiment_ids}")
#     loop = asyncio.get_event_loop()
#     assistant = None
#     mcp_manager = None
#     try:
#         pipeline_config = _get_pipeline_config(config_path_str)
#         mcp_manager = MCPManager(config=pipeline_config)
#         ra_config = pipeline_config.get("research_assistant_config", {})
#         assistant = ResearchAssistant(mcp_manager=mcp_manager, config=ra_config)
#
#         result = loop.run_until_complete(
#             assistant.analyze_experimental_results(experiment_ids)
#         )
#         print(f"Celery task: analyze_results_task completed for {experiment_ids}. Result: {result}")
#         return result
#     except FileNotFoundError as fnf:
#         print(f"Error in analyze_results_task: {fnf}")
#         return {"status": "error", "experiment_ids": experiment_ids, "error": str(fnf)}
#     except Exception as e:
#         print(f"Error in analyze_results_task for {experiment_ids}: {e}")
#         return {"status": "error", "experiment_ids": experiment_ids, "error": str(e)}
#     finally:
#         if mcp_manager:
#             loop.run_until_complete(_close_mcp_manager_clients(mcp_manager))
#         print(f"Celery task: analyze_results_task finished for {experiment_ids}")


# TODO:
# - Add tasks for AnalysisEngine: perform_deep_analysis_task
# - Add tasks for ReportGenerator: generate_report_task
# - Consider task chaining and error handling strategies for multi-step workflows.
# - Ensure robust configuration management for tasks (e.g., Celery config, pipeline config injection).
# - Test task execution thoroughly with a running Celery worker and broker.

print(
    "Celery tasks (updated for direct pipeline_config_dict) defined in backend_services.celery_tasks"
)

# --- Add new tasks below ---

from research_automation.analysis_engine import AnalysisEngine
from research_automation.report_generator import ReportGenerator
from research_automation.research_assistant import ResearchAssistant


@celery_app.task(name="pipeline_tasks.analyze_experiment_results")
def analyze_experiment_results_task(
    experiment_ids: List[str], pipeline_config_dict: dict
) -> dict:
    """Celery task to analyze experimental results using ResearchAssistant."""
    print(
        f"Celery task: analyze_experiment_results_task started for experiment_ids: {experiment_ids}"
    )
    loop = asyncio.get_event_loop()
    mcp_manager = None
    try:
        # pipeline_config = _get_pipeline_config(config_path_str) # No longer using path
        mcp_manager = MCPManager(config=pipeline_config_dict)
        ra_config = pipeline_config_dict.get("research_assistant_config", {})
        assistant = ResearchAssistant(mcp_manager=mcp_manager, config=ra_config)

        result = loop.run_until_complete(
            assistant.analyze_experimental_results(experiment_ids)
        )
        print(
            f"Celery task: analyze_experiment_results_task completed for {experiment_ids}. Result: {json.dumps(result, indent=2)}"
        )
        return result
    except FileNotFoundError as fnf:
        print(f"Error in analyze_experiment_results_task: {fnf}")
        return {
            "status": "error",
            "inputs": {"experiment_ids": experiment_ids},
            "error": str(fnf),
        }
    except Exception as e:
        print(f"Error in analyze_experiment_results_task for {experiment_ids}: {e}")
        return {
            "status": "error",
            "inputs": {"experiment_ids": experiment_ids},
            "error": str(e),
        }
    finally:
        if mcp_manager:
            loop.run_until_complete(_close_mcp_manager_clients(mcp_manager))
        print(
            f"Celery task: analyze_experiment_results_task finished for {experiment_ids}"
        )


@celery_app.task(name="pipeline_tasks.perform_deep_analysis")
def perform_deep_analysis_task(
    analysis_config: dict, pipeline_config_dict: dict
) -> dict:
    """Celery task to perform deep analysis using AnalysisEngine."""
    analysis_name = analysis_config.get("name", "Unnamed Analysis")
    print(
        "Celery task: perform_deep_analysis_task started for analysis: "{analysis_name}'"
    )
    loop = asyncio.get_event_loop()
    mcp_manager = None
    try:
        # pipeline_config = _get_pipeline_config(config_path_str) # No longer using path
        mcp_manager = MCPManager(config=pipeline_config_dict)
        ae_config = pipeline_config_dict.get("analysis_engine_config", {})
        engine = AnalysisEngine(mcp_manager=mcp_manager, config=ae_config)

        result = loop.run_until_complete(engine.perform_deep_analysis(analysis_config))
        print(
            "Celery task: perform_deep_analysis_task completed for "{analysis_name}'. Result: {json.dumps(result, indent=2)}"
        )
        return result
    except FileNotFoundError as fnf:
        print(f"Error in perform_deep_analysis_task: {fnf}")
        return {
            "status": "error",
            "inputs": {"analysis_config_name": analysis_name},
            "error": str(fnf),
        }
    except Exception as e:
        print("Error in perform_deep_analysis_task for "{analysis_name}': {e}")
        return {
            "status": "error",
            "inputs": {"analysis_config_name": analysis_name},
            "error": str(e),
        }
    finally:
        if mcp_manager:
            loop.run_until_complete(_close_mcp_manager_clients(mcp_manager))
        print("Celery task: perform_deep_analysis_task finished for "{analysis_name}'")


@celery_app.task(name="pipeline_tasks.generate_report")
def generate_report_task(report_config: dict, pipeline_config_dict: dict) -> dict:
    """Celery task to generate a report using ReportGenerator."""
    report_name = report_config.get("name", "Unnamed Report")
    print("Celery task: generate_report_task started for report: "{report_name}'")
    loop = asyncio.get_event_loop()
    mcp_manager = None
    try:
        # pipeline_config = _get_pipeline_config(config_path_str) # No longer using path
        mcp_manager = MCPManager(config=pipeline_config_dict)
        rg_config = pipeline_config_dict.get("report_generator_config", {})
        reporter = ReportGenerator(mcp_manager=mcp_manager, config=rg_config)

        result = loop.run_until_complete(reporter.generate_report(report_config))
        print(
            "Celery task: generate_report_task completed for "{report_name}'. Result: {json.dumps(result, indent=2)}"
        )
        return result
    except FileNotFoundError as fnf:
        print(f"Error in generate_report_task: {fnf}")
        return {
            "status": "error",
            "inputs": {"report_config_name": report_name},
            "error": str(fnf),
        }
    except Exception as e:
        print("Error in generate_report_task for "{report_name}': {e}")
        return {
            "status": "error",
            "inputs": {"report_config_name": report_name},
            "error": str(e),
        }
    finally:
        if mcp_manager:
            loop.run_until_complete(_close_mcp_manager_clients(mcp_manager))
        print("Celery task: generate_report_task finished for "{report_name}'")
