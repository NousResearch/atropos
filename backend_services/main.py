from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json # For loading actual config later
import os # For MCPManager test setup
import datetime # Moved import to top

# Assuming these are structured to be importable
# Adjust paths if necessary based on your project structure and PYTHONPATH
from backend_services.mcp_manager import MCPManager
from backend_services.experiment_orchestrator import ExperimentOrchestrator
from research_automation.analysis_engine import AnalysisEngine
from research_automation.report_generator import ReportGenerator

# Import Celery tasks
from .celery_app import celery_app # Import the app instance for task status
from .celery_tasks import (
    run_experiment_task,
    run_experiment_batch_task, 
    analyze_experiment_results_task,
    perform_deep_analysis_task,
    generate_report_task
)

# --- Application Setup ---
app = FastAPI(
    title="AI Research Pipeline API",
    description="API for managing experiments, triggering analyses, and generating reports. Uses Celery for async tasks.",
    version="0.3.0" # Version bump
)

# --- Configuration and Component Initialization ---

# Function to load pipeline configuration
def load_pipeline_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "pipeline_config.json") # Path relative to main.py
    # For testing or alternative structures, you might need to adjust this path or use an environment variable
    # e.g., config_path = os.environ.get("PIPELINE_CONFIG_PATH", "config/pipeline_config.json")
    
    print(f"Attempting to load pipeline configuration from: {os.path.abspath(config_path)}")

    if not os.path.exists(config_path):
        print(f"CRITICAL: Configuration file not found at {config_path}.")
        print("Please ensure 'config/pipeline_config.json' exists relative to the project root or where main.py is located.")
        print("Using a minimal fallback config for MCPManager to allow API to start with limited functionality.")
        # Fallback to a very minimal config if file not found, so MCPManager can at least initialize.
        # Other components will likely fail or be non-operational.
        return {
            "gcp_project_id": "fallback-gcp-project-id",
            "mcp_server_configurations": {
                # Minimal essential mock servers if config fails to load
                "gemini_fallback": {
                    "module_name": "gemini_mcp_server", "class_name": "GeminiMCPServer",
                    "config": {"api_key": "FALLBACK_KEY_WARN_NO_CONFIG", "model_name": "gemini-pro"}
                },
                "bq_fallback": {
                    "module_name": "bigquery_mcp_server", "class_name": "BigQueryMCPServer",
                    "config": {"project_id": "fallback-gcp-project-id"}
                },
                 "firestore_fallback": {
                    "module_name": "firestore_mcp_server", "class_name": "FirestoreMCPServer",
                    "config": {"project_id": "fallback-gcp-project-id"}
                }
            },
            "experiment_orchestrator_config": {"default_llm_mcp_server_name": "gemini_fallback"},
            "research_assistant_config": {"default_llm_mcp_server_name": "gemini_fallback"},
            "analysis_engine_config": {},
            "report_generator_config": {}
        }
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            print("Successfully loaded pipeline_config.json.")
            return config_data
    except json.JSONDecodeError as e:
        print(f"CRITICAL: Error decoding JSON from {config_path}: {e}")
        raise RuntimeError(f"Invalid JSON in configuration file: {config_path}") from e
    except Exception as e:
        print(f"CRITICAL: Could not load configuration file {config_path}: {e}")
        raise RuntimeError(f"Failed to load configuration: {config_path}") from e

# Ensure dummy __init__.py for mcp_servers for MCPManager dynamic loading to work.
# This might be created by MCPManager's own test block if run, but good to have for API startup.
mcp_servers_pkg_path = os.path.join(os.path.dirname(__file__), "mcp_servers")
if not os.path.exists(os.path.join(mcp_servers_pkg_path, "__init__.py")):
    if not os.path.exists(mcp_servers_pkg_path):
        os.makedirs(mcp_servers_pkg_path)
    with open(os.path.join(mcp_servers_pkg_path, "__init__.py"), "w") as f:
        f.write("# This file makes mcp_servers a Python package\n")
    print(f"Created dummy {os.path.join(mcp_servers_pkg_path, '__init__.py')} for API setup.")

PIPELINE_CONFIG = load_pipeline_config()

# Global instances (consider dependency injection for more complex apps)
mcp_manager_global: Optional[MCPManager] = None
experiment_orchestrator_global: Optional[ExperimentOrchestrator] = None
research_assistant_global: Optional[ResearchAssistant] = None
analysis_engine_global: Optional[AnalysisEngine] = None
report_generator_global: Optional[ReportGenerator] = None

@app.on_event("startup")
def startup_event():
    global mcp_manager_global, experiment_orchestrator_global, research_assistant_global, analysis_engine_global, report_generator_global
    print("FastAPI application startup: Initializing core components...")
    try:
        mcp_manager_global = MCPManager(config=PIPELINE_CONFIG)
        experiment_orchestrator_global = ExperimentOrchestrator(
            mcp_manager=mcp_manager_global, 
            config=PIPELINE_CONFIG.get("experiment_orchestrator_config")
        )
        research_assistant_global = ResearchAssistant(
            mcp_manager=mcp_manager_global, 
            config=PIPELINE_CONFIG.get("research_assistant_config")
        )
        analysis_engine_global = AnalysisEngine(
            mcp_manager=mcp_manager_global, 
            config=PIPELINE_CONFIG.get("analysis_engine_config")
        )
        report_generator_global = ReportGenerator(
            mcp_manager=mcp_manager_global, 
            config=PIPELINE_CONFIG.get("report_generator_config")
        )
        print("Core components initialized successfully.")
    except Exception as e:
        print(f"FATAL: Could not initialize core components during startup: {e}")
        # Prevent app from starting or run in degraded mode if critical components fail
        # For now, global vars will remain None, and endpoints will raise 503.

# --- Pydantic Models for API Requests/Responses ---

class TaskSubmissionResponse(BaseModel):
    task_id: str      # This will now be the Celery Task ID
    api_task_id: Optional[str] = None # The ID generated by the API endpoint, if different or for client tracking
    status: str
    message: str
    submitted_at: str

class ExperimentBatchRequest(BaseModel):
    batch_name: str = Field(..., example="Weekly Spatial Reasoning Batch")
    experiment_ids: List[str] = Field(..., example=["exp_sr_001", "exp_sr_002"])
    # Could add more, like a template to generate experiment_ids, or common overrides

# Model for submitting a single experiment via API
class ExperimentSingleRequestParams(BaseModel):
    # Optional: client can suggest an experiment_id, API might use it or generate one if part of config loading
    # For now, experiment_id is a path parameter, so this model is for potential future body params.
    custom_notes: Optional[str] = Field(None, example="Running this for preliminary check.")

class ResearchAnalysisRequest(BaseModel):
    experiment_ids: List[str] = Field(..., example=["exp_completed_001", "exp_completed_002"])
    # Optional context or parameters for this specific analysis run
    analysis_context: Optional[Dict[str, Any]] = Field(None, example={"focus_metric": "success_rate"})

class AnalysisTaskRequest(BaseModel):
    analysis_task_config: Dict[str, Any] = Field(..., example={
        "name": "Q4 Perf Deep Dive",
        "query": "SELECT ...",
        "analysis_type": "general_summary"
    })
    # Optional: analysis_task_id if client wants to suggest one (API might override)
    custom_task_id: Optional[str] = None 

class AnalysisTaskResponse(BaseModel):
    task_id: str 
    task_name: str
    status: str = Field(..., example="submitted" or "processing_async" or "completed")
    message: Optional[str] = None
    # For async, results might not be immediately available
    results_summary: Optional[Dict[str, Any]] = Field(None, description="Results may not be available if status is processing_async")
    storage_info: Optional[Dict[str, Any]] = Field(None, description="Storage info may not be available if status is processing_async")

class ReportGenerationRequest(BaseModel):
    report_task_config: Dict[str, Any]

class ReportGenerationResponse(BaseModel):
    report_name: str
    status: str
    message: Optional[str] = None
    gdoc_url: Optional[str] = None
    slack_message_ts: Optional[str] = None 

class MCPServerStatus(BaseModel):
    server_name: str
    status: str
    type: str

# --- Existing Endpoints (Root and Basic Experiment Placeholders) ---
@app.get("/", summary="Root endpoint", description="Provides a welcome message and API status.")
async def read_root():
    if mcp_manager_global is None:
         return {"message": "Welcome to the AI Research Pipeline API - WARNING: Core components may have FAILED to initialize."}
    return {"message": "Welcome to the AI Research Pipeline API", "status": "healthy", "core_components_initialized": True}

# (Keeping existing experiment endpoints as placeholders, new batch endpoint is preferred)
@app.post("/experiments", summary="Create a new experiment (placeholder - use /experiments/batch)", deprecated=True)
async def create_experiment_placeholder():
    return {"message": "This endpoint is a placeholder. Use POST /experiments/batch to run a batch."}

@app.get("/experiments/{experiment_id}", summary="Get experiment details (placeholder)", deprecated=True)
async def get_experiment_placeholder(experiment_id: str):
    return {"experiment_id": experiment_id, "details": "(placeholder - status tracking TBD)"}


# --- New Endpoints ---

@app.post("/experiments/batch", response_model=TaskSubmissionResponse, status_code=202)
async def trigger_experiment_batch_api(request: ExperimentBatchRequest):
    """
    Submits a new batch of experiments for asynchronous execution.
    The response indicates submission; actual status tracking needs a separate mechanism (e.g., querying a status endpoint or Pub/Sub notifications).
    """
    if not PIPELINE_CONFIG: # Ensure config is loaded
        raise HTTPException(status_code=503, detail="Pipeline configuration not loaded.")
    # experiment_orchestrator_global is not directly used here anymore, task handles its own.
    
    api_assigned_batch_id = f"api_batch_{request.batch_name.replace(' ', '_').lower()}_{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"
    # Pass the full pipeline config to the Celery task
    batch_config_for_task = {"name": request.batch_name, "experiment_ids": request.experiment_ids, "batch_id": api_assigned_batch_id}
    
    print(f"API: Submitting experiment batch '{request.batch_name}' (API ID: {api_assigned_batch_id}) to Celery.")
    try:
        # Pass pipeline_config_dict as the first argument to the Celery task
        celery_task = run_experiment_batch_task.delay(batch_config_dict=batch_config_for_task, pipeline_config_dict=PIPELINE_CONFIG)
        return TaskSubmissionResponse(
            task_id=celery_task.id, 
            api_task_id=api_assigned_batch_id,
            status="SUBMITTED_TO_CELERY", 
            message=f"Experiment batch '{request.batch_name}' submitted for asynchronous processing via Celery. Celery Task ID: {celery_task.id}",
            submitted_at=datetime.datetime.utcnow().isoformat()
        )
    except Exception as e: 
        print(f"API Error submitting batch '{request.batch_name}' to Celery: {e}")
        raise HTTPException(status_code=500, detail=f"Error submitting batch to Celery: {str(e)}")

@app.post("/experiments/run-single/{experiment_id}", response_model=TaskSubmissionResponse, status_code=202)
async def trigger_single_experiment_api(experiment_id: str, params: Optional[ExperimentSingleRequestParams] = None):
    """
    Submits a single experiment for asynchronous execution.
    """
    if not PIPELINE_CONFIG:
        raise HTTPException(status_code=503, detail="Pipeline configuration not loaded.")

    api_assigned_task_id = f"api_single_exp_{experiment_id}_{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"
    print(f"API: Submitting single experiment '{experiment_id}' (API ID: {api_assigned_task_id}) to Celery.")
    
    # Params from request body are currently not used by the task, but could be in future
    # For now, task only needs experiment_id and pipeline_config
    try:
        celery_task = run_experiment_task.delay(experiment_id=experiment_id, pipeline_config_dict=PIPELINE_CONFIG)
        return TaskSubmissionResponse(
            task_id=celery_task.id,
            api_task_id=api_assigned_task_id,
            status="SUBMITTED_TO_CELERY",
            message=f"Single experiment '{experiment_id}' submitted for asynchronous processing. Celery Task ID: {celery_task.id}",
            submitted_at=datetime.datetime.utcnow().isoformat()
        )
    except Exception as e:
        print(f"API Error submitting single experiment '{experiment_id}' to Celery: {e}")
        raise HTTPException(status_code=500, detail=f"Error submitting single experiment to Celery: {str(e)}")

@app.post("/analysis/run", response_model=TaskSubmissionResponse, status_code=202)
async def run_analysis_task_api(request: AnalysisTaskRequest):
    """
    Submits a new analysis task for asynchronous execution.
    """
    if not PIPELINE_CONFIG:
        raise HTTPException(status_code=503, detail="Pipeline configuration not loaded.")
    # analysis_engine_global is not directly used here anymore

    task_name = request.analysis_task_config.get("name", "Unnamed Analysis via API")
    api_assigned_task_id = request.custom_task_id or f"api_analysis_{task_name.replace(' ', '_').lower()}_{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"
    
    print(f"API: Submitting analysis '{task_name}' (API ID: {api_assigned_task_id}) to Celery.")
    try:
        # Pass pipeline_config_dict as the first argument
        celery_task = perform_deep_analysis_task.delay(
            analysis_config=request.analysis_task_config, 
            pipeline_config_dict=PIPELINE_CONFIG
        )
        return TaskSubmissionResponse(
            task_id=celery_task.id, 
            api_task_id=api_assigned_task_id,
            status="SUBMITTED_TO_CELERY",
            message=f"Analysis task '{task_name}' (API ID: {api_assigned_task_id}) submitted for asynchronous processing via Celery. Celery Task ID: {celery_task.id}",
            submitted_at=datetime.datetime.utcnow().isoformat()
        )
    except Exception as e:
        print(f"API Error submitting analysis '{task_name}' to Celery: {e}")
        raise HTTPException(status_code=500, detail=f"Error submitting analysis to Celery: {str(e)}")

@app.post("/research/analyze-results", response_model=TaskSubmissionResponse, status_code=202)
async def trigger_research_analysis_api(request: ResearchAnalysisRequest):
    """
    Submits a request to analyze results for a list of experiment IDs using ResearchAssistant.
    """
    if not PIPELINE_CONFIG:
        raise HTTPException(status_code=503, detail="Pipeline configuration not loaded.")

    if not request.experiment_ids:
        raise HTTPException(status_code=400, detail="experiment_ids list cannot be empty.")

    api_task_id = f"api_ra_analyze_{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"
    print(f"API: Submitting research analysis for IDs: {request.experiment_ids} (API ID: {api_task_id}) to Celery.")

    try:
        celery_task = analyze_experiment_results_task.delay(
            experiment_ids=request.experiment_ids,
            pipeline_config_dict=PIPELINE_CONFIG
            # research_assistant_config (if needed by task explicitly, but task gets it from pipeline_config_dict)
        )
        return TaskSubmissionResponse(
            task_id=celery_task.id,
            api_task_id=api_task_id,
            status="SUBMITTED_TO_CELERY",
            message=f"Research analysis for {len(request.experiment_ids)} experiment(s) submitted. Celery Task ID: {celery_task.id}",
            submitted_at=datetime.datetime.utcnow().isoformat()
        )
    except Exception as e:
        print(f"API Error submitting research analysis to Celery: {e}")
        raise HTTPException(status_code=500, detail=f"Error submitting research analysis to Celery: {str(e)}")

@app.post("/reports/generate", response_model=TaskSubmissionResponse, status_code=202)
async def generate_report_api(request: ReportGenerationRequest):
    """
    Submits a report generation task for asynchronous execution.
    """
    if not PIPELINE_CONFIG:
        raise HTTPException(status_code=503, detail="Pipeline configuration not loaded.")

    report_name_safe = request.report_task_config.get("name", "untitled_report").replace(' ', '_').lower()
    api_assigned_report_id = f"api_report_{report_name_safe}_{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"

    print(f"API: Submitting report generation task '{report_name_safe}' (API ID: {api_assigned_report_id}) to Celery.")
    try:
        celery_task = generate_report_task.delay(
            report_config=request.report_task_config,
            pipeline_config_dict=PIPELINE_CONFIG
        )
        return TaskSubmissionResponse(
            task_id=celery_task.id,
            api_task_id=api_assigned_report_id,
            status="SUBMITTED_TO_CELERY",
            message=f"Report generation task '{report_name_safe}' submitted. Celery Task ID: {celery_task.id}",
            submitted_at=datetime.datetime.utcnow().isoformat()
        )
    except Exception as e:
        print(f"API Error submitting report generation to Celery: {e}")
        raise HTTPException(status_code=500, detail=f"Error submitting report generation to Celery: {str(e)}")

@app.get("/mcp/servers", response_model=List[MCPServerStatus], summary="List available MCP Servers")
async def list_mcp_servers():
    """Lists all MCP servers currently loaded and their status by the MCPManager."""
    if not mcp_manager_global:
        raise HTTPException(status_code=503, detail="MCPManager not available.")
    
    servers_status = []
    server_names = mcp_manager_global.list_servers()
    for name in server_names:
        server_instance = mcp_manager_global.get_server(name)
        if server_instance:
            status_dict = server_instance.get_status() # Renamed from 'status' to avoid conflict
            servers_status.append(MCPServerStatus(server_name=name, status=status_dict.get("status"), type=status_dict.get("type")))
        else:
            servers_status.append(MCPServerStatus(server_name=name, status="error_not_found_in_manager_dict", type="Unknown"))
    return servers_status

# Example endpoint to check Celery task status (requires Celery result backend to be configured)
@app.get("/tasks/{task_id}/status", summary="Get Celery task status")
async def get_task_status(task_id: str):
    # This requires celery_pipeline_app to be accessible or task result fetched differently
    # Updated to use celery_app imported from .celery_app
    task_result = celery_app.AsyncResult(task_id)
    
    response = {
        "task_id": task_id,
        "status": task_result.status,
        "ready": task_result.ready(),
    }
    if task_result.successful():
        response["result"] = task_result.result
    elif task_result.failed():
        response["error"] = str(task_result.info)  # .info contains the exception
        response["traceback"] = task_result.traceback
    elif task_result.status == 'PENDING':
        response["message"] = "Task is pending or not found."
    elif task_result.status == 'STARTED':
        response["message"] = "Task has started."
        response["meta"] = task_result.info # Could contain custom meta if task updates it
    elif task_result.status == 'RETRY':
        response["message"] = "Task is scheduled for retry."
        response["meta"] = task_result.info
    else:
        response["meta"] = task_result.info # For other states like PROGRESS
        
    return response

# Add other necessary imports and routes as the project evolves 