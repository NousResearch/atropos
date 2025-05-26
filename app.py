import os
import logging

# --- Early ADC Check ---
try:
    import google.auth
    credentials, project = google.auth.default()
    logging.basicConfig(level=logging.INFO) # Ensure logging is configured early
    logger_adc_check = logging.getLogger("adc_check")
    logger_adc_check.info(f"ADC Check: Successfully obtained credentials. Project: {project}")
    if hasattr(credentials, 'service_account_email'):
        logger_adc_check.info(f"ADC Check: Credentials service account: {credentials.service_account_email}")
    else:
        logger_adc_check.info("ADC Check: Credentials are user credentials (not a service account).")
except Exception as e:
    logging.basicConfig(level=logging.CRITICAL)
    logger_adc_check = logging.getLogger("adc_check")
    logger_adc_check.critical(f"CRITICAL ADC Check FAILED: {e}", exc_info=True)
    # Depending on policy, you might want to exit or raise here to prevent app startup
# --- End Early ADC Check ---

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import json # For parsing request body if needed

# Assuming these files are in the same directory or accessible via PYTHONPATH
from production_research_pipeline import Production24x7Pipeline
from paper_generator import AutomatedPaperGenerator
# SimplePadresResearch and ResearchDataManager are used by the above classes

# Ensure logging is configured if not done by ADC check success path
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Automated Research Lab API",
    description="API for running AI research experiments and generating papers.",
    version="0.1.0"
)

# --- Global Objects (Initialize once) ---
try:
    # For local .env loading, ensure GOOGLE_CLOUD_PROJECT is set before this is called
    # In Cloud Run, env vars will be set in the service configuration.
    from dotenv import load_dotenv
    load_dotenv() # Load .env file for local development
    
    pipeline = Production24x7Pipeline() # Initializes researcher and data_manager
    paper_gen = AutomatedPaperGenerator(
        data_manager=pipeline.data_manager, # Reuse data_manager
        researcher=pipeline.researcher     # Reuse researcher
    )
    logger.info("FastAPI application and core components initialized successfully.")
except ImportError as e:
    logger.warning(f"python-dotenv not found, .env file will not be loaded. Error: {e}")
    # Proceed with initialization, relying on environment variables being set elsewhere (e.g., in Cloud Run)
    pipeline = Production24x7Pipeline()
    paper_gen = AutomatedPaperGenerator(
        data_manager=pipeline.data_manager,
        researcher=pipeline.researcher
    )
    logger.info("FastAPI application and core components initialized (without dotenv).")
except Exception as e:
    logger.error(f"Fatal error during application initialization: {e}", exc_info=True)
    # If core components fail to initialize, the app might not be usable.
    # Consider how to handle this, e.g., by having health check fail or exiting.
    pipeline = None
    paper_gen = None

# --- Pydantic Models for Request Bodies (Optional but good practice) ---
class ExperimentRequest(BaseModel):
    batch_size: int = 5

# --- API Endpoints ---

@app.post("/run-experiment")
async def run_experiment_endpoint(request_data: ExperimentRequest):
    """Run a batch of experiments and store results."""
    if not pipeline:
        logger.error("Attempted to run experiment, but pipeline is not initialized.")
        raise HTTPException(status_code=503, detail="Service not fully initialized: Pipeline unavailable.")
    
    logger.info(f"Received request to /run-experiment with batch_size: {request_data.batch_size}")
    try:
        results = pipeline.run_experiment_batch(batch_size=request_data.batch_size)
        logger.info(f"Experiment batch finished. Results: {results}")
        # The results from run_experiment_batch is already a dict structured for response
        return JSONResponse(content=results, status_code=200)
    except Exception as e:
        logger.error(f"Error during /run-experiment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to run experiments: {str(e)}")

@app.post("/generate-paper")
async def generate_paper_endpoint():
    """Generate a weekly research paper."""
    if not pipeline:
        logger.error("Attempted to generate paper, but pipeline is not initialized.")
        raise HTTPException(status_code=503, detail="Service not fully initialized: Pipeline unavailable.")

    logger.info("Received request to /generate-paper")
    try:
        # Production24x7Pipeline now has a generate_paper method
        paper_details = pipeline.generate_paper() 
        logger.info(f"Paper generation process finished. Details: {paper_details}")
        # paper_details is already a dict structured for response
        if "error" in paper_details:
             return JSONResponse(content=paper_details, status_code=500)
        return JSONResponse(content=paper_details, status_code=200)
    except Exception as e:
        logger.error(f"Error during /generate-paper: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate paper: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if pipeline: # Check if the core pipeline initialized successfully
        logger.info("Health check: OK")
        return JSONResponse(content={"status": "healthy", "pipeline_initialized": True}, status_code=200)
    else:
        logger.warning("Health check: DEGRADED - Core pipeline component not initialized.")
        return JSONResponse(content={"status": "degraded", "reason": "Core pipeline component not initialized"}, status_code=503)

# The following block for direct uvicorn execution is removed 
# as Gunicorn will be used as the process manager in production.
# logger.info("Starting FastAPI server locally with Uvicorn.")
# # Make sure GOOGLE_CLOUD_PROJECT is set in your .env or environment for local testing
# # Example: GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
# if not os.getenv("GOOGLE_CLOUD_PROJECT"):
#     logger.warning("GOOGLE_CLOUD_PROJECT environment variable is not set. "
#                    "Ensure it's in your .env file or environment for BigQueryManager to work correctly.")
# 
# uvicorn.run(app, host="0.0.0.0", port=8080) 