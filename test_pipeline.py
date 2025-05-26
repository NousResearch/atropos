import os
import uuid
import time
import logging
import json
from pathlib import Path
from datetime import datetime

from fastapi.testclient import TestClient
from dotenv import load_dotenv

# Attempt to import app and other necessary components
# These should be in the same directory or accessible via PYTHONPATH
try:
    from app import app  # Your FastAPI application instance
    from bigquery_manager import ResearchDataManager
    # Production24x7Pipeline and AutomatedPaperGenerator are used by app.py
    # SimplePadresResearch is used by Production24x7Pipeline
except ImportError as e:
    print(f"CRITICAL: Failed to import necessary modules: {e}")
    print("Ensure app.py, bigquery_manager.py, etc., are in the same directory or PYTHONPATH.")
    exit(1)

# Configure basic logging for the test script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - TEST - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Test Configuration & Helper Functions ---
TEST_RESULTS = []

def record_test_result(name: str, success: bool, message: str = ""):
    status = "PASS" if success else "FAIL"
    logger.info(f"Test {name}: {status} - {message}")
    TEST_RESULTS.append({"name": name, "status": status, "message": message})
    if not success:
        # Optionally, raise an error to stop on first failure or collect all results
        # raise AssertionError(f"Test {name} failed: {message}")
        pass 

# --- Test Functions ---

def test_health_check(client: TestClient):
    test_name = "Health Check Endpoint"
    logger.info(f"Running test: {test_name}")
    try:
        response = client.get("/health")
        if response.status_code == 200 and response.json().get("status") == "healthy":
            record_test_result(test_name, True, "Responded healthy.")
        else:
            record_test_result(test_name, False, f"Status: {response.status_code}, Body: {response.text}")
    except Exception as e:
        record_test_result(test_name, False, f"Exception: {e}")

def test_run_experiment_and_bigquery_storage(client: TestClient, data_manager: ResearchDataManager):
    test_name = "Run Experiment & BigQuery Storage"
    logger.info(f"Running test: {test_name}")
    experiment_id_to_check = None
    try:
        response = client.post("/run-experiment", json={"batch_size": 1}) # Run a single experiment for faster testing
        if response.status_code != 200:
            record_test_result(test_name, False, f"API call failed. Status: {response.status_code}, Body: {response.text}")
            return

        response_data = response.json()
        if not response_data.get("details") or not isinstance(response_data["details"], list) or not response_data["details"][0].get("status") == "success":
            record_test_result(test_name, False, f"Experiment run reported failure or unexpected response: {response_data}")
            return
        
        experiment_id_to_check = response_data["details"][0].get("experiment_id")
        if not experiment_id_to_check or experiment_id_to_check == 'N/A':
            record_test_result(test_name, False, f"Valid experiment_id not found in response: {response_data}")
            return
        
        logger.info(f"Experiment run via API reported success. Checking BigQuery for experiment_id: {experiment_id_to_check}")
        time.sleep(5) # Allow some time for potential BQ eventual consistency or async write
        
        # Query BQ for the specific experiment ID. This is tricky if ID is just a timestamp.
        # A more robust way would be to have a truly unique ID or query by a very recent timestamp range.
        # For this test, let's assume timestamp is somewhat unique for a test run.
        recent_experiments = data_manager.get_recent_experiments(days=1) # Check last day
        found_in_bq = False
        for exp in recent_experiments:
            # The ID from API might be a timestamp string, ensure consistent comparison or type conversion if needed.
            if str(exp.get("experiment_id")) == str(experiment_id_to_check) or \
               (isinstance(exp.get("timestamp"), datetime) and exp.get("timestamp").isoformat().startswith(experiment_id_to_check[:19])):
                found_in_bq = True
                logger.info(f"Found experiment {experiment_id_to_check} in BigQuery: {exp}")
                break
        
        if found_in_bq:
            record_test_result(test_name, True, f"Experiment {experiment_id_to_check} successfully run and found in BigQuery.")
        else:
            record_test_result(test_name, False, f"Experiment {experiment_id_to_check} run via API but NOT found in BigQuery. Recent experiments: {len(recent_experiments)} found.")
            if recent_experiments:
                logger.debug(f"First recent BQ entry for comparison: {recent_experiments[0]}")

    except Exception as e:
        record_test_result(test_name, False, f"Exception: {e}")

def test_paper_generation_creates_file(client: TestClient):
    test_name = "Paper Generation Creates File"
    logger.info(f"Running test: {test_name}")
    generated_paper_path = None
    try:
        response = client.post("/generate-paper")
        if response.status_code != 200:
            record_test_result(test_name, False, f"API call failed. Status: {response.status_code}, Body: {response.text}")
            return

        response_data = response.json()
        generated_filename = response_data.get("filename")
        if not generated_filename or not response_data.get("message", "").startswith("Research paper generated successfully"):
            record_test_result(test_name, False, f"Paper generation reported failure or unexpected response: {response_data}")
            return
        
        generated_paper_path = Path(generated_filename)
        if generated_paper_path.is_file() and generated_paper_path.stat().st_size > 0:
            record_test_result(test_name, True, f"Paper '{generated_filename}' created successfully.")
        else:
            record_test_result(test_name, False, f"Paper file '{generated_filename}' not found or is empty.")
            
    except Exception as e:
        record_test_result(test_name, False, f"Exception: {e}")
    finally:
        # Cleanup: Remove the generated paper file if it exists
        if generated_paper_path and generated_paper_path.exists():
            try:
                generated_paper_path.unlink()
                logger.info(f"Cleaned up generated paper: {generated_paper_path}")
            except OSError as e_unlink:
                logger.warning(f"Could not clean up paper {generated_paper_path}: {e_unlink}")

def test_direct_bigquery_operations(data_manager: ResearchDataManager):
    test_name = "Direct BigQuery Store and Retrieve"
    logger.info(f"Running test: {test_name}")
    unique_experiment_id = f"direct_bq_test_{uuid.uuid4()}"
    try:
        # Check if dataset and table exist (implicitly tested by ResearchDataManager init)
        logger.info(f"Dataset '{data_manager.dataset_id}' and table '{data_manager.table_id}' should exist due to RDM init.")
        
        test_data = {
            "experiment_id": unique_experiment_id,
            "timestamp": datetime.utcnow().isoformat(),
            "padres_success": True,
            "claude_analysis": "Direct BQ test analysis.",
            "perplexity_research": "Direct BQ test research.",
            "score": 0.99,
            "distance": 1.0,
            "task_completed": True,
            "raw_data": {"source": "direct_test", "id": unique_experiment_id}
        }
        
        store_success = data_manager.store_experiment(test_data)
        if not store_success:
            record_test_result(test_name, False, "Failed to store dummy experiment directly to BigQuery.")
            return
        
        logger.info(f"Dummy experiment {unique_experiment_id} stored. Attempting to retrieve...")
        time.sleep(2) # Brief delay for consistency
        
        recent_experiments = data_manager.get_recent_experiments(days=1)
        found_in_bq = False
        retrieved_exp_data = None
        for exp in recent_experiments:
            if exp.get("experiment_id") == unique_experiment_id:
                found_in_bq = True
                retrieved_exp_data = exp
                break
        
        if found_in_bq:
            record_test_result(test_name, True, f"Successfully stored and retrieved experiment {unique_experiment_id} directly.")
            logger.debug(f"Retrieved data: {retrieved_exp_data}")
        else:
            record_test_result(test_name, False, f"Failed to retrieve experiment {unique_experiment_id} after direct store.")
            
    except Exception as e:
        record_test_result(test_name, False, f"Exception: {e}")

# --- Main Test Execution --- 
def main():
    logger.info("========== Starting Research Pipeline Test Suite ==========")
    
    # Load .env file for local development configuration
    # In a CI/CD or production test environment, these would be set directly.
    load_dotenv()
    logger.info(".env file loaded (if present). Ensure required env vars are set.")

    gcp_project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    perplexity_key = os.getenv("PERPLEXITY_API_KEY")

    if not gcp_project_id:
        logger.error("CRITICAL: GOOGLE_CLOUD_PROJECT environment variable not set. Aborting tests.")
        return
    if not anthropic_key:
        logger.warning("ANTHROPIC_API_KEY not set. Tests requiring Claude may fail or be skipped by underlying logic.")
    if not perplexity_key:
        logger.warning("PERPLEXITY_API_KEY not set. Tests requiring Perplexity may fail or be skipped.")

    # Initialize FastAPI TestClient
    # The `app` object is imported from your app.py
    try:
        client = TestClient(app)
        logger.info("FastAPI TestClient initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize FastAPI TestClient: {e}. API tests will be skipped.")
        client = None

    # Initialize ResearchDataManager
    try:
        data_manager = ResearchDataManager(project_id=gcp_project_id)
        logger.info("ResearchDataManager initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize ResearchDataManager: {e}. BigQuery tests will be skipped.")
        data_manager = None

    # --- Run Tests --- 
    if client:
        test_health_check(client)
        if data_manager: # Only run if data_manager is available
            test_run_experiment_and_bigquery_storage(client, data_manager)
        test_paper_generation_creates_file(client)
    else:
        logger.warning("Skipping API endpoint tests due to TestClient initialization failure.")

    if data_manager:
        test_direct_bigquery_operations(data_manager)
    else:
        logger.warning("Skipping direct BigQuery tests due to ResearchDataManager initialization failure.")

    # --- Test Summary --- 
    logger.info("========== Test Suite Summary ==========")
    passed_count = sum(1 for r in TEST_RESULTS if r["status"] == "PASS")
    failed_count = len(TEST_RESULTS) - passed_count
    
    for result in TEST_RESULTS:
        logger.info(f"{result['name']}: {result['status']} - {result['message']}")
        
    logger.info(f"Total tests run: {len(TEST_RESULTS)}")
    logger.info(f"Passed: {passed_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info("=========================================")

    if failed_count > 0:
        exit(1) # Exit with error code if any tests failed

if __name__ == "__main__":
    main() 