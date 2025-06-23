import json
import logging
import os
from datetime import datetime

from google.api_core.exceptions import NotFound
from google.cloud import storage

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ResearchDataManager:
    """Manages storage and retrieval of research experiment data in Google Cloud Storage."""

    def __init__(self, project_id: str, bucket_name: str | None = None):
        """
        Initializes the ResearchDataManager.
        Args:
            project_id: The Google Cloud Project ID.
            bucket_name: The Google Cloud Storage bucket name. Defaults to {project_id}-research-data.
        """
        if not project_id:
            logger.error("Google Cloud Project ID is required.")
            raise ValueError("Google Cloud Project ID cannot be null or empty.")

        self.project_id = project_id
        try:
            self.client = storage.Client(project=self.project_id)
            logger.info(
                "Google Cloud Storage client initialized for project "{self.project_id}'."
            )
        except Exception as e:
            logger.error(
                f"Failed to initialize Google Cloud Storage client: {e}", exc_info=True
            )
            raise

        if bucket_name:
            self.bucket_name = bucket_name
        else:
            self.bucket_name = f"{project_id}-research-data"

        self.bucket = self._get_or_create_bucket(self.bucket_name)

    def _get_or_create_bucket(self, bucket_name: str) -> storage.Bucket:
        """Gets or creates a GCS bucket."""
        try:
            bucket = self.client.get_bucket(bucket_name)
            logger.info("Bucket "{bucket_name}' already exists.")
        except NotFound:
            logger.info("Bucket "{bucket_name}' not found. Creating new bucket...")
            try:
                bucket = self.client.create_bucket(bucket_name)
                logger.info("Bucket "{bucket_name}' created successfully.")
            except Exception as e:
                logger.error(
                    "Failed to create bucket "{bucket_name}': {e}", exc_info=True
                )
                raise
        except Exception as e:
            logger.error(
                "Error checking for bucket "{bucket_name}': {e}", exc_info=True
            )
            raise
        return bucket

    def store_experiment(self, experiment_data: dict):
        """
        Stores a single experiment data dictionary into a GCS bucket.
        Data is stored in a JSONL file named by the current date (YYYY-MM-DD.jsonl)
        within an 'experiments' folder.
        """
        try:
            today_date = datetime.utcnow().strftime("%Y-%m-%d")
            blob_name = f"experiments/{today_date}.jsonl"
            blob = self.bucket.blob(blob_name)

            json_string = json.dumps(experiment_data)

            if blob.exists():
                existing_content = blob.download_as_text()
                # Ensure there's a newline before appending if content exists
                if existing_content and not existing_content.endswith("\n"):
                    new_content = existing_content + "\n" + json_string
                else:
                    new_content = (
                        existing_content or ""
                    ) + json_string  # Handle case where file exists but is empty
                blob.upload_from_string(new_content, content_type="application/jsonl")
                # logger.debug(f"Appended experiment to gs://{self.bucket_name}/{blob_name}")
            else:
                blob.upload_from_string(json_string, content_type="application/jsonl")
                # logger.debug(f"Created new file and stored experiment to gs://{self.bucket_name}/{blob_name}")

            logger.info(
                "Successfully stored experiment "{experiment_data.get('experiment_id', 'N/A')}' to GCS: gs://{self.bucket_name}/{blob_name}"
            )

        except Exception as e:
            logger.error(
                f"Error storing experiment data to GCS (experiment_id: {experiment_data.get('experiment_id', 'N/A')}): {e}",
                exc_info=True,
            )
            # Depending on requirements, you might want to raise the exception
            # or implement a retry mechanism here.

    def get_all_experiments_for_paper(self) -> list[dict]:
        """
        Retrieves all experiment data from GCS for paper generation.
        This method will list all blobs in the 'experiments/' prefix,
        download them, parse the JSONL content, and return a list of experiment dicts.
        """
        all_experiments = []
        try:
            logger.info(
                "Fetching all experiments from GCS bucket "{self.bucket_name}', prefix 'experiments/'."
            )
            blobs = self.client.list_blobs(self.bucket_name, prefix="experiments/")
            for blob in blobs:
                if (
                    blob.name.endswith(".jsonl") and blob.size > 0
                ):  # Ensure it's a JSONL and not empty
                    logger.info(
                        f"Processing data from gs://{self.bucket_name}/{blob.name}..."
                    )
                    try:
                        jsonl_content = blob.download_as_text()
                        for line_number, line in enumerate(
                            jsonl_content.strip().split("\n")
                        ):
                            if line:  # ensure not an empty line after split
                                try:
                                    experiment = json.loads(line)
                                    all_experiments.append(experiment)
                                except json.JSONDecodeError as e:
                                    logger.warning(
                                        f"Skipping line {line_number + 1} due to JSON decode error in '{blob.name}': {e} - Line content (first 100 chars): '{line[:100]}...'"
                                    )
                    except Exception as e_blob:
                        logger.error(
                            "Error processing blob "{blob.name}': {e_blob}",
                            exc_info=True,
                        )
                elif not blob.name.endswith(".jsonl"):
                    logger.debug(f"Skipping non-JSONL file: {blob.name}")
                elif blob.size == 0:
                    logger.debug(f"Skipping empty file: {blob.name}")

            logger.info(
                f"Successfully retrieved {len(all_experiments)} experiments from GCS bucket '{self.bucket_name}'."
            )
            return all_experiments
        except Exception as e:
            logger.error(
                f"Error retrieving experiment data from GCS for paper generation: {e}",
                exc_info=True,
            )
            return []  # Return empty list on error


# Example Usage (optional, for local testing if this file is run directly)
if __name__ == "__main__":
    # This local test block requires GOOGLE_APPLICATION_CREDENTIALS to be set,
    # or to be run in an environment with default credentials (e.g., gcloud auth application-default login).
    # It also requires the GOOGLE_CLOUD_PROJECT environment variable to be set.

    gcp_project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not gcp_project_id:
        print(
            "GOOGLE_CLOUD_PROJECT environment variable not set. Please set it to run local tests."
        )
        # logger.error("GOOGLE_CLOUD_PROJECT environment variable not set. Please set it to run local tests.")
    else:
        logger.info(
            f"--- Running ResearchDataManager (GCS) local test for project: {gcp_project_id} ---"
        )

        # You can specify a custom bucket name for testing, or it will use the default.
        # test_bucket_name = f"{gcp_project_id}-research-data-test"
        # data_manager = ResearchDataManager(project_id=gcp_project_id, bucket_name=test_bucket_name)
        data_manager = ResearchDataManager(
            project_id=gcp_project_id
        )  # Uses default bucket
        logger.info(f"Using GCS Bucket: {data_manager.bucket_name}")

        # 1. Test store_experiment
        logger.info("\n--- Testing store_experiment ---")
        timestamp_now = datetime.utcnow().isoformat()
        test_experiment_1 = {
            "experiment_id": f"test_gcs_{int(datetime.utcnow().timestamp())}_001",
            "timestamp": timestamp_now,
            "params": {"size": 10, "shape": "circle", "attempt": 1},
            "padres_output": {"raw_sim": "simulation data for circle..."},
            "llm_analysis": {
                "summary": "Circle experiment looks good.",
                "reasoning_quality": "high",
            },
            "perplexity_context": {"related_studies": ["studyA", "studyB"]},
        }
        test_experiment_2 = {
            "experiment_id": f"test_gcs_{int(datetime.utcnow().timestamp())}_002",
            "timestamp": timestamp_now,  # Same day, should go to same file
            "params": {"size": 5, "shape": "square", "attempt": 1},
            "padres_output": {"raw_sim": "simulation data for square..."},
            "llm_analysis": {
                "summary": "Square experiment is interesting.",
                "reasoning_quality": "medium",
            },
            "perplexity_context": {"related_studies": ["studyC"]},
        }

        data_manager.store_experiment(test_experiment_1)
        data_manager.store_experiment(test_experiment_2)
        logger.info("Store experiment tests: Two experiments sent to GCS.")
        logger.info(
            f"Check your GCS bucket: gs://{data_manager.bucket_name}/experiments/{datetime.utcnow().strftime('%Y-%m-%d')}.jsonl"
        )

        # 2. Test get_all_experiments_for_paper
        logger.info("\n--- Testing get_all_experiments_for_paper ---")
        # Add a small delay to allow GCS to be eventually consistent if running immediately after writes
        # For local tests this might not be strictly necessary but good practice for cloud services
        import time

        time.sleep(2)

        retrieved_experiments = data_manager.get_all_experiments_for_paper()

        if retrieved_experiments:
            logger.info(
                f"Successfully retrieved {len(retrieved_experiments)} experiments from GCS."
            )
            # Print details of a few retrieved experiments for verification
            for i, exp in enumerate(retrieved_experiments[:3]):  # Print first 3
                logger.info(
                    f"  Retrieved Experiment {i+1}: ID '{exp.get('experiment_id', 'N/A')}', LLM Summary: '{exp.get('llm_analysis', {}).get('summary', 'N/A')}'"
                )
            if len(retrieved_experiments) > 3:
                logger.info(
                    f"  ... and {len(retrieved_experiments) - 3} more experiments."
                )
        else:
            logger.warning(
                "No experiments retrieved or an error occurred during retrieval from GCS."
            )

        logger.info("--- ResearchDataManager (GCS) local test finished ---")
