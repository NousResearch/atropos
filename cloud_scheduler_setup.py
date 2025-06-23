import json
import logging
import os

import google.auth  # Import the google.auth module
from google.api_core.exceptions import AlreadyExists, GoogleAPICallError, NotFound
from google.cloud import scheduler_v1
from google.cloud.scheduler_v1.types import HttpTarget, Job, OidcToken, RetryConfig

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CloudSchedulerManager:
    """Manages Cloud Scheduler jobs for the research pipeline."""

    def __init__(
        self,
        project_id: str,
        region: str,
        cloud_run_service_url: str,
        scheduler_sa_email: str,
    ):
        """
        Initializes the CloudSchedulerManager.
        Args:
            project_id: Your Google Cloud Project ID.
            region: The GCP region where the scheduler jobs will be created (e.g., 'us-central1').
            cloud_run_service_url: The HTTPS URL of your deployed Cloud Run service.
            scheduler_sa_email: The email of the service account Cloud Scheduler will use to invoke Cloud Run.
        """
        if not all([project_id, region, cloud_run_service_url, scheduler_sa_email]):
            msg = "Project ID, region, Cloud Run URL, and Scheduler SA email are required."
            logger.error(f"CRITICAL: {msg}")
            raise ValueError(msg)

        self.project_id = project_id
        self.location_path = f"projects/{project_id}/locations/{region}"
        self.cloud_run_service_url = cloud_run_service_url.rstrip(
            "/"
        )  # Ensure no trailing slash
        self.scheduler_sa_email = scheduler_sa_email
        try:
            # Explicitly get credentials with the necessary cloud-platform scope
            credentials, adc_project_id_detected = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )

            # Log which project ADC detected, but the script will use the passed `project_id` for constructing paths.
            logger.info(
                f"ADC detected project: {adc_project_id_detected}. Operations will use configured project: {self.project_id}"
            )

            # Initialize the client with these explicit credentials.
            self.client = scheduler_v1.CloudSchedulerClient(credentials=credentials)
            logger.info(
                "CloudSchedulerClient initialized successfully with explicitly scoped ADC."
            )
        except Exception as e:
            logger.error(
                f"Failed to initialize CloudSchedulerClient: {e}", exc_info=True
            )
            raise

    def _construct_job_path(self, job_id: str) -> str:
        return f"{self.location_path}/jobs/{job_id}"

    def create_or_update_http_job(
        self,
        job_id: str,
        description: str,
        schedule: str,  # Cron expression
        target_endpoint: str,  # e.g., "/run-experiment"
        http_method_str: str = "POST",  # Changed to string and renamed
        body: dict = None,
        time_zone: str = "Etc/UTC",
        attempt_deadline: str = "180s",  # e.g., "300s"
        retry_max_attempts: int = 3,
        retry_min_backoff: str = "5s",
        retry_max_backoff: str = "3600s",
        retry_max_doublings: int = 5,
    ):
        """
        Creates a new Cloud Scheduler job or updates it if it already exists.
        The job targets an HTTP endpoint (e.g., a Cloud Run service).
        """
        job_name = self._construct_job_path(job_id)
        target_uri = f"{self.cloud_run_service_url}{target_endpoint}"

        job_body_bytes = None
        headers = {"Content-Type": "application/json"}
        if body:
            try:
                job_body_bytes = json.dumps(body).encode("utf-8")
            except TypeError as e:
                logger.error("Error serializing body for job "{job_id}': {e}")
                raise

        # Ensure http_method_str is one of the supported string values if needed by the API
        # For HttpTarget, it directly takes the string like "POST", "GET"
        http_target = HttpTarget(
            uri=target_uri,
            http_method=http_method_str,  # Use the string directly
            headers=headers,
            body=job_body_bytes,
            oidc_token=OidcToken(service_account_email=self.scheduler_sa_email),
        )

        # Correctly instantiate RetryConfig
        retry_config_obj = RetryConfig(
            retry_count=retry_max_attempts,
            min_backoff_duration={"seconds": int(retry_min_backoff.rstrip("s"))},
            max_backoff_duration={"seconds": int(retry_max_backoff.rstrip("s"))},
            max_doublings=retry_max_doublings,
        )

        job_obj = Job(
            name=job_name,  # Will be used for update if job exists
            description=description,
            schedule=schedule,
            time_zone=time_zone,
            http_target=http_target,
            attempt_deadline={
                "seconds": int(attempt_deadline.rstrip("s"))
            },  # Duration string to seconds
            retry_config=retry_config_obj,  # Use the instantiated RetryConfig object
        )

        try:
            # Check if job exists to decide between create and update
            self.client.get_job(name=job_name)
            logger.info("Job "{job_id}' already exists. Attempting to update...")
            request = scheduler_v1.UpdateJobRequest(job=job_obj)
            updated_job = self.client.update_job(request=request)
            logger.info(f"Successfully updated job: {updated_job.name}")
            return updated_job
        except NotFound:
            logger.info("Job "{job_id}' not found. Attempting to create...")
            try:
                created_job = self.client.create_job(
                    parent=self.location_path, job=job_obj
                )
                logger.info(f"Successfully created job: {created_job.name}")
                return created_job
            except AlreadyExists:
                # This can happen in a race condition if another process created it
                logger.warning(
                    "Job "{job_id}' was created concurrently. Attempting to get and update."
                )
                # Fallback to update logic might be needed or just log and proceed
                return self.client.get_job(name=job_name)  # Or re-attempt update
            except GoogleAPICallError as e:
                logger.error("API error creating job "{job_id}': {e}", exc_info=True)
                raise
        except GoogleAPICallError as e:
            logger.error(
                "API error interacting with job "{job_id}': {e}", exc_info=True
            )
            raise
        except Exception as e:
            logger.error("Unexpected error with job "{job_id}': {e}", exc_info=True)
            raise

    def delete_job(self, job_id: str):
        """Deletes a Cloud Scheduler job."""
        job_name = self._construct_job_path(job_id)
        try:
            self.client.delete_job(name=job_name)
            logger.info(f"Successfully deleted job: {job_name}")
            return True
        except NotFound:
            logger.warning("Job "{job_id}' not found. Nothing to delete.")
            return False
        except GoogleAPICallError as e:
            logger.error("API error deleting job "{job_id}': {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(
                "Unexpected error deleting job "{job_id}': {e}", exc_info=True
            )
            raise

    def list_jobs(self) -> list:
        """Lists all Cloud Scheduler jobs in the configured location."""
        try:
            jobs_iterator = self.client.list_jobs(parent=self.location_path)
            jobs = list(jobs_iterator)
            logger.info(f"Found {len(jobs)} jobs in {self.location_path}.")
            return jobs
        except Exception as e:
            logger.error(f"Failed to list jobs: {e}", exc_info=True)
            return []  # Return empty list on error

    def setup_research_pipeline_jobs(self):
        """Creates or updates all standard jobs for the research pipeline."""
        logger.info("Setting up standard research pipeline Cloud Scheduler jobs...")

        # Job 1: Run Experiment Batch (every 2 hours)
        run_experiment_job_id = "run-spatial-experiments-batch"
        run_experiment_description = "Run spatial AI experiments batch (10 experiments) every 2 hours and store results."
        run_experiment_schedule = (
            "0 */2 * * *"  # Every 2 hours at the start of the hour
        )
        run_experiment_endpoint = "/run-experiment"
        run_experiment_body = {"batch_size": 10}  # DOUBLED BATCH SIZE HERE

        self.create_or_update_http_job(
            job_id=run_experiment_job_id,
            description=run_experiment_description,
            schedule=run_experiment_schedule,
            target_endpoint=run_experiment_endpoint,
            body=run_experiment_body,
            attempt_deadline="1500s",  # Increased to 25 minutes for larger batch
        )

        # Job 2: Generate Weekly Paper (every Sunday at 10:00 UTC)
        generate_paper_job_id = "generate-weekly-research-paper"
        generate_paper_description = "Generate the weekly research paper every Sunday."
        generate_paper_schedule = "0 10 * * 0"  # At 10:00 on Sunday
        generate_paper_endpoint = "/generate-paper"

        self.create_or_update_http_job(
            job_id=generate_paper_job_id,
            description=generate_paper_description,
            schedule=generate_paper_schedule,
            target_endpoint=generate_paper_endpoint,
            # http_method_str defaults to "POST", so no need to specify if POST
            attempt_deadline="1780s",  # Slightly less than a 30-min Cloud Run timeout
            retry_max_attempts=1,  # Paper generation might not be idempotent or cheap to retry often
        )

        logger.info("Standard research pipeline jobs setup process complete.")

    def delete_all_research_pipeline_jobs(self):
        """Deletes the standard jobs. Useful for cleanup."""
        logger.warning(
            "Attempting to delete standard research pipeline Cloud Scheduler jobs..."
        )
        # Add job_ids of jobs you want to delete
        job_ids_to_delete = [
            "run-spatial-experiments-batch",
            "generate-weekly-research-paper",
        ]
        for job_id in job_ids_to_delete:
            self.delete_job(job_id)
        logger.info("Deletion of standard jobs process complete.")


if __name__ == "__main__":
    # --- Configuration - IMPORTANT: Set these values ---
    try:
        # Load from environment variables (recommended for security and flexibility)
        GCP_PROJECT_ID = os.environ["GCP_PROJECT_ID"]
        GCP_REGION = os.environ.get("GCP_REGION", "us-central1")  # Default if not set

        # This is the HTTPS URL of your Cloud Run service that Cloud Scheduler will call.
        # Example: "https://your-service-name-abcdefghij-uc.a.run.app"
        CLOUD_RUN_INVOKE_URL = os.environ["CLOUD_RUN_INVOKE_URL"]

        # This is the service account Cloud Scheduler will use to authenticate to Cloud Run.
        # It needs the "Cloud Run Invoker" role on your Cloud Run service.
        # Format: your-scheduler-sa-name@your-project-id.iam.gserviceaccount.com
        # Often, this is the default Cloud Scheduler SA or a custom one you created.
        SCHEDULER_SERVICE_ACCOUNT_EMAIL = os.environ["SCHEDULER_SERVICE_ACCOUNT_EMAIL"]

        logger.info(f"Using GCP_PROJECT_ID: {GCP_PROJECT_ID}")
        logger.info(f"Using GCP_REGION: {GCP_REGION}")
        logger.info(f"Using CLOUD_RUN_INVOKE_URL: {CLOUD_RUN_INVOKE_URL}")
        logger.info(
            f"Using SCHEDULER_SERVICE_ACCOUNT_EMAIL: {SCHEDULER_SERVICE_ACCOUNT_EMAIL}"
        )

    except KeyError as e:
        logger.error(
            f"CRITICAL: Environment variable {e} not set. This script requires: "
            "GCP_PROJECT_ID, CLOUD_RUN_INVOKE_URL, SCHEDULER_SERVICE_ACCOUNT_EMAIL."
        )
        logger.error("Optionally, GCP_REGION (defaults to us-central1).")
        logger.error("Please set these environment variables before running.")
        logger.error('Example: export GCP_PROJECT_ID="your-project"')
        exit(1)

    # --- Script Actions ---
    manager = CloudSchedulerManager(
        project_id=GCP_PROJECT_ID,
        region=GCP_REGION,
        cloud_run_service_url=CLOUD_RUN_INVOKE_URL,
        scheduler_sa_email=SCHEDULER_SERVICE_ACCOUNT_EMAIL,
    )

    # To setup/update the jobs:
    manager.setup_research_pipeline_jobs()

    # To list jobs:
    # current_jobs = manager.list_jobs()
    # for job in current_jobs:
    #     logger.info(f"Found job: {job.name}, Schedule: {job.schedule}, Target: {job.http_target.uri if job.http_target else 'N/A'}")

    # To delete a specific job (example):
    # manager.delete_job("run-spatial-experiments-batch")

    # To delete all standard jobs defined in the script:
    # manager.delete_all_research_pipeline_jobs()

    logger.info("Cloud Scheduler setup script finished.")
