# Production Readiness Verification Checklist

This checklist is to ensure all components of the 24/7 AI Research Pipeline are correctly configured, integrated, and ready for production deployment and operation.

## I. Code & Dependencies

- [ ] **All Python dependencies listed in `requirements.txt`?**
    - `fastapi`, `uvicorn[standard]`, `gunicorn`
    - `python-dotenv`
    - `google-cloud-bigquery`
    - `google-cloud-scheduler` (for `cloud_scheduler_setup.py` script)
    - `google-cloud-secret-manager` (if interacting directly, though `deploy.sh` handles setup)
    - `requests`, `anthropic` (and any other libraries used by `SimplePadresResearch` or other core logic)
- [ ] **Imports between Python files are correct and resolve?**
    - `app.py` -> `production_research_pipeline`
    - `production_research_pipeline.py` -> `enhanced_padres_perplexity`, `bigquery_manager`, `paper_generator`
    - `paper_generator.py` -> `bigquery_manager`, `enhanced_padres_perplexity`
    - `test_pipeline.py` -> `app`, `bigquery_manager`
    - `cloud_scheduler_setup.py` -> `google.cloud.scheduler_v1`
- [ ] **`enhanced_padres_perplexity.py` (contains `SimplePadresResearch`) exists and is correctly named/imported?**
- [ ] **Error handling is comprehensive in all Python components?** (e.g., `try-except` blocks, logging, appropriate responses/exceptions)

## II. Environment & Configuration

- [ ] **Environment variables consistently named and used?**
    - `GOOGLE_CLOUD_PROJECT`
    - `ANTHROPIC_API_KEY`
    - `PERPLEXITY_API_KEY`
    - `CLOUD_RUN_INVOKE_URL` (for `cloud_scheduler_setup.py`)
    - `SCHEDULER_SERVICE_ACCOUNT_EMAIL` (for `cloud_scheduler_setup.py`)
- [ ] **Local Development: `.env` file template provided and `python-dotenv` used correctly?**
- [ ] **Production: Secrets are passed as environment variables to Cloud Run (not baked into Docker image)?**

## III. Docker & Containerization

- [ ] **`Dockerfile` is secure and efficient?**
    - Uses a specific Python version (e.g., `python:3.11-slim`).
    - Runs as a non-root user.
    - Copies only necessary files.
    - Does NOT copy `.env` or other secrets.
    - `CMD` correctly starts Gunicorn with Uvicorn workers (e.g., `CMD exec gunicorn --bind 0.0.0.0:$PORT ... app:app`).
    - `PORT` environment variable (default 8080) is exposed and used.
- [ ] **Docker image builds successfully using `deploy.sh` or `gcloud builds submit`?**

## IV. Google Cloud Platform Setup (via `deploy.sh` and manual checks)

- [ ] **Required GCP APIs enabled?** (Cloud Run, IAM, Secret Manager, Artifact Registry, Cloud Build, BigQuery).
- [ ] **Artifact Registry repository created for Docker images?**
- [ ] **Dedicated Service Account (`${SERVICE_NAME}-sa`) created for Cloud Run?**
- [ ] **Cloud Run Service Account has necessary IAM permissions?**
    - `roles/secretmanager.secretAccessor` (on the project or specific secrets).
    - `roles/bigquery.dataEditor`.
    - `roles/bigquery.user`.
    - (Optional) `roles/storage.objectAdmin` if GCS is used by `paper_generator.py`.
- [ ] **Secrets (`ANTHROPIC_API_KEY`, `PERPLEXITY_API_KEY`) created in Secret Manager?**
- [ ] **Cloud Run service account granted access to these specific secrets?**
- [ ] **BigQuery Dataset (`spatial_research_data`) creation handled?** (by `bigquery_manager.py` or `deploy.sh`)
- [ ] **BigQuery Table (`experiments`) schema matches data being stored by `bigquery_manager.py`?**
    - `experiment_id` (STRING, REQUIRED)
    - `timestamp` (TIMESTAMP, REQUIRED)
    - `padres_success` (BOOLEAN)
    - `claude_analysis` (STRING)
    - `perplexity_research` (STRING)
    - `score` (FLOAT)
    - `distance` (FLOAT)
    - `task_completed` (BOOLEAN)
    - `raw_data` (JSON)

## V. Deployment (`deploy.sh` script)

- [ ] **`deploy.sh` script variables are correctly configured (PROJECT_ID, REGION, names)?**
- [ ] **Cloud Run service deploys successfully using `deploy.sh`?**
    - Image is correctly specified from Artifact Registry.
    - Service account is assigned.
    - `--no-allow-unauthenticated` is set.
    - Memory (`2Gi`), CPU (`2`), and timeout (`1800s`) are configured.
    - Environment variables (GCP_PROJECT and secrets) are correctly injected.

## VI. Scheduling (`cloud_scheduler_setup.py` script)

- [ ] **`cloud_scheduler_setup.py` environment variables correctly set before running?** (`GCP_PROJECT_ID`, `CLOUD_RUN_INVOKE_URL`, `SCHEDULER_SERVICE_ACCOUNT_EMAIL`).
- [ ] **Cloud Scheduler Service Account (`SCHEDULER_SERVICE_ACCOUNT_EMAIL`) has `roles/run.invoker` permission on the deployed Cloud Run service?**
- [ ] **Scheduler jobs are created or updated successfully by `cloud_scheduler_setup.py`?**
    - Experiment job: `0 */2 * * *`, target `/run-experiment` (POST), OIDC token auth.
    - Paper generation job: `0 10 * * 0`, target `/generate-paper` (POST), OIDC token auth.
- [ ] **Scheduler job retry policies are configured as intended?**

## VII. Testing (`test_pipeline.py` script)

- [ ] **All tests in `test_pipeline.py` pass locally (using `.env` for secrets)?**
    - `/health` endpoint responds correctly.
    - `/run-experiment` completes, and data appears in BigQuery.
    - `/generate-paper` creates a paper file, and the file is cleaned up.
    - Direct BigQuery operations (store/retrieve) work.
- [ ] **(Post-Deployment) Tests pass when run against the deployed Cloud Run service?** (This might require a separate test script or configuration to target the deployed URL and handle OIDC authentication if testing authenticated endpoints).

## VIII. Monitoring & Operations

- [ ] **Logging is functional and provides adequate detail in Cloud Logging?**
- [ ] **Monitoring dashboards or alerts planned/configured for Cloud Run service (errors, latency, instance count)?**
- [ ] **BigQuery cost monitoring strategy in place?**
- [ ] **Procedure for updating API keys in Secret Manager understood?**
- [ ] **Rollback plan or strategy if a deployment introduces issues?**

## IX. Documentation

- [ ] **`README.md` is comprehensive and guides users through setup and deployment?**
- [ ] **`project_structure.md` accurately reflects the current file organization?** 