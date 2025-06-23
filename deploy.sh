#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Attempt to locate gcloud command ---
GCLOUD_CMD="gcloud"
if ! command -v gcloud &> /dev/null; then
    echo "WARN: 'gcloud' command not found in PATH. Attempting common locations..."
    COMMON_GCLOUD_PATHS=(
        "/usr/local/Caskroom/google-cloud-sdk/latest/google-cloud-sdk/bin/gcloud" # Common for Homebrew on macOS via Cask
        "~/google-cloud-sdk/bin/gcloud" # Common user installation
        "/snap/bin/gcloud" # Common for Snap on Linux
    )
    for path_attempt in "${COMMON_GCLOUD_PATHS[@]}"; do
        expanded_path=$(eval echo "${path_attempt}") # Expand ~ if present
        if [ -x "${expanded_path}" ]; then
            echo "INFO: Found gcloud at ${expanded_path}"
            GCLOUD_CMD="${expanded_path}"
            break
        fi
    done
    if [ "${GCLOUD_CMD}" == "gcloud" ]; then # Still not found
        echo "ERROR: 'gcloud' command could not be found. Please ensure the Google Cloud SDK is installed and its bin directory is in your PATH."
        exit 1
    fi
fi

# --- Configuration - Adjust these variables as needed ---
# GCP Project ID - IMPORTANT: Replace with your actual project ID or ensure it's set
PROJECT_ID="$(${GCLOUD_CMD} config get-value project 2>/dev/null)"
if [ -z "${PROJECT_ID}" ]; then
    echo "ERROR: GCP Project ID not set. Please set it using '${GCLOUD_CMD} config set project YOUR_PROJECT_ID' or hardcode it in the script."
    exit 1
fi

REGION="us-central1"  # Specific region for Cloud Run and Artifact Registry
# LOCATION_FOR_ARTIFACTS_AND_BQ="US" # Multi-region for BigQuery dataset, region for Artifact Registry
SERVICE_NAME="spatial-research-pipeline"
IMAGE_NAME="${SERVICE_NAME}-image"
ARTIFACT_REGISTRY_REPO="${SERVICE_NAME}-repo"
# SERVICE_URL for spatial-research-pipeline will be fetched after its deployment

SERVICE_ACCOUNT_NAME="${SERVICE_NAME}-sa"
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

CUSTOM_SCHEDULER_SA_NAME="custom-scheduler-sa"
CUSTOM_SCHEDULER_SA_EMAIL="${CUSTOM_SCHEDULER_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

# SECRET_ANTHROPIC_KEY_NAME="anthropic-api-key" # No longer needed
SECRET_PERPLEXITY_KEY_NAME="perplexity-api-key"
SECRET_GEMINI_KEY_NAME="gemini-api-key" # New secret for Gemini

# GCS Bucket for storing generated papers
PAPER_GCS_BUCKET_NAME="${PROJECT_ID}-research-papers"
# GCS Bucket for storing research data (JSONL files)
DATA_GCS_BUCKET_NAME="${PROJECT_ID}-research-data"

PADRES_API_SERVICE_URL="https://padres-api-service-312425595703.us-central1.run.app" # This is the URL for your separately deployed Padres API service

# --- Helper Functions ---
ask_for_api_key() {
    local key_name="$1"
    local env_var_name="$2"
    local key_value
    read -s -p "Enter your ${key_name} (will not be shown): " key_value
    echo
    if [ -z "${key_value}" ]; then
        echo "No ${key_name} provided. Skipping creation/update in Secret Manager for ${key_name}."
        return 1
    fi
    export "${env_var_name}"="${key_value}"
    return 0
}

# --- 1. Enable Necessary Google Cloud APIs ---
echo "Phase 1: Enabling Google Cloud APIs..."
"${GCLOUD_CMD}" services enable \
    run.googleapis.com \
    iam.googleapis.com \
    secretmanager.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    cloudscheduler.googleapis.com \
    generativelanguage.googleapis.com \
    storage.googleapis.com --project="${PROJECT_ID}"

echo "Phase 1b: Ensuring GCS Bucket for papers exists (gs://${PAPER_GCS_BUCKET_NAME}/)..."
if ! gsutil ls -b "gs://${PAPER_GCS_BUCKET_NAME}/" &>/dev/null; then
    echo "Bucket gs://${PAPER_GCS_BUCKET_NAME}/ not found. Creating bucket..."
    gsutil mb -p "${PROJECT_ID}" -l "${REGION}" "gs://${PAPER_GCS_BUCKET_NAME}/"
else
    echo "GCS Bucket gs://${PAPER_GCS_BUCKET_NAME}/ already exists."
fi

echo "Phase 1c: Ensuring GCS Bucket for research data exists (gs://${DATA_GCS_BUCKET_NAME}/)..."
if ! gsutil ls -b "gs://${DATA_GCS_BUCKET_NAME}/" &>/dev/null; then
    echo "Bucket gs://${DATA_GCS_BUCKET_NAME}/ not found. Creating bucket..."
    gsutil mb -p "${PROJECT_ID}" -l "${REGION}" "gs://${DATA_GCS_BUCKET_NAME}/"
else
    echo "GCS Bucket gs://${DATA_GCS_BUCKET_NAME}/ already exists."
fi

# --- 2. Create Artifact Registry Docker Repository (if it doesn't exist) ---
echo "Phase 2: Setting up Artifact Registry Docker repository..."
if ! "${GCLOUD_CMD}" artifacts repositories describe "${ARTIFACT_REGISTRY_REPO}" --location="${REGION}" --project="${PROJECT_ID}" &> /dev/null; then
    echo "Creating Artifact Registry repository: ${ARTIFACT_REGISTRY_REPO} in ${REGION}..."
    "${GCLOUD_CMD}" artifacts repositories create "${ARTIFACT_REGISTRY_REPO}" \
        --repository-format=docker \
        --location="${REGION}" \
        --description="Docker repository for ${SERVICE_NAME}" \
        --project="${PROJECT_ID}"
else
    echo "Artifact Registry repository '${ARTIFACT_REGISTRY_REPO}' already exists in ${REGION}."
fi

# --- 3. Create a Dedicated Service Account (if it doesn't exist) ---
echo "Phase 3: Setting up Service Account for Cloud Run service..."
if ! "${GCLOUD_CMD}" iam service-accounts describe "${SERVICE_ACCOUNT_EMAIL}" --project="${PROJECT_ID}" &> /dev/null; then
    echo "Creating service account: ${SERVICE_ACCOUNT_NAME}..."
    "${GCLOUD_CMD}" iam service-accounts create "${SERVICE_ACCOUNT_NAME}" \
        --display-name="Service Account for ${SERVICE_NAME}" \
        --project="${PROJECT_ID}"
else
    echo "Service account '${SERVICE_ACCOUNT_NAME}' already exists."
fi

# --- 3b. Create Custom Service Account for Cloud Scheduler (if it doesn't exist) ---
echo "Phase 3b: Setting up Custom Service Account for Cloud Scheduler..."
if ! "${GCLOUD_CMD}" iam service-accounts describe "${CUSTOM_SCHEDULER_SA_EMAIL}" --project="${PROJECT_ID}" &> /dev/null; then
    echo "Creating custom service account: ${CUSTOM_SCHEDULER_SA_NAME}..."
    "${GCLOUD_CMD}" iam service-accounts create "${CUSTOM_SCHEDULER_SA_NAME}" \
        --display-name="Custom SA for Cloud Scheduler" \
        --project="${PROJECT_ID}"
else
    echo "Custom Service account '${CUSTOM_SCHEDULER_SA_NAME}' already exists."
fi

# --- 4. Grant IAM Permissions to the Service Account ---
echo "Phase 4: Granting IAM permissions to the service account..."
"${GCLOUD_CMD}" projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/secretmanager.secretAccessor" \
    --condition=None > /dev/null

# Grant Cloud Run SA permission to write to the GCS buckets
gsutil iam ch "serviceAccount:${SERVICE_ACCOUNT_EMAIL}:objectAdmin" "gs://${PAPER_GCS_BUCKET_NAME}/" >/dev/null
gsutil iam ch "serviceAccount:${SERVICE_ACCOUNT_EMAIL}:objectAdmin" "gs://${DATA_GCS_BUCKET_NAME}/" >/dev/null
echo "Granted GCS permissions to Cloud Run SA '${SERVICE_ACCOUNT_EMAIL}' on buckets."

# New addition: Explicit bucket reader/writer for the data bucket
echo "Granting explicit GCS legacyBucketReader and legacyBucketWriter access to Cloud Run SA for data bucket gs://${PROJECT_ID}-research-data/..."
gsutil iam ch \
    serviceAccount:${SERVICE_ACCOUNT_EMAIL}:roles/storage.legacyBucketReader \
    serviceAccount:${SERVICE_ACCOUNT_EMAIL}:roles/storage.legacyBucketWriter \
    gs://${PROJECT_ID}-research-data/

CURRENT_USER=$("${GCLOUD_CMD}" config get-value account)
"${GCLOUD_CMD}" iam service-accounts add-iam-policy-binding "${CUSTOM_SCHEDULER_SA_EMAIL}" \
    --member="user:${CURRENT_USER}" \
    --role="roles/iam.serviceAccountUser" \
    --project="${PROJECT_ID}" > /dev/null

"${GCLOUD_CMD}" projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="user:${CURRENT_USER}" \
    --role="roles/cloudscheduler.admin" \
    --condition=None > /dev/null

echo "Service account permissions updated/verified."

# --- 5. Create Secrets in Secret Manager (if they don't exist) ---
echo "Phase 5: Setting up secrets in Secret Manager..."
# Gemini API Key
if ! "${GCLOUD_CMD}" secrets describe "${SECRET_GEMINI_KEY_NAME}" --project="${PROJECT_ID}" &> /dev/null; then
    echo "Secret '${SECRET_GEMINI_KEY_NAME}' not found."
    if ask_for_api_key "Gemini API Key" "GEMINI_KEY_VALUE"; then
        echo "Creating secret '${SECRET_GEMINI_KEY_NAME}'..."
        echo -n "${GEMINI_KEY_VALUE}" | "${GCLOUD_CMD}" secrets create "${SECRET_GEMINI_KEY_NAME}" \
            --data-file=- \
            --replication-policy=automatic \
            --project="${PROJECT_ID}"
        "${GCLOUD_CMD}" secrets add-iam-policy-binding "${SECRET_GEMINI_KEY_NAME}" \
            --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
            --role="roles/secretmanager.secretAccessor" \
            --project="${PROJECT_ID}" --condition=None > /dev/null
    else
        echo "Skipping Gemini API Key secret creation as no key was provided."
    fi
else
    echo "Secret '${SECRET_GEMINI_KEY_NAME}' already exists. To update, use '${GCLOUD_CMD} secrets versions add'."
fi

# Perplexity API Key
if ! "${GCLOUD_CMD}" secrets describe "${SECRET_PERPLEXITY_KEY_NAME}" --project="${PROJECT_ID}" &> /dev/null; then
    echo "Secret '${SECRET_PERPLEXITY_KEY_NAME}' not found."
    if ask_for_api_key "Perplexity API Key" "PERPLEXITY_KEY_VALUE"; then
        echo "Creating secret '${SECRET_PERPLEXITY_KEY_NAME}'..."
        echo -n "${PERPLEXITY_KEY_VALUE}" | "${GCLOUD_CMD}" secrets create "${SECRET_PERPLEXITY_KEY_NAME}" \
            --data-file=- \
            --replication-policy=automatic \
            --project="${PROJECT_ID}"
        "${GCLOUD_CMD}" secrets add-iam-policy-binding "${SECRET_PERPLEXITY_KEY_NAME}" \
            --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
            --role="roles/secretmanager.secretAccessor" \
            --project="${PROJECT_ID}" --condition=None > /dev/null
    fi
else
    echo "Secret '${SECRET_PERPLEXITY_KEY_NAME}' already exists. To update, use '${GCLOUD_CMD} secrets versions add'."
fi

# --- 6. Build Docker Image using Cloud Build and Push to Artifact Registry ---
echo "Phase 6: Building Docker image with Cloud Build and pushing to Artifact Registry..."
IMAGE_TAG="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REGISTRY_REPO}/${IMAGE_NAME}:latest"
"${GCLOUD_CMD}" builds submit . --tag="${IMAGE_TAG}" --project="${PROJECT_ID}"

# --- 7. Deploy to Cloud Run ---
echo "Phase 7: Deploying service to Cloud Run..."
DEPLOYED_SERVICE_URL=$("${GCLOUD_CMD}" run deploy "${SERVICE_NAME}" \
    --image="${IMAGE_TAG}" \
    --platform=managed \
    --region="${REGION}" \
    --service-account="${SERVICE_ACCOUNT_EMAIL}" \
    --no-allow-unauthenticated \
    --memory=2Gi \
    --cpu=2 \
    --timeout=1800s \
    --set-env-vars="GOOGLE_CLOUD_PROJECT=${PROJECT_ID},PADRES_API_URL=${PADRES_API_SERVICE_URL},PAPER_OUTPUT_GCS_BUCKET=${PAPER_GCS_BUCKET_NAME},DATA_GCS_BUCKET_NAME=${DATA_GCS_BUCKET_NAME}" \
    --set-secrets="PERPLEXITY_API_KEY=${SECRET_PERPLEXITY_KEY_NAME}:latest,GEMINI_API_KEY=${SECRET_GEMINI_KEY_NAME}:latest" \
    --project="${PROJECT_ID}" --format='value(status.url)')
echo "Cloud Run service deployed. URL: ${DEPLOYED_SERVICE_URL}"

# Grant CUSTOM_SCHEDULER_SA_EMAIL run.invoker role again AFTER service is deployed/updated to be sure
"${GCLOUD_CMD}" run services add-iam-policy-binding "${SERVICE_NAME}" --member="serviceAccount:${CUSTOM_SCHEDULER_SA_EMAIL}" --role="roles/run.invoker" --region="${REGION}" --platform=managed --project="${PROJECT_ID}" > /dev/null

# --- 8. Setting up Cloud Scheduler jobs using gcloud CLI ---
echo "Phase 8: Setting up Cloud Scheduler jobs using gcloud CLI..."

EXPERIMENT_JOB_ID="run-spatial-experiments-batch"
EXPERIMENT_JOB_DESCRIPTION="Run spatial AI experiments batch (10 experiments) every 2 hours (via gcloud)"
EXPERIMENT_SCHEDULE="0 */2 * * *"
EXPERIMENT_ENDPOINT="/run-experiment"
EXPERIMENT_BODY='{"batch_size": 10}'

# Check if job exists, then update or create
if "${GCLOUD_CMD}" scheduler jobs describe "${EXPERIMENT_JOB_ID}" --location="${REGION}" --project="${PROJECT_ID}" &>/dev/null; then
    echo "Updating existing experiment scheduler job: ${EXPERIMENT_JOB_ID}"
    "${GCLOUD_CMD}" scheduler jobs update http "${EXPERIMENT_JOB_ID}" \
        --schedule="${EXPERIMENT_SCHEDULE}" \
        --uri="${DEPLOYED_SERVICE_URL}${EXPERIMENT_ENDPOINT}" \
        --http-method=POST \
        --message-body="${EXPERIMENT_BODY}" \
        --oidc-service-account-email="${CUSTOM_SCHEDULER_SA_EMAIL}" \
        --oidc-token-audience="${DEPLOYED_SERVICE_URL}${EXPERIMENT_ENDPOINT}" \
        --location="${REGION}" \
        --time-zone="Etc/UTC" \
        --description="${EXPERIMENT_JOB_DESCRIPTION}" \
        --project="${PROJECT_ID}" \
        --attempt-deadline="1500s" \
        --max-retry-attempts=3 \
        --min-backoff=5s \
        --max-backoff=3600s \
        --max-doublings=5
else
    echo "Creating new experiment scheduler job: ${EXPERIMENT_JOB_ID}"
    "${GCLOUD_CMD}" scheduler jobs create http "${EXPERIMENT_JOB_ID}" \
        --schedule="${EXPERIMENT_SCHEDULE}" \
        --uri="${DEPLOYED_SERVICE_URL}${EXPERIMENT_ENDPOINT}" \
        --http-method=POST \
        --message-body="${EXPERIMENT_BODY}" \
        --oidc-service-account-email="${CUSTOM_SCHEDULER_SA_EMAIL}" \
        --oidc-token-audience="${DEPLOYED_SERVICE_URL}${EXPERIMENT_ENDPOINT}" \
        --location="${REGION}" \
        --time-zone="Etc/UTC" \
        --description="${EXPERIMENT_JOB_DESCRIPTION}" \
        --project="${PROJECT_ID}" \
        --attempt-deadline="1500s" \
        --max-retry-attempts=3 \
        --min-backoff=5s \
        --max-backoff=3600s \
        --max-doublings=5
fi

PAPER_JOB_ID="generate-weekly-research-paper"
PAPER_JOB_DESCRIPTION="Generate weekly research paper (via gcloud)"
PAPER_SCHEDULE="0 10 * * 0"
PAPER_ENDPOINT="/generate-paper"

if "${GCLOUD_CMD}" scheduler jobs describe "${PAPER_JOB_ID}" --location="${REGION}" --project="${PROJECT_ID}" &>/dev/null; then
    echo "Updating existing paper generation scheduler job: ${PAPER_JOB_ID}"
    "${GCLOUD_CMD}" scheduler jobs update http "${PAPER_JOB_ID}" \
        --schedule="${PAPER_SCHEDULE}" \
        --uri="${DEPLOYED_SERVICE_URL}${PAPER_ENDPOINT}" \
        --http-method=POST \
        --oidc-service-account-email="${CUSTOM_SCHEDULER_SA_EMAIL}" \
        --oidc-token-audience="${DEPLOYED_SERVICE_URL}${PAPER_ENDPOINT}" \
        --location="${REGION}" \
        --time-zone="Etc/UTC" \
        --description="${PAPER_JOB_DESCRIPTION}" \
        --project="${PROJECT_ID}" \
        --attempt-deadline="1780s" \
        --max-retry-attempts=1
else
    echo "Creating new paper generation scheduler job: ${PAPER_JOB_ID}"
    "${GCLOUD_CMD}" scheduler jobs create http "${PAPER_JOB_ID}" \
        --schedule="${PAPER_SCHEDULE}" \
        --uri="${DEPLOYED_SERVICE_URL}${PAPER_ENDPOINT}" \
        --http-method=POST \
        --oidc-service-account-email="${CUSTOM_SCHEDULER_SA_EMAIL}" \
        --oidc-token-audience="${DEPLOYED_SERVICE_URL}${PAPER_ENDPOINT}" \
        --location="${REGION}" \
        --time-zone="Etc/UTC" \
        --description="${PAPER_JOB_DESCRIPTION}" \
        --project="${PROJECT_ID}" \
        --attempt-deadline="1780s" \
        --max-retry-attempts=1
fi

echo "Cloud Scheduler jobs setup complete via gcloud CLI."

# --- Deployment Finished ---
echo " "
echo "🚀 Deployment of '${SERVICE_NAME}' service and scheduler setup complete!"
echo "Service URL: ${DEPLOYED_SERVICE_URL}"
echo "Remember: Access to this service is restricted."

# The following lines for running cloud_scheduler_setup.py are best run manually
# after confirming the deployment and ensuring the local environment is correctly set up with variables.
# echo "Consider setting up Cloud Scheduler to trigger your endpoints as needed using cloud_scheduler_setup.py."
# export GCP_PROJECT_ID="${PROJECT_ID}"
# export GCP_REGION="${REGION}"
# export CLOUD_RUN_INVOKE_URL="${DEPLOYED_SERVICE_URL}"
# # Ensure this SCHEDULER_SERVICE_ACCOUNT_EMAIL is the one with invoker permissions and you have actAs on it if it's not your user ADC.
# export SCHEDULER_SERVICE_ACCOUNT_EMAIL="custom-scheduler-sa@${PROJECT_ID}.iam.gserviceaccount.com"
# cd /Users/carlos/NOUS # Or your project path
# source .venv_x86_64_py310_hackathon/bin/activate # Or your venv
# pip install -r requirements.txt # Ensure google-cloud-scheduler is there
# python cloud_scheduler_setup.py
