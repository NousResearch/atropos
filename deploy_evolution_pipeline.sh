#!/bin/bash
set -e

PROJECT_ID=$(gcloud config get-value project)
REGION="us-central1" # Or your preferred region
SERVICE_NAME="vr-affordance-evolution" # Updated service name
IMAGE_TAG="latest" # Consider using more specific tags like v1.0, git commit SHA, etc.
IMAGE_URI="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:${IMAGE_TAG}"

# Ensure you are in the correct directory where your Dockerfile and source code are
# cd /path/to/your/spatial-research-pipeline # Uncomment and set if not running from project root

# Check if GOOGLE_CLOUD_PROJECT is set, if not, try to set it from gcloud config
if [ -z "${GOOGLE_CLOUD_PROJECT}" ]; then
  echo "GOOGLE_CLOUD_PROJECT not set. Trying to set from gcloud config."
  export GOOGLE_CLOUD_PROJECT=$(gcloud config get-value project)
  if [ -z "${GOOGLE_CLOUD_PROJECT}" ]; then
    echo "Error: GOOGLE_CLOUD_PROJECT could not be determined. Please set it manually or configure gcloud CLI."
    exit 1
  fi
fi
echo "Using GCP Project: ${GOOGLE_CLOUD_PROJECT}"


echo "🚀 Deploying ${SERVICE_NAME} to Cloud Run..."

# Build and push Docker image (assuming Dockerfile in current directory relative to script execution)
# Ensure your Dockerfile correctly copies the 'evolution' module and other dependencies.
echo "Building Docker image: ${IMAGE_URI}"
gcloud builds submit --tag ${IMAGE_URI} . --project=${GOOGLE_CLOUD_PROJECT}

# Deploy Cloud Run service with appropriate resources
# This assumes your app.py is set up to serve on port 8080 (default for Cloud Run Python)
echo "Deploying Cloud Run service: ${SERVICE_NAME}"
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_URI} \
    --platform managed \
    --region ${REGION} \
    --memory 8Gi \
    --cpu 4 \
    --timeout 3600 \
    --concurrency 10 \
    --max-instances 5 \
    --allow-unauthenticated \
    --set-env-vars="GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT},PYTHON_LOGGING_LEVEL=INFO" \
    --project=${GOOGLE_CLOUD_PROJECT} # Explicitly set project for deployment command
    # Add --service-account=YOUR_RUNTIME_SERVICE_ACCOUNT@${GOOGLE_CLOUD_PROJECT}.iam.gserviceaccount.com if needed for permissions

SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --platform managed --region ${REGION} --project=${GOOGLE_CLOUD_PROJECT} --format 'value(status.url)')

# Setup Cloud Scheduler for periodic paper generation/analysis (Example)
# Ensure the service account has 'Cloud Run Invoker' role on this service.
# And appropriate permissions if the endpoint modifies GCP resources (e.g. GCS).
SCHEDULER_JOB_NAME="weekly-evolution-paper-generation"
SCHEDULER_SERVICE_ACCOUNT_EMAIL="your-scheduler-sa@${PROJECT_ID}.iam.gserviceaccount.com" # REPLACE THIS
TARGET_ENDPOINT_PATH="/generate-paper" # Assuming your existing endpoint, or create a new one for evolution paper

echo "Checking if scheduler job '${SCHEDULER_JOB_NAME}' exists..."
if gcloud scheduler jobs describe ${SCHEDULER_JOB_NAME} --location ${REGION} --project=${PROJECT_ID} > /dev/null 2>&1; then
    echo "Scheduler job '${SCHEDULER_JOB_NAME}' already exists. Updating..."
    gcloud scheduler jobs update http ${SCHEDULER_JOB_NAME} \
        --schedule="0 2 * * 1" \
        --uri="${SERVICE_URL}${TARGET_ENDPOINT_PATH}" \
        --http-method POST \
        --oidc-service-account-email ${SCHEDULER_SERVICE_ACCOUNT_EMAIL} \
        --location ${REGION} \
        --project=${PROJECT_ID} \
        --description "Weekly generation of VR affordance evolution research paper draft."
else
    echo "Creating scheduler job '${SCHEDULER_JOB_NAME}'..."
    gcloud scheduler jobs create http ${SCHEDULER_JOB_NAME} \
        --schedule="0 2 * * 1" \
        --uri="${SERVICE_URL}${TARGET_ENDPOINT_PATH}" \
        --http-method POST \
        --oidc-service-account-email ${SCHEDULER_SERVICE_ACCOUNT_EMAIL} \
        --location ${REGION} \
        --project=${PROJECT_ID} \
        --description "Weekly generation of VR affordance evolution research paper draft."
fi

# Note on Monitoring Dashboard:
# Creating a monitoring dashboard (evolution-dashboard.json) via gcloud CLI is complex.
# It's usually done through the GCP Console UI or by using Terraform/Deployment Manager.
# The JSON structure is quite detailed.
# For now, instruct user to set this up manually or provide a template JSON they can import.
echo "\n--- Monitoring Dashboard Setup --- "
echo "Please create or update your GCP Monitoring Dashboard manually."
echo "Key metrics to track for '${SERVICE_NAME}':"
echo "- Request Count & Latency"
echo "- Instance Count & CPU/Memory Utilization"
echo "- Container crash counts & logs (especially for long-running evolution tasks)"
echo "- 4xx/5xx error rates"

echo "\n✅ ${SERVICE_NAME} deployment script finished."
echo "Service URL: ${SERVICE_URL}"
