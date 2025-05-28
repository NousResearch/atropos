#!/bin/bash
set -e

echo "🚀 Deploying AMIEN to Google Cloud Platform"
echo "Project ID: amien-research-pipeline"
echo "Region: us-central1"

# Set project
gcloud config set project amien-research-pipeline

# Enable required APIs
echo "📡 Enabling required APIs..."
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable secretmanager.googleapis.com
gcloud services enable scheduler.googleapis.com
gcloud services enable compute.googleapis.com
gcloud services enable monitoring.googleapis.com

# Create secrets (you'll need to add the actual API keys)
echo "🔐 Creating secrets..."
echo "YOUR_GEMINI_API_KEY" | gcloud secrets create gemini-api-key --data-file=-
echo "YOUR_OPENAI_API_KEY" | gcloud secrets create openai-api-key --data-file=-

# Build and deploy container images
echo "🏗️ Building container images..."
cd gcp_deployment

# Build API service
gcloud builds submit --tag gcr.io/amien-research-pipeline/amien-api-service .

# Deploy Cloud Run services
echo "☁️ Deploying Cloud Run services..."
gcloud run deploy amien-api-service \
    --image gcr.io/amien-research-pipeline/amien-api-service \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --max-instances 100

# Deploy with Terraform
echo "🏗️ Deploying infrastructure with Terraform..."
terraform init
terraform plan
terraform apply -auto-approve

# Create Cloud Scheduler jobs
echo "⏰ Creating scheduled jobs..."
gcloud scheduler jobs create http daily-research \
    --schedule="0 2 * * *" \
    --uri="$(gcloud run services describe amien-api-service --region us-central1 --format='value(status.url)')/research/generate" \
    --http-method=POST \
    --location=us-central1

gcloud scheduler jobs create http weekly-massive-experiments \
    --schedule="0 0 * * 1" \
    --uri="$(gcloud run services describe amien-api-service --region us-central1 --format='value(status.url)')/experiments/massive" \
    --http-method=POST \
    --message-body='{"sample_size": 5000}' \
    --location=us-central1

echo "✅ AMIEN deployment complete!"
echo "🌐 API URL: $(gcloud run services describe amien-api-service --region us-central1 --format='value(status.url)')"
echo "📊 Monitor at: https://console.cloud.google.com/monitoring"
echo "📅 Scheduler at: https://console.cloud.google.com/cloudscheduler"
