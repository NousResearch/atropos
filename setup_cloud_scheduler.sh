#!/bin/bash
set -e

echo "⏰ Setting up Cloud Scheduler for AMIEN automated research generation"

# Enable Cloud Scheduler API
echo "📡 Enabling Cloud Scheduler API..."
gcloud services enable cloudscheduler.googleapis.com

# Set location for scheduler jobs
LOCATION="us-central1"

# Create daily research generation job
echo "📅 Creating daily research generation job..."
gcloud scheduler jobs create http daily-research-generation \
    --location=$LOCATION \
    --schedule="0 2 * * *" \
    --uri="https://amien-api-service-643533604146.us-central1.run.app/research/generate" \
    --http-method=POST \
    --time-zone="UTC" \
    --description="Daily automated research generation using AMIEN evolution system" \
    --attempt-deadline=1800s \
    --max-retry-attempts=3 \
    --max-retry-duration=3600s \
    --min-backoff=60s \
    --max-backoff=300s \
    --max-doublings=3 || echo "Daily job already exists, updating..."

# Update daily job if it already exists
gcloud scheduler jobs update http daily-research-generation \
    --location=$LOCATION \
    --schedule="0 2 * * *" \
    --uri="https://amien-api-service-643533604146.us-central1.run.app/research/generate" \
    --http-method=POST \
    --time-zone="UTC" \
    --description="Daily automated research generation using AMIEN evolution system" \
    --attempt-deadline=1800s \
    --max-retry-attempts=3 \
    --max-retry-duration=3600s \
    --min-backoff=60s \
    --max-backoff=300s \
    --max-doublings=3 || echo "Failed to update daily job"

# Create weekly massive experiments job
echo "📊 Creating weekly massive experiments job..."
gcloud scheduler jobs create http weekly-massive-experiments \
    --location=$LOCATION \
    --schedule="0 1 * * 0" \
    --uri="https://amien-api-service-643533604146.us-central1.run.app/experiments/massive?sample_size=10000" \
    --http-method=POST \
    --time-zone="UTC" \
    --description="Weekly massive scale VR experiments with 10,000 simulated users" \
    --attempt-deadline=3600s \
    --max-retry-attempts=2 \
    --max-retry-duration=7200s \
    --min-backoff=300s \
    --max-backoff=600s \
    --max-doublings=2 || echo "Weekly job already exists, updating..."

# Update weekly job if it already exists
gcloud scheduler jobs update http weekly-massive-experiments \
    --location=$LOCATION \
    --schedule="0 1 * * 0" \
    --uri="https://amien-api-service-643533604146.us-central1.run.app/experiments/massive?sample_size=10000" \
    --http-method=POST \
    --time-zone="UTC" \
    --description="Weekly massive scale VR experiments with 10,000 simulated users" \
    --attempt-deadline=3600s \
    --max-retry-attempts=2 \
    --max-retry-duration=7200s \
    --min-backoff=300s \
    --max-backoff=600s \
    --max-doublings=2 || echo "Failed to update weekly job"

# Create monthly comprehensive analysis job
echo "📈 Creating monthly comprehensive analysis job..."
gcloud scheduler jobs create http monthly-comprehensive-analysis \
    --location=$LOCATION \
    --schedule="0 0 1 * *" \
    --uri="https://amien-api-service-643533604146.us-central1.run.app/experiments/massive?sample_size=50000" \
    --http-method=POST \
    --time-zone="UTC" \
    --description="Monthly comprehensive VR research with 50,000 simulated users" \
    --attempt-deadline=7200s \
    --max-retry-attempts=1 \
    --max-retry-duration=10800s \
    --min-backoff=600s \
    --max-backoff=1200s \
    --max-doublings=1 || echo "Monthly job already exists, updating..."

# Update monthly job if it already exists
gcloud scheduler jobs update http monthly-comprehensive-analysis \
    --location=$LOCATION \
    --schedule="0 0 1 * *" \
    --uri="https://amien-api-service-643533604146.us-central1.run.app/experiments/massive?sample_size=50000" \
    --http-method=POST \
    --time-zone="UTC" \
    --description="Monthly comprehensive VR research with 50,000 simulated users" \
    --attempt-deadline=7200s \
    --max-retry-attempts=1 \
    --max-retry-duration=10800s \
    --min-backoff=600s \
    --max-backoff=1200s \
    --max-doublings=1 || echo "Failed to update monthly job"

# List all scheduled jobs
echo "📋 Current scheduled jobs:"
gcloud scheduler jobs list --location=$LOCATION

echo "✅ Cloud Scheduler setup complete!"
echo ""
echo "📅 Scheduled Jobs:"
echo "  • Daily Research Generation: Every day at 2:00 AM UTC"
echo "  • Weekly Massive Experiments: Every Sunday at 1:00 AM UTC (10,000 users)"
echo "  • Monthly Comprehensive Analysis: 1st of every month at midnight UTC (50,000 users)"
echo ""
echo "🔧 To manually trigger a job:"
echo "  gcloud scheduler jobs run daily-research-generation --location=$LOCATION"
echo "  gcloud scheduler jobs run weekly-massive-experiments --location=$LOCATION"
echo "  gcloud scheduler jobs run monthly-comprehensive-analysis --location=$LOCATION" 