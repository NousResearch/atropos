#!/bin/bash

echo "🚀 Society Simulation Cloud Deployment Script"
echo "=============================================="

# Set project and region
PROJECT_ID="amien-research-pipeline"
REGION="us-central1"
SERVICE_NAME="society-simulation"

echo "📋 Configuration:"
echo "   Project: $PROJECT_ID"
echo "   Region: $REGION"
echo "   Service: $SERVICE_NAME"
echo ""

# Step 1: Authenticate and set project
echo "🔐 Step 1: Authentication and Project Setup"
echo "Please ensure you're authenticated with:"
echo "   gcloud auth login"
echo "   gcloud config set project $PROJECT_ID"
echo ""

# Step 2: Enable required APIs
echo "🔧 Step 2: Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable scheduler.googleapis.com
echo "✅ APIs enabled"
echo ""

# Step 3: Create storage bucket for results
echo "📦 Step 3: Creating Cloud Storage bucket..."
gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://$PROJECT_ID-simulation-results || echo "Bucket might already exist"
echo "✅ Storage bucket ready"
echo ""

# Step 4: Build and deploy
echo "🏗️ Step 4: Building and deploying container..."
cd gcp_deployment

# Build the container
echo "Building container image..."
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME:latest .

if [ $? -eq 0 ]; then
    echo "✅ Container built successfully"
else
    echo "❌ Container build failed"
    exit 1
fi

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image gcr.io/$PROJECT_ID/$SERVICE_NAME:latest \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 3600 \
    --max-instances 100 \
    --set-env-vars PROJECT_ID=$PROJECT_ID

if [ $? -eq 0 ]; then
    echo "✅ Cloud Run service deployed successfully"

    # Get the service URL
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")
    echo ""
    echo "🌐 Society Simulation API is now live!"
    echo "   URL: $SERVICE_URL"
    echo ""
    echo "📊 API Endpoints:"
    echo "   Health Check:     $SERVICE_URL/"
    echo "   Run Simulation:   $SERVICE_URL/simulation/run"
    echo "   Check Status:     $SERVICE_URL/simulation/status/{id}"
    echo "   List All:         $SERVICE_URL/simulation/list"
    echo "   Benchmark:        $SERVICE_URL/simulation/benchmark"
    echo ""
    echo "📝 Example usage:"
    echo "   curl -X POST $SERVICE_URL/simulation/run \\"
    echo "        -H 'Content-Type: application/json' \\"
    echo "        -d '{\"agents\": 500, \"steps\": 100}'"
    echo ""

    # Test the deployment
    echo "🧪 Testing deployment..."
    curl -s "$SERVICE_URL/" | jq . || echo "Service is starting up..."

else
    echo "❌ Cloud Run deployment failed"
    exit 1
fi

echo ""
echo "🎯 Deployment Complete!"
echo "Your society simulation is now running in the cloud!"
echo "You can scale to thousands of agents with the Cloud Run auto-scaling."
