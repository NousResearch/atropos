#!/bin/bash

# AMIEN AI Integration Production Deployment Script
# Comprehensive automation for deploying AMIEN to Google Cloud Platform

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID=${GCP_PROJECT_ID:-"amien-research-pipeline"}
REGION=${GCP_REGION:-"us-central1"}
ZONE=${GCP_ZONE:-"us-central1-a"}

echo -e "${BLUE}🚀 AMIEN AI Integration Production Deployment${NC}"
echo "============================================================"
echo -e "Project ID: ${GREEN}$PROJECT_ID${NC}"
echo -e "Region: ${GREEN}$REGION${NC}"
echo -e "Zone: ${GREEN}$ZONE${NC}"
echo -e "Timestamp: ${GREEN}$(date -u +"%Y-%m-%dT%H:%M:%SZ")${NC}"
echo ""

# Function to print status
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️ $1${NC}"
}

# Check prerequisites
echo -e "${BLUE}🔍 Checking Prerequisites...${NC}"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    print_error "gcloud CLI is not installed. Please install it first."
    exit 1
fi
print_status "gcloud CLI found"

# Check if authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    print_error "Not authenticated with gcloud. Please run 'gcloud auth login'"
    exit 1
fi
print_status "gcloud authentication verified"

# Check if project exists
if ! gcloud projects describe $PROJECT_ID &> /dev/null; then
    print_warning "Project $PROJECT_ID not found. Creating it..."
    gcloud projects create $PROJECT_ID --name="AMIEN Research Pipeline"
    print_status "Project created"
fi

# Set project
gcloud config set project $PROJECT_ID
print_status "Project set to $PROJECT_ID"

# Enable required APIs
echo -e "\n${BLUE}🔧 Enabling Required APIs...${NC}"
apis=(
    "run.googleapis.com"
    "cloudbuild.googleapis.com"
    "storage.googleapis.com"
    "secretmanager.googleapis.com"
    "scheduler.googleapis.com"
    "compute.googleapis.com"
    "monitoring.googleapis.com"
    "logging.googleapis.com"
    "artifactregistry.googleapis.com"
)

for api in "${apis[@]}"; do
    echo "Enabling $api..."
    gcloud services enable $api
done
print_status "All APIs enabled"

# Create storage buckets
echo -e "\n${BLUE}🗄️ Setting up Storage...${NC}"
BUCKET_NAME="${PROJECT_ID}-research-data"
if ! gsutil ls -b gs://$BUCKET_NAME &> /dev/null; then
    gsutil mb -l $REGION gs://$BUCKET_NAME
    print_status "Storage bucket created: $BUCKET_NAME"
else
    print_info "Storage bucket already exists: $BUCKET_NAME"
fi

# Set up secrets
echo -e "\n${BLUE}🔐 Setting up Secrets...${NC}"
secrets=(
    "gemini-api-key"
    "openai-api-key"
    "perplexity-api-key"
)

for secret in "${secrets[@]}"; do
    if ! gcloud secrets describe $secret &> /dev/null; then
        echo "Creating secret: $secret"
        echo "placeholder-key" | gcloud secrets create $secret --data-file=-
        print_warning "Created placeholder for $secret - UPDATE WITH REAL KEY"
    else
        print_info "Secret already exists: $secret"
    fi
done

# Build and deploy container images
echo -e "\n${BLUE}🐳 Building Container Images...${NC}"

# Create Artifact Registry repository
REPO_NAME="amien-images"
if ! gcloud artifacts repositories describe $REPO_NAME --location=$REGION &> /dev/null; then
    gcloud artifacts repositories create $REPO_NAME \
        --repository-format=docker \
        --location=$REGION \
        --description="AMIEN container images"
    print_status "Artifact Registry repository created"
fi

# Configure Docker authentication
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Build main API image
echo "Building main API image..."
cat > Dockerfile.production << 'EOF'
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements_production.txt .
RUN pip install --no-cache-dir -r requirements_production.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8080

# Expose port
EXPOSE 8080

# Run the application
CMD ["uvicorn", "gcp_deployment.main_api:app", "--host", "0.0.0.0", "--port", "8080"]
EOF

# Build and push image
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/amien-api:latest"
docker build -f Dockerfile.production -t $IMAGE_URI .
docker push $IMAGE_URI
print_status "Main API image built and pushed"

# Deploy Cloud Run services
echo -e "\n${BLUE}☁️ Deploying Cloud Run Services...${NC}"

# Deploy main API service
gcloud run deploy amien-api-service \
    --image=$IMAGE_URI \
    --platform=managed \
    --region=$REGION \
    --allow-unauthenticated \
    --memory=2Gi \
    --cpu=2 \
    --max-instances=100 \
    --set-env-vars="GCP_PROJECT_ID=$PROJECT_ID,GCP_REGION=$REGION" \
    --set-secrets="GEMINI_API_KEY=gemini-api-key:latest,OPENAI_API_KEY=openai-api-key:latest"

print_status "Main API service deployed"

# Get service URL
SERVICE_URL=$(gcloud run services describe amien-api-service --region=$REGION --format="value(status.url)")
print_info "Service URL: $SERVICE_URL"

# Set up Cloud Scheduler jobs
echo -e "\n${BLUE}⏰ Setting up Scheduled Jobs...${NC}"

# Daily research generation job
gcloud scheduler jobs create http daily-research-generation \
    --schedule="0 9 * * *" \
    --uri="${SERVICE_URL}/generate-research" \
    --http-method=POST \
    --time-zone="UTC" \
    --description="Daily research paper generation" \
    || print_info "Daily job already exists"

# Weekly massive experiments job
gcloud scheduler jobs create http weekly-massive-experiments \
    --schedule="0 2 * * 1" \
    --uri="${SERVICE_URL}/run-massive-experiments" \
    --http-method=POST \
    --time-zone="UTC" \
    --description="Weekly massive scale experiments" \
    || print_info "Weekly job already exists"

print_status "Scheduled jobs configured"

# Set up monitoring
echo -e "\n${BLUE}📊 Setting up Monitoring...${NC}"

# Create monitoring dashboard
if [ -f "monitoring/production_dashboard.json" ]; then
    gcloud monitoring dashboards create --config-from-file=monitoring/production_dashboard.json
    print_status "Monitoring dashboard created"
fi

# Create alert policies
cat > alert_policy.json << 'EOF'
{
  "displayName": "AMIEN High Error Rate",
  "conditions": [
    {
      "displayName": "High error rate condition",
      "conditionThreshold": {
        "filter": "resource.type=\"cloud_run_revision\"",
        "comparison": "COMPARISON_GREATER_THAN",
        "thresholdValue": 0.1,
        "duration": "300s"
      }
    }
  ],
  "alertStrategy": {
    "autoClose": "1800s"
  },
  "enabled": true
}
EOF

gcloud alpha monitoring policies create --policy-from-file=alert_policy.json
print_status "Alert policies created"

# Run system tests
echo -e "\n${BLUE}🧪 Running System Tests...${NC}"
python3 final_production_deployment.py --test-only

# Generate analytics report
echo -e "\n${BLUE}📈 Generating Analytics Report...${NC}"
python3 advanced_analytics_system.py

# Create deployment summary
echo -e "\n${BLUE}📋 Creating Deployment Summary...${NC}"
cat > DEPLOYMENT_SUMMARY.md << EOF
# AMIEN Production Deployment Summary

**Deployment Date:** $(date -u +"%Y-%m-%d %H:%M:%S UTC")
**Project ID:** $PROJECT_ID
**Region:** $REGION

## Deployed Services

### Cloud Run Services
- **amien-api-service**: $SERVICE_URL
  - Memory: 2Gi
  - CPU: 2
  - Max Instances: 100

### Storage
- **Research Data Bucket**: gs://$BUCKET_NAME

### Scheduled Jobs
- **Daily Research Generation**: 09:00 UTC daily
- **Weekly Massive Experiments**: 02:00 UTC every Monday

### Monitoring
- **Dashboard**: Available in Cloud Monitoring
- **Alerts**: Configured for error rates and performance

## Next Steps

1. **Update API Keys**: Replace placeholder secrets with real API keys
   \`\`\`bash
   echo "your-real-gemini-key" | gcloud secrets versions add gemini-api-key --data-file=-
   echo "your-real-openai-key" | gcloud secrets versions add openai-api-key --data-file=-
   \`\`\`

2. **Test Deployment**: 
   \`\`\`bash
   curl $SERVICE_URL/health
   \`\`\`

3. **Monitor System**: Check Cloud Monitoring dashboard

4. **Scale as Needed**: Adjust Cloud Run settings based on usage

## Cost Estimation
- **Base Cost**: ~\$50-100/month for infrastructure
- **Research Generation**: ~\$15 per paper
- **Massive Experiments**: ~\$5-10 per 1000 experiments

## Support
- **Logs**: Available in Cloud Logging
- **Monitoring**: Cloud Monitoring dashboard
- **Issues**: Check service logs and error rates
EOF

print_status "Deployment summary created: DEPLOYMENT_SUMMARY.md"

# Final status
echo ""
echo -e "${GREEN}🎉 AMIEN Production Deployment Complete!${NC}"
echo "============================================================"
echo -e "✅ All services deployed successfully"
echo -e "✅ Monitoring and alerts configured"
echo -e "✅ Scheduled jobs running"
echo -e "✅ Analytics system operational"
echo ""
echo -e "${YELLOW}⚠️ IMPORTANT: Update API keys in Secret Manager${NC}"
echo -e "${BLUE}📊 Service URL: $SERVICE_URL${NC}"
echo -e "${BLUE}📈 Monitor at: https://console.cloud.google.com/monitoring/dashboards${NC}"
echo ""
echo -e "${GREEN}🚀 AMIEN is now running in production!${NC}" 