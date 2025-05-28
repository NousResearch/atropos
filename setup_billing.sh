#!/bin/bash

echo "🏦 AMIEN Google Cloud Billing Setup"
echo "============================================================"

# Check if gcloud is installed and authenticated
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Please install it first."
    exit 1
fi

echo "🔍 Checking current authentication..."
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -1 > /dev/null; then
    echo "⚠️ Not authenticated. Running gcloud auth login..."
    gcloud auth login
else
    echo "✅ Already authenticated"
fi

echo ""
echo "🔍 Checking project configuration..."
PROJECT_ID="amien-research-pipeline"
gcloud config set project $PROJECT_ID

echo ""
echo "🔍 Checking billing status..."
BILLING_ENABLED=$(gcloud services list --enabled --filter="name:cloudbilling.googleapis.com" --format="value(name)" 2>/dev/null)

if [ -z "$BILLING_ENABLED" ]; then
    echo "❌ Billing is not enabled for project $PROJECT_ID"
    echo ""
    echo "📋 Next Steps:"
    echo "1. Go to: https://console.cloud.google.com/billing"
    echo "2. Create a billing account (requires credit card)"
    echo "3. Link billing account to project: $PROJECT_ID"
    echo "4. Run this script again to verify"
    echo ""
    echo "💡 Expected monthly costs: $50-200 for testing, $2000-5000 for production"
    exit 1
else
    echo "✅ Billing is enabled!"
fi

echo ""
echo "🔍 Checking required APIs..."
REQUIRED_APIS=(
    "run.googleapis.com"
    "artifactregistry.googleapis.com"
    "containerregistry.googleapis.com"
    "compute.googleapis.com"
    "storage.googleapis.com"
)

for api in "${REQUIRED_APIS[@]}"; do
    if gcloud services list --enabled --filter="name:$api" --format="value(name)" | grep -q "$api"; then
        echo "✅ $api enabled"
    else
        echo "⚠️ Enabling $api..."
        gcloud services enable $api
    fi
done

echo ""
echo "🎉 Billing Setup Complete!"
echo "============================================================"
echo "✅ Project: $PROJECT_ID"
echo "✅ Billing: Enabled"
echo "✅ APIs: Enabled"
echo ""
echo "🚀 Ready to deploy AMIEN!"
echo "Run: ./deploy_ai_integration.sh"
echo ""
echo "📊 Monitor costs at: https://console.cloud.google.com/billing" 