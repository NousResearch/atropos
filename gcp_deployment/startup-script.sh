#!/bin/bash
set -e

# Update system
apt-get update
apt-get install -y python3 python3-pip git curl

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Clone AMIEN repository
git clone https://github.com/your-repo/amien.git /opt/amien
cd /opt/amien

# Install Python dependencies
pip3 install -r requirements.txt

# Set up environment variables
export PROJECT_ID=$(curl -s "http://metadata.google.internal/computeMetadata/v1/project/project-id" -H "Metadata-Flavor: Google")
export REGION="us-central1"

# Run massive scale experiments
python3 scale_to_production.py

# Upload results to Cloud Storage
gsutil cp -r massive_scale_output/* gs://amien-experiment-results/$(date +%Y%m%d_%H%M%S)/

# Signal completion
echo "Massive scale experiments completed" | logger
