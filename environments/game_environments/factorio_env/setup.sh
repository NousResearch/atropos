#!/bin/bash
# Setup script for Factorio Learning Environment

echo "🎮 Setting up Factorio Learning Environment..."

# Check if we're on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM_FLAG="--platform linux/amd64"
    echo "📱 Detected macOS - will use linux/amd64 platform for Rosetta"
else
    PLATFORM_FLAG=""
    echo "🐧 Detected Linux/other - using native platform"
fi

# Update submodules
echo "📦 Updating git submodules..."
git submodule update --init --recursive

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install -e ./fle
pip install -r requirements.txt

# Build Docker image
echo "🐳 Building Factorio Docker image..."
cd fle/fle/cluster/docker
docker build -t factorio . $PLATFORM_FLAG

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully"
else
    echo "❌ Docker build failed"
    exit 1
fi

cd ../../../

echo "🚀 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Start Factorio server: docker-compose up -d factorio_0"
echo "2. Start LLM server on port 8080"
echo "3. Run agent: python llama_agent.py"
