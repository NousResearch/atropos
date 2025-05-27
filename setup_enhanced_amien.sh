#!/bin/bash

# Enhanced AMIEN Setup Script
# This script sets up the enhanced AMIEN system with AI Scientist and FunSearch integration

set -e

echo "🚀 Setting up Enhanced AMIEN Discovery Pipeline..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

# Check prerequisites
print_header "📋 Checking Prerequisites..."

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
required_version="3.8"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    print_error "Python 3.8+ required. Found: $python_version"
    exit 1
fi
print_status "Python version: $python_version ✓"

# Check Git
if ! command -v git &> /dev/null; then
    print_error "Git is required but not installed"
    exit 1
fi
print_status "Git available ✓"

# Check Google Cloud CLI (optional but recommended)
if command -v gcloud &> /dev/null; then
    print_status "Google Cloud CLI available ✓"
else
    print_warning "Google Cloud CLI not found. Install it for easier GCP management."
fi

# Create directory structure
print_header "📁 Creating Directory Structure..."

directories=(
    "logs"
    "experiments" 
    "papers"
    "functions"
    "data"
    "ai_scientist"
    "funsearch"
    "temp"
)

for dir in "${directories[@]}"; do
    mkdir -p "$dir"
    print_status "Created directory: $dir"
done

# Install Python dependencies
print_header "📦 Installing Python Dependencies..."

if [ -f "requirements.txt" ]; then
    print_status "Installing requirements from requirements.txt..."
    pip3 install -r requirements.txt
    print_status "Dependencies installed ✓"
else
    print_error "requirements.txt not found!"
    exit 1
fi

# Setup environment variables template
print_header "⚙️ Setting up Environment Configuration..."

if [ ! -f ".env" ]; then
    cat > .env << EOF
# Enhanced AMIEN Configuration
# Copy this file and update with your actual values

# Google Cloud Project
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json
GCP_PROJECT_ID=your-project-id

# Padres API Configuration
PADRES_API_URL=https://your-padres-api-url
PADRES_API_KEY=your-padres-api-key

# AI Research APIs
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
GEMINI_API_KEY=your-gemini-api-key
PERPLEXITY_API_KEY=your-perplexity-api-key

# Optional: Weights & Biases for experiment tracking
WANDB_API_KEY=your-wandb-api-key

# Discovery Pipeline Configuration
DISCOVERY_CYCLE_HOURS=4
MAX_PARALLEL_DISCOVERIES=10
PAPERS_PER_WEEK=50
FUNCTIONS_PER_WEEK=50
EOF
    print_status "Created .env template file"
    print_warning "Please update .env with your actual API keys and configuration"
else
    print_status ".env file already exists"
fi

# Clone AI research repositories (optional - will be done automatically by the system)
print_header "🔬 Preparing AI Research Repositories..."

print_status "AI Scientist and FunSearch repositories will be cloned automatically when needed"
print_status "This ensures you always get the latest versions"

# Create startup script
print_header "🚀 Creating Startup Scripts..."

# Make the Python startup script executable
if [ -f "start_enhanced_amien.py" ]; then
    chmod +x start_enhanced_amien.py
    print_status "Made start_enhanced_amien.py executable"
fi

# Create a simple bash wrapper
cat > start_amien.sh << 'EOF'
#!/bin/bash

# Enhanced AMIEN Startup Wrapper
echo "🚀 Starting Enhanced AMIEN Discovery Pipeline..."

# Load environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Start the enhanced AMIEN system
python3 start_enhanced_amien.py
EOF

chmod +x start_amien.sh
print_status "Created start_amien.sh wrapper script"

# Create Google Cloud Storage buckets setup script
cat > setup_gcs_buckets.sh << 'EOF'
#!/bin/bash

# Setup Google Cloud Storage buckets for Enhanced AMIEN

if [ -z "$GCP_PROJECT_ID" ]; then
    echo "Error: GCP_PROJECT_ID not set. Please update your .env file."
    exit 1
fi

echo "Creating GCS buckets for project: $GCP_PROJECT_ID"

# Create buckets
gsutil mb gs://${GCP_PROJECT_ID}-research-papers || echo "Bucket may already exist"
gsutil mb gs://${GCP_PROJECT_ID}-discovered-functions || echo "Bucket may already exist"
gsutil mb gs://${GCP_PROJECT_ID}-research-data || echo "Bucket may already exist"
gsutil mb gs://${GCP_PROJECT_ID}-experiment-logs || echo "Bucket may already exist"

# Set bucket permissions (adjust as needed)
echo "Setting bucket permissions..."
gsutil iam ch allUsers:objectViewer gs://${GCP_PROJECT_ID}-research-papers
gsutil iam ch allUsers:objectViewer gs://${GCP_PROJECT_ID}-discovered-functions

echo "✅ GCS buckets setup complete"
EOF

chmod +x setup_gcs_buckets.sh
print_status "Created setup_gcs_buckets.sh script"

# Create monitoring script
cat > monitor_amien.py << 'EOF'
#!/usr/bin/env python3
"""
Enhanced AMIEN Monitoring Script
Provides real-time monitoring of the discovery pipeline
"""

import asyncio
import json
import time
from datetime import datetime
from enhanced_research_orchestrator import EnhancedResearchOrchestrator

async def monitor_pipeline():
    """Monitor the Enhanced AMIEN pipeline"""
    
    config = {
        "project_id": "spatial-research-pipeline",  # Update with your project ID
        "use_ai_scientist_v2": True
    }
    
    orchestrator = EnhancedResearchOrchestrator(
        project_id=config["project_id"],
        config=config
    )
    
    print("📊 Enhanced AMIEN Pipeline Monitor")
    print("=" * 50)
    
    while True:
        try:
            stats = await orchestrator.get_discovery_statistics()
            
            print(f"\n🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("-" * 30)
            
            if stats:
                ai_stats = stats.get("ai_scientist", {})
                fun_stats = stats.get("funsearch", {})
                active_stats = stats.get("active_discoveries", {})
                rates = stats.get("discovery_rates", {})
                
                print(f"🤖 AI Scientist: {ai_stats.get('total_papers', 0)} papers")
                print(f"🔬 FunSearch: {fun_stats.get('total_functions', 0)} functions")
                print(f"🔄 Active: {sum(active_stats.values())} discoveries")
                print(f"📈 Rate: {rates.get('total_discoveries_per_day', 0):.1f}/day")
            else:
                print("⚠️  No statistics available")
            
            await asyncio.sleep(30)  # Update every 30 seconds
            
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(monitor_pipeline())
EOF

chmod +x monitor_amien.py
print_status "Created monitor_amien.py monitoring script"

# Final setup summary
print_header "✅ Enhanced AMIEN Setup Complete!"

cat << EOF

🎉 Enhanced AMIEN Discovery Pipeline is ready!

📋 Next Steps:
1. Update .env file with your API keys and configuration
2. Set up Google Cloud Storage buckets: ./setup_gcs_buckets.sh
3. Start the discovery pipeline: ./start_amien.sh
4. Monitor progress: python3 monitor_amien.py

📁 Directory Structure:
   • logs/           - System logs
   • experiments/    - Experiment data
   • papers/         - Generated research papers
   • functions/      - Discovered functions
   • data/           - Research data storage

🔧 Configuration Files:
   • .env                    - Environment variables
   • start_amien.sh         - Main startup script
   • setup_gcs_buckets.sh   - GCS bucket setup
   • monitor_amien.py       - Real-time monitoring

🚀 Expected Capabilities:
   • 50+ research papers per week
   • 50+ novel algorithms per week
   • Cross-domain inspiration from fireflies, casino psychology, etc.
   • Autonomous scientific discovery
   • Integration with AI Scientist v1/v2 and FunSearch

📖 Documentation:
   • integration_plan.md    - Detailed integration plan
   • enhanced_amien.log     - Runtime logs

⚠️  Important:
   • Update CONFIG['project_id'] in start_enhanced_amien.py
   • Ensure your GCP service account has necessary permissions
   • Set up API keys for OpenAI, Anthropic, Gemini, and Perplexity

Happy discovering! 🔬✨

EOF

print_status "Setup script completed successfully!" 