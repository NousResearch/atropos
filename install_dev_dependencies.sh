#!/bin/bash
set -e

echo "🚀 Installing 2,500-Agent LLM Society Simulation Dependencies"
echo "============================================================="

# Install standard requirements first
echo "📦 Installing standard requirements..."
pip install -r requirements.txt

# Install mesa-frames (check if it exists on PyPI, otherwise use git)
echo "🔧 Installing mesa-frames..."
pip install mesa-frames || pip install git+https://github.com/projectmesa/mesa-frames.git

# Install Atropos from NousResearch
echo "🤖 Installing Atropos from NousResearch..."
pip install git+https://github.com/NousResearch/atropos.git

# Install Point-E from OpenAI
echo "🎨 Installing Point-E from OpenAI..."
pip install git+https://github.com/openai/point-e.git

# Install development dependencies
echo "🛠️  Installing development dependencies..."
pip install -e ".[dev]"

# Verify installations
echo "✅ Verifying installations..."
python -c "import mesa; print(f'Mesa: {mesa.__version__}')"
python -c "import mesa_frames; print('Mesa-frames: OK')" || echo "Mesa-frames: Will install fallback"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

echo ""
echo "🎉 Installation complete!"
echo "💡 Next steps:"
echo "   1. Run: python src/main.py --help"
echo "   2. Check: python tests/test_installation.py"
echo "   3. Start development: python src/simulation/basic_demo.py"
