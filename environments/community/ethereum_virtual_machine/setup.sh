#!/bin/bash

# Ethereum Virtual Machine Environment Setup Script
# This script installs Foundry/Anvil and sets up the Python environment

set -e  # Exit on any error

echo "🔧 Setting up Ethereum Virtual Machine Environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# Check if Foundry is already installed
check_foundry() {
    if command -v anvil &> /dev/null && command -v cast &> /dev/null && command -v forge &> /dev/null; then
        print_status "Foundry is already installed"
        anvil --version
        return 0
    else
        return 1
    fi
}

# Install Foundry
install_foundry() {
    print_status "Installing Foundry..."

    # Download and install Foundry
    curl -L https://foundry.paradigm.xyz | bash

    # Source the profile to update PATH
    if [ -f ~/.bashrc ]; then
        source ~/.bashrc
    elif [ -f ~/.zshrc ]; then
        source ~/.zshrc
    fi

    # Run foundryup to install the latest version
    if command -v foundryup &> /dev/null; then
        foundryup
    else
        print_warning "foundryup not found in PATH. Please restart your terminal and run 'foundryup'"
        print_warning "Then run this setup script again."
        exit 1
    fi
}

# Verify installation
verify_installation() {
    print_status "Verifying installation..."

    if ! command -v anvil &> /dev/null; then
        print_error "Anvil not found. Installation may have failed."
        print_error "Please restart your terminal and try again."
        exit 1
    fi

    if ! command -v cast &> /dev/null; then
        print_error "Cast not found. Installation may have failed."
        exit 1
    fi

    print_status "✅ Foundry tools installed successfully:"
    echo "  - $(anvil --version)"
    echo "  - $(cast --version)"
    echo "  - $(forge --version)"
}

# Install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."

    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_status "Installed EVM-specific dependencies from requirements.txt"
    else
        print_warning "requirements.txt not found. Installing minimal dependencies..."
        pip install pyyaml>=6.0
    fi

    print_status "Note: Main dependencies (openai, pydantic, requests) are provided by atroposlib"
}

# Check for OpenAI API key
check_openai_key() {
    if [ -z "$OPENAI_API_KEY" ]; then
        print_warning "OPENAI_API_KEY environment variable not set"
        echo "  Set it with: export OPENAI_API_KEY='your-api-key-here'"
        echo "  This is required for question generation"
    else
        print_status "✅ OPENAI_API_KEY is set"
    fi
}

# Test the configuration
test_config() {
    print_status "Testing configuration..."

    python -c "
try:
    from anvil import AnvilConfig
    config = AnvilConfig()
    print('✅ Configuration loaded successfully')
    print(f'   - Config file: {config.config_file}')
    print(f'   - Network port: {config.anvil.port}')
    print(f'   - Fork URL: {config.anvil.fork_url}')
    print(f'   - Custom wallet: {config.funding.custom_wallet}')
except Exception as e:
    print(f'❌ Configuration test failed: {e}')
    exit(1)
"
}

# Main setup process
main() {
    echo "🚀 Starting setup process..."
    echo

    # Check if already installed
    if check_foundry; then
        print_status "Foundry already installed, skipping installation"
    else
        install_foundry
        verify_installation
    fi

    echo
    install_python_deps

    echo
    check_openai_key

    echo
    test_config

    echo
    print_status "🎉 Setup completed successfully!"
    echo
    echo "Next steps:"
    echo "  1. Configure configs/token_transfers.yaml if needed"
    echo "  2. Set OPENAI_API_KEY if not already set"
    echo "  3. Run inference: python evm_server.py process --env.data_path_to_save_groups evm_rollouts.jsonl"
    echo "  4. Or run training: python evm_server.py serve"
    echo
    echo "For troubleshooting, see README.md"
}

# Run main function
main "$@"
