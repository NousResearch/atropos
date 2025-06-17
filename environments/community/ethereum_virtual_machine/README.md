# Ethereum Virtual Machine (EVM) Transaction Agent Environment

Atropos environment for training language models to generate and execute profitable Ethereum transactions using Anvil (Foundry's local blockchain simulation).

## Overview

This environment trains language models to become proficient Ethereum transaction agents. It presents natural language transaction requests and rewards models for generating correct transaction JSON that successfully executes on a simulated blockchain. The agent learns to handle ETH transfers, ERC-20 token transfers, and complex DeFi interactions through reinforcement learning.

## Features

- **Complete EVM Training Environment**: Full implementation of the BaseEnv interface for Atropos
- **Anvil Blockchain Simulation**: Local Ethereum fork for safe transaction testing
- **Multi-Token Support**: ETH and major ERC-20 tokens (USDC, USDT, DAI, WETH, CRV)
- **Dynamic Question Generation**: LLM-powered generation of realistic transaction requests
- **Comprehensive Scoring System**: Multi-dimensional evaluation of transaction correctness
- **Adaptive Learning**: Performance-based question type selection for targeted improvement
- **Robust Cleanup**: Graceful handling of interruptions and proper resource management

## Files

- **evm_server.py**: Main environment implementation with transaction scoring logic
- **anvil.py**: Anvil blockchain backend management with integrated configuration
- **configs/token_transfers.yaml**: Blockchain simulation configuration
- **utils.py**: Cleanup handlers and utility functions

## Transaction Types

The environment trains on three primary transaction categories:

1. **ETH Transfer**: Simple Ether transfers between addresses
2. **ERC-20 Transfer (18 decimals)**: Standard token transfers (DAI, WETH, CRV)
3. **ERC-20 Transfer (non-18 decimals)**: Tokens with different decimal precision (USDC, USDT)

## Scoring System

The reward function evaluates transactions across five dimensions:

1. **Successful Execution (30%)**: Transaction executes without reverting
2. **Correct Balance Changes (50%)**: Expected token/ETH amounts transferred
3. **Thinking Quality (10%)**: Depth and quality of reasoning in `<think>` tags
4. **Address Accuracy (5%)**: Correct destination address matching
5. **Data Field Correctness (5%)**: Proper transaction data encoding

**Score Range**: -0.2 to 1.0 (penalties for missing thinking, rewards for correct execution)

## Prerequisites

### System Requirements
- Python 3.8+
- [Foundry](https://book.getfoundry.sh/) (includes Anvil and Cast)
- OpenAI API key

### Installing Foundry/Anvil

**Quick Install (Recommended)**
```bash
curl -L https://foundry.paradigm.xyz | bash
foundryup
```

**Verify Installation:**
```bash
anvil --version
cast --version
forge --version
```

## Setup

1. **Install Python dependencies:**
   ```bash
   pip install openai pydantic PyYAML
   ```

2. **Set OpenAI API key:**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

3. **Verify configuration:**
   ```bash
   python -c "from anvil import AnvilConfig; config = AnvilConfig(); print('Config loaded successfully')"
   ```

## Usage

### Running the Environment

**For inference-only rollouts:**
```bash
cd environments/community/ethereum_virtual_machine/
python evm_server.py process \
    --env.data_path_to_save_groups evm_rollouts.jsonl \
    --openai.model_name gpt-4o-mini
```

**For full training with server:**
```bash
python evm_server.py serve
```

### Configuration

The environment uses `configs/token_transfers.yaml` for blockchain configuration:

- **Network Settings**: Port (8545), chain ID, block time
- **Fork Configuration**: Mainnet fork at specific block
- **Wallet Setup**: Custom wallet funding and token swaps
- **Gas Settings**: Limit and price configuration
- **Token Addresses**: Whitelisted ERC-20 tokens

## Environment Architecture

### Question Generation
- Uses GPT-4o-mini to generate realistic transaction requests
- Adapts to current wallet balances and available tokens
- Varies tone and complexity (casual, formal, urgent styles)

### Transaction Execution
- Creates blockchain snapshots before execution
- Executes transactions using Foundry's `cast send`
- Measures balance changes for scoring
- Reverts to snapshot to maintain clean state

### Adaptive Learning
- Tracks performance by question type
- 80/20 strategy: focuses on weak areas while maintaining mastery
- Targets question types with <90% performance

## Training Applications

- **DeFi Agent Development**: Training models for decentralized finance interactions
- **Transaction Automation**: Building agents for routine blockchain operations
- **Smart Contract Interaction**: Learning to encode function calls and parameters
- **Risk Assessment**: Understanding transaction costs and failure modes
- **Multi-Chain Operations**: Foundation for cross-chain transaction agents
