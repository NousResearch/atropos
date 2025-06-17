# Ethereum Virtual Machine (EVM) Transaction Agent Environment

Atropos environment for training language models to generate and execute profitable Ethereum transactions.  An active forked version of the blockchain is created using Anvil (https://getfoundry.sh/guides/forking-mainnet-with-cast-anvil) to allow for execution and state inspection to verify transactions execute and perform the desired action.

## Overview

This environment trains language models to become proficient at text to transaction for EVM blockchains.  The existing config allows for ETH and ERC-20 transfers to be generated in natural language dynamically by LLM calls.  These are designed to target different types of transactions with increasing frequency towards those transaction types that the model is scoring poorly on. The agent learns to handle ETH transfers, ERC-20 token transfers, and complex DeFi interactions through reinforcement learning.

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

## Verified Scoring System with Anvil

Unlike traditional RL environments that rely on simulated or estimated rewards, this environment provides **cryptographically verified rewards** by executing transactions on a real Ethereum Virtual Machine simulation powered by Anvil. This ensures that scoring is based on actual blockchain state changes rather than heuristic approximations.

### Anvil-Powered Verification

**Anvil** (Foundry's blockchain simulator) enables true verification by:

- **Real EVM Execution**: Transactions run on an actual Ethereum Virtual Machine, not a simplified simulation
- **Mainnet Fork**: Uses real mainnet state with actual token contracts and balances
- **Cryptographic Verification**: Transaction success/failure is determined by EVM consensus rules
- **Atomic State Management**: Blockchain snapshots ensure clean evaluation without side effects
- **Gas Estimation**: Real gas consumption and fee calculation for realistic training

### Scoring Methodology

The environment employs a **snapshot-execute-verify-revert** cycle for each transaction:

```
1. Snapshot blockchain state
2. Record pre-execution balances
3. Execute agent's transaction
4. Measure actual state changes
5. Calculate verified score
6. Revert to clean snapshot
```

This process ensures that:
- ✅ **No False Positives**: Only correctly executed transactions receive rewards
- ✅ **Precise Measurement**: Exact balance changes are measured, not estimated
- ✅ **Isolated Evaluation**: Each transaction is evaluated independently
- ✅ **Real-World Validity**: Successful transactions would work on actual mainnet

### Five-Dimensional Scoring

The reward function evaluates transactions across five verified dimensions:

1. **Correct Balance Changes (0.5 points)**:
   - **Most Critical Component**: Measures actual on-chain balance differences
   - Compares pre/post execution balances with cryptographic precision
   - For ETH: Exact wei amounts transferred to destination
   - For ERC-20: Exact token units transferred (accounting for decimals)
   - Verified against real contract state, not estimated

2. **Successful Execution (0.3 points)**:
   - Verified by EVM status code (`0x1` = success)
   - Ensures transaction doesn't revert due to insufficient funds, gas, or logic errors
   - Only awarded if transaction is mined successfully

3. **Thinking Quality (±0.1 points)**:
4. **Destination Address Accuracy (0.05 points)**:
5. **Data Field Correctness (0.05 points)**:

**Total Score Range**: -0.2 to 1.0
- **Perfect execution**: 1.0 (all components correct)
- **Missing thinking**: -0.2 (penalty for unexplained decisions)
- **Partial success**: Proportional scoring based on verified components

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

## Potential Training Applications

- **DeFi Agent Development**: Training models for decentralized finance interactions
- **Transaction Automation**: Building agents for routine blockchain operations
- **Smart Contract Interaction**: Learning to encode function calls and parameters
- **Risk Assessment**: Understanding transaction costs and failure modes
- **Multi-Chain Operations**: Foundation for cross-chain transaction agents
