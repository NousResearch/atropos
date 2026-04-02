# Crypto DeFi Reasoning Environment

## Overview
Trains LLMs to reason about DeFi protocol health, tokenomics, liquidity metrics,
and on-chain data using structured mathematical analysis.

The model is given real-world style DeFi scenarios and must:
1. Reason through the data mathematically inside `<reasoning>` tags
2. Produce a verdict: **HEALTHY**, **OVERVALUED**, **UNDERVALUED**, or **RISKY**

## Scenario Types
| Type | Description |
|---|---|
| Tokenomics Health | FDV/MCap ratio, circulating supply pressure, volume analysis |
| Liquidity Analysis | TVL/MCap ratio, protocol revenue, inflation rate |
| DeFi Yield | APY sustainability, token emission pressure on TVL |
| On-Chain Activity | Active wallets, transaction volume vs market cap |

## Reward Signal
| Component | Score |
|---|---|
| Correct verdict label | 0.5 |
| Has `<reasoning>` block | 0.2 |
| Reasoning contains numerical calculations | 0.1 |
| Reasoning depth (>100 chars) | 0.1 |
| Has `<verdict>` block | 0.1 |

## Usage
```bash
python environments/community/crypto_defi_reasoning/crypto_defi_reasoning_server.py serve \
  --openai.model_name NousResearch/Hermes-3-Llama-3.1-8B \
  --slurm false
```

## Author
Investorquab — community contributor
