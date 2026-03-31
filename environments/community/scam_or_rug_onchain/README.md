# Scam or Rug On-Chain Environment

A reinforcement learning environment that trains LLMs to detect token scams and rug pulls from the perspective of an average Web3 user.

## Overview

This environment challenges the model to analyze raw on-chain token data and classify tokens as `SCAM`, `RUG_RISK`, or `LEGITIMATE` — then calculate the estimated price impact if malicious actors dump their holdings.

The detection logic is grounded in real-world Web3 experience, covering the most common attack vectors used in BSC, Ethereum, and Solana token scams.

## Detection Layers

1. **Cluster Holdings** — identifies connected wallets funded from the same source holding large % of supply (bundled supply attack)
2. **Liquidity Pool** — checks LP holder type, lock duration, and whether LP is genuinely burned or just unlocked
3. **Mint Authority** — flags tokens where dev can still print new supply
4. **Tax Mechanism** — scores buy/sell tax: 0–1% healthy, 1–5% caution, >5% red flag, >10% danger zone / semi-honeypot
5. **Burn Validity** — distinguishes real burn addresses (`0x000...000`, `0x000...dead`) from fake ones, and checks for `recoverTokens()` / `emergencyWithdraw()` backdoors
6. **Honeypot Detection** — checks whether holders can actually sell, and flags suspiciously high sell tax
7. **Wash Trading** — analyzes buy/sell ratio and unique trader count to detect artificial volume inflation

## Task Format

The model receives structured on-chain data and must respond in this format:
```
CLASSIFICATION: <SCAM|RUG_RISK|LEGITIMATE>
REASONING: <explanation across all detection layers>
DUMP IMPACT: <X>% price drop if cluster dumps (or N/A if LEGITIMATE)
CALCULATION: <show math>
```

## Reward Function

Scores are computed across three components:

| Component | Weight | Description |
|---|---|---|
| Classification | 0.4 | Correct label gets full score, adjacent label (SCAM↔RUG_RISK) gets partial |
| Reasoning | 0.3 | Keywords matched across all 7 detection dimensions |
| Math accuracy | 0.3 | Dump impact within 2% → full, within 5% → partial, within 10% → minimal |

## Dump Impact Formula
```
tokens_dumped = total_supply × (cluster_pct / 100)
current_price = lp_value / total_supply
new_price = lp_value / (total_supply + tokens_dumped)
price_drop = (current_price - new_price) / current_price × 100
```

## Data Generation

All token data is synthetically generated with realistic value ranges derived from real-world scam patterns observed across EVM and Solana chains. Each training round randomizes all variable values while keeping the underlying detection logic consistent — forcing the model to generalize rather than memorize.

## Scope

This environment covers non-utility tokens with fixed supply (not gas tokens or established governance tokens). Upgradeable contract detection is included as a proxy for potential mint-after-burn attacks.

## Author

Contributed by [@ILKokoron](https://github.com/ILKokoron)