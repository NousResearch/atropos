# BLEUBERI Environment for Atropos

This environment implements the BLEUBERI approach for instruction-following using BLEU scores as rewards. BLEUBERI (BLEU-Based Enhanced Utility for Better Evaluating Reward in Instruction-following) demonstrates that BLEU scores, when paired with high-quality references from strong LLMs, can be highly effective rewards for training models to follow instructions.

## Overview

BLEUBERI uses BLEU scores (a simple n-gram matching metric) directly as rewards in a Group Relative Policy Optimization (GRPO) training framework. The approach:

1. Collects high-quality reference responses from top LLMs (Claude, Gemini, etc.)
2. Computes BLEU scores by comparing model outputs to these references
3. Uses these scores as rewards to train models through GRPO

## Features

- BLEU-based reward functions (with support for multiple reference models)
- Compatible with the Atropos asynchronous environment framework
- Support for both SFT and GRPO training approaches
- Evaluation on instruction-following benchmarks

## Usage

```bash
# Run the BLEUBERI environment
python -m atroposlib.cli.dpo --env-module environments.bleuberi.bleuberi_env

# Generate data with pre-collected references
python -m environments.bleuberi.bleuberi_env process --config environments/bleuberi/configs/default.yaml
```

## Configuration

See the `configs/` directory for example configurations. The environment supports:

- Using pre-collected references or generating references on-the-fly
- Multiple reference models for more robust BLEU scoring
- Various BLEU calculation parameters
- Different dataset sources (default: Tulu3 mixture)

## References

This implementation is based on the paper [BLEUBERI: BLEU is a surprisingly effective reward for instruction following](https://arxiv.org/abs/2505.11080) and its [original implementation](https://github.com/lilakk/BLEUBERI).
