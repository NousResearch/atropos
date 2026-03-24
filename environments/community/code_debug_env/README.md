# Code Debug Environment

An Atropos RL environment for training LLMs to debug and fix buggy Python code.

## Overview

This environment uses the [HumanEvalPack](https://huggingface.co/datasets/bigcode/humanevalpack) dataset (Python subset, HumanEvalFix task), which contains 164 buggy Python functions with associated test suites. The model receives a buggy function and must output the corrected version inside `\boxed{}`. Scoring is done by executing the fixed code against the original test cases.

## Architecture

```
code_debug_env.py    # Main env (extends BaseEnv)
code_executor.py     # Safe subprocess execution with timeout
test_code_debug.py   # Unit tests
README.md            # This file
```

## Reward Design

| Outcome | Score | Description |
|---------|-------|-------------|
| All tests pass | **1.0** | Perfect fix |
| Partial improvement | **-0.5 to 0.9** | More tests pass than buggy version |
| No improvement | **-0.5** | Code runs but doesn't fix anything |
| Compilation error / regression | **-1.0** | Fix is worse than the original |

When all rollouts in a group score 1.0, a **length penalty** is applied to encourage concise solutions (same pattern as `sql_query_env`).

## Setup

```bash
# Install dependencies (datasets is the only extra)
pip install datasets

# Run tests
cd environments/community/code_debug_env
python -m pytest test_code_debug.py -v
```

## Usage

```bash
# Process mode (offline data generation)
python code_debug_env.py process \
    --env.data_path_to_save_groups data/code_debug.jsonl \
    --env.group_size 8 \
    --openai.base_url http://localhost:8000/v1 \
    --openai.model_name "NousResearch/DeepHermes-3-Llama-3-3B-Preview"

# Serve mode (online RL training)
python code_debug_env.py serve \
    --openai.base_url http://localhost:9001/v1 \
    --openai.model_name "NousResearch/DeepHermes-3-Llama-3-3B-Preview"

# Evaluate mode
python code_debug_env.py evaluate \
    --openai.base_url http://localhost:8000/v1 \
    --openai.model_name "NousResearch/DeepHermes-3-Llama-3-3B-Preview"
```

## WandB Metrics

| Metric | Description |
|--------|-------------|
| `train/percent_correct` | Fraction of rollouts that pass all tests |
| `train/avg_score` | Average reward across rollouts |
| `train/partial_fix_rate` | Fraction of rollouts that partially fix the code |
| `eval/percent_correct` | Eval set accuracy |

## Dataset

- **Source**: [bigcode/humanevalpack](https://huggingface.co/datasets/bigcode/humanevalpack) (Python subset)
- **License**: Apache 2.0
- **Size**: 164 problems
- **Split**: 80% train / 20% test

## Compute Footprint

- **RAM**: < 1 GB (dataset is small, execution is in subprocess)
- **CPU**: < 5s per verification (subprocess with 10s timeout)
- **GPU**: Only needed for the inference server
