# NL2Bash Generation Environment

Train LLMs to translate natural language instructions into Bash commands.

## Overview

This environment uses the [NL2SH-ALFA](https://huggingface.co/datasets/westenfelder/NL2SH-ALFA) dataset to train language models on natural language to Bash translation. Commands are verified by **string matching** against gold standard commands.

## Dataset

- **Source**: [westenfelder/NL2SH-ALFA](https://huggingface.co/datasets/westenfelder/NL2SH-ALFA)
- **Paper**: [LLM-Supported Natural Language to Bash Translation](https://arxiv.org/abs/2502.06858) (NAACL 2025)
- **Training Set**: 40,939 instruction-command pairs
- **Test Set**: 300 manually verified pairs with alternative commands and difficulty levels

### Sample Data

```json
{
  "nl": "find all files in the current directory with the extension .txt and delete them",
  "bash": "find . -name \"*.txt\" -delete",
  "bash2": "find . -type f -name \"*.txt\" -exec rm {} +",
  "difficulty": 1
}
```

## Usage

### Training Mode (with API Server)

```bash
# Terminal 1: Start the Atropos API
run-api

# Terminal 2: Run the environment
python bash_env.py serve --slurm False
```

### Local Testing (without API)

```bash
python bash_env.py process --env.data_path_to_save_groups bash_output.jsonl
```

This generates `bash_output.jsonl` and `bash_output.html` for inspection.

### With Local vLLM Server

```bash
python bash_env.py process \
    --env.data_path_to_save_groups bash_output.jsonl \
    --openai.base_url http://localhost:9001/v1 \
    --openai.model_name YOUR_MODEL_NAME
```

## Reward Function

| Score | Condition |
|-------|-----------|
| **1.0** | Generated command matches gold or alternative (exact or normalized) |
| **-1.0** | Command does not match or could not be extracted |

String matching is used instead of execution-based verification because:
1. Bash execution without sandboxing is unsafe
2. Many commands have side effects (file creation/deletion, network calls)
3. The dataset was designed for string-based evaluation

## Prompt Format

The model receives a natural language instruction:

```
Instruction: find all files in the current directory with the extension .txt and delete them
```

Output should be in boxed format:
```
<think>
[Chain of thought reasoning]
</think>

\boxed{find . -name "*.txt" -delete}
```

## Unit Tests

```bash
# Run unit tests
python -m pytest test_bash_utils.py -v
```

Tests cover:
- Bash command normalization
- `\boxed{}` extraction patterns
- String matching with alternatives
- Basic syntax validation

## Integration Test

```bash
# Run with a local vLLM server
python test_integration.py --base_url http://localhost:8000/v1 --model Qwen/Qwen3-8B

# Test on training set instead
python test_integration.py --base_url http://localhost:8000/v1 --model Qwen/Qwen3-8B --use_train
```

The test reports overall accuracy and difficulty-stratified accuracy (easy/medium/hard).

## Files

| File | Description |
|------|-------------|
| `bash_env.py` | Main environment implementation |
| `bash_utils.py` | Bash command processing utilities |
| `nl2bash_loader.py` | NL2SH-ALFA dataset loader |
| `test_bash_utils.py` | Unit tests for utilities |
| `test_integration.py` | LLM integration test |

## Evaluation Metrics

The environment logs the following metrics to WandB:

- `train/percent_correct` - Training accuracy
- `eval/percent_correct` - Overall test accuracy
- `eval/accuracy_easy` - Accuracy on easy problems (difficulty=0)
- `eval/accuracy_medium` - Accuracy on medium problems (difficulty=1)
- `eval/accuracy_hard` - Accuracy on hard problems (difficulty=2)
