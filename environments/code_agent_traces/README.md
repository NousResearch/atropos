# Code Agent Traces

Interleaved reasoning pipeline for code generation with **two modes**:

| Mode | File | Purpose |
|------|------|---------|
| **Trace Generator** | `trace_generator.py` | Standalone JSONL generation for fine-tuning |
| **RL Environment** | `interleaved_code_env.py` | Atropos integration for online RL training |

## Quick Start

```bash
# Set API key
export OLLAMA_API_KEY=your_key

# Generate synthetic traces → JSONL
python trace_generator.py --output traces.jsonl --num-traces 10

# Or run RL training with Atropos
python interleaved_code_env.py serve --config config.yaml
```

## Interleaved Reasoning Format

Models use `[THINK]`, `[CODE]`, `[VERIFY]` markers:

```
[THINK] I need a hash map for O(n) lookup
[CODE]
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        if target - num in seen:
            return [seen[target - num], i]
        seen[num] = i
[/CODE]
[VERIFY]
Test: [2,7,11,15], target=9 → [0,1] ✓
[/VERIFY]
```

## Architecture

```
code_agent_traces/
├── trace_generator.py       # Standalone JSONL generation
├── interleaved_code_env.py  # Atropos RL environment
├── interleaved_agent.py     # Interactive demo
├── run_structured_pipeline.py # Structured pipeline demo
├── local_executor.py        # Safe code execution
└── README.md
```

## Requirements

```bash
# Core
pip install aiohttp rich

# For trace generator
pip install datasets  # optional, for HumanEval

# For RL environment (Atropos)
pip install atroposlib wandb transformers
```

## Configuration

```bash
# Ollama Cloud
export OLLAMA_BASE_URL=https://ollama.com
export OLLAMA_API_KEY=your_api_key
export OLLAMA_MODEL=deepseek-v3.2

# Local Ollama
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=qwen2.5-coder:7b
```

---

## Mode 1: Trace Generator (JSONL)

Standalone script for generating fine-tuning data.

### Usage

```bash
# Basic usage (single-call, model may output monolithic response)
python trace_generator.py --output traces.jsonl --num-traces 10

# RECOMMENDED: Force true interleaving via multi-turn conversation
python trace_generator.py --output traces.jsonl --num-traces 10 --force-interleave

# Only successful traces (score > 0)
python trace_generator.py --output traces.jsonl --only-success --force-interleave

# Simple chat format for fine-tuning
python trace_generator.py --output traces.jsonl --chat-format --force-interleave
```

### Interleaving Modes

| Mode | Flag | Description |
|------|------|-------------|
| Single-call | (default) | Fast but model often outputs monolithic Plan→Code→Verify |
| **Forced interleave** | `--force-interleave` | Multi-turn conversation forces Think→Code→Think→Code... |

**Recommended**: Use `--force-interleave` for proper training data with granular reasoning steps.

### Output Format (Full)

```json
{
    "problem": "Write a function two_sum...",
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "[THINK]...[CODE]...[VERIFY]..."}
    ],
    "code": "def two_sum(nums, target):\n    ...",
    "score": 1.0,
    "tests_passed": 4,
    "tests_total": 4,
    "think_count": 3,
    "has_verify": true,
    "trace": [
        {"type": "think", "content": "I need a hash map..."},
        {"type": "code", "content": "def two_sum..."},
        {"type": "verify", "content": "Test: [2,7,11,15]..."}
    ]
}
```

### Output Format (Chat - for fine-tuning)

```json
{"messages": [...], "score": 1.0}
```

---

## Mode 2: RL Environment (Atropos)

Full Atropos integration for online RL training.

### Usage

```bash
# Serve for training
python interleaved_code_env.py serve --config config.yaml

# Process mode (generate without training)
python interleaved_code_env.py process \
    --env--total_steps 100 \
    --env--data_path_to_save_groups data/traces.jsonl

# Evaluation only
python interleaved_code_env.py evaluate
```

### Config Example

```yaml
# config.yaml
env:
  group_size: 8
  max_token_length: 4096
  dataset_name: openai/openai_humaneval
  partial_credit: true
  think_bonus: 0.1
  verify_bonus: 0.1

openai:
  base_url: http://localhost:9004/v1
  model_name: your-model
```

### Reward Structure

| Component | Reward |
|-----------|--------|
| All tests pass | +1.0 |
| Partial success | -1.0 + 2×(passed/total) |
| Execution error | -1.0 |
| ≥2 [THINK] markers | +0.1 bonus |
| [VERIFY] marker | +0.1 bonus |

### WandB Metrics

- `train/percent_correct` - Success rate
- `train/avg_think_count` - Average [THINK] usage
- `train/verify_rate` - [VERIFY] usage rate
- `eval/accuracy` - Test set accuracy

---

## Demo Scripts

```bash
# Interactive interleaved agent
python interleaved_agent.py --example 0

# Structured Planning-Action-Reflection pipeline
python run_structured_pipeline.py --example 0
```

---

## Supported Models

**Cloud (ollama.com):**
- `deepseek-v3.2` - Recommended
- `deepseek-v3.1`

**Local:**
- `qwen2.5-coder:7b`
- `deepseek-r1:7b`
- Any Ollama-compatible model
