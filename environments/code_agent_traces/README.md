# Code Agent Traces

Interleaved reasoning pipeline for code generation with **four modes**:

| Mode | File | Interleaving | Tool Use | Purpose |
|------|------|:------------:|:--------:|---------|
| Marker-Based | `trace_generator.py` | ✓ (forced) | ✗ | Fast JSONL generation |
| Tool-Based | `trace_generator_tools.py` | ✗ | ✓ | Execution feedback |
| **Interleaved+Tools** | `trace_generator_interleaved_tools.py` | ✓ | ✓ | **IDEAL training data** |
| RL Environment | `interleaved_code_env.py` | ✓ | ✓ | Online RL training |

## Quick Start

```bash
# Set API key
export OLLAMA_API_KEY=your_key

# RECOMMENDED: Interleaved + Tool traces (best quality)
python trace_generator_interleaved_tools.py --output traces.jsonl --num-traces 10

# Only ideal traces (interleaved + tool + success)
python trace_generator_interleaved_tools.py --output traces.jsonl --only-ideal

# Marker-based (fast, no execution feedback)
python trace_generator.py --output traces.jsonl --num-traces 10 --force-interleave

# Tool-based (execution feedback, but monolithic code)
python trace_generator_tools.py --output traces.jsonl --num-traces 10

# Or run RL training with Atropos
python interleaved_code_env.py serve --config config.yaml
```

## Two Generation Approaches

### Approach 1: Marker-Based (trace_generator.py)

```
┌─────────┐  prompt   ┌─────────┐   text    ┌─────────┐
│ Script  │ ────────► │  LLM    │ ────────► │ Parser  │
└─────────┘           └─────────┘           └─────────┘
                                                 │
                                                 ▼
                                           regex parsing
                                           [THINK]/[CODE]
                                                 │
                                                 ▼
                                           ┌─────────┐
                                           │Executor │ (post-hoc)
                                           └─────────┘
```

- Model outputs entire solution with markers
- Code executed AFTER generation to score
- Fast but model doesn't see execution results

### Approach 2: Tool-Based (trace_generator_tools.py)

```
┌─────────┐  prompt   ┌─────────┐
│ Script  │ ────────► │  LLM    │
└─────────┘           └─────────┘
     ▲                     │
     │                     ▼
     │              [CODE] block
     │                     │
     │                     ▼
     │              ┌─────────┐
     │              │Executor │
     │              └─────────┘
     │                     │
     │    [RESULT]/[ERROR] │
     └─────────────────────┘
```

- Multi-turn conversation with execution feedback
- Model sees test results/errors and can fix bugs
- Creates richer training data with error-recovery patterns

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
├── trace_generator.py       # Marker-based JSONL generation
├── trace_generator_tools.py # Tool-based JSONL (with execution feedback)
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

## Mode 2: Tool-Based Generator (trace_generator_tools.py)

Generates traces with **actual code execution feedback**. Model can iterate and fix bugs.

### Usage

```bash
# Basic usage
python trace_generator_tools.py --output traces.jsonl --num-traces 10

# Training format (single assistant message with full trace)
python trace_generator_tools.py --output traces.jsonl --training-format

# Only successful traces
python trace_generator_tools.py --output traces.jsonl --only-success

# Limit iterations per problem
python trace_generator_tools.py --output traces.jsonl --max-iterations 3
```

### Example Trace Output

```
[THINK] I need to find two indices that sum to target. I'll use a hash map.
[CODE]
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        if target - num in seen:
            return [seen[target-num], i]
        seen[num] = i
[/CODE]
[RESULT]
Test 1: PASS - two_sum([2,7,11,15], 9)
Test 2: FAIL - two_sum([3,3], 6) expected [0, 1]
2/4 tests passed
[/RESULT]
[THINK] Test 2 failed - need to check duplicate handling. Actually the logic is correct,
but I forgot to return [] when no solution. Let me trace through [3,3]:
- i=0: num=3, complement=3, seen={} -> not found, add seen[3]=0
- i=1: num=3, complement=3, seen={3:0} -> found! return [0,1]
Wait, this should work. The issue must be the missing return statement at the end.
[CODE]
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        if target - num in seen:
            return [seen[target-num], i]
        seen[num] = i
    return []
[/CODE]
[RESULT]
All 4 tests passed!
[/RESULT]
[VERIFY]
The solution uses O(n) time with hash map lookup. For each number,
we check if (target - num) exists in seen. The duplicate case [3,3]
works because we check before storing.
[/VERIFY]
```

### Output Format

```json
{
    "problem": "Write a function two_sum...",
    "code": "def two_sum(nums, target):\n    ...",
    "score": 1.0,
    "tests_passed": 4,
    "tests_total": 4,
    "think_count": 2,
    "code_iterations": 2,
    "has_verify": true,
    "had_errors": true,
    "trace": [
        {"type": "think", "content": "I need to find..."},
        {"type": "code", "content": "def two_sum..."},
        {"type": "result", "content": "Test 1: PASS..."},
        {"type": "think", "content": "Test 2 failed..."},
        {"type": "code", "content": "def two_sum..."},
        {"type": "result", "content": "All 4 tests passed!"},
        {"type": "verify", "content": "The solution uses..."}
    ]
}
```

### Why Use This?

| Metric | Marker-Based | Tool-Based |
|--------|--------------|------------|
| Speed | Fast (1 LLM call) | Slower (multi-turn) |
| Error recovery | No | Yes, model fixes bugs |
| Training quality | Good | Better (shows debugging) |
| Trace richness | Think→Code→Verify | Think→Code→Result→Fix→Result→Verify |

---

## Mode 3: Interleaved + Tools (IDEAL)

**`trace_generator_interleaved_tools.py`** - Combines BOTH dimensions for optimal training data.

### The Two Dimensions

```
                    Interleaved Reasoning
                           ↑
        ┌──────────────────┼──────────────────┐
        │   trace_         │   interleaved_   │
        │   generator.py   │   tools.py       │
        │   (forced)       │   (THIS ONE)     │
        │                  │   ★ IDEAL ★      │
────────┼──────────────────┼──────────────────┼────→ Tool Use
        │   trace_         │   trace_         │
        │   generator.py   │   generator_     │
        │   (default)      │   tools.py       │
        └──────────────────┴──────────────────┘
```

### Usage

```bash
# Generate ideal traces
python trace_generator_interleaved_tools.py --output traces.jsonl --num-traces 10

# Only keep ideal traces (interleaved + tool + success)
python trace_generator_interleaved_tools.py --output traces.jsonl --only-ideal

# Training format (single assistant message)
python trace_generator_interleaved_tools.py --output traces.jsonl --training-format
```

### Example Output

```
[THINK] I'll start by defining the function and initializing pointers.
[CODE]
def binary_search(nums, target):
    left, right = 0, len(nums) - 1
[/CODE]

[THINK] Now I need a loop while the interval is valid.
[CODE]
    while left <= right:
        mid = (left + right) // 2
[/CODE]

[THINK] Compare mid element with target, handle three cases.
[CODE]
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
[/CODE]

[THINK] If we exit the loop, target wasn't found.
[CODE]
    return -1
[/CODE]

[RESULT]
Test 1: PASS - binary_search([1,2,3,4,5], 3) = 2
Test 2: PASS - binary_search([1,2,3,4,5], 6) = -1
All 4 tests passed!
[/RESULT]

[VERIFY]
Binary search divides search space in half each iteration.
For [1,2,3,4,5] with target=3:
- left=0, right=4, mid=2, nums[2]=3 == target -> return 2
```

### Quality Metrics

| Metric | Description | Ideal Value |
|--------|-------------|-------------|
| `think_count` | Number of [THINK] blocks | ≥ 3 |
| `code_block_count` | Number of [CODE] blocks | ≥ 3 |
| `execution_count` | Real code executions | ≥ 1 |
| `is_ideal` | All conditions met + success | `true` |

---

## Mode 4: RL Environment (Atropos)

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
