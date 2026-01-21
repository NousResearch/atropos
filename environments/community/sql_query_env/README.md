# SQL Query Generation Environment

Train LLMs to generate correct SQL queries from natural language questions.

## Overview

This environment uses the [Salesforce/WikiSQL](https://huggingface.co/datasets/Salesforce/wikisql) dataset to train language models on text-to-SQL tasks. Queries are verified by **executing** the generated SQL against in-memory SQLite databases and comparing results to ground truth.

## Dataset

- **Source**: [Salesforce/WikiSQL](https://huggingface.co/datasets/Salesforce/wikisql)
- **Size**: 80,654 examples (train + validation + test)
- **Format**: Natural language questions with table schemas and ground truth SQL

## Usage

### Training Mode (with API Server)

```bash
# Terminal 1: Start the Atropos API
run-api

# Terminal 2: Run the environment
python sql_query_env.py serve --slurm False
```

### Local Testing (without API)

```bash
python sql_query_env.py process --env.data_path_to_save_groups sql_output.jsonl
```

This generates `sql_output.jsonl` and `sql_output.html` for inspection.

### With Local vLLM Server

```bash
python sql_query_env.py process \
    --env.data_path_to_save_groups sql_output.jsonl \
    --openai.base_url http://localhost:9001/v1 \
    --openai.model_name YOUR_MODEL_NAME
```

## Reward Function

| Score | Condition |
|-------|-----------|
| **1.0** | Generated SQL executes and returns same result as gold SQL |
| **-1.0** | SQL fails to execute or returns incorrect result |

When all responses in a group are correct, a length penalty is applied to encourage concise solutions.

## Prompt Format

The model receives a table schema and question:

```
Table: data
Columns: col1, col2, col3
Sample data:
  value1 | value2 | value3

Question: What is the value of col1 where col2 equals X?
```

Output should be in boxed format:
```
<think>
[Chain of thought reasoning]
</think>

\boxed{SELECT col1 FROM data WHERE col2 = 'X'}
```

## Unit Tests

```bash
# Run unit tests
python -m pytest test_sql_executor.py -v
```

All 19 tests cover:
- Table creation with special column names
- SQL execution and error handling
- `\boxed{}` extraction patterns
- Result comparison and normalization
- End-to-end scoring integration

## LLM Integration Test

The environment has been verified with Qwen3-8B on an NVIDIA H200:

```bash
# Run integration test with a local vLLM server
python test_integration.py --base_url http://localhost:8000/v1 --model Qwen/Qwen3-8B
```

Test results:
- **40% accuracy** on 10 random WikiSQL examples
- SQL extraction from `\boxed{}` working correctly
- Execution-based scoring producing correct reward signals

## Files

| File | Description |
|------|-------------|
| `sql_query_env.py` | Main environment implementation |
| `sql_executor.py` | SQLite execution and scoring utilities |
| `wikisql_loader.py` | WikiSQL dataset loader (from GitHub) |
| `test_sql_executor.py` | Unit tests (19 tests) |
| `test_integration.py` | LLM integration test |

## Author

Community contribution to Atropos.
