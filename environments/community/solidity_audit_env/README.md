# Solidity Smart Contract Security Audit Environment

An Atropos RL environment that trains LLMs to detect security vulnerabilities in Solidity smart contracts.

## Overview

This environment presents Solidity code snippets to an LLM and asks it to:
1. Determine if the code is vulnerable
2. Identify the vulnerability category (reentrancy, access control, integer overflow, etc.)
3. Explain the vulnerability
4. Suggest a fix

## Dataset

Uses [darkknight25/Smart_Contract_Vulnerability_Dataset](https://huggingface.co/datasets/darkknight25/Smart_Contract_Vulnerability_Dataset) from HuggingFace:
- ~2,000 Solidity code snippets with labeled vulnerabilities
- Fields: `code_snippet`, `category`, `description`, `severity`, `vulnerable`
- MIT License

## Scoring

Multi-component reward function (0.0 - 1.0):

| Component | Weight | Logic |
|---|---|---|
| Vulnerability detection | 0.25 | Binary match (vulnerable: true/false) |
| Category match | 0.35 | Fuzzy string similarity (SequenceMatcher) |
| Description quality | 0.25 | Keyword Jaccard similarity with ground truth |
| Format compliance | 0.15 | Valid YAML, all fields present, boxed format |

Additional behaviors:
- **Length penalty**: When all scores >= 0.9, shorter responses are rewarded
- **No learning signal**: Returns None when all scores are identical (standard Atropos pattern)

## Expected Output Format

```yaml
\boxed{
vulnerable: true
category: "reentrancy"
description: "The withdraw function calls an external address before updating the balance state variable"
fix: "Move the state update before the external call (checks-effects-interactions pattern)"
}
```

## Setup

```bash
pip install -r requirements.txt
```

## Running

### Process Mode (data generation, no training server needed)

```bash
cd environments/community/solidity_audit_env

python solidity_audit_env.py process \
  --env.data_path_to_save_groups audit_output.jsonl \
  --openai.base_url https://openrouter.ai/api/v1 \
  --openai.api_key $OPENROUTER_API_KEY \
  --openai.model_name qwen/qwen3-8b
```

### Training Mode (requires Atropos training server)

```bash
python solidity_audit_env.py run
```

## Testing

```bash
cd environments/community/solidity_audit_env
python -m pytest test_scoring.py -v
```

## File Structure

```
solidity_audit_env/
├── README.md                 # This file
├── solidity_audit_env.py     # Main environment (BaseEnv subclass)
├── scoring.py                # Reward function helpers
├── dataset_loader.py         # HuggingFace dataset loading & preprocessing
├── test_scoring.py           # Unit tests for scoring logic
└── requirements.txt          # Dependencies
```

## WandB Metrics

- `train/avg_reward` - Average reward across training batches
- `train/vuln_detection_accuracy` - Binary vulnerability detection accuracy
- `train/category_accuracy` - Category matching score
- `eval/avg_reward` - Average reward on evaluation set
