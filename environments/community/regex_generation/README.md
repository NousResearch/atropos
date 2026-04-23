# Regex Generation Environment

An RL environment that trains language models to generate correct Python-compatible regular expressions from natural language descriptions and example test cases.

## How it works

Each problem gives the model:
- A natural language description of the pattern to match
- A set of strings that **should** match
- A set of strings that **should not** match

The model must produce a regex pattern inside `<answer>` tags. The pattern is tested using `re.fullmatch()` against all provided examples.

## Reward signal

The reward is the fraction of test cases passed (both positive and negative). A score of 1.0 means the regex correctly matches all positive examples and rejects all negative ones. Groups where all rollouts score identically are discarded (no learning signal).

## Problem set

The environment ships with 28 hand-crafted regex problems across three difficulty levels:

- **Easy**: Basic patterns (digits only, starts with X, exact match)
- **Medium**: Emails, dates, phone numbers, hex colors, zip codes
- **Hard**: IPv4 addresses, semantic versioning, URLs, repeated words

Problems are split 80/20 into train/test sets.

## Running

```bash
# Basic training
python regex_env.py serve \
    --env.tokenizer_name="NousResearch/DeepHermes-3-Llama-3-3B-Preview" \
    --openai.base_url http://localhost:9001/v1

# Only easy/medium problems
python regex_env.py serve \
    --env.difficulties='["easy", "medium"]'
```

## Config options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `difficulties` | list[str] | `["easy", "medium", "hard"]` | Difficulty levels to include |
| `score_threshold` | float | `1.0` | Min score to count as "correct" in metrics |

Standard `BaseEnvConfig` options (`group_size`, `max_token_length`, etc.) also apply.

## Eval metrics

| Metric | Description |
|--------|-------------|
| `eval/avg_score` | Average fraction of test cases passed |
| `eval/percent_perfect` | Fraction of problems with all tests passing |
| `eval/percent_valid_regex` | Fraction of responses with syntactically valid regex |
| `train/percent_correct` | Training accuracy (problems scoring above threshold) |

## Dependencies

No extra dependencies beyond what Atropos already provides. Uses only Python's built-in `re` module for regex validation.
