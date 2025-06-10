# Environments

This directory contains various environments for training and evaluating language models on different tasks. Each environment implements a specific task with its own input format, reward function, and evaluation metrics.

## Available Environments

---

###  MCQA Thinking Environment (`mcqa_thinking_env.py`)

Multiple Choice Question Answering environment that requires models to think through problems systematically.

**Input Format:**
- Questions from the MMLU (Massive Multitask Language Understanding) dataset
- Each item contains:
  - `prompt`: The question text
  - `answer`: Index of correct answer
  - `ground_truth`: Letter (A, B, C, D) of correct answer
  - `options`: List of possible answers

**System Prompt:**
```
You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem.
```

**Reward Function:**
- Score of 1.0 if the model's answer matches the ground truth letter
- Score of 0.0 if incorrect or invalid response (multiple think tags, malformed thinking sections)
- Length penalty applied if all responses are correct:
  - No penalty for responses under 50% of max token length
  - Linear penalty scaling from 1.0 down to 0.0 for responses between 50% and 100% of max length
  - Returns None if all scores are identical (no learning signal)

---

### GSM8K Environment (`gsm8k_server.py`)

Mathematical reasoning environment using the GSM8K dataset.

**Input Format:**
- Questions from GSM8K dataset
- Each item contains:
  - `question`: The math problem
  - `answer`: The numerical answer

**System Prompt:**
```
You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem.

You are allocated a maximum of 2048 tokens, please strive to use less.

You will then provide your answer like this: \boxed{your answer here}
It is important that you provide your answer in the correct format.
If you do not, you will not receive credit for your answer.
So please end your answer with \boxed{your answer here}
```

**Reward Function:**
- Score of 1.0 if the model's answer matches the ground truth (using LaTeX verification)
- Score of 0.0 if incorrect or if ground truth is not parseable
- Length penalty applied if all responses are correct:
  - No penalty for responses under 50% of max token length
  - Linear penalty scaling from 1.0 down to 0.0 for responses between 50% and 100% of max length
  - Returns None if all scores are identical (no learning signal)

---

### Tool Calling Environment (`tool_calling_server.py`)

Environment for training models to make function calls in a structured format.

**Input Format:**
- Conversations from ShareGPT-Hermes function call dataset
- Each item contains:
  - `conversations`: List of messages with roles (system, human, gpt)
  - Expected tool calls in JSON format

**System Prompt:**
```
You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem.
```

**Reward Function:**
- Score of 1.0 if all expected tool calls are present and match exactly (including nested JSON fields)
- Score of 0.0 if any tool calls are missing, incorrect, or malformed
- Length penalty applied if all responses are correct:
  - No penalty for responses under 50% of max token length
  - Linear penalty scaling from 1.0 down to 0.0 for responses between 50% and 100% of max length
  - Returns None if all scores are identical (no learning signal)

---

### LLM Equivalency Judge Environment (`lm_equivalence_judge_environment.py`)

Textbook question answering environment that uses LLM-based judging to evaluate response equivalency.

**Input Format:**
- Questions from the TextbooksToRLQuestions-100k dataset
- Each item contains:
  - `question`: The textbook question
  - `solution`: The ground truth answer/solution

**System Prompt:**
```
You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem.
```

**Reward Function:**
- Score of 1.0 if the model's answer is judged equivalent to ground truth by LLM judge
- Score of 0.0 if:
  - Response was cut off due to length limits
  - Missing or malformed thinking tags (not exactly one `<think></think>` pair)
  - LLM judge determines answer is not equivalent to ground truth
- Uses either OpenAI API or local model for equivalency judging
- Judge evaluates semantic equivalence, numerical reasoning, and format compliance
- Returns None if all scores are identical (no learning signal)

**Judge Configuration:**
- Configurable to use OpenAI API (gpt-4.1-nano by default) or local model
- Automatic fallback to local model if OpenAI API is unavailable
- Environment variables: `OAI_API_KEY`/`OPENAI_API_KEY`, `OAI_MODEL`
- Detailed evaluation logging for debugging judge decisions

## Common Features

All environments share these common features:

1. **Training/Test Split:**
   - 98% training, 2% test split
   - Random shuffling with fixed seed (42)

2. **Metrics Tracking:**
   - Percent correct buffer
   - Completion lengths
   - Wandb integration for visualization
   - Rollout tracking

3. **Token Management:**
   - Maximum token length limits
   - Token length statistics tracking
   - Length penalty for excessive responses

4. **Evaluation:**
   - Separate evaluation on test set
   - Comprehensive metrics logging
   - Support for multiple model completions per prompt

## Usage

Each environment can be initialized with:
- `config`: BaseEnvConfig object
- `server_configs`: List of OpenAI API configurations
- `slurm`: Boolean for distributed training
- `testing`: Boolean for testing mode

The environments follow a common interface with methods for:
- `setup()`: Loading and preparing datasets
- `get_next_item()`: Retrieving next training item
- `collect_trajectories()`: Generating model responses
- `score()`: Computing rewards
- `evaluate()`: Running evaluation on test set
- `wandb_log()`: Logging metrics to Weights & Biases
