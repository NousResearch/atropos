# Atropos Evaluation Environments

This directory contains **30+ evaluation environments** for benchmarking language models across diverse capabilities: reasoning, coding, math, instruction following, creative writing judgment, and more.

## Table of Contents

- [Quick Start](#quick-start)
- [Environment Categories](#environment-categories)
- [Common Configuration Options](#common-configuration-options)
- [Knowledge & Reasoning Benchmarks](#knowledge--reasoning-benchmarks)
- [Math Benchmarks](#math-benchmarks)
- [Code Generation](#code-generation)
- [Instruction Following](#instruction-following)
- [LLM-as-Judge Benchmarks](#llm-as-judge-benchmarks)
- [Open-Ended QA](#open-ended-qa)
- [Shared Utilities](#shared-utilities)
- [Advanced Usage](#advanced-usage)

---

## Quick Start

All evaluation environments follow the same CLI pattern:

```bash
python <environment>.py evaluate \
    --openai.base_url <API_ENDPOINT> \
    --openai.api_key <API_KEY> \
    --openai.model_name <MODEL_NAME> \
    --env.data_dir_to_save_evals <OUTPUT_DIR>
```

### Example: Run MMLU on GPT-4o

```bash
cd environments/eval_environments

python mmlu_eval.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o \
    --env.data_dir_to_save_evals ../evals/mmlu/gpt-4o
```

### Example: Run on Local vLLM Server

```bash
python mmlu_eval.py evaluate \
    --openai.base_url http://localhost:8000/v1 \
    --openai.api_key xxx \
    --openai.model_name Qwen/Qwen2.5-72B-Instruct \
    --env.thinking_mode True \
    --env.data_dir_to_save_evals ../evals/mmlu/qwen-72b
```

### Example: Run on OpenRouter

```bash
python gpqa_eval.py evaluate \
    --openai.base_url https://openrouter.ai/api/v1 \
    --openai.api_key $OPENROUTER_API_KEY \
    --openai.model_name anthropic/claude-sonnet-4 \
    --env.data_dir_to_save_evals ../evals/gpqa/claude-sonnet
```

---

## Environment Categories

| Category | Environments | Description |
|----------|-------------|-------------|
| **Knowledge/Reasoning** | MMLU, MMLU-Pro, GPQA, AGIEval, OBQA, BBH | Multiple-choice QA |
| **Math** | GSM8K, MATH, MATH-500, AIME, AIMO, OlympiadBench | Mathematical reasoning |
| **Code** | LiveCodeBench (LCB) | Code generation with execution |
| **Instruction Following** | IFEval | Format/constraint adherence |
| **Reading Comprehension** | DROP, MuSR, PubMedQA, HLE | Text understanding |
| **Open-Ended** | SimpleQA | Factuality verification |
| **LLM-as-Judge** | MT-Bench, MixEval, Arena-Hard, RefusalBench, JudgeMark | Model evaluation |
| **Pairwise Judgment** | PairwiseJudgement | RewardBench-2 evaluation |

---

## Common Configuration Options

All environments support these common options:

```bash
# Thinking mode (chain-of-thought reasoning)
--env.thinking_mode True/False

# Custom system prompts
--env.custom_system_prompt "You are a helpful assistant."
--env.custom_thinking_prompt "Think step by step..."

# Token limits (0 = model default)
--env.eval_max_tokens 4096

# Temperature
--env.eval_temperature 0.0

# Debug mode (saves full responses)
--env.full_debug True

# Output directory
--env.data_dir_to_save_evals ./results/
```

---

## Knowledge & Reasoning Benchmarks

### MMLU (`mmlu_eval.py`)

**Massive Multitask Language Understanding** - 57 subjects from STEM to humanities.

```bash
# Full MMLU evaluation
python mmlu_eval.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o \
    --env.data_dir_to_save_evals ../evals/mmlu/gpt-4o

# With thinking mode (recommended for reasoning models)
python mmlu_eval.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o \
    --env.thinking_mode True \
    --env.data_dir_to_save_evals ../evals/mmlu/gpt-4o-thinking

# Test on specific subjects only
python mmlu_eval.py evaluate \
    --openai.base_url http://localhost:8000/v1 \
    --openai.api_key xxx \
    --openai.model_name Hermes-4-14B \
    --env.subjects '["abstract_algebra", "anatomy"]' \
    --env.data_dir_to_save_evals ../evals/mmlu/hermes-subset
```

### MMLU-Pro (`mmlu_pro_eval.py`)

**Harder version of MMLU** with 10 answer choices instead of 4.

```bash
python mmlu_pro_eval.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o \
    --env.thinking_mode True \
    --env.data_dir_to_save_evals ../evals/mmlu-pro/gpt-4o
```

### GPQA Diamond (`gpqa_eval.py`)

**Graduate-level science questions** - PhD-level difficulty.

```bash
# GPQA Diamond (default, hardest subset)
python gpqa_eval.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o \
    --env.thinking_mode True \
    --env.data_dir_to_save_evals ../evals/gpqa/gpt-4o

# Different subset
python gpqa_eval.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o \
    --env.subset gpqa_extended \
    --env.data_dir_to_save_evals ../evals/gpqa/gpt-4o-extended
```

### AGIEval (`agieval_eval.py`)

**Human-centric benchmark** from admission and qualification exams (SAT, LSAT, GRE, etc.).

```bash
# All AGIEval subsets
python agieval_eval.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o \
    --env.thinking_mode True \
    --env.data_dir_to_save_evals ../evals/agieval/gpt-4o

# Specific subset (e.g., SAT Math)
python agieval_eval.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o \
    --env.subset sat-math \
    --env.data_dir_to_save_evals ../evals/agieval/gpt-4o-sat-math
```

### OpenBookQA (`obqa_eval.py`)

**Common sense reasoning** with science facts.

```bash
python obqa_eval.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o \
    --env.thinking_mode True \
    --env.data_dir_to_save_evals ../evals/obqa/gpt-4o
```

### BigBench Hard (`bbh_eval.py`)

**23 challenging tasks** from BIG-Bench.

```bash
# All BBH tasks
python bbh_eval.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o \
    --env.thinking_mode True \
    --env.data_dir_to_save_evals ../evals/bbh/gpt-4o

# Specific task
python bbh_eval.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o \
    --env.subset boolean_expressions \
    --env.data_dir_to_save_evals ../evals/bbh/gpt-4o-boolean
```

---

## Math Benchmarks

All math benchmarks expect answers in `\boxed{}` LaTeX format and use `math_verify` for robust symbolic comparison.

### GSM8K (`gsm8k_eval.py`)

**Grade school math** word problems.

```bash
python gsm8k_eval.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o \
    --env.thinking_mode True \
    --env.data_dir_to_save_evals ../evals/gsm8k/gpt-4o
```

### MATH (`math_eval.py`)

**Competition math** problems (algebra, geometry, number theory, etc.).

```bash
python math_eval.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o \
    --env.thinking_mode True \
    --env.data_dir_to_save_evals ../evals/math/gpt-4o

# Filter by difficulty level
python math_eval.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o \
    --env.level "Level 5" \
    --env.data_dir_to_save_evals ../evals/math/gpt-4o-level5
```

### MATH-500 (`math500_eval.py`)

**500 hardest MATH problems** curated subset.

```bash
python math500_eval.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o \
    --env.thinking_mode True \
    --env.data_dir_to_save_evals ../evals/math500/gpt-4o
```

### AIME (`aime_eval.py`)

**American Invitational Mathematics Examination** - integer answers 0-999.

```bash
python aime_eval.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o \
    --env.thinking_mode True \
    --env.data_dir_to_save_evals ../evals/aime/gpt-4o
```

### AIMO (`aimo_eval.py`)

**AI Math Olympiad** problems.

```bash
python aimo_eval.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o \
    --env.thinking_mode True \
    --env.data_dir_to_save_evals ../evals/aimo/gpt-4o
```

### OlympiadBench (`olympiadbench_eval.py`)

**Olympiad-level math and physics** problems.

```bash
python olympiadbench_eval.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o \
    --env.thinking_mode True \
    --env.data_dir_to_save_evals ../evals/olympiad/gpt-4o
```

---

## Code Generation

### LiveCodeBench (`lcb_eval.py`)

**Code generation** with actual execution against test cases. Supports Modal sandbox for secure execution.

```bash
# First, deploy Modal sandbox (one-time setup)
pip install modal
modal token new
modal deploy modal_sandbox.py

# Run with Modal sandbox (secure)
python lcb_eval.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o \
    --env.use_modal True \
    --env.thinking_mode True \
    --env.data_dir_to_save_evals ../evals/lcb/gpt-4o

# Run with local execution (faster, but not sandboxed - for trusted code only)
python lcb_eval.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o \
    --env.use_modal False \
    --env.data_dir_to_save_evals ../evals/lcb/gpt-4o-local

# Different dataset version
python lcb_eval.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o \
    --env.subset release_latest \
    --env.data_dir_to_save_evals ../evals/lcb/gpt-4o-latest
```

---

## Instruction Following

### IFEval (`ifeval_eval.py`)

**Instruction Following Evaluation** - tests adherence to formatting constraints.

```bash
python ifeval_eval.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o \
    --env.thinking_mode True \
    --env.data_dir_to_save_evals ../evals/ifeval/gpt-4o
```

---

## LLM-as-Judge Benchmarks

These benchmarks evaluate models' ability to judge other models' outputs.

### JudgeMark v2 (`judgemark_eval.py`)

**Creative writing judgment** - evaluates how well a model can judge creative writing quality.

```bash
# Requires Judgemark-v2 data (clone to atropos root)
git clone https://github.com/EQ-bench/Judgemark-v2.git

python judgemark_eval.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o \
    --env.data_dir_to_save_evals ../evals/judgemark/gpt-4o

# Quick test with limited samples
python judgemark_eval.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o \
    --env.max_samples 20 \
    --env.full_debug True \
    --env.data_dir_to_save_evals ../evals/judgemark/gpt-4o-test
```

### MT-Bench (`mtbench_eval.py`)

**Multi-turn conversation** benchmark with LLM judge.

```bash
python mtbench_eval.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o \
    --env.judge_model gpt-4o \
    --env.data_dir_to_save_evals ../evals/mtbench/gpt-4o

# Use different judge model
python mtbench_eval.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o-mini \
    --env.judge_model gpt-4o \
    --env.judge_base_url https://api.openai.com/v1 \
    --env.judge_api_key_env OPENAI_API_KEY \
    --env.data_dir_to_save_evals ../evals/mtbench/gpt-4o-mini-judged-by-4o
```

### MixEval (`mixeval_eval.py`)

**Dynamic benchmark** mixing multiple evaluation types with LLM judge.

```bash
python mixeval_eval.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o \
    --env.data_dir_to_save_evals ../evals/mixeval/gpt-4o
```

### Arena-Hard (`arena_hard_environment.py`)

**Challenging real-world queries** from Chatbot Arena with Claude as judge.

```bash
python arena_hard_environment.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o \
    --env.thinking_mode True \
    --env.data_dir_to_save_evals ../evals/arena-hard/gpt-4o
```

### RefusalBench (`refusalbench_environment.py`)

**Safety refusal evaluation** with LLM judge.

```bash
python refusalbench_environment.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o \
    --env.thinking_mode True \
    --env.data_dir_to_save_evals ../evals/refusalbench/gpt-4o
```

### Pairwise Judgment (`pairwise_judgement_environment.py`)

**RewardBench-2** evaluation for pairwise response comparison.

```bash
python pairwise_judgement_environment.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o \
    --env.thinking_mode True \
    --env.data_dir_to_save_evals ../evals/rewardbench/gpt-4o

# Specific categories
python pairwise_judgement_environment.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o \
    --env.eval_categories '["MATH", "SAFETY"]' \
    --env.data_dir_to_save_evals ../evals/rewardbench/gpt-4o-math-safety
```

---

## Open-Ended QA

### SimpleQA (`simpleqa_eval.py`)

**Factuality benchmark** with exact/fuzzy matching or optional LLM judge.

```bash
# Default: string matching (no LLM judge)
python simpleqa_eval.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o \
    --env.thinking_mode True \
    --env.data_dir_to_save_evals ../evals/simpleqa/gpt-4o

# With LLM judge
python simpleqa_eval.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o \
    --env.use_llm_judge True \
    --env.judge_model_name gpt-4o \
    --env.data_dir_to_save_evals ../evals/simpleqa/gpt-4o-llm-judge
```

### DROP (`drop_eval.py`)

**Reading comprehension** requiring discrete reasoning over passages.

```bash
python drop_eval.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o \
    --env.thinking_mode True \
    --env.data_dir_to_save_evals ../evals/drop/gpt-4o
```

### MuSR (`musr_eval.py`)

**Multi-step reasoning** in long narratives.

```bash
python musr_eval.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o \
    --env.thinking_mode True \
    --env.data_dir_to_save_evals ../evals/musr/gpt-4o
```

### PubMedQA (`pubmedqa_eval.py`)

**Biomedical research QA** from PubMed abstracts.

```bash
python pubmedqa_eval.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o \
    --env.thinking_mode True \
    --env.data_dir_to_save_evals ../evals/pubmedqa/gpt-4o
```

### HLE (`hle_eval.py`)

**Humanity's Last Exam** - challenging collaborative QA.

```bash
python hle_eval.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o \
    --env.thinking_mode True \
    --env.data_dir_to_save_evals ../evals/hle/gpt-4o
```

---

## Advanced Usage

### Running on Multiple Models (Batch Script)

```bash
#!/bin/bash
# batch_eval.sh

MODELS=("gpt-4o" "gpt-4o-mini" "claude-sonnet-4")
BENCHMARKS=("mmlu_eval.py" "gpqa_eval.py" "gsm8k_eval.py")

for model in "${MODELS[@]}"; do
    for benchmark in "${BENCHMARKS[@]}"; do
        bench_name=$(basename "$benchmark" .py)
        echo "Running $bench_name on $model..."
        
        python "$benchmark" evaluate \
            --openai.base_url https://api.openai.com/v1 \
            --openai.api_key $OPENAI_API_KEY \
            --openai.model_name "$model" \
            --env.thinking_mode True \
            --env.data_dir_to_save_evals "../evals/${bench_name}/${model}"
    done
done
```

### Comparing Thinking vs Non-Thinking Mode

```bash
# Without thinking (baseline)
python mmlu_eval.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o \
    --env.thinking_mode False \
    --env.data_dir_to_save_evals ../evals/mmlu/gpt-4o-no-think

# With thinking
python mmlu_eval.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o \
    --env.thinking_mode True \
    --env.data_dir_to_save_evals ../evals/mmlu/gpt-4o-thinking
```

### Custom System Prompts

```bash
python mmlu_eval.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o \
    --env.thinking_mode True \
    --env.custom_thinking_prompt "You are a brilliant scientist. Reason through this problem methodically using <think></think> tags." \
    --env.custom_system_prompt "Always show your work clearly." \
    --env.data_dir_to_save_evals ../evals/mmlu/gpt-4o-custom-prompt
```

### Using Different API Providers

```bash
# OpenAI
python mmlu_eval.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o

# Anthropic (via OpenAI-compatible endpoint)
python mmlu_eval.py evaluate \
    --openai.base_url https://api.anthropic.com/v1/ \
    --openai.api_key $ANTHROPIC_API_KEY \
    --openai.model_name claude-sonnet-4-20250514

# Together AI
python mmlu_eval.py evaluate \
    --openai.base_url https://api.together.xyz/v1 \
    --openai.api_key $TOGETHER_API_KEY \
    --openai.model_name meta-llama/Llama-3.3-70B-Instruct-Turbo

# Local vLLM
python mmlu_eval.py evaluate \
    --openai.base_url http://localhost:8000/v1 \
    --openai.api_key xxx \
    --openai.model_name Qwen/Qwen2.5-72B-Instruct

# OpenRouter
python mmlu_eval.py evaluate \
    --openai.base_url https://openrouter.ai/api/v1 \
    --openai.api_key $OPENROUTER_API_KEY \
    --openai.model_name anthropic/claude-sonnet-4

# Fireworks AI
python mmlu_eval.py evaluate \
    --openai.base_url https://api.fireworks.ai/inference/v1 \
    --openai.api_key $FIREWORKS_API_KEY \
    --openai.model_name accounts/fireworks/models/llama-v3p1-70b-instruct
```

### Debug Mode for Development

```bash
# Full debug mode saves all responses
python mmlu_eval.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o \
    --env.full_debug True \
    --env.data_dir_to_save_evals ../evals/mmlu/gpt-4o-debug
```

### Temperature and Token Settings

```bash
# Deterministic evaluation (temperature=0)
python mmlu_eval.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o \
    --env.eval_temperature 0.0 \
    --env.data_dir_to_save_evals ../evals/mmlu/gpt-4o-deterministic

# Higher temperature for diversity
python mmlu_eval.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o \
    --env.eval_temperature 0.7 \
    --env.data_dir_to_save_evals ../evals/mmlu/gpt-4o-temp07

# Custom max tokens
python gsm8k_eval.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key $OPENAI_API_KEY \
    --openai.model_name gpt-4o \
    --env.eval_max_tokens 8192 \
    --env.data_dir_to_save_evals ../evals/gsm8k/gpt-4o-8k
```

---

## Shared Utilities

### `eval_helpers.py`

Contains shared functions used across environments:

- **Answer extraction**: `extract_letter_from_answer_tag`, `extract_freeform_from_answer_tag`
- **Math verification**: `score_math_answer_async`, `extract_boxed_answers`
- **Thinking mode**: `create_system_content`, `get_default_thinking_prompt`
- **Results saving**: `save_eval_results`, `load_eval_results`

---

## Output Format

All evaluations produce:

1. **`metrics.json`**: Summary statistics (accuracy, F1, etc.)
2. **`results.jsonl`**: Per-item results (one JSON per line)
3. **`evaluate_config.yaml`**: Configuration used for the run

Example `metrics.json`:
```json
{
  "accuracy": 0.847,
  "total_samples": 14042,
  "correct": 11892,
  "per_category_accuracy": {
    "stem": 0.823,
    "humanities": 0.891,
    "social_sciences": 0.856
  }
}
```

---

## Dependencies

Core dependencies (most are in the main requirements.txt):

```bash
pip install datasets openai pydantic tqdm wandb scipy numpy
```

For specific environments:
- **Math evals**: `pip install math_verify latex2sympy2_extended`
- **LiveCodeBench**: `pip install modal` (for secure sandbox)
- **JudgeMark**: Clone `Judgemark-v2` repo to atropos root

---

## Contributing

When adding a new evaluation environment:

1. Follow the existing patterns (inherit from `BaseEnv`)
2. Use `eval_helpers.py` for common functions
3. Support thinking mode with `<think></think>` tags
4. Use `<answer></answer>` tags for answer extraction (or `\boxed{}` for math)
5. Save results using `save_eval_results()`
6. Add examples to this README

---

## License

See the main Atropos repository license.
