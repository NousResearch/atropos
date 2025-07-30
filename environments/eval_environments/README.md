# Evaluation Environments

This directory contains environments that are **primarily designed for evaluation and benchmarking** of language models. While these environments can technically be used for reinforcement learning training, they are optimized for assessment and comparison of model capabilities rather than training.

**Key Characteristics:**
- **Evaluation-First Design**: These environments are built around established benchmarks and evaluation methodologies
- **Limited Training Data**: Most lack comprehensive training datasets and are designed for assessment rather than learning
- **Benchmark Compliance**: Implementations strictly follow official evaluation protocols for consistency with published results

**Future Development:** If comprehensive training datasets are developed for any of these environments, they may be moved back to the main `environments/` directory to better support both training and evaluation use cases.

---

## Available Evaluation Environments

### 🏆 Pairwise Judgment Environment (`pairwise_judgement_environment.py`) - **BENCHMARK**

**⚠️ PRIMARY USE CASE: EVALUATION BENCHMARK** - This environment implements the official RewardBench-2 evaluation suite for measuring how well models can judge the quality of AI assistant responses. Use this to benchmark your models against state-of-the-art judgment capabilities.

A comprehensive benchmark environment based on the official RewardBench-2 dataset that evaluates models' ability to judge AI assistant responses through two evaluation modes:
- **Choice Mode**: Compare 4 responses (A/B/C/D) and select the best one
- **Ties Mode**: Rate individual responses on a 1-10 scale to identify winners

**Benchmark Categories:**
- **Factuality**: Factual accuracy and correctness
- **Focus**: Staying on topic and following instructions
- **Math**: Mathematical reasoning and problem-solving
- **Precise IF**: Precise instruction following
- **Safety**: Harmful content detection and safety
- **Ties**: Multiple correct responses requiring nuanced judgment

**Input Format:**
- **Choice Mode**: Question + 4 AI responses (A, B, C, D) → Select best response
- **Ties Mode**: Question + individual response → Rate 1-10 scale
- Each item contains:
  - `prompt`: The user question
  - `chosen`: List of high-quality responses
  - `rejected`: List of lower-quality responses
  - `subset`: Category (Factuality, Math, Safety, etc.)

**System Prompt (Thinking Mode):**
```
You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem.
You are allowed to monologue in freeform at times, but the majority of your reasoning must be done using markdown, with point, bolding and italics used appropriately to structure your reasoning.
```

**Evaluation Methodology:**
- **Choice Mode**: Score 1.0 if model selects the correct response (A/B/C/D format: `[[A]]`)
- **Ties Mode**: Rate all responses → Find maximum rating → Score 1.0 if any max-rated response is correct
- **Format Compliance**: Must use exact format (`[[A]]` for choice, trailing number for rating)
- **Robust to Parsing Errors**: Partial failures don't invalidate entire samples

**Key Features:**
- **Dual Evaluation Modes**: Automatic detection of choice vs. ties samples
- **Category Filtering**: Evaluate specific RewardBench categories
- **Thinking Mode Support**: Full `<think></think>` tag parsing and evaluation
- **Chat Completions**: Uses modern chat completion API endpoints
- **Comprehensive Metrics**: A-bias detection, compliance rates, rating distributions
- **Original RewardBench Compliance**: Matches official methodology exactly

**Configuration Options:**
- `thinking_mode`: Enable `<think></think>` reasoning mode (default: True)
- `num_choices`: Number of response choices (2-26, default: 4)
- `eval_categories`: Filter to specific categories (default: all)
- `max_ties_responses`: Limit ties responses for cost control (default: 100)
- `eval_temperature`: Temperature for evaluation (default: 0.6)
- `eval_max_tokens`: Max tokens for evaluation (default: 16384)

**Benchmark Usage (Primary Use Case):**
```bash
# Evaluate OpenAI models
python pairwise_judgement_environment.py evaluate \
    --openai.base_url https://api.openai.com/v1 \
    --openai.api_key sk-YOURAPIKEY \
    --openai.model_name gpt-4o \
    --env.data_dir_to_save_evals ./evals/rewardbench-gpt-4o

# Evaluate specific categories only
python pairwise_judgement_environment.py evaluate \
    --openai.model_name gpt-4o-mini \
    --env.eval_categories='["MATH", "SAFETY"]' \
    --env.data_dir_to_save_evals ./evals/math-safety-only

# Evaluate with custom settings
python pairwise_judgement_environment.py evaluate \
    --openai.model_name gpt-4o \
    --env.thinking_mode=False \
    --env.eval_temperature=0.0 \
    --env.max_ties_responses=50 \
    --env.data_dir_to_save_evals ./evals/deterministic-eval
```

**Training Usage - Will Require Your Own Custom Dataset(Secondary Use Case):**
```bash
# Train with synthetic judgment data
python pairwise_judgement_environment.py serve \
    --env.thinking_mode=True \
    --env.num_choices=4 \
    --env.rollout_temperature=0.8
```

**Benchmark Metrics:**
- `eval/percent_correct`: Overall accuracy across all categories
- `eval/percent_correct_{category}`: Per-category accuracy scores
- `eval/choice_format_compliance_rate`: Format compliance for choice mode
- `eval/ties_format_compliance_rate`: Format compliance for ties mode
- `eval/ties_error_rate`: Proportion of unparseable rating attempts
- `eval/wrong_answer_a_bias_rate`: A-bias detection in incorrect responses
- `eval/avg_ties_rating`: Average rating in ties mode
- `eval/ties_rating_freq_{1-10}`: Distribution of ratings 1-10

**Dependencies:**
- `datasets` (for RewardBench-2 dataset)
- `wandb` (for metrics tracking)
- `tqdm` (for progress bars)

**Citation:**
```bibtex
@misc{lambert2024rewardbenchevaluatingrewardmodels,
      title={RewardBench: Evaluating Reward Models for Language Modeling},
      author={Nathan Lambert and Valentina Pyatkin and Jacob Morrison and LJ Miranda and Bill Yuchen Lin and Khyathi Chandu and Nouha Dziri and Sachin Kumar and Tom Zick and Yejin Choi and Noah A. Smith and Hannaneh Hajishirzi},
      year={2024},
      eprint={2403.13787},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2403.13787},
}
```

---

### Arena-Hard Environment (`arena_hard_environment.py`)

A high-quality benchmark environment implementing the Arena-Hard evaluation pipeline with Claude Sonnet 4 as judge, designed to train and evaluate models against challenging real-world user queries from Chatbot Arena.

**Based on:** [Arena-Hard-Auto v0.1](https://lmsys.org/blog/2024-04-19-arena-hard/) by LMSYS ORG
- **Citation:** Li, T., Chiang, W. L., Frick, E., Dunlap, L., Zhu, B., Gonzalez, J. E., & Stoica, I. (2024). From Live Data to High-Quality Benchmarks: The Arena-Hard Pipeline. LMSYS ORG Blog.

**Key Features:**
- **Claude Sonnet 4 Judge**: Uses state-of-the-art Claude Sonnet 4 model for robust response evaluation
- **Dual-Round Judging**: Implements Arena-Hard methodology with two judgment rounds to reduce position bias
- **Thinking Mode Support**: Full `<think></think>` tag parsing and validation for advanced reasoning
- **GPT-4 Baseline Comparison**: Evaluates model responses against high-quality GPT-4-0314 baseline responses
- **Real-World Queries**: 500 challenging prompts extracted from 200K+ user queries in Chatbot Arena
- **Comprehensive Metrics**: Win rates, category breakdowns, and Arena-Hard compatible scoring

**Input Format:**
Each training/evaluation item contains:
- `uid`: Unique identifier for prompt-baseline pairing
- `prompt`: The user query from Arena-Hard dataset
- `answer`: GPT-4-0314 baseline response (for comparison)
- `category`: Optional category classification
- `cluster`: Optional topic cluster assignment

**Dataset Schema:**
- **Training Prompts**: `NousResearch/arena-hard-v1-prompts` (HuggingFace) or local JSONL
- **Training Baselines**: `NousResearch/gpt-4-0314-baseline-arenahard` (HuggingFace) or local JSONL
- **Evaluation Prompts**: Same as training by default, configurable separately
- **Evaluation Baselines**: Same as training by default, configurable separately

**System Prompt (Thinking Mode - Default):**
```
You are a deep thinking AI assistant. Before providing your response, you should think through the problem carefully. Use <think></think> tags to enclose your internal reasoning and thought process, then provide your final response after the thinking tags.
```

**System Prompt (Non-Thinking Mode):**
- Uses `custom_system_prompt` if provided, otherwise no system prompt

**Judge System Prompt:**
```
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by providing a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "[[A>B]]" if assistant A is better, "[[B>A]]" if assistant B is better, and "[[A=B]]" for a tie.
```

**Evaluation Methodology:**
1. **Model Response Generation**: Generate response to Arena-Hard prompt using configured temperature/tokens
2. **Thinking Validation**: If thinking mode enabled, validate exactly one `<think></think>` pair and extract content after tags
3. **Dual-Round Judging**:
   - Round 1: Judge model response (A) vs GPT-4 baseline (B)
   - Round 2: Judge GPT-4 baseline (A) vs model response (B)
4. **Score Combination**: Average the two judgment scores using Arena-Hard logic
5. **Arena Score Conversion**: Convert [-1,1] range to [0,1] winrate format

**Reward Function:**
- **Training**: Scores range from -1.0 to 1.0 based on combined judgment results
  - 1.0: Model response clearly better than baseline
  - 0.0: Tie between model and baseline
  - -1.0: Baseline clearly better than model response
- **Invalid Thinking**: Automatic 0.0 score for malformed `<think></think>` tags
- **Evaluation**: Converted to Arena-Hard winrate format (0.0 to 1.0)

**Configuration Options (`ArenaHardConfig`):**

**Thinking Mode:**
- `thinking_mode`: Enable `<think></think>` reasoning mode (default: False)
- `custom_thinking_prompt`: Custom thinking prompt (default: uses built-in prompt)
- `custom_system_prompt`: Additional system prompt to append (default: None)

**Judge Settings:**
- `judge_temperature`: Temperature for Claude Sonnet 4 judge (default: 0.2)
- `judge_max_tokens`: Max tokens for judge responses (default: 4096)

**Model Generation:**
- `eval_temperature`: Temperature for evaluation completions (default: 0.6)
- `rollout_temperature`: Temperature for training rollouts (default: 1.0)
- `eval_max_tokens`: Max tokens for evaluation (default: 40960)
- `train_max_tokens`: Max tokens for training (default: 16384)

**Dataset Configuration:**
- `train_prompt_dataset`: Training prompts dataset/path (default: "NousResearch/arena-hard-v1-prompts")
- `train_baseline_dataset`: Training baselines dataset/path (default: "NousResearch/gpt-4-0314-baseline-arenahard")
- `eval_prompt_dataset`: Evaluation prompts dataset/path (default: same as training)
- `eval_baseline_dataset`: Evaluation baselines dataset/path (default: same as training)
- `train_split`/`eval_split`: Dataset splits to use (default: "train")

**Reliability:**
- `max_retries`: Maximum API call retries (default: 3)
- `retry_delay`: Delay between retries in seconds (default: 1.0)
- `min_response_length`: Minimum valid response length (default: 10)

**Usage Examples:**

**Training:**
```bash
# Basic training with thinking mode
python arena_hard_environment.py serve \
    --env.thinking_mode=True \
    --env.rollout_temperature=1.0 \
    --env.group_size=8

# Training without thinking mode
python arena_hard_environment.py serve \
    --env.thinking_mode=False \
    --env.custom_system_prompt="You are a helpful assistant." \
    --env.eval_temperature=0.0

# Training with custom datasets
python arena_hard_environment.py serve \
    --env.train_prompt_dataset="/path/to/custom_prompts.jsonl" \
    --env.train_baseline_dataset="/path/to/custom_baselines.jsonl"
```

**Evaluation:**
```bash
# Evaluate model performance
python arena_hard_environment.py evaluate \
    --env.thinking_mode=True \
    --env.eval_temperature=0.0 \
    --env.judge_temperature=0.0

# Evaluate with debug logging
python arena_hard_environment.py evaluate \
    --env.full_debug=True \
    --env.eval_max_tokens=8192
```

**Evaluation Metrics:**
- `eval/overall_winrate`: Overall Arena-Hard winrate (0.0 to 1.0)
- `eval/winrate_{category}`: Per-category winrates when available
- `eval/win_count`/`eval/tie_count`/`eval/loss_count`: Raw judgment counts
- `eval/win_rate`/`eval/tie_rate`/`eval/loss_rate`: Judgment proportions
- `eval/total_samples`: Number of evaluation samples processed

**Training Metrics:**
- `train/winrate`: Training winrate based on judgment outcomes
- `train/win_rate`/`train/tie_rate`/`train/loss_rate`: Training judgment distributions
- `train/total_judgments`: Total judgments made during training
- `config/thinking_mode`: Whether thinking mode is enabled (1.0/0.0)

**Dependencies:**
- `openai` (for Claude Sonnet 4 API via Anthropic's OpenAI-compatible endpoint)
- `datasets` (for HuggingFace dataset loading)
- `tiktoken` (for tokenization)
- `wandb` (for metrics tracking)
- `tqdm` (for progress bars)

**Environment Variables Required:**
- `ANTHROPIC_API_KEY`: API key for Claude Sonnet 4 judge

**Key Implementation Details:**
- **Position Bias Reduction**: Dual-round judging with position swapping
- **Robust Parsing**: Multiple regex patterns for judgment extraction ([[A>B]], [[B>A]], [[A=B]])
- **Thinking Validation**: Strict validation of thinking tag format and content extraction
- **Error Handling**: Comprehensive retry logic with exponential backoff
- **Arena-Hard Compatibility**: Scores and metrics match original Arena-Hard methodology 