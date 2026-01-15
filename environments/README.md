# Environments

This directory contains various environments for training and evaluating language models on different tasks. Each environment implements a specific task with its own input format, reward function, and evaluation metrics.

## Directory Structure

- **Main Environments**: Training-focused environments with comprehensive datasets
- **[Evaluation Environments](eval_environments/)**: Benchmark-focused environments primarily designed for model evaluation (see [eval_environments/README.md](eval_environments/README.md))

## Available Environments

---

### Prime Intellect Verifiers Integration

A flexible environment that integrates with the [Verifiers](https://docs.primeintellect.ai/) ecosystem, allowing you to use any registered Prime environment for RL training, SFT data generation, or evaluation.

**Files:**
- `environments/verifiers_server.py` - Training and SFT data generation
- `environments/eval_environments/verifiers_eval.py` - Standalone evaluation

**Dependencies:**

- `verifiers` Python package (install via `pip install verifiers` or include in your environment)
- Prime CLI for environment management (`uv tool install prime`)
- Prime CLI login required (`prime login`)
- Environment installation (`prime env install owner/env_name`)

**Supported Modes:**

| Mode | File | Description |
|------|------|-------------|
| `serve` | `verifiers_server.py` | RL training with local inference server (requires ManagedServer for logprobs) |
| `process` | `verifiers_server.py` | SFT data generation with ANY API (OpenAI, Claude, local, etc.) |
| `evaluate` | `verifiers_server.py` | Quick evaluation using ManagedServer |
| `evaluate` | `verifiers_eval.py` | Standalone evaluation with detailed metrics and retry logic |

**Input Format:**

- Loaded dynamically from the specified Prime environment via `vf.load_environment()`
- Each item contains:
  - `question`: The problem/prompt
  - `answer`: The expected answer for verification

**System Prompt:**

- Dynamically loaded from the Prime environment's `system_prompt` configuration

**Reward Function:**

- Uses the environment's **rubric** system with:
  - `parser`: Extracts answers from completions (e.g., `parser.parse_answer(completion)`)
  - `funcs`: List of reward functions that receive `(parser, completion, answer)`
  - `weights`: Weights for combining reward functions (normalized to sum to 1.0)
- Final score is weighted sum of all reward function outputs

**W&B Metrics Logged (Training - `verifiers_server.py`):**

| Metric | Description |
|--------|-------------|
| `train/percent_correct` | Average score from verifiers reward functions (0-1) |
| `train/rollouts` | Table of tokenized completions with scores |
| `train/completion_lengths_*` | Response length statistics (std, min, max, p95) |
| `server/server_0_request_time_*` | API latency metrics (avg, std, 99p) |
| `eval/avg_total_score` | Average score on evaluation dataset |

**Output (Evaluation - `verifiers_eval.py`):**

Uses `evaluate_log()` from `atroposlib.envs.eval` to output:
- Console: Metrics table with accuracy, avg_score, time, and per-reward function breakdown
- File: `metrics.json` and `samples.jsonl` (when `--eval-dir` is specified)

**Configuration Options (`VfEnvConfig` for `verifiers_server.py`):**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `vf_env_name` | str | `""` | Prime environment identifier (e.g., `"will/wordle"`, `"primeintellect/gsm8k"`) |
| `env_args` | Dict | `{}` | Additional arguments passed to `vf.load_environment()`. Read environment specific documentation to get these args. |

**CLI Options (`verifiers_eval.py`):**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--server-url` | str | `http://localhost:8000/v1` | URL of the inference server |
| `--model-name` | str | (required) | Model name to evaluate |
| `--api-key` | str | `$OPENAI_API_KEY` | API key (defaults to env var) |
| `--vf-env-name` | str | `primeintellect/gsm8k` | Prime environment identifier |
| `--temperature` | float | `0.0` | Temperature for generation |
| `--max-tokens` | int | `2048` | Maximum tokens per completion |
| `--max-eval-items` | int | `-1` | Maximum items to evaluate (-1 for all) |
| `--max-concurrent` | int | `64` | Maximum concurrent requests |
| `--eval-dir` | str | `None` | Directory to save evaluation results |

**Usage Examples:**

```bash
# RL Training (requires local vLLM/SGLang server)
python verifiers_server.py serve \
    --env.vf_env_name "will/wordle" \
    --openai.base_url http://localhost:9001/v1 \
    --slurm false

# SFT Data Generation with OpenAI GPT-4o
python verifiers_server.py process \
    --env.vf_env_name "will/wordle" \
    --env.data_path_to_save_groups gpt4o_sft_data.jsonl \
    --env.total_steps 100 \
    --env.group_size 4 \
    --openai.model_name gpt-4o \
    --openai.base_url https://api.openai.com/v1

# SFT Data Generation with local server
python verifiers_server.py process \
    --env.vf_env_name "will/wordle" \
    --env.data_path_to_save_groups local_sft_data.jsonl \
    --openai.base_url http://localhost:9001/v1

# Quick Evaluation via verifiers_server.py
python verifiers_server.py evaluate \
    --env.vf_env_name "will/wordle" \
    --openai.base_url http://localhost:9001/v1

# Standalone Evaluation with OpenAI (verifiers_eval.py)
python eval_environments/verifiers_eval.py \
    --server-url https://api.openai.com/v1 \
    --model-name gpt-4o \
    --vf-env-name primeintellect/gsm8k

# Quick test run with limited items
python eval_environments/verifiers_eval.py \
    --server-url https://api.openai.com/v1 \
    --model-name gpt-4o-mini \
    --vf-env-name primeintellect/alphabet-sort \
    --max-eval-items 10

# Evaluation with local server and results saved
python eval_environments/verifiers_eval.py \
    --server-url http://localhost:9001/v1 \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --vf-env-name primeintellect/gsm8k \
    --eval-dir ./eval_results
```

**Key Implementation Details:**

- **RL Training Mode (`serve`)**: Uses `ManagedServer` for proper token/logprob alignment required by policy gradient methods (GRPO, PPO, REINFORCE). Returns `ScoredDataGroup` with `tokens`, `masks`, `scores`, and `inference_logprobs`.
- **SFT Datagen Mode (`process`)**: Uses `tokenize_for_trainer` to tokenize API responses with your target model's tokenizer (e.g., GPT-4o responses tokenized for Qwen/Llama). Does NOT require logprobs.
- **Evaluation (`verifiers_eval.py`)**: Standalone evaluation script using `EvalBase` pattern. Uses verifiers' native batch evaluation for efficiency and outputs results via `evaluate_log()`. Works with any OpenAI-compatible API.

**Prime Environment Installation:**
```bash
# Install Prime CLI
uv tool install prime

# Login to Prime
prime login

# Install an environment (e.g., Wordle, GSM8K)
prime env install will/wordle
prime env install primeintellect/gsm8k

# List available environments
prime env list
```

### Letter Counting Environment (`letter_counting_environment.py`)

A comprehensive environment for training models to count letters in words, sentences, and text passages with configurable difficulty and data modes.

**Input Format:**
- Single letter counting: "How many 'a's are in the word 'banana'?"
- Multiple letter counting: "Count the occurrences of the letters 'e', 'o', and 't' in the following text: 'The quick brown fox jumps over the lazy dog'"
- Each item contains:
  - `prompt`: The counting question with instructions
  - `correct_counts`: Dictionary mapping letters to their counts
  - `text`: The source text (word, sentence, or passage)
  - `target_letters`: List of letters to count

**System Prompt:**
```
You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem.
```

**Data Modes:**
- **Word Mode**: Uses NLTK's words corpus (236k+ English words)
- **Mixed Mode**: Combines words and text passages from OpenWebText-10k dataset
- **Text Passage Mode**: Uses OpenWebText-10k dataset with character-based text extraction

**Key Features:**
- **Multi-letter counting**: Configurable simultaneous counting of multiple letters with JSON responses
- **Letter selection bias**: Configurable bias toward letters present in the text (reduces zero-count questions)
- **Random string generation**: Optional random strings (80% alphabetical) mixed with real words
- **Word capitalization**: Optional uppercase and title case transformations
- **Punctuation/space handling**: Configurable inclusion in letter counting
- **Training thresholds**: Skip groups that are too easy based on group average scores
- **Data dumping**: Save rollouts from groups with appropriate difficulty to JSONL files
- **Comprehensive metrics**: Letter distribution, text lengths, error rates, group average scores

**Answer Formats:**
- Single letter: `<answer>3</answer>`
- Multiple letters: `<answer>{"e": 4, "o": 4, "t": 2}</answer>`

**Reward Function:**
- Score of 1.0 if the model's answer exactly matches the expected count(s)
- Score of 0.0 if incorrect, malformed, or missing answer
- Groups with identical scores (no learning signal) return None
- Groups with average score > `max_group_average_for_training` are skipped for training for difficulty control/curriculum

**Configuration Options:**
- `use_text_passages`: Enable mixed mode with text passages (default: False)
- `text_passage_percentage`: Ratio of passages to words in mixed mode (default: 0.5)
- `max_letters_to_count`: Maximum simultaneous letters (default: 1)
- `multi_letter_probability`: Probability of multi-letter questions (default: 0.0)
- `present_letter_bias`: Bias toward letters present in text (default: 0.5)
- `include_punctuation_in_count`: Include punctuation in counting (default: True)
- `include_spaces_in_count`: Include spaces in counting (default: False)
- `max_group_average_for_training`: Skip easy groups threshold (default: 1.0)
- `dump_rollouts`: Save rollouts to JSONL files (default: False)
- `debug_logging`: Enable verbose per-item scoring details (default: False)

**Evaluation Metrics:**
- `eval/accuracy`: Overall accuracy on test set
- `eval/letter_distribution_entropy`: Entropy of letter selection distribution
- `eval/avg_word_length`: Average length of test items
- `eval/format_error_rate`: Rate of malformed responses
- `eval/think_tag_usage`: Percentage using think tags
- `train/group_average_scores`: Distribution of group difficulty scores

**Dependencies:**
- `nltk` (for words corpus)
- `datasets` (for OpenWebText-10k when using text passages)

**Usage Example:**
```bash
# Word-only mode
python letter_counting_environment.py serve \
    --env.use_text_passages=False \
    --env.max_letters_to_count=1 \
    --env.max_group_average_for_training=0.75

# Mixed mode with multi-letter counting
python letter_counting_environment.py serve \
    --env.use_text_passages=True \
    --env.text_passage_percentage=0.3 \
    --env.max_letters_to_count=4 \
    --env.multi_letter_probability=0.2

# Data dumping mode
python letter_counting_environment.py serve \
    --env.dump_rollouts=True \
    --env.dump_batch_size=100 \
    --env.max_group_average_for_training=0.75
```

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

### RLAIF Server Environment (`rlaif_server.py`)

Environment for Reinforcement Learning from AI Feedback (RLAIF). Used for aligning models to specific personalities or styles based on AI-generated preferences or reward signals.

**Input Format:**
- Typically involves prompts for which responses are generated and then evaluated by a reward model or preference model to guide the LLM's behavior. Specifics depend on the RLAIF setup.

**System Prompt:**
- Varies based on the desired personality/style (e.g., "Egregore," "Ascension Maze").

**Reward Function:**
- Based on the output of an AI judge/reward model, designed to score responses according to the target alignment criteria.

---

### Financial Fundamentals Prediction Environment (`fundamental_prediction_environment.py`)

Environment for training models to predict financial fundamentals using the "NousResearch/company-fundamentals-prediction-lite" dataset.

**Input Format:**
- Items include `context` (company fundamentals, news, macroeconomic data), `fundamental_metric` (e.g., revenue, EPS), and ground truth `answer` ("maintained", "raised", or "reduced") and `magnitude` (percentage change). The model analyzes the `context` to predict the `answer` and `magnitude` for the given `fundamental_metric`.

**Task:**
- Predict directional changes and magnitude for company financial fundamentals.

**Reward Function:**
- Based on the accuracy of predictions for both direction and magnitude.

---

### Math Server Environment (`math_server.py`)

A versatile math problem-solving environment supporting multiple datasets and operational modes.

**Datasets:**
- Integrates `gsm8k` (various subsets), `competition_math`, `math_qa`, and `MetaMathQA`.

**Operational Modes:**
- Supports standard problem solving, RLAIF (Reinforcement Learning from AI Feedback) for preference learning between solutions, a "judge" mode for evaluating solution correctness, and a "retry/self-correct" mode utilizing feedback on previous attempts.

**Input Format:**
- Mathematical problems, varying slightly by operational mode (e.g., including solutions for judging/RLAIF).

**System Prompt:**
- Dynamically constructed based on the operational mode. For standard problem solving, the prompt focuses on the problem itself. Other modes include specific instructions for judging, preference selection, or self-correction.

**Reward Function:**
- Based on the correctness of the mathematical solution, with variations depending on the mode (e.g., preference scores in RLAIF).

---

### Math Server Zero Environment (`math_server_zero.py`)

A math problem-solving environment using the "zwhe99/DeepMath-103K" dataset, with a structured prompt format inspired by the Open-Reasoner-Zero project.

**Input Format:**
- Mathematical problems from the "zwhe99/DeepMath-103K" dataset.

**System Prompt Structure:**
- Utilizes a specific conversational format where the AI is instructed to first think (using `<think> </think>` tags) and then provide the answer (using `<answer> </answer>` tags, with the final numerical answer in `\boxed{}`). The overall prompt guides the model through this structured reasoning and response process.
  - `prompt_format = "A conversation between User and Assistant... User: {prompt}\nAssistant: <think>"`
  - `problem_format = "You must put your answer inside <answer> </answer> tags... This is the problem:\n{problem}"`

**Reward Function:**
- Based on the correctness of the mathematical solution within the `<answer>` tag, verified using LaTeX parsing.

---

### Coding Server Environment (`code_execution_server/coding_server.py`)

Environment for training models to generate and potentially execute code.

**Input Format:**
- Coding problems or prompts (e.g., from datasets like MBPP, HumanEval).

**System Prompt:**
- Instructs the model to generate code for a given problem.

**Reward Function:**
- Based on correctness of the generated code, often involving execution and unit test passing.
- The `code_execution_server/` directory also contains a `Dockerfile` for containerized execution.

---

### Dataset Environment (`dataset_environment/dataset_env.py`)

A highly configurable environment for working with Hugging Face datasets. For more details, see the [Dataset Environment README](dataset_environment/README.md).

**Purpose:**
- Allows users to easily define RL environments using existing datasets from Hugging Face Hub.

**Input Format:**
- Defined by the chosen Hugging Face dataset (user specifies prompt and answer fields).

**System Prompt:**
- Customizable by the user.

**Reward Function:**
- Highly flexible, supports a registry of predefined reward functions (e.g., `accuracy`, `format`, `cosine_scaled`) and allows users to create and register custom reward functions. Multiple reward functions can be combined with weights.

**Configuration:**
- Primarily through YAML files specifying dataset details, generation parameters, and reward functions.

---

### Multimodal DPO Environments (`multimodal_dpo/`)

A collection of environments for Direct Preference Optimization (DPO) with multimodal inputs. These environments are designed for tasks that involve processing both text and images.

**Files:**
- `ocr_vqa.py`
- `pixmo_clocks.py`
- `pixmo_count.py`
- `pixmo_point_explanations.py`
- `clevr_cogen_a_train.py`
- `clevr_complex.py`

**Purpose:**
- Training models on tasks such as Optical Character Recognition VQA, visual counting, and interpreting complex visual scenes (e.g., Clevr).

**Input Format:**
- Typically pairs of (image, text prompt) and corresponding preferred/dispreferred responses.

**Reward Function:**
- Based on the DPO mechanism, implicitly learned from preference data.

---

### Game Environments (`game_environments/`)

This section covers environments based on interactive games.

#### Gymnasium Taxi (`game_environments/gymnasium/gym_taxi.py`)

- **Game:** Based on the classic Gymnasium Taxi-v3 environment.
- **Task:** The agent controls a taxi to pick up a passenger and drop them off at the correct location.
- **Objective:** Optimize for efficient navigation and task completion.

#### Gymnasium Blackjack (`game_environments/gymnasium/blackjack/`)

Two Blackjack environment implementations are provided. For more details, see the [Blackjack README](game_environments/gymnasium/blackjack/README.md).

- **`blackjack_env_no_thinking.py` (Standard Blackjack):**
    - **Gameplay:** A standard version of Blackjack.
    - **Objective:** Achieve a hand total closer to 21 than the dealer without exceeding 21.
    - **Interaction:** Designed for shorter episodes without complex intermediate "thinking" steps. Aiming to teach the LLM to be a better policy model in uncertain environments.

- **`blackjack_env_thinking.py` (Blackjack with Windowed Decision Making & Counterfactuals):**
    - **Gameplay:** A more complex version designed for agents that produce long interaction sequences, including "thinking" steps.
    - **Features:** Windowed decision making, local alternative generation, value-based pruning, and counterfactual data for training (GRPO).
    - **Use Case:** Ideal for training LLMs that engage in explicit multi-step reasoning before action. Teaches the model to be more "confident" about selecting optimal moves & taking informed risks in uncertain environments, even with the knowledge that it might still lose with optimal play.

### Instruction Following Environment (`instruction_following_algorithm_environment.py`)

**Dependencies:**
- `datasets` (Hugging Face)
- `langdetect`

This environment was inspired by AllenAI's RLVR-IFEVAL environment and uses AllenAI's dataset from their Tulu3 paper and project:
- Dataset: https://huggingface.co/datasets/allenai/RLVR-IFeval
- Paper: https://arxiv.org/abs/2411.15124

Environment for training models to follow natural language instructions and constraints, based on the `allenai/RLVR-IFeval` dataset with advanced adaptive curriculum learning and comprehensive data management.

**Input Format:**
- Each item from the processed `allenai/RLVR-IFeval` dataset contains:
  - `prompt`: The user's instruction string.
  - `func_name`: The string name of the verifier function (from a predefined map) used to check if the instruction is followed.
  - `args`: A dictionary of arguments for the specified verifier function.

**System Prompt:**
```
You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem.
```

**Reward Function:**
- Score of 1.0 if the model's response correctly follows the instruction, as determined by the specific verifier function associated with the input prompt.
- Score of 0.0 if the response fails the verifier function or has malformed `<think>` tags (must have exactly one opening and one closing tag).
- Length penalty applied if all responses in a batch are correct (receive a score of 1.0 before penalty):
  - No penalty for responses under 75% of max token length.
  - Linear penalty scaling from 1.0 down to 0.0 for responses between 75% and 100% of max length.
  - Returns None if all scores are identical after potential penalties (no learning signal).

**Key Features:**

**1. Adaptive Curriculum System:**
- **Cycling Queue**: Items are managed in an active training queue where solved items are removed from circulation
- **Flexible Solving Criteria**: Items can be marked as "solved" based on:
  - Group average score > `max_group_average_for_training` (default: 0.75) - too easy for training
  - Group average score ≥ 0.9 - mastered through high performance
  - Single correct rollout when `solve_on_single_correct=True` - immediate removal on any success
- **Attempt Tracking**: Tracks how many times each item has been attempted
- **Queue Reset**: When all items are solved, the queue resets with previously solved items for continued training
- **Comprehensive Logging**: Shows task names, group average scores, solve reasons, and contextual messages

**2. Dataset State Persistence:**
- **Automatic Dumping**: Saves active queue every 100 iterations to `atropos/environments/datasets/remaining_unsolved.jsonl`
- **Rich Metadata**: Includes attempt counts, queue positions, iteration info, and curriculum state
- **Resume Capability**: `resume_from_unsolved_dataset` config option to load from saved state
- **Conflict Handling**: When both `dataset_name` and `resume_from_unsolved_dataset` are set:
  - Training items come from resume file (overrides dataset_name)
  - Test/evaluation items come from dataset_name for consistent evaluation
  - System validates compatibility and warns about mismatches

**3. Data Dumping Infrastructure:**
- **Structured Conversations**: Saves rollouts as proper chat conversations with role/content format
- **Group Format**: Data saved with group-level metadata including constraint details and group average scores
- **Configurable Thresholds**: `rollout_save_score_threshold` (default: 0.7) for filtering quality rollouts
- **Failed Rollout Tracking**: Separate `dump_failed_rollouts` option for debugging constraint violations
- **Batch Processing**: Automatic saving when buffers reach size limits (100 for rollouts, 50 for failed)
- **Unique Identifiers**: Each run gets a UUID for file organization
- **Save Location**: `atropos/environments/data_dumps/` with descriptive filenames

**4. Enhanced Logging and Monitoring:**
- **Log Suppression**: `suppress_base_env_logs` (default: True) reduces verbose base environment, httpx, and httpcore logs
- **Curriculum Metrics**: WandB tracking of active items, solved items, percent solved, and average attempts
- **Group-Level Insights**: Shows which tasks are being mastered vs. which remain challenging
- **Training Progress**: Clear indication when groups are skipped for being too easy vs. used for training

**Configuration Options (`IFConfig`):**
- `dataset_name`: Primary dataset (default: "allenai/RLVR-IFeval")
- `dataset_config_name`: Optional dataset configuration
- `test_set_ratio`: Test set proportion (default: 0.05)
- `dump_rollouts`: Enable successful rollout saving (default: False)
- `dump_failed_rollouts`: Enable failed rollout saving for debugging (default: False)
- `rollout_save_score_threshold`: Minimum score for saving rollouts (default: 0.7)
- `max_group_average_for_training`: Skip groups above this score (default: 0.75)
- `dataset_shuffle_seed`: Reproducible dataset shuffling (default: 42)
- `resume_from_unsolved_dataset`: Path to resume file (default: None)
- `suppress_base_env_logs`: Reduce verbose logging (default: True)
- `solve_on_single_correct`: Mark item as solved if any rollout gets it correct (default: False)

**Verifier Functions:**
Comprehensive map of 24 verifier functions (`IF_FUNCTIONS_MAP`) covering diverse constraints:
- **Content Requirements**: `verify_keywords`, `verify_keyword_frequency`, `validate_forbidden_words`
- **Format Constraints**: `validate_json_format`, `validate_title`, `validate_quotation`
- **Structure Requirements**: `verify_paragraph_count`, `verify_bullet_points`, `validate_sections`
- **Language Constraints**: `validate_response_language`, `validate_uppercase`, `validate_lowercase`
- **Length Requirements**: `validate_word_constraint`, `verify_sentence_constraint`
- **Special Formatting**: `verify_postscript`, `validate_placeholders`, `validate_highlighted_sections`
- **Response Patterns**: `validate_repeat_prompt`, `validate_two_responses`, `validate_end`
- **Character Constraints**: `verify_letter_frequency`, `validate_no_commas`
- **Advanced Features**: `validate_choice`, `validate_frequency_capital_words`

**Usage Examples:**
```bash
# Basic training
python instruction_following_algorithm_environment.py serve

# With data dumping enabled
python instruction_following_algorithm_environment.py serve \
    --env.dump_rollouts=True \
    --env.rollout_save_score_threshold=0.8

# Resume from previous session
python instruction_following_algorithm_environment.py serve \
    --env.resume_from_unsolved_dataset="atropos/environments/datasets/remaining_unsolved.jsonl"

# Adjust difficulty threshold
python instruction_following_algorithm_environment.py serve \
    --env.max_group_average_for_training=0.8

# Enable single-correct solving (remove items immediately when any rollout succeeds)
python instruction_following_algorithm_environment.py serve \
    --env.solve_on_single_correct=True
```

**Evaluation Metrics:**
- `eval/percent_correct`: Overall accuracy on test set
- `curriculum/active_items`: Number of items still in training circulation
- `curriculum/solved_items`: Number of items removed as solved
- `curriculum/percent_solved`: Percentage of total items solved
- `curriculum/avg_attempts_active`: Average attempts for items still in circulation
- `train/percent_correct`: Training accuracy with group-level insights

**Specialized Dataset Processing:**
- Robust parsing of `allenai/RLVR-IFeval` format with comprehensive error handling
- Extraction of user instructions, verifier function names, and arguments
- Validation of verifier function availability in `IF_FUNCTIONS_MAP`
- Fallback to dummy dataset if primary dataset loading fails
- Configurable dataset shuffling for reproducible experiments

---



### SWE-RL Environment (`swe_rl_env.py`)

Software Engineering Reinforcement Learning environment for training models to fix bugs based on issue descriptions and code context.

**Dependencies:**
- `datasets` (Hugging Face)
- `difflib`
- `wandb`
- `pydantic`

**Dataset:**
- Default: `princeton-nlp/SWE-bench_Lite_oracle`
- Configurable via `SWERLEnvConfig` (e.g., `dataset_name`, `dataset_split_train`, `dataset_split_eval`).

**Input Format (for the model via prompts):**
- `problem_statement`: The issue text.
- `content`: Relevant code segments from one or more files.

**System Prompts:**
1.  **Thinking System Prompt:**
    ```
    You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem.
    ```
2.  **Task System Prompt:**
    ```
    A user will ask you to solve a task. You should generate the solution. Your response format must follow the template below:
    ```
    (Followed by instructions on the SEARCH/REPLACE format)

**User Prompt Template:**
```
We are currently solving the following issue within our repository. Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---
Below are some code segments, each from a relevant file. One or more of these files may contain bugs.
--- BEGIN FILE ---
``` {content} ```
--- END FILE ---
Please first localize the bug based on the issue statement, and then generate *SEARCH/REPLACE* edits to fix the issue.
Every *SEARCH/REPLACE* edit must use this format:
1. The file path
2. The start of search block: <<<<<<< SEARCH
3. A contiguous chunk of lines to search for in the existing source code
4. The dividing line: =======
5. The lines to replace into the source code
6. The end of the replace block: >>>>>>> REPLACE
Here is an example:
```python
### mathweb/flask/app.py
import math
from flask import Flask
```
Please note that the *SEARCH/REPLACE* edit REQUIRES PROPER INDENTATION. If you would like to add the line ’ print(x)’, you must fully write that out, with all those spaces before the code!
Wrap each *SEARCH/REPLACE* edit in a code block as shown in the example above. If you have multiple *SEARCH/REPLACE* edits, use a separate code block for each one.
```

**Reward Function:**
- Primary reward is based on the `SequenceMatcher` ratio between the model's reconstructed generated patch and the oracle patch.
- A score of -1.0 is given initially.
- If the model's response has a `finish_reason` of "length", or if `<think>` tags are present but malformed, the reward remains -1.0 and advantage is set to zero for "length".
- If the SEARCH/REPLACE patch format is correctly parsed from the model's output (after potentially extracting content from `<think> </think>` tags):
    - The `SequenceMatcher.ratio()` between the reconstructed predicted patch and the `oracle_patch_str` is used as the reward.
- Buffers track:
    - `percent_format_correct_buffer`: Percentage of responses with correctly formatted patches.
    - `similarity_score_buffer`: List of similarity scores for correctly formatted patches.
    - `think_tags_present_buffer`: Percentage of responses where `<think>` tags were present.
    - `think_tags_well_formed_buffer`: Percentage of responses where `<think>` tags were present AND well-formed.

**Evaluation Metrics:**
- `eval/avg_similarity_score_correct_patch_format`: Average similarity score for responses that had a correctly formatted patch.
- `eval/patch_format_accuracy`: Proportion of evaluation items where the patch was correctly formatted.
- `eval/pass_at_1`: Proportion of evaluation items where the patch was correct and achieved a similarity score of 1.0.
- `eval/avg_think_tags_present`: Average presence of think tags in evaluation responses.
- `eval/avg_think_tags_well_formed`: Average well-formedness of think tags in evaluation responses.

**Unique Configuration and Features:**
- **Dataset Handling:** Loads training and test data from Hugging Face datasets, specifically tailored for SWE-bench like formats.
- **Patch Parsing:** Implements robust parsing for a specific SEARCH/REPLACE patch format.
- **Thinking Tag Processing:** Extracts content after `<think> </think>`

---

### Text Reversal Environment (`text_reversal_environment.py`)

Environment for training and evaluating exact string reversal with optional thinking and split train/eval context lengths.

**Dataset:**
- `PrimeIntellect/Reverse-Text-SFT`

**Input Format:**
- Each item contains two `prompt` messages and one `completion` message:
  - `prompt`: list of messages with roles {`system`, `user`}
  - `completion`: list with a single assistant message containing the reversed text, wrapped in `<reversed_text>...</reversed_text>`

**Prompt Construction:**
- The dataset's system text is NOT used as a system message to the model.
- Instead, it is prepended to the user content with two newline separators and sent as the user turn:
  - Effective user content: `"{dataset_system}\n\n{dataset_user}"`
- Optional thinking system prompt is included only when `use_thinking=True`.

**Reward Function:**
- Extract the model output after the first closing `</think>` tag (if present), trim whitespace.
- Score is 1.0 if the remaining output EXACTLY matches the dataset assistant `completion` content; otherwise 0.0.

**Optional CoT Length Penalty (for correct rollouts only):**
- Enabled by default (`length_penalty_enabled=True`).
- Within each training group, compute CoT token lengths from the content inside the first `<think>...</think>` block of correct rollouts.
- Let L̄ be the average of those lengths. A deadband δ (default 5 tokens) defines a threshold `L̄ + δ`.
- Any correct rollout with length above this threshold is penalized: `score = 1 - α * ((excess / L̄)^p)`, clipped to `[penalty_min_score, 1]`.
  - Defaults: `α=0.5`, `p=2`, `penalty_min_score=0.2`.
- Incorrect rollouts remain at 0.0. If no valid think block (or thinking disabled), penalty is skipped for that rollout.

**Curriculum: One-Epoch + Hard Retries (optional):**
- Controlled by `curriculum_one_epoch_enabled` (default: True).
- First pass (one epoch): each item is attempted once. If any rollout in the group is correct (≥1/N), the item is considered solved and never revisited. If the group has zero correct (0/N), the item is marked “hard” and placed into a retry pool.
- Retry phase: only begins after the first pass over all training items completes. Items in the retry pool are revisited up to `hard_retry_max_attempts` times (default: 3). If still unsolved, they are dropped and training completes naturally when the retry pool is exhausted.
- Tip: Use a large `total_steps`. The environment will stop serving items once the one-epoch + retries queues are exhausted (it raises completion in `get_next_item`).

**Configuration Options (`TextReversalEnvConfig`):**
- `use_thinking` (bool, default: False): include thinking system prompt.
- `dataset_name` (str, default: `PrimeIntellect/Reverse-Text-SFT`): training dataset.
- `eval_dataset_name` (Optional[str], default: None): static eval dataset to use (full split). If `None`, the environment samples `test_set_size` examples from the training dataset for eval.
- `test_set_size` (int, default: 100): number of samples for eval when `eval_dataset_name=None`.
- `max_train_token_length` (int, default: 16384): max tokens for training generations.
- `max_eval_token_length` (int, default: 32768): max tokens for eval generations.
- `length_penalty_enabled` (bool, default: True): enable within-group CoT length penalty for correct rollouts.
- `penalty_deadband_tokens` (int, default: 5): δ deadband added above average length before penalizing.
- `penalty_alpha` (float, default: 0.5): penalty scale.
- `penalty_power` (float, default: 2.0): penalty exponent (quadratic by default).
- `penalty_min_score` (float, default: 0.2): lower bound for penalized correct rollouts.
- `curriculum_one_epoch_enabled` (bool, default: True): enables one-pass training plus a late retry phase for hard items.
- `hard_retry_max_attempts` (int, default: 3): maximum retry attempts per hard item in the retry phase.

**Usage Examples:**
```bash
# Basic training with default 16k train context, 32k eval context, and sampled eval set (100 examples)
python text_reversal_environment.py serve

# Enable thinking system prompt
python text_reversal_environment.py serve \
  --env.use_thinking=True

# Use a static eval dataset instead of sampling from train
python text_reversal_environment.py serve \
  --env.eval_dataset_name="someorg/Reverse-Text-EVAL"

# Override max token lengths if needed
python text_reversal_environment.py serve \
  --env.max_train_token_length=12000 \
  --env.max_eval_token_length=28000

# Adjust/disable the CoT length penalty for correct rollouts
python text_reversal_environment.py serve \
  --env.length_penalty_enabled=False \
  --env.penalty_deadband_tokens=8 \
  --env.penalty_alpha=0.6 \
  --env.penalty_power=2.0 \
  --env.penalty_min_score=0.3

# Enable one-epoch + retries curriculum and set max retries
python text_reversal_environment.py serve \
  --env.curriculum_one_epoch_enabled=True \
  --env.hard_retry_max_attempts=3
```

**Evaluation Metric:**
- `eval/percent_correct`: strict exact-match accuracy on the eval set.
