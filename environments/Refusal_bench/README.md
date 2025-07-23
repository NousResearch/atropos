# Refusal Benchmark Environment

This environment evaluates model refusal behavior by testing whether models appropriately refuse to answer certain types of questions.

## Recent Changes

### YAML Configuration Support
- **API Keys**: API keys are now loaded from YAML configuration files instead of being hardcoded
- **Configurable Test Set**: The test dataset file is now configurable via the `data_file` parameter
- **Minos Classifier**: Added support for "minos" classifier type in addition to "openai"

## Configuration

The environment now supports YAML-based configuration. The system looks for configuration files in the following order:

1. `configs/refusal_benchmark.yaml`
2. `refusal_benchmark.yaml`
3. `<environment_directory>/configs/refusal_benchmark.yaml`
4. `<environment_directory>/refusal_benchmark.yaml`

### Environment Variables

Set up your API keys as environment variables:

```bash
export OPENAI_API_KEY="your_openai_api_key_here"
export MINOS_API_KEY="your_minos_api_key_here"  # If using Minos classifier
```

### Configuration Files

#### Basic Configuration (`configs/refusal_benchmark.yaml`)

```yaml
# Basic environment settings
tokenizer_name: "gpt2"
group_size: 16
use_wandb: true
total_steps: 500
batch_size: 512

# Environment-specific configuration
data_file: "processed_test_label1.jsonl"  # Path to your test dataset
classifier_type: "openai"  # Options: "openai" or "minos"

# Server configurations
server_configs:
  # Main model server
  - model_name: "gpt-4o-mini"
    base_url: "https://api.openai.com/v1"
    api_key: "${OPENAI_API_KEY}"  # Environment variable reference
    num_max_requests_at_once: 8
    num_requests_for_eval: 32
    server_type: "openai"
    timeout: 120
  
  # Classifier server
  - model_name: "gpt-4o-mini"
    base_url: "https://api.openai.com/v1"
    api_key: "${OPENAI_API_KEY}"
    num_max_requests_at_once: 8
    num_requests_for_eval: 32
    server_type: "openai"
    timeout: 120
```

#### Using Minos Classifier

To use the Minos classifier, update your configuration:

```yaml
classifier_type: "minos"

server_configs:
  # Main model server (OpenAI)
  - model_name: "gpt-4o-mini"
    base_url: "https://api.openai.com/v1"
    api_key: "${OPENAI_API_KEY}"
    num_max_requests_at_once: 8
    num_requests_for_eval: 32
    server_type: "openai"
    timeout: 120
  
  # Minos classifier server
  - model_name: "minos-classifier"
    base_url: "http://localhost:8001/v1"  # Your Minos server URL
    api_key: "${MINOS_API_KEY}"
    num_max_requests_at_once: 8
    num_requests_for_eval: 32
    server_type: "minos"
    timeout: 120
```

## Features

### Classifiers

1. **OpenAI Classifier**: Uses OpenAI models to classify whether responses are refusals
   - Prompts the model to respond with `<answer>Refusal</answer>` or `<answer>Non-Refusal</answer>`
   - Provides confidence scoring based on response clarity

2. **Minos Classifier**: Uses a Minos classifier system for refusal detection
   - Expects responses in format `[CLASSIFICATION]:[CONFIDENCE]`
   - More structured approach to refusal classification

## Configuration Reference

### Core Environment Parameters (inherited from BaseEnvConfig)

These parameters control the basic behavior of the environment:

- **`tokenizer_name`** (default: `"gpt2"`)
  - HuggingFace tokenizer used for encoding/decoding text
  - Should match or be compatible with your model's tokenizer

- **`group_size`** (default: `16`)
  - Number of responses generated per training item
  - Higher values provide more diverse training data but increase cost/time

- **`use_wandb`** (default: `true`)
  - Enable Weights & Biases logging for experiment tracking
  - Set to `false` for local runs without logging

- **`max_num_workers_per_node`** (default: `16`)
  - Maximum concurrent workers for parallel processing
  - Adjust based on your hardware and API rate limits

- **`total_steps`** (default: `500`)
  - Total number of training steps to execute
  - In evaluation mode, this controls the number of items processed

- **`batch_size`** (default: `512`)
  - Batch size for training (set by trainer, not environment)
  - Used for memory management during training

- **`steps_per_eval`** (default: `25`)
  - How frequently to run evaluation during training
  - Set to 0 to disable periodic evaluation

- **`max_token_length`** (default: `1024`)
  - Maximum sequence length for tokenization
  - Should accommodate your prompts + responses

### Dataset Configuration

- **`data_file`** (default: `"processed_test_label1.jsonl"`)
  - Path to your JSONL test dataset
  - Each line should contain: `{"user": "question", "assistant": "response", "original_label": 1}`
  - Can be relative path or absolute path
  - **Usage**: Point to your specific refusal benchmark dataset

- **`max_eval_samples`** (default: `4000`)
  - Maximum number of samples to evaluate from the dataset
  - Use smaller values (e.g., 100) for quick testing
  - Set to `-1` to evaluate all available samples
  - **Usage**: Control evaluation time and cost

### Classifier Configuration

- **`classifier_type`** (default: `"openai"`)
  - Type of refusal classifier to use
  - **Options**:
    - `"openai"`: Uses OpenAI models with structured prompting
      - Expects responses in `<answer>Refusal</answer>` or `<answer>Non-Refusal</answer>` format
      - More reliable for general use cases
    - `"minos"`: Uses Minos classifier system
      - Expects responses in `[CLASSIFICATION]:[CONFIDENCE]` format
      - Requires separate Minos server setup
  - **Usage**: Choose based on your classifier infrastructure

### Generation Parameters

- **`eval_temperature`** (default: `0.7`)
  - Controls randomness in evaluation responses
  - **Range**: 0.0 (deterministic) to 1.0 (very random)
  - **Recommendation**: 0.1-0.3 for consistent evaluation results
  - **Usage**: Lower values for reproducible evaluations

- **`train_temperature`** (default: `0.8`)
  - Controls randomness in training responses
  - **Range**: 0.0 (deterministic) to 1.0 (very random)
  - **Recommendation**: 0.7-1.0 for diverse training examples
  - **Usage**: Higher values generate more varied training data

- **`eval_max_tokens`** (default: `1024`)
  - Maximum tokens generated per evaluation response
  - Prevents overly long completions during evaluation
  - **Usage**: Set based on expected response length (typical refusals are short)

- **`train_max_tokens`** (default: `1024`)
  - Maximum tokens generated per training response
  - Can be different from `eval_max_tokens`
  - **Usage**: May want longer responses for training diversity

### Training Configuration

- **`use_label_0_for_training`** (default: `true`)
  - Whether to use label 0 data for training
  - **Label 0**: Questions that should be answered helpfully (non-refusal examples)
  - **Label 1**: Questions that should be refused (used for evaluation only)
  - **Usage**: Set to `false` if you only want evaluation mode

### Output Configuration

- **`data_dir_to_save_evals`** (default: `"results/refusal_evaluation"`)
  - Directory where evaluation results are saved
  - **Creates multiple files**:
    - `metrics.json`: Standard evaluation metrics
    - `samples.jsonl`: Individual sample results
    - `detailed_samples_YYYYMMDD_HHMMSS.jsonl`: Enhanced samples with metadata
    - `evaluation_summary_YYYYMMDD_HHMMSS.json`: Complete evaluation summary
  - Set to `null` to disable file output
  - **Usage**: Organize results by experiment/model

### Server Configuration

The `server_configs` section defines API endpoints:

```yaml
server_configs:
  # Main model server (for the model being evaluated)
  - model_name: "gpt-4o-mini"
    base_url: "https://api.openai.com/v1"
    api_key: "${OPENAI_API_KEY}"
    num_max_requests_at_once: 8
    num_requests_for_eval: 32
    server_type: "openai"
    timeout: 120
  
  # Classifier server (for refusal detection)
  - model_name: "gpt-4o-mini"  # Can be different model
    base_url: "https://api.openai.com/v1"  # Can be different endpoint
    api_key: "${OPENAI_API_KEY}"
    num_max_requests_at_once: 8
    num_requests_for_eval: 32
    server_type: "openai"
    timeout: 120
```

**Key Points**:
- First server is used for the main model being evaluated
- Second server is used for the refusal classifier
- They can be the same model/endpoint or different
- Environment variable references like `${OPENAI_API_KEY}` are supported

## Configuration Examples

### Quick Start (Trial Configuration)

For quick testing and debugging, use `trial.yml`:

```yaml
# Small sample size for fast evaluation
max_eval_samples: 100

# Conservative API settings to avoid rate limits
server_configs:
  - num_max_requests_at_once: 1  # One request at a time
    timeout: 120
  - num_max_requests_at_once: 1
    timeout: 30                  # Faster timeout for classifier
```

### Production Configuration

For comprehensive evaluation, use `configs/refusal_benchmark.yaml`:

```yaml
# Larger sample size for thorough evaluation
max_eval_samples: 4000

# Higher throughput settings
server_configs:
  - num_max_requests_at_once: 8   # More concurrent requests
    timeout: 120
  - num_max_requests_at_once: 8
    timeout: 120
```

### Cost Optimization

To reduce API costs:

```yaml
# Use smaller, cheaper models
server_configs:
  - model_name: "gpt-4o-mini"      # Main model
  - model_name: "gpt-3.5-turbo"   # Cheaper classifier

# Reduce sample size and token limits
max_eval_samples: 1000
eval_max_tokens: 512
train_max_tokens: 512
```

### Local Model Setup

To evaluate local models with OpenAI classifier:

```yaml
server_configs:
  # Local vLLM server
  - model_name: "meta-llama/Llama-2-7b-chat-hf"
    base_url: "http://localhost:8000/v1"
    api_key: "dummy"
    num_max_requests_at_once: 32   # Higher for local
    server_type: "openai"          # vLLM uses OpenAI API
  
  # OpenAI classifier
  - model_name: "gpt-4o-mini"
    base_url: "https://api.openai.com/v1"
    api_key: "${OPENAI_API_KEY}"
    server_type: "openai"
```

### Evaluation-Only Mode

To disable training and only run evaluation:

```yaml
use_label_0_for_training: false   # No training data
total_steps: 1                    # Single evaluation run
steps_per_eval: 0                 # Disable periodic evaluation
```

## Troubleshooting Configuration

### Common Issues

1. **API Key Not Found**: Ensure environment variables are set
   ```bash
   export OPENAI_API_KEY="your_key_here"
   ```

2. **Rate Limits**: Reduce concurrent requests
   ```yaml
   num_max_requests_at_once: 1
   ```

3. **Timeouts**: Increase timeout values
   ```yaml
   timeout: 300  # 5 minutes
   ```

4. **Large Datasets**: Limit evaluation samples
   ```yaml
   max_eval_samples: 500
   ```

### Configuration Validation

The environment validates configuration on startup and will warn about:
- Missing API keys
- Invalid classifier types
- Unreachable server endpoints
- Invalid parameter ranges

## Usage

1. **Set up environment variables:**
   ```bash
   export OPENAI_API_KEY="your_api_key"
   export MINOS_API_KEY="your_minos_key"  # If using Minos classifier
   ```

2. **Create or modify configuration file:**
   ```bash
   # Copy and edit the example configuration
   cp configs/refusal_benchmark.yaml my_config.yaml
   # Edit my_config.yaml with your settings
   ```

3. **Run the environment:**

   **For evaluation only:**
   ```bash
   python refusal_benchmark_environment.py evaluate --config my_config.yaml
   ```

   **For local testing/processing:**
   ```bash
   python refusal_benchmark_environment.py process --config my_config.yaml
   ```

   **For serving (connecting to Atropos trainer):**
   ```bash
   python refusal_benchmark_environment.py serve --config my_config.yaml
   ```

### Available Commands

- **`evaluate`**: Runs evaluation on the test dataset and outputs metrics
- **`process`**: Runs the environment locally for testing and data generation  
- **`serve`**: Runs the environment as a server that connects to the Atropos training infrastructure

### Command Line Options

You can also override configuration values directly from the command line:

```bash
# Override specific settings
python refusal_benchmark_environment.py evaluate \
  --config my_config.yaml \
  --env--classifier_type minos \
  --env--max_eval_samples 500

# Run without config file (uses defaults)
python refusal_benchmark_environment.py evaluate \
  --env--data_file "my_test_data.jsonl" \
  --openai--api_key "$OPENAI_API_KEY"
```

## Data Format

The test dataset should be in JSONL format with the following structure:

```json
{"user": "Question text", "assistant": "Original response", "original_label": 1}
```

Where:
- `user`: The question/prompt
- `assistant`: The original assistant response (optional, for analysis)
- `original_label`: 1 for questions that should be refused, 0 for questions that should be answered

## Evaluation Output

The environment saves comprehensive evaluation results to structured files:

### Files Generated

1. **`metrics.json`** - Standard evaluation metrics in nous-evals format
2. **`samples.jsonl`** - Individual sample results from the framework
3. **`detailed_samples_YYYYMMDD_HHMMSS.jsonl`** - Enhanced sample data with metadata
4. **`evaluation_summary_YYYYMMDD_HHMMSS.json`** - Complete evaluation summary

### Sample Output Structure

Each evaluation sample includes:

```json
{
  "messages": [...],
  "question": "The user's question",
  "model_response": "Model's response",
  "is_refusal": true,
  "refusal_confidence": 0.95,
  "classifier_type": "openai",
  "score": 1.0,
  "correct": true,
  "expected_refusal": true,
  "actual_refusal": true,
  "item_id": "item_123",
  "original_label": 1,
  "evaluation_timestamp": "20241201_143022",
  "generation_params": {
    "temperature": 0.7,
    "max_tokens": 1024
  },
  "token_usage": {
    "prompt_tokens": 45,
    "completion_tokens": 12,
    "total_tokens": 57
  }
}
```

### Summary Metrics

The evaluation summary includes:
- **Accuracy**: Percentage of correctly handled questions
- **Refusal Rate**: Percentage of questions that were refused
- **Confidence Statistics**: Mean, min, max classifier confidence
- **Sample Breakdown**: Counts of refused vs non-refused samples
- **Environment Configuration**: All relevant settings used
