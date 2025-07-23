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

### Configurable Parameters

- `data_file`: Path to your test dataset (JSONL format)
- `classifier_type`: Choose between "openai" or "minos" classifiers
- `max_eval_samples`: Maximum number of samples to evaluate
- `eval_temperature`: Temperature for evaluation completions
- `train_temperature`: Temperature for training completions
- `eval_max_tokens`: Maximum tokens for evaluation
- `train_max_tokens`: Maximum tokens for training

## Usage

1. **Set up environment variables:**
   ```bash
   export OPENAI_API_KEY="your_api_key"
   ```

2. **Create or modify configuration file:**
   ```bash
   cp configs/refusal_benchmark.yaml my_config.yaml
   # Edit my_config.yaml with your settings
   ```

3. **Run the environment:**
   ```python
   from atropos.environments.Refusal_bench.refusal_benchmark_environment import RefusalBenchmarkEnv
   
   # The environment will automatically load configuration from YAML
   env_config, server_configs = RefusalBenchmarkEnv.config_init()
   env = RefusalBenchmarkEnv(env_config, server_configs)
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

## Migration from Old Version

If you're upgrading from the previous version:

1. **Remove hardcoded API keys** from your code
2. **Set up environment variables** for your API keys
3. **Create a YAML configuration file** using the examples above
4. **Update data file paths** in the configuration if needed

The environment maintains backward compatibility - if no YAML configuration is found, it will use default values and attempt to load API keys from environment variables. 