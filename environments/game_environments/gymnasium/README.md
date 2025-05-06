# Gymnasium Environments

This directory contains RL environments based on OpenAI's Gymnasium.

## Blackjack Environment

A reinforcement learning environment for Blackjack that uses a best-of-n approach to select actions. The model is trained to make function calls with proper formatting to choose between "hit" and "stick" actions.

### Features

- Uses Gymnasium's Blackjack-v1 environment
- Teaches models to use tool-calling format for actions
- Combines environment rewards with format rewards for better training
- Configurable via YAML files

## Usage

### Running with Default Configuration

To run the Blackjack environment with the default configuration:

```bash
python environments/game_environments/gymnasium/blackjack_local_server.py
```

This will use the default configuration from `configs/envs/blackjack.yaml`.

### Custom Configuration

You can specify a custom configuration file:

```bash
python environments/game_environments/gymnasium/blackjack_local_server.py --config my_custom_config
```

The `--config` parameter can be:

1. A name (without `.yaml` extension) which will be looked up in `configs/envs/`
2. A relative or absolute path to a YAML file

For example:
```bash
# Using a config in configs/envs/
python environments/game_environments/gymnasium/blackjack_local_server.py --config blackjack_hard

# Using a config with full path
python environments/game_environments/gymnasium/blackjack_local_server.py --config /path/to/my/config.yaml
```

## Configuration Structure

The configuration file follows this structure:

```yaml
# Base environment parameters
tokenizer_name: "NousResearch/DeepHermes-3-Llama-3-8B-Preview"
group_size: 1
use_wandb: false
# ... other base parameters

# Blackjack specific configuration
blackjack:
  # Environment parameters
  env_name: "Blackjack-v1"
  temperature: 0.7
  top_p: 0.9
  # ... other Blackjack specific parameters
  
  # Reward function configuration
  reward_functions: ["format"]
  format_reward_weight: 0.2

# Server configuration
server_configs:
  - model_name: "${OPENAI_MODEL:gpt-4.1-nano}"
    base_url: "${OPENAI_API_BASE}"
    api_key: "${OPENAI_API_KEY}"
    num_requests_for_eval: 256
```

### Important Configuration Parameters

#### Base Parameters

- `tokenizer_name`: The tokenizer to use for encoding/decoding text
- `group_size`: Number of model responses to evaluate for each action (best-of-n)
- `max_token_length`: Maximum token length for generation

#### Blackjack Specific Parameters

- `env_name`: The Gymnasium environment name ("Blackjack-v1")
- `temperature`: Temperature for model generation
- `top_p`: Top-p sampling parameter
- `max_steps`: Maximum steps per episode
- `reward_functions`: List of reward functions to apply (e.g., ["format"])
- `format_reward_weight`: Weight for the format reward

#### Server Configuration

- `model_name`: LLM model to use (can use environment variables with ${VAR_NAME:default} syntax)
- `base_url`: Base URL for the model API (optional)
- `api_key`: API key for the model
- `num_requests_for_eval`: Number of evaluation requests to allocate

## Environment Variables

The server configuration supports using environment variables with optional default values:

- `${VARIABLE_NAME}`: Uses the value of the environment variable
- `${VARIABLE_NAME:default}`: Uses the default value if the environment variable is not set

For example:
```yaml
model_name: "${OPENAI_MODEL:gpt-4.1-nano}"
```

This will use the value of the OPENAI_MODEL environment variable if set, otherwise it will use "gpt-4.1-nano". 