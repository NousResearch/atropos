# Pydantic Schema Following Environment

This environment trains language models to generate JSON objects that conform to Pydantic schemas. The environment has been updated to load schemas dynamically from a dataset instead of using hardcoded schemas.

## Dataset Format

The environment expects a dataset with the following columns:

- **problem_id** (string): Unique identifier for each problem
- **task_type** (string): Should be "pydantic_adherance"
- **prompt** (string): The user prompt that asks the model to generate JSON for a specific Pydantic schema
- **verification_info** (string): JSON string containing:
  - `pydantic_config`: The complete Pydantic model definition as executable Python code
  - `model_name`: The name of the target Pydantic model class
- **metadata** (string): Additional metadata (optional)

### Example Dataset Entry

```json
{
  "problem_id": "pydantic_adherance_PuXNOOXO",
  "task_type": "pydantic_adherance",
  "prompt": "Below you see a pydantic model named FestivalLineup. Return a json that, when parsed to a dict, is compatible with the model. Here is the pydantic config:\n\n```python\nfrom pydantic import BaseModel, model_validator, ConfigDict, ValidationError, HttpUrl\nfrom typing import List, Dict, Literal\nfrom datetime import date, time\n\nclass Artist(BaseModel):\n    model_config = ConfigDict(extra=\"forbid\")\n    name: str\n    genre: Literal['rock', 'electronic', 'jazz', 'pop', 'hiphop']\n    popularity_score: int\n    social_links: Dict[str, HttpUrl] = {}\n\n    @model_validator(mode='after')\n    def check_popularity(cls, m):\n        if not (0 <= m.popularity_score <= 100):\n            raise ValidationError(...)\n        return m\n\n# ... more model definitions ...\n```\n\nReturn the json and nothing else.",
  "verification_info": "{\"pydantic_config\": \"from pydantic import BaseModel, model_validator, ConfigDict, ValidationError, HttpUrl\\nfrom typing import List, Dict, Literal\\nfrom datetime import date, time\\n\\nclass Artist(BaseModel):\\n    model_config = ConfigDict(extra=\\\"forbid\\\")\\n    name: str\\n    genre: Literal['rock', 'electronic', 'jazz', 'pop', 'hiphop']\\n    popularity_score: int\\n    social_links: Dict[str, HttpUrl] = {}\\n\\n    @model_validator(mode='after')\\n    def check_popularity(cls, m):\\n        if not (0 <= m.popularity_score <= 100):\\n            raise ValidationError(...)\\n        return m\\n\\n# ... complete model definitions ...\", \"model_name\": \"FestivalLineup\"}",
  "metadata": "{\"difficulty\": 0}"
}
```

## Key Changes from Previous Version

1. **Dynamic Schema Loading**: Instead of importing hardcoded Pydantic schemas, the environment now dynamically creates Pydantic models from the `pydantic_config` code in the dataset.

2. **Dataset Integration**: Uses HuggingFace `datasets` library to load schema definitions and prompts.

3. **Flexible Prompts**: The prompts come directly from the dataset, allowing for more varied and sophisticated prompt engineering.

4. **Model Caching**: Dynamically created Pydantic models are cached to avoid recompilation.

## Usage

### Basic Setup

```python
from atropos.environments.pydantic_schema_following_environment.pydantic_schema_following_environment import PydanticSchemaFollowingEnv
from atroposlib.envs.base import BaseEnvConfig, APIServerConfig

# Configure for your dataset
env_config = BaseEnvConfig(
    dataset_name="your_username/your_pydantic_dataset",
    dataset_split="train",
    # ... other config options
)

server_configs = [
    APIServerConfig(
        model_name="your_model_name",
        base_url="your_api_endpoint",
        # ... other server config
    )
]

# Create environment
env = PydanticSchemaFollowingEnv(
    config=env_config,
    server_configs=server_configs
)
```

### Using the Example Configuration

```python
from atropos.environments.pydantic_schema_following_environment.example_config import create_config_with_dataset

# Create configuration for your dataset
env_config, server_configs = create_config_with_dataset("your_username/your_pydantic_dataset")

# Initialize environment
env = PydanticSchemaFollowingEnv(env_config, server_configs)
```

## How It Works

1. **Setup Phase**:
   - Loads the dataset using HuggingFace `datasets`
   - Splits into train/test sets (80/20 by default)
   - Initializes model cache for dynamic Pydantic models

2. **Training Loop**:
   - Gets next item from dataset (cycles through training set)
   - Sends prompt to language model
   - Collects multiple completions per prompt
   - Scores each completion by validating against the Pydantic schema

3. **Scoring**:
   - Extracts JSON from model response using `<json_output>` tags
   - Dynamically creates Pydantic model from `pydantic_config`
   - Validates JSON against the model (1.0 for valid, 0.0 for invalid)
   - Applies length penalty if responses are too verbose

4. **Evaluation**:
   - Runs on test set with lower temperature
   - Reports average score and percentage of perfect scores

## Dynamic Model Creation

The environment can handle any Pydantic model definition by executing the `pydantic_config` code in a controlled namespace that includes all necessary imports:

- `BaseModel`, `Field`, `model_validator`, `field_validator`
- `ConfigDict`, `ValidationError`, `HttpUrl`, `EmailStr`
- Type hints: `List`, `Dict`, `Optional`, `Union`, `Any`, `Literal`
- Date/time types: `datetime`, `date`, `time`, `timedelta`
- Other types: `Enum`, `Decimal`, `UUID`

## Monitoring and Logging

The environment integrates with Weights & Biases (wandb) for monitoring:

- **Training metrics**: Percentage of perfect scores
- **Evaluation metrics**: Average score, percentage perfect
- **Rollout tables**: Sample conversations, scores, and extracted JSON
- **Model information**: Tracks which Pydantic models are being used

## Error Handling

- **Dataset loading errors**: Falls back to empty dataset with warning
- **Model creation errors**: Logs detailed error information and skips problematic items
- **JSON extraction errors**: Assigns score of 0.0
- **Validation errors**: Assigns score of 0.0

## Performance Considerations

- **Model caching**: Avoids recompiling identical Pydantic models
- **Batch processing**: Processes multiple rollouts per item efficiently
- **Evaluation limits**: Limits evaluation to 50 items for faster feedback
- **Length penalties**: Discourages overly verbose responses

This updated environment provides much more flexibility for training models on diverse Pydantic schemas while maintaining the same core training loop and evaluation methodology.
