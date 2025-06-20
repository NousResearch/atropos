# Pydantic Schema Following Environment

This environment trains language models to both generate JSON objects that conform to Pydantic schemas and edit erroneous JSON to fix validation errors. The environment supports two task types: **generation** (creating valid JSON from scratch) and **editing** (correcting invalid JSON). Schemas are loaded dynamically from a dataset instead of using hardcoded schemas.

## Dataset Format

The environment expects a dataset with the following columns:

- **problem_id** (string): Unique identifier for each problem
- **task_type** (string): Either "generation" or "editing"
- **prompt** (string): The user prompt that asks the model to generate or correct JSON for a specific Pydantic schema
- **verification_info** (string): JSON string containing:
  - `pydantic_config`: The complete Pydantic model definition as executable Python code
  - `model_name`: The name of the target Pydantic model class
- **erroneous_data** (dict, optional): For editing tasks, contains JSON data with intentional errors that need correction
- **metadata** (string): Additional metadata (optional)

### Example Dataset Entries

**Generation Task:**
```json
{
  "problem_id": "pydantic_adherance_PuXNOOXO",
  "task_type": "generation",
  "prompt": "Below you see a pydantic model named FestivalLineup. Return a json that, when parsed to a dict, is compatible with the model. Here is the pydantic config:\n\n```python\nfrom pydantic import BaseModel, model_validator, ConfigDict, ValidationError, HttpUrl\nfrom typing import List, Dict, Literal\nfrom datetime import date, time\n\nclass Artist(BaseModel):\n    model_config = ConfigDict(extra=\"forbid\")\n    name: str\n    genre: Literal['rock', 'electronic', 'jazz', 'pop', 'hiphop']\n    popularity_score: int\n    social_links: Dict[str, HttpUrl] = {}\n\n    @model_validator(mode='after')\n    def check_popularity(cls, m):\n        if not (0 <= m.popularity_score <= 100):\n            raise ValidationError(...)\n        return m\n\n# ... more model definitions ...\n```\n\nReturn the json and nothing else.",
  "verification_info": "{\"pydantic_config\": \"from pydantic import BaseModel, model_validator, ConfigDict, ValidationError, HttpUrl\\nfrom typing import List, Dict, Literal\\nfrom datetime import date, time\\n\\nclass Artist(BaseModel):\\n    model_config = ConfigDict(extra=\\\"forbid\\\")\\n    name: str\\n    genre: Literal['rock', 'electronic', 'jazz', 'pop', 'hiphop']\\n    popularity_score: int\\n    social_links: Dict[str, HttpUrl] = {}\\n\\n    @model_validator(mode='after')\\n    def check_popularity(cls, m):\\n        if not (0 <= m.popularity_score <= 100):\\n            raise ValidationError(...)\\n        return m\\n\\n# ... complete model definitions ...\", \"model_name\": \"FestivalLineup\"}",
  "metadata": "{\"difficulty\": 0}"
}
```

**Editing Task:**
```json
{
  "problem_id": "pydantic_editing_user_profile_001",
  "task_type": "editing",
  "prompt": "The following JSON data contains validation errors that violate the UserProfile Pydantic model constraints. Please identify and correct all errors to make the data valid.\n\nPydantic Model Definition:\n```python\nfrom pydantic import BaseModel, EmailStr, Field, model_validator\nfrom typing import Literal, Optional\nfrom datetime import date\n\nclass UserProfile(BaseModel):\n    name: str = Field(min_length=2, max_length=50)\n    age: int = Field(ge=0, le=120)\n    email: EmailStr\n    status: Literal['active', 'inactive', 'pending']\n    join_date: date\n    score: Optional[int] = Field(default=None, ge=0, le=100)\n    \n    @model_validator(mode='after')\n    def validate_profile(cls, values):\n        if values.age < 13 and values.status == 'active':\n            raise ValueError('Users under 13 cannot have active status')\n        return values\n```\n\nErroneous JSON to fix:\n```json\n{\n  \"name\": \"\",\n  \"age\": -5,\n  \"email\": \"not-an-email\",\n  \"status\": \"unknown_status\",\n  \"join_date\": \"invalid-date-format\",\n  \"score\": 150,\n  \"extra_field\": \"should_not_exist\"\n}\n```\n\nReturn the corrected JSON in the following format:\n<think>\n[Your analysis of the errors and corrections needed]\n</think>\n\n<json_output>\n[Corrected JSON that passes all validation]\n</json_output>",
  "verification_info": "{\"pydantic_config\": \"from pydantic import BaseModel, EmailStr, Field, model_validator\\nfrom typing import Literal, Optional\\nfrom datetime import date\\n\\nclass UserProfile(BaseModel):\\n    name: str = Field(min_length=2, max_length=50)\\n    age: int = Field(ge=0, le=120)\\n    email: EmailStr\\n    status: Literal['active', 'inactive', 'pending']\\n    join_date: date\\n    score: Optional[int] = Field(default=None, ge=0, le=100)\\n    \\n    @model_validator(mode='after')\\n    def validate_profile(cls, values):\\n        if values.age < 13 and values.status == 'active':\\n            raise ValueError('Users under 13 cannot have active status')\\n        return values\", \"model_name\": \"UserProfile\"}",
  "erroneous_data": {
    "name": "",
    "age": -5,
    "email": "not-an-email",
    "status": "unknown_status",
    "join_date": "invalid-date-format",
    "score": 150,
    "extra_field": "should_not_exist"
  },
  "metadata": "{\"difficulty\": 2, \"error_types\": [\"constraint_error\", \"format_error\", \"enum_error\", \"extra_field\"]}"
}
```

**Expected Model Response:**
```
<think>
I need to analyze the erroneous JSON against the UserProfile model:

1. "name": "" - violates min_length=2 constraint
2. "age": -5 - violates ge=0 constraint
3. "email": "not-an-email" - invalid email format
4. "status": "unknown_status" - not in allowed Literal values ['active', 'inactive', 'pending']
5. "join_date": "invalid-date-format" - invalid date format
6. "score": 150 - violates le=100 constraint
7. "extra_field": "should_not_exist" - not defined in model schema

I'll fix each error:
- Set name to a valid string ≥2 characters
- Set age to a valid integer 0-120
- Use a valid email format
- Use an allowed status value
- Use proper ISO date format
- Set score within 0-100 range
- Remove the extra field
</think>

<json_output>
{
  "name": "John Doe",
  "age": 25,
  "email": "john.doe@example.com",
  "status": "active",
  "join_date": "2023-01-15",
  "score": 85
}
</json_output>
```

**Error Types Demonstrated:**
- **constraint_error**: `name` (min_length), `age` (ge=0), `score` (le=100)
- **format_error**: `email` (EmailStr format), `join_date` (date format)
- **enum_error**: `status` (invalid Literal value)
- **extra_field**: `extra_field` (not in model schema)

## Key Changes from Previous Version

1. **Editing Task Support**: Added comprehensive editing task functionality alongside generation tasks, training models to correct erroneous JSON data.

2. **Sophisticated Error Introduction**: Intelligent error generation system that creates realistic validation errors including type mismatches, constraint violations, format errors, and missing required fields.

3. **Dynamic Schema Loading**: Instead of importing hardcoded Pydantic schemas, the environment now dynamically creates Pydantic models from the `pydantic_config` code in the dataset.

4. **Dataset Integration**: Uses HuggingFace `datasets` library to load schema definitions and prompts.

5. **Flexible Prompts**: The prompts come directly from the dataset, allowing for more varied and sophisticated prompt engineering.

6. **Model Caching**: Dynamically created Pydantic models are cached to avoid recompilation.

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

## Error Introduction Configuration

For editing tasks, the environment includes error introduction. Configure these parameters in your config:

```python
env_config = PydanticEnvConfig(
    # Task configuration
    task_type="editing",  # instead of "generation"

    # Error introduction settings
    error_types_enabled=[
        "type_error",           # Type mismatches (str -> int, etc.)
        "constraint_error",     # Constraint violations (min/max length, numeric bounds)
        "format_error",         # Invalid formats (email, URL, UUID)
        "enum_error",           # Invalid enum values
        "required_field_missing" # Missing required fields
    ],
    max_errors_per_item=1, # How many errors should we introduce?
    error_introduction_probability=1.0, # What percentage of the time should we introduce errors?
    error_introduction_seed=42,  # For reproducible errors

    # ... other config options
)
```

**Available Error Types:**
- **type_error**: Changes field types (string to number, etc.)
- **constraint_error**: Violates field constraints (length, numeric bounds)
- **format_error**: Creates invalid formats for emails, URLs, UUIDs
- **enum_error**: Uses invalid enum values
- **required_field_missing**: Removes required fields
- **extra_field**: Adds unexpected fields
- **nested_error**: Introduces errors in nested objects
- **list_error**: Creates invalid list structures

## How It Works

1. **Setup Phase**:
   - Loads the dataset using HuggingFace `datasets`
   - Splits into train/test sets (80/20 by default)
   - Initializes model cache for dynamic Pydantic models

2. **Training Loop**:
   - Gets next item from dataset (cycles through training set)
   - **For Generation Tasks**: Sends prompt directly to language model
   - **For Editing Tasks**:
     - Automatically generates erroneous data if not provided
     - Creates editing prompt with Pydantic model and erroneous JSON
     - Asks model to identify and fix validation errors
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
- **Error introduction failures**: Falls back to simple error patterns or skips editing tasks
- **JSON extraction errors**: Assigns score of 0.0
- **Validation errors**: Assigns score of 0.0

## Performance Considerations

- **Model caching**: Avoids recompiling identical Pydantic models
- **Batch processing**: Processes multiple rollouts per item efficiently
- **Evaluation limits**: Limits evaluation to 50 items for faster feedback
- **Length penalties**: Discourages overly verbose responses

This updated environment provides comprehensive training capabilities for both JSON generation and editing tasks on diverse Pydantic schemas, with error introduction for realistic editing challenges, while maintaining efficient training loops and robust evaluation methodology.
