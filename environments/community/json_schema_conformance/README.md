# JSON Schema Conformance Environment

This environment targets the ability of LLMs to generate and correct JSON data that conforms to a given, randomly generated JSON schema.

This is a challenging task for modern LLMs. As noted by [OpenRouter](https://x.com/OpenRouterAI/status/1802495026602070389), even top-tier models can have significant JSON violation rates in production, highlighting the need for improved training. This environment is designed to provide that by generating a wide variety of complex, realistic schema-based challenges.

## Task Description

The environment supports two distinct tasks, configurable via the `--env.task_type` flag:

1.  **`generation` (Default):** The model is given a randomly generated JSON schema and must produce a valid JSON object from scratch that conforms to it.
2.  **`editing`:** The model is given a schema and a *non-conforming* JSON object. Its task is to identify and fix the error, returning a valid, conforming object.

This dual-task structure allows for training models that are not only good generators but also effective, schema-aware editors.

**Example Generation Prompt:**
```
Please generate a valid JSON object that strictly conforms to the following JSON schema. Your response should only be the JSON object itself, with no other text or explanations.

JSON Schema:
```json
{
  "type": "object",
  "properties": {
    "name": {
      "type": "string"
    },
    "email": {
      "type": "string",
      "format": "email"
    },
    "age": {
      "type": "integer",
      "minimum": 18
    }
  },
  "required": [
    "name",
    "email",
    "age"
  ]
}
```

**Example Editing Prompt:**
```
The following JSON object should conform to the provided schema, but it contains an error. Please correct the object and return only the fixed, valid JSON in your response.

JSON Schema:
```json
{
  "type": "object",
  "properties": {
    "name": {
      "type": "string"
    },
    "email": {
      "type": "string",
      "format": "email"
    },
    "age": {
      "type": "integer",
      "minimum": 18
    }
  },
  "required": [
    "name",
    "email",
    "age"
  ]
}
```

Erroneous JSON:
```json
{
  "name": "Jane Doe",
  "email": "jane.doe@not-an-email",
  "age": "seventeen"
}
```

## Scoring

The rewards, which can be adjusted, are as follows:
- **`1.0`**: The model's output is a syntactically correct JSON object that successfully validates against the provided schema.
- **`0.2`**: The output is valid JSON but *fails* schema validation. This provides partial credit for getting the syntax right.
- **`0.0`**: The output is not valid JSON.

## Configuration

The complexity and nature of the tasks can be controlled via the `JSONSchemaConformanceEnvConfig` class or command-line arguments.

| Flag | Description | Default |
|---|---|---|
| `task_type` | Task to perform: `generation` or `editing`. | `generation`|
| `max_properties` | Max properties in a generated object schema. | `5` |
| `max_nesting_depth`| Max nesting depth for generated schemas. | `2` |
| `property_required_prob` | Probability for an object property to be required. | `0.5` |
| `string_pattern_prob`| Probability of adding a regex `pattern` to a string. | `0.2` |
| `string_format_prob` | Probability of adding a `format` (e.g., 'email') to a string. | `0.2` |
| `number_range_prob`| Probability of adding `minimum` and `maximum` to a number. | `0.3` |
| `array_items_range_prob`| Probability of adding `minItems` and `maxItems` to an array. | `0.3` |
| `enum_prob` | Probability of converting a type to a strict `enum`. | `0.1` |
| `debug_logging`| Enable detailed debug logging. | `False` |
| `dump_rollouts`| Dump successful rollouts to JSONL files for analysis. | `True` |
| `dump_failed_rollouts` | Dump failed rollouts (all scores < 0.2) to JSONL files. | `False` |
| `rollout_save_score_threshold` | Minimum score to save a rollout (1.0 saves only perfect ones). | `0.9` |
| `eval_set_size` | Number of fixed schemas to generate for the evaluation set. | `50` |

## Quick Start

To run the environment, first install its dependencies:
```bash
pip install -r environments/community/json_schema_conformance/requirements.txt
```

2.  **Configure LLM Server**: In the `environments/community/json_schema_conformance/` directory, create a file named `llm_config.json`. This file will store your API credentials. Add your model details, API key, and base URL like this:
    ```json
    [
        {
            "model_name": "gpt-4o-mini-2024-07-18",
            "api_key": "YOUR_API_KEY_HERE",
            "base_url": "https://api.openai.com/v1"
        }
    ]
    ```

3.  **Run the Environment**: You can now run the environment using the `process` command. The server configuration will be loaded automatically from your `llm_config.json` file.

    To run the default **generation** task for local testing:
    ```bash
    python environments/community/json_schema_conformance/json_schema_conformance_server.py process --env.data_path_to_save_groups output_generation.jsonl
    ```

    To run the **editing** task:
    ```bash
    python environments/community/json_schema_conformance/json_schema_conformance_server.py process --env.task_type editing --env.data_path_to_save_groups output_editing.jsonl
    ```
