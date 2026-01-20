# Atropos-SmolaGents Integration

This integration enables the use of SmolaGents' agent capabilities with Atropos' server-based LLM architecture for high-quality training data generation.

## Overview

The integration consists of:

1. **SmolagentsEnv**: A full-fledged Atropos environment for generating high-quality agent trajectories
2. **Process-based execution**: Robust parallel execution of agents in isolated processes
3. **Tools**: File manipulation and web searching tools for the agent to use
4. **Scoring system**: Automatic evaluation of agent responses based on correctness and efficiency

## Files

- `smolagents_env.py`: The complete Atropos environment for generating training data.
- `agent_process_runner.py`: Module for running agents in separate processes.
- `server_proxy.py`: Proxy mechanism for communication between processes and Atropos server.
- `smolagents_model.py`: Process-safe Atropos server model implementation for SmolaGents.
- `download_gaia.py`: Script to download and set up the GAIA benchmark dataset.

**Tools:**
- `tools/file_tools.py`: Tools for reading, writing, and appending to files
- `tools/tavily_tools.py`: Web search and page extraction tools powered by Tavily

## Installation

1. First, make sure you have Atropos installed.
2. Install SmolaGents and dependencies:
   ```bash
   # Install via pyproject.toml (recommended)
   pip install -e .[smolagents-integration]

   # Or install manually (legacy method)
   pip install smolagents
   pip install datasets pandas huggingface_hub
   pip install tavily-python  # Required for web search tools (needed by most GAIA tasks)
   ```

## Environment Variables

The integration uses the following environment variables:

- `OPENAI_API_KEY`: Required for OpenAI API access when using LiteLLM model (test mode) or when using OpenAI models with Atropos.
- `TAVILY_API_KEY`: Required for web search and page extraction tools. You can get a key from [Tavily](https://tavily.com/).

If you need to use a different API key per run, you can also provide them as command-line arguments:
```
--api-key your_api_key
```

## Quick Start

Test the environment with a small batch:

```bash
# Set your API keys
export OPENAI_API_KEY='your-openai-key'
export TAVILY_API_KEY='your-tavily-key'  # Get free key at https://tavily.com

# Run a quick test (processes 2 examples)
python -m environments.smolagents_integration.smolagents_env process \
  --env.group_size 2 \
  --env.total_steps 1 \
  --env.use_wandb false \
  --openai.model_name "gpt-4o-mini" \
  --openai.base_url "https://api.openai.com/v1"
```

**Note:** The GAIA dataset will be automatically downloaded to `data/gaia` on first run if not present. You need HuggingFace access to the GAIA dataset.

## Using the Integration

### Using SmolagentsEnv for Training Data Generation

Generate SFT training data with the following command:

```bash
atropos-sft-gen output.jsonl --tokenizer NousResearch/DeepHermes-3-Llama-3-8B-Preview \
  --save-messages --env smolagents
```

For local testing using OpenAI API directly:

```bash
# Minimal test (processes just 2 examples)
python -m environments.smolagents_integration.smolagents_env process \
  --env.data_path_to_save_groups output/gaia/smolagents_output.jsonl \
  --env.total_steps 1 \
  --env.group_size 2 \
  --env.include_messages true \
  --env.max_concurrent_processes 8 \
  --env.use_chat_completion true \
  --openai.model_name "gpt-4o" \
  --openai.base_url "https://api.openai.com/v1" \
  --openai.api_key x
```

```bash
# Standard run (processes 10 groups of 2 examples each = 20 total examples)
python -m environments.smolagents_integration.smolagents_env process \
  --env.data_path_to_save_groups output/gaia/smolagents_output.jsonl \
  --env.total_steps 10 \
  --env.group_size 2 \
  --env.include_messages true \
  --env.max_concurrent_processes 8 \
  --env.use_chat_completion true \
  --openai.model_name "gpt-4o" \
  --openai.base_url "https://api.openai.com/v1" \
  --openai.api_key x
```

Note: The command syntax uses dots (`.`) to separate namespaces. Also, the OpenAI API key should be set in your environment variables as `OPENAI_API_KEY` or in a `.env` file in the project root.

If you want to use a local server instead of OpenAI:

```bash
python -m environments.smolagents_integration.smolagents_env process \
  --env.data_path_to_save_groups output/gaia/smolagents_output.jsonl \
  --env.total_steps 10 \
  --env.group_size 2 \
  --env.include_messages true \
  --env.max_concurrent_processes 8 \
  --env.use_chat_completion true \
  --openai.model_name "your-model-name" \
  --openai.base_url "http://localhost:8000/v1"
```

To serve the environment for a trainer:

```bash
python -m environments.smolagents_integration.smolagents_env serve \
  --env.rollout_server_url "http://localhost:8000" \
  --env.use_chat_completion true \
  --env.max_concurrent_processes 5 \
  --env.group_size 8 \
  --openai.model_name "your-model-name" \
  --openai.base_url "http://localhost:8000/v1"
```

## How It Works

### SmolagentsEnv

The `SmolagentsEnv` class provides a complete environment for generating high-quality agent trajectories:

1. Loads tasks from the GAIA benchmark dataset
2. Creates a process-safe model implementation for SmolaGents
3. Initializes a CodeAgent with configurable tools
4. Manages agent execution and trajectory collection in parallel processes
5. Scores trajectories based on correctness, efficiency, and reasoning quality
6. Integrates with Atropos SFT generation pipeline

### Process-Based Isolation

The SmolaGents integration supports true parallel execution of agent processes using multiprocessing. This allows for significantly better performance when running multiple agents simultaneously.

#### How It Works

The process-based isolation implementation:
1. Creates a server proxy mechanism to communicate with the Atropos server from child processes
2. Spawns separate Python processes for each agent execution
3. Manages inter-process communication through queues
4. Collects and processes results from all agents

#### Configuration Options

The process-based isolation can be configured through the following options:

```
# Set the maximum number of concurrent processes (default: 8)
--env.max_concurrent_processes=8

# Set the timeout for agent processes in seconds (default: 240)
--env.process_timeout=240
```

## GAIA Benchmark Dataset

The SmolaGents integration uses the GAIA benchmark dataset for generating high-quality agent trajectories. The dataset includes a variety of tasks with file attachments that test reasoning, problem-solving, and tool usage capabilities.

### Automatic Download

The environment automatically downloads the GAIA dataset on first run if it's not found. The dataset will be saved to `data/gaia` by default.

**Requirements:**
- You need access to the GAIA dataset on HuggingFace
- Ensure you're logged in with `huggingface-cli login` or have your `HF_TOKEN` set

### Manual Download (Optional)

If you prefer to download the dataset manually or to a custom location, the integration includes a dedicated script:

```bash
# Download to the default location (data/gaia)
python -m environments.smolagents_integration.download_gaia

# Specify a custom output directory
python -m environments.smolagents_integration.download_gaia --output-dir /path/to/custom/dir

# Use the raw dataset version instead of the annotated version
python -m environments.smolagents_integration.download_gaia --use-raw
```

This script:
1. Downloads the appropriate GAIA dataset from HuggingFace
2. Creates a `GAIA.py` loader file that works with the HuggingFace datasets library
3. Validates that all necessary files were downloaded correctly

To use the dataset, you need to have access to the GAIA repository on HuggingFace. The script will automatically handle authentication using your HuggingFace credentials.

### Dataset Structure

The GAIA dataset is organized into:
- `test/` - Tasks for evaluation
- `validation/` - Additional tasks that can be used for development or validation

Each task includes:
- A question prompt
- A ground truth answer
- Associated files (images, documents, spreadsheets, etc.)

## Troubleshooting

- **Process-related errors**: When using process-based isolation, ensure your code is serializable for multiprocessing. Also, check that proxy communication is working properly.
- **Message format errors**: Check that message conversions between SmolaGents and Atropos formats are correct.
- **Missing GAIA data**: Make sure you've downloaded the GAIA benchmark data correctly. If needed, run `python -m environments.smolagents_integration.download_gaia`. If you encounter permission errors, you might need to request access to the HuggingFace dataset.
- **Web tool errors**: If Tavily tools aren't working, make sure you have set the `TAVILY_API_KEY` environment variable and have installed the `tavily-python` package.
- **Tool import errors**: If you see errors about missing tool modules, ensure your working directory allows proper imports of the tools folder.
- **Permission errors with file tools**: Ensure your process has the correct permissions to read/write files in the directories being accessed.
- **Memory issues**: If you encounter memory usage problems, try lowering the `max_concurrent_processes` parameter.
