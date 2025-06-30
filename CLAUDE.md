# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Atropos is a reinforcement learning environment microservice framework for async RL with LLMs, developed by Nous Research. It provides a scalable platform for training language models through diverse interactive environments and now includes SFT data generation capabilities.

## Key Development Commands

### Environment Setup
```bash
# Basic installation
pip install -e .                # for using
pip install -e .[dev]           # for development  
pip install -e .[examples]      # for running examples
pip install -e .[all]           # for everything

# Pre-commit hooks (required for contributors)
pre-commit install
```

### Core Commands
```bash
# Start the central API server
run-api

# Run environments in serve mode (distributed with API server)
python environments/gsm8k_server.py serve --openai.model_name Qwen/Qwen2.5-1.5B-Instruct --slurm false
python environments/gsm8k_server.py serve --config environments/configs/example.yaml

# Run environments in process mode (local testing/data generation)
python environments/gsm8k_server.py process --env.data_path_to_save_groups output.jsonl

# Data processing tools
view-run                         # Gradio UI for inspecting rollouts
atropos-sft-gen output.jsonl --tokenizer Qwen/Qwen2.5-1.5B-Instruct  # Convert to SFT format
atropos-dpo-gen output.jsonl --tokenizer Qwen/Qwen2.5-1.5B-Instruct  # Convert to DPO format
```

### Testing
```bash
pytest                          # Run tests from atroposlib/tests directory
```

## Architecture Overview

### Core Components

1. **API Server** (`atroposlib/api/server.py`): Central FastAPI server that coordinates between environments and trainers
   - Manages trajectory queues and batch distribution
   - Handles environment registration and status tracking
   - Provides endpoints for scored data submission and retrieval
   - Supports both RL training and SFT data collection modes

2. **Base Environment** (`atroposlib/envs/base.py`): Abstract base class for all environments
   - `BaseEnv` class with key methods: `collect_trajectories()`, `get_next_item()`, `evaluate()`
   - Built-in CLI support with `serve` and `process` subcommands
   - Automatic wandb integration and checkpoint management
   - Server management for inference endpoints

3. **Environment Ecosystem** (`environments/`): Collection of ready-to-use RL environments
   - Dataset environments (GSM8K, MMLU, MATH)
   - Interactive games (Blackjack, Chess) 
   - Code execution environments
   - RLAIF/RLHF environments
   - InternBootcamp reasoning tasks
   - Community contributions in `environments/community/`

4. **Training Integration**:
   - **RL Training**: `example_trainer/` with GRPO implementation
   - **SFT Data Collection**: Environments can dump rollouts for supervised fine-tuning
   - Integration with SGLang for high-performance inference

### Key Configuration Patterns

- **Environment Configuration**: Environments use `BaseEnvConfig` with CLI override support
- **Server Configuration**: Uses `APIServerConfig` or `OpenaiConfig` for OpenAI-compatible inference endpoints
- **CLI Pattern**: All environments support `python env_script.py serve|process` with YAML config files
- **Namespace Convention**: CLI args use `--env.param` and `--openai.param` namespacing

### Data Flow Architecture

```
Environment → collect_trajectories() → ScoredDataGroup → API Server Queue
                     ↑                                        ↓
             Inference Server                          Trainer get_batch()
                     ↑                                        ↓
                Model Updates ← Training Step ← Batch Processing

For SFT Data Collection:
Environment → collect_trajectories() → dump_rollouts → JSONL files → atropos-sft-gen → SFT dataset
```

### Important Base Classes

- `BaseEnv`: Main environment interface requiring `collect_trajectories()`, `get_next_item()`, `evaluate()`
- `ScoredDataGroup`/`ScoredDataItem`: Standard data format for trajectories with scores
- `Message`: TypedDict for chat message format with optional rewards
- `APIServerConfig`/`OpenaiConfig`: Configuration for OpenAI-compatible inference endpoints

### File Organization

- `atroposlib/`: Core library with base classes and utilities
- `environments/`: Environment implementations (add new envs to `community/` subdirectory)
- `example_trainer/`: Reference RL training implementation
- `intern_bootcamp_datagen.slurm`: SLURM script for BS200 cluster data generation
- Configuration docs: `CONFIG.md`, `atroposlib/envs/README.md`, `environments/community/README.md`

## InternBootcamp Data Generation on BS200 Cluster

### Overview
The InternBootcamp environment generates verifiable reasoning tasks for training LLMs. The setup uses `serve` mode with SFT data collection and a fake trainer to enable data dumping without actual RL training.

### Key Files
- **SLURM Script**: `intern_bootcamp_datagen.slurm`
- **Serve Config**: `environments/intern_bootcamp/config_serve.yaml`
- **Environment**: `environments/intern_bootcamp/intern_bootcamp_env.py`
- **Fake Trainer**: `fake_trainer.py` - Required to fetch batches from API server
- **Output Data**: `/home/maxpaperclips/atropos/data/intern_bootcamp_*.jsonl`

### Architecture
1. **SGLang Server**: Runs on 8 B200 GPUs (180GB VRAM each)
   - Tensor Parallelism (TP): 8 GPUs
   - Model: deepseek-ai/DeepSeek-R1 (high quality reasoning)
   - Port: 9000
   
2. **Atropos API Server**: Manages trajectory collection
   - Launched with `run-api` (non-blocking)
   - Requires a trainer to register and fetch batches
   - Port: 8000
   
3. **Fake Trainer**: Enables data collection without RL training
   - Registers with API server as a trainer
   - Continuously fetches batches to allow environment to proceed
   - Required because API server blocks environments until trainer starts
   
4. **InternBootcamp Environment**: Generates problems and collects responses
   - Runs in `serve` mode with `dump_rollouts: true`
   - Generates 16 responses per problem for rejection sampling
   - Saves data every 100 problems

### Running Data Generation
```bash
# Submit the job
sbatch intern_bootcamp_datagen.slurm

# Monitor progress
tail -f logs/$SLURM_JOB_ID/api.log
tail -f logs/$SLURM_JOB_ID/sglang.log  
tail -f logs/$SLURM_JOB_ID/intern_bootcamp.log
tail -f logs/$SLURM_JOB_ID/fake_trainer.log

# Check data generation (note: may be in literal ${HOME} directory due to bug)
ls -la ~/atropos/data/intern_bootcamp*.jsonl
ls -la '${HOME}'/atropos/data/intern_bootcamp*.jsonl
```

### Configuration Details
- **Model**: deepseek-ai/DeepSeek-R1 (configured in SLURM script)
- **Data Output**: `/home/maxpaperclips/atropos/data/`
- **Total Steps**: 50,000 problems (800,000 total responses with group_size=16)
- **Max Token Length**: 16,384 (appropriate for 8B model training)
- **Temperature**: 0.7, Top-p: 0.9
- **Logs**: `logs/$SLURM_JOB_ID/`

### Critical Fixes Applied

1. **Git Submodule**: InternBootcamp library must be initialized
   ```bash
   git submodule update --init --recursive
   cd environments/intern_bootcamp/internbootcamp_lib
   uv pip install -e .
   ```

2. **Disable WandB**: Add `--env.use_wandb false` to environment launch

3. **Import Errors**: Fixed in `sft_loader_server.py`
   - Changed `OpenaiConfig` to `APIServerConfig`
   - Import from `atroposlib.envs.server_handling.server_baseline`

4. **Fake Trainer Required**: API server requires active trainer
   - Created `fake_trainer.py` that registers and fetches batches
   - Without this, environment gets stuck "waiting for trainer to start"

5. **Path Expansion Bug**: Config uses absolute paths
   - Environment doesn't expand `${HOME}` variables
   - Use absolute paths like `/home/maxpaperclips/atropos/data/`

### Post-Processing
After data generation completes:
```bash
# Convert rollouts to SFT format
atropos-sft-gen ~/atropos/data/intern_bootcamp_rollouts_*.jsonl \
    --tokenizer deepseek-ai/DeepSeek-R1 \
    --output ~/atropos/data/intern_bootcamp_sft.jsonl

# If data is in ${HOME} directory:
atropos-sft-gen '${HOME}'/atropos/data/intern_bootcamp_rollouts_*.jsonl \
    --tokenizer deepseek-ai/DeepSeek-R1 \
    --output ~/atropos/data/intern_bootcamp_sft.jsonl

# Filter high-quality responses (optional)
jq 'select(.score > 0.5)' ~/atropos/data/intern_bootcamp_sft.jsonl > ~/atropos/data/intern_bootcamp_sft_filtered.jsonl
```

### Common Issues and Solutions

1. **ModuleNotFoundError: internbootcamp**
   - Initialize git submodule and install with uv pip

2. **Environment stuck "waiting for trainer"**
   - Launch fake_trainer.py to register and fetch batches
   - API server requires active trainer even for data collection

3. **No data files appearing**
   - Check literal `${HOME}` directory due to path expansion bug
   - Environment saves every 100 problems, be patient

4. **Import errors in sft_loader_server**
   - Use APIServerConfig instead of OpenaiConfig
   - Import from correct module path

5. **SGLang launch failures**
   - Use working command with TP=8, xgrammar backend, triton attention
   - Ensure sufficient GPU memory (8x B200 GPUs)

### Data Output Structure
- **Main data file**: `intern_bootcamp_serve_data_N.jsonl` (increments if exists)
- **Rollout files**: `intern_bootcamp_rollouts_UUID_NNNN.jsonl`
- Each problem generates 16 responses for rejection sampling
- Files include full conversations with reasoning traces

## Development Guidelines

- New environments should inherit from `BaseEnv` and be placed in `environments/community/`
- Follow the established CLI pattern with `serve` and `process` subcommands
- Use the built-in wandb integration for logging and metrics
- Environments should support both distributed (`serve`) and local testing (`process`) modes
- All environments must implement the core abstract methods: `collect_trajectories()`, `get_next_item()`, `evaluate()`, `setup()`
- For SFT data collection, set `dump_rollouts: true` in environment config

## Testing and Debugging

- Use `process` subcommand for local environment testing without full distributed setup
- `view-run` provides Gradio interface for inspecting generated rollouts
- Output files are automatically saved as both `.jsonl` and `.html` formats for easy inspection
- Built-in retry logic and error handling for robust distributed operation
- For cluster debugging, check logs in `logs/$SLURM_JOB_ID/`