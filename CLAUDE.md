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
The InternBootcamp environment generates verifiable reasoning tasks for training LLMs. The new setup uses the `serve` mode with SFT data collection instead of the older `process` mode.

### Key Files
- **SLURM Script**: `intern_bootcamp_datagen.slurm`
- **Serve Config**: `environments/intern_bootcamp/config_serve.yaml`
- **Environment**: `environments/intern_bootcamp/intern_bootcamp_env.py`
- **Output Data**: `~/atropos/data/intern_bootcamp_rollouts_*.jsonl`

### Architecture
1. **SGLang Server**: Runs on 8 B200 GPUs (180GB VRAM each)
   - Data Parallelism (DP): 4 replicas
   - Tensor Parallelism (TP): 2 GPUs per replica
   - Single endpoint on port 9000
   
2. **Atropos API Server**: Manages trajectory collection on CPUs
   - Launched with `run-api` (non-blocking)
   - Coordinates between environment and inference
   
3. **InternBootcamp Environment**: Generates problems and collects responses
   - Runs in `serve` mode with SFT data dumping enabled
   - Saves rollouts periodically to JSONL files

### Running Data Generation
```bash
# Submit the job
sbatch intern_bootcamp_datagen.slurm

# Monitor progress
tail -f logs/$SLURM_JOB_ID/api.log
tail -f logs/$SLURM_JOB_ID/sglang.log  
tail -f logs/$SLURM_JOB_ID/intern_bootcamp.log

# Check data generation
ls -la ~/atropos/data/intern_bootcamp_rollouts_*.jsonl
```

### Configuration Details
- **Model**: Configure in SLURM script (`MODEL_NAME` variable)
- **Data Output**: `~/atropos/data/` (no shared `/data/` on BS200)
- **Logs**: `logs/$SLURM_JOB_ID/`
- **Rollout Dumping**: Enabled via `dump_rollouts: true`
- **Batch Size**: 16 responses per problem for rejection sampling
- **Total Steps**: 1000 problems (16,000 total responses)

### Post-Processing
After data generation completes:
```bash
# Convert rollouts to SFT format
atropos-sft-gen ~/atropos/data/intern_bootcamp_rollouts_*.jsonl \
    --tokenizer NousResearch/Hermes-3-Llama-3.1-8B \
    --output ~/atropos/data/intern_bootcamp_sft.jsonl

# Filter high-quality responses (optional)
jq 'select(.score > 0.5)' ~/atropos/data/intern_bootcamp_sft.jsonl > ~/atropos/data/intern_bootcamp_sft_filtered.jsonl
```

### Monitoring and Debugging
- WandB tracking enabled for real-time metrics
- Check SGLang health: `curl http://localhost:9000/v1/models`
- API server status: `curl http://localhost:8000/health`
- Environment saves rollouts every 100 items to prevent data loss

### Common Issues
1. **SGLang not starting**: Check VRAM usage and model size
2. **API server connection failed**: Ensure run-api started successfully
3. **No data output**: Check `dump_rollouts: true` in config
4. **Out of memory**: Adjust `--mem-fraction-static` in SGLang or reduce DP/TP
5. **SLURM job stuck in PD**: Check available nodes with `sinfo` and ensure sufficient GPU resources

### Current Setup Status (2025-06-30)
The SGLang server launch command has been updated to use the working configuration:
```bash
nohup python3 -m sglang.launch_server \
    --model ${MODEL_NAME} \
    --tp 8 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 9000 \
    --disable-outlines-disk-cache \
    --grammar-backend xgrammar \
    --attention-backend triton
```

**Next Steps**:
1. Cancel pending job if needed: `scancel 624`
2. Resubmit from head node: `sbatch intern_bootcamp_datagen.slurm`
3. Monitor job startup: `squeue -u $USER`
4. Once running, check logs: `tail -f logs/$SLURM_JOB_ID/*.log`
5. Verify SGLang health: `curl http://localhost:9000/v1/models`

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