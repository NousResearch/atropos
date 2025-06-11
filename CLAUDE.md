# CLAUDE.md - Atropos Project Guide

This document contains project-specific instructions and learnings for Claude assistants working on the Atropos RL training framework.

## Active Data Generation Jobs (as of June 4, 2025)

### InternBootcamp Dataset Generation Status
**Previous Job 13590**: Completed with partial data
- **Output**: `data/intern_bootcamp_deephermes24b_dataset_13590.jsonl`
- **Results**: ~102 lines written, growing slowly due to configuration issues

### Current Issues Being Debugged (Job 13611)
- **Problem**: Environment stuck at step 0, no data being written
- **Root Causes Identified**:
  1. **Batch Size Mismatch**: atropos-sft-gen expects `batch_size = group_size * 8` (512), but environment was using 64
  2. **Token Length Exceeded**: Many responses exceed 14,000 token limit with detailed reasoning
     - Warning in logs: "Token length is too long in a group, skipping..."
     - This causes valid scored groups to be discarded
  3. **Serve Mode Configuration**: Using serve mode with parallel workers for better throughput

### Next Steps to Try
1. **Increase token limit** to 16384 or higher to accommodate long reasoning
2. **Or reduce response length** by adjusting temperature/top_p or system prompt
3. **Monitor token usage** to find optimal balance between reasoning depth and token limits

### Configuration Fixes Applied
1. **Added `batch_size: 512`** to config_process.yaml
2. **Added `--env.batch_size 512`** to SLURM script command line
3. **Added `--env.ensure_scores_are_not_same false`** to keep all rejection samples
4. **Using multi-node setup**: Node 1 runs API+environment, Node 2 runs data collection

### Key Fixes Applied
1. **Error Handling**: Added try-catch in `RandomTask.case_generator()` to skip failing bootcamp tasks
2. **Multi-node Setup**: Using 2-node configuration to separate API/environment from data collection
3. **Network Fix**: Using proper node hostnames for cross-node communication
4. **Batch Size Fix**: Configured batch_size=512 to match atropos-sft-gen expectations
5. **Score Filtering**: Disabled score deduplication for rejection sampling

### Data Consolidation TODO
- Multiple previous runs exist with partial data:
  - `data/intern_bootcamp_deephermes24b_dataset_13563.jsonl` (215MB)
  - `data/intern_bootcamp_deephermes24b_dataset_13576.jsonl` (126MB)
  - `data/intern_bootcamp_deephermes24b_dataset_13572.jsonl` (18MB)
- Also filtered SFT versions available
- Plan to consolidate all datasets before weekend training run

## Project Overview

**Atropos** is a large-scale reinforcement learning framework for training LLMs, primarily using the GRPO (Generalized Preference Optimization) algorithm. It supports both online learning (real-time environment interaction) and offline learning (from stored datasets/trajectories).

### Core Components

- **atroposlib/**: Core training infrastructure with GRPO implementation
- **environments/**: Collection of RL environments for training different capabilities
- **environments/intern_bootcamp/**: Reasoning task environment using InternBootcamp verification

## Build/Test Commands

### General Commands
- **Run environment locally**: `uv run python -m environments.intern_bootcamp.intern_bootcamp_env process --config config.yaml`
- **Filter dataset**: `uv run python environments/intern_bootcamp/filter_dataset.py input.jsonl --format sft --min-score 0.0`
- **Submit SLURM job**: `sbatch environments/intern_bootcamp/run_dataset_generation.slurm`

### Environment Commands
- **Process mode (data generation)**: Uses `process` command instead of `serve` for dataset creation
- **Serve mode (training)**: Uses `serve` command for live RL training with trainer integration

## InternBootcamp Environment

### Purpose
The InternBootcamp environment integrates with the InternBootcamp reasoning task collection to generate high-quality reasoning datasets through rejection sampling.

### Key Features
- **RandomTask mode**: Automatically samples from 1000+ available reasoning tasks
- **Rejection sampling**: Generates multiple responses per problem (default: 16) for quality filtering
- **Verification scoring**: Uses InternBootcamp's built-in verification system for accurate scoring
- **Format requirements**: Expects answers in `[answer]...[/answer]` tags, not LaTeX `\boxed{}`

### Configuration Files
- `config_process.yaml`: Configuration for dataset generation mode
- `filter_dataset.py`: Script to filter by scores and convert to ShareGPT format
- `run_dataset_generation.slurm`: SLURM script for large-scale generation

### Model Integration
- **Current setup**: Uses DeepHermes-3-Mistral-24B-Preview via NousResearch API
- **API endpoint**: `https://inference-api.nousresearch.com/v1`
- **Tokenizer**: Uses same model for tokenization to ensure chat template compatibility

### System Prompt Design
The system prompt correctly:
- ✅ Specifies `<think>` tags for reasoning only
- ✅ Defers to task-specific answer formatting
- ✅ Does NOT specify default answer formats like `\boxed{}`
- ✅ Emphasizes following problem-specific format instructions

### Data Pipeline

1. **Generation**: `process` mode generates raw JSONL with scored groups
2. **Filtering**: `filter_dataset.py` filters by score threshold (typically > 0.0)
3. **Format conversion**: Converts to ShareGPT format for SFT training
4. **Task labeling**: Automatically extracts and labels task types

### ShareGPT Format Output
```json
{
    "conversations": [
        {"from": "system", "value": "..."},
        {"from": "human", "value": "..."},
        {"from": "gpt", "value": "..."}
    ],
    "task_name": "digit_operations",
    "score": 1.0,
    "source": "intern_bootcamp"
}
```

## Scoring System

### InternBootcamp Verification
- **Score ≥ 1.0**: Correct answer with proper format
- **Score > 0**: Correct format but wrong answer
- **Score = 0**: Wrong answer and/or format
- **Score < 0**: Penalty scores (e.g., -0.5 for format violations)

### Common Issues
- **Format mismatch**: Models defaulting to LaTeX `\boxed{}` instead of required `[answer]` tags
- **Incomplete responses**: Complex reasoning tasks may exceed token limits
- **Low success rates**: Expected for challenging reasoning tasks

## Dataset Generation Workflow

### Small Scale Testing (2 steps)
```bash
cd /home/maxpaperclips/atropos
uv run python -m environments.intern_bootcamp.intern_bootcamp_env process \
    --config environments/intern_bootcamp/config_process.yaml
```

### Large Scale Production (10,000 steps)
```bash
sbatch environments/intern_bootcamp/run_dataset_generation.slurm
```

### Post-Processing
```bash
# Filter for good responses (score > 0)
uv run python environments/intern_bootcamp/filter_dataset.py \
    data/intern_bootcamp_deephermes24b_dataset_JOBID.jsonl \
    --format sft --min-score 0.0 --verbose
```

## Architecture Notes

### Process vs Serve Modes
- **Process mode**: For dataset generation, sets `do_send_to_api=False`, saves to JSONL
- **Serve mode**: For live training, sends scored data to trainer via API
- **Key differences**: Process mode has different defaults (ensure_scores_are_not_same=False, include_messages=True)

### Base Environment Structure
- **collect_trajectories()**: Generates multiple responses per item
- **score()**: Applies verification and scoring logic
- **handle_send_to_api()**: Routes scored data to trainer or file output
- **process_manager()**: Main loop for dataset generation mode

### Rejection Sampling Benefits
- Generates diverse attempts at difficult problems
- Enables filtering for high-quality responses
- Provides in-distribution data when using same model family
- Supports curriculum learning through task difficulty analysis

## Lessons Learned

### Model Behavior
- DeepHermes models have strong reasoning capabilities but format compliance issues
- Models trained on mathematical content default to LaTeX formatting
- System prompts correctly designed; issue is model override of specific instructions

### Scaling Considerations
- 10,000 steps × 16 responses = 160,000 total responses
- Expected retention rate varies by task difficulty (5-20% for challenging reasoning)
- API rate limiting requires timeout and retry logic
- SLURM jobs should include sufficient time buffers (24 hours for 10k steps)

### Quality Metrics
- Track score distributions to identify task difficulty patterns
- Monitor task diversity through RandomTask selection
- Use retention rates to tune score thresholds for SFT datasets

## File Locations

### Configuration
- `environments/intern_bootcamp/config_process.yaml`: Main configuration
- `environments/intern_bootcamp/run_dataset_generation.slurm`: SLURM script

### Scripts
- `environments/intern_bootcamp/filter_dataset.py`: Filtering and format conversion
- `environments/intern_bootcamp/run_dataset_generation.sh`: Local testing script

### Outputs
- `data/intern_bootcamp_deephermes24b_dataset_*.jsonl`: Raw scored groups
- `data/intern_bootcamp_deephermes24b_dataset_*_sft.jsonl`: ShareGPT format
- `environments/intern_bootcamp/logs/`: SLURM job logs

## Monitoring Current Job

```bash
# Check progress
wc -l data/intern_bootcamp_deephermes24b_dataset_13590.jsonl
ls -lh data/intern_bootcamp_deephermes24b_dataset_13590.jsonl

# Check job status
squeue -u $USER -j 13590

# View recent scoring activity
tail -20 environments/intern_bootcamp/logs/13590_env.log | grep -E "(Scored|current_step)"

# Check API activity
tail -10 environments/intern_bootcamp/logs/13590_api.log | grep -E "(batch|scored_data)"
```

## Data Format
Each line in the JSONL contains a scored data group with:
- Full conversation (system prompt + user + assistant)
- Score from InternBootcamp verification
- Task metadata including the specific bootcamp task name
- ~10KB per line average (long reasoning responses)

## Future Improvements

### Environment Enhancements
- Add support for step-based learning (vs episode-level)
- Implement curriculum learning progression
- Add multi-task training support

### Data Quality
- Improve format compliance through better prompting
- Add response validation before scoring
- Implement active learning for hard examples

### Scaling
- Support distributed generation across multiple nodes
- Add checkpointing for long-running jobs
- Implement streaming data processing for memory efficiency

## Important Notes for Claude Assistants

1. **Always use `uv run`** instead of direct `python` commands
2. **Process mode** is for data generation, **serve mode** is for training
3. **Score thresholds** should typically be > 0.0 for SFT datasets
4. **Task diversity** is ensured through RandomTask mode
5. **Format compliance** is the primary issue, not reasoning capability
6. **Rejection sampling** provides multiple attempts per problem for quality filtering

## Recent Work: InternBootcamp Task Labeling Improvements ✅ (In Progress)

**Problem**: The filter script was lumping most tasks into generic "reasoning_task" category instead of showing the actual InternBootcamp task names.

**Root Cause**: The `RandomTask` class was storing actual bootcamp names in `identity["_bootcamp_name"]` but this wasn't being preserved in the final `ScoredDataGroup`, so the filter script had to guess task types from message content.

**Solution Implemented**:
1. **Created `InternBootcampScoredDataGroup`** - extends `ScoredDataGroup` with `bootcamp_names: Optional[List[str]]` field
2. **Updated `score()` method in intern_bootcamp_env.py** - extracts `bootcamp_name = identity.get("_bootcamp_name", self.config.task_name)` and stores it in `scored_data["bootcamp_names"]`
3. **Updated filter_dataset.py** - modified `extract_task_name_from_messages()` to accept `bootcamp_name` parameter and use it directly instead of parsing content
4. **Updated SFT extraction** - reads `bootcamp_names` from scored groups and passes to task name extraction

**Files Modified**:
- `/environments/intern_bootcamp/intern_bootcamp_env.py`: Added `InternBootcampScoredDataGroup` class and updated scoring
- `/environments/intern_bootcamp/filter_dataset.py`: Enhanced to use actual bootcamp names from scored data

**Expected Result**: Instead of seeing 217 "reasoning_task" conversations, we should see the actual bootcamp task names like "Cconstanzesmachinebootcamp", "aquariumbootcamp", etc. - one per actual InternBootcamp task used.

**Status**: ✅ **COMPLETED AND TESTED SUCCESSFULLY**

**Test Results** (June 2, 2025):
1. ✅ **Generation Test**: Successfully ran 2-step generation completing 1 task (16 responses)
2. ✅ **Task Name Preservation**: `bootcamp_names` field correctly populated with actual task names
3. ✅ **Filtering Success**: Output shows **`Dqueuebootcamp: 7 conversations`** instead of generic "reasoning_task"
4. ✅ **Quality Metrics**: 43.8% retention rate (7/16 responses scored ≥ 0.0), average score 1.000

**Before vs After Comparison**:
- **Before**: `reasoning_task: 217 conversations` (generic catch-all)
- **After**: `Dqueuebootcamp: 7 conversations` (actual InternBootcamp task name)

**Ready for Production**: The improvements are working correctly and ready for the full 10,000-step generation job. Expected output will show diverse actual bootcamp task names like "Aquariumbootcamp", "Cconstanzesmachinebootcamp", etc.

## Recent Work: Parallel Processing for Process Command ✅ (June 4, 2025)

**Problem**: The `process` command was running sequentially, processing only 1 group at a time, making large-scale data generation slow (~1 group/second).

**Solution Implemented**: Added parallel processing capability to `base.py` that allows processing multiple groups concurrently, similar to how the `serve` command works.

**Key Changes**:
1. **Added `use_parallel_processing` config field** - Enables/disables parallel mode (default: True for process mode)
2. **Implemented `parallel_process_manager()` method** - Uses worker pool pattern to process multiple groups concurrently
3. **Created `handle_process_group()` method** - Handles individual group processing (adapted from `handle_env`)
4. **Added `_handle_group_completion()` callback** - Tracks completion and shows progress with ✓/✗ indicators
5. **Updated process mode defaults** - Sets `max_num_workers=8` and `use_parallel_processing=True` by default

**Files Created/Modified**:
- `/atroposlib/envs/base.py`: Added parallel processing implementation
- `/environments/intern_bootcamp/run_datagen_parallel_sglang.slurm`: SLURM script for parallel generation
- `/environments/intern_bootcamp/test_parallel_processing.py`: Local test script with dotenv support

**Performance Improvements**:
- **Sequential Mode**: 1 group at a time → ~1 group/second
- **Parallel Mode**: Up to 32 groups in parallel → ~10-30 groups/second (depending on API latency)
- **Expected speedup**: 10x-50x for network-bound tasks (API calls)
- **Time reduction**: 10,000 groups from ~3 hours to ~30-60 minutes

**Usage**:
```bash
# Parallel processing is enabled by default
uv run python -m environments.intern_bootcamp.intern_bootcamp_env process \
    --config config.yaml \
    --env.use_parallel_processing true \
    --env.max_num_workers 32

# Or use the new SLURM script
sbatch environments/intern_bootcamp/run_datagen_parallel_sglang.slurm
```

**Features**:
- **Backwards compatible**: Original sequential mode still available with `--env.use_parallel_processing false`
- **Progress tracking**: Shows real-time progress with completion statistics
- **Error handling**: Failed groups don't stop other workers
- **Resource management**: Reuses proven worker pool pattern from serve mode
- **Automatic throughput calculation**: Shows groups/second metrics

**Test Results** (June 4, 2025):
- Successfully tested with 3 groups, 4 responses each, 2 parallel workers
- Workers correctly spawned and managed
- Progress tracking working with ✓/✗ indicators
- JSONL output correctly formatted
- Parallel execution confirmed

**Status**: ✅ **IMPLEMENTED AND READY FOR PRODUCTION**

**Branch**: `parallel-processing-clean` (created off latest main)

## Recent Work: Parallel Processing for Process Command ✅ (June 4, 2025)

**Problem**: The `process` command was running sequentially, processing only 1 group at a time, making large-scale data generation slow (~1 group/second).

**Solution Implemented**: Added parallel processing capability to `base.py` that allows processing multiple groups concurrently, similar to how the `serve` command works.

**Key Changes**:
1. **Added `use_parallel_processing` config field** - Enables/disables parallel mode (default: True for process mode)
2. **Implemented `parallel_process_manager()` method** - Uses worker pool pattern to process multiple groups concurrently
3. **Created `handle_process_group()` method** - Handles individual group processing (adapted from `handle_env`)
4. **Added `_handle_group_completion()` callback** - Tracks completion and shows progress with ✓/✗ indicators
5. **Updated process mode defaults** - Sets `max_num_workers=8` and `use_parallel_processing=True` by default

**Files Modified**:
- `/atroposlib/envs/base.py`: Added parallel processing implementation
- Created `/environments/intern_bootcamp/run_datagen_parallel_sglang.slurm`: SLURM script for parallel generation
- Created `/environments/intern_bootcamp/test_parallel_processing.py`: Local test script with dotenv support

**Performance Improvements**:
- **Sequential Mode**: 1 group at a time → ~1 group/second
- **Parallel Mode**: Up to 32 groups in parallel → ~10-30 groups/second (depending on API latency)
- **Expected speedup**: 10x-50x for network-bound tasks (API calls)
- **Time reduction**: 10,000 groups from ~3 hours to ~30-60 minutes

**Usage**:
```bash
# Parallel processing is enabled by default
uv run python -m environments.intern_bootcamp.intern_bootcamp_env process \
    --config config.yaml \
    --env.use_parallel_processing true \
    --env.max_num_workers 32

# Or use the new SLURM script
sbatch environments/intern_bootcamp/run_datagen_parallel_sglang.slurm
```

**Features**:
- **Backwards compatible**: Original sequential mode still available with `--env.use_parallel_processing false`
- **Progress tracking**: Shows real-time progress with completion statistics
- **Error handling**: Failed groups don't stop other workers
- **Resource management**: Reuses proven worker pool pattern from serve mode
- **Automatic throughput calculation**: Shows groups/second metrics

**Status**: ✅ **IMPLEMENTED AND READY FOR TESTING**

## Recent Work: TextWorld Environment with SGLang and VR-CLI (June 10, 2025)

### Problem Solved
Needed to set up TextWorld environment to work with SGLang server for access to logprobs, remove unnecessary smolagents dependency, and implement environment state copying for VR-CLI evaluation.

### Solution Implemented

#### 1. SGLang Integration
- **Updated**: `textworld_local_server.py` to use DeepHermes-3-Mistral-24B via local SGLang server
- SGLang server runs with `--tp 8` for tensor parallelism across 8 GPUs
- Provides access to logprobs which OpenAI API doesn't expose
- Server endpoint: `http://localhost:30000/v1`

#### 2. Removed smolagents Dependency
- **Issue**: AtroposAgent inherited from smolagents.Model causing property conflicts
- **Solution**: Simplified AtroposAgent to not inherit from Model
- Removed all smolagents imports and converted helper functions
- Agent now directly manages token counting and API calls

#### 3. TextWorld State Copying via Replay
- **Challenge**: TextWorld doesn't support state saving/loading or deepcopy
- **Solution**: Implemented replay-based state copying:
  - Extract canonical action history from agent's game log
  - Create new environment instance with same game file
  - Replay all actions to reach current state
  - Used for VR-CLI evaluation of alternative actions

#### 4. Key Files Modified
- `/environments/game_environments/textworld/textworld_local_server.py`: SGLang configuration
- `/environments/game_environments/textworld/agents/atropos_agent.py`: Removed smolagents
- `/environments/game_environments/textworld/textworld_env.py`: Added replay-based copying

### TextWorld VR-CLI Implementation

VR-CLI (Value-Ranked Counterfactual Learning from Implicit feedback) evaluates multiple action alternatives by:
1. Agent generates N alternative actions with outcome predictions
2. For each alternative, create environment copy via replay
3. Execute action and compare predicted vs actual outcome
4. Score based on prediction accuracy (lower perplexity = better)
5. Select best-scoring action to execute

### Current Status
- ✅ SGLang server integration working with logprobs
- ✅ smolagents dependency removed successfully
- ✅ Environment replay for state copying implemented
- ✅ TextWorld games running with VR-CLI evaluation
- ✅ Agent successfully playing through game objectives

### TODO for TextWorld Environment

1. **Create SLURM scripts** for distributed training:
   - Dataset generation script for rejection sampling
   - Multi-node training script with SGLang inference
   - Configure appropriate node allocation

2. **Implement additional reward functions**:
   - Game completion reward
   - Step efficiency reward
   - VR-CLI prediction accuracy reward

3. **Test with different TextWorld challenges**:
   - tw-simple (current)
   - tw-cooking
   - tw-treasure_hunter
   - Custom challenges

4. **Performance optimization**:
   - Cache environment copies for similar states
   - Batch VR-CLI evaluations
   - Optimize replay mechanism

5. **Data generation**:
   - Generate diverse game instances
   - Create SFT datasets from successful trajectories
   - Filter by VR-CLI scores for quality