# CLAUDE.md - Atropos Project Guide

This document contains project-specific instructions and learnings for Claude assistants working on the Atropos RL training framework.

## IMPORTANT: Always Use UV Run
- **NEVER use `python` directly** - always use `uv run python`
- This project uses UV for dependency management
- Example: `uv run python test_improved_prompt.py`

## Current Work: Inline Memory Generation System (June 11, 2025)

### Background
The TextWorld environment currently uses a separate LLM call to generate memory summaries after each turn. We're implementing an inline memory generation system where the model generates memories as XML blocks during its response, which will be more efficient and allow the model to learn memory generation through RL.

### Current Memory System Analysis
1. **Memory Generation Process**:
   - After each turn, `record_selected_action_and_learn_from_turn()` is called
   - This method calls `summarize_turn_for_memory()` which makes a separate LLM API call
   - The LLM generates a concise memory summary based on the turn's events
   - The summary is stored in a FAISS index for similarity-based retrieval

2. **Memory Storage**:
   - Uses `AtroposMemoryManager` with sentence-transformers for embeddings
   - Stores text summaries in a FAISS index (vector database)
   - Retrieves top-k relevant memories based on cosine similarity
   - Retrieved memories are prepended to observations in `generate_action()`

3. **Thinking Block Summarization**:
   - Separate process that happens in `postprocess_histories()`
   - Uses `summarize_thinking_block()` to compress long reasoning chains
   - Preserves key reasoning steps while reducing token count
   - **NOTE**: We'll initially try WITHOUT thinking block summarization to see if inline memories are sufficient
   - Only re-enable if model starts refusing to produce thinking blocks due to overconditioning on previous messages

### Proposed Inline Memory Generation System

#### Implementation Plan
1. **Update System Prompt**:
   - Add instructions for generating `<memory>` blocks after `<think>` and before `<tool_call>`
   - Memory blocks should:
     - Build upon and reference previous memories (shown as "Relevant Memories" in observation)
     - Track outcomes of previous actions and update understanding
     - Maintain continuity of goals and strategic plans
     - Record new discoveries while preserving important context

2. **Memory Generation Guidelines** (to include in prompt):
   - Review previous memories shown in the observation
   - Note the outcome of your last action (did it match expectations?)
   - Update any goals or plans based on new information
   - Preserve important state information (inventory, location, objectives)
   - Be concise but comprehensive (aim for 1-3 sentences)

3. **Modify AtroposAgent**:
   - Add memory extraction logic in `generate_action()` method
   - Parse XML to extract content from `<memory>` tags
   - Store extracted memories directly in the FAISS index
   - Remove the need for `summarize_turn_for_memory()` calls

4. **Benefits**:
   - **Efficiency**: Eliminates extra LLM API call per turn
   - **Context**: Memories generated with full context awareness
   - **RL Integration**: Memory generation becomes part of the RL training
   - **Continuity**: Model learns to maintain coherent memory chains
   - **Strategic Planning**: Memories can track multi-step plans

5. **Challenges & Solutions**:
   - **Memory Coherence**: Model must learn to build on previous memories
   - **Information Decay**: Important early memories might get lost
   - **Token Usage**: Monitor if inline memories increase response length significantly
   - **Thinking Block Overconditioning**: If model stops producing thinking blocks because previous messages lack them, we may need to:
     - Re-enable thinking block summarization for very long blocks (>2000 tokens)
     - Or adjust training to ensure model still produces thinking blocks despite their absence in history

### Implementation Tasks
- [x] Update system prompt in `textworld_env.py` to include memory block instructions
- [x] Add memory extraction logic to `AtroposAgent.record_selected_action_and_learn_from_turn()`
- [x] Create XML parsing utilities for memory blocks in `utils/memory_parser.py`
- [x] Update memory storage to use inline memories (with fallback to LLM summarization)
- [ ] Test with 8B model to ensure format compliance
- [ ] Compare memory quality between inline and separate generation
- [ ] Monitor token usage and adjust if needed

### Next Steps for Testing (IMPORTANT - RUN ON GPU NODE)
1. **Allocate a GPU node**:
   ```bash
   srun --gpus=8 --pty bash
   ```

2. **Launch SGLang server with 8B model**:
   ```bash
   cd /home/maxpaperclips/atropos
   ./launch_sglang_nohup.sh --model "NousResearch/DeepHermes-3-Llama-3-8B-Preview" --tp 4
   ```

3. **Wait for server to be ready**:
   ```bash
   # Check server status
   curl http://localhost:30000/health
   ```

4. **Run the inline memory test**:
   ```bash
   cd /home/maxpaperclips/atropos
   uv run python test_inline_memory.py
   ```

5. **Run full TextWorld environment test**:
   ```bash
   cd /home/maxpaperclips/atropos/environments/game_environments/textworld
   uv run python textworld_local_server.py
   ```

### What We've Implemented
1. **Memory Parser** (`utils/memory_parser.py`):
   - `extract_memory_block()`: Extracts content from `<memory>` tags
   - `validate_memory_content()`: Ensures memory meets quality requirements
   - `extract_thinking_and_memory()`: Extracts both thinking and memory blocks

2. **Updated AtroposAgent**:
   - Modified `record_selected_action_and_learn_from_turn()` to first try extracting inline memory
   - Falls back to LLM summarization if no valid inline memory found
   - Logs whether inline or LLM-generated memory was used

3. **Updated System Prompt**:
   - Added memory generation instructions between thinking and tool call
   - Included example responses showing memory blocks
   - Emphasized building upon previous memories for continuity

4. **Test Script** (`test_inline_memory.py`):
   - Tests memory generation across multiple turns
   - Shows how memories should build upon each other
   - Validates memory extraction and quality

### Example Expected Output
```xml
<think>
Looking at my previous memories, I was exploring the kitchen to find cooking ingredients.
I successfully opened the fridge and found eggs, milk, and flour. My goal is still to
cook something. Now I need to take these ingredients and find a recipe or mixing bowl.
The previous action of opening the fridge worked as expected.
</think>
<memory>
Found eggs, milk, and flour in kitchen fridge. Still need mixing bowl or recipe to cook.
Previous exploration of kitchen successful - have stove and ingredients located.
</memory>
<tool_call>
{"name": "execute_command", "arguments": {"command": "take eggs", "expected_outcome": "I take the eggs from the fridge and add them to my inventory"}}
</tool_call>
```

### Memory Continuity Example
**Turn 1 Memory**: "Kitchen has stove and fridge. Main objective is cooking. Need to find ingredients."

**Turn 2 Memory**: "Found eggs, milk, flour in fridge. Still need mixing bowl or recipe. Kitchen layout understood."

**Turn 3 Memory**: "Have eggs in inventory. Milk and flour still in fridge. Next: find mixing bowl, then gather remaining ingredients."

This creates a coherent narrative that the model can follow and build upon.

### Inline Memory System Testing Results (June 15, 2025)

#### Test Script Results
Successfully tested inline memory generation with `test_inline_memory.py`:
- **Test Case 1**: ✅ Model generated proper `<memory>` block with validation passing
- **Test Case 2**: ✅ Model generated `<memory>` block that built upon previous memories
- **Test Case 3**: ❌ Model improvised with `<remember>` tags instead of `<memory>` tags

#### Key Findings
1. **Inline memory generation works** when model uses correct XML tags
2. **8B model can follow format** but may occasionally improvise
3. **Max tokens must be 16384** to allow room for long thinking chains + memory blocks
4. **Memory extraction and validation** working correctly in AtroposAgent

#### Full Episode Testing
Ran complete TextWorld episode with `textworld_local_server.py`:
- ✅ **Memory generation working** - Model successfully generated memory blocks
- ✅ **Memory retrieval attempted** - System checked for relevant memories (though none found on turn 1 as expected)
- ✅ **VR-CLI scoring functioning** - Action predictions being evaluated
- ✅ **Registry system working** - Generated "quest_puzzle_hard" game
- ✅ **Complex objective following** - Agent navigating multi-step puzzle correctly

### Data Generation Progress (June 16, 2025)

#### Job Fixes Applied
1. **Argument error fixed** - Removed incorrect atropos_agent_config parameters from SLURM script
2. **Asyncio error fixed** - Changed from `asyncio.run(main_cli())` to `TextWorldEnv.cli()` pattern
3. **SGLang logprobs error fixed** - Changed `logprobs=0` to `logprobs=1` for VR-CLI scoring compatibility

#### Data Generation Status (Job 14120)
- **Running successfully** with 4 parallel workers
- **Inline memory working** - Model generating `<memory>` blocks correctly
- **VR-CLI scoring operational** - Getting perplexity-based scores for predictions
- **Episode tracking working** - Each ScoredDataGroup has episode_id and turn_number
- **Game variety confirmed** - Using 70% generated games, 30% pre-built challenges

#### Key Metrics Observed
- **VR-CLI scores**: 75.8% are zero (expected for poor predictions)
- **Non-zero VR-CLI range**: 0.00015 to 0.91 (good diversity)
- **Average episode length**: 4 turns
- **Memory generation rate**: Successfully generating inline memories

#### Analysis Tools Created
- **analyze_episodes.py**: Script to reconstruct episodes and analyze scoring distributions
- Allows episode-by-episode analysis and VR-CLI score investigation

### VR-CLI Implementation Updates (Completed: December 2024) ✅

#### Summary of Implementation Session
We successfully updated the VR-CLI implementation to match the paper's specifications and added token length penalties:

1. **VR-CLI Formula Fixed** ✅
   - Updated from continuous scoring to percentage improvement: `[1 - PPL(y|x,a)/PPL(y|x)] × 100`
   - Implemented discrete reward levels: 0.0 (<0.05), 0.5 (0.05-1), 0.9 (1-2), 1.0 (≥2)
   - Verified per-token perplexity calculation is correct

2. **Reward Weighting Adjusted** ✅
   - Changed VR-CLI weight from 0.7 to 0.3
   - Disabled LaTRo (weight = 0.0) for now
   - Environment reward now gets 0.7 weight (1.0 - 0.3 - 0.0)
   - Properly handles three-way reward split for future LaTRo integration

3. **Credit Assignment Enhanced** ✅
   - Verified discount factor (γ = 0.99) is properly applied
   - Implemented credit assignment for unselected alternatives with same action
   - Future returns now propagated to all alternatives that would have taken the same action

4. **Analysis Tools Updated** ✅
   - analyze_episodes.py now shows discrete VR-CLI reward distribution
   - Per-turn analysis displays reward level counts
   - Overall statistics show percentage breakdown by reward level

5. **Token Length Penalty Added** ✅
   - Implemented configurable token length penalty/bonus system
   - Baseline: 500 tokens (neutral point)
   - Up to 10% reward adjustment based on response length
   - Shorter responses get bonus when outcomes are good
   - Longer responses get extra penalty when outcomes are bad
   - Helps encourage concise, efficient reasoning

### Current Implementation State (Ready for GPU Testing)

## Recent Debugging Session: TextWorld Data Generation Hanging (June 16, 2025)

### Problem
TextWorld data generation jobs are hanging after initial setup. Jobs 14118, 14119, 14174, 14175, 14177, and 14178 all failed to generate data, while job 14120 successfully generated 1.6GB.

### What We Tried
1. **Fixed GPU allocation**: Changed from 4 GPUs (tp=4) to 8 GPUs (tp=8) in the SLURM script
2. **Added debug logging**: 
   - Added logging to `get_next_item()` and `collect_trajectories()` in textworld_env.py
   - Added logging to `_next_step()` to trace episode execution
   - Added logging to `parallel_process_manager()` in base.py
   - Set `debug_mode: true` in config (though it showed as false in output)
3. **Added wandb_name**: Set `wandb_name: "textworld-datagen"` in config
4. **Observations**:
   - SGLang server starts successfully
   - Config and server_configs are printed
   - Process hangs after printing server configs
   - No debug output appears after config print
   - SGLang logs show only one POST request to /generate endpoint
   - WandB initializes despite `use_wandb: false` in config

### Unknowns
- Is the hang in the TextWorld environment initialization or SGLang?
- Why does job 14120 work but others fail?
- Is the parallel processing manager even starting?
- Could SGLang be hanging on the first request?

### Problem Resolved ✅ (June 16, 2025)

The issue was that `do_send_to_api` was not a field in `BaseEnvConfig`, causing the process command to run `env_manager` which waits for a training backend instead of running standalone data generation.

#### Solution Implemented
1. **Added `do_send_to_api` field to `BaseEnvConfig`** with default value `True` (for serve mode)
2. **Set `do_send_to_api=False` in process mode defaults** to ensure data generation runs standalone
3. **Verified fix works** - Job 14186 running successfully without hanging

#### Key Code Changes
- Added to `BaseEnvConfig` in `/home/maxpaperclips/atropos/atroposlib/envs/base.py`:
  ```python
  do_send_to_api: bool = Field(
      default=True,
      description="Whether to send data to the API server (True for serve mode, False for process mode)",
  )
  ```
- Updated `PROCESS_MODE_ENV_DEFAULT_CONFIG` to include `do_send_to_api=False`

This ensures that:
- `serve` command: Sends data to training API (default `True`)
- `process` command: Saves data to file without waiting for backend (overridden to `False`)

### Current Implementation State (Ready for GPU Testing)

#### Key Configuration Parameters
```yaml
# Reward weights (in textworld_env.py and config_process.yaml)
vrcli_weight: 0.3          # VR-CLI contribution
latro_weight: 0.0          # LaTRo disabled for now
# Environment weight: 0.7   # (1.0 - 0.3 - 0.0)

# Token length penalty
token_length_penalty_enabled: true
token_length_penalty_weight: 0.1      # Max 10% adjustment
token_length_baseline: 500            # Neutral point (tokens)
token_length_penalty_scale: 0.0002    # Penalty per token over baseline

# Credit assignment
vrcli_discount_factor: 0.99          # For multi-step episodes
```

#### Expected Behavior During Testing
1. **VR-CLI Rewards**: Should see mostly 0.0 and 0.5 scores, with occasional 0.9 and rare 1.0
2. **Token Length**: Responses around 500 tokens get no adjustment; shorter get bonus, longer get penalty
3. **Credit Assignment**: Final episode outcomes propagate back through all steps
4. **Alternative Actions**: Unselected alternatives that chose the same action also get future returns

#### Testing Commands
```bash
# On GPU node with SGLang server running
cd /home/maxpaperclips/atropos/environments/game_environments/textworld
uv run python textworld_local_server.py

# To analyze generated data
uv run python analyze_episodes.py data/textworld_deephermes8b_dataset_*.jsonl --stats-only
```

### Remaining Tasks

#### Testing & Validation (IMMEDIATE - ON GPU NODE)
**Tasks:**
- [ ] Test updated VR-CLI implementation with different game types
- [ ] Verify discrete reward distribution matches expectations
- [ ] Check token length penalty is working as intended
- [ ] Compare with any existing baseline data if available

#### Post-Testing Adjustments
**Based on test results, may need to:**
- [ ] Adjust VR-CLI percentage thresholds if too many 0.0 rewards
- [ ] Fine-tune token length baseline (currently 500)
- [ ] Modify penalty scale if too harsh/lenient
- [ ] Consider game-specific adjustments

#### Memory System Analysis (LOWER PRIORITY)
**Tasks:**
- [ ] Add memory retrieval statistics to analyze_episodes.py
- [ ] Track memory similarity scores when retrieving
- [ ] Analyze memory coherence across turns
- [ ] Test different memory_top_k values (currently 3)

### Future Work & Ablations

#### LaTRo Implementation (Postponed for Ablation Study)
**Implement cross-entropy based rewards as alternative/complement to VR-CLI**

**Future Tasks:**
- [ ] Implement LaTRo reward calculation: `log π(correct_action | state + reasoning)`
- [ ] Add KL divergence penalty term
- [ ] Create hybrid reward: VR-CLI + LaTRo + environment
- [ ] Compare performance with VR-CLI alone
- [ ] Test on different game types (puzzle vs navigation vs quest)

#### Additional Analysis & Optimization
**Lower priority tasks for system improvement**

1. **Game Diversity Analysis**
   - [ ] Add game type tracking to metadata
   - [ ] Analyze distribution of generated vs pre-built games
   - [ ] Track difficulty levels used
   - [ ] Check if certain game types lead to better learning

2. **Sparse Reward Analysis**
   - [ ] Track episodes with zero environment rewards throughout
   - [ ] Compare VR-CLI guidance in sparse vs dense reward episodes
   - [ ] Analyze if VR-CLI helps discover winning strategies

3. **Format Compliance & Error Analysis**
   - [ ] Track tool call parsing failures
   - [ ] Analyze when model fails to generate memory blocks
   - [ ] Monitor token usage vs max_tokens limit

4. **Performance Optimization**
   - [ ] Monitor SGLang server memory usage
   - [ ] Optimize batch sizes for throughput
   - [ ] Profile VR-CLI perplexity calculations

5. **Evaluation Metrics**
   - [ ] Implement win rate tracking over time
   - [ ] Create learning efficiency metrics
   - [ ] Measure sample efficiency (wins per 1000 episodes)

6. **Configuration Experiments**
   - [ ] Test different temperature values for action generation
   - [ ] Vary number of alternatives (currently 16)
   - [ ] Experiment with thinking summarization thresholds

### Previous Work: TextWorld Environment Testing (Completed)
We tested the TextWorld environment with different models to compare format compliance:
- **DeepHermes-3-Mistral-24B-Preview**: Had formatting issues with tool calls
- **DeepHermes-3-Llama-3-8B-Preview**: Successfully handles the XML format correctly

### Key Files and Locations
- **SGLang launcher scripts**:
  - `/home/maxpaperclips/atropos/launch_sglang.sh` - Original blocking version
  - `/home/maxpaperclips/atropos/launch_sglang_nohup.sh` - Non-blocking version with nohup
- **TextWorld test files**:
  - `/home/maxpaperclips/atropos/test_improved_prompt.py` - Tests TextWorld environment prompting
  - `/home/maxpaperclips/atropos/test_direct_tool_call.py` - Direct test of model's tool call format
- **TextWorld environment**:
  - `/home/maxpaperclips/atropos/environments/game_environments/textworld/`
  - `textworld_local_server.py` - Local server for testing with SGLang
  - `textworld_env.py` - Main environment implementation
  - `textworld_registry.py` - Registry system for game selection

### Running Tests
```bash
# Launch SGLang server (non-blocking)
./launch_sglang_nohup.sh --model "NousResearch/DeepHermes-3-Llama-3-8B-Preview" --tp 4

# Check server status
curl http://localhost:30000/health

# Run TextWorld tests
uv run python test_improved_prompt.py
uv run python test_direct_tool_call.py

# Run the full TextWorld local server test
cd environments/game_environments/textworld
uv run python -m environments.game_environments.textworld.textworld_local_server
```

### Issues Found and Solutions (June 11, 2025)

#### Problem: SGLang Tool Call Parser
- SGLang automatically tries to parse tool calls from chat completions, causing "Tool Call Parser Not Given!" 400 errors
- This cannot be disabled in SGLang configuration
- Both 24B and 8B models were affected when using chat completions endpoint

#### Solution Implemented
- Modified `AtroposAgent` in `atropos_agent.py` to use **completions endpoint** instead of chat completions
- Changed both `generate()` and `generate_action()` methods to:
  1. Convert messages to prompt using `tokenizer.apply_chat_template()`
  2. Use `server_client.completion()` instead of `server_client.chat_completion()`
  3. Extract text from `choice.text` instead of `choice.message.content`

#### Results
- ✅ **8B model** (DeepHermes-3-Llama-3-8B-Preview) successfully generates proper `<think>` and `<tool_call>` XML blocks
- Model follows the format correctly with very long reasoning chains (as expected for a reasoning model)
- No more 400 errors from SGLang

#### 24B Model Test Results (June 11, 2025)
- ❌ **24B model** (DeepHermes-3-Mistral-24B-Preview) does NOT follow the XML format correctly
- Even with the completions endpoint fix, the 24B model:
  - Does not generate `<think>` tags
  - Does not generate `<tool_call>` tags
  - Produces repetitive text and appears to include internal tokens like `<|start_header_id|>`
  - Generates "Thought Process:" instead of using XML tags

#### Conclusion
- **Use the 8B model** for TextWorld environment as it correctly follows the XML tool call format
- The 24B model appears to have different training/prompting requirements and doesn't work with this format

## Recent Work: TextWorld Registry System Complete ✅ (June 11, 2025)

### Problem Solved
Fixed all issues with the TextWorld registry system for managing 1000+ game variations through random sampling, mixing pre-built challenges and procedurally generated games.

### Key Fixes Applied
1. **Object Placement Pattern**: Fixed `maker.move()` errors by using `room.add(obj)` for new objects
2. **Quest Parameters**: Adjusted hard difficulty settings to make generation feasible
3. **Door Creation**: Fixed navigation generator by creating doors on paths, not as standalone objects
4. **Game Variance**: Added random treasure placement and dead-end branches to prevent predictable patterns
5. **Cache Conflicts**: Implemented automatic seed regeneration with `compile_game_with_retry()`
6. **Challenge Parameters**: Fixed tw-cooking to use correct parameter types (int, bool) and `recipe_seed`
7. **Parameter Randomization**: All challenges now have randomized parameters for variety

### Current Status
- ✅ All 4 generators working (Quest, Puzzle, Navigation, Mixed)
- ✅ All 4 challenges working (tw-simple, tw-cooking, tw-coin_collector, tw-treasure_hunter)
- ✅ Registry system operational with 70% generated / 30% pre-built ratio
- ✅ **100% success rate** on random generation (with automatic retry on conflicts)
- ✅ Sparse rewards preferred for better RL training
- ✅ Ready for large-scale RL training without overfitting

See full details in the TextWorld section below.

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

## Recent Work: TextWorld Registry System Complete & Tested ✅ (June 11, 2025)

### Problem Solved
User needed a registry system for TextWorld similar to bootcamp_registry.py to manage 1000+ game variations through random sampling, mixing pre-built challenges and procedurally generated games.

### Latest Updates (June 11, 2025)
- ✅ **Launched SGLang server** with DeepHermes-3-Mistral-24B-Preview on port 30000
- ✅ **Updated textworld_local_server.py** to use new registry system with 70% generated/30% pre-built ratio
- ✅ **Verified cleanup mechanism** - Registry properly removes both .z8 and .ni files after use
- ✅ **Ran full integration test** - VR-CLI scoring working with registry-selected games
- ⚠️ **Issue Found**: LLM struggling with output format (malformed tool calls) - needs investigation

### Solution Implemented

#### 1. Created Comprehensive Registry System
- **File**: `/environments/game_environments/textworld/textworld_registry.py`
- Three main classes:
  - `TextWorldChallengeRegistry`: Manages pre-built challenges (tw-simple, tw-cooking, tw-coin_collector, tw-treasure_hunter)
  - `TextWorldGenerator`: Base class for custom game generation with difficulty settings
  - `TextWorldEnvironmentRegistry`: Main registry combining both systems
- Features:
  - Random selection between generated (default 70%) and pre-built (30%) games
  - Difficulty levels: easy, medium, hard, expert, random
  - Automatic cleanup of generated files
  - LRU cache for frequently used configurations

#### 2. Updated Generation Utilities
- **File**: `/environments/game_environments/textworld/generation_utils.py`
- Added `cleanup_on_error` parameter for better error handling
- Ensures partial files are cleaned up on generation failure

#### 3. Integrated Registry with TextWorldEnv
- **File**: `/environments/game_environments/textworld/textworld_env.py`
- Added registry configuration fields to TextWorldEnvConfig:
  - `use_registry`: Enable/disable registry usage (default: True)
  - `registry_mode`: Selection mode - random, generated, challenge
  - `registry_generation_ratio`: Ratio of generated vs pre-built games (0.0-1.0)
  - `registry_difficulty`: Difficulty level selection
  - `registry_game_type`: Specific game type selection
- Modified `_setup_new_episode` to use registry when enabled
- Updated cleanup methods to handle registry cleanup

#### 4. Test Results
- ✅ Successfully tested all registry functionality
- ✅ Pre-built challenges working correctly
- ✅ Generated games with difficulty scaling functional
- ✅ Proper file cleanup confirmed
- ✅ Integration with TextWorldEnv verified

### Key Features
- **Dynamic game selection**: Mixes pre-built and generated games based on configured ratio
- **Difficulty support**: Easy, medium, hard, expert, or random selection
- **Game types**: Quest, puzzle, navigation, mixed (for generated games)
- **Automatic cleanup**: Tracks and cleans up all generated game files
- **Seamless integration**: Works transparently with existing TextWorldEnv

### Usage Example
```python
# Create environment with registry enabled
config = TextWorldEnvConfig(
    use_registry=True,
    registry_mode="random",
    registry_generation_ratio=0.7,  # 70% generated, 30% pre-built
    registry_difficulty="random",    # Randomize difficulty
    max_steps=30
)
```

### Background: SGLang Integration (Previously Completed)
- Updated textworld_local_server.py to use SGLang instead of OpenAI for logprobs access
- Removed smolagents dependency from AtroposAgent (simplified implementation)
- Implemented environment replay for state copying (required for VR-CLI evaluation)
- Successfully tested with DeepHermes-3-Mistral-24B-Preview model

### Specialized Game Generators ✅ (June 10, 2025)

#### Completed Today
Created four specialized game generators for TextWorld:

1. **quest_generator.py** ✅
   - Quest types: fetch, delivery, rescue, exploration, puzzle
   - Difficulty-based room/object scaling
   - Custom quest generation with commands
   - Uses GameMaker API for programmatic game creation

2. **puzzle_generator.py** ✅
   - Puzzle types: door_sequence, combination_lock, weight_puzzle, light_puzzle, container_puzzle
   - Specialized puzzle mechanics (keys, combinations, weights)
   - Progressive difficulty with more complex puzzles

3. **navigation_generator.py** ✅
   - Navigation types: maze, labyrinth, exploration, dungeon
   - Layout algorithms: grid maze, branching tree, hub-and-spoke
   - Landmarks and navigation aids based on difficulty
   - Dead ends and complex pathfinding challenges

4. **mixed_generator.py** ✅
   - Mixed types: quest_puzzle, navigation_puzzle, exploration_quest, dungeon_crawler, adventure
   - Combines elements from other generators
   - Complex multi-stage games with various mechanics

#### Files Created/Modified
- `/environments/game_environments/textworld/generators/quest_generator.py`
- `/environments/game_environments/textworld/generators/puzzle_generator.py`
- `/environments/game_environments/textworld/generators/navigation_generator.py`
- `/environments/game_environments/textworld/generators/mixed_generator.py`
- `/environments/game_environments/textworld/generators/__init__.py`
- Updated `/environments/game_environments/textworld/textworld_registry.py` to use new generators

### Issues Fixed ✅ (June 11, 2025)

All major issues have been resolved:

1. **GameMaker API Issues** ✅:
   - Fixed `'NoneType' object has no attribute 'remove'` error
   - Discovered correct pattern: use `room.add(obj)` for new objects, `maker.move(obj, room)` for existing objects
   - Doors must be created on paths using `maker.new_door(path)`, not as standalone objects

2. **Quest Generation Parameters** ✅:
   - Adjusted hard difficulty settings: reduced `quest_breadth` from 3 to 2
   - Increased `quest_depth` to 10 to compensate
   - All difficulty levels now generate successfully

3. **Challenge Generation** ✅:
   - Added missing `recipe_seed` parameter to tw-cooking challenge
   - Fixed all generator object placement patterns
   - Registry system successfully mixing challenges and generated games

### Key Learnings

#### TextWorld Object Creation Pattern
```python
# For newly created objects - use room.add()
obj = maker.new(type='o', name='item')
room.add(obj)  # NOT maker.move()

# For existing objects (already placed) - use maker.move()
maker.move(obj, new_room)
```

#### Door Creation
```python
# Doors must be created on paths between rooms
path = maker.connect(room1.exits["east"], room2.exits["west"])
door = maker.new_door(path, name="wooden door")
# NOT: door = maker.new(type="d", name="door")
```

#### Game Variance Implementation
- Random treasure placement (not always in last room)
- Dead-end branches in harder difficulties
- Decoy objects to prevent predictable patterns
- Multiple valid endpoints to avoid "last room = treasure" learning

### Completed Tasks ✅

1. **Bug Fixes**:
   - [x] Fixed room connection errors in all generators
   - [x] Adjusted quest generation parameters for feasibility
   - [x] Added proper error handling with traceback for debugging
   - [x] Tested each generator individually

2. **All Generators Working**:
   - [x] Quest Generator (fetch, delivery, rescue, exploration, puzzle)
   - [x] Puzzle Generator (door_sequence, combination_lock, weight_puzzle, etc.)
   - [x] Navigation Generator (maze, labyrinth, exploration, dungeon)
   - [x] Mixed Generator (quest_puzzle, navigation_puzzle, dungeon_crawler, etc.)

3. **Registry System Complete**:
   - [x] 70% generated / 30% pre-built game ratio
   - [x] Automatic cleanup of generated files
   - [x] LRU cache for frequently used configurations
   - [x] Support for difficulty and game type selection

### Current Status

The TextWorld registry system is fully operational and ready for large-scale RL training:
- **Success rate**: ~70% (some edge cases with specific seeds/parameters)
- **Game variety**: 1000+ possible variations through mixing
- **Performance**: Fast generation with caching
- **Integration**: Ready to use with TextWorldEnv

### Architecture Notes

The generator system now provides:
- **Modular design**: Each generator focuses on specific game mechanics
- **Difficulty scaling**: All generators support easy/medium/hard/expert
- **Flexible configuration**: Can specify game types and sub-types
- **Programmatic generation**: Uses TextWorld's GameMaker API correctly
- **Registry integration**: Seamlessly works with the existing environment

### Usage Example
```python
from environments.game_environments.textworld.textworld_registry import create_textworld_registry

# Create registry with 70% generated games
registry = create_textworld_registry(generation_ratio=0.7, seed=42)

# Get a random game
game_file, config = registry.get_environment(mode="random")

# Get specific type
game_file, config = registry.get_environment(
    mode="generated",
    difficulty="medium", 
    game_type="puzzle"
)
```

### Technical Details

#### Cache Conflict Resolution
Implemented `compile_game_with_retry()` in `generation_utils.py` that:
- Detects "same id have different structures" errors
- Automatically generates a new seed and retries (up to 5 attempts)
- Cleans up conflicting .z8 and .ni files

#### Challenge Parameter Fixes
- **tw-cooking**: Required integer parameters (not strings), `recipe_seed` (not `recipe-seed`), and `split='train'`
- **Parameter randomization**: Each challenge now randomly selects from predefined parameter ranges
- **Constraint enforcement**: tw-cooking ensures `take <= recipe` automatically

#### Registry Configuration
```python
# 70% generated games, 30% pre-built challenges
registry = create_textworld_registry(generation_ratio=0.7, seed=42)

# Challenges prefer sparse rewards for better RL training
"tw-simple": {
    "rewards": ["sparse", "balanced", "dense"],  # sparse is default
    ...
}
```

The system is production-ready for large-scale RL training without overfitting!