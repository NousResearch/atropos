# InternBootcamp Data Generation on BS200 Cluster

This setup runs data generation for the InternBootcamp environment using the new SFT data collection service with `serve` mode.

## Quick Start

```bash
sbatch intern_bootcamp_datagen.slurm
```

## Configuration

- **Model**: deepseek-ai/DeepSeek-R1 (configured in SLURM script)
- **Data Output**: `~/atropos/data/` (Note: may save to literal `${HOME}/atropos/data/` due to path expansion bug)
- **Logs**: `logs/$SLURM_JOB_ID/`
- **Config**: `environments/intern_bootcamp/config_serve.yaml`
- **Target**: 50,000 problems (800,000 total responses with group_size=16)

## Architecture

1. **SGLang** runs on all 8 B200 GPUs with:
   - Tensor Parallelism (TP): 8 GPUs
   - Port: 9000
   - Using xgrammar backend and triton attention

2. **Atropos API Server** handles trajectory collection
   - Requires fake trainer to fetch batches

3. **Fake Trainer** enables data collection without RL training
   - Required because API server blocks until trainer registers

4. **InternBootcamp Environment** generates problems and collects responses
   - Generates 16 responses per problem for rejection sampling
   - Saves data every 100 problems

## SLURM Configuration

The script includes `--requeue` flag for automatic restart on error. This helps handle unexpected shutdowns (e.g., API server crashes).

## Monitoring

Check logs during execution:
```bash
# Overall job output
tail -f logs/$SLURM_JOB_ID.out

# Individual services
tail -f logs/$SLURM_JOB_ID/api.log
tail -f logs/$SLURM_JOB_ID/sglang.log
tail -f logs/$SLURM_JOB_ID/intern_bootcamp.log
tail -f logs/$SLURM_JOB_ID/fake_trainer.log
```

## Output Files

- **Main data**: `intern_bootcamp_serve_data_N.jsonl` (increments if exists)
- **Rollouts**: `intern_bootcamp_rollouts_UUID_NNNN.jsonl`
- **Location**: Check both `~/atropos/data/` and literal `'${HOME}'/atropos/data/`

## Known Issues

1. **Path Expansion Bug**: Environment may save to literal `${HOME}` directory
2. **Early Termination**: API server may shut down before completion (restart helps)
3. **Progress Tracking**: Each restart begins from scratch (data files auto-increment)

## Post-Processing

After generation, convert to SFT format:
```bash
# If in normal location
atropos-sft-gen ~/atropos/data/intern_bootcamp_rollouts_*.jsonl \
    --tokenizer deepseek-ai/DeepSeek-R1 \
    --output ~/atropos/data/intern_bootcamp_sft.jsonl

# If in literal ${HOME} directory
atropos-sft-gen '${HOME}'/atropos/data/intern_bootcamp_rollouts_*.jsonl \
    --tokenizer deepseek-ai/DeepSeek-R1 \
    --output ~/atropos/data/intern_bootcamp_sft.jsonl
```

## Recent Runs

- **Job 630**: Generated 171/50,000 problems (~2,736 responses) before API server shutdown after 5 hours
- **Job 732**: Currently running with automatic requeue enabled