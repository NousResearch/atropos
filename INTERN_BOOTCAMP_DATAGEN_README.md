# InternBootcamp Data Generation on BS200 Cluster

This setup runs data generation for the InternBootcamp environment using the new SFT data collection service with `serve` mode.

## Quick Start

```bash
sbatch intern_bootcamp_datagen.slurm
```

## Configuration

- **Model**: Configured in the SLURM script (`MODEL_NAME` variable)
- **Data Output**: `~/atropos/data/`
- **Logs**: `logs/$SLURM_JOB_ID/`
- **Config**: `environments/intern_bootcamp/config_serve.yaml`

## Architecture

1. **SGLang** runs on all 8 GPUs with:
   - Data Parallelism (DP): 4 replicas
   - Tensor Parallelism (TP): 2 GPUs per replica
   - Port: 9000

2. **Atropos API Server** handles trajectory collection on CPUs

3. **InternBootcamp Environment** generates problems and collects responses

## Monitoring

Check logs during execution:
```bash
tail -f logs/$SLURM_JOB_ID/api.log
tail -f logs/$SLURM_JOB_ID/sglang.log
tail -f logs/$SLURM_JOB_ID/intern_bootcamp.log
```

## Output Files

- Raw rollouts: `~/atropos/data/intern_bootcamp_rollouts_*.jsonl`
- Processed data: `~/atropos/data/intern_bootcamp_serve_data.jsonl`

## Post-Processing

After generation, convert to SFT format:
```bash
atropos-sft-gen ~/atropos/data/intern_bootcamp_rollouts_*.jsonl --tokenizer NousResearch/Hermes-3-Llama-3.1-8B
```