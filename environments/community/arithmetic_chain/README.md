# Arithmetic Chain

Self-contained RL environment: procedurally generated multi-step integer problems (add / subtract / multiply from a starting value). The model must answer with `\boxed{integer}`; rewards use the same `math_verify` path as GSM8K.

**No Hugging Face dataset** — training items are sampled on the fly.

## Run (serve)

From the repo root, with Atropos API and an OpenAI-compatible inference server configured in `config_init` or via CLI overrides:

```bash
python environments/community/arithmetic_chain/arithmetic_chain_server.py serve --slurm false
```

## Process (debug rollouts)

```bash
python environments/community/arithmetic_chain/arithmetic_chain_server.py process \
  --env.data_path_to_save_groups rollouts.jsonl \
  --slurm false
```

Uses `ManagedServer` for token/logprob tracking (compatible with trainers that expect Atropos’ standard scored groups).
