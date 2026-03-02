# GRPO Example Trainer

This guide explains how to run the `example_trainer` integration with Atropos using GRPO.

The trainer is a reference implementation for end-to-end wiring (`environment -> run-api -> rollout server -> optimizer`), with multiple synchronization modes with vLLM.

## Supported Modes

- `shared_vllm`: single-copy training via CUDA IPC (trainer updates shared vLLM tensors in place)
- `lora_only`: LoRA adapter training with HTTP hot-swap (slow due to eager mode)
- `lora_restart`: LoRA adapter training with periodic vLLM restart (faster than `lora_only`)
- `none`: legacy full-checkpoint flow with vLLM reloads

## Prerequisites

1. Python 3.10+
2. CUDA-capable PyTorch environment for GPU training
3. Atropos API server available (`run-api`)
4. An environment process producing trajectories (for example GSM8K server)

## Installation

From repository root:

```bash
pip install -e ".[example_trainer]"
```

Optional (all extras):

```bash
pip install -e ".[all]"
```

## CLI Entry Points

After install, you can use either module invocation or script entrypoints:

- `python -m example_trainer.grpo` or `atropos-grpo`
- `python -m example_trainer.run` or `atropos-grpo-run`

## Minimal End-to-End Startup

### 1) Start Atropos API

```bash
run-api --port 8002
```

### 2) Start an environment

```bash
python environments/gsm8k_server.py serve \
  --env.rollout_server_url "http://localhost:8002" \
  --openai.server_type vllm \
  --openai.base_url "http://localhost:9001/v1" \
  --openai.api_key "dummy"
```

### 3) Start vLLM server (shared-weights example)

```bash
VLLM_ENABLE_SHARED_WEIGHTS=1 LOGDIR=/tmp/grpo_training \
python -m example_trainer.vllm_api_server \
  --model Qwen/Qwen3-1.7B-Base \
  --port 9001 \
  --gpu-memory-utilization 0.45 \
  --enforce-eager
```

### 4) Start trainer

```bash
atropos-grpo \
  --model-name Qwen/Qwen3-1.7B-Base \
  --weight-bridge-mode shared_vllm \
  --vllm-port 9001 \
  --vllm-config-path /tmp/grpo_training/vllm_bridge_config.json \
  --atropos-url "http://localhost:8002" \
  --batch-size 1 \
  --gradient-accumulation-steps 64 \
  --warmup-steps 5 \
  --training-steps 30 \
  --clip-eps 0.2
```

## Objective Notes

- GRPO uses rollout `inference_logprobs` for importance-ratio computation.
- The trainer currently uses clipped importance-ratio updates without a separate frozen-reference-model KL term.

## Outputs

- Trainer logs to stdout (and optional W&B if enabled)
- Checkpoints under `--save-path`
- Mode-specific logs/checkpoints when using matrix/orchestration scripts

## Troubleshooting

- If vLLM health checks time out, inspect `vllm.log`, `trainer.log`, and `env.log`.
- If targeted shared-layer runs lose gradients, ensure non-reentrant checkpointing is enabled in shared mode.
- If environment workers time out at 600s, reduce env concurrency (`--env.max_num_workers_per_node`) and batch pressure.
