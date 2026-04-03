# Atropos DEO: Production Deployment Guide

Technical specification for deploying the Atropos Dynamic Environment Orchestrator (DEO) in GPU-accelerated training clusters.

## Cluster-Scale Deployment

### 1. Scaling LLM Workers (GPU Isolation)
The DEO leverages `CUDA_VISIBLE_DEVICES` to ensure that each worker has dedicated, non-overlapping access to GPUs.

**Example: Launching Tensor-Parallel Workers (TP=2)**
To run workers that each consume 2 GPUs on an 8-GPU node:
```bash
python -m atroposlib.cli.orchestrate \
    --env-command "python main.py --tp 2 --port \$PORT" \
    --gpus-per-actor 2 \
    --max-actors 4
```
The DEO will automatically slice the device list:
- Worker 0: `CUDA_VISIBLE_DEVICES=0,1`
- Worker 1: `CUDA_VISIBLE_DEVICES=2,3`
- ...

### 2. Multi-Node Expansion
Use the `RemoteActor` strategy to manage a distributed fleet from a single controller. Ensure passwordless SSH and identical environment paths across the cluster.

---

## Production Resilience Patterns

### 1. Hardware Cordoning (Thermal Guard)
The DEO continuously monitors GPU health via NVML/SMI. If a GPU enters a `ThermalThrottled` or `HardwareFault` state (`0x0000000000000008`), the DEO will:
1.  **Cordon** the GPU (mark it as unavailable).
2.  **Skip** scale-up attempts that would utilize that hardware.
3.  Log a `CRITICAL` alert to prevent training performance degradation.

### 2. CrashLoopBackOff (Self-Healing)
To prevent "Launch Storms" (rapidly failing workers consuming CPU/IO), the DEO tracks failure frequency.
- **Trigger**: 3 failures within a 60-second window.
- **Action**: Scale-up is **HALTED** until the cooldown period expires or the operator intervenes.

### 3. Graceful Draining (Zero Data Loss)
During scale-down (e.g., training efficiency adjustment or node maintenance), the DEO sends `SIGUSR1`. 
- Workers finish the current rollout.
- Checkpoints are saved.
- Data is securely synchronized before process exit.

---

## Maintenance & Observability

### Diagnostic Audit
Run the status command to audit the current resource allocation:
```bash
python -m atroposlib.cli.orchestrate --status
```

### WandB Integration
All orchestration metadata is synchronized to WandB, allowing infra teams to monitor:
- `deo/rollout_pressure` (Scaling demand)
- `deo/num_draining` (Capacity withdrawal status)
- `deo/free_vram_mb` (Memory headroom)

---

## Troubleshooting
- **Zombie Processes**: If the DEO is killed via `SIGKILL`, some CUDA kernels may remain active. Restart the DEO; its **Warm Startup** logic will automatically adopt these orphans and reclaim them gracefully.
- **Port Hijacking**: The DEO performs a socket-level pre-flight check before every launch to prevent conflicts with unmanaged system processes.
