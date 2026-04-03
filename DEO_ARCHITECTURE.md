# Atropos DEO: Architecture & Scaling State Machine

The **Dynamic Environment Orchestrator (DEO)** is a resilient, latency-aware scaling engine designed to manage environment performers (workers) for RL training at scale.

## 🏗️ Core Components

1.  **ScalingController**: The "Brain". Uses a dampened PID-like calculation with hysteresis to decide the `target_actors` based on "Rollout Pressure" (Queue/BatchSize).
2.  **ScalingStrategy**: The "Hands". 
    - `LocalActor`: Manages subprocesses on the local node with port isolation.
    - `RemoteActor`: Manages processes on remote nodes via SSH.
3.  **MetricsCollector**: The "Sensors". Polls the Atropos API server for global workload telemetry. Includes a 3-poll grace period for network resilience.

## 🔄 The Scaling State Machine

Workers transition through four distinct phases to ensure zero data loss and cluster stability.

```mermaid
state_chart
    [*] --> Pending : Launched (Port Assigned)
    Pending --> Connected : Registered with API Server
    Connected --> Draining : SIGUSR1 (Scale Down Triggered)
    Draining --> [*] : Rollout Finished (Process Exit)
    
    Pending --> [*] : Boot Timeout / Failure
    Connected --> [*] : Crash / Termination
```

### 1. Pending Phase
- Orchestrator subtracts `pending` counts from scaling decisions to prevent **Launch Storms**.
- Tracked via PID and launch timestamp.

### 2. Connected Phase
- Worker is executing rollouts and contributing to the global throughput.
- Orchestrator monitors "Rollout Pressure" to decide if more are needed.

### 3. Draining Phase (Nous Maintainer Standard)
- When scaling down, the orchestrator DOES NOT kill the process immediately.
- It sends `SIGUSR1` to the worker.
- The worker stops accepting new tasks and finishes its current rollout.
- The orchestrator moves the worker to a `draining` list and continues managing the rest of the cluster.

### 4. Adoption Logic (Warm Startup)
- Upon restart, the DEO scans the process table for orphans matching the environment command.
- It "adopts" these workers into its management loop, preventing duplicate launches and port conflicts.

## 🛡️ Resilience Features
- **Port Isolation**: Dedicated port pools (`8001:8020`) for multi-instance scaling on a single IP.
- **Heartbeat Grace Period**: 3-poll window (~30s) where stale metrics are used during network flaps to prevent accidental mass scale-down.
- **Process Group Isolation**: `os.killpg` ensures that even a worker's sub-processes (e.g., a CUDA kernel launcher) are reaped correctly.
