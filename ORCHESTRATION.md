# Atropos Elastic Orchestrator (DEO)

The Atropos Elastic Orchestrator (DEO) is a standalone microservice designed to dynamically scale environment workers based on trainer demand. It monitors the Atropos server's rollout queue and automatically adjusts the number of active environment instances to prevent trainer starvation and optimize resource usage.

## Architecture

DEO operates as a decoupled control loop:
1.  **Metric Collection**: Polls the Atropos server's `/global-status` endpoint for a metric called **Rollout Pressure (RP)**.
    *   `RP = Current_Queue_Size / Target_Batch_Size`
2.  **Scaling Decision**: A threshold-based controller determines if the system should scale UP (RP > 1.0) or scale DOWN (RP < 1.0).
3.  **Actor Management**: Currently manages local environment servers as subprocesses.

## Usage

DEO is integrated into the `atroposlib` CLI as `atropos-orchestrate`. 

### Basic Example

```bash
atropos-orchestrate \
  --server-url "http://localhost:8000" \
  --env-command "python environments/gsm8k_server.py serve" \
  --min-actors 1 \
  --max-actors 10
```

### Advanced Configuration

| Flag | Description | Default |
| :--- | :--- | :--- |
| `--target-pressure` | The ideal RP to maintain (1.0 = 1 batch ready) | 1.0 |
| `--cooldown` | Seconds to wait between scaling actions | 60 |
| `--max-step` | Maximum actors to add/remove in one poll | 4 |
| `--poll-interval` | How often to check metrics (seconds) | 10 |
| `--wandb` | Enable logging of orchestration metrics to WandB | False |
| `--verbose` | Enable debug logs for scaling decisions | False |

## Monitoring & Observability

### WandB Integration
If `--wandb` is enabled, DEO will automatically fetch the current project/group from the Atropos server and log the following metrics:
- `deo/rollout_pressure`: The current workload intensity.
- `deo/num_actors`: The number of active environment instances.
- `deo/queue_size`: The raw number of rollouts waiting in the server.

### Local Logging
In `--verbose` mode, DEO provides detailed decision traces:
```text
2026-04-03 01:54:43 [INFO] DEO: Starting DEO against http://localhost:8000...
2026-04-03 01:54:48 [DEBUG] Controller: In cooldown (55s remaining). Holding at 1 actors.
2026-04-03 01:55:03 [INFO] Controller DECISION: Scale UP 1 -> 5 (Pressure: 5.00)
```

## Best Practices

1.  **Warm-up**: Always start with `--min-actors 1` to ensure the trainer has at least one source of data at launch.
2.  **Resource Limits**: Use `--max-actors` to prevent DEO from saturating your host's CPU/GPU.
3.  **Hysteresis**: The default `scaling_threshold` in the controller logic prevents "jitter" (scaling up and down too rapidly when the pressure is near 1.0).
