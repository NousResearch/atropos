# SkyRL Integration (SHM Transport)

This directory contains the `skyrl_adapter.py`, which enables Atropos to act as a high-performance reasoning environment provider for the SkyRL training framework.

## Architecture

The integration utilizes a **Zero-Copy Shared Memory (SHM)** transport to eliminate the "JSON Tax" during reasoning-dense RL collection.

* **Transport**: `atroposlib.api.shm_buffer.ZeroCopySHMBuffer`
* **Adapter**: `atroposlib.envs.skyrl_adapter.SkyRLAdapter`

## Performance

Benchmarks on RTX 3090 hardware show an **~8x throughput gain** compared to standard HTTP/JSON transport:
- **Baseline (HTTP)**: ~2,000 trajectories/sec
- **Hardened (SHM)**: **16,500+ trajectories/sec**

## Usage

To enable the SHM transport, initialize the environment with `TransportType.SHM`:

```python
from atroposlib.envs.base import TransportType
from atroposlib.envs.skyrl_adapter import SkyRLAdapter

env = SkyRLAdapter(
    transport=TransportType.SHM,
    shm_name="atropos_shm_run1",
    # ... other config
)
```

## Testing

A dedicated end-to-end verification script for the SHM bridge is available in the root directory:

```bash
bash test_shm.sh
```

This script verifies the atomic index synchronization and data integrity without requiring a full GPU cluster.
