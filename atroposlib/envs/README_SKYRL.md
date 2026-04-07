# SkyRL Integration (SHM Transport)

This directory contains `skyrl_adapter.py`, enabling Atropos to provide reasoning environments for the SkyRL training framework.

## Architecture

The integration uses a **Zero-Copy Shared Memory (SHM)** transport to reduce serialization overhead during reasoning-dense RL collection.

* **Transport**: `atroposlib.api.shm_buffer.ZeroCopySHMBuffer`
* **Adapter**: `atroposlib.envs.skyrl_adapter.SkyRLAdapter`

## Performance

Benchmarks on RTX 3090 hardware:
- **Baseline (HTTP)**: ~2,000 trajectories/sec
- **Hardened (SHM)**: **16,500+ trajectories/sec** (~8x throughput gain)

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
pytest -v atroposlib/tests/test_skyrl_shm_e2e.py
```

This script verifies the atomic index synchronization and data integrity without requiring a full GPU cluster.
